"""Race-resistant source and snapshot revalidation for recalled memory."""

import errno
import hashlib
import os
import stat
from enum import StrEnum
from pathlib import Path, PurePosixPath

from repocompass.memory.contracts import (
    MemoryFreshnessStatus,
    MemoryStalenessReason,
    RepositoryMemory,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.tools.gateway import ToolExecutionContext

MAX_MEMORY_SOURCE_BYTES = 4 * 1024 * 1024


class MemorySourceReadReason(StrEnum):
    """Internal safe-reader outcome mapped to sanitized freshness reasons."""

    INVALID_PATH = "invalid_path"
    MISSING = "missing"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    OVERSIZED = "oversized"
    CHANGED_DURING_READ = "changed_during_read"
    INVALID_REPOSITORY = "invalid_repository"


class MemorySourceReadError(RuntimeError):
    """A source could not be read through the bounded no-follow boundary."""

    def __init__(self, reason: MemorySourceReadReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def read_memory_source_digest(
    repository_root: Path,
    relative_path: str,
) -> str:
    """Hash one bounded regular file using component-wise ``openat`` calls."""

    parts = _validated_path_parts(relative_path)
    if not isinstance(repository_root, Path):
        raise MemorySourceReadError(MemorySourceReadReason.INVALID_REPOSITORY)
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise MemorySourceReadError(
            MemorySourceReadReason.INVALID_REPOSITORY
        ) from error
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    try:
        try:
            current_descriptor = os.open(root, directory_flags)
        except OSError as error:
            raise MemorySourceReadError(
                MemorySourceReadReason.INVALID_REPOSITORY
            ) from error
        descriptors.append(current_descriptor)
        for component in parts[:-1]:
            try:
                current_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
            except OSError as error:
                raise MemorySourceReadError(
                    _reason_from_open_error(error, parent=True)
                ) from error
            descriptors.append(current_descriptor)
        try:
            file_descriptor = os.open(
                parts[-1],
                file_flags,
                dir_fd=current_descriptor,
            )
        except OSError as error:
            raise MemorySourceReadError(
                _reason_from_open_error(error, parent=False)
            ) from error
        descriptors.append(file_descriptor)

        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise MemorySourceReadError(MemorySourceReadReason.SPECIAL_FILE)
        if opened_stat.st_size > MAX_MEMORY_SOURCE_BYTES:
            raise MemorySourceReadError(MemorySourceReadReason.OVERSIZED)
        chunks: list[bytes] = []
        remaining = MAX_MEMORY_SOURCE_BYTES + 1
        while remaining > 0:
            chunk = os.read(file_descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        finished_stat = os.fstat(file_descriptor)
        if len(content) > MAX_MEMORY_SOURCE_BYTES:
            raise MemorySourceReadError(MemorySourceReadReason.OVERSIZED)
        if (
            finished_stat.st_dev != opened_stat.st_dev
            or finished_stat.st_ino != opened_stat.st_ino
            or finished_stat.st_size != opened_stat.st_size
            or finished_stat.st_mtime_ns != opened_stat.st_mtime_ns
            or len(content) != opened_stat.st_size
        ):
            raise MemorySourceReadError(MemorySourceReadReason.CHANGED_DURING_READ)
        return hashlib.sha256(content).hexdigest()
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def classify_repository_memory_freshness(
    memory: RepositoryMemory,
    context: ToolExecutionContext,
) -> tuple[MemoryFreshnessStatus, MemoryStalenessReason | None]:
    """Classify one recalled fact without ever auto-upgrading an old source."""

    if not isinstance(memory, RepositoryMemory):
        raise TypeError("memory must be a RepositoryMemory")
    if not isinstance(context, ToolExecutionContext):
        raise TypeError("context must be a ToolExecutionContext")
    identity = resolve_memory_repository_identity(context)
    if memory.repository_key != identity.repository_key:
        raise ValueError("memory repository key does not match current repository")
    if context.commit_sha is None:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.UNVERSIONED_SNAPSHOT,
        )
    if memory.commit_sha != context.commit_sha:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.COMMIT_MISMATCH,
        )
    if memory.source_repository_id != context.repository_id:
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH,
        )

    for source in memory.sources:
        try:
            current_digest = read_memory_source_digest(
                context.repository_root,
                source.location.path,
            )
        except MemorySourceReadError as error:
            if error.reason is MemorySourceReadReason.CHANGED_DURING_READ:
                reason = MemoryStalenessReason.SOURCE_CHANGED
            else:
                reason = MemoryStalenessReason.SOURCE_MISSING
            return MemoryFreshnessStatus.STALE, reason
        if current_digest != source.source_digest:
            return (
                MemoryFreshnessStatus.STALE,
                MemoryStalenessReason.SOURCE_CHANGED,
            )

    if not repository_snapshot_is_clean(context):
        return (
            MemoryFreshnessStatus.REVALIDATION_REQUIRED,
            MemoryStalenessReason.DIRTY_WORKTREE,
        )
    return MemoryFreshnessStatus.CURRENT, None


def repository_snapshot_is_clean(context: ToolExecutionContext) -> bool:
    """Reuse the Gateway's public immutable-snapshot decision boundary."""

    request = context.bind_call(
        call_id="memory-snapshot-check",
        tool_name="read_code",
        arguments={},
    )
    return context.allows_result_reuse(request)


def _validated_path_parts(relative_path: str) -> tuple[str, ...]:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise MemorySourceReadError(MemorySourceReadReason.INVALID_PATH)
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or relative_path == "."
        or ".." in path.parts
        or path.as_posix() != relative_path
    ):
        raise MemorySourceReadError(MemorySourceReadReason.INVALID_PATH)
    return path.parts


def _reason_from_open_error(error: OSError, *, parent: bool) -> MemorySourceReadReason:
    if error.errno == errno.ENOENT:
        return MemorySourceReadReason.MISSING
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        return MemorySourceReadReason.SYMLINK
    return (
        MemorySourceReadReason.INVALID_REPOSITORY
        if parent and error.errno in {errno.EACCES, errno.EPERM}
        else MemorySourceReadReason.SPECIAL_FILE
    )
