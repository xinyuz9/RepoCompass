"""Bounded component-wise source reads shared by tools, memory and indexing."""

import errno
import hashlib
import os
import stat
from enum import StrEnum
from pathlib import Path, PurePosixPath

MAX_SOURCE_BYTES = 4 * 1024 * 1024


class SourceReadReason(StrEnum):
    """Internal safe-reader outcome mapped to sanitized freshness reasons."""

    INVALID_PATH = "invalid_path"
    MISSING = "missing"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    OVERSIZED = "oversized"
    CHANGED_DURING_READ = "changed_during_read"
    INVALID_REPOSITORY = "invalid_repository"


class SourceReadError(RuntimeError):
    """A source could not be read through the bounded no-follow boundary."""

    def __init__(self, reason: SourceReadReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def read_source_digest(
    repository_root: Path,
    relative_path: str,
) -> str:
    """Hash one bounded regular file using component-wise ``openat`` calls."""

    return hashlib.sha256(read_source_bytes(repository_root, relative_path)).hexdigest()


def read_source_bytes(
    repository_root: Path,
    relative_path: str,
    *,
    max_bytes: int = MAX_SOURCE_BYTES,
) -> bytes:
    """Read bounded source through the same no-follow boundary; never execute it."""

    if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_SOURCE_BYTES:
        raise ValueError("source byte limit is invalid")

    if not all(
        getattr(os, name, 0) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        # No path-based fallback on platforms lacking the required boundary.
        raise SourceReadError(SourceReadReason.INVALID_REPOSITORY)

    parts = _validated_path_parts(relative_path)
    if not isinstance(repository_root, Path):
        raise SourceReadError(SourceReadReason.INVALID_REPOSITORY)
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise SourceReadError(SourceReadReason.INVALID_REPOSITORY) from error
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
            raise SourceReadError(SourceReadReason.INVALID_REPOSITORY) from error
        descriptors.append(current_descriptor)
        for component in parts[:-1]:
            try:
                current_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
            except OSError as error:
                raise SourceReadError(
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
            raise SourceReadError(
                _reason_from_open_error(error, parent=False)
            ) from error
        descriptors.append(file_descriptor)

        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SourceReadError(SourceReadReason.SPECIAL_FILE)
        if opened_stat.st_size > max_bytes:
            raise SourceReadError(SourceReadReason.OVERSIZED)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(file_descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        finished_stat = os.fstat(file_descriptor)
        if len(content) > max_bytes:
            raise SourceReadError(SourceReadReason.OVERSIZED)
        if (
            finished_stat.st_dev != opened_stat.st_dev
            or finished_stat.st_ino != opened_stat.st_ino
            or finished_stat.st_size != opened_stat.st_size
            or finished_stat.st_mtime_ns != opened_stat.st_mtime_ns
            or len(content) != opened_stat.st_size
        ):
            raise SourceReadError(SourceReadReason.CHANGED_DURING_READ)
        return content
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validated_path_parts(relative_path: str) -> tuple[str, ...]:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise SourceReadError(SourceReadReason.INVALID_PATH)
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or relative_path == "."
        or ".." in path.parts
        or path.as_posix() != relative_path
    ):
        raise SourceReadError(SourceReadReason.INVALID_PATH)
    return path.parts


def _reason_from_open_error(error: OSError, *, parent: bool) -> SourceReadReason:
    if error.errno == errno.ENOENT:
        return SourceReadReason.MISSING
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        return SourceReadReason.SYMLINK
    return (
        SourceReadReason.INVALID_REPOSITORY
        if parent and error.errno in {errno.EACCES, errno.EPERM}
        else SourceReadReason.SPECIAL_FILE
    )
