"""Clean-snapshot and AST-backed authority construction for focused improve."""

from __future__ import annotations

import hashlib
import stat
import subprocess
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath

from repocompass.improve.contracts import (
    MAX_IMPROVE_FILE_BYTES,
    FocusedImproveScope,
    ImproveTargetSelection,
    PythonVerificationProfile,
    ScopedImproveTarget,
    VerificationSpec,
    create_focused_scope_contract,
    validate_improve_path,
)
from repocompass.structure.contracts import PythonSymbolKind
from repocompass.structure.python_index import PythonIndexError, build_python_file_index
from repocompass.tools.gateway import ToolExecutionContext

_GIT_TIMEOUT_SECONDS = 5


class ImproveScopeErrorReason(StrEnum):
    """Stable, non-sensitive scope-construction denial reasons."""

    INVALID_REPOSITORY = "invalid_repository"
    GIT_REQUIRED = "git_required"
    DIRTY_SNAPSHOT = "dirty_snapshot"
    SNAPSHOT_CHANGED = "snapshot_changed"
    INVALID_PATH = "invalid_path"
    PATH_NOT_FOUND = "path_not_found"
    PATH_ESCAPE = "path_escape"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    UNTRACKED_FILE = "untracked_file"
    UNSUPPORTED_FILE = "unsupported_file"
    OVERSIZED_FILE = "oversized_file"
    ENCODING_ERROR = "encoding_error"
    FILE_CHANGED = "file_changed"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    INVALID_SYMBOL = "invalid_symbol"


class ImproveScopeError(ValueError):
    """Fail-closed scope error that never includes an absolute repository path."""

    def __init__(self, reason: ImproveScopeErrorReason) -> None:
        if not isinstance(reason, ImproveScopeErrorReason):
            raise TypeError("reason must be an ImproveScopeErrorReason")
        self.reason = reason
        super().__init__(f"focused improve scope rejected: {reason.value}")


def create_focused_improve_scope(
    repository_path: Path,
    *,
    objective: str,
    targets: Sequence[ImproveTargetSelection],
    verifications: Sequence[VerificationSpec] | None = None,
    max_changed_lines: int = 200,
    max_revision_count: int = 3,
) -> FocusedImproveScope:
    """Create immutable write authority for exact files/symbols in a clean commit."""

    if not isinstance(repository_path, Path):
        raise TypeError("repository_path must be a pathlib.Path")
    if not targets or not all(
        isinstance(item, ImproveTargetSelection) for item in targets
    ):
        raise ValueError("targets must contain at least one ImproveTargetSelection")
    try:
        context = ToolExecutionContext(repository_path)
    except (OSError, TypeError, ValueError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.INVALID_REPOSITORY) from error
    if context.commit_sha is None:
        raise ImproveScopeError(ImproveScopeErrorReason.GIT_REQUIRED)
    top_level = _git_top_level(context.repository_root)
    _require_clean(top_level)
    initial_identity = (context.repository_id, context.commit_sha)
    selections = _canonical_selections(targets)
    scoped_targets = tuple(
        _scope_target(context, top_level, selection) for selection in selections
    )
    _require_non_overlapping_targets(scoped_targets)
    final_context = ToolExecutionContext(context.repository_root)
    _require_clean(top_level)
    if (
        final_context.repository_id,
        final_context.commit_sha,
    ) != initial_identity or final_context.repository_root != context.repository_root:
        raise ImproveScopeError(ImproveScopeErrorReason.SNAPSHOT_CHANGED)
    resolved_verifications = (
        tuple(verifications)
        if verifications is not None
        else (
            VerificationSpec(
                profile=PythonVerificationProfile.COMPILEALL,
                arguments=tuple(sorted({target.path for target in scoped_targets})),
                timeout_seconds=30,
            ),
        )
    )
    return create_focused_scope_contract(
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
        objective=objective,
        targets=scoped_targets,
        verifications=resolved_verifications,
        max_changed_lines=max_changed_lines,
        max_revision_count=max_revision_count,
    )


def load_scope_sources(
    repository_path: Path,
    scope: FocusedImproveScope,
) -> dict[str, str]:
    """Revalidate an exact clean scope and return each unique trusted source."""

    if not isinstance(scope, FocusedImproveScope):
        raise TypeError("scope must be a FocusedImproveScope")
    try:
        context = ToolExecutionContext(repository_path)
    except (OSError, TypeError, ValueError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.INVALID_REPOSITORY) from error
    if (
        context.repository_id != scope.repository_id
        or context.commit_sha != scope.commit_sha
    ):
        raise ImproveScopeError(ImproveScopeErrorReason.SNAPSHOT_CHANGED)
    top_level = _git_top_level(context.repository_root)
    _require_clean(top_level)
    targets_by_path: dict[str, ScopedImproveTarget] = {}
    for target in scope.targets:
        targets_by_path.setdefault(target.path, target)
    sources: dict[str, str] = {}
    for path, target in targets_by_path.items():
        source_bytes, source = _read_tracked_source(
            context.repository_root,
            top_level,
            path,
        )
        if (
            hashlib.sha256(source_bytes).hexdigest() != target.base_sha256
            or len(source_bytes) != target.byte_count
        ):
            raise ImproveScopeError(ImproveScopeErrorReason.SNAPSHOT_CHANGED)
        sources[path] = source
    _require_clean(top_level)
    return sources


def _scope_target(
    context: ToolExecutionContext,
    top_level: Path,
    selection: ImproveTargetSelection,
) -> ScopedImproveTarget:
    source_bytes, source = _read_tracked_source(
        context.repository_root,
        top_level,
        selection.path,
    )
    digest = hashlib.sha256(source_bytes).hexdigest()
    line_count = source.count("\n") + 1
    if selection.symbol is None:
        return ScopedImproveTarget(
            path=selection.path,
            base_sha256=digest,
            byte_count=len(source_bytes),
            line_count=line_count,
        )
    try:
        index = build_python_file_index(
            context.repository_root,
            Path(*PurePosixPath(selection.path).parts),
            repository_id=context.repository_id,
            commit_sha=context.commit_sha,
        )
    except PythonIndexError as error:
        raise ImproveScopeError(ImproveScopeErrorReason.INVALID_SYMBOL) from error
    matches = tuple(
        symbol
        for symbol in index.symbols
        if symbol.qualified_name == selection.symbol
        and symbol.kind is not PythonSymbolKind.MODULE
    )
    if len(matches) != 1:
        raise ImproveScopeError(ImproveScopeErrorReason.SYMBOL_NOT_FOUND)
    symbol = matches[0]
    if index.source_digest != digest:
        raise ImproveScopeError(ImproveScopeErrorReason.FILE_CHANGED)
    return ScopedImproveTarget(
        path=selection.path,
        base_sha256=digest,
        byte_count=len(source_bytes),
        line_count=line_count,
        symbol=symbol.qualified_name,
        line_start=symbol.definition.line_start,
        line_end=symbol.definition.line_end,
    )


def _read_tracked_source(
    repository_root: Path,
    top_level: Path,
    path_text: str,
) -> tuple[bytes, str]:
    try:
        path_text = validate_improve_path(path_text)
    except (TypeError, ValueError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.INVALID_PATH) from error
    if not path_text.endswith(".py"):
        raise ImproveScopeError(ImproveScopeErrorReason.UNSUPPORTED_FILE)
    relative = Path(*PurePosixPath(path_text).parts)
    candidate = repository_root / relative
    current = repository_root
    try:
        for part in relative.parts:
            current = current / part
            item_stat = current.lstat()
            if stat.S_ISLNK(item_stat.st_mode):
                raise ImproveScopeError(ImproveScopeErrorReason.SYMLINK)
    except FileNotFoundError as error:
        raise ImproveScopeError(ImproveScopeErrorReason.PATH_NOT_FOUND) from error
    except OSError as error:
        raise ImproveScopeError(ImproveScopeErrorReason.SPECIAL_FILE) from error
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, ValueError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.PATH_ESCAPE) from error
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ImproveScopeError(ImproveScopeErrorReason.SPECIAL_FILE)
    tracked_path = resolved.relative_to(top_level).as_posix()
    stage = _git(top_level, "ls-files", "--stage", "--", tracked_path)
    if not stage.stdout or not stage.stdout.startswith((b"100644 ", b"100755 ")):
        raise ImproveScopeError(ImproveScopeErrorReason.UNTRACKED_FILE)
    try:
        with resolved.open("rb") as source_file:
            source_bytes = source_file.read(MAX_IMPROVE_FILE_BYTES + 1)
    except OSError as error:
        raise ImproveScopeError(ImproveScopeErrorReason.SPECIAL_FILE) from error
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ImproveScopeError(ImproveScopeErrorReason.FILE_CHANGED)
    if len(source_bytes) > MAX_IMPROVE_FILE_BYTES:
        raise ImproveScopeError(ImproveScopeErrorReason.OVERSIZED_FILE)
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImproveScopeError(ImproveScopeErrorReason.ENCODING_ERROR) from error
    if "\x00" in source:
        raise ImproveScopeError(ImproveScopeErrorReason.ENCODING_ERROR)
    return source_bytes, source


def _canonical_selections(
    targets: Sequence[ImproveTargetSelection],
) -> tuple[ImproveTargetSelection, ...]:
    indexed: dict[tuple[str, str], ImproveTargetSelection] = {}
    by_path: dict[str, list[ImproveTargetSelection]] = {}
    for target in targets:
        key = (target.path, target.symbol or "")
        if key in indexed:
            raise ValueError("focused targets must be unique")
        indexed[key] = target
        by_path.setdefault(target.path, []).append(target)
    for path_targets in by_path.values():
        if len(path_targets) > 1 and any(item.symbol is None for item in path_targets):
            raise ValueError("file target cannot overlap symbol targets")
    return tuple(indexed[key] for key in sorted(indexed))


def _require_non_overlapping_targets(targets: Sequence[ScopedImproveTarget]) -> None:
    by_path: dict[str, list[tuple[int, int]]] = {}
    for target in targets:
        if target.line_start is not None and target.line_end is not None:
            by_path.setdefault(target.path, []).append(
                (target.line_start, target.line_end)
            )
    for spans in by_path.values():
        ordered = sorted(spans)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] <= previous[1]:
                raise ImproveScopeError(ImproveScopeErrorReason.INVALID_SYMBOL)


def _git_top_level(repository_root: Path) -> Path:
    result = _git(repository_root, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise ImproveScopeError(ImproveScopeErrorReason.GIT_REQUIRED)
    try:
        value = result.stdout.decode("utf-8").strip()
        top_level = Path(value).resolve(strict=True)
        repository_root.relative_to(top_level)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.GIT_REQUIRED) from error
    return top_level


def _require_clean(top_level: Path) -> None:
    status_result = _git(
        top_level,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status_result.stdout:
        raise ImproveScopeError(ImproveScopeErrorReason.DIRTY_SNAPSHOT)


def _git(
    cwd: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise ImproveScopeError(ImproveScopeErrorReason.GIT_REQUIRED) from error
    if check and completed.returncode != 0:
        raise ImproveScopeError(ImproveScopeErrorReason.GIT_REQUIRED)
    return completed
