"""Deterministic, bounded queries over Python structural indexes."""

import os
import keyword
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from repocompass.structure.contracts import PythonFileIndex
from repocompass.structure.python_index import (
    PythonIndexError,
    PythonIndexErrorReason,
    build_python_file_index,
)
from repocompass.tools.contracts import COMMIT_SHA_PATTERN, SHA256_PATTERN
from repocompass.tools.repository import IGNORED_DIRECTORY_NAMES

MAX_PYTHON_SCAN_FILES = 500
MAX_PYTHON_SCAN_ENTRIES = 20_000
MAX_PYTHON_SCAN_DEPTH = 32
MAX_STRUCTURAL_QUERY_CHARACTERS = 2_048
MAX_STRUCTURAL_RESULTS = 500


@dataclass(frozen=True, slots=True)
class PythonFileDiscovery:
    """One deterministic and explicitly truncated repository scan."""

    paths: tuple[Path, ...]
    truncated: bool


def list_python_symbols(
    repository_root: Path,
    relative_path: str,
    *,
    max_results: int,
    repository_id: str,
    commit_sha: str | None,
) -> dict[str, object]:
    """Return bounded AST-verified definitions for one Python file."""

    _validate_query_contract(repository_id, commit_sha, max_results=max_results)
    index = _build_index(
        repository_root,
        Path(relative_path),
        repository_id=repository_id,
        commit_sha=commit_sha,
    )
    records = index.symbols[:max_results]
    return {
        **_file_identity(index),
        "returned_results": len(records),
        "results_truncated": len(records) < len(index.symbols),
        "symbols": [_json_record(record) for record in records],
        "total_results": len(index.symbols),
    }


def inspect_python_imports(
    repository_root: Path,
    relative_path: str,
    *,
    max_results: int,
    repository_id: str,
    commit_sha: str | None,
) -> dict[str, object]:
    """Return bounded AST-verified imports for one Python file."""

    _validate_query_contract(repository_id, commit_sha, max_results=max_results)
    index = _build_index(
        repository_root,
        Path(relative_path),
        repository_id=repository_id,
        commit_sha=commit_sha,
    )
    records = index.imports[:max_results]
    return {
        **_file_identity(index),
        "imports": [_json_record(record) for record in records],
        "returned_results": len(records),
        "results_truncated": len(records) < len(index.imports),
        "total_results": len(index.imports),
    }


def find_python_symbol_definitions(
    repository_root: Path,
    name: str,
    *,
    relative_path: str | None,
    max_files: int,
    max_results: int,
    repository_id: str,
    commit_sha: str | None,
) -> dict[str, object]:
    """Find exact qualified or simple definitions without name resolution."""

    _validate_query_contract(
        repository_id,
        commit_sha,
        max_results=max_results,
        name=name,
    )
    scan = _create_index_scan(
        repository_root,
        relative_path=relative_path,
        max_files=max_files,
        repository_id=repository_id,
        commit_sha=commit_sha,
    )
    matched: list[dict[str, object]] = []
    total_results = 0
    qualified_query = "." in name
    for index in scan.iter_indexes():
        for record in index.symbols:
            matches = (
                record.qualified_name == name
                if qualified_query
                else record.name == name
            )
            if not matches:
                continue
            total_results += 1
            if len(matched) < max_results:
                matched.append(_json_record(record))
    return _repository_query_result(
        repository_id=repository_id,
        commit_sha=commit_sha,
        query=name,
        match_semantics="exact_simple_or_qualified_definition",
        result_key="definitions",
        records=matched,
        total_results=total_results,
        scan=scan,
    )


def find_python_references(
    repository_root: Path,
    name: str,
    *,
    relative_path: str | None,
    max_files: int,
    max_results: int,
    repository_id: str,
    commit_sha: str | None,
) -> dict[str, object]:
    """Find syntactic references; no cross-file binding is claimed."""

    _validate_query_contract(
        repository_id,
        commit_sha,
        max_results=max_results,
        name=name,
    )
    scan = _create_index_scan(
        repository_root,
        relative_path=relative_path,
        max_files=max_files,
        repository_id=repository_id,
        commit_sha=commit_sha,
    )
    matched: list[dict[str, object]] = []
    total_results = 0
    qualified_query = "." in name
    for index in scan.iter_indexes():
        for record in index.references:
            matches = (
                record.name == name
                if qualified_query
                else record.name == name or record.name.rsplit(".", 1)[-1] == name
            )
            if not matches:
                continue
            total_results += 1
            if len(matched) < max_results:
                matched.append(_json_record(record))
    return _repository_query_result(
        repository_id=repository_id,
        commit_sha=commit_sha,
        query=name,
        match_semantics="syntactic_exact_or_terminal_component",
        result_key="references",
        records=matched,
        total_results=total_results,
        scan=scan,
    )


def discover_python_files(
    repository_root: Path,
    *,
    max_files: int,
) -> PythonFileDiscovery:
    """Discover Python files in stable order without following symlinks."""

    if (
        not isinstance(max_files, int)
        or isinstance(max_files, bool)
        or not 1 <= max_files <= MAX_PYTHON_SCAN_FILES
    ):
        raise ValueError("max_files is outside the structural scan limit")
    try:
        root = repository_root.resolve(strict=True)
    except (AttributeError, OSError) as error:
        raise PythonIndexError(PythonIndexErrorReason.INVALID_REPOSITORY) from error
    if not root.is_dir():
        raise PythonIndexError(PythonIndexErrorReason.INVALID_REPOSITORY)

    paths: list[Path] = []
    entries_seen = 0
    truncated = False
    stack: list[tuple[Path, int]] = [(Path(), 0)]
    while stack and len(paths) <= max_files:
        relative_directory, depth = stack.pop()
        directory = root / relative_directory
        remaining_entries = MAX_PYTHON_SCAN_ENTRIES - entries_seen
        if remaining_entries <= 0:
            truncated = True
            break
        try:
            directory_entries, overflow = _bounded_directory_entries(
                directory,
                max_entries=remaining_entries,
            )
        except OSError:
            truncated = True
            continue
        if overflow:
            truncated = True
            break
        entries_seen += len(directory_entries)
        child_directories: list[Path] = []
        for entry in sorted(directory_entries, key=lambda item: item.name):
            relative_entry = relative_directory / entry.name
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in IGNORED_DIRECTORY_NAMES:
                        continue
                    if depth >= MAX_PYTHON_SCAN_DEPTH:
                        truncated = True
                        continue
                    child_directories.append(relative_entry)
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(
                    ".py"
                ):
                    paths.append(relative_entry)
                    if len(paths) > max_files:
                        truncated = True
                        break
            except OSError:
                truncated = True
        stack.extend((path, depth + 1) for path in reversed(child_directories))

    return PythonFileDiscovery(paths=tuple(paths[:max_files]), truncated=truncated)


def _bounded_directory_entries(
    directory: Path,
    *,
    max_entries: int,
) -> tuple[list[os.DirEntry[str]], bool]:
    """Read one directory with a no-follow fd when the platform supports it."""

    if os.scandir in os.supports_fd:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(directory, flags)
        try:
            return _collect_directory_entries(
                os.scandir(descriptor),
                max_entries=max_entries,
            )
        finally:
            os.close(descriptor)
    if directory.is_symlink():
        raise OSError("directory symlinks are not allowed")
    return _collect_directory_entries(
        os.scandir(directory),
        max_entries=max_entries,
    )


def _collect_directory_entries(
    iterator: Iterator[os.DirEntry[str]],
    *,
    max_entries: int,
) -> tuple[list[os.DirEntry[str]], bool]:
    entries: list[os.DirEntry[str]] = []
    with iterator:
        for entry in iterator:
            entries.append(entry)
            if len(entries) > max_entries:
                return entries, True
    return entries, False


@dataclass(slots=True)
class _PythonIndexScan:
    repository_root: Path
    repository_id: str
    commit_sha: str | None
    direct_path: bool
    discovery: PythonFileDiscovery
    indexed_files: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def iter_indexes(self) -> Iterator[PythonFileIndex]:
        """Yield one index at a time so repository scans retain bounded memory."""

        for path in self.discovery.paths:
            try:
                index = _build_index(
                    self.repository_root,
                    path,
                    repository_id=self.repository_id,
                    commit_sha=self.commit_sha,
                )
            except PythonIndexError as error:
                if self.direct_path:
                    raise
                key = error.reason.value
                self.skipped_by_reason[key] = self.skipped_by_reason.get(key, 0) + 1
                continue
            self.indexed_files += 1
            yield index


def _create_index_scan(
    repository_root: Path,
    *,
    relative_path: str | None,
    max_files: int,
    repository_id: str,
    commit_sha: str | None,
) -> _PythonIndexScan:
    discovery = (
        discover_python_files(repository_root, max_files=max_files)
        if relative_path is None
        else PythonFileDiscovery(paths=(Path(relative_path),), truncated=False)
    )
    return _PythonIndexScan(
        repository_root=repository_root,
        repository_id=repository_id,
        commit_sha=commit_sha,
        direct_path=relative_path is not None,
        discovery=discovery,
    )


def _build_index(
    repository_root: Path,
    relative_path: Path,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> PythonFileIndex:
    return build_python_file_index(
        repository_root,
        relative_path,
        repository_id=repository_id,
        commit_sha=commit_sha,
    )


def _file_identity(index: PythonFileIndex) -> dict[str, object]:
    return {
        "commit_sha": index.commit_sha,
        "index_id": index.index_id,
        "module_name": index.module_name,
        "path": index.path,
        "repository_id": index.repository_id,
        "schema_version": index.schema_version,
        "source_digest": index.source_digest,
    }


def _repository_query_result(
    *,
    repository_id: str,
    commit_sha: str | None,
    query: str,
    match_semantics: str,
    result_key: str,
    records: list[dict[str, object]],
    total_results: int,
    scan: _PythonIndexScan,
) -> dict[str, object]:
    return {
        "candidate_files": len(scan.discovery.paths),
        "commit_sha": commit_sha,
        "file_scan_truncated": scan.discovery.truncated,
        "indexed_files": scan.indexed_files,
        "match_semantics": match_semantics,
        "query": query,
        "repository_id": repository_id,
        result_key: records,
        "returned_results": len(records),
        "results_truncated": len(records) < total_results,
        "schema_version": 1,
        "skipped_by_reason": dict(sorted(scan.skipped_by_reason.items())),
        "total_results": total_results,
    }


def _json_record(record: object) -> dict[str, object]:
    model_dump = getattr(record, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("structural record must support model_dump")
    value = model_dump(mode="json")
    if not isinstance(value, dict):
        raise TypeError("structural record must serialize to an object")
    return value


def _validate_query_contract(
    repository_id: str,
    commit_sha: str | None,
    *,
    max_results: int,
    name: str | None = None,
) -> None:
    if (
        not isinstance(repository_id, str)
        or re.fullmatch(SHA256_PATTERN, repository_id) is None
        or (
            commit_sha is not None
            and (
                not isinstance(commit_sha, str)
                or re.fullmatch(COMMIT_SHA_PATTERN, commit_sha) is None
            )
        )
    ):
        raise PythonIndexError(PythonIndexErrorReason.INVALID_IDENTITY)
    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or not 1 <= max_results <= MAX_STRUCTURAL_RESULTS
    ):
        raise ValueError("max_results is outside the structural query limit")
    if name is None:
        return
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= MAX_STRUCTURAL_QUERY_CHARACTERS
        or any(
            not part.isidentifier() or keyword.iskeyword(part)
            for part in name.split(".")
        )
    ):
        raise ValueError("name must be a dotted Python identifier")
