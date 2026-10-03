import json
import os
import selectors
import subprocess
from pathlib import Path
from time import monotonic
from typing import TypedDict

from repocompass.source_reader import (
    SourceReadError,
    SourceReadReason,
    read_source_bytes,
)

# ========================================================
# List tree
# ========================================================

IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    }
)


def list_tree(
    repo_path: Path,
    max_depth: int = 3,
    max_entries: int = 200,
) -> list[str]:
    """List repository entries as relative POSIX-style paths.

    Args:
        repo_path: Path to the repository root directory.
        max_depth: Maximum traversal depth. The repository root is depth 0.
        max_entries: Maximum number of entries to return

    Returns:
        A stable list of relative paths. Directory paths end with ``/``,
        and the repository root itself is excluded.

    Raises:
        FileNotFoundError: If the repository path does not exist.
        NotADirectoryError: If the repository path is not a directory.
        ValueError: If max_depth is negative or max_entries is not positive.
    """

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    if max_depth < 0:
        raise ValueError(
            f"Max_depth must be non-negative, current max_depth is {max_depth}"
        )

    if max_entries <= 0:
        raise ValueError(
            f"Max_entries must be positive, current max_entries is {max_entries}"
        )

    current_depth = 0
    entries = []
    _walk(repo_path, repo_path, entries, current_depth, max_depth, max_entries)

    return entries


def _walk(
    dir_path: Path,
    repo_path: Path,
    entries: list[str],
    current_depth: int = 0,
    max_depth: int = 3,
    max_entries: int = 200,
) -> None:
    """Recursive helper of list tree"""
    if current_depth >= max_depth or len(entries) >= max_entries:
        return

    for item in sorted(dir_path.iterdir()):
        if len(entries) >= max_entries:
            return

        if not item.is_symlink() and item.is_dir():
            if item.name in IGNORED_DIRECTORY_NAMES:
                continue

            entries.append(item.relative_to(repo_path).as_posix() + "/")
            child_depth = current_depth + 1

            if child_depth < max_depth:
                _walk(item, repo_path, entries, child_depth, max_depth, max_entries)

            continue

        entries.append(item.relative_to(repo_path).as_posix())

    return


# ========================================================
# Read Code
# ========================================================


class ReadCodeResult(TypedDict):
    """Structured result returned by read_code."""

    path: str
    start_line: int
    end_line: int
    content: str


def read_code(
    repo_path: Path, relative_path: Path, start_line: int, end_line: int
) -> ReadCodeResult:
    """Read a numbered range of UTF-8 lines from a repository file.

    Args:
        repo_path: Path to the repository root directory.
        relative_path: Path to the file, relative to the repository root.
        start_line: First line to read, inclusive and starting from 1.
        end_line: Last line to read, inclusive. Values beyond EOF are truncated.

    Returns:
        A structured result containing the POSIX-style relative file path,
        actual start and end lines, and content prefixed with line numbers.

    Raises:
        FileNotFoundError: If the repository or target file does not exist.
        NotADirectoryError: If the repository path is not a directory.
        IsADirectoryError: If the target path is a directory.
        ValueError: If the line range is invalid, exceeds 400 lines, or the
            target is unsafe, changes during reading, or exceeds 4 MiB.
        UnicodeDecodeError: If the target file is not valid UTF-8 text.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    if type(start_line) is not int or type(end_line) is not int:
        raise ValueError("Line numbers must be integers")
    if start_line < 1:
        raise ValueError(f"Start line should be positive, start line {start_line}")

    if start_line > end_line:
        raise ValueError(
            f"Start line must smaller or equal to end line, start line: {start_line}, end line: {end_line}"
        )

    if end_line - start_line >= 400:
        raise ValueError("Read more than 400 lines")

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Source path must stay relative to the repository")
    if (repo_path / relative_path).is_dir():
        raise IsADirectoryError(f"Source is a directory: {relative_path.as_posix()}")
    # Preflight policy alone is not authority to follow a path later. Open each
    # component relative to a pinned directory descriptor in the worker itself.
    try:
        content = read_source_bytes(repo_path, relative_path.as_posix())
    except SourceReadError as error:
        if error.reason is SourceReadReason.MISSING:
            raise FileNotFoundError(
                f"Source not found: {relative_path.as_posix()}"
            ) from error
        raise ValueError(f"Source read rejected: {error.reason.value}") from error
    lines = content.decode("utf-8").splitlines()
    file_length = len(lines)

    if start_line > file_length:
        raise ValueError(
            f"Start line exceeds file length, start line: {start_line}, file length: {file_length}"
        )

    end_line = min(file_length, end_line)

    selected_lines = lines[start_line - 1 : end_line]
    numbered_lines = []
    for line_num, line_txt in enumerate(selected_lines, start=start_line):
        line = f"{line_num} | {line_txt}"
        numbered_lines.append(line)

    result = {
        "path": relative_path.as_posix(),
        "start_line": start_line,
        "end_line": end_line,
        "content": "\n".join(numbered_lines),
    }

    return result


# ========================================================
# search_text
# ========================================================


class SearchMatch(TypedDict):
    """A single text match returned by search_text."""

    path: str
    line: int
    text: str


SEARCH_TIMEOUT_SECONDS = 5
MAX_SEARCH_STDOUT_BYTES = 2 * 1024 * 1024
MAX_SEARCH_RECORD_BYTES = 128 * 1024
MAX_SEARCH_STDERR_BYTES = 8192


def search_text(
    repo_path: Path,
    query: str,
    globs: list[str] | None = None,
    max_results: int = 100,
) -> list[SearchMatch]:
    """Search repository text with ripgrep and return structured matches.

    Args:
        repo_path: Path to the repository root directory.
        query: Literal text to search for.
        globs: Optional ripgrep glob patterns used to restrict searched files.
        max_results: Maximum number of matches to return.

    Returns:
        A stable list of matches containing relative POSIX-style paths, line
        numbers, and complete matching lines.

    Raises:
        FileNotFoundError: If the repository path does not exist.
        NotADirectoryError: If the repository path is not a directory.
        ValueError: If query is empty or max_results is not positive.
        RuntimeError: If ripgrep is unavailable, times out, exceeds a stream
            limit, emits invalid JSON/UTF-8 data, or exits with an error.

    Stops the child after max_results. This is a bounded prefix, not proof that
    the entire repository was scanned. Pipe limits do not bound ripgrep's RSS.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    if not query:
        raise ValueError("Query must not be empty")

    if max_results <= 0:
        raise ValueError(
            f"Max_results must be positive, current max_results is {max_results}"
        )

    resolved_path = repo_path.resolve()
    command = [
        "rg",
        # User ripgrep config may enable --follow or executable --pre filters.
        # A read-only Agent tool must not inherit those additional capabilities.
        "--no-config",
        "--no-follow",
        "--json",
        "--line-buffered",
        "--fixed-strings",
        "--sort",
        "path",
    ]

    for glob in globs or []:
        command.extend(["--glob", glob])

    command.extend(["--", query, "."])

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=resolved_path,
            shell=False,
        ) as process:
            try:
                return _read_search_output(process, max_results)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
    except FileNotFoundError as exc:
        raise RuntimeError("ripgrep executable 'rg' was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ripgrep search timed out after {SEARCH_TIMEOUT_SECONDS} seconds"
        ) from exc


def _read_search_output(
    process: subprocess.Popen[bytes], max_results: int
) -> list[SearchMatch]:
    deadline = monotonic() + SEARCH_TIMEOUT_SECONDS
    pending = bytearray()
    errors = bytearray()
    stdout_bytes = 0
    matches: list[SearchMatch] = []
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, SEARCH_TIMEOUT_SECONDS)
            for key, _ in selector.select(remaining):
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    if len(errors) + len(chunk) > MAX_SEARCH_STDERR_BYTES:
                        raise RuntimeError("ripgrep stderr exceeded byte limit")
                    errors.extend(chunk)
                    continue
                stdout_bytes += len(chunk)
                if stdout_bytes > MAX_SEARCH_STDOUT_BYTES:
                    raise RuntimeError(
                        "ripgrep stdout exceeded byte limit; narrow the query"
                    )
                pending.extend(chunk)
                while (end := pending.find(b"\n")) >= 0:
                    if end > MAX_SEARCH_RECORD_BYTES:
                        raise RuntimeError("ripgrep JSON record exceeded byte limit")
                    record = bytes(pending[:end])
                    del pending[: end + 1]
                    match = _parse_search_record(record)
                    if match is not None:
                        matches.append(match)
                        if len(matches) >= max_results:
                            return matches
                if len(pending) > MAX_SEARCH_RECORD_BYTES:
                    raise RuntimeError("ripgrep JSON record exceeded byte limit")
    if pending:
        raise RuntimeError("ripgrep emitted an incomplete JSON record")
    returncode = process.wait(timeout=max(0, deadline - monotonic()))
    if returncode not in (0, 1):
        detail = errors.decode("utf-8", errors="replace").strip() or "no error message"
        raise RuntimeError(f"ripgrep failed with return code {returncode}: {detail}")
    return matches


def _parse_search_record(record: bytes) -> SearchMatch | None:
    try:
        event = json.loads(record)
        if event["type"] in ("begin", "end", "summary"):
            return None
        if event["type"] != "match":
            raise ValueError("unexpected event")
        data = event["data"]
        path = data["path"]["text"]
        text = data["lines"]["text"]
        line = data["line_number"]
        if not isinstance(path, str) or not isinstance(text, str):
            raise ValueError("non-text match")
        if type(line) is not int or line < 1:
            raise ValueError("invalid line number")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("non-relative path")
        return {
            "path": relative.as_posix(),
            "line": line,
            "text": text.removesuffix("\n").removesuffix("\r"),
        }
    except (ValueError, KeyError, TypeError) as error:
        # ripgrep uses base64 fields for non-UTF-8 data; never turn those into
        # plausible but lossy source paths or content.
        raise RuntimeError(
            "ripgrep emitted invalid or non-UTF-8 JSON match data"
        ) from error
