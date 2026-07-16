import subprocess
from pathlib import Path
from typing import TypedDict

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
            target path is outside the repository.
        UnicodeDecodeError: If the target file is not valid UTF-8 text.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    if relative_path.is_absolute():
        raise ValueError("Relative path need to be relative not absolute path")

    resolved_repo = repo_path.resolve()
    resolved_target = (resolved_repo / relative_path).resolve()

    try:
        resolved_target.relative_to(resolved_repo)
    except ValueError as exc:
        raise ValueError(
            f"Resolve target not in repository root, resolve target is {resolved_target}"
        ) from exc

    if not resolved_target.exists():
        raise FileNotFoundError(f"File path does not exist: {resolved_target}")

    if resolved_target.is_dir():
        raise IsADirectoryError(f"File path is a directory: {resolved_target}")

    lines = resolved_target.read_text(encoding="utf-8").splitlines()
    file_length = len(lines)

    if start_line < 1:
        raise ValueError(f"Start line should be positive, start line {start_line}")

    if start_line > end_line:
        raise ValueError(
            f"Start line must smaller or equal to end line, start line: {start_line}, end line: {end_line}"
        )

    if end_line - start_line >= 400:
        raise ValueError("Read more than 400 lines")

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
        RuntimeError: If ripgrep is unavailable, times out, or exits with an
            unexpected error.
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
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color",
        "never",
        "--fixed-strings",
        "--sort",
        "path",
    ]

    for glob in globs or []:
        command.extend(["--glob", glob])

    command.extend(["--", query, "."])

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=resolved_path,
            timeout=5,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ripgrep executable 'rg' was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ripgrep search timed out after 5 seconds") from exc

    if completed_process.returncode == 1:
        return []

    if completed_process.returncode != 0:
        error_message = completed_process.stderr.strip() or "no error message"
        raise RuntimeError(
            f"ripgrep failed with return code {completed_process.returncode}: "
            f"{error_message}"
        )

    matches: list[SearchMatch] = []
    output_lines = completed_process.stdout.splitlines()
    for output_line in output_lines:
        path_text, line_text, matched_text = output_line.split(":", 2)
        matches.append(
            {
                "path": Path(path_text).as_posix(),
                "line": int(line_text),
                "text": matched_text,
            }
        )

        if len(matches) >= max_results:
            break

    return matches
