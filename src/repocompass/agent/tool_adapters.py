from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from repocompass.tools.repository import list_tree as list_tree_impl
from repocompass.tools.repository import read_code as read_code_impl
from repocompass.tools.repository import search_text as search_text_impl
from repocompass.tools.repository_tools import (
    MAX_LIST_TREE_ENTRIES,
    MAX_SEARCH_TEXT_RESULTS,
)

MAX_SEARCH_TEXT_RESULT = MAX_SEARCH_TEXT_RESULTS


@tool("list_tree")
def list_tree_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    max_depth: int = 3,
    max_entries: int = 100,
) -> str:
    """Inspect the repository structure when relevant file locations are unknown.

    Returns bounded, relative file and directory paths. Directories end with ``/``.
    """
    entries = list_tree_impl(
        repo_path,
        max_depth=max_depth,
        max_entries=min(max_entries, MAX_LIST_TREE_ENTRIES),
    )

    if not entries:
        return "No visible repository entries were found."

    return "\n".join(entries)


@tool("read_code")
def read_code_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    relative_path: Path,
    start_line: int,
    end_line: int,
) -> str:
    """Read a numbered line range from a known repository-relative file.

    Use this after locating a relevant file with ``list_tree`` or ``search_text``.
    The result identifies the file and actual line range before the code content.
    """

    result = read_code_impl(
        repo_path,
        relative_path=relative_path,
        start_line=start_line,
        end_line=end_line,
    )

    path = f"File: {result['path']}\n"
    line_range = f"Lines: {result['start_line']}-{result['end_line']}\n"
    content = result["content"]

    return path + line_range + content


@tool("search_text")
def search_text_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    query: str,
    globs: list[str] | None = None,
    max_results: int = 100,
) -> str:
    """Locate literal text in repository files when the exact file is unknown.

    Use optional glob patterns to narrow the search. Results are bounded and each
    match contains a repository-relative path, line number, and matching line.
    """

    max_results = min(max_results, MAX_SEARCH_TEXT_RESULT)
    matches = search_text_impl(repo_path, query, globs, max_results)

    if not matches:
        return "No match found"

    return "\n".join(
        [
            ":".join([match["path"], str(match["line"]), match["text"]])
            for match in matches
        ]
    )


REPOSITORY_TOOLS = (
    list_tree_tool,
    search_text_tool,
    read_code_tool,
)
