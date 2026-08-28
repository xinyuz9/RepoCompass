"""Schema-only model tools whose execution is owned by the Tool Gateway."""

from pathlib import Path
from typing import Annotated, NoReturn

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from repocompass.tools.repository_tools import (
    MAX_LIST_TREE_DEPTH,
    MAX_LIST_TREE_ENTRIES,
    MAX_PYTHON_SCAN_FILES,
    MAX_SEARCH_TEXT_RESULTS,
    MAX_STRUCTURAL_QUERY_CHARACTERS,
    MAX_STRUCTURAL_RESULTS,
)


@tool("list_tree")
def list_tree_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    max_depth: Annotated[int, Field(ge=0, le=MAX_LIST_TREE_DEPTH)] = 3,
    max_entries: Annotated[int, Field(ge=1, le=MAX_LIST_TREE_ENTRIES)] = 100,
) -> str:
    """Inspect a bounded repository tree when file locations are unknown."""
    del repo_path, max_depth, max_entries
    return _gateway_only()


@tool("read_code")
def read_code_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    relative_path: str,
    start_line: Annotated[int, Field(ge=1)],
    end_line: Annotated[int, Field(ge=1)],
) -> str:
    """Read a bounded numbered range from one known repository-relative file."""
    del repo_path, relative_path, start_line, end_line
    return _gateway_only()


@tool("search_text")
def search_text_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    query: str,
    globs: list[str] | None = None,
    max_results: Annotated[int, Field(ge=1, le=MAX_SEARCH_TEXT_RESULTS)] = 100,
) -> str:
    """Locate bounded literal-text matches with repository-relative locations."""
    del repo_path, query, globs, max_results
    return _gateway_only()


@tool("list_symbols")
def list_symbols_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    relative_path: str,
    max_results: Annotated[int, Field(ge=1, le=MAX_STRUCTURAL_RESULTS)] = 100,
) -> str:
    """List AST-verified symbols and exact source spans in one Python file."""
    del repo_path, relative_path, max_results
    return _gateway_only()


@tool("inspect_imports")
def inspect_imports_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    relative_path: str,
    max_results: Annotated[int, Field(ge=1, le=MAX_STRUCTURAL_RESULTS)] = 100,
) -> str:
    """Inspect AST-verified imports and lexical scopes in one Python file."""
    del repo_path, relative_path, max_results
    return _gateway_only()


@tool("find_symbol_definition")
def find_symbol_definition_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    name: Annotated[
        str,
        Field(min_length=1, max_length=MAX_STRUCTURAL_QUERY_CHARACTERS),
    ],
    relative_path: str | None = None,
    max_files: Annotated[int, Field(ge=1, le=MAX_PYTHON_SCAN_FILES)] = 200,
    max_results: Annotated[int, Field(ge=1, le=MAX_STRUCTURAL_RESULTS)] = 100,
) -> str:
    """Find exact Python definitions with content-addressed source provenance."""
    del repo_path, name, relative_path, max_files, max_results
    return _gateway_only()


@tool("find_references")
def find_references_tool(
    repo_path: Annotated[Path, InjectedState("repo_path")],
    name: Annotated[
        str,
        Field(min_length=1, max_length=MAX_STRUCTURAL_QUERY_CHARACTERS),
    ],
    relative_path: str | None = None,
    max_files: Annotated[int, Field(ge=1, le=MAX_PYTHON_SCAN_FILES)] = 200,
    max_results: Annotated[int, Field(ge=1, le=MAX_STRUCTURAL_RESULTS)] = 100,
) -> str:
    """Find syntactic Python references without claiming name resolution."""
    del repo_path, name, relative_path, max_files, max_results
    return _gateway_only()


def _gateway_only() -> NoReturn:
    raise RuntimeError("Repository tools may execute only through the Tool Gateway")


LEXICAL_REPOSITORY_TOOLS = (
    list_tree_tool,
    search_text_tool,
    read_code_tool,
)

STRUCTURAL_REPOSITORY_TOOLS = (
    list_symbols_tool,
    inspect_imports_tool,
    find_symbol_definition_tool,
    find_references_tool,
)

REPOSITORY_TOOLS = (*LEXICAL_REPOSITORY_TOOLS, *STRUCTURAL_REPOSITORY_TOOLS)
