"""Unit tests for the LLM-visible repository tool schemas."""

import pytest
from langchain_core.tools import BaseTool

from repocompass.agent.tool_adapters import (
    find_references_tool,
    find_symbol_definition_tool,
    inspect_imports_tool,
    list_tree_tool,
    list_symbols_tool,
    read_code_tool,
    search_text_tool,
)


@pytest.mark.parametrize(
    ("repository_tool", "expected_properties", "expected_required"),
    [
        pytest.param(
            list_tree_tool,
            {"max_depth", "max_entries"},
            set(),
            id="list-tree",
        ),
        pytest.param(
            search_text_tool,
            {"query", "globs", "max_results"},
            {"query"},
            id="search-text",
        ),
        pytest.param(
            read_code_tool,
            {"relative_path", "start_line", "end_line"},
            {"relative_path", "start_line", "end_line"},
            id="read-code",
        ),
        pytest.param(
            list_symbols_tool,
            {"relative_path", "max_results"},
            {"relative_path"},
            id="list-symbols",
        ),
        pytest.param(
            inspect_imports_tool,
            {"relative_path", "max_results"},
            {"relative_path"},
            id="inspect-imports",
        ),
        pytest.param(
            find_symbol_definition_tool,
            {"name", "relative_path", "max_files", "max_results"},
            {"name"},
            id="find-symbol-definition",
        ),
        pytest.param(
            find_references_tool,
            {"name", "relative_path", "max_files", "max_results"},
            {"name"},
            id="find-references",
        ),
    ],
)
def test_repository_tool_schema_hides_repo_path(
    repository_tool: BaseTool,
    expected_properties: set[str],
    expected_required: set[str],
) -> None:
    """Only safe, model-controlled arguments should appear in each schema."""

    schema = repository_tool.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == expected_properties
    assert set(schema.get("required", [])) == expected_required
    assert "repo_path" not in schema["properties"]

    internal_schema = repository_tool.get_input_schema().model_json_schema()
    assert "repo_path" in internal_schema["properties"]


@pytest.mark.parametrize(
    ("repository_tool", "arguments"),
    [
        (list_tree_tool, {}),
        (search_text_tool, {"query": "needle"}),
        (
            read_code_tool,
            {"relative_path": "app.py", "start_line": 1, "end_line": 1},
        ),
        (list_symbols_tool, {"relative_path": "app.py"}),
        (inspect_imports_tool, {"relative_path": "app.py"}),
        (find_symbol_definition_tool, {"name": "Runner"}),
        (find_references_tool, {"name": "Runner"}),
    ],
)
def test_model_visible_tools_cannot_execute_outside_gateway(
    tmp_path,
    repository_tool: BaseTool,
    arguments: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="Tool Gateway"):
        repository_tool.func(repo_path=tmp_path, **arguments)
