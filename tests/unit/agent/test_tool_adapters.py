"""Unit tests for the LLM-visible repository tool schemas."""

import pytest
from langchain_core.tools import BaseTool

from repocompass.agent.tool_adapters import (
    list_tree_tool,
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
