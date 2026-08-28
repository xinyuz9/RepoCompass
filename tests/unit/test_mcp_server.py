"""In-memory MCP protocol tests for the seven public investigation tools."""

import asyncio
from pathlib import Path

from mcp import Client
import pytest

from repocompass.mcp_server import create_mcp_server, main
from tests.mcp_stub_service import StubInvestigationService

PUBLIC_TOOLS = {
    "start_investigation",
    "get_localization_result",
    "get_code_graph",
    "expand_symbol",
    "get_context_pack",
    "resume_investigation",
    "get_evaluator_report",
}


def test_mcp_server_requires_an_existing_operator_bound_repository(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        create_mcp_server(
            StubInvestigationService(),
            repository_root=tmp_path / "missing",
        )


def test_mcp_http_transport_rejects_non_loopback_bind_before_model_setup(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--repo-path",
                str(tmp_path),
                "--transport",
                "streamable-http",
                "--host",
                "0.0.0.0",
            ]
        )

    assert raised.value.code == 2


def test_mcp_lists_only_high_level_tools_with_safety_annotations(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with Client(
            create_mcp_server(
                StubInvestigationService(),
                repository_root=tmp_path,
            )
        ) as client:
            listed = await client.list_tools()

        tools = {tool.name: tool for tool in listed.tools}
        assert set(tools) == PUBLIC_TOOLS
        for name in {"start_investigation", "resume_investigation"}:
            assert tools[name].annotations is not None
            assert tools[name].annotations.read_only_hint is False
            assert tools[name].annotations.destructive_hint is False
            assert tools[name].annotations.idempotent_hint is True
            assert tools[name].annotations.open_world_hint is False
        assert (
            "repo_path" not in tools["start_investigation"].input_schema["properties"]
        )
        for name in PUBLIC_TOOLS - {
            "start_investigation",
            "resume_investigation",
        }:
            assert tools[name].annotations is not None
            assert tools[name].annotations.read_only_hint is True
            assert tools[name].annotations.open_world_hint is False
        assert "list_tree" not in tools
        assert "search_text" not in tools
        assert "read_code" not in tools

    asyncio.run(scenario())


def test_mcp_start_and_getter_return_valid_structured_content(tmp_path: Path) -> None:
    service = StubInvestigationService()

    async def scenario() -> None:
        async with Client(
            create_mcp_server(service, repository_root=tmp_path)
        ) as client:
            started = await client.call_tool(
                "start_investigation",
                {
                    "request_id": "mcp-start-1",
                    "objective": "Locate authentication.",
                    "max_tool_calls_per_step": 7,
                },
            )
            localization = await client.call_tool(
                "get_localization_result",
                {"thread_id": "mcp-thread"},
            )

        assert started.is_error is False
        assert started.structured_content is not None
        assert started.structured_content["thread_id"] == "mcp-thread"
        assert localization.is_error is False
        assert localization.structured_content is not None
        assert localization.structured_content["availability"] == "available"
        assert localization.structured_content["result"]["relevant_files"] == [
            "src/auth.py"
        ]

    asyncio.run(scenario())
    assert len(service.start_requests) == 1
    assert service.start_requests[0].repo_path == tmp_path.resolve()
    assert service.start_requests[0].max_tool_calls_per_step == 7


def test_mcp_known_errors_are_model_readable_and_do_not_leak_tracebacks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with Client(
            create_mcp_server(
                StubInvestigationService(),
                repository_root=tmp_path,
            )
        ) as client:
            result = await client.call_tool(
                "expand_symbol",
                {
                    "thread_id": "mcp-thread",
                    "canonical_name": "src.auth.authenticate",
                },
            )

        assert result.is_error is True
        rendered = "\n".join(
            block.text for block in result.content if block.type == "text"
        )
        assert "code graph is unavailable" in rendered
        assert "Traceback" not in rendered
        assert "/home/" not in rendered

    asyncio.run(scenario())
