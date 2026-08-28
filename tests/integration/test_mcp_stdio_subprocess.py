"""Actual stdio subprocess interoperability smoke for the installed MCP SDK."""

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_stdio_subprocess_lists_and_calls_high_level_tool() -> None:
    async def scenario() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_ROOT / "tests/mcp_stdio_driver.py")],
            cwd=str(PROJECT_ROOT),
        )
        async with Client(server, read_timeout_seconds=20) as client:
            listed = await client.list_tools()
            result = await client.call_tool(
                "get_localization_result",
                {"thread_id": "stdio-thread"},
            )

        assert {tool.name for tool in listed.tools} == {
            "start_investigation",
            "get_localization_result",
            "get_code_graph",
            "expand_symbol",
            "get_context_pack",
            "resume_investigation",
            "get_evaluator_report",
        }
        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["thread_id"] == "stdio-thread"

    asyncio.run(scenario())
