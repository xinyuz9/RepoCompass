"""Standalone stdio MCP server with no model or credential dependency."""

from pathlib import Path

from repocompass.mcp_server import create_mcp_server
from mcp_stub_service import StubInvestigationService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    create_mcp_server(
        StubInvestigationService(),
        repository_root=PROJECT_ROOT,
    ).run("stdio")
