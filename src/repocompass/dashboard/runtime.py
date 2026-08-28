"""Credential-free, loopback-only runtime for the RepoCompass dashboard."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import uvicorn
from langchain_core.language_models import BaseChatModel

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.store import MemoryStoreError, open_memory_store
from repocompass.models import WorkflowModels
from repocompass.pricing import load_pricing_table_file
from repocompass.storage.raw_store import RawStoreError, open_raw_run_store

DEFAULT_DASHBOARD_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")
DEFAULT_DASHBOARD_RAW_STORE_DATABASE = Path(".repocompass/raw_runs.sqlite")
DEFAULT_DASHBOARD_MEMORY_DATABASE = Path(".repocompass/memory.sqlite")
DEFAULT_DASHBOARD_TRACE_DIRECTORY = Path(".repocompass/traces")
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8766
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class _InspectionOnlyModel:
    """Compile graph readers while making accidental model execution fail closed."""

    def with_structured_output(self, *_: Any, **__: Any) -> "_InspectionOnlyModel":
        return self

    def bind_tools(self, *_: Any, **__: Any) -> "_InspectionOnlyModel":
        return self

    def invoke(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("dashboard inspection runtime cannot invoke models")


def create_inspection_models() -> WorkflowModels:
    """Build graph-compatible placeholders without loading provider credentials."""

    model = cast(BaseChatModel, _InspectionOnlyModel())
    return WorkflowModels.from_single_model(model)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-dashboard",
        description=(
            "Serve the repository-bound RepoCompass investigation dashboard on "
            "a loopback interface. No provider credentials are required for reads."
        ),
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="trusted repository root selected by the local operator",
    )
    parser.add_argument(
        "--host",
        choices=_LOOPBACK_HOSTS,
        default=DEFAULT_DASHBOARD_HOST,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=DEFAULT_DASHBOARD_CHECKPOINT_DATABASE,
    )
    parser.add_argument(
        "--raw-store-db",
        type=Path,
        default=DEFAULT_DASHBOARD_RAW_STORE_DATABASE,
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=DEFAULT_DASHBOARD_MEMORY_DATABASE,
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_DASHBOARD_TRACE_DIRECTORY,
    )
    parser.add_argument("--pricing-file", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Open local stores and serve the read-only dashboard until shutdown."""

    parser = create_parser()
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")

    try:
        pricing_table = (
            None
            if arguments.pricing_file is None
            else load_pricing_table_file(arguments.pricing_file)
        )
        with (
            open_sqlite_checkpointer(arguments.checkpoint_db) as checkpointer,
            open_raw_run_store(arguments.raw_store_db) as raw_run_store,
            open_memory_store(arguments.memory_db) as memory_store,
        ):
            runner = AgentRunner(
                create_inspection_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_run_store,
                memory_store=memory_store,
                pricing_table=pricing_table,
            )
            app = create_dashboard_app(
                DashboardQueryService(
                    runner,
                    raw_run_store=raw_run_store,
                    trace_store=JsonlTraceStore(arguments.trace_dir),
                    repository_root=arguments.repo_path,
                    memory_store=memory_store,
                    pricing_table=pricing_table,
                )
            )
            uvicorn.run(
                app,
                host=arguments.host,
                port=arguments.port,
                log_level=arguments.log_level,
                access_log=True,
                date_header=False,
                server_header=False,
                proxy_headers=False,
                forwarded_allow_ips="",
                limit_concurrency=32,
                timeout_keep_alive=5,
            )
    except (
        MemoryStoreError,
        OSError,
        RawStoreError,
        ValueError,
        sqlite3.Error,
    ) as error:
        print(f"repocompass-dashboard: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
