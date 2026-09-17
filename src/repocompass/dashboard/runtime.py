"""Credential-free, loopback-only runtime for the RepoCompass dashboard."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import uvicorn
from langchain_core.language_models import BaseChatModel

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.cancellation import DashboardCancellationService
from repocompass.storage.cancellation import CancellationError, open_cancellation_store
from repocompass.dashboard.demo import DashboardDemoError, create_dashboard_demo
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.store import MemoryStoreError, open_memory_store
from repocompass.models import WorkflowModels, create_workflow_models
from repocompass.pricing import load_pricing_table_file
from repocompass.service import (
    InvestigationService,
    ServiceOperationStoreError,
    open_service_operation_store,
)
from repocompass.storage.raw_store import RawStoreError, open_raw_run_store
from repocompass.tools.browser import (
    BrowserPolicy,
    add_browser_arguments,
    validate_browser_runtime,
)

DEFAULT_DASHBOARD_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")
DEFAULT_DASHBOARD_RAW_STORE_DATABASE = Path(".repocompass/raw_runs.sqlite")
DEFAULT_DASHBOARD_MEMORY_DATABASE = Path(".repocompass/memory.sqlite")
DEFAULT_DASHBOARD_OPERATION_DATABASE = Path(".repocompass/service_operations.sqlite")
DEFAULT_DASHBOARD_CANCELLATION_DATABASE = Path(".repocompass/cancellations.sqlite")
DEFAULT_DASHBOARD_TRACE_DIRECTORY = Path(".repocompass/traces")
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8766
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_DEMO_CONFLICT_OPTIONS = frozenset(
    {
        "--repo-path",
        "--checkpoint-db",
        "--raw-store-db",
        "--memory-db",
        "--operation-db",
        "--cancellation-db",
        "--trace-dir",
        "--pricing-file",
        "--enable-lifecycle-mutations",
        "--enable-memory-mutations",
        "--browser-page",
    }
)


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
            "a loopback interface. Reads need no provider credentials; lifecycle "
            "and memory mutations are absent unless explicitly enabled."
        ),
    )
    add_browser_arguments(parser)
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=False,
        help="trusted repository root selected by the local operator",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "serve a sanitized, read-only, ephemeral investigation generated "
            "through the production graph without provider credentials"
        ),
    )
    parser.add_argument(
        "--host",
        choices=_LOOPBACK_HOSTS,
        default=DEFAULT_DASHBOARD_HOST,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument(
        "--demo-turns",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="number of scripted production-graph turns in the ephemeral demo",
    )
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
        "--operation-db",
        type=Path,
        default=DEFAULT_DASHBOARD_OPERATION_DATABASE,
        help="durable idempotency ledger used only by lifecycle mutations",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_DASHBOARD_TRACE_DIRECTORY,
    )
    parser.add_argument(
        "--cancellation-db",
        type=Path,
        default=DEFAULT_DASHBOARD_CANCELLATION_DATABASE,
        help="independent durable control ledger for lifecycle worker cancellation",
    )
    parser.add_argument("--pricing-file", type=Path)
    parser.add_argument(
        "--enable-lifecycle-mutations",
        action="store_true",
        help=(
            "load configured workflow models and expose idempotent start, "
            "resume, follow-up, and pending-review commands"
        ),
    )
    parser.add_argument(
        "--enable-memory-mutations",
        action="store_true",
        help="expose only exact, confirmed, audited memory deletion",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Open local stores and serve a read-first dashboard until shutdown."""

    parser = create_parser()
    argument_vector = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(argument_vector)
    if arguments.browser_page and not arguments.enable_lifecycle_mutations:
        parser.error("--browser-page requires --enable-lifecycle-mutations")
    try:
        browser_policy = (
            BrowserPolicy(tuple(arguments.browser_page))
            if arguments.browser_page
            else None
        )
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if arguments.demo_turns != 1 and not arguments.demo:
        parser.error("--demo-turns requires --demo")
    if arguments.demo:
        conflicts = sorted(
            option
            for option in _DEMO_CONFLICT_OPTIONS
            if _option_was_supplied(argument_vector, option)
        )
        if conflicts:
            parser.error("--demo cannot be combined with " + ", ".join(conflicts))
    elif arguments.repo_path is None:
        parser.error("--repo-path is required unless --demo is selected")

    try:
        with ExitStack() as stack:
            if browser_policy is not None:
                validate_browser_runtime()
            if arguments.demo:
                demo_root = Path(
                    stack.enter_context(
                        TemporaryDirectory(prefix="repocompass-dashboard-demo-")
                    )
                )
                demo = (
                    create_dashboard_demo(demo_root)
                    if arguments.demo_turns == 1
                    else create_dashboard_demo(demo_root, turns=arguments.demo_turns)
                )
                repository_path = demo.repository_path
                checkpoint_database = demo.checkpoint_database
                raw_store_database = demo.raw_store_database
                memory_database = demo.memory_database
                trace_directory = demo.trace_directory
                pricing_table = demo.pricing_table
                print(
                    "repocompass-dashboard: sanitized ephemeral demo loaded; "
                    "provider calls and mutation routes are disabled",
                    file=sys.stderr,
                )
            else:
                repository_path = arguments.repo_path
                checkpoint_database = arguments.checkpoint_db
                raw_store_database = arguments.raw_store_db
                memory_database = arguments.memory_db
                trace_directory = arguments.trace_dir
                pricing_table = (
                    None
                    if arguments.pricing_file is None
                    else load_pricing_table_file(arguments.pricing_file)
                )
            if repository_path is None:  # pragma: no cover - parser owns this branch
                raise ValueError("repository path is required")
            checkpointer = stack.enter_context(
                open_sqlite_checkpointer(checkpoint_database)
            )
            raw_run_store = stack.enter_context(open_raw_run_store(raw_store_database))
            memory_store = stack.enter_context(open_memory_store(memory_database))
            operation_store = (
                stack.enter_context(
                    open_service_operation_store(arguments.operation_db)
                )
                if arguments.enable_lifecycle_mutations
                else None
            )
            models = (
                create_workflow_models()
                if arguments.enable_lifecycle_mutations
                else create_inspection_models()
            )
            cancellation_store = (
                stack.enter_context(open_cancellation_store(arguments.cancellation_db))
                if arguments.enable_lifecycle_mutations
                else None
            )
            browser_dependencies = (
                {} if browser_policy is None else {"browser_policy": browser_policy}
            )
            runner = AgentRunner(
                models,
                checkpointer=checkpointer,
                raw_run_store=raw_run_store,
                memory_store=memory_store,
                pricing_table=pricing_table,
                cancellation_store=cancellation_store,
                **browser_dependencies,
            )
            query_service = DashboardQueryService(
                runner,
                raw_run_store=raw_run_store,
                trace_store=JsonlTraceStore(trace_directory),
                repository_root=repository_path,
                memory_store=memory_store,
                pricing_table=pricing_table,
                lifecycle_mutations_enabled=arguments.enable_lifecycle_mutations,
                cancellation_enabled=cancellation_store is not None,
                memory_mutations_enabled=arguments.enable_memory_mutations,
                demo_mode=arguments.demo,
                static_browser_page_ids=(
                    ()
                    if browser_policy is None
                    else tuple(page.page_id for page in browser_policy.pages)
                ),
            )
            lifecycle_service = (
                InvestigationService(
                    runner,
                    operation_store=operation_store,
                    raw_run_store=raw_run_store,
                    pricing_table=pricing_table,
                    trace_sink=JsonlTraceSink(),
                    trace_directory=trace_directory,
                )
                if operation_store is not None
                else None
            )
            command_service = (
                DashboardCommandService(
                    query_service,
                    lifecycle_service=lifecycle_service,
                    memory_store=(
                        memory_store if arguments.enable_memory_mutations else None
                    ),
                )
                if (
                    arguments.enable_lifecycle_mutations
                    or arguments.enable_memory_mutations
                )
                else None
            )
            app = create_dashboard_app(
                query_service,
                commands=command_service,
                cancellation=(
                    None
                    if cancellation_store is None
                    else DashboardCancellationService(
                        cancellation_store,
                        raw_run_store,
                        repository_path,
                    )
                ),
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
        CancellationError,
        DashboardDemoError,
        MemoryStoreError,
        OSError,
        RawStoreError,
        ServiceOperationStoreError,
        ValueError,
        sqlite3.Error,
    ) as error:
        print(f"repocompass-dashboard: error: {error}", file=sys.stderr)
        return 1
    return 0


def _option_was_supplied(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


if __name__ == "__main__":
    raise SystemExit(main())
