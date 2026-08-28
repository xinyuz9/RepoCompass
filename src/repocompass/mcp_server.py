"""Official MCP adapter for RepoCompass's high-level investigation service."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanDecisionAction,
    HumanReviewPolicy,
)
from repocompass.agent.runner import (
    AgentRunner,
    DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_NO_PROGRESS_ROUNDS,
    DEFAULT_MAX_PLAN_REVISIONS,
    DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
)
from repocompass.agent.context_builder import DEFAULT_CONTEXT_BUDGET_TOKENS
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.memory.store import MemoryStoreError, open_memory_store
from repocompass.models import create_workflow_models
from repocompass.pricing import load_pricing_table_file
from repocompass.service.contracts import (
    CodeGraphView,
    ContextPackView,
    EvaluatorReportView,
    ExpandedSymbolView,
    ExpandSymbolRequest,
    InvestigationOperationResult,
    LocalizationResultView,
    ResumeInvestigationRequest,
    StartInvestigationRequest,
    ThreadArtifactRequest,
)
from repocompass.service.investigation import (
    InvestigationService,
    InvestigationServiceError,
)
from repocompass.service.operation_store import (
    ServiceOperationStoreError,
    open_service_operation_store,
)
from repocompass.storage.raw_store import RawStoreError, open_raw_run_store

DEFAULT_MCP_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")
DEFAULT_MCP_RAW_STORE_DATABASE = Path(".repocompass/raw_runs.sqlite")
DEFAULT_MCP_MEMORY_DATABASE = Path(".repocompass/memory.sqlite")
DEFAULT_MCP_OPERATION_DATABASE = Path(".repocompass/service_operations.sqlite")
DEFAULT_MCP_TRACE_DIRECTORY = Path(".repocompass/traces")
MCP_SERVER_VERSION = "0.1.0"
MCP_MAX_REQUEST_BODY_BYTES = 1_048_576

_READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    open_world_hint=False,
)
_IDEMPOTENT_MUTATION_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


class MCPInvestigationService(Protocol):
    def start_investigation(
        self, request: StartInvestigationRequest
    ) -> InvestigationOperationResult: ...

    def resume_investigation(
        self, request: ResumeInvestigationRequest
    ) -> InvestigationOperationResult: ...

    def get_localization_result(
        self, request: ThreadArtifactRequest
    ) -> LocalizationResultView: ...

    def get_code_graph(self, request: ThreadArtifactRequest) -> CodeGraphView: ...

    def expand_symbol(self, request: ExpandSymbolRequest) -> ExpandedSymbolView: ...

    def get_context_pack(self, request: ThreadArtifactRequest) -> ContextPackView: ...

    def get_evaluator_report(
        self, request: ThreadArtifactRequest
    ) -> EvaluatorReportView: ...


_ResultT = TypeVar("_ResultT")


def _invoke_service(operation: Callable[[], _ResultT]) -> _ResultT:
    try:
        return operation()
    except ValidationError as error:
        raise ToolError("tool arguments violate the RepoCompass contract") from error
    except (InvestigationServiceError, ServiceOperationStoreError) as error:
        raise ToolError(str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise ToolError("investigation storage operation failed") from error
    except ValueError as error:
        message = str(error)
        if message.startswith("Checkpoint thread does not exist:"):
            public_message = "investigation thread does not exist"
        elif message.startswith("Human ") or "human-review" in message.casefold():
            public_message = "investigation lifecycle request was rejected"
        else:
            public_message = (
                "investigation request was rejected by the service boundary"
            )
        raise ToolError(public_message) from error


def create_mcp_server(
    service: MCPInvestigationService,
    *,
    repository_root: Path,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> MCPServer:
    """Register exactly seven high-level tools over one shared service."""

    required_methods = (
        "start_investigation",
        "get_localization_result",
        "get_code_graph",
        "expand_symbol",
        "get_context_pack",
        "resume_investigation",
        "get_evaluator_report",
    )
    if not all(callable(getattr(service, name, None)) for name in required_methods):
        raise TypeError("service does not implement the MCP investigation boundary")
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a Path")
    trusted_repository_root = repository_root.expanduser().resolve(strict=True)
    if not trusted_repository_root.is_dir():
        raise ValueError("repository_root must identify a directory")

    server = MCPServer(
        name="RepoCompass",
        title="RepoCompass Repository Investigation",
        description="Bounded, evidence-grounded repository investigation tools.",
        instructions=(
            "Start or resume a bounded RepoCompass investigation, then read its "
            "localization, task-specific graph, context, and evaluator artifacts. "
            "The repository root is bound by the operator at server startup. "
            "Internal lexical/source tools are intentionally unavailable."
        ),
        version=MCP_SERVER_VERSION,
        log_level=log_level,
    )

    @server.tool(annotations=_IDEMPOTENT_MUTATION_ANNOTATIONS)
    def start_investigation(
        request_id: str,
        objective: str,
        max_tool_calls_per_step: int = DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        max_explorer_calls_per_step: int = DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
        max_no_progress_rounds: int = DEFAULT_MAX_NO_PROGRESS_ROUNDS,
        max_plan_revisions: int = DEFAULT_MAX_PLAN_REVISIONS,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
        max_attempt_cost_usd: str | None = None,
        human_review_policy: Literal["never", "before_finalize"] = "never",
        memory_retrieval_enabled: bool = False,
        max_repository_memory_results: int = DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
        max_episodic_memory_results: int = DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
    ) -> InvestigationOperationResult:
        """Start one idempotent investigation inside the operator-bound repository."""

        return _invoke_service(
            lambda: service.start_investigation(
                StartInvestigationRequest(
                    request_id=request_id,
                    repo_path=trusted_repository_root,
                    objective=objective,
                    max_tool_calls_per_step=max_tool_calls_per_step,
                    max_explorer_calls_per_step=max_explorer_calls_per_step,
                    max_no_progress_rounds=max_no_progress_rounds,
                    max_plan_revisions=max_plan_revisions,
                    context_budget_tokens=context_budget_tokens,
                    max_attempt_cost_usd=max_attempt_cost_usd,
                    human_review_policy=HumanReviewPolicy(human_review_policy),
                    memory_retrieval_enabled=memory_retrieval_enabled,
                    max_repository_memory_results=max_repository_memory_results,
                    max_episodic_memory_results=max_episodic_memory_results,
                )
            )
        )

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS)
    def get_localization_result(thread_id: str) -> LocalizationResultView:
        """Get the final file/symbol localization without embedding the full graph."""

        return _invoke_service(
            lambda: service.get_localization_result(
                ThreadArtifactRequest(thread_id=thread_id)
            )
        )

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS)
    def get_code_graph(thread_id: str) -> CodeGraphView:
        """Get the bounded task-specific evidence graph and reading path."""

        return _invoke_service(
            lambda: service.get_code_graph(ThreadArtifactRequest(thread_id=thread_id))
        )

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS)
    def expand_symbol(
        thread_id: str,
        entity_id: str | None = None,
        canonical_name: str | None = None,
        max_edges: int = 20,
    ) -> ExpandedSymbolView:
        """Expand one symbol already present in the result graph; never query source."""

        return _invoke_service(
            lambda: service.expand_symbol(
                ExpandSymbolRequest(
                    thread_id=thread_id,
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    max_edges=max_edges,
                )
            )
        )

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS)
    def get_context_pack(thread_id: str) -> ContextPackView:
        """Get the latest bounded Active Context and complete selection ledger."""

        return _invoke_service(
            lambda: service.get_context_pack(ThreadArtifactRequest(thread_id=thread_id))
        )

    @server.tool(annotations=_IDEMPOTENT_MUTATION_ANNOTATIONS)
    def resume_investigation(
        request_id: str,
        thread_id: str,
        interrupt_id: str | None = None,
        decision: Literal["approve", "revise", "cancel"] | None = None,
        feedback: str | None = None,
    ) -> InvestigationOperationResult:
        """Resume checkpoint work or answer one exact pending Human review."""

        def invoke() -> InvestigationOperationResult:
            if decision is None:
                if feedback is not None:
                    raise ValueError("feedback requires a Human decision")
                human_decision = None
            else:
                human_decision = HumanDecision(
                    action=HumanDecisionAction(decision),
                    feedback=feedback,
                )
            return service.resume_investigation(
                ResumeInvestigationRequest(
                    request_id=request_id,
                    thread_id=thread_id,
                    interrupt_id=interrupt_id,
                    decision=human_decision,
                )
            )

        return _invoke_service(invoke)

    @server.tool(annotations=_READ_ONLY_ANNOTATIONS)
    def get_evaluator_report(thread_id: str) -> EvaluatorReportView:
        """Get the current validated evaluator report and revision counters."""

        return _invoke_service(
            lambda: service.get_evaluator_report(
                ThreadArtifactRequest(thread_id=thread_id)
            )
        )

    return server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-mcp",
        description="Serve RepoCompass's seven high-level tools over MCP.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="trusted repository root bound by the operator, never by the MCP client",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=DEFAULT_MCP_CHECKPOINT_DATABASE,
    )
    parser.add_argument(
        "--raw-store-db",
        type=Path,
        default=DEFAULT_MCP_RAW_STORE_DATABASE,
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=DEFAULT_MCP_MEMORY_DATABASE,
    )
    parser.add_argument(
        "--operation-db",
        type=Path,
        default=DEFAULT_MCP_OPERATION_DATABASE,
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_MCP_TRACE_DIRECTORY,
    )
    parser.add_argument("--pricing-file", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run local stdio by default, or loopback-only Streamable HTTP."""

    parser = create_parser()
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if arguments.transport == "streamable-http" and arguments.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        parser.error("the v0.1 MCP HTTP transport is loopback-only")

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
            open_service_operation_store(arguments.operation_db) as operation_store,
        ):
            runner_dependencies: dict[str, object] = {
                "memory_store": memory_store,
            }
            if pricing_table is not None:
                runner_dependencies["pricing_table"] = pricing_table
            runner = AgentRunner(
                create_workflow_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_run_store,
                **runner_dependencies,
            )
            server = create_mcp_server(
                InvestigationService(
                    runner,
                    operation_store=operation_store,
                    trace_sink=JsonlTraceSink(),
                    trace_directory=arguments.trace_dir,
                ),
                repository_root=arguments.repo_path,
                log_level=arguments.log_level,
            )
            if arguments.transport == "stdio":
                server.run("stdio")
            else:
                server.run(
                    "streamable-http",
                    host=arguments.host,
                    port=arguments.port,
                    streamable_http_path="/mcp",
                    max_request_body_size=MCP_MAX_REQUEST_BODY_BYTES,
                )
    except (
        MemoryStoreError,
        OSError,
        RawStoreError,
        ServiceOperationStoreError,
        ValueError,
        sqlite3.Error,
    ) as error:
        print(f"repocompass-mcp: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
