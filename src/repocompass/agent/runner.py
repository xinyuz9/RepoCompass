"""Orchestration boundary for one RepoCompass investigation run.

The Runner will connect trusted input, RepoCompassState construction, LangGraph
execution, and the local domain trace. It must reuse graph behavior rather than
duplicating node routing, tool execution, or budget enforcement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from repocompass.agent.context import ContextItem, ContextPack
from repocompass.agent.context_builder import DEFAULT_CONTEXT_BUDGET_TOKENS
from repocompass.agent.cost import (
    AttemptUsageCostReport,
    PricingTable,
    validate_cost_budget_usd,
)
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.invocation import RawRunRecorder
from repocompass.agent.lifecycle import (
    ContinueRequest,
    ExecutionStatus,
    HumanDecision,
    HumanResponseRequest,
    HumanReviewPolicy,
    HumanReviewRequest,
    decode_human_review_interrupt,
    encode_human_decision,
)
from repocompass.agent.result import FinalResult, revalidate_final_result
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.structural_artifacts import InvestigationStructure
from repocompass.agent.state import (
    RepoCompassState,
    create_continuation_update,
    create_initial_state,
)
from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.models import WorkflowModels
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunStore,
)
from repocompass.tools.contracts import (
    NoProgressDecision,
    ToolAuditRecord,
    decode_no_progress_decision,
    decode_tool_audit_record,
)
from repocompass.tools.gateway import ToolExecutionContext

if TYPE_CHECKING:
    from repocompass.memory.consolidation import EpisodeConsolidationResult
    from repocompass.memory.store import MemoryStore

RunIdFactory = Callable[[], str]
ThreadIdFactory = Callable[[], str]
EpisodeClockFactory = Callable[[], int]
ExecutionMode = Literal["start", "resume", "continue", "respond"]
DEFAULT_MAX_TOOL_CALLS_PER_STEP = 16
DEFAULT_MAX_EXPLORER_CALLS_PER_STEP = 12
DEFAULT_MAX_PLAN_REVISIONS = 2
DEFAULT_MAX_NO_PROGRESS_ROUNDS = 3
DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS = 5
DEFAULT_MAX_EPISODIC_MEMORY_RESULTS = 5

_SENSITIVE_TRACE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "password",
        "proxy_authorization",
        "reasoning",
        "reasoning_content",
        "refresh_token",
        "secret",
        "token",
    }
)


def generate_run_id() -> str:
    """Return a new opaque identifier for one investigation run.

    Returns:
        A UUID string suitable for correlating every event in one
        :class:`RunTrace`.

    Side effects:
        Generates a random UUID locally. It performs no file or network I/O.
    """

    return str(uuid4())


def generate_thread_id() -> str:
    """Return a new opaque identifier for one logical investigation thread."""

    return str(uuid4())


def current_time_ms() -> int:
    """Return the current Unix timestamp in integer milliseconds."""

    return time_ns() // 1_000_000


def _normalize_identifier(value: object, *, name: str) -> str:
    """Validate and normalize one generated run or thread identifier."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")

    return normalized


def _thread_config(
    thread_id: str,
    *,
    run_id: str | None = None,
    turn_number: int | None = None,
) -> RunnableConfig:
    """Build the LangGraph configuration for one checkpoint thread."""

    configurable: dict[str, str | int] = {"thread_id": thread_id}
    if run_id is not None:
        configurable["repocompass_run_id"] = run_id
    if turn_number is not None:
        configurable["repocompass_turn_number"] = turn_number
    return {"configurable": configurable}


def resolve_repository_path(repo_path: Path) -> Path:
    """Resolve one repository root and require an existing directory."""

    resolved_path = repo_path.expanduser().resolve(strict=True)
    if not resolved_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {resolved_path}")

    return resolved_path


def _is_sensitive_trace_key(key: str) -> bool:
    """Return whether one mapping key must be omitted from local traces."""

    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_TRACE_KEYS or normalized.endswith(
        (
            "_access_token",
            "_api_key",
            "_password",
            "_refresh_token",
            "_secret",
        )
    )


def _to_json_value(value: object) -> JsonValue:
    """Convert supported graph-update values to JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseModel):
        return _to_json_value(value.model_dump(mode="json"))

    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if _is_sensitive_trace_key(normalized_key):
                continue
            normalized[normalized_key] = _to_json_value(item)
        return normalized

    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    raise TypeError(
        f"Graph update contains non-JSON-compatible value: {type(value).__name__}"
    )


def _context_pack_trace_metadata(pack: ContextPack) -> dict[str, JsonValue]:
    return {
        "pack_id": pack.pack_id,
        "build_id": pack.build_report.build_id,
        "budget_tokens": pack.build_report.budget_tokens,
        "estimated_tokens": pack.estimated_tokens,
        "selected_count": pack.build_report.selected_count,
        "excluded_count": pack.build_report.excluded_count,
        "items": [
            {
                "item_id": item.item_id,
                "source_id": item.source.source_id,
                "source_kind": item.source.kind.value,
                "raw_artifact_id": item.source.raw_artifact_id,
                "section": item.section.value,
                "verification_status": item.verification_status.value,
                "estimated_tokens": item.estimated_tokens,
                "pinned": item.pinned,
                "location": (
                    None
                    if item.source.location is None
                    else item.source.location.model_dump(mode="json")
                ),
                "memory": (
                    None
                    if item.source.memory is None
                    else item.source.memory.model_dump(mode="json")
                ),
            }
            for item in pack.items
        ],
        "decisions": [
            {
                "item_id": decision.item_id,
                "outcome": decision.outcome.value,
                "reason": decision.reason.value,
                "estimated_tokens": decision.estimated_tokens,
                "selected_position": decision.selected_position,
            }
            for decision in pack.build_report.decisions
        ],
    }


def _structure_trace_metadata(
    artifact: InvestigationStructure,
) -> dict[str, JsonValue]:
    edges = {edge.assertion_id: edge for edge in artifact.graph.edges}
    return {
        "artifact_id": artifact.artifact_id,
        "repository_id": artifact.repository_id,
        "commit_sha": artifact.commit_sha,
        "indexed_paths": list(artifact.indexed_paths),
        "graph_id": artifact.graph.graph_id,
        "graph_node_count": len(artifact.graph.nodes),
        "graph_edge_count": len(artifact.graph.edges),
        "reading_path_id": artifact.reading_path.reading_path_id,
        "reading_step_count": len(artifact.reading_path.steps),
        "claim_lineage_count": len(artifact.reading_path.claim_lineages),
        "slice_truncated": artifact.reading_path.graph_slice.truncated,
        "slice_truncation_reasons": [
            reason.value
            for reason in artifact.reading_path.graph_slice.truncation_reasons
        ],
        "relevant_symbols": [
            {
                "binding_id": item.binding_id,
                "canonical_name": item.canonical_name,
                "entity_id": item.entity_id,
                "path": item.path,
            }
            for item in artifact.relevant_symbols
        ],
        "candidate_relations": [
            {
                "assertion_id": assertion_id,
                "relation": edges[assertion_id].relation.value,
                "verification_status": edges[assertion_id].verification_status.value,
            }
            for assertion_id in artifact.candidate_relation_assertion_ids
        ],
    }


def _normalize_update(update: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return a JSON-compatible copy of one node's partial state update."""

    normalized: dict[str, JsonValue] = {}
    for key, value in update.items():
        if key == "memory_context_items":
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, ContextItem) for item in value
            ):
                raise TypeError("Graph memory_context_items are invalid")
            normalized[key] = [
                {
                    "item_id": item.item_id,
                    "source_id": item.source.source_id,
                    "memory": (
                        None
                        if item.source.memory is None
                        else item.source.memory.model_dump(mode="json")
                    ),
                }
                for item in value
            ]
            continue
        if key == "human_feedback":
            normalized["human_feedback_provided"] = value is not None
            continue
        if key == "active_context_pack":
            if value is None:
                normalized[key] = None
            elif isinstance(value, ContextPack):
                normalized[key] = _context_pack_trace_metadata(value)
            else:
                raise TypeError(
                    "Graph active_context_pack is not a ContextPack or None"
                )
            continue
        if key == "investigation_structure":
            if value is None:
                normalized[key] = None
            elif isinstance(value, InvestigationStructure):
                normalized[key] = _structure_trace_metadata(value)
            else:
                raise TypeError(
                    "Graph investigation_structure is not an "
                    "InvestigationStructure or None"
                )
            continue
        if key == "final_result":
            if value is None:
                normalized[key] = None
            elif isinstance(value, FinalResult):
                public_fields = value.model_dump(
                    mode="json",
                    exclude={"structure"},
                )
                normalized_result = _to_json_value(public_fields)
                if not isinstance(normalized_result, dict):
                    raise TypeError("Graph final_result did not serialize to an object")
                normalized_result["structure"] = (
                    None
                    if value.structure is None
                    else _structure_trace_metadata(value.structure)
                )
                normalized[key] = normalized_result
            else:
                raise TypeError("Graph final_result is not a FinalResult or None")
            continue
        if _is_sensitive_trace_key(key):
            continue
        normalized[key] = _to_json_value(value)
    return normalized


class RunRequest(BaseModel):
    """Validated, trusted input required to start one investigation.

    Attributes:
        repo_path: Application-supplied repository root. The Runner must resolve
            it and reject paths that are not existing directories.
        objective: Non-empty repository investigation question.
        max_tool_calls_per_step: Positive tool-call budget for one active plan
            step.
        max_explorer_calls_per_step: Positive Explorer invocation budget for one
            active plan step.
        max_no_progress_rounds: Positive consecutive stalled-round limit for one
            active plan step.
        max_plan_revisions: Positive global budget for model-based plan
            revisions.
        max_attempt_cost_usd: Optional canonical positive USD limit for each
            lifecycle attempt, enforced from confirmed provider usage.
        context_budget_tokens: Positive hard budget for dynamic active-context
            JSONL records in each model invocation.
        human_review_policy: Optional final-candidate review gate.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    repo_path: Path
    objective: str = Field(min_length=1)
    max_tool_calls_per_step: int = Field(gt=0)
    max_explorer_calls_per_step: int = Field(gt=0)
    max_no_progress_rounds: int = Field(default=DEFAULT_MAX_NO_PROGRESS_ROUNDS, gt=0)
    max_plan_revisions: int = Field(default=DEFAULT_MAX_PLAN_REVISIONS, gt=0)
    context_budget_tokens: int = Field(default=DEFAULT_CONTEXT_BUDGET_TOKENS, gt=0)
    max_attempt_cost_usd: str | None = None
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.NEVER
    memory_retrieval_enabled: bool = False
    max_repository_memory_results: int = Field(
        default=DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
        ge=1,
        le=10,
    )
    max_episodic_memory_results: int = Field(
        default=DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
        ge=1,
        le=10,
    )

    @field_validator("max_attempt_cost_usd", mode="before")
    @classmethod
    def require_canonical_attempt_cost_budget(cls, value: object) -> object:
        if value is None:
            return None
        try:
            return validate_cost_budget_usd(value)
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error


class ResumeRequest(BaseModel):
    """Validated input required to resume one persisted investigation."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    thread_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Completed public result of one Runner invocation.

    Attributes:
        thread_id: Stable checkpoint identity of the logical investigation.
        final_result: Structured report for a completed attempt, otherwise None.
        termination_status: Final accepted, exhausted, or cancelled status.
        step_stop_reason: Reason the final active step stopped exploring, if any.
        trace: Ordered local domain trace for this run.
        execution_status: Whether this attempt completed or paused at review.
        human_review_request: Pending public review payload when interrupted.
        interrupt_id: Exact pending interrupt identity when interrupted.

    ``RunOutcome`` deliberately omits the complete RepoCompassState so persistent
    message history and internal orchestration data do not become public API.
    """

    thread_id: str
    final_result: FinalResult | None
    termination_status: TerminationStatus | None
    step_stop_reason: str | None
    trace: RunTrace
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED
    human_review_request: HumanReviewRequest | None = None
    interrupt_id: str | None = None
    tool_audit_records: tuple[ToolAuditRecord, ...] = ()
    no_progress_decision: NoProgressDecision | None = None
    consecutive_no_progress_rounds: int = 0
    usage_cost_report: AttemptUsageCostReport | None = None
    episode_consolidation: EpisodeConsolidationResult | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous combinations of terminal and interrupt fields."""

        if self.episode_consolidation is not None:
            from repocompass.memory.consolidation import EpisodeConsolidationResult

            if not isinstance(self.episode_consolidation, EpisodeConsolidationResult):
                raise TypeError(
                    "episode_consolidation must be an EpisodeConsolidationResult"
                )
        if self.execution_status is ExecutionStatus.COMPLETED:
            if self.final_result is None or self.termination_status is None:
                raise ValueError(
                    "Completed outcome requires final_result and termination_status"
                )
            if self.human_review_request is not None:
                raise ValueError(
                    "Completed outcome must not contain a human review request"
                )
            if self.interrupt_id is not None:
                raise ValueError("Completed outcome must not contain an interrupt ID")
            return

        if self.execution_status is ExecutionStatus.INTERRUPTED:
            if self.human_review_request is None:
                raise ValueError("Interrupted outcome requires a human review request")
            if not self.interrupt_id:
                raise ValueError("Interrupted outcome requires an interrupt ID")
            if self.final_result is not None or self.termination_status is not None:
                raise ValueError(
                    "Interrupted outcome must not contain terminal result fields"
                )
            if (
                self.episode_consolidation is not None
                and self.episode_consolidation.episode is not None
            ):
                raise ValueError(
                    "Interrupted outcome cannot contain a consolidated episode"
                )
            return

        raise ValueError(f"Unknown execution status: {self.execution_status!r}")


def _read_step_stop_reason(state: Mapping[str, object]) -> str | None:
    """Read and validate the optional active-step stop reason."""

    step_stop_reason = state.get("step_stop_reason")
    if step_stop_reason is not None and not isinstance(step_stop_reason, str):
        raise TypeError("Graph step_stop_reason is not a string or None")

    return step_stop_reason


def _read_attempt_tool_audits(
    state: Mapping[str, object],
    *,
    run_id: str,
) -> tuple[ToolAuditRecord, ...]:
    records = state.get("tool_audit_records", [])
    if not isinstance(records, list):
        raise TypeError("Graph tool_audit_records are invalid")
    try:
        decoded_records = tuple(
            decode_tool_audit_record(payload) for payload in records
        )
    except ValueError as error:
        raise TypeError("Graph tool_audit_records are invalid") from error
    return tuple(record for record in decoded_records if record.run_id == run_id)


def _read_progress_metadata(
    state: Mapping[str, object],
) -> tuple[NoProgressDecision | None, int]:
    stalled_rounds = state.get("consecutive_no_progress_rounds", 0)
    if (
        not isinstance(stalled_rounds, int)
        or isinstance(stalled_rounds, bool)
        or stalled_rounds < 0
    ):
        raise TypeError("Graph consecutive_no_progress_rounds are invalid")
    payload = state.get("no_progress_decision")
    if payload is None:
        return None, stalled_rounds
    try:
        decision = decode_no_progress_decision(payload)
    except ValueError as error:
        raise TypeError("Graph no_progress_decision is invalid") from error
    return decision, stalled_rounds


def _usage_cost_trace_metadata(
    report: AttemptUsageCostReport,
) -> dict[str, JsonValue]:
    """Return a bounded report reference without copying invocation payloads."""

    total = report.breakdowns[0]
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "pricing_table_id": report.pricing_table_id,
        "pricing_version": report.pricing_version,
        "pricing_effective_from": report.pricing_effective_from,
        "invocation_count": total.invocation_count,
        "cost_coverage": total.cost.coverage.value,
        "estimated_cost_usd": total.cost.estimated_cost_usd,
        "disclaimer": report.disclaimer,
    }


def _record_terminal_trace_event(
    outcome: RunOutcome,
    *,
    execution_mode: ExecutionMode,
    usage_cost_report: AttemptUsageCostReport | None,
    episode_consolidation: EpisodeConsolidationResult | None,
) -> None:
    """Append the lifecycle terminal only after optional ledger finalization."""

    if outcome.execution_status is ExecutionStatus.INTERRUPTED:
        review_request = outcome.human_review_request
        interrupt_id = outcome.interrupt_id
        if review_request is None or interrupt_id is None:
            raise ValueError("Interrupted outcome is missing review metadata")
        kind = TraceEventKind.RUN_INTERRUPTED
        payload: dict[str, JsonValue] = {
            "thread_id": outcome.thread_id,
            "execution_mode": execution_mode,
            "interrupt_id": interrupt_id,
            "turn_number": review_request.turn_number,
        }
    else:
        if outcome.termination_status is None:
            raise ValueError("Completed outcome is missing termination status")
        kind = TraceEventKind.RUN_FINISHED
        payload = {
            "thread_id": outcome.thread_id,
            "execution_mode": execution_mode,
            "termination_status": outcome.termination_status,
            "step_stop_reason": outcome.step_stop_reason,
        }
    if usage_cost_report is not None:
        payload["usage_cost_report"] = _usage_cost_trace_metadata(usage_cost_report)
    if episode_consolidation is not None:
        payload["episode_consolidation"] = {
            "schema_version": episode_consolidation.schema_version,
            "disposition": episode_consolidation.disposition.value,
            "reason": episode_consolidation.reason.value,
            "episode_id": (
                None
                if episode_consolidation.episode is None
                else episode_consolidation.episode.episode_id
            ),
        }
    outcome.trace.record(kind, payload=payload)


def _build_interrupted_outcome(
    snapshot: StateSnapshot,
    *,
    thread_id: str,
    trace: RunTrace,
    execution_mode: ExecutionMode,
) -> RunOutcome:
    """Map one pending checkpoint interrupt to the public Runner contract."""

    if len(snapshot.interrupts) != 1:
        raise ValueError(
            "Graph paused with an unsupported number of interrupts: "
            f"{len(snapshot.interrupts)}"
        )
    if (
        snapshot.values.get("final_result") is not None
        or snapshot.values.get("termination_status") is not None
    ):
        raise ValueError("Interrupted graph contains terminal result fields")

    pending_interrupt = snapshot.interrupts[0]
    interrupt_id = _normalize_identifier(
        pending_interrupt.id,
        name="interrupt_id",
    )
    review_request = decode_human_review_interrupt(pending_interrupt.value)
    progress_decision, stalled_rounds = _read_progress_metadata(snapshot.values)

    return RunOutcome(
        thread_id=thread_id,
        final_result=None,
        termination_status=None,
        step_stop_reason=_read_step_stop_reason(snapshot.values),
        trace=trace,
        execution_status=ExecutionStatus.INTERRUPTED,
        human_review_request=review_request,
        interrupt_id=interrupt_id,
        tool_audit_records=_read_attempt_tool_audits(
            snapshot.values,
            run_id=trace.run_id,
        ),
        no_progress_decision=progress_decision,
        consecutive_no_progress_rounds=stalled_rounds,
    )


class AgentRunner:
    """Execute the compiled graph and translate its updates into domain events."""

    def __init__(
        self,
        models: WorkflowModels,
        *,
        checkpointer: BaseCheckpointSaver,
        raw_run_store: RawRunStore | None = None,
        pricing_table: PricingTable | None = None,
        memory_store: MemoryStore | None = None,
        episode_clock_factory: EpisodeClockFactory = current_time_ms,
        run_id_factory: RunIdFactory = generate_run_id,
        thread_id_factory: ThreadIdFactory = generate_thread_id,
    ) -> None:
        """Create a reusable Runner with explicit dependencies.

        Args:
            models: Explicit model roles used to compile the workflow.
            checkpointer: Persistence backend shared by start and resume calls.
            raw_run_store: Optional durable store for attempts, active context,
                and exact model I/O provenance.
            pricing_table: Optional immutable snapshot used for reports and
                required when an attempt cost budget is configured.
            memory_store: Optional local memory boundary that receives only
                deterministic terminal episode consolidation.
            episode_clock_factory: Local millisecond clock used only when a
                memory store is configured.
            run_id_factory: Zero-argument factory for run IDs. Tests can inject
                a deterministic factory; production uses UUID strings.
            thread_id_factory: Zero-argument factory for logical investigation
                IDs. A thread ID remains stable across resume attempts.

        Returns:
            None.

        Side effects:
            Stores dependencies only. It must not call the model, execute the
            graph, create traces, or access the repository.
        """

        self._models = models
        self._checkpointer = checkpointer
        if pricing_table is not None and not isinstance(pricing_table, PricingTable):
            raise TypeError("pricing_table must be a PricingTable or None")
        self._pricing_table = pricing_table
        if memory_store is not None:
            from repocompass.memory.store import MemoryStore

            if not isinstance(memory_store, MemoryStore):
                raise TypeError("memory_store must be a MemoryStore or None")
        if not callable(episode_clock_factory):
            raise TypeError("episode_clock_factory must be callable")
        self._memory_store = memory_store
        self._episode_clock_factory = episode_clock_factory
        self._raw_run_recorder = (
            None
            if raw_run_store is None
            else RawRunRecorder(raw_run_store, pricing_table=pricing_table)
        )
        self._run_id_factory = run_id_factory
        self._thread_id_factory = thread_id_factory

    def run(self, request: RunRequest) -> RunOutcome:
        """Start one new repository investigation.

        Args:
            request: Validated repository, objective, and budget input.

        Returns:
            The completed result, stable checkpoint thread ID, and trace for
            this start attempt.

        Raises:
            ValueError: If the repository, generated identity, or final graph
                state is invalid.
            OSError: If repository resolution or graph execution encounters an
                operating-system failure.

        Side effects:
            Invokes the configured model and read-only repository tools through
            the graph and persists graph checkpoints through the injected
            checkpointer. It does not write the completed domain trace to disk.
        """

        if request.max_attempt_cost_usd is not None and (
            self._raw_run_recorder is None or self._pricing_table is None
        ):
            raise ValueError(
                "attempt cost budget requires a raw store and pricing table"
            )
        if request.memory_retrieval_enabled and self._memory_store is None:
            raise ValueError("memory retrieval requires a memory store")
        repo_path = resolve_repository_path(request.repo_path)
        thread_id = _normalize_identifier(
            self._thread_id_factory(),
            name="thread_id",
        )
        initial_state = create_initial_state(
            repo_path,
            request.objective,
            request.max_tool_calls_per_step,
            request.max_explorer_calls_per_step,
            request.max_plan_revisions,
            request.human_review_policy,
            max_no_progress_rounds=request.max_no_progress_rounds,
            context_budget_tokens=request.context_budget_tokens,
            max_attempt_cost_usd=request.max_attempt_cost_usd,
            cost_budget_pricing_table_id=(
                self._pricing_table.table_id
                if request.max_attempt_cost_usd is not None
                and self._pricing_table is not None
                else None
            ),
            memory_retrieval_enabled=request.memory_retrieval_enabled,
            max_repository_memory_results=request.max_repository_memory_results,
            max_episodic_memory_results=request.max_episodic_memory_results,
        )
        graph = create_agent_graph(
            self._models,
            checkpointer=self._checkpointer,
            raw_run_recorder=self._raw_run_recorder,
            memory_store=self._memory_store,
            memory_clock=self._episode_clock_factory,
        )
        config = _thread_config(thread_id)
        if graph.get_state(config).values:
            raise ValueError(f"Generated thread ID already exists: {thread_id}")

        return self._execute_graph(
            graph,
            initial_state,
            thread_id=thread_id,
            execution_mode="start",
        )

    def resume(self, request: ResumeRequest) -> RunOutcome:
        """Resume the latest checkpoint for one persisted investigation."""

        thread_id = _normalize_identifier(
            request.thread_id,
            name="thread_id",
        )

        graph = create_agent_graph(
            self._models,
            checkpointer=self._checkpointer,
            raw_run_recorder=self._raw_run_recorder,
            memory_store=self._memory_store,
            memory_clock=self._episode_clock_factory,
        )

        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise ValueError(f"Checkpoint thread does not exist: {thread_id}")

        return self._execute_graph(
            graph,
            None,
            thread_id=thread_id,
            execution_mode="resume",
        )

    def respond(self, request: HumanResponseRequest) -> RunOutcome:
        """Consume one exact pending Human-review interrupt."""

        thread_id = _normalize_identifier(request.thread_id, name="thread_id")
        graph = create_agent_graph(
            self._models,
            checkpointer=self._checkpointer,
            raw_run_recorder=self._raw_run_recorder,
            memory_store=self._memory_store,
            memory_clock=self._episode_clock_factory,
        )
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise ValueError(f"Checkpoint thread does not exist: {thread_id}")
        if not snapshot.interrupts:
            raise ValueError(
                f"Checkpoint thread is not awaiting a Human response: {thread_id}"
            )
        if len(snapshot.interrupts) != 1:
            raise ValueError(
                "Checkpoint thread has an unsupported number of interrupts: "
                f"{len(snapshot.interrupts)}"
            )

        pending_interrupt = snapshot.interrupts[0]
        pending_interrupt_id = _normalize_identifier(
            pending_interrupt.id,
            name="interrupt_id",
        )
        response_interrupt_id = _normalize_identifier(
            request.interrupt_id,
            name="interrupt_id",
        )
        if response_interrupt_id != pending_interrupt_id:
            raise ValueError(
                "Human response interrupt ID does not match the pending interrupt"
            )

        review_request = decode_human_review_interrupt(pending_interrupt.value)
        if request.decision.action not in review_request.allowed_actions:
            raise ValueError(
                "Human decision action is not allowed by the pending review: "
                f"{request.decision.action.value}"
            )

        return self._execute_graph(
            graph,
            Command(resume=encode_human_decision(request.decision)),
            thread_id=thread_id,
            execution_mode="respond",
            human_decision=request.decision,
        )

    def continue_thread(self, request: ContinueRequest) -> RunOutcome:
        """Start a new Human turn on one completed checkpoint thread."""

        thread_id = _normalize_identifier(request.thread_id, name="thread_id")
        graph = create_agent_graph(
            self._models,
            checkpointer=self._checkpointer,
            raw_run_recorder=self._raw_run_recorder,
            memory_store=self._memory_store,
            memory_clock=self._episode_clock_factory,
        )
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise ValueError(f"Checkpoint thread does not exist: {thread_id}")
        if snapshot.next or snapshot.interrupts:
            raise ValueError(f"Checkpoint thread is not completed: {thread_id}")

        continuation_update = create_continuation_update(
            snapshot.values,
            request.prompt,
        )
        graph.update_state(
            config,
            continuation_update,
            as_node="__start__",
        )
        return self._execute_graph(
            graph,
            None,
            thread_id=thread_id,
            execution_mode="continue",
        )

    def _execute_graph(
        self,
        graph: CompiledStateGraph,
        graph_input: RepoCompassState | Command | None,
        *,
        thread_id: str,
        execution_mode: ExecutionMode,
        human_decision: HumanDecision | None = None,
    ) -> RunOutcome:
        """Stream one lifecycle attempt and publish its observable outcome."""

        checkpoint_config = _thread_config(thread_id)
        metadata_state: Mapping[str, object]
        existing_snapshot: StateSnapshot | None = None
        if isinstance(graph_input, Mapping):
            metadata_state = graph_input
        else:
            existing_snapshot = graph.get_state(checkpoint_config)
            metadata_state = existing_snapshot.values

        configured_cost_budget = metadata_state.get("max_attempt_cost_usd")
        configured_pricing_table_id = metadata_state.get("cost_budget_pricing_table_id")
        if (
            metadata_state.get("memory_retrieval_enabled", False)
            and self._memory_store is None
        ):
            raise ValueError("memory retrieval requires a memory store")
        if configured_cost_budget is not None:
            configured_cost_budget = validate_cost_budget_usd(configured_cost_budget)
            if self._raw_run_recorder is None or self._pricing_table is None:
                raise ValueError(
                    "attempt cost budget requires a raw store and pricing table"
                )
            if configured_pricing_table_id != self._pricing_table.table_id:
                raise ValueError(
                    "attempt cost budget pricing snapshot does not match the "
                    "configured pricing table"
                )
        elif configured_pricing_table_id is not None:
            raise ValueError(
                "cost budget pricing snapshot exists without an attempt cost budget"
            )
        trace = RunTrace(
            _normalize_identifier(
                self._run_id_factory(),
                name="run_id",
            )
        )
        turn_number = metadata_state["turn_number"]
        if not isinstance(turn_number, int) or isinstance(turn_number, bool):
            raise TypeError("Graph turn_number is not an integer")
        if turn_number < 1:
            raise ValueError("Graph turn_number must be positive")
        config = _thread_config(
            thread_id,
            run_id=trace.run_id,
            turn_number=turn_number,
        )
        started_payload: dict[str, JsonValue] = {
            "thread_id": thread_id,
            "execution_mode": execution_mode,
            "turn_number": turn_number,
            "repo_path": _to_json_value(metadata_state["repo_path"]),
            "objective": _to_json_value(metadata_state["objective"]),
            "max_tool_calls_per_step": _to_json_value(
                metadata_state["max_tool_calls_per_step"]
            ),
            "max_explorer_calls_per_step": _to_json_value(
                metadata_state["max_explorer_calls_per_step"]
            ),
            "max_no_progress_rounds": _to_json_value(
                metadata_state.get(
                    "max_no_progress_rounds", DEFAULT_MAX_NO_PROGRESS_ROUNDS
                )
            ),
            "max_plan_revisions": _to_json_value(metadata_state["max_plan_revisions"]),
            "context_budget_tokens": _to_json_value(
                metadata_state.get(
                    "context_budget_tokens", DEFAULT_CONTEXT_BUDGET_TOKENS
                )
            ),
        }
        if metadata_state.get("memory_retrieval_enabled", False):
            started_payload["memory_retrieval_enabled"] = True
            started_payload["max_repository_memory_results"] = _to_json_value(
                metadata_state.get(
                    "max_repository_memory_results",
                    DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
                )
            )
            started_payload["max_episodic_memory_results"] = _to_json_value(
                metadata_state.get(
                    "max_episodic_memory_results",
                    DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
                )
            )
        if self._pricing_table is not None:
            started_payload["pricing_table_id"] = self._pricing_table.table_id
        if configured_cost_budget is not None:
            started_payload["max_attempt_cost_usd"] = configured_cost_budget
            assert isinstance(configured_pricing_table_id, str)
            assert configured_pricing_table_id == self._pricing_table.table_id
        if human_decision is not None:
            started_payload["human_decision"] = {
                "action": human_decision.action.value,
                "feedback_provided": human_decision.feedback is not None,
            }
        trace.record(TraceEventKind.RUN_STARTED, payload=started_payload)

        recorder = self._raw_run_recorder
        if recorder is not None:
            recorder.start_attempt(
                run_id=trace.run_id,
                thread_id=thread_id,
                execution_mode=AttemptExecutionMode(execution_mode),
                state=metadata_state,
            )
        try:
            outcome = self._stream_graph_attempt(
                graph,
                graph_input,
                thread_id=thread_id,
                execution_mode=execution_mode,
                existing_snapshot=existing_snapshot,
                trace=trace,
                config=config,
            )
            episode_consolidation = None
            if self._memory_store is not None:
                from repocompass.memory.consolidation import (
                    consolidate_terminal_episode,
                )

                repo_path = metadata_state.get("repo_path")
                objective = metadata_state.get("objective")
                if not isinstance(repo_path, Path):
                    raise TypeError("Graph repo_path is not a Path")
                if not isinstance(objective, str):
                    raise TypeError("Graph objective is not a string")
                episode_consolidation = consolidate_terminal_episode(
                    self._memory_store,
                    context=ToolExecutionContext(repo_path),
                    outcome=outcome,
                    objective=objective,
                    turn_number=turn_number,
                    created_at_ms=self._episode_clock_factory(),
                )
        except BaseException:
            if recorder is not None:
                recorder.finish_attempt(trace.run_id, status=AttemptStatus.FAILED)
            raise

        usage_cost_report = None
        if recorder is not None:
            status = (
                AttemptStatus.INTERRUPTED
                if outcome.execution_status is ExecutionStatus.INTERRUPTED
                else AttemptStatus.COMPLETED
            )
            recorder.finish_attempt(trace.run_id, status=status)
            usage_cost_report = recorder.get_usage_cost_report(
                state=metadata_state,
                config=config,
            )
        _record_terminal_trace_event(
            outcome,
            execution_mode=execution_mode,
            usage_cost_report=usage_cost_report,
            episode_consolidation=episode_consolidation,
        )
        return replace(
            outcome,
            usage_cost_report=usage_cost_report,
            episode_consolidation=episode_consolidation,
        )

    @staticmethod
    def _stream_graph_attempt(
        graph: CompiledStateGraph,
        graph_input: RepoCompassState | Command | None,
        *,
        thread_id: str,
        execution_mode: ExecutionMode,
        existing_snapshot: StateSnapshot | None,
        trace: RunTrace,
        config: RunnableConfig,
    ) -> RunOutcome:
        """Execute a started attempt without owning its persistence lifecycle."""

        if (
            graph_input is None
            and existing_snapshot is not None
            and existing_snapshot.interrupts
        ):
            return _build_interrupted_outcome(
                existing_snapshot,
                thread_id=thread_id,
                trace=trace,
                execution_mode=execution_mode,
            )

        for part in graph.stream(
            graph_input,
            config=config,
            stream_mode="updates",
            durability="sync",
            version="v2",
        ):
            if part["type"] != "updates":
                continue

            for node_name, update in part["data"].items():
                if node_name.startswith("__"):
                    continue
                if not isinstance(update, Mapping):
                    raise TypeError(f"Node {node_name!r} returned a non-mapping update")

                trace.record(
                    TraceEventKind.NODE_COMPLETED,
                    node=node_name,
                    payload=_normalize_update(update),
                )

        snapshot = graph.get_state(config)
        final_state = snapshot.values
        step_stop_reason = _read_step_stop_reason(final_state)

        if snapshot.interrupts:
            return _build_interrupted_outcome(
                snapshot,
                thread_id=thread_id,
                trace=trace,
                execution_mode=execution_mode,
            )

        final_result = final_state.get("final_result")
        if not isinstance(final_result, FinalResult):
            raise ValueError("Graph finished without a FinalResult")
        final_result = revalidate_final_result(final_result)

        termination_status = final_state.get("termination_status")
        if termination_status not in ("accepted", "exhausted", "cancelled"):
            raise ValueError("Graph finished without a termination status")
        progress_decision, stalled_rounds = _read_progress_metadata(final_state)

        return RunOutcome(
            thread_id=thread_id,
            final_result=final_result,
            termination_status=termination_status,
            step_stop_reason=step_stop_reason,
            trace=trace,
            tool_audit_records=_read_attempt_tool_audits(
                final_state,
                run_id=trace.run_id,
            ),
            no_progress_decision=progress_decision,
            consecutive_no_progress_rounds=stalled_rounds,
        )
