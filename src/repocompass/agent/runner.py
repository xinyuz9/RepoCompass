"""Orchestration boundary for one RepoCompass investigation run.

The Runner will connect trusted input, RepoCompassState construction, LangGraph
execution, and the local domain trace. It must reuse graph behavior rather than
duplicating node routing, tool execution, or budget enforcement.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from repocompass.agent.graph import create_agent_graph
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
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.state import (
    RepoCompassState,
    create_continuation_update,
    create_initial_state,
)
from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.models import WorkflowModels

RunIdFactory = Callable[[], str]
ThreadIdFactory = Callable[[], str]
ExecutionMode = Literal["start", "resume", "continue", "respond"]
DEFAULT_MAX_TOOL_CALLS_PER_STEP = 16
DEFAULT_MAX_EXPLORER_CALLS_PER_STEP = 12
DEFAULT_MAX_PLAN_REVISIONS = 2

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


def _normalize_identifier(value: object, *, name: str) -> str:
    """Validate and normalize one generated run or thread identifier."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")

    return normalized


def _thread_config(thread_id: str) -> RunnableConfig:
    """Build the LangGraph configuration for one checkpoint thread."""

    return {"configurable": {"thread_id": thread_id}}


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


def _normalize_update(update: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return a JSON-compatible copy of one node's partial state update."""

    normalized: dict[str, JsonValue] = {}
    for key, value in update.items():
        if key == "human_feedback":
            normalized["human_feedback_provided"] = value is not None
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
        max_plan_revisions: Positive global budget for model-based plan
            revisions.
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
    max_plan_revisions: int = Field(default=DEFAULT_MAX_PLAN_REVISIONS, gt=0)
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.NEVER


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

    def __post_init__(self) -> None:
        """Reject ambiguous combinations of terminal and interrupt fields."""

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
            return

        raise ValueError(f"Unknown execution status: {self.execution_status!r}")


def _read_step_stop_reason(state: Mapping[str, object]) -> str | None:
    """Read and validate the optional active-step stop reason."""

    step_stop_reason = state.get("step_stop_reason")
    if step_stop_reason is not None and not isinstance(step_stop_reason, str):
        raise TypeError("Graph step_stop_reason is not a string or None")

    return step_stop_reason


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
    trace.record(
        TraceEventKind.RUN_INTERRUPTED,
        payload={
            "thread_id": thread_id,
            "execution_mode": execution_mode,
            "interrupt_id": interrupt_id,
            "turn_number": review_request.turn_number,
        },
    )

    return RunOutcome(
        thread_id=thread_id,
        final_result=None,
        termination_status=None,
        step_stop_reason=_read_step_stop_reason(snapshot.values),
        trace=trace,
        execution_status=ExecutionStatus.INTERRUPTED,
        human_review_request=review_request,
        interrupt_id=interrupt_id,
    )


class AgentRunner:
    """Execute the compiled graph and translate its updates into domain events."""

    def __init__(
        self,
        models: WorkflowModels,
        *,
        checkpointer: BaseCheckpointSaver,
        run_id_factory: RunIdFactory = generate_run_id,
        thread_id_factory: ThreadIdFactory = generate_thread_id,
    ) -> None:
        """Create a reusable Runner with explicit dependencies.

        Args:
            models: Explicit model roles used to compile the workflow.
            checkpointer: Persistence backend shared by start and resume calls.
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
        )
        graph = create_agent_graph(
            self._models,
            checkpointer=self._checkpointer,
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

        config = _thread_config(thread_id)
        metadata_state: Mapping[str, object]
        existing_snapshot: StateSnapshot | None = None
        if isinstance(graph_input, Mapping):
            metadata_state = graph_input
        else:
            existing_snapshot = graph.get_state(config)
            metadata_state = existing_snapshot.values

        trace = RunTrace(
            _normalize_identifier(
                self._run_id_factory(),
                name="run_id",
            )
        )
        started_payload: dict[str, JsonValue] = {
            "thread_id": thread_id,
            "execution_mode": execution_mode,
            "turn_number": _to_json_value(metadata_state["turn_number"]),
            "repo_path": _to_json_value(metadata_state["repo_path"]),
            "objective": _to_json_value(metadata_state["objective"]),
            "max_tool_calls_per_step": _to_json_value(
                metadata_state["max_tool_calls_per_step"]
            ),
            "max_explorer_calls_per_step": _to_json_value(
                metadata_state["max_explorer_calls_per_step"]
            ),
            "max_plan_revisions": _to_json_value(metadata_state["max_plan_revisions"]),
        }
        if human_decision is not None:
            started_payload["human_decision"] = {
                "action": human_decision.action.value,
                "feedback_provided": human_decision.feedback is not None,
            }
        trace.record(TraceEventKind.RUN_STARTED, payload=started_payload)

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

        termination_status = final_state.get("termination_status")
        if termination_status not in ("accepted", "exhausted", "cancelled"):
            raise ValueError("Graph finished without a termination status")

        trace.record(
            TraceEventKind.RUN_FINISHED,
            payload={
                "thread_id": thread_id,
                "execution_mode": execution_mode,
                "termination_status": termination_status,
                "step_stop_reason": step_stop_reason,
            },
        )

        return RunOutcome(
            thread_id=thread_id,
            final_result=final_result,
            termination_status=termination_status,
            step_stop_reason=step_stop_reason,
            trace=trace,
        )
