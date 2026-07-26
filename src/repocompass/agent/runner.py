"""Orchestration boundary for one RepoCompass investigation run.

The Runner will connect trusted input, AgentState construction, LangGraph
execution, and the local domain trace. It must reuse graph behavior rather than
duplicating node routing, tool execution, or budget enforcement.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from repocompass.agent.graph import create_agent_graph
from repocompass.agent.result import FinalResult
from repocompass.agent.state import create_initial_state
from repocompass.agent.trace import RunTrace, TraceEventKind

RunIdFactory = Callable[[], str]


def generate_run_id() -> str:
    """Return a new opaque identifier for one investigation run.

    Returns:
        A UUID string suitable for correlating every event in one
        :class:`RunTrace`.

    Side effects:
        Generates a random UUID locally. It performs no file or network I/O.
    """

    return str(uuid4())


def resolve_repository_path(repo_path: Path) -> Path:
    """Resolve one repository root and require an existing directory."""

    resolved_path = repo_path.expanduser().resolve(strict=True)
    if not resolved_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {resolved_path}")

    return resolved_path


def _to_json_value(value: object) -> JsonValue:
    """Convert supported graph-update values to JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseModel):
        return cast(JsonValue, value.model_dump(mode="json"))

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    raise TypeError(
        f"Graph update contains non-JSON-compatible value: {type(value).__name__}"
    )


def _normalize_update(update: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return a JSON-compatible copy of one node's partial state update."""

    return {
        key: _to_json_value(value)
        for key, value in update.items()
    }


class RunRequest(BaseModel):
    """Validated, trusted input required to start one investigation.

    Attributes:
        repo_path: Application-supplied repository root. The Runner must resolve
            it and reject paths that are not existing directories.
        objective: Non-empty repository investigation question.
        max_tool_calls: Positive tool-call budget enforced by the graph.
        max_iterations: Positive Agent model-invocation budget enforced by the
            graph.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    repo_path: Path
    objective: str = Field(min_length=1)
    max_tool_calls: int = Field(gt=0)
    max_iterations: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Completed public result of one Runner invocation.

    Attributes:
        final_result: Structured repository investigation report.
        stop_reason: Graph stop reason, or ``None`` after normal completion.
        trace: Ordered local domain trace for this run.

    ``RunOutcome`` deliberately omits the complete AgentState so persistent
    message history and internal orchestration data do not become public API.
    """

    final_result: FinalResult
    stop_reason: str | None
    trace: RunTrace


class AgentRunner:
    """Execute the compiled graph and translate its updates into domain events."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        run_id_factory: RunIdFactory = generate_run_id,
    ) -> None:
        """Create a reusable Runner with explicit dependencies.

        Args:
            model: Chat model used to compile the RepoCompass graph.
            run_id_factory: Zero-argument factory for run IDs. Tests can inject
                a deterministic factory; production uses UUID strings.

        Returns:
            None.

        Side effects:
            Stores dependencies only. It must not call the model, execute the
            graph, create traces, or access the repository.
        """

        self._model = model
        self._run_id_factory = run_id_factory

    def run(self, request: RunRequest) -> RunOutcome:
        """Run one repository investigation and return its result and trace.

        Each LangGraph node update becomes one ``NODE_COMPLETED`` event whose
        payload is a JSON-compatible copy of that partial state update.

        Args:
            request: Already validated investigation input.

        Returns:
            A :class:`RunOutcome` containing the structured result, final stop
            reason, and complete in-memory domain trace.

        Raises:
            ValueError: If the repository, run ID, or final graph state is
                invalid.
            OSError: If repository resolution or graph execution encounters an
                operating-system failure.

        Side effects:
            Invokes the configured model and read-only repository tools through
            the existing graph. It mutates only the newly created in-memory
            trace and performs no trace-file persistence.
        """

        repo_path = resolve_repository_path(request.repo_path)

        trace = RunTrace(self._run_id_factory())
        trace.record(
            TraceEventKind.RUN_STARTED,
            payload={
                "repo_path": str(repo_path),
                "objective": request.objective,
                "max_tool_calls": request.max_tool_calls,
                "max_iterations": request.max_iterations,
            },
        )

        initial_state = create_initial_state(
            repo_path,
            [HumanMessage(content=request.objective)],
            request.max_tool_calls,
            request.max_iterations,
        )
        graph = create_agent_graph(self._model)
        final_result: FinalResult | None = None
        stop_reason: str | None = None

        for part in graph.stream(
            initial_state,
            stream_mode="updates",
            version="v2",
        ):
            if part["type"] != "updates":
                continue

            for node_name, update in part["data"].items():
                if node_name.startswith("__"):
                    continue
                if not isinstance(update, Mapping):
                    raise TypeError(
                        f"Node {node_name!r} returned a non-mapping update"
                    )

                trace.record(
                    TraceEventKind.NODE_COMPLETED,
                    node=node_name,
                    payload=_normalize_update(update),
                )

                if "stop_reason" in update:
                    candidate_stop_reason = update["stop_reason"]
                    if candidate_stop_reason is not None and not isinstance(
                        candidate_stop_reason, str
                    ):
                        raise TypeError("Graph stop_reason is not a string or None")
                    stop_reason = candidate_stop_reason

                if "final_result" in update:
                    candidate_result = update["final_result"]
                    if not isinstance(candidate_result, FinalResult):
                        raise TypeError("Graph final_result is not a FinalResult")
                    final_result = candidate_result

        if final_result is None:
            raise ValueError("Graph finished without a FinalResult")

        trace.record(
            TraceEventKind.RUN_FINISHED,
            payload={"stop_reason": stop_reason},
        )

        return RunOutcome(
            final_result=final_result,
            stop_reason=stop_reason,
            trace=trace,
        )
