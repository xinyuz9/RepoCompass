"""State-mutating nodes and node factories for the RepoCompass graph."""

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError
from repocompass.agent.admission import PRECALL_BUDGET_STOP_REASONS, PrecallBudgetDenied

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.context import ContextItem, ContextPack, MemoryRecallReport
from repocompass.agent.context_builder import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    ContextSelectionStrategy,
)
from repocompass.agent.context_gathering import (
    GatheredContext,
    gather_candidate_context,
    gather_evaluator_context,
    gather_explorer_context,
    gather_planner_context,
    gather_replanner_context,
    render_model_context,
)
from repocompass.agent.cost import AttemptCostBudgetSummary
from repocompass.agent.evaluation import (
    EvaluationReport,
    EvidenceExcerpt,
    SemanticAssessment,
    ValidationIssue,
    validate_candidate,
)
from repocompass.agent.invocation import (
    CostBudgetExceededError,
    RawRunRecorder,
    describe_model,
)
from repocompass.agent.lifecycle import (
    HumanReviewInterruptPayload,
    HumanReviewPolicy,
    HumanReviewRequest,
    allowed_human_decision_actions,
    decode_human_decision,
    decode_human_review_interrupt,
    encode_human_review_interrupt,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.prompts import (
    CANDIDATE_BUILDING_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    REPLANNING_SYSTEM_PROMPT,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import (
    ATTEMPT_COST_BUDGET_STOP_REASON,
    ControllerAction,
    HumanDecisionActionValue,
    TerminationStatus,
)
from repocompass.agent.structural_artifacts import (
    InvestigationStructure,
    build_investigation_structure,
    revalidate_investigation_structure,
)
from repocompass.agent.usage import ModelRole
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    TOOL_NAME_PATTERN,
    ToolAuditPayload,
    ToolCallRequest,
    ToolDecisionReason,
    decode_tool_audit_record,
    encode_tool_audit_record,
)
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    create_repository_tool_registry,
)

if TYPE_CHECKING:
    from repocompass.memory.store import MemoryStore


def _context_budget(state: Mapping[str, object]) -> int:
    """Read a validated context budget while supporting legacy checkpoints."""

    value = state.get("context_budget_tokens", DEFAULT_CONTEXT_BUDGET_TOKENS)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("context_budget_tokens must be a positive integer")
    return value


def _next_context_build_count(state: Mapping[str, object]) -> int:
    """Return the next monotonic context-build sequence number."""

    value = state.get("context_build_count", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("context_build_count must be a non-negative integer")
    return value + 1


@dataclass(frozen=True, slots=True)
class _ModelCostBudgetDenied:
    budget: AttemptCostBudgetSummary | None
    stop_reason: str = ATTEMPT_COST_BUDGET_STOP_REASON


class CostBudgetDeniedUpdate(TypedDict):
    cost_budget: AttemptCostBudgetSummary | None
    step_stop_reason: str


def _cost_budget_denied_update(
    budget: AttemptCostBudgetSummary | None,
    *,
    stop_reason: str = ATTEMPT_COST_BUDGET_STOP_REASON,
) -> CostBudgetDeniedUpdate:
    return {
        "cost_budget": budget,
        "step_stop_reason": stop_reason,
    }


def _invoke_model(
    *,
    recorder: RawRunRecorder | None,
    state: Mapping[str, object],
    config: RunnableConfig | None,
    node_name: str,
    role: ModelRole,
    model: BaseChatModel,
    gathered_context: GatheredContext,
    messages: list[AnyMessage],
    plan_step_id: str | None,
    invoke: Callable[[list[AnyMessage]], object],
    parse: Callable[[object], object],
) -> object | _ModelCostBudgetDenied:
    """Invoke directly or through the fail-closed Raw Run recording boundary."""

    if recorder is None:
        if state.get("max_attempt_cost_usd") is not None:
            raise ValueError("attempt cost budget requires Raw Run recording")
        return parse(invoke(messages))
    if config is None:
        raise ValueError(
            "Runnable config is required when Raw Run recording is enabled"
        )
    try:
        return recorder.invoke_model(
            state=state,
            config=config,
            node_name=node_name,
            role=role,
            identity=describe_model(model),
            gathered_context=gathered_context,
            messages=messages,
            plan_step_id=plan_step_id,
            invoke=invoke,
            parse=parse,
        )
    except CostBudgetExceededError as error:
        return _ModelCostBudgetDenied(error.budget)
    except PrecallBudgetDenied as error:
        return _ModelCostBudgetDenied(
            recorder.assess_cost_budget(state=state, config=config),
            stop_reason=error.stop_reason,
        )


def _parse_structured_response(
    response: object,
    *,
    expected_type: type[BaseModel],
    error_message: str,
) -> BaseModel:
    """Extract a validated parsed value while retaining include_raw envelopes."""

    parsed = response
    if isinstance(response, Mapping):
        parsed = response.get("parsed")
        if (
            not isinstance(parsed, expected_type)
            and response.get("parsing_error") is not None
        ):
            parsed = _parse_strict_json_tool_call(
                response.get("raw"),
                expected_type=expected_type,
            )
    if not isinstance(parsed, expected_type):
        raise TypeError(error_message)
    return parsed


def _parse_strict_json_tool_call(
    raw_response: object,
    *,
    expected_type: type[BaseModel],
) -> BaseModel | None:
    """Revalidate one exact function call at the strict JSON transport boundary."""

    if not isinstance(raw_response, AIMessage) or raw_response.invalid_tool_calls:
        return None
    if len(raw_response.tool_calls) != 1:
        return None
    tool_call = raw_response.tool_calls[0]
    schema_title = expected_type.model_json_schema().get("title")
    if not isinstance(schema_title, str) or tool_call.get("name") != schema_title:
        return None
    arguments = tool_call.get("args")
    if not isinstance(arguments, Mapping) or not all(
        isinstance(key, str) for key in arguments
    ):
        return None
    try:
        payload = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return expected_type.model_validate_json(payload)
    except (TypeError, ValueError, ValidationError):
        return None


def _parse_explorer_response(response: object) -> AIMessage:
    if not isinstance(response, AIMessage):
        raise TypeError("Explorer model did not return AIMessage")
    return response


class PlannerInput(TypedDict):
    """Minimal initial-planning view without repository content or messages."""

    objective: str
    max_tool_calls_per_step: int
    max_explorer_calls_per_step: int
    context_budget_tokens: int
    context_build_count: int
    max_attempt_cost_usd: str | None


class PlannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    active_context_pack: ContextPack
    context_build_count: int


PlannerNode = Callable[[PlannerInput], PlannerNodeUpdate]


def build_planner_messages(
    state: PlannerInput,
    *,
    gathered_context: GatheredContext | None = None,
) -> list[AnyMessage]:
    """Build an isolated initial-planning invocation."""

    gathered = gathered_context or gather_planner_context(
        objective=state["objective"],
        max_tool_calls_per_step=state["max_tool_calls_per_step"],
        max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
        budget_tokens=_context_budget(state),
    )
    return [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=render_model_context(ModelRole.PLANNER, gathered.pack)),
    ]


def create_plan_investigation_node(
    model: BaseChatModel,
    recorder: RawRunRecorder | None = None,
    context_selection_strategy: ContextSelectionStrategy = (
        ContextSelectionStrategy.PRIORITIZED
    ),
) -> PlannerNode:
    """Create the node that generates the initial investigation plan."""

    structured_model = model.with_structured_output(
        InvestigationPlan,
        method="function_calling",
        include_raw=True,
    )

    def planner_node(
        state: PlannerInput,
        config: RunnableConfig | None = None,
    ) -> PlannerNodeUpdate:
        """Generate the initial plan and activate its designated step."""

        gathered = gather_planner_context(
            objective=state["objective"],
            max_tool_calls_per_step=state["max_tool_calls_per_step"],
            max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
            budget_tokens=_context_budget(state),
            selection_strategy=context_selection_strategy,
        )
        messages = build_planner_messages(state, gathered_context=gathered)
        response = _invoke_model(
            recorder=recorder,
            state=state,
            config=config,
            node_name="plan_investigation",
            role=ModelRole.PLANNER,
            model=model,
            gathered_context=gathered,
            messages=messages,
            plan_step_id=None,
            invoke=structured_model.invoke,
            parse=lambda value: _parse_structured_response(
                value,
                expected_type=InvestigationPlan,
                error_message="Planner model did not return InvestigationPlan",
            ),
        )
        if isinstance(response, _ModelCostBudgetDenied):
            return _cost_budget_denied_update(
                response.budget, stop_reason=response.stop_reason
            )  # type: ignore[return-value]
        assert isinstance(response, InvestigationPlan)

        return {
            "investigation_plan": response,
            "active_context_pack": gathered.pack,
            "context_build_count": _next_context_build_count(state),
        }

    return planner_node


def seed_tool_wrapper_plan(state: PlannerInput) -> dict[str, InvestigationPlan]:
    """Create one deterministic active step without a planner model call."""

    objective = state["objective"].strip()
    if not objective:
        raise ValueError("Tool-wrapper baseline requires a non-empty objective")
    return {
        "investigation_plan": InvestigationPlan(
            steps=[
                PlanStep(
                    id="tool-wrapper-step",
                    objective=objective,
                    expected_evidence=[
                        "Repository-relative files and exact source locations"
                    ],
                    status="active",
                )
            ]
        )
    }


class MemoryRecallNodeInput(TypedDict):
    """Minimal state required for one optional new-turn recall."""

    repo_path: Path
    objective: str
    investigation_plan: InvestigationPlan | None
    memory_retrieval_enabled: bool
    max_repository_memory_results: int
    max_episodic_memory_results: int


class MemoryRecallNodeUpdate(TypedDict):
    memory_recall_report: MemoryRecallReport | None
    memory_context_items: list[ContextItem]


MemoryRecallNode = Callable[
    [MemoryRecallNodeInput, RunnableConfig | None],
    MemoryRecallNodeUpdate,
]


def create_recall_memory_node(
    store: "MemoryStore | None",
    clock: Callable[[], int],
) -> MemoryRecallNode:
    """Create one checkpointed recall node; disabled mode performs no query."""

    if not callable(clock):
        raise TypeError("memory recall clock must be callable")

    def recall_node(
        state: MemoryRecallNodeInput,
        config: RunnableConfig | None = None,
    ) -> MemoryRecallNodeUpdate:
        if not state["memory_retrieval_enabled"]:
            return {"memory_recall_report": None, "memory_context_items": []}
        if store is None:
            raise ValueError("memory retrieval is enabled without a memory store")
        plan = state["investigation_plan"]
        if plan is None:
            raise ValueError("memory recall requires an InvestigationPlan")
        from repocompass.memory.retrieval import recall_memory_context

        result = recall_memory_context(
            store,
            repo_path=state["repo_path"],
            objective=state["objective"],
            active_step=plan.active_step,
            max_repository_results=state["max_repository_memory_results"],
            max_episodic_results=state["max_episodic_memory_results"],
            audit_at_ms=clock(),
            run_id=_memory_recall_run_id(config),
        )
        return {
            "memory_recall_report": result.report,
            "memory_context_items": list(result.context_items),
        }

    return recall_node


def _memory_recall_run_id(config: RunnableConfig | None) -> str | None:
    if config is None:
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("repocompass_run_id")
    if isinstance(value, str) and re.fullmatch(IDENTIFIER_PATTERN, value):
        return value
    return None


class AgentNodeInput(MessagesState):
    """State visible to one Explorer model invocation."""

    repo_path: Path
    objective: str
    investigation_plan: InvestigationPlan | None
    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    evaluation_report: EvaluationReport | None
    context_budget_tokens: int
    context_build_count: int
    max_attempt_cost_usd: str | None
    memory_context_items: list[ContextItem]
    last_tool_batch_audits: list[ToolAuditPayload]


class AgentNodeUpdate(TypedDict):
    messages: list[AnyMessage]
    step_explorer_call_count: int
    active_context_pack: ContextPack
    context_build_count: int


AgentNode = Callable[[AgentNodeInput], AgentNodeUpdate]


def build_agent_messages(
    state: AgentNodeInput,
    *,
    gathered_context: GatheredContext | None = None,
) -> list[AnyMessage]:
    """Build one Explorer invocation without mutating persistent messages."""

    investigation_plan = state["investigation_plan"]
    if investigation_plan is None:
        raise ValueError("Cannot investigate without an InvestigationPlan")

    gathered = gathered_context or gather_explorer_context(
        repo_path=state["repo_path"],
        objective=state["objective"],
        active_step=investigation_plan.active_step,
        remaining_tool_calls=max(
            0,
            state["max_tool_calls_per_step"] - state["step_tool_call_count"],
        ),
        remaining_explorer_calls=max(
            0,
            state["max_explorer_calls_per_step"] - state["step_explorer_call_count"],
        ),
        messages=tuple(state["messages"]),
        last_tool_batch_audits=tuple(
            decode_tool_audit_record(item)
            for item in state.get("last_tool_batch_audits", [])
        ),
        evaluation_report=state["evaluation_report"],
        budget_tokens=_context_budget(state),
        memory_items=tuple(state.get("memory_context_items", [])),
    )
    return [
        SystemMessage(content=INVESTIGATION_SYSTEM_PROMPT),
        HumanMessage(content=render_model_context(ModelRole.EXPLORER, gathered.pack)),
    ]


def create_agent_node(
    model: BaseChatModel,
    recorder: RawRunRecorder | None = None,
    repository_tools: tuple[BaseTool, ...] = REPOSITORY_TOOLS,
    context_selection_strategy: ContextSelectionStrategy = (
        ContextSelectionStrategy.PRIORITIZED
    ),
) -> AgentNode:
    """Bind repository tools once and create the graph's Explorer node."""

    if not isinstance(repository_tools, tuple) or not repository_tools:
        raise TypeError("repository_tools must be a non-empty tuple")
    bound_model = model.bind_tools(repository_tools)

    def agent_node(
        state: AgentNodeInput,
        config: RunnableConfig | None = None,
    ) -> AgentNodeUpdate:
        """Call the tool-bound model once and return its state update."""

        investigation_plan = state["investigation_plan"]
        if investigation_plan is None:
            raise ValueError("Cannot investigate without an InvestigationPlan")
        gathered = gather_explorer_context(
            repo_path=state["repo_path"],
            objective=state["objective"],
            active_step=investigation_plan.active_step,
            remaining_tool_calls=max(
                0,
                state["max_tool_calls_per_step"] - state["step_tool_call_count"],
            ),
            remaining_explorer_calls=max(
                0,
                state["max_explorer_calls_per_step"]
                - state["step_explorer_call_count"],
            ),
            messages=tuple(state["messages"]),
            last_tool_batch_audits=tuple(
                decode_tool_audit_record(item)
                for item in state.get("last_tool_batch_audits", [])
            ),
            evaluation_report=state["evaluation_report"],
            budget_tokens=_context_budget(state),
            memory_items=tuple(state.get("memory_context_items", [])),
            selection_strategy=context_selection_strategy,
        )
        messages = build_agent_messages(state, gathered_context=gathered)
        response = _invoke_model(
            recorder=recorder,
            state=state,
            config=config,
            node_name="agent",
            role=ModelRole.EXPLORER,
            model=model,
            gathered_context=gathered,
            messages=messages,
            plan_step_id=investigation_plan.active_step.id,
            invoke=bound_model.invoke,
            parse=_parse_explorer_response,
        )
        if isinstance(response, _ModelCostBudgetDenied):
            return _cost_budget_denied_update(
                response.budget, stop_reason=response.stop_reason
            )  # type: ignore[return-value]
        assert isinstance(response, AIMessage)
        return {
            "messages": [response],
            "step_explorer_call_count": state["step_explorer_call_count"] + 1,
            "active_context_pack": gathered.pack,
            "context_build_count": _next_context_build_count(state),
        }

    return agent_node


class ExplorerBudgetInput(TypedDict):
    """Empty state view for a constant budget-exhaustion update."""


class ExplorerBudgetUpdate(TypedDict):
    step_stop_reason: str


def mark_step_explorer_budget_exhausted(
    _state: ExplorerBudgetInput,
) -> ExplorerBudgetUpdate:
    """Record that the active step cannot invoke Explorer again."""

    return {"step_stop_reason": "step_explorer_budget_exhausted"}


class ToolGatewayNodeInput(MessagesState):
    """Trusted state required to execute one model-requested tool batch."""

    repo_path: Path
    step_tool_call_count: int
    max_tool_calls_per_step: int
    tool_audit_records: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: int | None


class ToolGatewayNodeUpdate(TypedDict):
    messages: list[ToolMessage]
    step_tool_call_count: int
    step_stop_reason: str | None
    tool_audit_records: list[ToolAuditPayload]
    last_tool_batch_audits: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: int


ToolGatewayNode = Callable[
    [ToolGatewayNodeInput, RunnableConfig],
    ToolGatewayNodeUpdate,
]


def create_execute_tools_node(
    gateway: ToolGateway | None = None,
) -> ToolGatewayNode:
    """Create the only Graph node authorized to execute repository tools."""

    configured_gateway = gateway or ToolGateway(create_repository_tool_registry())

    def execute_tools_node(
        state: ToolGatewayNodeInput,
        config: RunnableConfig,
    ) -> ToolGatewayNodeUpdate:
        """Validate, authorize, execute, normalize, and audit one tool batch."""

        if not state["messages"]:
            raise ValueError("Cannot execute tools without a pending message")
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Pending tool message must be an AIMessage")

        run_id = _tool_run_id(config)
        decoded_records = tuple(
            decode_tool_audit_record(payload) for payload in state["tool_audit_records"]
        )
        prior_run_records = tuple(
            record for record in decoded_records if record.run_id == run_id
        )
        configured_floor = state["step_tool_audit_sequence_floor"]
        if configured_floor is None:
            effective_floor = len(prior_run_records)
        elif not isinstance(configured_floor, int) or isinstance(
            configured_floor,
            bool,
        ):
            raise TypeError("step_tool_audit_sequence_floor must be an integer or None")
        elif configured_floor < 0:
            raise ValueError("step_tool_audit_sequence_floor must not be negative")
        else:
            effective_floor = min(configured_floor, len(prior_run_records))
        audit_log = ToolAuditLog(
            run_id,
            records=prior_run_records,
            deduplication_floor=effective_floor,
        )
        prior_run_record_count = len(prior_run_records)
        context = ToolExecutionContext(state["repo_path"])
        contains_invalid_call = bool(last_message.invalid_tool_calls)
        pending_calls = [
            *[(tool_call, True) for tool_call in last_message.invalid_tool_calls],
            *[(tool_call, False) for tool_call in last_message.tool_calls],
        ]
        remaining_budget = max(
            0,
            state["max_tool_calls_per_step"] - state["step_tool_call_count"],
        )
        batch_exceeds_budget = (
            not contains_invalid_call and len(pending_calls) > remaining_budget
        )
        messages: list[ToolMessage] = []
        budget_consumed = 0
        budget_denied = False

        for tool_call, is_invalid_call in pending_calls:
            original_name = tool_call.get("name")
            original_call_id = tool_call.get("id")
            if original_call_id is None:
                raise ValueError("Tool call id is None")
            request, ingress_invalid = _bind_gateway_request(
                context,
                call_id=original_call_id,
                tool_name=original_name,
                arguments=tool_call.get("args"),
                force_invalid=is_invalid_call,
            )

            if contains_invalid_call or ingress_invalid:
                result = configured_gateway.reject(
                    request,
                    context=context,
                    reason=ToolDecisionReason.INVALID_ARGUMENTS,
                    audit_log=audit_log,
                )
            elif batch_exceeds_budget:
                result = configured_gateway.reject(
                    request,
                    context=context,
                    reason=ToolDecisionReason.BUDGET_EXHAUSTED,
                    audit_log=audit_log,
                )
            else:
                result = configured_gateway.execute(
                    request,
                    context=context,
                    available_budget=remaining_budget,
                    audit_log=audit_log,
                )
                remaining_budget -= result.budget_consumed

            budget_consumed += result.budget_consumed
            if result.decision.reason is ToolDecisionReason.BUDGET_EXHAUSTED:
                budget_denied = True
            messages.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=original_call_id,
                    name=original_name or request.tool_name,
                    status="success" if result.succeeded else "error",
                )
            )

        new_audit_records = [
            encode_tool_audit_record(record)
            for record in audit_log.records[prior_run_record_count:]
        ]
        return {
            "messages": messages,
            "step_tool_call_count": (state["step_tool_call_count"] + budget_consumed),
            "step_stop_reason": (
                "step_tool_call_budget_exhausted" if budget_denied else None
            ),
            "tool_audit_records": new_audit_records,
            "last_tool_batch_audits": new_audit_records,
            "step_tool_audit_sequence_floor": effective_floor,
        }

    return execute_tools_node


def _tool_run_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    configured_run_id = configurable.get("repocompass_run_id")
    if isinstance(configured_run_id, str) and re.fullmatch(
        IDENTIFIER_PATTERN,
        configured_run_id,
    ):
        return configured_run_id
    return "graph-invoke"


def _bind_gateway_request(
    context: ToolExecutionContext,
    *,
    call_id: object,
    tool_name: object,
    arguments: object,
    force_invalid: bool,
) -> tuple[ToolCallRequest, bool]:
    if not force_invalid and isinstance(arguments, dict):
        try:
            return (
                context.bind_call(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                ),
                False,
            )
        except (TypeError, ValueError, ValidationError):
            pass

    digest = _invalid_ingress_digest(tool_name, arguments)
    safe_call_id = _safe_gateway_identifier(call_id, prefix="invalid-call")
    safe_tool_name = (
        tool_name
        if isinstance(tool_name, str) and re.fullmatch(TOOL_NAME_PATTERN, tool_name)
        else "invalid_tool"
    )
    return (
        context.bind_call(
            call_id=safe_call_id,
            tool_name=safe_tool_name,
            arguments={"invalid_call_digest": digest},
        ),
        True,
    )


def _invalid_ingress_digest(tool_name: object, arguments: object) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"repocompass:invalid-tool-call:v1\0")
    for value in (tool_name, arguments):
        if isinstance(value, str):
            encoded = value.encode("utf-8", errors="backslashreplace")
        else:
            try:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError):
                encoded = type(value).__name__.encode("ascii", errors="replace")
        hasher.update(len(encoded).to_bytes(8, byteorder="big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _safe_gateway_identifier(value: object, *, prefix: str) -> str:
    if isinstance(value, str) and re.fullmatch(IDENTIFIER_PATTERN, value):
        return value
    digest = hashlib.sha256(
        str(value).encode("utf-8", errors="backslashreplace")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


class CandidateBuilderInput(MessagesState):
    """Evidence-bearing state visible to candidate construction."""

    repo_path: Path
    objective: str
    investigation_plan: InvestigationPlan | None
    step_stop_reason: str | None
    context_budget_tokens: int
    context_build_count: int
    max_attempt_cost_usd: str | None
    evaluation_report: EvaluationReport | None


class CandidateBuilderUpdate(TypedDict):
    candidate_result: CandidateResult
    active_context_pack: ContextPack
    context_build_count: int
    investigation_structure: None


CandidateBuilderNode = Callable[[CandidateBuilderInput], CandidateBuilderUpdate]


def build_candidate_messages(
    state: CandidateBuilderInput,
    *,
    gathered_context: GatheredContext | None = None,
) -> list[AnyMessage]:
    """Build an isolated candidate invocation from selected evidence only."""

    investigation_plan = state["investigation_plan"]
    if investigation_plan is None:
        raise ValueError("Cannot build a candidate without an InvestigationPlan")

    gathered = gathered_context or gather_candidate_context(
        repo_path=state["repo_path"],
        objective=state["objective"],
        active_step=investigation_plan.active_step,
        step_stop_reason=state["step_stop_reason"] or "none",
        messages=tuple(state["messages"]),
        budget_tokens=_context_budget(state),
        evaluation_report=state.get("evaluation_report"),
    )
    return [
        SystemMessage(content=CANDIDATE_BUILDING_SYSTEM_PROMPT),
        HumanMessage(
            content=render_model_context(ModelRole.CANDIDATE_BUILDER, gathered.pack)
        ),
    ]


def create_build_candidate_node(
    model: BaseChatModel,
    recorder: RawRunRecorder | None = None,
    context_selection_strategy: ContextSelectionStrategy = (
        ContextSelectionStrategy.PRIORITIZED
    ),
) -> CandidateBuilderNode:
    """Create a node that turns Explorer evidence into an internal candidate."""

    structured_model = model.with_structured_output(
        CandidateResult,
        method="function_calling",
        include_raw=True,
    )

    def build_candidate_node(
        state: CandidateBuilderInput,
        config: RunnableConfig | None = None,
    ) -> CandidateBuilderUpdate:
        """Generate one structured candidate from the collected evidence."""

        investigation_plan = state["investigation_plan"]
        if investigation_plan is None:
            raise ValueError("Cannot build a candidate without an InvestigationPlan")
        gathered = gather_candidate_context(
            repo_path=state["repo_path"],
            objective=state["objective"],
            active_step=investigation_plan.active_step,
            step_stop_reason=state["step_stop_reason"] or "none",
            messages=tuple(state["messages"]),
            budget_tokens=_context_budget(state),
            evaluation_report=state.get("evaluation_report"),
            selection_strategy=context_selection_strategy,
        )
        messages = build_candidate_messages(state, gathered_context=gathered)
        response = _invoke_model(
            recorder=recorder,
            state=state,
            config=config,
            node_name="build_candidate",
            role=ModelRole.CANDIDATE_BUILDER,
            model=model,
            gathered_context=gathered,
            messages=messages,
            plan_step_id=investigation_plan.active_step.id,
            invoke=structured_model.invoke,
            parse=lambda value: _parse_structured_response(
                value,
                expected_type=CandidateResult,
                error_message="Candidate builder model did not return CandidateResult",
            ),
        )
        if isinstance(response, _ModelCostBudgetDenied):
            return _cost_budget_denied_update(
                response.budget, stop_reason=response.stop_reason
            )  # type: ignore[return-value]
        assert isinstance(response, CandidateResult)

        return {
            "candidate_result": response,
            "active_context_pack": gathered.pack,
            "context_build_count": _next_context_build_count(state),
            "investigation_structure": None,
        }

    return build_candidate_node


class EvaluatorInput(TypedDict):
    """Minimal trusted state visible to candidate evaluation."""

    repo_path: Path
    objective: str
    investigation_plan: InvestigationPlan | None
    candidate_result: CandidateResult | None
    context_budget_tokens: int
    context_build_count: int
    max_attempt_cost_usd: str | None
    investigation_structure: InvestigationStructure | None


class EvaluationPromptInput(TypedDict):
    """Validated data rendered into one isolated evaluator invocation."""

    repo_path: Path
    objective: str
    active_step: PlanStep
    candidate_result: CandidateResult
    deterministic_issues: tuple[ValidationIssue, ...]
    evidence_pack: tuple[EvidenceExcerpt, ...]
    context_budget_tokens: int
    context_build_count: int
    investigation_structure: InvestigationStructure | None


class EvaluatorNodeUpdate(TypedDict):
    evaluation_report: EvaluationReport
    active_context_pack: ContextPack
    context_build_count: int
    investigation_structure: InvestigationStructure | None


EvaluatorNode = Callable[[EvaluatorInput], EvaluatorNodeUpdate]


def build_evaluator_messages(
    state: EvaluationPromptInput,
    *,
    gathered_context: GatheredContext | None = None,
) -> list[AnyMessage]:
    """Build one evaluator invocation without Explorer message history."""

    gathered = gathered_context or gather_evaluator_context(
        repo_path=state["repo_path"],
        objective=state["objective"],
        active_step=state["active_step"],
        candidate_result=state["candidate_result"],
        deterministic_issues=state["deterministic_issues"],
        evidence_pack=state["evidence_pack"],
        budget_tokens=_context_budget(state),
        investigation_structure=state.get("investigation_structure"),
    )
    return [
        SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
        HumanMessage(content=render_model_context(ModelRole.EVALUATOR, gathered.pack)),
    ]


def create_evaluate_candidate_node(
    model: BaseChatModel,
    recorder: RawRunRecorder | None = None,
    context_selection_strategy: ContextSelectionStrategy = (
        ContextSelectionStrategy.PRIORITIZED
    ),
) -> EvaluatorNode:
    """Create the isolated semantic evaluator with deterministic validation."""

    structured_model = model.with_structured_output(
        SemanticAssessment,
        method="function_calling",
        include_raw=True,
    )

    def evaluate_candidate_node(
        state: EvaluatorInput,
        config: RunnableConfig | None = None,
    ) -> EvaluatorNodeUpdate:
        """Validate one candidate and return a merged evaluation report."""

        candidate = state["candidate_result"]
        if candidate is None:
            raise ValueError("Cannot evaluate without a CandidateResult")
        investigation_plan = state["investigation_plan"]
        if investigation_plan is None:
            raise ValueError("Cannot evaluate without an InvestigationPlan")

        validation = validate_candidate(state["repo_path"], candidate)
        structure_result = build_investigation_structure(
            repo_path=state["repo_path"],
            objective=state["objective"],
            candidate=candidate,
            valid_evidence_indices=frozenset(
                excerpt.evidence_index for excerpt in validation.evidence_pack
            ),
        )
        structure_issues = tuple(
            ValidationIssue(
                code=issue.code,
                message=issue.message,
                path=issue.path,
                evidence_index=issue.evidence_index,
                relation_index=issue.relation_index,
            )
            for issue in structure_result.issues
        )
        deterministic_issues = (*validation.issues, *structure_issues)
        prompt_input: EvaluationPromptInput = {
            "repo_path": state["repo_path"],
            "objective": state["objective"],
            "active_step": investigation_plan.active_step,
            "candidate_result": candidate,
            "deterministic_issues": deterministic_issues,
            "evidence_pack": validation.evidence_pack,
            "context_budget_tokens": _context_budget(state),
            "context_build_count": state.get("context_build_count", 0),
            "investigation_structure": structure_result.artifact,
        }
        gathered = gather_evaluator_context(
            repo_path=prompt_input["repo_path"],
            objective=prompt_input["objective"],
            active_step=prompt_input["active_step"],
            candidate_result=prompt_input["candidate_result"],
            deterministic_issues=prompt_input["deterministic_issues"],
            evidence_pack=prompt_input["evidence_pack"],
            budget_tokens=_context_budget(prompt_input),
            investigation_structure=prompt_input["investigation_structure"],
            selection_strategy=context_selection_strategy,
        )
        messages = build_evaluator_messages(prompt_input, gathered_context=gathered)
        response = _invoke_model(
            recorder=recorder,
            state=state,
            config=config,
            node_name="evaluate_candidate",
            role=ModelRole.EVALUATOR,
            model=model,
            gathered_context=gathered,
            messages=messages,
            plan_step_id=investigation_plan.active_step.id,
            invoke=structured_model.invoke,
            parse=lambda value: _parse_structured_response(
                value,
                expected_type=SemanticAssessment,
                error_message="Evaluator model did not return SemanticAssessment",
            ),
        )
        if isinstance(response, _ModelCostBudgetDenied):
            return _cost_budget_denied_update(
                response.budget, stop_reason=response.stop_reason
            )  # type: ignore[return-value]
        assert isinstance(response, SemanticAssessment)

        return {
            "evaluation_report": EvaluationReport(
                deterministic_issues=list(deterministic_issues),
                semantic_assessment=response,
            ),
            "active_context_pack": gathered.pack,
            "context_build_count": _next_context_build_count(state),
            "investigation_structure": structure_result.artifact,
        }

    return evaluate_candidate_node


class ControllerInput(TypedDict):
    """State required for one deterministic post-evaluation decision."""

    investigation_plan: InvestigationPlan | None
    evaluation_report: EvaluationReport | None
    step_stop_reason: str | None
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    plan_revision_count: int
    max_plan_revisions: int


class ControllerNodeUpdate(TypedDict):
    next_action: ControllerAction


def decide_next_action(state: ControllerInput) -> ControllerNodeUpdate:
    """Select the next workflow action after one active-step evaluation.

    A completed active step advances before the overall answer can be accepted,
    so pending plan work cannot be bypassed. This deterministic node must not
    call a model or mutate the investigation plan.
    """
    if state["investigation_plan"] is None:
        raise ValueError("InvestigationPlan is missing")

    if state["evaluation_report"] is None:
        raise ValueError("EvaluationReport is missing")

    report = state["evaluation_report"]
    plan = state["investigation_plan"]
    semantic_assessment = report.semantic_assessment

    has_revision_budget = state["plan_revision_count"] < state["max_plan_revisions"]

    # Model approval cannot override invalid source/structure evidence. There
    # is no reliable per-step issue attribution in the cumulative candidate,
    # so any deterministic defect keeps the active step uncompleted.
    if semantic_assessment.active_step_completed and not report.deterministic_issues:
        if plan.next_pending_step() is not None:
            return {"next_action": "advance_step"}

        if report.passed:
            return {"next_action": "accepted"}

    elif (
        state["step_stop_reason"] is None
        and state["step_explorer_call_count"] < state["max_explorer_calls_per_step"]
    ):
        return {"next_action": "continue_step"}

    return {"next_action": "revise_plan" if has_revision_budget else "exhausted"}


class PlanAdvanceInput(TypedDict):
    """State required to complete the active step and activate the next one."""

    investigation_plan: InvestigationPlan | None


class PlanAdvanceUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    step_tool_call_count: int
    step_explorer_call_count: int
    last_tool_batch_audits: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: None
    progress_snapshot: None
    consecutive_no_progress_rounds: int
    step_stop_reason: None
    candidate_result: None
    evaluation_report: None
    investigation_structure: None
    next_action: None


def advance_plan_step(state: PlanAdvanceInput) -> PlanAdvanceUpdate:
    """Complete the active step, activate its successor, and reset local state."""

    plan = state["investigation_plan"]
    if plan is None:
        raise ValueError("Cannot advance without an InvestigationPlan")

    active_step = plan.active_step
    next_step = plan.next_pending_step()
    if next_step is None:
        raise ValueError("Cannot advance without a pending plan step")

    revised_steps = []
    for step in plan.steps:
        if step.id == active_step.id:
            step = step.model_copy(update={"status": "completed"})
        elif step.id == next_step.id:
            step = step.model_copy(update={"status": "active"})
        revised_steps.append(step)

    revised_plan = InvestigationPlan(steps=revised_steps)

    return {
        "investigation_plan": revised_plan,
        "step_tool_call_count": 0,
        "step_explorer_call_count": 0,
        "last_tool_batch_audits": [],
        "step_tool_audit_sequence_floor": None,
        "progress_snapshot": None,
        "consecutive_no_progress_rounds": 0,
        "step_stop_reason": None,
        "candidate_result": None,
        "evaluation_report": None,
        "investigation_structure": None,
        "next_action": None,
    }


class ReplannerInput(TypedDict):
    """Validated state visible to the model-based replanner."""

    objective: str
    investigation_plan: InvestigationPlan | None
    candidate_result: CandidateResult | None
    evaluation_report: EvaluationReport | None
    human_feedback: str | None
    max_tool_calls_per_step: int
    max_explorer_calls_per_step: int
    plan_revision_count: int
    max_plan_revisions: int
    context_budget_tokens: int
    context_build_count: int
    max_attempt_cost_usd: str | None
    memory_context_items: list[ContextItem]
    last_tool_batch_audits: list[ToolAuditPayload]


class ReplannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    step_tool_call_count: int
    step_explorer_call_count: int
    last_tool_batch_audits: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: None
    progress_snapshot: None
    consecutive_no_progress_rounds: int
    plan_revision_count: int
    step_stop_reason: None
    candidate_result: None
    evaluation_report: None
    investigation_structure: None
    next_action: None
    human_review_payload: None
    human_decision_action: None
    human_feedback: None
    active_context_pack: ContextPack
    context_build_count: int


class RejectedRevisionUpdate(TypedDict):
    """A consumed model response rejected without changing the existing plan."""

    termination_status: TerminationStatus
    step_stop_reason: str
    plan_revision_count: int
    final_result: FinalResult
    active_context_pack: ContextPack
    context_build_count: int


ReplannerNode = Callable[[ReplannerInput], ReplannerNodeUpdate | RejectedRevisionUpdate]


def build_replanner_messages(
    state: ReplannerInput,
    *,
    gathered_context: GatheredContext | None = None,
) -> list[AnyMessage]:
    """Build an isolated replanning invocation from evaluated state."""

    investigation_plan = state["investigation_plan"]
    candidate = state["candidate_result"]
    evaluation_report = state["evaluation_report"]
    if investigation_plan is None:
        raise ValueError("Cannot revise without an InvestigationPlan")
    if candidate is None:
        raise ValueError("Cannot revise without a CandidateResult")
    if evaluation_report is None:
        raise ValueError("Cannot revise without an EvaluationReport")

    gathered = gathered_context or gather_replanner_context(
        objective=state["objective"],
        investigation_plan=investigation_plan,
        candidate_result=candidate,
        evaluation_report=evaluation_report,
        human_feedback=state["human_feedback"],
        plan_revision_count=state["plan_revision_count"],
        last_tool_batch_audits=tuple(
            decode_tool_audit_record(item)
            for item in state.get("last_tool_batch_audits", [])
        ),
        max_tool_calls_per_step=state["max_tool_calls_per_step"],
        max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
        budget_tokens=_context_budget(state),
        memory_items=tuple(state.get("memory_context_items", [])),
    )
    return [
        SystemMessage(content=REPLANNING_SYSTEM_PROMPT),
        HumanMessage(content=render_model_context(ModelRole.REPLANNER, gathered.pack)),
    ]


def create_revise_plan_node(
    model: BaseChatModel,
    recorder: RawRunRecorder | None = None,
    context_selection_strategy: ContextSelectionStrategy = (
        ContextSelectionStrategy.PRIORITIZED
    ),
) -> ReplannerNode:
    """Create the model node that revises only unfinished plan work."""

    structured_model = model.with_structured_output(
        InvestigationPlan,
        method="function_calling",
        include_raw=True,
    )

    def revise_plan_node(
        state: ReplannerInput,
        config: RunnableConfig | None = None,
    ) -> ReplannerNodeUpdate | RejectedRevisionUpdate:
        """Generate a bounded revision and reset step-local execution state."""

        current_plan = state["investigation_plan"]
        if current_plan is None:
            raise ValueError("Cannot revise without an InvestigationPlan")
        candidate = state["candidate_result"]
        if candidate is None:
            raise ValueError("Cannot revise without a CandidateResult")
        evaluation_report = state["evaluation_report"]
        if evaluation_report is None:
            raise ValueError("Cannot revise without an EvaluationReport")
        if state["plan_revision_count"] >= state["max_plan_revisions"]:
            raise ValueError(
                "Cannot revise after the plan-revision budget is exhausted"
            )

        gathered = gather_replanner_context(
            objective=state["objective"],
            investigation_plan=current_plan,
            candidate_result=candidate,
            evaluation_report=evaluation_report,
            human_feedback=state["human_feedback"],
            plan_revision_count=state["plan_revision_count"],
            last_tool_batch_audits=tuple(
                decode_tool_audit_record(item)
                for item in state.get("last_tool_batch_audits", [])
            ),
            max_tool_calls_per_step=state["max_tool_calls_per_step"],
            max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
            budget_tokens=_context_budget(state),
            memory_items=tuple(state.get("memory_context_items", [])),
            selection_strategy=context_selection_strategy,
        )
        messages = build_replanner_messages(state, gathered_context=gathered)
        try:
            response = _invoke_model(
                recorder=recorder,
                state=state,
                config=config,
                node_name="revise_plan",
                role=ModelRole.REPLANNER,
                model=model,
                gathered_context=gathered,
                messages=messages,
                plan_step_id=current_plan.active_step.id,
                invoke=structured_model.invoke,
                parse=lambda value: _parse_plan_revision(value, current_plan),
            )
        except PlanRevisionIntegrityError:
            # The recorder has already retained the failed response and usage.
            # Never accept the altered history or retry the same checkpoint
            # implicitly. Preserve candidate/plan diagnostics, but do not publish
            # their unvalidated claims as a usable final answer.
            message = (
                "Investigation stopped because the proposed plan revision added "
                "or changed completed steps. The existing plan was preserved; "
                "the candidate was not accepted. No automatic retry was made."
            )
            return {
                "termination_status": "exhausted",
                "step_stop_reason": "invalid_plan_revision",
                "plan_revision_count": state["plan_revision_count"] + 1,
                "final_result": FinalResult(
                    relevant_files=[],
                    evidence=[],
                    summary=message,
                    unresolved_questions=[state["objective"]],
                ),
                "active_context_pack": gathered.pack,
                "context_build_count": _next_context_build_count(state),
            }
        if isinstance(response, _ModelCostBudgetDenied):
            return _cost_budget_denied_update(
                response.budget, stop_reason=response.stop_reason
            )  # type: ignore[return-value]
        assert isinstance(response, InvestigationPlan)

        return {
            "investigation_plan": response,
            "step_tool_call_count": 0,
            "step_explorer_call_count": 0,
            "last_tool_batch_audits": [],
            "step_tool_audit_sequence_floor": None,
            "progress_snapshot": None,
            "consecutive_no_progress_rounds": 0,
            "plan_revision_count": state["plan_revision_count"] + 1,
            "step_stop_reason": None,
            "candidate_result": None,
            "evaluation_report": None,
            "investigation_structure": None,
            "next_action": None,
            "human_review_payload": None,
            "human_decision_action": None,
            "human_feedback": None,
            "active_context_pack": gathered.pack,
            "context_build_count": _next_context_build_count(state),
        }

    return revise_plan_node


class PlanRevisionIntegrityError(ValueError):
    """A well-formed model plan violated controller-owned completion history."""


def _parse_plan_revision(
    value: object, current: InvestigationPlan
) -> InvestigationPlan:
    """Validate completion provenance inside recorded response parsing."""

    revised = _parse_structured_response(
        value,
        expected_type=InvestigationPlan,
        error_message="Replanner model did not return InvestigationPlan",
    )
    completed = [step for step in current.steps if step.status == "completed"]
    revised_completed = [step for step in revised.steps if step.status == "completed"]
    if revised_completed != completed:
        raise PlanRevisionIntegrityError(
            "Replanner added or modified a completed plan step"
        )
    return revised


class CostBudgetFinalizerInput(TypedDict):
    """State available to the model-free cost-budget fallback."""

    objective: str
    candidate_result: CandidateResult | None
    cost_budget: AttemptCostBudgetSummary | None
    investigation_structure: InvestigationStructure | None
    step_stop_reason: NotRequired[str | None]


class CostBudgetFinalizerUpdate(TypedDict):
    termination_status: TerminationStatus
    step_stop_reason: str
    final_result: FinalResult


def finalize_cost_budget_exhausted(
    state: CostBudgetFinalizerInput,
) -> CostBudgetFinalizerUpdate:
    """Publish the best available result without making another model call."""

    budget = state["cost_budget"]
    stop_reason = state.get("step_stop_reason")
    precall_denied = stop_reason in PRECALL_BUDGET_STOP_REASONS
    if not precall_denied and (budget is None or budget.next_invocation_allowed):
        raise ValueError("cost budget fallback requires a denied decision")
    objective = state["objective"].strip()
    if not objective:
        raise ValueError("cost budget fallback requires a non-empty objective")
    if precall_denied:
        budget_message = (
            "Investigation stopped before another model invocation because its "
            f"cost reservation did not fit the configured spending limits ({stop_reason})."
        )
    else:
        assert budget is not None
        stop_reason = ATTEMPT_COST_BUDGET_STOP_REASON
        budget_message = (
            "Investigation stopped before another model invocation because the "
            f"attempt cost budget denied it ({budget.reason.value})."
        )
    structure = state.get("investigation_structure")
    if structure is not None and not isinstance(structure, InvestigationStructure):
        raise TypeError("investigation_structure is invalid")
    if structure is not None:
        structure = revalidate_investigation_structure(structure)
    candidate = state["candidate_result"]
    if candidate is None:
        result = FinalResult(
            relevant_files=[],
            evidence=[],
            summary=budget_message,
            unresolved_questions=[objective],
        )
    else:
        result = FinalResult(
            relevant_files=list(
                dict.fromkeys(item.path for item in candidate.evidence)
            ),
            evidence=[item.claim for item in candidate.evidence],
            summary=f"{candidate.summary}\n\n{budget_message}",
            unresolved_questions=list(
                dict.fromkeys([*candidate.unresolved_questions, budget_message])
            ),
            relevant_symbols=(
                [item.canonical_name for item in structure.relevant_symbols]
                if structure is not None
                else []
            ),
            structure=structure,
        )
    return {
        "termination_status": "exhausted",
        "step_stop_reason": stop_reason,
        "final_result": result,
    }


class TerminationInput(TypedDict):
    """Empty state view for a constant terminal update."""


class TerminationUpdate(TypedDict):
    termination_status: TerminationStatus


def mark_investigation_accepted(_state: TerminationInput) -> TerminationUpdate:
    """Record that the evaluated investigation result was accepted."""

    return {"termination_status": "accepted"}


def mark_investigation_exhausted(_state: TerminationInput) -> TerminationUpdate:
    """Record that no bounded retry or plan revision remains."""

    return {"termination_status": "exhausted"}


def mark_investigation_cancelled(_state: TerminationInput) -> TerminationUpdate:
    """Record that Human review cancelled the current turn."""

    return {"termination_status": "cancelled"}


class PrepareHumanReviewInput(TypedDict):
    """State required to materialize a checkpoint-safe review request."""

    human_review_policy: HumanReviewPolicy
    turn_number: int
    candidate_result: CandidateResult | None
    evaluation_report: EvaluationReport | None
    plan_revision_count: int
    max_plan_revisions: int


class PrepareHumanReviewUpdate(TypedDict):
    human_review_payload: HumanReviewInterruptPayload | None
    human_decision_action: None
    human_feedback: None


def prepare_human_review(
    state: PrepareHumanReviewInput,
) -> PrepareHumanReviewUpdate:
    """Persist a JSON review envelope before entering the interrupting node."""

    policy = state["human_review_policy"]
    if policy is HumanReviewPolicy.NEVER:
        return {
            "human_review_payload": None,
            "human_decision_action": None,
            "human_feedback": None,
        }
    if policy is not HumanReviewPolicy.BEFORE_FINALIZE:
        raise ValueError(f"Unknown human review policy: {policy!r}")

    candidate = state["candidate_result"]
    if candidate is None:
        raise ValueError("Cannot request Human review without a CandidateResult")
    evaluation_report = state["evaluation_report"]
    if evaluation_report is None:
        raise ValueError("Cannot request Human review without an EvaluationReport")

    request = HumanReviewRequest(
        turn_number=state["turn_number"],
        candidate_result=candidate,
        evaluation_report=evaluation_report,
        allowed_actions=allowed_human_decision_actions(
            state["plan_revision_count"],
            state["max_plan_revisions"],
        ),
    )
    return {
        "human_review_payload": encode_human_review_interrupt(request),
        "human_decision_action": None,
        "human_feedback": None,
    }


class HumanReviewNodeInput(TypedDict):
    """State required by the optional final-candidate review gate."""

    human_review_policy: HumanReviewPolicy
    human_review_payload: HumanReviewInterruptPayload | None


class HumanReviewNodeUpdate(TypedDict):
    human_review_policy: HumanReviewPolicy
    human_decision_action: HumanDecisionActionValue
    human_feedback: str | None


def review_final_candidate(
    state: HumanReviewNodeInput,
) -> HumanReviewNodeUpdate:
    """Pause for a versioned decision before accepted finalization."""

    policy = state["human_review_policy"]
    if policy is HumanReviewPolicy.NEVER:
        return {
            "human_review_policy": policy,
            "human_decision_action": "approve",
            "human_feedback": None,
        }
    if policy is not HumanReviewPolicy.BEFORE_FINALIZE:
        raise ValueError(f"Unknown human review policy: {policy!r}")

    payload = state["human_review_payload"]
    if payload is None:
        raise ValueError("Cannot request Human review without a prepared payload")
    review_request = decode_human_review_interrupt(payload)
    decision = decode_human_decision(interrupt(payload))
    if decision.action not in review_request.allowed_actions:
        raise ValueError(
            f"Human decision action is not allowed: {decision.action.value}"
        )

    return {
        "human_review_policy": policy,
        "human_decision_action": decision.action.value,
        "human_feedback": decision.feedback,
    }


class FinalizerInput(TypedDict):
    """Minimal input for deterministic candidate publication."""

    candidate_result: CandidateResult | None
    investigation_structure: InvestigationStructure | None


class FinalizeNodeUpdate(TypedDict):
    final_result: FinalResult


def finalize_candidate(state: FinalizerInput) -> FinalizeNodeUpdate:
    """Publish the current candidate through the stable public result contract."""

    candidate = state["candidate_result"]
    if candidate is None:
        raise ValueError("Cannot finalize without a CandidateResult")
    structure = state.get("investigation_structure")
    if structure is not None and not isinstance(structure, InvestigationStructure):
        raise TypeError("investigation_structure is invalid")
    if structure is not None:
        structure = revalidate_investigation_structure(structure)

    return {
        "final_result": FinalResult(
            relevant_files=list(
                dict.fromkeys(item.path for item in candidate.evidence)
            ),
            evidence=[item.claim for item in candidate.evidence],
            summary=candidate.summary,
            unresolved_questions=candidate.unresolved_questions,
            relevant_symbols=(
                [item.canonical_name for item in structure.relevant_symbols]
                if structure is not None
                else []
            ),
            structure=structure,
        )
    }
