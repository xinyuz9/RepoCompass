"""State-mutating nodes and node factories for the RepoCompass graph."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.types import interrupt
from pydantic import ValidationError

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import (
    EvaluationReport,
    EvidenceExcerpt,
    SemanticAssessment,
    ValidationIssue,
    validate_candidate,
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
    CANDIDATE_BUILDING_REQUEST_TEMPLATE,
    CANDIDATE_BUILDING_SYSTEM_PROMPT,
    EVALUATION_REQUEST_TEMPLATE,
    EVALUATION_SYSTEM_PROMPT,
    INVESTIGATION_CONTEXT_TEMPLATE,
    INVESTIGATION_FEEDBACK_TEMPLATE,
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_REQUEST_TEMPLATE,
    PLANNING_SYSTEM_PROMPT,
    REPLANNING_REQUEST_TEMPLATE,
    REPLANNING_SYSTEM_PROMPT,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import (
    ControllerAction,
    HumanDecisionActionValue,
    TerminationStatus,
)
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


class PlannerInput(TypedDict):
    """Minimal initial-planning view without repository content or messages."""

    objective: str
    max_tool_calls_per_step: int
    max_explorer_calls_per_step: int


class PlannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan


PlannerNode = Callable[[PlannerInput], PlannerNodeUpdate]


def build_planner_messages(state: PlannerInput) -> list[AnyMessage]:
    """Build an isolated initial-planning invocation."""

    return [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(
            content=PLANNING_REQUEST_TEMPLATE.format(
                objective=state["objective"],
                max_tool_calls_per_step=state["max_tool_calls_per_step"],
                max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
            )
        ),
    ]


def create_plan_investigation_node(model: BaseChatModel) -> PlannerNode:
    """Create the node that generates the initial investigation plan."""

    structured_model = model.with_structured_output(
        InvestigationPlan,
        method="function_calling",
    )

    def planner_node(state: PlannerInput) -> PlannerNodeUpdate:
        """Generate the initial plan and activate its designated step."""

        response = structured_model.invoke(build_planner_messages(state))
        if not isinstance(response, InvestigationPlan):
            raise TypeError("Planner model did not return InvestigationPlan")

        return {"investigation_plan": response}

    return planner_node


class AgentNodeInput(MessagesState):
    """State visible to one Explorer model invocation."""

    objective: str
    investigation_plan: InvestigationPlan | None
    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    evaluation_report: EvaluationReport | None


class AgentNodeUpdate(TypedDict):
    messages: list[AnyMessage]
    step_explorer_call_count: int


AgentNode = Callable[[AgentNodeInput], AgentNodeUpdate]


def build_agent_messages(state: AgentNodeInput) -> list[AnyMessage]:
    """Build one Explorer invocation without mutating persistent messages."""

    investigation_plan = state["investigation_plan"]
    if investigation_plan is None:
        raise ValueError("Cannot investigate without an InvestigationPlan")

    investigation_context = INVESTIGATION_CONTEXT_TEMPLATE.format(
        objective=state["objective"],
        active_objective=investigation_plan.active_step.objective,
        active_step=investigation_plan.active_step.model_dump_json(indent=2),
        remaining_tool_calls=max(
            0,
            state["max_tool_calls_per_step"] - state["step_tool_call_count"],
        ),
        remaining_explorer_calls=max(
            0,
            state["max_explorer_calls_per_step"] - state["step_explorer_call_count"],
        ),
    )
    messages: list[AnyMessage] = [
        SystemMessage(
            content=f"{INVESTIGATION_SYSTEM_PROMPT}\n\n{investigation_context}"
        ),
        *state["messages"],
    ]
    evaluation_report = state["evaluation_report"]
    if evaluation_report is not None and not evaluation_report.passed:
        messages.append(
            HumanMessage(
                content=INVESTIGATION_FEEDBACK_TEMPLATE.format(
                    missing_evidence=json.dumps(
                        evaluation_report.semantic_assessment.missing_evidence,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    recommended_focus=json.dumps(
                        evaluation_report.semantic_assessment.recommended_focus,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            )
        )

    return messages


def create_agent_node(model: BaseChatModel) -> AgentNode:
    """Bind repository tools once and create the graph's Explorer node."""

    bound_model = model.bind_tools(REPOSITORY_TOOLS)

    def agent_node(state: AgentNodeInput) -> AgentNodeUpdate:
        """Call the tool-bound model once and return its state update."""

        response = bound_model.invoke(build_agent_messages(state))
        return {
            "messages": [response],
            "step_explorer_call_count": state["step_explorer_call_count"] + 1,
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


class ToolGatewayNodeUpdate(TypedDict):
    messages: list[ToolMessage]
    step_tool_call_count: int
    step_stop_reason: str | None
    tool_audit_records: list[ToolAuditPayload]


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
        audit_log = ToolAuditLog(
            run_id,
            records=prior_run_records,
            deduplication_floor=len(prior_run_records),
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

    objective: str
    investigation_plan: InvestigationPlan | None
    step_stop_reason: str | None


class CandidateBuilderUpdate(TypedDict):
    candidate_result: CandidateResult


CandidateBuilderNode = Callable[[CandidateBuilderInput], CandidateBuilderUpdate]


def build_candidate_messages(state: CandidateBuilderInput) -> list[AnyMessage]:
    """Build an isolated structured-candidate invocation.

    Tool results are rendered as ordinary JSON data so historical repository
    tool calls cannot be mistaken for tools available to the structured model.
    """

    investigation_plan = state["investigation_plan"]
    if investigation_plan is None:
        raise ValueError("Cannot build a candidate without an InvestigationPlan")

    collected_tool_results = [
        {
            "tool": message.name,
            "content": message.content,
        }
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.status == "success"
    ]
    step_stop_reason = state["step_stop_reason"] or "none"
    return [
        SystemMessage(content=CANDIDATE_BUILDING_SYSTEM_PROMPT),
        HumanMessage(
            content=CANDIDATE_BUILDING_REQUEST_TEMPLATE.format(
                objective=state["objective"],
                active_objective=investigation_plan.active_step.objective,
                active_step=investigation_plan.active_step.model_dump_json(indent=2),
                step_stop_reason=step_stop_reason,
                collected_tool_results=json.dumps(
                    collected_tool_results,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ),
    ]


def create_build_candidate_node(model: BaseChatModel) -> CandidateBuilderNode:
    """Create a node that turns Explorer evidence into an internal candidate."""

    structured_model = model.with_structured_output(
        CandidateResult,
        method="function_calling",
    )

    def build_candidate_node(
        state: CandidateBuilderInput,
    ) -> CandidateBuilderUpdate:
        """Generate one structured candidate from the collected evidence."""

        response = structured_model.invoke(build_candidate_messages(state))
        if not isinstance(response, CandidateResult):
            raise TypeError("Candidate builder model did not return CandidateResult")

        return {"candidate_result": response}

    return build_candidate_node


class EvaluatorInput(TypedDict):
    """Minimal trusted state visible to candidate evaluation."""

    repo_path: Path
    objective: str
    investigation_plan: InvestigationPlan | None
    candidate_result: CandidateResult | None


class EvaluationPromptInput(TypedDict):
    """Validated data rendered into one isolated evaluator invocation."""

    objective: str
    active_step: PlanStep
    candidate_result: CandidateResult
    deterministic_issues: tuple[ValidationIssue, ...]
    evidence_pack: tuple[EvidenceExcerpt, ...]


class EvaluatorNodeUpdate(TypedDict):
    evaluation_report: EvaluationReport


EvaluatorNode = Callable[[EvaluatorInput], EvaluatorNodeUpdate]


def build_evaluator_messages(state: EvaluationPromptInput) -> list[AnyMessage]:
    """Build one evaluator invocation without Explorer message history."""

    deterministic_issues = [
        issue.model_dump(mode="json") for issue in state["deterministic_issues"]
    ]
    evidence_pack = [
        excerpt.model_dump(mode="json") for excerpt in state["evidence_pack"]
    ]
    return [
        SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
        HumanMessage(
            content=EVALUATION_REQUEST_TEMPLATE.format(
                objective=state["objective"],
                active_objective=state["active_step"].objective,
                expected_evidence=json.dumps(
                    state["active_step"].expected_evidence,
                    ensure_ascii=False,
                    indent=2,
                ),
                candidate_result=state["candidate_result"].model_dump_json(indent=2),
                deterministic_issues=json.dumps(
                    deterministic_issues,
                    ensure_ascii=False,
                    indent=2,
                ),
                evidence_pack=json.dumps(
                    evidence_pack,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ),
    ]


def create_evaluate_candidate_node(model: BaseChatModel) -> EvaluatorNode:
    """Create the isolated semantic evaluator with deterministic validation."""

    structured_model = model.with_structured_output(
        SemanticAssessment,
        method="function_calling",
    )

    def evaluate_candidate_node(state: EvaluatorInput) -> EvaluatorNodeUpdate:
        """Validate one candidate and return a merged evaluation report."""

        candidate = state["candidate_result"]
        if candidate is None:
            raise ValueError("Cannot evaluate without a CandidateResult")
        investigation_plan = state["investigation_plan"]
        if investigation_plan is None:
            raise ValueError("Cannot evaluate without an InvestigationPlan")

        validation = validate_candidate(state["repo_path"], candidate)
        prompt_input: EvaluationPromptInput = {
            "objective": state["objective"],
            "active_step": investigation_plan.active_step,
            "candidate_result": candidate,
            "deterministic_issues": validation.issues,
            "evidence_pack": validation.evidence_pack,
        }
        response = structured_model.invoke(build_evaluator_messages(prompt_input))
        if not isinstance(response, SemanticAssessment):
            raise TypeError("Evaluator model did not return SemanticAssessment")

        return {
            "evaluation_report": EvaluationReport(
                deterministic_issues=list(validation.issues),
                semantic_assessment=response,
            )
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

    if semantic_assessment.active_step_completed:
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
    step_stop_reason: None
    candidate_result: None
    evaluation_report: None
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
        "step_stop_reason": None,
        "candidate_result": None,
        "evaluation_report": None,
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


class ReplannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    step_tool_call_count: int
    step_explorer_call_count: int
    plan_revision_count: int
    step_stop_reason: None
    candidate_result: None
    evaluation_report: None
    next_action: None
    human_review_payload: None
    human_decision_action: None
    human_feedback: None


ReplannerNode = Callable[[ReplannerInput], ReplannerNodeUpdate]


def build_replanner_messages(state: ReplannerInput) -> list[AnyMessage]:
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

    return [
        SystemMessage(content=REPLANNING_SYSTEM_PROMPT),
        HumanMessage(
            content=REPLANNING_REQUEST_TEMPLATE.format(
                objective=state["objective"],
                human_feedback=state["human_feedback"] or "None provided.",
                investigation_plan=investigation_plan.model_dump_json(indent=2),
                candidate_result=candidate.model_dump_json(indent=2),
                evaluation_report=evaluation_report.model_dump_json(indent=2),
                plan_revision_count=state["plan_revision_count"],
                max_tool_calls_per_step=state["max_tool_calls_per_step"],
                max_explorer_calls_per_step=state["max_explorer_calls_per_step"],
            )
        ),
    ]


def create_revise_plan_node(model: BaseChatModel) -> ReplannerNode:
    """Create the model node that revises only unfinished plan work."""

    structured_model = model.with_structured_output(
        InvestigationPlan,
        method="function_calling",
    )

    def revise_plan_node(state: ReplannerInput) -> ReplannerNodeUpdate:
        """Generate a bounded revision and reset step-local execution state."""

        current_plan = state["investigation_plan"]
        if current_plan is None:
            raise ValueError("Cannot revise without an InvestigationPlan")
        if state["plan_revision_count"] >= state["max_plan_revisions"]:
            raise ValueError(
                "Cannot revise after the plan-revision budget is exhausted"
            )

        response = structured_model.invoke(build_replanner_messages(state))
        if not isinstance(response, InvestigationPlan):
            raise TypeError("Replanner model did not return InvestigationPlan")

        revised_by_id = {step.id: step for step in response.steps}
        for completed_step in (
            step for step in current_plan.steps if step.status == "completed"
        ):
            if revised_by_id.get(completed_step.id) != completed_step:
                raise ValueError("Replanner modified a completed plan step")

        return {
            "investigation_plan": response,
            "step_tool_call_count": 0,
            "step_explorer_call_count": 0,
            "plan_revision_count": state["plan_revision_count"] + 1,
            "step_stop_reason": None,
            "candidate_result": None,
            "evaluation_report": None,
            "next_action": None,
            "human_review_payload": None,
            "human_decision_action": None,
            "human_feedback": None,
        }

    return revise_plan_node


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


class FinalizeNodeUpdate(TypedDict):
    final_result: FinalResult


def finalize_candidate(state: FinalizerInput) -> FinalizeNodeUpdate:
    """Publish the current candidate through the stable public result contract."""

    candidate = state["candidate_result"]
    if candidate is None:
        raise ValueError("Cannot finalize without a CandidateResult")

    return {
        "final_result": FinalResult(
            relevant_files=list(
                dict.fromkeys(item.path for item in candidate.evidence)
            ),
            evidence=[item.claim for item in candidate.evidence],
            summary=candidate.summary,
            unresolved_questions=candidate.unresolved_questions,
        )
    }
