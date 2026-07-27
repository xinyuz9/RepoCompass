"""State-mutating nodes and node factories for the RepoCompass graph."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import MessagesState

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import (
    EvaluationReport,
    EvidenceExcerpt,
    SemanticAssessment,
    ValidationIssue,
    validate_candidate,
)
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.prompts import (
    CANDIDATE_BUILDING_REQUEST_TEMPLATE,
    CANDIDATE_BUILDING_SYSTEM_PROMPT,
    EVALUATION_REQUEST_TEMPLATE,
    EVALUATION_SYSTEM_PROMPT,
    INVESTIGATION_CONTEXT_TEMPLATE,
    INVESTIGATION_SYSTEM_PROMPT,
    PLANNING_REQUEST_TEMPLATE,
    PLANNING_SYSTEM_PROMPT,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS


class PlannerInput(TypedDict):
    """Minimal initial-planning view without repository content or messages."""

    objective: str
    max_tool_calls_per_step: int
    max_explorer_calls_per_step: int


class PlannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    active_objective: str


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

        active_step = next(
            step for step in response.steps if step.id == response.active_step_id
        )
        return {
            "investigation_plan": response,
            "active_objective": active_step.objective,
        }

    return planner_node


class AgentNodeInput(MessagesState):
    """State visible to one Explorer model invocation."""

    objective: str
    active_objective: str
    investigation_plan: InvestigationPlan | None
    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int


class AgentNodeUpdate(TypedDict):
    messages: list[AnyMessage]
    step_explorer_call_count: int


AgentNode = Callable[[AgentNodeInput], AgentNodeUpdate]


def build_agent_messages(state: AgentNodeInput) -> list[AnyMessage]:
    """Build one Explorer invocation without mutating persistent messages."""

    investigation_plan = (
        "No investigation plan is available."
        if not state["investigation_plan"]
        else state["investigation_plan"].model_dump_json(indent=2)
    )
    investigation_context = INVESTIGATION_CONTEXT_TEMPLATE.format(
        objective=state["objective"],
        active_objective=state["active_objective"],
        investigation_plan=investigation_plan,
        remaining_tool_calls=max(
            0,
            state["max_tool_calls_per_step"] - state["step_tool_call_count"],
        ),
        remaining_explorer_calls=max(
            0,
            state["max_explorer_calls_per_step"] - state["step_explorer_call_count"],
        ),
    )
    return [
        SystemMessage(
            content=f"{INVESTIGATION_SYSTEM_PROMPT}\n\n{investigation_context}"
        ),
        *state["messages"],
    ]


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
    step_explorer_call_count: int
    max_explorer_calls_per_step: int


class ExplorerBudgetUpdate(TypedDict):
    stop_reason: str


def mark_step_explorer_budget_exhausted(
    _state: ExplorerBudgetInput,
) -> ExplorerBudgetUpdate:
    """Record that the active step cannot invoke Explorer again."""

    return {"stop_reason": "step_explorer_budget_exhausted"}


class ToolBudgetInput(MessagesState):
    step_tool_call_count: int


ToolCountUpdate = dict[str, int]


def reserve_tool_budget(state: ToolBudgetInput) -> ToolCountUpdate:
    """Reserve budget for every tool call in the latest AI message."""

    batch_tool_call_count = len(state["messages"][-1].tool_calls)
    return {
        "step_tool_call_count": batch_tool_call_count + state["step_tool_call_count"]
    }


class ToolRejectionInput(MessagesState):
    pass


class ToolRejectionUpdate(TypedDict):
    messages: list[ToolMessage]
    stop_reason: str


def reject_pending_tool_calls(
    state: ToolRejectionInput,
) -> ToolRejectionUpdate:
    """Return error messages for calls rejected by the execution budget."""

    last_message = state["messages"][-1]
    messages: list[ToolMessage] = []
    for tool_call in last_message.tool_calls + last_message.invalid_tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because the tool-call budget was "
                    "exhausted. Answer using the available evidence."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status="error",
            )
        )

    return {
        "messages": messages,
        "stop_reason": "step_tool_call_budget_exhausted",
    }


class InvalidToolBatchInput(MessagesState):
    step_tool_call_count: int


class InvalidToolBatchUpdate(TypedDict):
    messages: list[ToolMessage]
    step_tool_call_count: int


def handle_invalid_tool_batch(
    state: InvalidToolBatchInput,
) -> InvalidToolBatchUpdate:
    """Return model-visible errors for a batch containing invalid tool calls."""

    last_message = state["messages"][-1]
    messages: list[ToolMessage] = []
    batch_tool_call_count = len(last_message.invalid_tool_calls) + len(
        last_message.tool_calls
    )

    for invalid_tool_call in last_message.invalid_tool_calls:
        error_message = (
            invalid_tool_call["error"] or "Tool arguments could not be parsed."
        )
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because its arguments could not be "
                    f"parsed. Parser error: {error_message} Submit a corrected tool "
                    "call with valid JSON arguments."
                ),
                tool_call_id=invalid_tool_call["id"],
                name=invalid_tool_call["name"],
                status="error",
            )
        )

    for tool_call in last_message.tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because another call in the same "
                    "batch had invalid arguments. Resubmit the complete batch with "
                    "corrected arguments."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status="error",
            )
        )

    return {
        "messages": messages,
        "step_tool_call_count": batch_tool_call_count + state["step_tool_call_count"],
    }


class CandidateBuilderInput(MessagesState):
    """Evidence-bearing state visible to candidate construction."""

    objective: str
    active_objective: str
    investigation_plan: InvestigationPlan | None
    stop_reason: str | None


class CandidateBuilderUpdate(TypedDict):
    candidate_result: CandidateResult


CandidateBuilderNode = Callable[[CandidateBuilderInput], CandidateBuilderUpdate]


def build_candidate_messages(state: CandidateBuilderInput) -> list[AnyMessage]:
    """Build an isolated structured-candidate invocation."""

    investigation_plan = (
        "No investigation plan is available."
        if not state["investigation_plan"]
        else state["investigation_plan"].model_dump_json(indent=2)
    )
    stop_reason = state["stop_reason"] or "none"
    return [
        SystemMessage(content=CANDIDATE_BUILDING_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(
            content=CANDIDATE_BUILDING_REQUEST_TEMPLATE.format(
                objective=state["objective"],
                active_objective=state["active_objective"],
                investigation_plan=investigation_plan,
                stop_reason=stop_reason,
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
    candidate_result: CandidateResult | None


class EvaluationPromptInput(TypedDict):
    """Validated data rendered into one isolated evaluator invocation."""

    objective: str
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

        validation = validate_candidate(state["repo_path"], candidate)
        prompt_input: EvaluationPromptInput = {
            "objective": state["objective"],
            "candidate_result": candidate,
            "deterministic_issues": validation.issues,
            "evidence_pack": validation.evidence_pack,
        }
        response = structured_model.invoke(build_evaluator_messages(prompt_input))
        if not isinstance(response, SemanticAssessment):
            raise TypeError("Evaluator model did not return SemanticAssessment")

        passed = (
            not validation.issues
            and response.answers_objective
            and response.evidence_supports_summary
            and not response.has_critical_omissions
        )
        return {
            "evaluation_report": EvaluationReport(
                deterministic_issues=list(validation.issues),
                semantic_assessment=response,
                missing_evidence=response.missing_evidence,
                recommended_focus=response.recommended_focus,
                passed=passed,
            )
        }

    return evaluate_candidate_node


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
            relevant_files=candidate.relevant_files,
            evidence=[item.claim for item in candidate.evidence],
            summary=candidate.summary,
            unresolved_questions=candidate.unresolved_questions,
        )
    }
