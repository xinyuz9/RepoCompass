import re
from collections.abc import Mapping
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict, cast

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import MessagesState

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.context import (
    ContextItem,
    ContextPack,
    MemoryRecallReport,
)
from repocompass.agent.context_builder import DEFAULT_CONTEXT_BUDGET_TOKENS
from repocompass.agent.cost import (
    AttemptCostBudgetSummary,
    validate_cost_budget_usd,
)
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import (
    HumanReviewInterruptPayload,
    HumanReviewHistory,
    HumanReviewPolicy,
    TurnRecord,
)
from repocompass.agent.review_history import validated_review_history
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult, revalidate_final_result
from repocompass.agent.routing import (
    ControllerAction,
    HumanDecisionActionValue,
    TerminationStatus,
)
from repocompass.agent.structural_artifacts import InvestigationStructure
from repocompass.tools.contracts import (
    SHA256_PATTERN,
    NoProgressDecisionPayload,
    ProgressSnapshotPayload,
    ToolAuditPayload,
)


class RepoCompassState(MessagesState):
    """Canonical internal state for one repository investigation."""

    repo_path: Path
    objective: str
    turn_number: int
    turn_history: list[TurnRecord]
    review_history: HumanReviewHistory | None
    cancelled_run_id: str | None
    cancellation_request_id: str | None
    human_review_policy: HumanReviewPolicy
    human_review_payload: HumanReviewInterruptPayload | None
    human_decision_action: HumanDecisionActionValue | None
    human_feedback: str | None
    investigation_plan: InvestigationPlan | None
    memory_retrieval_enabled: bool
    max_repository_memory_results: int
    max_episodic_memory_results: int
    memory_recall_report: MemoryRecallReport | None
    memory_context_items: list[ContextItem]
    tool_audit_records: Annotated[list[ToolAuditPayload], add]
    last_tool_batch_audits: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: int | None

    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    progress_snapshot: ProgressSnapshotPayload | None
    no_progress_decision: NoProgressDecisionPayload | None
    consecutive_no_progress_rounds: int
    max_no_progress_rounds: int
    plan_revision_count: int
    max_plan_revisions: int
    context_budget_tokens: int
    context_build_count: int
    active_context_pack: ContextPack | None
    max_attempt_cost_usd: str | None
    cost_budget_pricing_table_id: str | None
    cost_budget: AttemptCostBudgetSummary | None

    step_stop_reason: str | None
    termination_status: TerminationStatus | None
    candidate_result: CandidateResult | None
    candidate_output_contract: str | None
    evaluation_report: EvaluationReport | None
    investigation_structure: InvestigationStructure | None
    next_action: ControllerAction | None
    final_result: FinalResult | None


class ContinuationStateUpdate(TypedDict):
    """State update that starts a new turn on one completed thread."""

    messages: list[AnyMessage]
    repo_path: Path
    objective: str
    turn_number: int
    turn_history: list[TurnRecord]
    review_history: HumanReviewHistory
    cancelled_run_id: None
    cancellation_request_id: None
    human_review_policy: HumanReviewPolicy
    human_review_payload: None
    human_decision_action: None
    human_feedback: None
    investigation_plan: None
    memory_retrieval_enabled: bool
    max_repository_memory_results: int
    max_episodic_memory_results: int
    memory_recall_report: None
    memory_context_items: list[ContextItem]
    tool_audit_records: list[ToolAuditPayload]
    last_tool_batch_audits: list[ToolAuditPayload]
    step_tool_audit_sequence_floor: int
    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    progress_snapshot: None
    no_progress_decision: None
    consecutive_no_progress_rounds: int
    max_no_progress_rounds: int
    plan_revision_count: int
    max_plan_revisions: int
    step_stop_reason: None
    context_budget_tokens: int
    context_build_count: int
    active_context_pack: None
    max_attempt_cost_usd: str | None
    cost_budget_pricing_table_id: str | None
    cost_budget: None
    termination_status: None
    candidate_result: None
    candidate_output_contract: None
    evaluation_report: None
    investigation_structure: None
    next_action: None
    final_result: None


def _validate_cost_budget_pricing_table_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("cost budget pricing table ID must be a string")
    if re.fullmatch(SHA256_PATTERN, value) is None:
        raise ValueError("cost budget pricing table ID must be a SHA-256 digest")
    return value


def _require_cost_budget_pricing_binding(
    max_cost_usd: str | None,
    pricing_table_id: object,
) -> str | None:
    if (max_cost_usd is None) != (pricing_table_id is None):
        raise ValueError(
            "attempt cost budget and pricing table ID must be configured together"
        )
    return (
        None
        if pricing_table_id is None
        else _validate_cost_budget_pricing_table_id(pricing_table_id)
    )


def create_initial_state(
    repo_path: Path,
    objective: str,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
    max_plan_revisions: int = 2,
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.NEVER,
    max_no_progress_rounds: int = 3,
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    max_attempt_cost_usd: str | None = None,
    cost_budget_pricing_table_id: str | None = None,
    memory_retrieval_enabled: bool = False,
    max_repository_memory_results: int = 5,
    max_episodic_memory_results: int = 5,
) -> RepoCompassState:
    objective = objective.strip()
    if not objective:
        raise ValueError("Objective must not be empty")
    if max_tool_calls_per_step < 1:
        raise ValueError(
            f"Max tool calls per step must be positive, not {max_tool_calls_per_step}"
        )
    if max_explorer_calls_per_step < 1:
        raise ValueError(
            "Max Explorer calls per step must be positive, "
            f"not {max_explorer_calls_per_step}"
        )
    if not isinstance(max_no_progress_rounds, int) or isinstance(
        max_no_progress_rounds,
        bool,
    ):
        raise TypeError("Max no-progress rounds must be an integer")
    if max_no_progress_rounds < 1:
        raise ValueError("Max no-progress rounds must be positive")
    if max_plan_revisions < 1:
        raise ValueError(
            f"Max plan revisions must be positive, not {max_plan_revisions}"
        )

    if not isinstance(context_budget_tokens, int) or isinstance(
        context_budget_tokens,
        bool,
    ):
        raise TypeError("Context budget must be an integer")
    if context_budget_tokens < 1:
        raise ValueError("Context budget must be positive")
    if type(memory_retrieval_enabled) is not bool:
        raise TypeError("memory_retrieval_enabled must be a bool")
    _validate_memory_result_limit(
        max_repository_memory_results,
        name="max_repository_memory_results",
    )
    _validate_memory_result_limit(
        max_episodic_memory_results,
        name="max_episodic_memory_results",
    )
    if max_attempt_cost_usd is not None:
        max_attempt_cost_usd = validate_cost_budget_usd(max_attempt_cost_usd)
    cost_budget_pricing_table_id = _require_cost_budget_pricing_binding(
        max_attempt_cost_usd, cost_budget_pricing_table_id
    )
    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path.resolve(),
        "objective": objective,
        "turn_number": 1,
        "turn_history": [],
        "review_history": HumanReviewHistory(),
        "cancelled_run_id": None,
        "cancellation_request_id": None,
        "human_review_policy": human_review_policy,
        "human_review_payload": None,
        "human_decision_action": None,
        "human_feedback": None,
        "investigation_plan": None,
        "memory_retrieval_enabled": memory_retrieval_enabled,
        "max_repository_memory_results": max_repository_memory_results,
        "max_episodic_memory_results": max_episodic_memory_results,
        "memory_recall_report": None,
        "memory_context_items": [],
        "tool_audit_records": [],
        "last_tool_batch_audits": [],
        "step_tool_audit_sequence_floor": 0,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": max_tool_calls_per_step,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": max_explorer_calls_per_step,
        "progress_snapshot": None,
        "no_progress_decision": None,
        "consecutive_no_progress_rounds": 0,
        "max_no_progress_rounds": max_no_progress_rounds,
        "plan_revision_count": 0,
        "max_plan_revisions": max_plan_revisions,
        "step_stop_reason": None,
        "termination_status": None,
        "candidate_result": None,
        "candidate_output_contract": None,
        "context_budget_tokens": context_budget_tokens,
        "context_build_count": 0,
        "active_context_pack": None,
        "max_attempt_cost_usd": max_attempt_cost_usd,
        "cost_budget_pricing_table_id": cost_budget_pricing_table_id,
        "cost_budget": None,
        "evaluation_report": None,
        "investigation_structure": None,
        "next_action": None,
        "final_result": None,
    }


def create_continuation_update(
    state: Mapping[str, object],
    prompt: str,
) -> ContinuationStateUpdate:
    """Archive a completed turn and reset state for the next Human prompt."""

    objective = prompt.strip()
    if not objective:
        raise ValueError("Continuation prompt must not be empty")

    final_result = state.get("final_result")
    termination_status = state.get("termination_status")
    if not isinstance(final_result, FinalResult) or termination_status not in (
        "accepted",
        "exhausted",
        "cancelled",
    ):
        raise ValueError("Continuation requires a completed turn")
    final_result = revalidate_final_result(final_result)

    turn_number = state.get("turn_number")
    previous_objective = state.get("objective")
    turn_history = state.get("turn_history")
    repo_path = state.get("repo_path")
    human_review_policy = state.get("human_review_policy")
    max_tool_calls = state.get("max_tool_calls_per_step")
    max_explorer_calls = state.get("max_explorer_calls_per_step")
    max_no_progress_rounds = state.get("max_no_progress_rounds")
    max_plan_revisions = state.get("max_plan_revisions")
    memory_retrieval_enabled = state.get("memory_retrieval_enabled", False)
    max_repository_memory_results = state.get("max_repository_memory_results", 5)
    max_episodic_memory_results = state.get("max_episodic_memory_results", 5)
    if type(memory_retrieval_enabled) is not bool:
        raise TypeError("Graph memory_retrieval_enabled must be a bool")
    _validate_memory_result_limit(
        max_repository_memory_results,
        name="max_repository_memory_results",
    )
    _validate_memory_result_limit(
        max_episodic_memory_results,
        name="max_episodic_memory_results",
    )
    if not isinstance(turn_number, int) or isinstance(turn_number, bool):
        raise TypeError("Graph turn_number is not an integer")
    if turn_number < 1:
        raise ValueError("Graph turn_number must be positive")
    context_budget_tokens = state.get(
        "context_budget_tokens", DEFAULT_CONTEXT_BUDGET_TOKENS
    )
    max_attempt_cost_usd = state.get("max_attempt_cost_usd")
    cost_budget_pricing_table_id = state.get("cost_budget_pricing_table_id")
    if max_attempt_cost_usd is not None:
        max_attempt_cost_usd = validate_cost_budget_usd(max_attempt_cost_usd)
    cost_budget_pricing_table_id = _require_cost_budget_pricing_binding(
        max_attempt_cost_usd, cost_budget_pricing_table_id
    )
    if not isinstance(previous_objective, str) or not previous_objective.strip():
        raise ValueError("Completed turn objective is missing")
    if not isinstance(turn_history, list) or not all(
        isinstance(record, TurnRecord) for record in turn_history
    ):
        raise TypeError("Graph turn_history does not contain TurnRecord values")
    if not isinstance(repo_path, Path):
        raise TypeError("Graph repo_path is not a Path")
    if not isinstance(human_review_policy, HumanReviewPolicy):
        raise TypeError("Graph human_review_policy is invalid")
    for name, value in (
        ("max_tool_calls_per_step", max_tool_calls),
        ("max_explorer_calls_per_step", max_explorer_calls),
        ("max_no_progress_rounds", max_no_progress_rounds),
        ("max_plan_revisions", max_plan_revisions),
        ("context_budget_tokens", context_budget_tokens),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Graph {name} must be a positive integer")

    archived_turn = TurnRecord(
        turn_number=turn_number,
        objective=previous_objective,
        final_result=final_result,
        termination_status=cast(TerminationStatus, termination_status),
        review_history=validated_review_history(state.get("review_history")),
    )
    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path,
        "objective": objective,
        "turn_number": turn_number + 1,
        "turn_history": [*turn_history, archived_turn],
        "review_history": HumanReviewHistory(),
        "cancelled_run_id": None,
        "cancellation_request_id": None,
        "human_review_policy": human_review_policy,
        "human_review_payload": None,
        "human_decision_action": None,
        "human_feedback": None,
        "investigation_plan": None,
        "memory_retrieval_enabled": memory_retrieval_enabled,
        "max_repository_memory_results": max_repository_memory_results,
        "max_episodic_memory_results": max_episodic_memory_results,
        "memory_recall_report": None,
        "memory_context_items": [],
        "tool_audit_records": [],
        "last_tool_batch_audits": [],
        "step_tool_audit_sequence_floor": 0,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": max_tool_calls,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": max_explorer_calls,
        "progress_snapshot": None,
        "no_progress_decision": None,
        "consecutive_no_progress_rounds": 0,
        "max_no_progress_rounds": max_no_progress_rounds,
        "plan_revision_count": 0,
        "max_plan_revisions": max_plan_revisions,
        "step_stop_reason": None,
        "termination_status": None,
        "candidate_result": None,
        "candidate_output_contract": None,
        "evaluation_report": None,
        "investigation_structure": None,
        "context_budget_tokens": context_budget_tokens,
        "context_build_count": 0,
        "active_context_pack": None,
        "max_attempt_cost_usd": max_attempt_cost_usd,
        "cost_budget_pricing_table_id": cost_budget_pricing_table_id,
        "cost_budget": None,
        "next_action": None,
        "final_result": None,
    }


def _validate_memory_result_limit(value: object, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError(f"{name} must be between 1 and 10")
