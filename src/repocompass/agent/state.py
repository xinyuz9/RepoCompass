from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import MessagesState

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import HumanReviewPolicy, TurnRecord
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import ControllerAction, TerminationStatus


class RepoCompassState(MessagesState):
    """Canonical internal state for one repository investigation."""

    repo_path: Path
    objective: str
    turn_number: int
    turn_history: list[TurnRecord]
    human_review_policy: HumanReviewPolicy
    investigation_plan: InvestigationPlan | None

    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    plan_revision_count: int
    max_plan_revisions: int

    step_stop_reason: str | None
    termination_status: TerminationStatus | None
    candidate_result: CandidateResult | None
    evaluation_report: EvaluationReport | None
    next_action: ControllerAction | None
    final_result: FinalResult | None


class ContinuationStateUpdate(TypedDict):
    """State update that starts a new turn on one completed thread."""

    messages: list[AnyMessage]
    repo_path: Path
    objective: str
    turn_number: int
    turn_history: list[TurnRecord]
    human_review_policy: HumanReviewPolicy
    investigation_plan: None
    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    plan_revision_count: int
    max_plan_revisions: int
    step_stop_reason: None
    termination_status: None
    candidate_result: None
    evaluation_report: None
    next_action: None
    final_result: None


def create_initial_state(
    repo_path: Path,
    objective: str,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
    max_plan_revisions: int = 2,
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.NEVER,
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
    if max_plan_revisions < 1:
        raise ValueError(
            f"Max plan revisions must be positive, not {max_plan_revisions}"
        )

    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path.resolve(),
        "objective": objective,
        "turn_number": 1,
        "turn_history": [],
        "human_review_policy": human_review_policy,
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": max_tool_calls_per_step,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": max_explorer_calls_per_step,
        "plan_revision_count": 0,
        "max_plan_revisions": max_plan_revisions,
        "step_stop_reason": None,
        "termination_status": None,
        "candidate_result": None,
        "evaluation_report": None,
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
    ):
        raise ValueError("Continuation requires a completed turn")

    turn_number = state.get("turn_number")
    previous_objective = state.get("objective")
    turn_history = state.get("turn_history")
    repo_path = state.get("repo_path")
    human_review_policy = state.get("human_review_policy")
    max_tool_calls = state.get("max_tool_calls_per_step")
    max_explorer_calls = state.get("max_explorer_calls_per_step")
    max_plan_revisions = state.get("max_plan_revisions")
    if not isinstance(turn_number, int) or isinstance(turn_number, bool):
        raise TypeError("Graph turn_number is not an integer")
    if turn_number < 1:
        raise ValueError("Graph turn_number must be positive")
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
        ("max_plan_revisions", max_plan_revisions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Graph {name} must be a positive integer")

    archived_turn = TurnRecord(
        turn_number=turn_number,
        objective=previous_objective,
        final_result=final_result,
        termination_status=cast(TerminationStatus, termination_status),
    )
    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path,
        "objective": objective,
        "turn_number": turn_number + 1,
        "turn_history": [*turn_history, archived_turn],
        "human_review_policy": human_review_policy,
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": max_tool_calls,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": max_explorer_calls,
        "plan_revision_count": 0,
        "max_plan_revisions": max_plan_revisions,
        "step_stop_reason": None,
        "termination_status": None,
        "candidate_result": None,
        "evaluation_report": None,
        "next_action": None,
        "final_result": None,
    }
