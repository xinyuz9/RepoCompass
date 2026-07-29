from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import ControllerAction, TerminationStatus


class RepoCompassState(MessagesState):
    """Canonical internal state for one repository investigation."""

    repo_path: Path
    objective: str
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


def create_initial_state(
    repo_path: Path,
    objective: str,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
    max_plan_revisions: int = 2,
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
