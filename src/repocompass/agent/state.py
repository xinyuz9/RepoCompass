from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult


class RepoCompassState(MessagesState):
    """Canonical internal state for one repository investigation."""

    repo_path: Path
    objective: str
    active_objective: str
    investigation_plan: InvestigationPlan | None

    step_tool_call_count: int
    max_tool_calls_per_step: int
    step_explorer_call_count: int
    max_explorer_calls_per_step: int

    stop_reason: str | None
    candidate_result: CandidateResult | None
    evaluation_report: EvaluationReport | None
    final_result: FinalResult | None


def create_initial_state(
    repo_path: Path,
    objective: str,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
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

    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path.resolve(),
        "objective": objective,
        "active_objective": objective,
        "investigation_plan": None,
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": max_tool_calls_per_step,
        "step_explorer_call_count": 0,
        "max_explorer_calls_per_step": max_explorer_calls_per_step,
        "stop_reason": None,
        "candidate_result": None,
        "evaluation_report": None,
        "final_result": None,
    }
