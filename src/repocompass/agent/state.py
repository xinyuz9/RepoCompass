from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from repocompass.agent.contracts import FinalResult, InvestigationPlan

DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_MAX_ITERATIONS = 4


class RepoCompassState(MessagesState):
    """Canonical internal state for one repository investigation."""

    repo_path: Path
    objective: str
    active_objective: str
    investigation_plan: InvestigationPlan | None

    tool_call_count: int
    max_tool_calls: int
    iteration_count: int
    max_iterations: int

    stop_reason: str | None
    final_result: FinalResult | None

def create_initial_state(
    repo_path: Path,
    objective: str,
    max_tool_calls: int,
    max_iterations: int,
) -> RepoCompassState:
    objective = objective.strip()
    if not objective:
        raise ValueError("Objective must not be empty")
    if max_tool_calls < 1:
        raise ValueError(f"Max tool calls must be positive, not {max_tool_calls}")
    if max_iterations < 1:
        raise ValueError(f"Max iterations must be positive, not {max_iterations}")

    return {
        "messages": [HumanMessage(content=objective)],
        "repo_path": repo_path.resolve(),
        "objective": objective,
        "active_objective": objective,
        "investigation_plan": None,
        "tool_call_count": 0,
        "max_tool_calls": max_tool_calls,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "stop_reason": None,
        "final_result": None,
    }


class ToolRoutingInput(MessagesState):
    tool_call_count: int
    max_tool_calls: int


class AgentNodeInput(MessagesState):
    objective: str
    active_objective: str
    investigation_plan: InvestigationPlan | None
    tool_call_count: int
    max_tool_calls: int
    iteration_count: int
    max_iterations: int


class PlannerInput(TypedDict):
    """Minimal initial-planning view without repository content or messages."""

    objective: str
    max_tool_calls: int
    max_iterations: int


class AgentInvocationInput(TypedDict):
    iteration_count: int
    max_iterations: int


class ToolBudgetInput(MessagesState):
    tool_call_count: int


class ToolRejectionInput(MessagesState):
    pass


class InvalidToolBatchInput(MessagesState):
    tool_call_count: int


class FinalizerInput(MessagesState):
    pass
