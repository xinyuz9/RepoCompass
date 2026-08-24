"""Pure conditional routing decisions for the RepoCompass graph."""

from collections.abc import Mapping
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

from repocompass.agent.cost import AttemptCostBudgetSummary

ATTEMPT_COST_BUDGET_STOP_REASON = "attempt_cost_budget_exhausted"
ControllerAction = Literal[
    "accepted",
    "continue_step",
    "advance_step",
    "revise_plan",
    "exhausted",
]
HumanDecisionActionValue = Literal["approve", "revise", "cancel"]
TerminationStatus = Literal["accepted", "exhausted", "cancelled"]


class ControllerRoutingInput(TypedDict):
    next_action: ControllerAction | None


def route_controller(state: ControllerRoutingInput) -> ControllerAction:
    """Route using the decision already recorded by the Controller node."""

    next_action = state["next_action"]
    if next_action is None:
        raise ValueError("Controller did not select a next action")

    return next_action


class HumanDecisionRoutingInput(TypedDict):
    human_decision_action: HumanDecisionActionValue | None


def route_human_decision(
    state: HumanDecisionRoutingInput,
) -> HumanDecisionActionValue:
    """Route only the validated action returned by the review node."""

    action = state["human_decision_action"]
    if action is None:
        raise ValueError("Human decision action is missing")

    return action


class ExplorerInvocationInput(TypedDict):
    step_explorer_call_count: int
    max_explorer_calls_per_step: int
    step_stop_reason: str | None
    cost_budget: AttemptCostBudgetSummary | None


ExplorerInvocationRoute = Literal[
    "agent",
    "step_explorer_budget_exhausted",
    "cost_budget_exhausted",
]


def route_explorer_invocation(
    state: ExplorerInvocationInput,
) -> ExplorerInvocationRoute:
    """Allow another Explorer call only while the active step has budget."""

    if route_after_model_invocation(state) == "cost_budget_exhausted":
        return "cost_budget_exhausted"
    if state["step_explorer_call_count"] < state["max_explorer_calls_per_step"]:
        return "agent"

    return "step_explorer_budget_exhausted"


class ToolRoutingInput(MessagesState):
    pass
    step_stop_reason: str | None
    cost_budget: AttemptCostBudgetSummary | None


ToolCallRoute = Literal[
    "execute_tools",
    "build_candidate",
    "cost_budget_exhausted",
]


def route_tool_calls(state: ToolRoutingInput) -> ToolCallRoute:
    """Choose the next tool-call path without modifying graph state."""

    if route_after_model_invocation(state) == "cost_budget_exhausted":
        return "cost_budget_exhausted"
    if not state["messages"]:
        raise ValueError("Message list is empty")

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(
            f"Last message is not an AIMessage, {type(last_message).__name__}"
        )

    pending_tool_calls = last_message.invalid_tool_calls + last_message.tool_calls
    pending_tool_call_count = len(pending_tool_calls)
    if pending_tool_call_count == 0:
        return "build_candidate"

    for tool_call in pending_tool_calls:
        if tool_call["id"] is None:
            raise ValueError("Tool call id is None")

    return "execute_tools"


ModelInvocationRoute = Literal["continue", "cost_budget_exhausted"]


def route_after_model_invocation(
    state: Mapping[str, object],
) -> ModelInvocationRoute:
    """Close only when a model node recorded a valid budget denial."""

    stop_reason = state.get("step_stop_reason")
    budget = state.get("cost_budget")
    if budget is not None and not isinstance(budget, AttemptCostBudgetSummary):
        raise TypeError("Graph cost_budget is invalid")
    if stop_reason == ATTEMPT_COST_BUDGET_STOP_REASON:
        if budget is None or budget.next_invocation_allowed:
            raise ValueError("cost budget stop requires a denied budget decision")
        return "cost_budget_exhausted"
    if budget is not None and not budget.next_invocation_allowed:
        raise ValueError("denied cost budget is missing its stable stop reason")
    return "continue"


class PostToolExecutionInput(TypedDict):
    step_stop_reason: str | None
    step_explorer_call_count: int
    max_explorer_calls_per_step: int


PostToolExecutionRoute = Literal[
    "agent",
    "build_candidate",
    "step_explorer_budget_exhausted",
]


def route_after_tool_execution(
    state: PostToolExecutionInput,
) -> PostToolExecutionRoute:
    """Stop on a deterministic step reason, otherwise enforce Explorer budget."""

    if state["step_stop_reason"] in {
        "step_tool_call_budget_exhausted",
        "step_no_progress",
    }:
        return "build_candidate"
    if state["step_explorer_call_count"] < state["max_explorer_calls_per_step"]:
        return "agent"
    return "step_explorer_budget_exhausted"
