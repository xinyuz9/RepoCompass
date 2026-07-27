"""Pure conditional routing decisions for the RepoCompass graph."""

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState


class ExplorerInvocationInput(TypedDict):
    step_explorer_call_count: int
    max_explorer_calls_per_step: int


ExplorerInvocationRoute = Literal["agent", "step_explorer_budget_exhausted"]


def route_explorer_invocation(
    state: ExplorerInvocationInput,
) -> ExplorerInvocationRoute:
    """Allow another Explorer call only while the active step has budget."""

    if state["step_explorer_call_count"] < state["max_explorer_calls_per_step"]:
        return "agent"

    return "step_explorer_budget_exhausted"


class ToolRoutingInput(MessagesState):
    step_tool_call_count: int
    max_tool_calls_per_step: int


ToolCallRoute = Literal[
    "authorize_tools",
    "build_candidate",
    "reject_tools",
    "handle_invalid_tools",
]


def route_tool_calls(state: ToolRoutingInput) -> ToolCallRoute:
    """Choose the next tool-call path without modifying graph state."""

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

    if (
        pending_tool_call_count + state["step_tool_call_count"]
        > state["max_tool_calls_per_step"]
    ):
        return "reject_tools"
    if last_message.invalid_tool_calls:
        return "handle_invalid_tools"

    return "authorize_tools"
