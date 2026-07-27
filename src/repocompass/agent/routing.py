"""Conditional routing decisions for repository tool calls."""

from typing import Literal

from langchain_core.messages import AIMessage

from repocompass.agent.state import AgentInvocationInput, ToolRoutingInput

ToolCallRoute = Literal[
    "authorize_tools",
    "finalize",
    "reject_tools",
    "handle_invalid_tools",
]
AgentInvocationRoute = Literal["agent", "iteration_budget_exhausted"]


def route_agent_invocation(state: AgentInvocationInput) -> AgentInvocationRoute:
    """Allow Agent reasoning only while an iteration remains."""

    if state["iteration_count"] < state["max_iterations"]:
        return "agent"

    return "iteration_budget_exhausted"


def route_tool_calls(state: ToolRoutingInput) -> ToolCallRoute:
    """Choose the next tool-call path without modifying agent state."""

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
        return "finalize"

    for tool_call in pending_tool_calls:
        if tool_call["id"] is None:
            raise ValueError("Tool call id is None")

    if pending_tool_call_count + state["tool_call_count"] > state["max_tool_calls"]:
        return "reject_tools"

    if last_message.invalid_tool_calls:
        return "handle_invalid_tools"

    return "authorize_tools"
