from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel

from repocompass.agent.nodes import (
    create_agent_node,
    create_finalize_node,
    handle_invalid_tool_batch,
    mark_iteration_budget_exhausted,
    reject_pending_tool_calls,
    reserve_tool_budget,
)
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS
from repocompass.agent.routing import route_agent_invocation, route_tool_calls
from repocompass.agent.state import AgentState


def create_agent_graph(model: BaseChatModel):
    """Compile the M0 repository tool-calling graph for the provided model."""

    builder = StateGraph(AgentState)

    builder.add_node("agent", create_agent_node(model))
    builder.add_node("reject_tools", reject_pending_tool_calls)
    builder.add_node("handle_invalid_tools", handle_invalid_tool_batch)
    builder.add_node("authorize_tools", reserve_tool_budget)
    builder.add_node("tools", ToolNode(REPOSITORY_TOOLS))
    builder.add_node(
        "iteration_budget_exhausted",
        mark_iteration_budget_exhausted,
    )
    builder.add_node("finalize", create_finalize_node(model))

    agent_invocation_routes = {
        "agent": "agent",
        "iteration_budget_exhausted": "iteration_budget_exhausted",
    }
    builder.add_conditional_edges(
        START,
        route_agent_invocation,
        agent_invocation_routes,
    )
    builder.add_conditional_edges(
        "agent",
        route_tool_calls,
        {
            "authorize_tools": "authorize_tools",
            "finalize": "finalize",
            "reject_tools": "reject_tools",
            "handle_invalid_tools": "handle_invalid_tools",
        },
    )
    builder.add_edge("authorize_tools", "tools")
    builder.add_conditional_edges(
        "tools",
        route_agent_invocation,
        agent_invocation_routes,
    )
    builder.add_edge("reject_tools", "finalize")
    builder.add_conditional_edges(
        "handle_invalid_tools",
        route_agent_invocation,
        agent_invocation_routes,
    )
    builder.add_edge("iteration_budget_exhausted", "finalize")
    builder.add_edge("finalize", END)

    compiled = builder.compile(name="repocompass_m0")

    return compiled
