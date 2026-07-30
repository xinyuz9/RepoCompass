from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from repocompass.agent.nodes import (
    advance_plan_step,
    create_agent_node,
    create_build_candidate_node,
    create_evaluate_candidate_node,
    create_plan_investigation_node,
    create_revise_plan_node,
    decide_next_action,
    finalize_candidate,
    handle_invalid_tool_batch,
    mark_investigation_accepted,
    mark_investigation_exhausted,
    mark_step_explorer_budget_exhausted,
    reject_pending_tool_calls,
    reserve_tool_budget,
)
from repocompass.agent.routing import (
    route_controller,
    route_explorer_invocation,
    route_tool_calls,
)
from repocompass.agent.state import RepoCompassState
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS
from repocompass.models import WorkflowModels


def create_agent_graph(
    models: WorkflowModels,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the current graph with an optional persistence dependency."""

    builder = StateGraph(RepoCompassState)

    builder.add_node(
        "plan_investigation",
        create_plan_investigation_node(models.planner),
    )
    builder.add_node("agent", create_agent_node(models.explorer))
    builder.add_node("reject_tools", reject_pending_tool_calls)
    builder.add_node("handle_invalid_tools", handle_invalid_tool_batch)
    builder.add_node("authorize_tools", reserve_tool_budget)
    builder.add_node("tools", ToolNode(REPOSITORY_TOOLS))
    builder.add_node(
        "step_explorer_budget_exhausted",
        mark_step_explorer_budget_exhausted,
    )
    builder.add_node(
        "build_candidate",
        create_build_candidate_node(models.explorer),
    )
    builder.add_node(
        "evaluate_candidate",
        create_evaluate_candidate_node(models.evaluator),
    )
    builder.add_node("decide_next_action", decide_next_action)
    builder.add_node("advance_plan_step", advance_plan_step)
    builder.add_node("revise_plan", create_revise_plan_node(models.planner))
    builder.add_node("mark_investigation_accepted", mark_investigation_accepted)
    builder.add_node("mark_investigation_exhausted", mark_investigation_exhausted)
    builder.add_node("finalize", finalize_candidate)

    explorer_invocation_routes = {
        "agent": "agent",
        "step_explorer_budget_exhausted": "step_explorer_budget_exhausted",
    }
    builder.add_edge(START, "plan_investigation")
    builder.add_conditional_edges(
        "plan_investigation",
        route_explorer_invocation,
        explorer_invocation_routes,
    )
    builder.add_conditional_edges(
        "agent",
        route_tool_calls,
        {
            "authorize_tools": "authorize_tools",
            "build_candidate": "build_candidate",
            "reject_tools": "reject_tools",
            "handle_invalid_tools": "handle_invalid_tools",
        },
    )
    builder.add_edge("authorize_tools", "tools")
    builder.add_conditional_edges(
        "tools",
        route_explorer_invocation,
        explorer_invocation_routes,
    )
    builder.add_edge("reject_tools", "build_candidate")
    builder.add_conditional_edges(
        "handle_invalid_tools",
        route_explorer_invocation,
        explorer_invocation_routes,
    )
    builder.add_edge("step_explorer_budget_exhausted", "build_candidate")
    builder.add_edge("build_candidate", "evaluate_candidate")
    builder.add_edge("evaluate_candidate", "decide_next_action")
    builder.add_conditional_edges(
        "decide_next_action",
        route_controller,
        {
            "accepted": "mark_investigation_accepted",
            "continue_step": "agent",
            "advance_step": "advance_plan_step",
            "revise_plan": "revise_plan",
            "exhausted": "mark_investigation_exhausted",
        },
    )
    builder.add_edge("advance_plan_step", "agent")
    builder.add_edge("revise_plan", "agent")
    builder.add_edge("mark_investigation_accepted", "finalize")
    builder.add_edge("mark_investigation_exhausted", "finalize")
    builder.add_edge("finalize", END)

    compiled = builder.compile(
        checkpointer=checkpointer,
        name="repocompass_m1",
    )

    return compiled
