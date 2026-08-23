from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from repocompass.agent.nodes import (
    advance_plan_step,
    create_agent_node,
    create_build_candidate_node,
    create_evaluate_candidate_node,
    create_execute_tools_node,
    create_plan_investigation_node,
    create_revise_plan_node,
    decide_next_action,
    finalize_candidate,
    mark_investigation_accepted,
    mark_investigation_cancelled,
    mark_investigation_exhausted,
    mark_step_explorer_budget_exhausted,
    prepare_human_review,
    review_final_candidate,
)
from repocompass.agent.progress import assess_tool_progress
from repocompass.agent.routing import (
    route_after_tool_execution,
    route_controller,
    route_explorer_invocation,
    route_human_decision,
    route_tool_calls,
)
from repocompass.agent.state import RepoCompassState
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
    builder.add_node("tools", create_execute_tools_node())
    builder.add_node("assess_progress", assess_tool_progress)
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
    builder.add_node("mark_investigation_cancelled", mark_investigation_cancelled)
    builder.add_node("mark_investigation_exhausted", mark_investigation_exhausted)
    builder.add_node("prepare_human_review", prepare_human_review)
    builder.add_node("review_final_candidate", review_final_candidate)
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
            "execute_tools": "tools",
            "build_candidate": "build_candidate",
        },
    )
    builder.add_edge("tools", "assess_progress")
    builder.add_conditional_edges(
        "assess_progress",
        route_after_tool_execution,
        {
            **explorer_invocation_routes,
            "build_candidate": "build_candidate",
        },
    )
    builder.add_edge("step_explorer_budget_exhausted", "build_candidate")
    builder.add_edge("build_candidate", "evaluate_candidate")
    builder.add_edge("evaluate_candidate", "decide_next_action")
    builder.add_conditional_edges(
        "decide_next_action",
        route_controller,
        {
            "accepted": "prepare_human_review",
            "continue_step": "agent",
            "advance_step": "advance_plan_step",
            "revise_plan": "revise_plan",
            "exhausted": "mark_investigation_exhausted",
        },
    )
    builder.add_edge("advance_plan_step", "agent")
    builder.add_edge("revise_plan", "agent")
    builder.add_edge("prepare_human_review", "review_final_candidate")
    builder.add_conditional_edges(
        "review_final_candidate",
        route_human_decision,
        {
            "approve": "mark_investigation_accepted",
            "revise": "revise_plan",
            "cancel": "mark_investigation_cancelled",
        },
    )
    builder.add_edge("mark_investigation_accepted", "finalize")
    builder.add_edge("mark_investigation_cancelled", "finalize")
    builder.add_edge("mark_investigation_exhausted", "finalize")
    builder.add_edge("finalize", END)

    compiled = builder.compile(
        checkpointer=checkpointer,
        name="repocompass_m1",
    )

    return compiled
