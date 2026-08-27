from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from repocompass.agent.invocation import RawRunRecorder
from repocompass.agent.nodes import (
    advance_plan_step,
    create_agent_node,
    create_build_candidate_node,
    create_evaluate_candidate_node,
    create_execute_tools_node,
    create_plan_investigation_node,
    create_recall_memory_node,
    create_revise_plan_node,
    decide_next_action,
    finalize_candidate,
    finalize_cost_budget_exhausted,
    mark_investigation_accepted,
    mark_investigation_cancelled,
    mark_investigation_exhausted,
    mark_step_explorer_budget_exhausted,
    prepare_human_review,
    review_final_candidate,
)
from repocompass.agent.progress import assess_tool_progress
from repocompass.agent.routing import (
    route_after_model_invocation,
    route_after_tool_execution,
    route_controller,
    route_explorer_invocation,
    route_human_decision,
    route_tool_calls,
)
from repocompass.agent.state import RepoCompassState
from repocompass.models import WorkflowModels

if TYPE_CHECKING:
    from repocompass.memory.store import MemoryStore


def _route_after_initial_plan(state: RepoCompassState) -> str:
    """Recall only when enabled; disabled runs preserve the legacy graph path."""

    route = route_explorer_invocation(state)
    if route == "agent" and state.get("memory_retrieval_enabled", False):
        return "recall_memory"
    return route


def create_agent_graph(
    models: WorkflowModels,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    raw_run_recorder: RawRunRecorder | None = None,
    memory_store: MemoryStore | None = None,
    memory_clock: Callable[[], int] = lambda: 0,
):
    """Compile the graph with optional checkpoint and Raw Run persistence."""

    builder = StateGraph(RepoCompassState)

    builder.add_node(
        "plan_investigation",
        create_plan_investigation_node(models.planner, raw_run_recorder),
    )
    builder.add_node(
        "recall_memory",
        create_recall_memory_node(memory_store, memory_clock),
    )
    builder.add_node("agent", create_agent_node(models.explorer, raw_run_recorder))
    builder.add_node("tools", create_execute_tools_node())
    builder.add_node("assess_progress", assess_tool_progress)
    builder.add_node(
        "step_explorer_budget_exhausted",
        mark_step_explorer_budget_exhausted,
    )
    builder.add_node(
        "build_candidate",
        create_build_candidate_node(models.explorer, raw_run_recorder),
    )
    builder.add_node(
        "evaluate_candidate",
        create_evaluate_candidate_node(models.evaluator, raw_run_recorder),
    )
    builder.add_node("decide_next_action", decide_next_action)
    builder.add_node("advance_plan_step", advance_plan_step)
    builder.add_node(
        "revise_plan",
        create_revise_plan_node(models.planner, raw_run_recorder),
    )
    builder.add_node("mark_investigation_accepted", mark_investigation_accepted)
    builder.add_node("mark_investigation_cancelled", mark_investigation_cancelled)
    builder.add_node("mark_investigation_exhausted", mark_investigation_exhausted)
    builder.add_node(
        "finalize_cost_budget_exhausted",
        finalize_cost_budget_exhausted,
    )
    builder.add_node("prepare_human_review", prepare_human_review)
    builder.add_node("review_final_candidate", review_final_candidate)
    builder.add_node("finalize", finalize_candidate)

    explorer_invocation_routes = {
        "agent": "agent",
        "step_explorer_budget_exhausted": "step_explorer_budget_exhausted",
        "cost_budget_exhausted": "finalize_cost_budget_exhausted",
    }
    post_model_routes = {
        "continue": "evaluate_candidate",
        "cost_budget_exhausted": "finalize_cost_budget_exhausted",
    }
    builder.add_edge(START, "plan_investigation")
    builder.add_conditional_edges(
        "plan_investigation",
        _route_after_initial_plan,
        {**explorer_invocation_routes, "recall_memory": "recall_memory"},
    )
    builder.add_edge("recall_memory", "agent")
    builder.add_conditional_edges(
        "agent",
        route_tool_calls,
        {
            "execute_tools": "tools",
            "build_candidate": "build_candidate",
            "cost_budget_exhausted": "finalize_cost_budget_exhausted",
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
    builder.add_conditional_edges(
        "build_candidate",
        route_after_model_invocation,
        post_model_routes,
    )
    builder.add_conditional_edges(
        "evaluate_candidate",
        route_after_model_invocation,
        {
            "continue": "decide_next_action",
            "cost_budget_exhausted": "finalize_cost_budget_exhausted",
        },
    )
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
    builder.add_conditional_edges(
        "revise_plan",
        route_after_model_invocation,
        {
            "continue": "agent",
            "cost_budget_exhausted": "finalize_cost_budget_exhausted",
        },
    )
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
    builder.add_edge("finalize_cost_budget_exhausted", END)

    compiled = builder.compile(
        checkpointer=checkpointer,
        name="repocompass_m1",
    )

    return compiled
