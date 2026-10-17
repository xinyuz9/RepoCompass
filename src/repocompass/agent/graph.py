from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig, RunnableLambda

from repocompass.agent.architecture import (
    DEFAULT_AGENT_ARCHITECTURE,
    AgentArchitectureProfile,
)
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
    seed_tool_wrapper_plan,
    validate_quick_candidate,
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
from repocompass.agent.tool_adapters import (
    LEXICAL_REPOSITORY_TOOLS,
    REPOSITORY_TOOLS,
    create_browser_tool,
)
from repocompass.models import WorkflowModels
from repocompass.models.control import check_attempt_deadline
from repocompass.tools.gateway import ToolGateway, create_repository_tool_registry
from repocompass.tools.browser import BrowserPolicy
from repocompass.tools.contracts import ToolCapability

if TYPE_CHECKING:
    from repocompass.memory.store import MemoryStore


def _route_after_initial_plan(state: RepoCompassState) -> str:
    """Recall only when enabled; disabled runs preserve the legacy graph path."""

    route = route_explorer_invocation(state)
    if route == "agent" and state.get("memory_retrieval_enabled", False):
        return "recall_memory"
    return route


def _route_after_revision(state: RepoCompassState) -> str:
    if state.get("step_stop_reason") == "invalid_plan_revision":
        return "rejected_revision"
    return route_after_model_invocation(state)


def create_agent_graph(
    models: WorkflowModels,
    *,
    architecture_profile: AgentArchitectureProfile = DEFAULT_AGENT_ARCHITECTURE,
    checkpointer: BaseCheckpointSaver | None = None,
    raw_run_recorder: RawRunRecorder | None = None,
    memory_store: MemoryStore | None = None,
    memory_clock: Callable[[], int] = lambda: 0,
    cancellation_check: Callable[[RunnableConfig], None] | None = None,
    browser_policy: BrowserPolicy | None = None,
):
    """Compile the graph with optional checkpoint and Raw Run persistence."""

    if not isinstance(architecture_profile, AgentArchitectureProfile):
        raise TypeError("architecture_profile must be an AgentArchitectureProfile")
    repository_tools = (
        REPOSITORY_TOOLS
        if architecture_profile.structural_tools_enabled
        else LEXICAL_REPOSITORY_TOOLS
    )
    if browser_policy is not None:
        if not isinstance(browser_policy, BrowserPolicy):
            raise TypeError("browser_policy must be a BrowserPolicy or None")
        repository_tools = (*repository_tools, create_browser_tool(browser_policy))
    allowed_capabilities = {ToolCapability.FILESYSTEM_READ}
    if browser_policy is not None:
        allowed_capabilities.update({ToolCapability.NETWORK, ToolCapability.PROCESS})
    gateway = ToolGateway(
        create_repository_tool_registry(
            include_structural=architecture_profile.structural_tools_enabled,
            browser_policy=browser_policy,
        ),
        allowed_capabilities=frozenset(allowed_capabilities),
    )
    context_strategy = architecture_profile.context_selection_strategy
    builder = StateGraph(RepoCompassState)

    def add_node(name, action):
        def before_node(state, config: RunnableConfig):
            check_attempt_deadline()
            if cancellation_check is not None:
                cancellation_check(config)
            return state

        action = RunnableLambda(before_node) | action
        builder.add_node(name, action)

    add_node(
        "plan_investigation",
        (
            create_plan_investigation_node(
                models.planner,
                raw_run_recorder,
                context_strategy,
            )
            if architecture_profile.planner_enabled
            else seed_tool_wrapper_plan
        ),
    )
    add_node(
        "recall_memory",
        create_recall_memory_node(memory_store, memory_clock),
    )
    add_node(
        "agent",
        create_agent_node(
            models.explorer,
            raw_run_recorder,
            repository_tools,
            context_strategy,
        ),
    )
    add_node("tools", create_execute_tools_node(gateway))
    add_node("assess_progress", assess_tool_progress)
    add_node(
        "step_explorer_budget_exhausted",
        mark_step_explorer_budget_exhausted,
    )
    add_node(
        "build_candidate",
        create_build_candidate_node(
            models.explorer,
            raw_run_recorder,
            context_strategy,
        ),
    )
    add_node(
        "evaluate_candidate",
        create_evaluate_candidate_node(
            models.evaluator,
            raw_run_recorder,
            context_strategy,
        ),
    )
    add_node("decide_next_action", decide_next_action)
    add_node("advance_plan_step", advance_plan_step)
    add_node(
        "revise_plan",
        create_revise_plan_node(
            models.planner,
            raw_run_recorder,
            context_strategy,
        ),
    )
    add_node("mark_investigation_accepted", mark_investigation_accepted)
    add_node("mark_investigation_cancelled", mark_investigation_cancelled)
    add_node("mark_investigation_exhausted", mark_investigation_exhausted)
    add_node(
        "finalize_cost_budget_exhausted",
        finalize_cost_budget_exhausted,
    )
    add_node("prepare_human_review", prepare_human_review)
    add_node("review_final_candidate", review_final_candidate)
    add_node("finalize", finalize_candidate)
    add_node("validate_quick_candidate", validate_quick_candidate)

    explorer_invocation_routes = {
        "agent": "agent",
        "step_explorer_budget_exhausted": "step_explorer_budget_exhausted",
        "cost_budget_exhausted": "finalize_cost_budget_exhausted",
    }
    post_model_routes = {
        "continue": "evaluate_candidate",
        "cost_budget_exhausted": "finalize_cost_budget_exhausted",
    }
    post_candidate_routes = (
        post_model_routes
        if architecture_profile.evaluator_enabled
        else {
            "continue": "validate_quick_candidate",
            "cost_budget_exhausted": "finalize_cost_budget_exhausted",
        }
    )
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
        post_candidate_routes,
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
        _route_after_revision,
        {
            "continue": "agent",
            "cost_budget_exhausted": "finalize_cost_budget_exhausted",
            "rejected_revision": END,
        },
    )
    builder.add_edge("prepare_human_review", "review_final_candidate")
    builder.add_conditional_edges(
        "validate_quick_candidate",
        lambda state: "invalid" if state.get("final_result") is not None else "valid",
        {"invalid": END, "valid": "prepare_human_review"},
    )
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
