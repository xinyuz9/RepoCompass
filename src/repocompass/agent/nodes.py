"""Node factories for the RepoCompass agent graph."""

from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

from repocompass.agent.prompts import (
    FINALIZATION_INSTRUCTION,
    build_agent_messages,
    build_planner_messages,
)
from repocompass.agent.contracts import FinalResult, InvestigationPlan
from repocompass.agent.state import (
    AgentInvocationInput,
    AgentNodeInput,
    FinalizerInput,
    InvalidToolBatchInput,
    PlannerInput,
    ToolBudgetInput,
    ToolRejectionInput,
)
from repocompass.agent.tool_adapters import REPOSITORY_TOOLS


class AgentNodeUpdate(TypedDict):
    messages: list[AnyMessage]
    iteration_count: int


AgentNode = Callable[[AgentNodeInput], AgentNodeUpdate]
ToolCountUpdate = dict[str, int]


class PlannerNodeUpdate(TypedDict):
    investigation_plan: InvestigationPlan
    active_objective: str


PlannerNode = Callable[[PlannerInput], PlannerNodeUpdate]


class ToolRejectionUpdate(TypedDict):
    messages: list[ToolMessage]
    stop_reason: str


class InvalidToolBatchUpdate(TypedDict):
    messages: list[ToolMessage]
    tool_call_count: int


class FinalizeNodeUpdate(TypedDict):
    final_result: FinalResult


class IterationBudgetUpdate(TypedDict):
    stop_reason: str


FinalizeNode = Callable[[FinalizerInput], FinalizeNodeUpdate]


def create_agent_node(model: BaseChatModel) -> AgentNode:
    """Bind repository tools once and create the graph's agent node."""

    _bound_model = model.bind_tools(REPOSITORY_TOOLS)

    def agent_node(state: AgentNodeInput) -> AgentNodeUpdate:
        """Call the tool-bound model once and return its message as a state update."""

        messages_for_agent = build_agent_messages(state)
        response = _bound_model.invoke(messages_for_agent)
        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
        }

    return agent_node


def mark_iteration_budget_exhausted(
    _state: AgentInvocationInput,
) -> IterationBudgetUpdate:
    """Record that another Agent model invocation was not allowed."""

    return {"stop_reason": "iteration_budget_exhausted"}


def reserve_tool_budget(state: ToolBudgetInput) -> ToolCountUpdate:
    """Reserve budget for every tool call in the latest AI message."""

    batch_tool_call_count = len(state["messages"][-1].tool_calls)
    return {"tool_call_count": batch_tool_call_count + state["tool_call_count"]}


def reject_pending_tool_calls(
    state: ToolRejectionInput,
) -> ToolRejectionUpdate:
    """Return error messages for tool calls rejected by the execution budget."""

    last_message = state["messages"][-1]

    messages: list[ToolMessage] = []
    for tool_call in last_message.tool_calls + last_message.invalid_tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because the tool-call budget was "
                    "exhausted. Answer using the available evidence."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status="error",
            )
        )

    return {
        "messages": messages,
        "stop_reason": "tool_call_budget_exhausted",
    }


def handle_invalid_tool_batch(
    state: InvalidToolBatchInput,
) -> InvalidToolBatchUpdate:
    """Return model-visible errors for a batch containing invalid tool calls."""

    last_message = state["messages"][-1]
    messages: list[ToolMessage] = []
    batch_tool_call_count = len(last_message.invalid_tool_calls) + len(
        last_message.tool_calls
    )

    for invalid_tool_call in last_message.invalid_tool_calls:
        error_message = (
            invalid_tool_call["error"] or "Tool arguments could not be parsed."
        )
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because its arguments could not be "
                    f"parsed. Parser error: {error_message} Submit a corrected tool "
                    "call with valid JSON arguments."
                ),
                tool_call_id=invalid_tool_call["id"],
                name=invalid_tool_call["name"],
                status="error",
            )
        )

    for tool_call in last_message.tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    "Tool call was not executed because another call in the same "
                    "batch had invalid arguments. Resubmit the complete batch with "
                    "corrected arguments."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status="error",
            )
        )

    return {
        "messages": messages,
        "tool_call_count": batch_tool_call_count + state["tool_call_count"],
    }


def create_finalize_node(model: BaseChatModel) -> FinalizeNode:
    """Create a node that turns collected repository evidence into a final report."""

    _structured_model = model.with_structured_output(
        FinalResult,
        method="function_calling",
    )

    def finalize_node(state: FinalizerInput) -> FinalizeNodeUpdate:
        """Generate a structured final-result update from the message history."""

        messages_for_finalizer = [
            *state["messages"],
            HumanMessage(content=FINALIZATION_INSTRUCTION),
        ]

        response = _structured_model.invoke(messages_for_finalizer)

        if not isinstance(response, FinalResult):
            raise TypeError("Finalizer model did not return FinalResult")

        return {
            "final_result": response,
        }

    return finalize_node


def create_plan_investigation_node(
    model: BaseChatModel,
) -> PlannerNode:
    """Create a node that generates the initial structured investigation plan.

    The model produces an :class:`InvestigationPlan`. The returned node converts
    that domain object into a partial graph update and derives
    ``active_objective`` from the plan's validated active step.
    """

    _structured_model = model.with_structured_output(
        InvestigationPlan,
        method="function_calling",
    )

    def planner_node(state: PlannerInput) -> PlannerNodeUpdate:
        """Generate the initial plan and activate its designated step."""

        messages_for_planner = build_planner_messages(state)

        response = _structured_model.invoke(messages_for_planner)

        if not isinstance(response, InvestigationPlan):
            raise TypeError("Planner model did not return InvestigationPlan")

        active_step = next(
            step for step in response.steps if step.id == response.active_step_id
        )

        return {
            "investigation_plan": response,
            "active_objective": active_step.objective,
        }

    return planner_node
