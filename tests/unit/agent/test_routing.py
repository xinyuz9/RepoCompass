"""Unit tests for the repository tool-call routing decision table."""

from typing import get_type_hints

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
)

from repocompass.agent.routing import (
    ExplorerInvocationInput,
    ExplorerInvocationRoute,
    ToolRoutingInput,
    ToolCallRoute,
    route_explorer_invocation,
    route_tool_calls,
)


def test_route_tool_calls_declares_narrow_input_schema() -> None:
    assert get_type_hints(route_tool_calls)["state"] is ToolRoutingInput


def test_route_explorer_invocation_declares_narrow_input_schema() -> None:
    assert get_type_hints(route_explorer_invocation)["state"] is ExplorerInvocationInput


def _state_for_message(
    message: AIMessage,
    *,
    used_tool_calls: int,
    max_tool_calls_per_step: int,
) -> ToolRoutingInput:
    """Build only the state fields required for tool-call routing."""

    return {
        "messages": [message],
        "step_tool_call_count": used_tool_calls,
        "max_tool_calls_per_step": max_tool_calls_per_step,
    }


@pytest.mark.parametrize(
    (
        "message",
        "used_tool_calls",
        "max_tool_calls_per_step",
        "expected_route",
    ),
    [
        pytest.param(
            AIMessage(content="Enough evidence is available."),
            0,
            3,
            "build_candidate",
            id="no-calls-build-candidate",
        ),
        pytest.param(
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="list_tree",
                        args={"max_depth": 2, "max_entries": 10},
                        id="call-tree",
                        type="tool_call",
                    )
                ],
            ),
            2,
            3,
            "authorize_tools",
            id="exact-remaining-budget-authorizes",
        ),
        pytest.param(
            AIMessage(
                content="",
                invalid_tool_calls=[
                    InvalidToolCall(
                        name="search_text",
                        args='{"query":',
                        id="call-invalid-search",
                        error="Invalid JSON arguments",
                        type="invalid_tool_call",
                    )
                ],
            ),
            0,
            3,
            "handle_invalid_tools",
            id="invalid-call-is-returned-for-repair",
        ),
        pytest.param(
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="list_tree",
                        args={},
                        id="call-tree",
                        type="tool_call",
                    ),
                    ToolCall(
                        name="search_text",
                        args={"query": "authenticate"},
                        id="call-search",
                        type="tool_call",
                    ),
                ],
            ),
            2,
            3,
            "reject_tools",
            id="batch-over-budget-is-rejected",
        ),
    ],
)
def test_route_tool_calls_decision_table(
    message: AIMessage,
    used_tool_calls: int,
    max_tool_calls_per_step: int,
    expected_route: ToolCallRoute,
) -> None:
    """Route each valid state according to calls, validity, and budget."""

    state = _state_for_message(
        message,
        used_tool_calls=used_tool_calls,
        max_tool_calls_per_step=max_tool_calls_per_step,
    )

    assert route_tool_calls(state) == expected_route


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="list_tree",
                        args={},
                        id=None,
                        type="tool_call",
                    )
                ],
            ),
            id="valid-call-shape",
        ),
        pytest.param(
            AIMessage(
                content="",
                invalid_tool_calls=[
                    InvalidToolCall(
                        name="search_text",
                        args='{"query":',
                        id=None,
                        error="Invalid JSON arguments",
                        type="invalid_tool_call",
                    )
                ],
            ),
            id="invalid-call-shape",
        ),
    ],
)
def test_route_tool_calls_rejects_missing_call_id(
    message: AIMessage,
) -> None:
    """Every pending call needs an ID so a ToolMessage can answer it."""

    state = _state_for_message(
        message,
        used_tool_calls=0,
        max_tool_calls_per_step=3,
    )

    with pytest.raises(ValueError, match="Tool call id is None"):
        route_tool_calls(state)


def test_route_tool_calls_rejects_empty_history() -> None:
    """Routing without an Agent response is a graph wiring error."""

    state: ToolRoutingInput = {
        "messages": [],
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": 3,
    }

    with pytest.raises(ValueError, match="Message list is empty"):
        route_tool_calls(state)


def test_route_tool_calls_requires_ai_message() -> None:
    """The router must run immediately after Agent, not after another node."""

    state: ToolRoutingInput = {
        "messages": [HumanMessage(content="Inspect the repository.")],
        "step_tool_call_count": 0,
        "max_tool_calls_per_step": 3,
    }

    with pytest.raises(TypeError, match="Last message is not an AIMessage"):
        route_tool_calls(state)


@pytest.mark.parametrize(
    (
        "step_explorer_call_count",
        "max_explorer_calls_per_step",
        "expected_route",
    ),
    [
        pytest.param(0, 1, "agent", id="first-invocation-is-allowed"),
        pytest.param(1, 2, "agent", id="remaining-iteration-is-allowed"),
        pytest.param(
            2,
            2,
            "step_explorer_budget_exhausted",
            id="limit-prevents-another-invocation",
        ),
    ],
)
def test_route_explorer_invocation_enforces_step_budget(
    step_explorer_call_count: int,
    max_explorer_calls_per_step: int,
    expected_route: ExplorerInvocationRoute,
) -> None:
    state: ExplorerInvocationInput = {
        "step_explorer_call_count": step_explorer_call_count,
        "max_explorer_calls_per_step": max_explorer_calls_per_step,
    }

    assert route_explorer_invocation(state) == expected_route
