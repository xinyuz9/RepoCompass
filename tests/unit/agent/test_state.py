"""Unit tests for RepoCompass agent state initialization."""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from repocompass.agent.state import create_initial_state


def test_create_initial_state_sets_required_fields(tmp_path: Path) -> None:
    message = HumanMessage(content="Find the authentication code")
    state = create_initial_state(
        tmp_path,
        [message],
        max_tool_calls=3,
        max_iterations=4,
    )

    assert state["messages"] == [message]
    assert state["repo_path"] == tmp_path.resolve()
    assert state["tool_call_count"] == 0
    assert state["max_tool_calls"] == 3
    assert state["iteration_count"] == 0
    assert state["max_iterations"] == 4
    assert state["stop_reason"] is None
    assert state["final_result"] is None


@pytest.mark.parametrize("max_tool_calls", [0, -1])
def test_create_initial_state_rejects_non_positive_budget(
    tmp_path: Path,
    max_tool_calls: int,
) -> None:
    message = HumanMessage(content="Find the authentication code")

    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            [message],
            max_tool_calls=max_tool_calls,
            max_iterations=3,
        )


@pytest.mark.parametrize("max_iterations", [0, -1])
def test_create_initial_state_rejects_non_positive_iteration_budget(
    tmp_path: Path,
    max_iterations: int,
) -> None:
    message = HumanMessage(content="Find the authentication code")

    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            [message],
            max_tool_calls=3,
            max_iterations=max_iterations,
        )
