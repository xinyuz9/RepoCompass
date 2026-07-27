"""Unit tests for RepoCompass agent state initialization."""

from pathlib import Path

import pytest
from repocompass.agent.state import create_initial_state


def test_create_initial_state_sets_required_fields(tmp_path: Path) -> None:
    objective = "Find the authentication code"
    state = create_initial_state(
        tmp_path,
        objective,
        max_tool_calls=3,
        max_iterations=4,
    )

    assert state["messages"][0].content == objective
    assert state["repo_path"] == tmp_path.resolve()
    assert state["objective"] == objective
    assert state["active_objective"] == objective
    assert state["investigation_plan"] is None
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
    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            "Find the authentication code",
            max_tool_calls=max_tool_calls,
            max_iterations=3,
        )


@pytest.mark.parametrize("max_iterations", [0, -1])
def test_create_initial_state_rejects_non_positive_iteration_budget(
    tmp_path: Path,
    max_iterations: int,
) -> None:
    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            "Find the authentication code",
            max_tool_calls=3,
            max_iterations=max_iterations,
        )


@pytest.mark.parametrize("objective", ["", "   "])
def test_create_initial_state_rejects_empty_objective(
    tmp_path: Path,
    objective: str,
) -> None:
    with pytest.raises(ValueError, match="Objective must not be empty"):
        create_initial_state(
            tmp_path,
            objective,
            max_tool_calls=3,
            max_iterations=3,
        )
