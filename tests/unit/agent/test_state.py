"""Unit tests for RepoCompass agent state initialization."""

from pathlib import Path

import pytest
from repocompass.agent.state import RepoCompassState, create_initial_state


def test_create_initial_state_sets_required_fields(tmp_path: Path) -> None:
    objective = "Find the authentication code"
    state = create_initial_state(
        tmp_path,
        objective,
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=4,
    )

    assert state["messages"][0].content == objective
    assert state["repo_path"] == tmp_path.resolve()
    assert state["objective"] == objective
    assert state["human_review_payload"] is None
    assert state["human_decision_action"] is None
    assert state["human_feedback"] is None
    assert state["investigation_plan"] is None
    assert state["step_tool_call_count"] == 0
    assert state["max_tool_calls_per_step"] == 3
    assert state["step_explorer_call_count"] == 0
    assert state["max_explorer_calls_per_step"] == 4
    assert state["plan_revision_count"] == 0
    assert state["max_plan_revisions"] == 2
    assert state["step_stop_reason"] is None
    assert state["termination_status"] is None
    assert state["candidate_result"] is None
    assert state["evaluation_report"] is None
    assert state["next_action"] is None
    assert state["final_result"] is None


def test_global_state_omits_derived_and_obsolete_fields() -> None:
    assert "active_objective" not in RepoCompassState.__annotations__
    assert "decision_reason" not in RepoCompassState.__annotations__


@pytest.mark.parametrize("max_tool_calls_per_step", [0, -1])
def test_create_initial_state_rejects_non_positive_budget(
    tmp_path: Path,
    max_tool_calls_per_step: int,
) -> None:
    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            "Find the authentication code",
            max_tool_calls_per_step=max_tool_calls_per_step,
            max_explorer_calls_per_step=3,
        )


@pytest.mark.parametrize("max_explorer_calls_per_step", [0, -1])
def test_create_initial_state_rejects_non_positive_explorer_budget(
    tmp_path: Path,
    max_explorer_calls_per_step: int,
) -> None:
    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            "Find the authentication code",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=max_explorer_calls_per_step,
        )


@pytest.mark.parametrize("max_plan_revisions", [0, -1])
def test_create_initial_state_rejects_non_positive_revision_budget(
    tmp_path: Path,
    max_plan_revisions: int,
) -> None:
    with pytest.raises(ValueError):
        create_initial_state(
            tmp_path,
            "Find the authentication code",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=3,
            max_plan_revisions=max_plan_revisions,
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
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=3,
        )
