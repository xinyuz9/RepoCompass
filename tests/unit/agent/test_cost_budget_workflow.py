"""Cost-budget routing and deterministic fallback publication."""

import pytest

from repocompass.agent.cost import (
    AttemptCostBudgetSummary,
    CostBudgetReason,
    CostBudgetStatus,
)
from repocompass.agent.nodes import finalize_cost_budget_exhausted
from repocompass.agent.routing import route_after_model_invocation


def _within_budget() -> AttemptCostBudgetSummary:
    return AttemptCostBudgetSummary(
        limit_usd="1",
        status=CostBudgetStatus.WITHIN_BUDGET,
        reason=CostBudgetReason.WITHIN_BUDGET,
        confirmed_cost_usd="0.25",
        remaining_cost_usd="0.75",
        next_invocation_allowed=True,
    )


def _overshot() -> AttemptCostBudgetSummary:
    return AttemptCostBudgetSummary(
        limit_usd="0.1",
        status=CostBudgetStatus.OVERSHOT,
        reason=CostBudgetReason.POST_RESPONSE_OVERSHOOT,
        confirmed_cost_usd="0.12",
        remaining_cost_usd="0",
        overshoot_usd="0.02",
        next_invocation_allowed=False,
    )


@pytest.mark.parametrize(
    "budget",
    [None, _within_budget()],
)
def test_model_invocation_route_continues_when_budget_allows(
    budget: AttemptCostBudgetSummary | None,
) -> None:
    assert (
        route_after_model_invocation(
            {
                "step_stop_reason": None,
                "cost_budget": budget,
            }
        )
        == "continue"
    )


def test_model_invocation_route_closes_on_stable_budget_reason() -> None:
    assert (
        route_after_model_invocation(
            {
                "step_stop_reason": "attempt_cost_budget_exhausted",
                "cost_budget": _overshot(),
            }
        )
        == "cost_budget_exhausted"
    )


def test_budget_fallback_produces_public_result_without_another_model_call() -> None:
    update = finalize_cost_budget_exhausted(
        {
            "objective": "Locate authentication.",
            "candidate_result": None,
            "cost_budget": _overshot(),
        }
    )

    assert update["termination_status"] == "exhausted"
    assert update["step_stop_reason"] == "attempt_cost_budget_exhausted"
    result = update["final_result"]
    assert result.relevant_files == []
    assert result.evidence == []
    assert "cost budget" in result.summary
    assert result.unresolved_questions == ["Locate authentication."]


def test_budget_fallback_rejects_missing_or_allowing_decision() -> None:
    with pytest.raises(ValueError, match="denied"):
        finalize_cost_budget_exhausted(
            {
                "objective": "Locate authentication.",
                "candidate_result": None,
                "cost_budget": _within_budget(),
            }
        )


@pytest.mark.parametrize(
    "reason", ["attempt_cost_reservation_denied", "batch_cost_reservation_denied"]
)
@pytest.mark.parametrize("budget", [None, _within_budget()])
def test_precall_denial_does_not_require_fabricated_overshoot(reason, budget):
    state = {
        "objective": "Locate authentication.",
        "candidate_result": None,
        "cost_budget": budget,
        "step_stop_reason": reason,
    }
    assert route_after_model_invocation(state) == "cost_budget_exhausted"
    result = finalize_cost_budget_exhausted(state)
    assert result["termination_status"] == "exhausted"
    assert result["step_stop_reason"] == reason
    assert "reservation" in result["final_result"].summary
    assert "overshoot" not in result["final_result"].summary
