"""Unit tests for structured investigation planning contracts."""

import pytest
from pydantic import ValidationError

from repocompass.agent.contracts import FinalResult, InvestigationPlan, PlanStep

VALID_RESULT = {
    "relevant_files": ["src/example.py"],
    "evidence": ["src/example.py:1 contains the relevant definition."],
    "summary": "The relevant implementation was identified.",
    "unresolved_questions": [],
}


@pytest.mark.parametrize("missing_field", VALID_RESULT)
def test_final_result_requires_every_field(missing_field: str) -> None:
    """Omitting any report section should fail validation."""

    incomplete_result = {
        key: value for key, value in VALID_RESULT.items() if key != missing_field
    }

    with pytest.raises(ValidationError):
        FinalResult.model_validate(incomplete_result)


def test_final_result_rejects_extra_fields() -> None:
    """State-only metadata must not leak into the report schema."""

    with pytest.raises(ValidationError):
        FinalResult.model_validate(
            {
                **VALID_RESULT,
                "stop_reason": "tool_call_budget_exhausted",
            }
        )


def _step(
    step_id: str,
    *,
    status: str,
) -> PlanStep:
    return PlanStep.model_validate(
        {
            "id": step_id,
            "objective": f"Complete {step_id}",
            "expected_evidence": [f"Evidence for {step_id}"],
            "status": status,
        }
    )


def test_investigation_plan_accepts_one_matching_active_step() -> None:
    plan = InvestigationPlan(
        task_summary="Locate and explain authentication.",
        success_criteria=["Identify the entry point."],
        steps=[
            _step("locate", status="active"),
            _step("explain", status="pending"),
        ],
        active_step_id="locate",
    )

    assert plan.active_step_id == "locate"
    assert plan.steps[0].status == "active"


def test_investigation_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="step IDs must be unique",
    ):
        InvestigationPlan(
            task_summary="Locate authentication.",
            success_criteria=["Identify the entry point."],
            steps=[
                _step("locate", status="active"),
                _step("locate", status="pending"),
            ],
            active_step_id="locate",
        )


def test_investigation_plan_rejects_non_active_selected_step() -> None:
    with pytest.raises(
        ValidationError,
        match="must identify an active step",
    ):
        InvestigationPlan(
            task_summary="Locate authentication.",
            success_criteria=["Identify the entry point."],
            steps=[
                _step("locate", status="pending"),
                _step("explain", status="active"),
            ],
            active_step_id="locate",
        )


def test_investigation_plan_rejects_multiple_active_steps() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one active step",
    ):
        InvestigationPlan(
            task_summary="Locate authentication.",
            success_criteria=["Identify the entry point."],
            steps=[
                _step("locate", status="active"),
                _step("explain", status="active"),
            ],
            active_step_id="locate",
        )
