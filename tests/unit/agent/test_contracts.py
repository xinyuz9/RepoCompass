"""Unit tests for structured investigation planning contracts."""

import pytest
from pydantic import ValidationError

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.planning import MAX_PLAN_STEPS, InvestigationPlan, PlanStep
from repocompass.agent.result import FinalResult

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
                "stop_reason": "step_tool_call_budget_exhausted",
            }
        )


def test_candidate_result_contains_structured_evidence() -> None:
    evidence = EvidenceItem(
        path="src/auth.py",
        line_start=10,
        line_end=14,
        symbol="authenticate",
        claim="authenticate validates the supplied credentials.",
    )

    candidate = CandidateResult(
        relevant_files=["src/auth.py"],
        evidence=[evidence],
        summary="Authentication is handled by authenticate.",
        unresolved_questions=[],
    )

    assert candidate.evidence == [evidence]
    assert candidate.evidence[0].line_start == 10


@pytest.mark.parametrize(
    ("line_start", "line_end", "message"),
    [
        (None, 3, "line_end requires line_start"),
        (5, 3, "line_end must not precede line_start"),
    ],
)
def test_evidence_item_rejects_invalid_line_range(
    line_start: int | None,
    line_end: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        EvidenceItem(
            path="src/auth.py",
            line_start=line_start,
            line_end=line_end,
            claim="Invalid range.",
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


def test_investigation_plan_rejects_too_many_steps() -> None:
    steps = [_step("active", status="active")]
    steps.extend(
        _step(f"pending-{index}", status="pending") for index in range(MAX_PLAN_STEPS)
    )

    with pytest.raises(ValidationError):
        InvestigationPlan(
            task_summary="Overly long plan.",
            success_criteria=["Stay within the plan-size contract."],
            steps=steps,
            active_step_id="active",
        )
