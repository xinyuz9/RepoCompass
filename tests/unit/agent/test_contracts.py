"""Unit tests for structured investigation planning contracts."""

import pytest
from pydantic import ValidationError

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import MAX_PLAN_STEPS, InvestigationPlan, PlanStep
from repocompass.agent.result import FinalResult

VALID_RESULT = {
    "relevant_files": ["src/example.py"],
    "evidence": ["src/example.py:1 contains the relevant definition."],
    "summary": "The relevant implementation was identified.",
    "unresolved_questions": [],
}


def test_internal_contract_fields_are_minimal() -> None:
    assert set(InvestigationPlan.model_fields) == {"steps"}
    assert set(CandidateResult.model_fields) == {
        "evidence",
        "relations",
        "summary",
        "unresolved_questions",
    }
    assert set(EvaluationReport.model_fields) == {
        "deterministic_issues",
        "semantic_assessment",
        "structural_warnings",
        "plan_coverage",
    }
    assert set(EvaluationReport.model_computed_fields) == {"passed"}


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
                "step_stop_reason": "step_tool_call_budget_exhausted",
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
        steps=[
            _step("locate", status="active"),
            _step("explain", status="pending"),
        ],
    )

    assert plan.active_step == plan.steps[0]


def test_investigation_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="step IDs must be unique",
    ):
        InvestigationPlan(
            steps=[
                _step("locate", status="active"),
                _step("locate", status="pending"),
            ],
        )


def test_investigation_plan_rejects_missing_active_step() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one active step",
    ):
        InvestigationPlan(
            steps=[
                _step("locate", status="pending"),
                _step("explain", status="pending"),
            ],
        )


def test_investigation_plan_rejects_multiple_active_steps() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one active step",
    ):
        InvestigationPlan(
            steps=[
                _step("locate", status="active"),
                _step("explain", status="active"),
            ],
        )


def test_investigation_plan_rejects_too_many_steps() -> None:
    steps = [_step("active", status="active")]
    steps.extend(
        _step(f"pending-{index}", status="pending") for index in range(MAX_PLAN_STEPS)
    )

    with pytest.raises(ValidationError):
        InvestigationPlan(
            steps=steps,
        )
