"""Deterministic no-progress assessment tests."""

from copy import deepcopy

import pytest

from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.progress import (
    STEP_NO_PROGRESS_REASON,
    ProgressAssessmentInput,
    assess_tool_progress,
)
from repocompass.tools.contracts import (
    NoProgressAction,
    NoProgressReason,
    ToolAuditRecord,
    ToolCallFingerprint,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolPolicyDecision,
    decode_no_progress_decision,
    decode_progress_snapshot,
    encode_tool_audit_record,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40


def _plan(step_id: str = "inspect") -> InvestigationPlan:
    return InvestigationPlan(
        steps=[
            PlanStep(
                id=step_id,
                objective="Inspect repository evidence.",
                expected_evidence=["A verified source location."],
                status="active",
            )
        ]
    )


def _evaluation_report(focus: str) -> EvaluationReport:
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=SemanticAssessment(
            active_step_completed=False,
            answers_objective=False,
            evidence_supports_summary=False,
            has_critical_omissions=True,
            missing_evidence=[f"Missing {focus}."],
            recommended_focus=[focus],
            summary="More repository evidence is required.",
        ),
    )


def _audit(
    *,
    fingerprint: str,
    outcome: ToolDecisionOutcome,
    reason: ToolDecisionReason,
    result_digest: str | None = None,
    sequence: int = 0,
):
    successful = outcome in {
        ToolDecisionOutcome.ALLOWED,
        ToolDecisionOutcome.CACHED,
    }
    return encode_tool_audit_record(
        ToolAuditRecord(
            sequence=sequence,
            run_id="progress-run",
            call_id=f"call-{sequence}",
            tool_name="list_tree",
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
            fingerprint=ToolCallFingerprint(digest=fingerprint),
            decision=ToolPolicyDecision(outcome=outcome, reason=reason),
            latency_ms=1,
            output_bytes=16 if successful else None,
            result_digest=result_digest if successful else None,
        )
    )


def _state(
    audits: list[dict],
    *,
    threshold: int = 3,
) -> ProgressAssessmentInput:
    return {
        "investigation_plan": _plan(),
        "plan_revision_count": 0,
        "evaluation_report": None,
        "progress_snapshot": None,
        "consecutive_no_progress_rounds": 0,
        "max_no_progress_rounds": threshold,
        "last_tool_batch_audits": audits,
        "step_stop_reason": None,
    }


def _next_state(
    state: ProgressAssessmentInput,
    update: dict,
    audits: list[dict],
) -> ProgressAssessmentInput:
    next_state = deepcopy(state)
    next_state.update(update)
    next_state["last_tool_batch_audits"] = audits
    return next_state


def test_new_allowed_fingerprint_and_evidence_are_progress() -> None:
    state = _state(
        [
            _audit(
                fingerprint="1" * 64,
                outcome=ToolDecisionOutcome.ALLOWED,
                reason=ToolDecisionReason.ALLOWED,
                result_digest="2" * 64,
            )
        ]
    )

    update = assess_tool_progress(state)
    snapshot = decode_progress_snapshot(update["progress_snapshot"])
    decision = decode_no_progress_decision(update["no_progress_decision"])

    assert snapshot.active_step_id == "inspect"
    assert snapshot.successful_tool_fingerprints == ("1" * 64,)
    assert snapshot.evidence_fingerprints == ("2" * 64,)
    assert update["consecutive_no_progress_rounds"] == 0
    assert update["step_stop_reason"] is None
    assert decision.action is NoProgressAction.CONTINUE
    assert decision.reason is NoProgressReason.PROGRESS_OBSERVED


def test_new_result_digest_for_known_allowed_call_is_progress() -> None:
    state = _state(
        [
            _audit(
                fingerprint="1" * 64,
                outcome=ToolDecisionOutcome.ALLOWED,
                reason=ToolDecisionReason.ALLOWED,
                result_digest="2" * 64,
            )
        ]
    )
    first_update = assess_tool_progress(state)
    changed_result = _audit(
        fingerprint="1" * 64,
        outcome=ToolDecisionOutcome.ALLOWED,
        reason=ToolDecisionReason.ALLOWED,
        result_digest="3" * 64,
        sequence=1,
    )

    update = assess_tool_progress(_next_state(state, first_update, [changed_result]))
    snapshot = decode_progress_snapshot(update["progress_snapshot"])
    decision = decode_no_progress_decision(update["no_progress_decision"])

    assert snapshot.successful_tool_fingerprints == ("1" * 64,)
    assert snapshot.evidence_fingerprints == ("2" * 64, "3" * 64)
    assert update["consecutive_no_progress_rounds"] == 0
    assert decision.reason is NoProgressReason.PROGRESS_OBSERVED


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ToolDecisionOutcome.CACHED, ToolDecisionReason.CACHE_HIT),
        (
            ToolDecisionOutcome.DEDUPLICATED,
            ToolDecisionReason.DUPLICATE_CALL,
        ),
        (ToolDecisionOutcome.DENIED, ToolDecisionReason.INVALID_ARGUMENTS),
    ],
)
def test_cache_duplicate_and_denial_do_not_count_as_progress(
    outcome: ToolDecisionOutcome,
    reason: ToolDecisionReason,
) -> None:
    first_state = _state(
        [
            _audit(
                fingerprint="1" * 64,
                outcome=ToolDecisionOutcome.ALLOWED,
                reason=ToolDecisionReason.ALLOWED,
                result_digest="2" * 64,
            )
        ]
    )
    first_update = assess_tool_progress(first_state)
    repeated_audit = _audit(
        fingerprint="1" * 64,
        outcome=outcome,
        reason=reason,
        result_digest="2" * 64 if outcome is ToolDecisionOutcome.CACHED else None,
        sequence=1,
    )

    update = assess_tool_progress(
        _next_state(first_state, first_update, [repeated_audit])
    )
    decision = decode_no_progress_decision(update["no_progress_decision"])

    assert update["consecutive_no_progress_rounds"] == 1
    assert update["step_stop_reason"] is None
    assert decision.action is NoProgressAction.CONTINUE
    assert decision.reason is NoProgressReason.BELOW_THRESHOLD


def test_stalled_rounds_stop_exactly_at_hard_threshold() -> None:
    denied = _audit(
        fingerprint="3" * 64,
        outcome=ToolDecisionOutcome.DENIED,
        reason=ToolDecisionReason.INVALID_ARGUMENTS,
    )
    state = _state([denied], threshold=2)
    first_update = assess_tool_progress(state)

    second_update = assess_tool_progress(_next_state(state, first_update, [denied]))
    decision = decode_no_progress_decision(second_update["no_progress_decision"])

    assert first_update["consecutive_no_progress_rounds"] == 1
    assert second_update["consecutive_no_progress_rounds"] == 2
    assert second_update["step_stop_reason"] == STEP_NO_PROGRESS_REASON
    assert decision.action is NoProgressAction.STOP
    assert decision.reason is NoProgressReason.NO_PROGRESS_LIMIT


def test_new_allowed_call_resets_stalled_rounds() -> None:
    denied = _audit(
        fingerprint="3" * 64,
        outcome=ToolDecisionOutcome.DENIED,
        reason=ToolDecisionReason.INVALID_ARGUMENTS,
    )
    state = _state([denied])
    stalled_update = assess_tool_progress(state)
    allowed = _audit(
        fingerprint="4" * 64,
        outcome=ToolDecisionOutcome.ALLOWED,
        reason=ToolDecisionReason.ALLOWED,
        result_digest="5" * 64,
        sequence=1,
    )

    progress_update = assess_tool_progress(
        _next_state(state, stalled_update, [allowed])
    )
    decision = decode_no_progress_decision(progress_update["no_progress_decision"])

    assert progress_update["consecutive_no_progress_rounds"] == 0
    assert decision.reason is NoProgressReason.PROGRESS_OBSERVED


def test_new_evaluator_request_is_progress_once() -> None:
    denied = _audit(
        fingerprint="3" * 64,
        outcome=ToolDecisionOutcome.DENIED,
        reason=ToolDecisionReason.INVALID_ARGUMENTS,
    )
    state = _state([denied])
    first_update = assess_tool_progress(state)
    focused_state = _next_state(state, first_update, [denied])
    focused_state["evaluation_report"] = _evaluation_report("middleware")

    focused_update = assess_tool_progress(focused_state)
    repeated_update = assess_tool_progress(
        _next_state(focused_state, focused_update, [denied])
    )

    assert focused_update["consecutive_no_progress_rounds"] == 0
    assert repeated_update["consecutive_no_progress_rounds"] == 1


def test_new_step_scope_does_not_carry_stalled_rounds() -> None:
    denied = _audit(
        fingerprint="3" * 64,
        outcome=ToolDecisionOutcome.DENIED,
        reason=ToolDecisionReason.INVALID_ARGUMENTS,
    )
    state = _state([denied], threshold=3)
    first_update = assess_tool_progress(state)
    second_update = assess_tool_progress(_next_state(state, first_update, [denied]))
    new_step_state = _next_state(state, second_update, [denied])
    new_step_state["investigation_plan"] = _plan("focused-retry")
    new_step_state["plan_revision_count"] = 1

    update = assess_tool_progress(new_step_state)

    assert second_update["consecutive_no_progress_rounds"] == 2
    assert update["consecutive_no_progress_rounds"] == 1
    assert (
        decode_progress_snapshot(update["progress_snapshot"]).active_step_id
        == "focused-retry"
    )


def test_existing_budget_stop_reason_has_precedence() -> None:
    denied = _audit(
        fingerprint="3" * 64,
        outcome=ToolDecisionOutcome.DENIED,
        reason=ToolDecisionReason.BUDGET_EXHAUSTED,
    )
    state = _state([denied], threshold=1)
    state["step_stop_reason"] = "step_tool_call_budget_exhausted"

    update = assess_tool_progress(state)

    assert update["step_stop_reason"] == "step_tool_call_budget_exhausted"
    assert (
        decode_no_progress_decision(update["no_progress_decision"]).action
        is NoProgressAction.STOP
    )


@pytest.mark.parametrize("threshold", [0, -1, True])
def test_progress_assessment_rejects_invalid_threshold(threshold: object) -> None:
    state = _state([])
    state["max_no_progress_rounds"] = threshold  # type: ignore[typeddict-item]

    with pytest.raises((TypeError, ValueError), match="max_no_progress_rounds"):
        assess_tool_progress(state)


def test_progress_assessment_rejects_malformed_checkpoint_payload() -> None:
    state = _state([])
    state["progress_snapshot"] = {"active_step_id": "inspect", "extra": True}

    with pytest.raises(ValueError, match="Invalid progress snapshot payload"):
        assess_tool_progress(state)
