"""Bounded, replay-safe explicit review records; no reconstruction from messages."""

import pytest

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanDecisionAction,
    HumanReviewHistory,
    HumanReviewRequest,
    MAX_REVIEW_HISTORY,
    MAX_REVIEW_TEXT,
)
from repocompass.agent.review_history import (
    record_review_decision,
    record_review_request,
    validated_review_history,
)


def request(summary="Candidate"):
    return HumanReviewRequest(
        turn_number=1,
        candidate_result=CandidateResult(
            evidence=[], summary=summary, unresolved_questions=[]
        ),
        evaluation_report=EvaluationReport(
            deterministic_issues=[],
            semantic_assessment=SemanticAssessment(
                active_step_completed=True,
                answers_objective=True,
                evidence_supports_summary=True,
                has_critical_omissions=False,
                missing_evidence=[],
                recommended_focus=[],
                summary="Ready",
            ),
        ),
    )


def test_request_replay_is_idempotent_and_different_pending_request_fails():
    history = record_review_request(HumanReviewHistory(), request())
    assert record_review_request(history, request()) == history
    assert history.records[0].action is None
    assert not history.prior_history_unavailable
    with pytest.raises(ValueError, match="unanswered"):
        record_review_request(history, request("Different"))


@pytest.mark.parametrize("action", list(HumanDecisionAction))
def test_decision_preserves_request_and_is_not_replayed(action):
    history = record_review_request(HumanReviewHistory(), request())
    decision = HumanDecision(action=action, feedback="Inspect tests")
    decided = record_review_decision(history, request(), decision)
    assert len(decided.records) == 1
    assert decided.records[0].action is action
    assert decided.records[0].feedback == "Inspect tests"
    assert decided.records[0].request_digest == history.records[0].request_digest
    with pytest.raises(ValueError, match="pending record"):
        record_review_decision(decided, request(), decision)
    with pytest.raises(ValueError, match="pending record"):
        record_review_decision(history, request("Other"), decision)


def test_legacy_resume_marks_unknown_prefix_but_new_empty_history_fails_closed():
    decision = HumanDecision(action=HumanDecisionAction.APPROVE)
    legacy = record_review_decision(None, request(), decision)
    assert legacy.prior_history_unavailable
    assert legacy.records[0].action is HumanDecisionAction.APPROVE
    assert validated_review_history(None) is None
    with pytest.raises(ValueError, match="no recorded request"):
        record_review_decision(HumanReviewHistory(), request(), decision)


def test_history_tail_and_text_are_bounded_with_explicit_loss():
    review = request("中" * (MAX_REVIEW_TEXT + 1))
    history = HumanReviewHistory()
    decision = HumanDecision(
        action=HumanDecisionAction.REVISE, feedback="请" * (MAX_REVIEW_TEXT + 1)
    )
    for _ in range(MAX_REVIEW_HISTORY + 3):
        history = record_review_request(history, review)
        history = record_review_decision(history, review, decision)
    assert history.omitted_count == 3
    assert [r.review_number for r in history.records] == list(range(4, 36))
    assert not history.prior_history_unavailable
    assert all(
        r.text_truncated
        and len(r.candidate_summary) == MAX_REVIEW_TEXT
        and len(r.feedback) == MAX_REVIEW_TEXT
        for r in history.records
    )
    assert validated_review_history(history) == history


def test_corrupt_history_and_illegal_decision_fail_closed():
    history = record_review_request(HumanReviewHistory(), request())
    for value in [
        {},
        [],
        history.model_copy(update={"omitted_count": True}),
        history.model_copy(update={"records": (history.records[0],) * 2}),
    ]:
        with pytest.raises(ValueError):
            validated_review_history(value)
    review = request().model_copy(
        update={
            "allowed_actions": (HumanDecisionAction.APPROVE, HumanDecisionAction.CANCEL)
        }
    )
    history = record_review_request(HumanReviewHistory(), review)
    with pytest.raises(ValueError, match="not allowed"):
        record_review_decision(
            history,
            review,
            HumanDecision(action=HumanDecisionAction.REVISE, feedback="Change"),
        )


def test_stock_msgpack_list_containers_are_restored_without_coercing_fields():
    history = record_review_request(HumanReviewHistory(), request())
    record = history.records[0]
    revived = history.model_copy(
        update={
            "records": [
                record.model_copy(
                    update={
                        "allowed_actions": list(record.allowed_actions),
                    }
                )
            ]
        }
    )
    assert validated_review_history(revived) == history
    malformed = history.model_copy(
        update={
            "records": [
                record.model_copy(
                    update={
                        "review_number": True,
                    }
                )
            ]
        }
    )
    with pytest.raises(ValueError):
        validated_review_history(malformed)
