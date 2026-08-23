"""Unit contracts for the persisted final-candidate review interrupt."""

import json

import pytest

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    HumanDecisionAction,
    HumanReviewRequest,
    decode_human_review_interrupt,
    encode_human_review_interrupt,
)


def _review_request() -> HumanReviewRequest:
    return HumanReviewRequest(
        turn_number=2,
        candidate_result=CandidateResult(
            evidence=[
                EvidenceItem(
                    path="src/auth.py",
                    line_start=10,
                    line_end=14,
                    symbol="authenticate",
                    claim="authenticate validates credentials.",
                )
            ],
            summary="Authentication is implemented by authenticate.",
            unresolved_questions=[],
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
                summary="The candidate is ready for review.",
            ),
        ),
    )


def test_human_review_interrupt_payload_is_versioned_json_and_round_trips() -> None:
    review = _review_request()

    payload = encode_human_review_interrupt(review)

    assert payload["kind"] == "human_review"
    assert payload["schema_version"] == 1
    assert payload["request"]["allowed_actions"] == [
        action.value for action in HumanDecisionAction
    ]
    assert "passed" not in payload["request"]["evaluation_report"]
    assert json.loads(json.dumps(payload)) == payload
    assert decode_human_review_interrupt(payload) == review


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {
            "kind": "another_interrupt",
            "schema_version": 1,
            "request": {},
        },
        {
            "kind": "human_review",
            "schema_version": 2,
            "request": {},
        },
        {
            "kind": "human_review",
            "schema_version": 1,
            "request": "not-an-object",
        },
        {
            "kind": "human_review",
            "schema_version": 1,
            "request": {},
            "unexpected": True,
        },
    ],
)
def test_human_review_interrupt_decoder_rejects_malformed_payload(
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="human-review interrupt"):
        decode_human_review_interrupt(payload)


def test_human_review_interrupt_decoder_rejects_computed_or_tampered_fields() -> None:
    payload = encode_human_review_interrupt(_review_request())
    payload["request"]["evaluation_report"]["passed"] = True

    with pytest.raises(ValueError, match="human-review interrupt"):
        decode_human_review_interrupt(payload)

    payload = encode_human_review_interrupt(_review_request())
    payload["request"]["allowed_actions"] = ["approve"]

    with pytest.raises(ValueError, match="allowed_actions"):
        decode_human_review_interrupt(payload)
