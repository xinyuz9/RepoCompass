"""Explicit review exchanges persisted at the two human-review node boundaries."""

from hashlib import sha256
import json

from repocompass.agent.lifecycle import (
    MAX_REVIEW_HISTORY,
    MAX_REVIEW_TEXT,
    HumanDecision,
    HumanReviewHistory,
    HumanReviewRecord,
    HumanReviewRequest,
)


def validated_review_history(value: object) -> HumanReviewHistory | None:
    """None means a legacy checkpoint, never proof that no review occurred."""
    if value is None:
        return None
    if not isinstance(value, HumanReviewHistory):
        raise ValueError("invalid checkpoint review history")
    # The stock InMemorySaver may revive strict models via model_construct,
    # leaving msgpack arrays as lists. Normalize only sequence containers;
    # never coerce decision values, booleans, counters or arbitrary mappings.
    fields = dict(value.__dict__)
    records = fields.get("records")
    if isinstance(records, (list, tuple)):
        normalized = []
        for record in records:
            if not isinstance(record, HumanReviewRecord):
                raise ValueError("invalid checkpoint review record")
            record_fields = dict(record.__dict__)
            actions = record_fields.get("allowed_actions")
            if isinstance(actions, list):
                record_fields["allowed_actions"] = tuple(actions)
            normalized.append(
                HumanReviewRecord.model_validate(record_fields, strict=True)
            )
        fields["records"] = tuple(normalized)
    return HumanReviewHistory.model_validate(fields, strict=True)


def _digest(request: HumanReviewRequest) -> str:
    encoded = json.dumps(
        request.model_dump(mode="json", exclude_computed_fields=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(encoded.encode()).hexdigest()


def record_review_request(
    value: object, request: HumanReviewRequest
) -> HumanReviewHistory:
    previous = validated_review_history(value)
    history = previous or HumanReviewHistory(prior_history_unavailable=True)
    if history.records and history.records[-1].action is None:
        # A checkpoint replay may prepare the same request; never duplicate it.
        if history.records[-1].request_digest == _digest(request):
            return history
        raise ValueError("cannot replace an unanswered review")
    summary = request.candidate_result.summary
    record = HumanReviewRecord(
        review_number=history.omitted_count + len(history.records) + 1,
        request_digest=_digest(request),
        candidate_summary=summary[:MAX_REVIEW_TEXT],
        evaluation_passed=request.evaluation_report.passed,
        deterministic_issue_count=len(request.evaluation_report.deterministic_issues),
        allowed_actions=request.allowed_actions,
        text_truncated=len(summary) > MAX_REVIEW_TEXT,
    )
    records = (*history.records, record)
    overflow = max(0, len(records) - MAX_REVIEW_HISTORY)
    return HumanReviewHistory(
        records=records[overflow:],
        omitted_count=history.omitted_count + overflow,
        prior_history_unavailable=history.prior_history_unavailable,
    )


def record_review_decision(
    value: object,
    request: HumanReviewRequest,
    decision: HumanDecision,
) -> HumanReviewHistory:
    history = validated_review_history(value)
    if history is None:
        # Resume an old interrupt: retain the known request/decision, flag unknown prefix.
        history = record_review_request(None, request)
    if not history.records:
        raise ValueError("review decision has no recorded request")
    pending = history.records[-1]
    if pending.action is not None or pending.request_digest != _digest(request):
        raise ValueError("review decision does not match the pending record")
    feedback = decision.feedback
    record = HumanReviewRecord(
        **{
            **pending.model_dump(mode="python"),
            "action": decision.action,
            "feedback": None if feedback is None else feedback[:MAX_REVIEW_TEXT],
            "text_truncated": pending.text_truncated
            or (feedback is not None and len(feedback) > MAX_REVIEW_TEXT),
        }
    )
    return HumanReviewHistory(
        records=(*history.records[:-1], record),
        omitted_count=history.omitted_count,
        prior_history_unavailable=history.prior_history_unavailable,
    )
