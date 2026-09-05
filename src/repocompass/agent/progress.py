"""Deterministic active-step progress assessment and stopping policy."""

import hashlib
import json
from typing import TypedDict

from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import InvestigationPlan
from repocompass.tools.contracts import (
    NoProgressAction,
    NoProgressDecision,
    NoProgressDecisionPayload,
    NoProgressReason,
    ProgressSnapshot,
    ProgressSnapshotPayload,
    ToolAuditPayload,
    ToolDecisionOutcome,
    decode_progress_snapshot,
    decode_tool_audit_record,
    encode_no_progress_decision,
    encode_progress_snapshot,
    fingerprint_progress_snapshot,
)

STEP_NO_PROGRESS_REASON = "step_no_progress"


class ProgressAssessmentInput(TypedDict):
    """Narrow checkpoint-safe state consumed after one tool batch."""

    investigation_plan: InvestigationPlan | None
    plan_revision_count: int
    evaluation_report: EvaluationReport | None
    progress_snapshot: ProgressSnapshotPayload | None
    consecutive_no_progress_rounds: int
    max_no_progress_rounds: int
    last_tool_batch_audits: list[ToolAuditPayload]
    step_stop_reason: str | None


class ProgressAssessmentUpdate(TypedDict):
    """Checkpoint-safe update emitted by deterministic progress assessment."""

    progress_snapshot: ProgressSnapshotPayload
    no_progress_decision: NoProgressDecisionPayload
    consecutive_no_progress_rounds: int
    step_stop_reason: str | None


def assess_tool_progress(
    state: ProgressAssessmentInput,
) -> ProgressAssessmentUpdate:
    """Compare cumulative active-step signals and apply the hard stall limit."""

    plan = state["investigation_plan"]
    if not isinstance(plan, InvestigationPlan):
        raise ValueError("Cannot assess progress without an InvestigationPlan")
    plan_revision_count = _non_negative_integer(
        state["plan_revision_count"],
        name="plan_revision_count",
    )
    previous_stalled_rounds = _non_negative_integer(
        state["consecutive_no_progress_rounds"],
        name="consecutive_no_progress_rounds",
    )
    threshold = _positive_integer(
        state["max_no_progress_rounds"],
        name="max_no_progress_rounds",
    )
    raw_audits = state["last_tool_batch_audits"]
    if not isinstance(raw_audits, list):
        raise TypeError("last_tool_batch_audits must be a list")

    previous_payload = state["progress_snapshot"]
    previous_snapshot = (
        None if previous_payload is None else decode_progress_snapshot(previous_payload)
    )
    active_step_id = plan.active_step.id
    same_scope = (
        previous_snapshot is not None
        and previous_snapshot.active_step_id == active_step_id
        and previous_snapshot.plan_revision_count == plan_revision_count
    )
    if same_scope:
        assert previous_snapshot is not None
        evidence_fingerprints = set(previous_snapshot.evidence_fingerprints)
        successful_tool_fingerprints = set(
            previous_snapshot.successful_tool_fingerprints
        )
        previous_evaluator_digest = previous_snapshot.evaluator_request_digest
        base_stalled_rounds = previous_stalled_rounds
    else:
        evidence_fingerprints = set()
        successful_tool_fingerprints = set()
        previous_evaluator_digest = None
        base_stalled_rounds = 0

    new_tool_progress_observed = False
    for payload in raw_audits:
        audit = decode_tool_audit_record(payload)
        if audit.decision.outcome is not ToolDecisionOutcome.ALLOWED:
            continue
        fingerprint = audit.fingerprint.digest
        if fingerprint not in successful_tool_fingerprints:
            successful_tool_fingerprints.add(fingerprint)
        # A new query that returns identical evidence is not progress. Keep its
        # request identity for audit, but only novel observed results reset stalls.
        if (
            audit.result_digest is not None
            and audit.result_digest not in evidence_fingerprints
        ):
            evidence_fingerprints.add(audit.result_digest)
            new_tool_progress_observed = True

    evaluator_digest = _evaluator_request_digest(state["evaluation_report"])
    if evaluator_digest is None and same_scope:
        evaluator_digest = previous_evaluator_digest
    evaluator_progress = (
        evaluator_digest is not None and evaluator_digest != previous_evaluator_digest
    )
    progress_observed = new_tool_progress_observed or evaluator_progress

    snapshot = ProgressSnapshot(
        active_step_id=active_step_id,
        evidence_fingerprints=tuple(sorted(evidence_fingerprints)),
        successful_tool_fingerprints=tuple(sorted(successful_tool_fingerprints)),
        evaluator_request_digest=evaluator_digest,
        plan_revision_count=plan_revision_count,
    )
    if progress_observed:
        stalled_rounds = 0
        action = NoProgressAction.CONTINUE
        reason = NoProgressReason.PROGRESS_OBSERVED
    else:
        stalled_rounds = base_stalled_rounds + 1
        if stalled_rounds >= threshold:
            action = NoProgressAction.STOP
            reason = NoProgressReason.NO_PROGRESS_LIMIT
        else:
            action = NoProgressAction.CONTINUE
            reason = NoProgressReason.BELOW_THRESHOLD

    decision = NoProgressDecision(
        action=action,
        reason=reason,
        consecutive_stalled_rounds=stalled_rounds,
        threshold=threshold,
        snapshot_digest=fingerprint_progress_snapshot(snapshot),
    )
    step_stop_reason = state["step_stop_reason"]
    if step_stop_reason is not None and not isinstance(step_stop_reason, str):
        raise TypeError("step_stop_reason must be a string or None")
    if action is NoProgressAction.STOP and step_stop_reason is None:
        step_stop_reason = STEP_NO_PROGRESS_REASON

    return {
        "progress_snapshot": encode_progress_snapshot(snapshot),
        "no_progress_decision": encode_no_progress_decision(decision),
        "consecutive_no_progress_rounds": stalled_rounds,
        "step_stop_reason": step_stop_reason,
    }


def _evaluator_request_digest(report: EvaluationReport | None) -> str | None:
    if report is None:
        return None
    if not isinstance(report, EvaluationReport):
        raise TypeError("evaluation_report must be an EvaluationReport or None")
    assessment = report.semantic_assessment
    payload = {
        "missing_evidence": sorted(set(assessment.missing_evidence)),
        "recommended_focus": sorted(set(assessment.recommended_focus)),
    }
    if not payload["missing_evidence"] and not payload["recommended_focus"]:
        return None
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"repocompass:evaluator-request:v1\0" + encoded).hexdigest()


def _non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _non_negative_integer(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized
