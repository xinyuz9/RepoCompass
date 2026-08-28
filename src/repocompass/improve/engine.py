"""Bounded producer/evaluator loop for optional focused improvements."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.improve.contracts import FocusedChangeProposal, FocusedImproveScope
from repocompass.improve.proposal import ProposalValidationError
from repocompass.improve.sandbox import (
    ProposalVerificationReport,
    SandboxExecutionError,
    SandboxUnavailableError,
    VerificationDecisionReason,
    VerificationStatus,
)
from repocompass.tools.contracts import Sha256Digest

MAX_FEEDBACK_OUTPUT_CHARACTERS = 2_048


class _EngineContractModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class ImproveStopReason(StrEnum):
    """Stable terminal reason for one bounded producer/evaluator loop."""

    ACCEPTED = "accepted"
    PROPOSAL_EXHAUSTED = "proposal_exhausted"
    PRODUCER_ERROR = "producer_error"
    DUPLICATE_PROPOSAL = "duplicate_proposal"
    EQUIVALENT_PROPOSAL = "equivalent_proposal"
    INVALID_PROPOSAL_SEQUENCE = "invalid_proposal_sequence"
    NO_PROGRESS = "no_progress"
    REVISION_LIMIT = "revision_limit"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    SANDBOX_ERROR = "sandbox_error"
    VERIFIER_CONTRACT_ERROR = "verifier_contract_error"


class ImproveIterationOutcome(StrEnum):
    """Deterministic result of one unique proposal evaluation."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ImproveCheckFeedback(_EngineContractModel):
    """Sanitized bounded check signal available to a proposal producer."""

    schema_version: Literal[1] = 1
    profile: str = Field(min_length=1, max_length=64)
    status: VerificationStatus
    output_sha256: Sha256Digest
    output_excerpt: str = Field(max_length=MAX_FEEDBACK_OUTPUT_CHARACTERS)


class ImproveEvaluationFeedback(_EngineContractModel):
    """Machine-readable evaluator feedback without filesystem/process authority."""

    schema_version: Literal[1] = 1
    signal_digest: Sha256Digest
    codes: tuple[str, ...] = Field(min_length=1, max_length=16)
    checks: tuple[ImproveCheckFeedback, ...] = Field(default=(), max_length=8)
    side_effect_paths: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("codes", "checks", "side_effect_paths", mode="before")
    @classmethod
    def canonicalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_signal(self) -> "ImproveEvaluationFeedback":
        if self.codes != tuple(sorted(set(self.codes))):
            raise ValueError("feedback codes must be unique and sorted")
        if self.side_effect_paths != tuple(sorted(set(self.side_effect_paths))):
            raise ValueError("feedback side-effect paths must be unique and sorted")
        expected = derive_feedback_signal(
            codes=self.codes,
            checks=self.checks,
            side_effect_paths=self.side_effect_paths,
        )
        if self.signal_digest != expected:
            raise ValueError("signal_digest does not match evaluator feedback")
        return self


class ImproveIterationRecord(_EngineContractModel):
    """One unique proposal and its deterministic evaluator result."""

    schema_version: Literal[1] = 1
    revision_number: int = Field(ge=1, le=3)
    proposal_id: Sha256Digest
    outcome: ImproveIterationOutcome
    validation_error: str | None = Field(default=None, min_length=1, max_length=128)
    verification: ProposalVerificationReport | None = None
    feedback: ImproveEvaluationFeedback | None = None

    @model_validator(mode="after")
    def require_coherent_iteration(self) -> "ImproveIterationRecord":
        if self.verification is None and self.validation_error is None:
            raise ValueError("iteration requires validation or verification evidence")
        if self.verification is not None:
            if self.verification.proposal_id != self.proposal_id:
                raise ValueError("verification proposal identity does not reconcile")
            expected_outcome = (
                ImproveIterationOutcome.ACCEPTED
                if self.verification.accepted
                else ImproveIterationOutcome.REJECTED
            )
            if self.outcome is not expected_outcome:
                raise ValueError("iteration outcome does not match verification")
        if self.outcome is ImproveIterationOutcome.ACCEPTED:
            if self.validation_error is not None or self.feedback is not None:
                raise ValueError("accepted iteration cannot carry rejection feedback")
        elif self.feedback is None:
            raise ValueError("rejected iteration requires bounded feedback")
        return self


class FocusedImproveRunReport(_EngineContractModel):
    """Content-addressed terminal report for a bounded improve attempt."""

    schema_version: Literal[1] = 1
    run_report_id: Sha256Digest
    scope_id: Sha256Digest
    accepted: bool
    stop_reason: ImproveStopReason
    iterations: tuple[ImproveIterationRecord, ...] = Field(default=(), max_length=3)
    accepted_proposal_id: Sha256Digest | None = None

    @field_validator("iterations", mode="before")
    @classmethod
    def canonicalize_iterations(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_terminal_report(self) -> "FocusedImproveRunReport":
        if tuple(item.revision_number for item in self.iterations) != tuple(
            range(1, len(self.iterations) + 1)
        ):
            raise ValueError("iteration revisions must be contiguous from one")
        expected_acceptance = self.stop_reason is ImproveStopReason.ACCEPTED
        if self.accepted is not expected_acceptance:
            raise ValueError("accepted does not match stop_reason")
        if self.accepted:
            if (
                not self.iterations
                or self.iterations[-1].outcome is not ImproveIterationOutcome.ACCEPTED
                or self.accepted_proposal_id != self.iterations[-1].proposal_id
            ):
                raise ValueError("accepted proposal identity is not terminal iteration")
        elif self.accepted_proposal_id is not None:
            raise ValueError("rejected run cannot expose an accepted proposal")
        expected_id = derive_run_report_id(
            scope_id=self.scope_id,
            accepted=self.accepted,
            stop_reason=self.stop_reason,
            iterations=self.iterations,
            accepted_proposal_id=self.accepted_proposal_id,
        )
        if self.run_report_id != expected_id:
            raise ValueError("run_report_id does not match run contents")
        return self


class ProposalProducer(Protocol):
    """Narrow producer boundary with no filesystem or process handle."""

    def propose(
        self,
        scope: FocusedImproveScope,
        feedback: ImproveEvaluationFeedback | None,
    ) -> FocusedChangeProposal | None: ...


class ProposalVerifier(Protocol):
    """Narrow verifier boundary implemented by BubblewrapVerifier."""

    def verify(
        self,
        repository_path: Path,
        scope: FocusedImproveScope,
        proposal: FocusedChangeProposal,
    ) -> ProposalVerificationReport: ...


class SequenceProposalProducer:
    """Deterministic offline producer used by fixtures and the M9 CLI."""

    def __init__(self, proposals: Sequence[FocusedChangeProposal]) -> None:
        if not all(isinstance(item, FocusedChangeProposal) for item in proposals):
            raise TypeError("proposals must contain FocusedChangeProposal values")
        self._proposals = tuple(proposals)
        self._index = 0
        self._feedback_history: list[ImproveEvaluationFeedback | None] = []

    @property
    def feedback_history(self) -> tuple[ImproveEvaluationFeedback | None, ...]:
        return tuple(self._feedback_history)

    def propose(
        self,
        scope: FocusedImproveScope,
        feedback: ImproveEvaluationFeedback | None,
    ) -> FocusedChangeProposal | None:
        if not isinstance(scope, FocusedImproveScope):
            raise TypeError("scope must be a FocusedImproveScope")
        if feedback is not None and not isinstance(feedback, ImproveEvaluationFeedback):
            raise TypeError("feedback must be ImproveEvaluationFeedback or None")
        self._feedback_history.append(feedback)
        if self._index >= len(self._proposals):
            return None
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal


class FocusedImproveEngine:
    """Bound proposals, verification, feedback, and deterministic loop stops."""

    def __init__(self, verifier: ProposalVerifier) -> None:
        if not callable(getattr(verifier, "verify", None)):
            raise TypeError("verifier must provide verify")
        self._verifier = verifier

    def run(
        self,
        repository_path: Path,
        scope: FocusedImproveScope,
        producer: ProposalProducer,
    ) -> FocusedImproveRunReport:
        if not isinstance(repository_path, Path):
            raise TypeError("repository_path must be a pathlib.Path")
        if not isinstance(scope, FocusedImproveScope):
            raise TypeError("scope must be a FocusedImproveScope")
        if not callable(getattr(producer, "propose", None)):
            raise TypeError("producer must provide propose")
        iterations: list[ImproveIterationRecord] = []
        feedback: ImproveEvaluationFeedback | None = None
        proposal_ids: set[str] = set()
        replacement_fingerprints: set[str] = set()
        feedback_signals: set[str] = set()
        stop_reason = ImproveStopReason.REVISION_LIMIT

        for expected_revision in range(1, scope.max_revision_count + 1):
            try:
                proposal = producer.propose(scope, feedback)
            except Exception:
                stop_reason = ImproveStopReason.PRODUCER_ERROR
                break
            if proposal is None:
                stop_reason = ImproveStopReason.PROPOSAL_EXHAUSTED
                break
            if not isinstance(proposal, FocusedChangeProposal):
                stop_reason = ImproveStopReason.PRODUCER_ERROR
                break
            if proposal.proposal_id in proposal_ids:
                stop_reason = ImproveStopReason.DUPLICATE_PROPOSAL
                break
            fingerprint = _replacement_fingerprint(proposal)
            if fingerprint in replacement_fingerprints:
                stop_reason = ImproveStopReason.EQUIVALENT_PROPOSAL
                break
            if (
                proposal.scope_id != scope.scope_id
                or proposal.revision_number != expected_revision
                or proposal.revision_number > scope.max_revision_count
            ):
                stop_reason = ImproveStopReason.INVALID_PROPOSAL_SEQUENCE
                break
            proposal_ids.add(proposal.proposal_id)
            replacement_fingerprints.add(fingerprint)
            try:
                verification = self._verifier.verify(
                    repository_path,
                    scope,
                    proposal,
                )
            except ProposalValidationError as error:
                feedback = _validation_feedback(error)
                iterations.append(
                    ImproveIterationRecord(
                        revision_number=expected_revision,
                        proposal_id=proposal.proposal_id,
                        outcome=ImproveIterationOutcome.REJECTED,
                        validation_error=error.reason.value,
                        feedback=feedback,
                    )
                )
            except SandboxUnavailableError:
                stop_reason = ImproveStopReason.SANDBOX_UNAVAILABLE
                break
            except SandboxExecutionError:
                stop_reason = ImproveStopReason.SANDBOX_ERROR
                break
            else:
                if (
                    verification.scope_id != scope.scope_id
                    or verification.proposal_id != proposal.proposal_id
                ):
                    stop_reason = ImproveStopReason.VERIFIER_CONTRACT_ERROR
                    break
                if verification.accepted:
                    iterations.append(
                        ImproveIterationRecord(
                            revision_number=expected_revision,
                            proposal_id=proposal.proposal_id,
                            outcome=ImproveIterationOutcome.ACCEPTED,
                            verification=verification,
                        )
                    )
                    return create_run_report(
                        scope_id=scope.scope_id,
                        stop_reason=ImproveStopReason.ACCEPTED,
                        iterations=iterations,
                        accepted_proposal_id=proposal.proposal_id,
                    )
                feedback = _verification_feedback(verification)
                iterations.append(
                    ImproveIterationRecord(
                        revision_number=expected_revision,
                        proposal_id=proposal.proposal_id,
                        outcome=ImproveIterationOutcome.REJECTED,
                        verification=verification,
                        feedback=feedback,
                    )
                )

            if feedback.signal_digest in feedback_signals:
                stop_reason = ImproveStopReason.NO_PROGRESS
                break
            feedback_signals.add(feedback.signal_digest)
        return create_run_report(
            scope_id=scope.scope_id,
            stop_reason=stop_reason,
            iterations=iterations,
            accepted_proposal_id=None,
        )


def create_run_report(
    *,
    scope_id: str,
    stop_reason: ImproveStopReason,
    iterations: Sequence[ImproveIterationRecord],
    accepted_proposal_id: str | None,
) -> FocusedImproveRunReport:
    normalized_iterations = tuple(iterations)
    accepted = stop_reason is ImproveStopReason.ACCEPTED
    report_id = derive_run_report_id(
        scope_id=scope_id,
        accepted=accepted,
        stop_reason=stop_reason,
        iterations=normalized_iterations,
        accepted_proposal_id=accepted_proposal_id,
    )
    return FocusedImproveRunReport(
        run_report_id=report_id,
        scope_id=scope_id,
        accepted=accepted,
        stop_reason=stop_reason,
        iterations=normalized_iterations,
        accepted_proposal_id=accepted_proposal_id,
    )


def derive_feedback_signal(
    *,
    codes: Sequence[str],
    checks: Sequence[ImproveCheckFeedback],
    side_effect_paths: Sequence[str],
) -> str:
    payload = {
        "codes": list(codes),
        "checks": [check.model_dump(mode="json") for check in checks],
        "side_effect_paths": list(side_effect_paths),
    }
    return _content_id(b"repocompass:improve-feedback:v1\0", payload)


def derive_run_report_id(
    *,
    scope_id: str,
    accepted: bool,
    stop_reason: ImproveStopReason,
    iterations: Sequence[ImproveIterationRecord],
    accepted_proposal_id: str | None,
) -> str:
    payload = {
        "scope_id": scope_id,
        "accepted": accepted,
        "stop_reason": stop_reason.value,
        "iterations": [item.model_dump(mode="json") for item in iterations],
        "accepted_proposal_id": accepted_proposal_id,
    }
    return _content_id(b"repocompass:improve-run-report:v1\0", payload)


def _validation_feedback(error: ProposalValidationError) -> ImproveEvaluationFeedback:
    codes = (error.reason.value,)
    signal = derive_feedback_signal(codes=codes, checks=(), side_effect_paths=())
    return ImproveEvaluationFeedback(
        signal_digest=signal,
        codes=codes,
    )


def _verification_feedback(
    report: ProposalVerificationReport,
) -> ImproveEvaluationFeedback:
    codes = tuple(
        sorted(
            reason.value
            for reason in report.decision_reasons
            if reason is not VerificationDecisionReason.ACCEPTED
        )
    )
    checks = tuple(
        ImproveCheckFeedback(
            profile=check.profile.value,
            status=check.status,
            output_sha256=check.output_sha256,
            output_excerpt=check.output[:MAX_FEEDBACK_OUTPUT_CHARACTERS],
        )
        for check in report.checks
        if check.required and check.status is not VerificationStatus.PASSED
    )
    side_effect_paths = tuple(sorted(set(report.side_effect_paths)))
    signal = derive_feedback_signal(
        codes=codes,
        checks=checks,
        side_effect_paths=side_effect_paths,
    )
    return ImproveEvaluationFeedback(
        signal_digest=signal,
        codes=codes,
        checks=checks,
        side_effect_paths=side_effect_paths,
    )


def _replacement_fingerprint(proposal: FocusedChangeProposal) -> str:
    payload = [
        {
            "path": edit.path,
            "base_sha256": edit.base_sha256,
            "replacement_sha256": edit.replacement_sha256,
        }
        for edit in proposal.edits
    ]
    return _content_id(b"repocompass:replacement-set:v1\0", payload)


def _content_id(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()
