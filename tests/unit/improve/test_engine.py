"""Bounded focused-improve evaluator and revision-loop tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from repocompass.improve.contracts import (
    PythonVerificationProfile,
    ScopedImproveTarget,
    VerificationSpec,
    create_change_proposal,
    create_focused_scope_contract,
)
from repocompass.improve.engine import (
    FocusedImproveEngine,
    ImproveStopReason,
    SequenceProposalProducer,
)
from repocompass.improve.sandbox import (
    SandboxProvenance,
    VerificationCommandResult,
    VerificationDecisionReason,
    VerificationStatus,
    create_verification_report,
)


def test_engine_revises_from_bounded_feedback_until_accepted() -> None:
    scope, original = _scope()
    proposals = (
        _proposal(scope, original, "return 2", revision=1),
        _proposal(scope, original, "return 3", revision=2),
    )
    producer = SequenceProposalProducer(proposals)
    verifier = _Verifier(
        (
            (VerificationStatus.FAILED, "one assertion failed"),
            (VerificationStatus.PASSED, "1 passed"),
        )
    )

    report = FocusedImproveEngine(verifier).run(
        Path("/trusted/repository"), scope, producer
    )

    assert report.accepted is True
    assert report.stop_reason is ImproveStopReason.ACCEPTED
    assert report.accepted_proposal_id == proposals[1].proposal_id
    assert len(report.iterations) == 2
    assert verifier.calls == [proposal.proposal_id for proposal in proposals]
    assert producer.feedback_history[0] is None
    assert producer.feedback_history[1] is not None
    assert producer.feedback_history[1].codes == ("check_failed",)
    assert "assertion failed" in producer.feedback_history[1].checks[0].output_excerpt


def test_exact_duplicate_proposal_stops_without_reverification() -> None:
    scope, original = _scope()
    proposal = _proposal(scope, original, "return 2", revision=1)
    producer = SequenceProposalProducer((proposal, proposal))
    verifier = _Verifier(
        (
            (VerificationStatus.FAILED, "failed once"),
            (VerificationStatus.PASSED, "must not run"),
        )
    )

    report = FocusedImproveEngine(verifier).run(
        Path("/trusted/repository"), scope, producer
    )

    assert report.accepted is False
    assert report.stop_reason is ImproveStopReason.DUPLICATE_PROPOSAL
    assert len(report.iterations) == 1
    assert verifier.calls == [proposal.proposal_id]


def test_equivalent_replacements_stop_even_when_metadata_changes() -> None:
    scope, original = _scope()
    first = _proposal(scope, original, "return 2", revision=1, rationale="first")
    equivalent = _proposal(
        scope,
        original,
        "return 2",
        revision=2,
        rationale="different rationale",
    )
    verifier = _Verifier(((VerificationStatus.FAILED, "failed once"),))

    report = FocusedImproveEngine(verifier).run(
        Path("/trusted/repository"),
        scope,
        SequenceProposalProducer((first, equivalent)),
    )

    assert report.stop_reason is ImproveStopReason.EQUIVALENT_PROPOSAL
    assert verifier.calls == [first.proposal_id]


def test_repeated_evaluator_signal_stops_no_progress_loop() -> None:
    scope, original = _scope()
    first = _proposal(scope, original, "return 2", revision=1)
    second = _proposal(scope, original, "return 3", revision=2)
    verifier = _Verifier(
        (
            (VerificationStatus.FAILED, "same deterministic failure"),
            (VerificationStatus.FAILED, "same deterministic failure"),
        )
    )

    report = FocusedImproveEngine(verifier).run(
        Path("/trusted/repository"),
        scope,
        SequenceProposalProducer((first, second)),
    )

    assert report.stop_reason is ImproveStopReason.NO_PROGRESS
    assert len(report.iterations) == 2
    assert verifier.calls == [first.proposal_id, second.proposal_id]


def test_revision_budget_and_invalid_sequence_are_deterministic() -> None:
    scope, original = _scope(max_revisions=2)
    first = _proposal(scope, original, "return 2", revision=1)
    second = _proposal(scope, original, "return 3", revision=2)
    third = _proposal(scope, original, "return 4", revision=3)
    verifier = _Verifier(
        (
            (VerificationStatus.FAILED, "failure-a"),
            (VerificationStatus.FAILED, "failure-b"),
        )
    )
    exhausted = FocusedImproveEngine(verifier).run(
        Path("/trusted/repository"),
        scope,
        SequenceProposalProducer((first, second, third)),
    )

    assert exhausted.stop_reason is ImproveStopReason.REVISION_LIMIT
    assert len(exhausted.iterations) == 2
    assert len(verifier.calls) == 2

    wrong_revision = _proposal(scope, original, "return 9", revision=2)
    invalid = FocusedImproveEngine(_Verifier(())).run(
        Path("/trusted/repository"),
        scope,
        SequenceProposalProducer((wrong_revision,)),
    )
    assert invalid.stop_reason is ImproveStopReason.INVALID_PROPOSAL_SEQUENCE
    assert invalid.iterations == ()


class _Verifier:
    def __init__(self, outcomes: tuple[tuple[VerificationStatus, str], ...]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[str] = []

    def verify(self, _repository_path, scope, proposal):
        self.calls.append(proposal.proposal_id)
        status, output = next(self._outcomes)
        check = VerificationCommandResult(
            profile=PythonVerificationProfile.PYTEST,
            arguments=("tests",),
            required=True,
            status=status,
            exit_code=0 if status is VerificationStatus.PASSED else 1,
            duration_ms=10,
            output=output,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            output_truncated=False,
        )
        reasons = (
            (VerificationDecisionReason.ACCEPTED,)
            if status is VerificationStatus.PASSED
            else (VerificationDecisionReason.CHECK_FAILED,)
        )
        return create_verification_report(
            scope_id=scope.scope_id,
            proposal_id=proposal.proposal_id,
            sandbox=SandboxProvenance(
                cpu_limit_seconds=10,
                address_space_limit_bytes=2_147_483_648,
                process_limit=128,
                output_limit_bytes=262_144,
            ),
            checks=(check,),
            side_effect_paths=(),
            decision_reasons=reasons,
        )


def _scope(*, max_revisions: int = 3):
    original = "def allowed() -> int:\n    return 1\n"
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    scope = create_focused_scope_contract(
        repository_id="a" * 64,
        commit_sha="b" * 40,
        objective="Return a different value.",
        targets=(
            ScopedImproveTarget(
                path="pkg/example.py",
                base_sha256=digest,
                byte_count=len(original.encode("utf-8")),
                line_count=3,
                symbol="pkg.example.allowed",
                line_start=1,
                line_end=2,
            ),
        ),
        verifications=(
            VerificationSpec(
                profile=PythonVerificationProfile.PYTEST,
                arguments=("tests",),
                timeout_seconds=30,
            ),
        ),
        max_changed_lines=20,
        max_revision_count=max_revisions,
    )
    return scope, original


def _proposal(
    scope,
    original: str,
    replacement_line: str,
    *,
    revision: int,
    rationale: str = "focused revision",
):
    return create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("return 1", replacement_line),
        },
        rationale=rationale,
        revision_number=revision,
    )
