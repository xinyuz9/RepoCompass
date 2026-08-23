"""Deterministic Active Context Builder tests."""

import json

import pytest
from pydantic import ValidationError

from repocompass.agent.context import (
    ContextItem,
    ContextSection,
    ContextSelectionOutcome,
    ContextSelectionReason,
    ContextSourceKind,
    ProvenanceKind,
    ProvenanceLink,
    RawArtifactKind,
    RepositoryLocation,
    VerificationStatus,
    create_context_item,
    create_context_source,
    create_raw_artifact_metadata,
)
from repocompass.agent.context_builder import (
    ContextBudgetError,
    ContextBudgetPolicy,
    ContextCandidateError,
    ContextSectionReservation,
    build_active_context,
    estimate_context_item_tokens,
    render_context_pack,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40


def _item(
    name: str,
    content: str,
    *,
    kind: ContextSourceKind = ContextSourceKind.SOURCE_EXCERPT,
    section: ContextSection = ContextSection.EVIDENCE,
    verification_status: VerificationStatus | None = None,
    priority: int = 50,
    pinned: bool = False,
    raw_artifact_id: str | None = None,
    path: str | None = None,
) -> ContextItem:
    if raw_artifact_id is None and kind in {
        ContextSourceKind.SOURCE_EXCERPT,
        ContextSourceKind.TOOL_RESULT,
    }:
        raw_artifact_id = create_raw_artifact_metadata(
            kind=RawArtifactKind.SOURCE_EXCERPT,
            media_type="text/plain",
            content=content,
        ).artifact_id
    location = None
    if kind is ContextSourceKind.SOURCE_EXCERPT:
        location = RepositoryLocation(
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
            path=path or f"src/{name}.py",
            line_start=1,
            line_end=max(1, content.count("\n") + 1),
        )
    provenance_kind = (
        ProvenanceKind.OBJECTIVE
        if kind is ContextSourceKind.OBJECTIVE
        else ProvenanceKind.PLAN_STEP
    )
    source = create_context_source(
        kind=kind,
        raw_artifact_id=raw_artifact_id,
        location=location,
        provenance=(ProvenanceLink(kind=provenance_kind, id=f"{name}-origin"),),
    )
    if verification_status is None:
        verification_status = (
            VerificationStatus.VERIFIED
            if kind is ContextSourceKind.SOURCE_EXCERPT
            else VerificationStatus.UNVERIFIED
        )
    return create_context_item(
        source=source,
        section=section,
        verification_status=verification_status,
        content=content,
        estimated_tokens=1,
        priority=priority,
        pinned=pinned,
    )


def _policy(
    budget_tokens: int,
    *,
    reservations: tuple[ContextSectionReservation, ...] = (),
    compression_enabled: bool = False,
    min_compressed_content_bytes: int = 16,
) -> ContextBudgetPolicy:
    return ContextBudgetPolicy(
        budget_tokens=budget_tokens,
        section_reservations=reservations,
        compression_enabled=compression_enabled,
        min_compressed_content_bytes=min_compressed_content_bytes,
    )


def _decision(pack, item_id: str):
    return next(
        decision
        for decision in pack.build_report.decisions
        if decision.item_id == item_id
    )


def test_policy_rejects_zero_budget_duplicate_sections_and_over_reservation() -> None:
    with pytest.raises(ValidationError, match="budget_tokens"):
        _policy(0)
    duplicate = (
        ContextSectionReservation(section=ContextSection.EVIDENCE, reserved_tokens=10),
        ContextSectionReservation(section=ContextSection.EVIDENCE, reserved_tokens=5),
    )
    with pytest.raises(ValidationError, match="unique"):
        _policy(20, reservations=duplicate)
    excessive = (
        ContextSectionReservation(section=ContextSection.EVIDENCE, reserved_tokens=11),
        ContextSectionReservation(section=ContextSection.FEEDBACK, reserved_tokens=10),
    )
    with pytest.raises(ValidationError, match="reservations"):
        _policy(20, reservations=excessive)


def test_builder_rejects_empty_or_duplicate_candidate_identity() -> None:
    with pytest.raises(ContextCandidateError, match="at least one"):
        build_active_context(candidates=(), policy=_policy(100))

    item = _item("same", "same content")
    with pytest.raises(ContextCandidateError, match="unique"):
        build_active_context(candidates=(item, item), policy=_policy(1_000))


def test_required_objective_exactly_fits_hard_budget() -> None:
    objective = _item(
        "objective",
        "Explain authentication flow.",
        kind=ContextSourceKind.OBJECTIVE,
        section=ContextSection.OBJECTIVE,
    )
    exact_budget = estimate_context_item_tokens(objective)

    pack = build_active_context(
        candidates=(objective,),
        policy=_policy(exact_budget),
    )

    assert pack.estimated_tokens == exact_budget
    assert pack.items[0].estimated_tokens == exact_budget
    assert _decision(pack, objective.item_id).reason is (
        ContextSelectionReason.REQUIRED_OBJECTIVE
    )
    assert len(render_context_pack(pack).encode("utf-8")) == exact_budget


def test_required_objective_one_token_over_budget_fails_closed() -> None:
    objective = _item(
        "objective",
        "Explain authentication flow.",
        kind=ContextSourceKind.OBJECTIVE,
        section=ContextSection.OBJECTIVE,
    )

    with pytest.raises(ContextBudgetError, match="mandatory"):
        build_active_context(
            candidates=(objective,),
            policy=_policy(estimate_context_item_tokens(objective) - 1),
        )


def test_builder_recomputes_untrusted_token_estimate_before_selection() -> None:
    evidence = _item("large", "x" * 256)
    actual_tokens = estimate_context_item_tokens(evidence)
    assert evidence.estimated_tokens == 1
    assert actual_tokens > 1

    pack = build_active_context(
        candidates=(evidence,),
        policy=_policy(actual_tokens - 1),
    )

    assert pack.items == ()
    decision = _decision(pack, evidence.item_id)
    assert decision.outcome is ContextSelectionOutcome.EXCLUDED
    assert decision.reason is ContextSelectionReason.BUDGET_EXCEEDED
    assert decision.estimated_tokens == actual_tokens


def test_selection_and_structured_order_are_independent_of_input_order() -> None:
    objective = _item(
        "objective",
        "Find the request entry point.",
        kind=ContextSourceKind.OBJECTIVE,
        section=ContextSection.OBJECTIVE,
    )
    plan = _item(
        "plan",
        "Inspect the router.",
        kind=ContextSourceKind.PLAN_STEP,
        section=ContextSection.PLAN,
    )
    feedback = _item(
        "feedback",
        "Verify middleware ordering.",
        kind=ContextSourceKind.EVALUATOR_REQUEST,
        section=ContextSection.FEEDBACK,
    )
    evidence = _item("evidence", "router.py calls middleware.py")
    candidates = (evidence, feedback, objective, plan)
    budget = sum(estimate_context_item_tokens(item) for item in candidates)

    first = build_active_context(candidates=candidates, policy=_policy(budget))
    second = build_active_context(
        candidates=tuple(reversed(candidates)),
        policy=_policy(budget),
    )

    assert first == second
    assert tuple(item.section for item in first.items) == (
        ContextSection.OBJECTIVE,
        ContextSection.PLAN,
        ContextSection.FEEDBACK,
        ContextSection.EVIDENCE,
    )


def test_same_raw_source_and_digest_is_deduplicated_with_auditable_reason() -> None:
    content = "def authenticate(): pass"
    raw_artifact_id = create_raw_artifact_metadata(
        kind=RawArtifactKind.SOURCE_EXCERPT,
        media_type="text/plain",
        content=content,
    ).artifact_id
    preferred = _item(
        "preferred",
        content,
        priority=90,
        raw_artifact_id=raw_artifact_id,
        path="src/auth.py",
    )
    duplicate = _item(
        "duplicate",
        content,
        priority=10,
        raw_artifact_id=raw_artifact_id,
        path="src/auth.py",
    )

    pack = build_active_context(
        candidates=(duplicate, preferred),
        policy=_policy(estimate_context_item_tokens(preferred)),
    )

    assert tuple(item.item_id for item in pack.items) == (preferred.item_id,)
    assert _decision(pack, duplicate.item_id).reason is ContextSelectionReason.DUPLICATE


def test_section_reservation_protects_evidence_from_higher_priority_history() -> None:
    evidence = _item("evidence", "verified implementation evidence", priority=10)
    history = _item(
        "history",
        "recent but unverified history",
        kind=ContextSourceKind.CANDIDATE,
        section=ContextSection.HISTORY,
        priority=100,
    )
    evidence_tokens = estimate_context_item_tokens(evidence)

    pack = build_active_context(
        candidates=(history, evidence),
        policy=_policy(
            evidence_tokens,
            reservations=(
                ContextSectionReservation(
                    section=ContextSection.EVIDENCE,
                    reserved_tokens=evidence_tokens,
                ),
            ),
        ),
    )

    assert tuple(item.item_id for item in pack.items) == (evidence.item_id,)
    assert _decision(pack, evidence.item_id).reason is (
        ContextSelectionReason.VERIFIED_EVIDENCE
    )
    assert _decision(pack, history.item_id).reason is (
        ContextSelectionReason.LOWER_PRIORITY
    )


def test_pinned_item_is_mandatory_and_selected_before_relevance() -> None:
    pinned = _item("pinned", "keep this exact source", priority=0, pinned=True)
    relevant = _item("relevant", "higher ranked source", priority=100)

    pack = build_active_context(
        candidates=(relevant, pinned),
        policy=_policy(estimate_context_item_tokens(pinned)),
    )

    assert tuple(item.item_id for item in pack.items) == (pinned.item_id,)
    assert _decision(pack, pinned.item_id).reason is ContextSelectionReason.PINNED


def test_evaluator_request_and_verified_evidence_have_explicit_reasons() -> None:
    evaluator_request = _item(
        "request",
        "Find the missing regression test.",
        kind=ContextSourceKind.EVALUATOR_REQUEST,
        section=ContextSection.FEEDBACK,
    )
    evidence = _item("verified", "tests/test_auth.py covers login")
    budget = sum(
        estimate_context_item_tokens(item) for item in (evaluator_request, evidence)
    )

    pack = build_active_context(
        candidates=(evidence, evaluator_request),
        policy=_policy(budget),
    )

    assert _decision(pack, evaluator_request.item_id).reason is (
        ContextSelectionReason.EVALUATOR_REQUEST
    )
    assert _decision(pack, evidence.item_id).reason is (
        ContextSelectionReason.VERIFIED_EVIDENCE
    )


def test_traceable_compression_downgrades_verified_excerpt_to_unverified_summary() -> (
    None
):
    original = _item("large", "line = True\n" * 80)
    full_tokens = estimate_context_item_tokens(original)
    budget = full_tokens // 2

    pack = build_active_context(
        candidates=(original,),
        policy=_policy(
            budget,
            compression_enabled=True,
            min_compressed_content_bytes=32,
        ),
    )

    assert len(pack.items) == 1
    compressed = pack.items[0]
    assert compressed.item_id != original.item_id
    assert compressed.source.kind is ContextSourceKind.SUMMARY
    assert compressed.source.raw_artifact_id == original.source.raw_artifact_id
    assert compressed.source.location == original.source.location
    assert compressed.verification_status is VerificationStatus.UNVERIFIED
    assert "[TRUNCATED:" in compressed.content
    assert pack.estimated_tokens <= budget
    assert _decision(pack, original.item_id).outcome is (
        ContextSelectionOutcome.EXCLUDED
    )
    assert _decision(pack, compressed.item_id).reason is (
        ContextSelectionReason.COMPRESSED_FIT
    )


def test_compression_disabled_excludes_oversized_optional_item() -> None:
    original = _item("large", "line = True\n" * 80)

    pack = build_active_context(
        candidates=(original,),
        policy=_policy(estimate_context_item_tokens(original) // 2),
    )

    assert pack.items == ()
    assert _decision(pack, original.item_id).reason is (
        ContextSelectionReason.BUDGET_EXCEEDED
    )


def test_rendered_jsonl_preserves_joinable_source_provenance() -> None:
    evidence = _item("evidence", "src/auth.py validates the token")
    pack = build_active_context(
        candidates=(evidence,),
        policy=_policy(estimate_context_item_tokens(evidence)),
    )

    rendered = render_context_pack(pack)
    records = [json.loads(line) for line in rendered.splitlines()]

    assert len(records) == 1
    assert records[0]["item_id"] == evidence.item_id
    assert records[0]["source"]["raw_artifact_id"] == evidence.source.raw_artifact_id
    assert records[0]["source"]["location"]["path"] == "src/evidence.py"
    assert records[0]["verification_status"] == "verified"
    assert len(rendered.encode("utf-8")) == pack.estimated_tokens
