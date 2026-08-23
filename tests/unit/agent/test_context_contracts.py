"""Contracts for traceable active context and raw-artifact metadata."""

import json
import pytest
from pydantic import ValidationError

from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.context import (
    ContextBuildReport,
    ContextItem,
    ContextSection,
    ContextSelectionDecision,
    ContextSelectionOutcome,
    ContextSelectionReason,
    ContextSource,
    ContextSourceKind,
    ProvenanceKind,
    ProvenanceLink,
    RawArtifactKind,
    RepositoryLocation,
    VerificationStatus,
    create_context_build_report,
    create_context_item,
    create_context_pack,
    create_context_source,
    create_raw_artifact_metadata,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40


def _artifact(content: str = "def authenticate():\n    return True\n"):
    return create_raw_artifact_metadata(
        kind=RawArtifactKind.TOOL_OUTPUT,
        media_type="text/plain",
        content=content,
    )


def _location(path: str = "src/auth.py") -> RepositoryLocation:
    return RepositoryLocation(
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
        path=path,
        line_start=1,
        line_end=2,
        symbol="authenticate",
    )


def _source(
    *,
    kind: ContextSourceKind = ContextSourceKind.SOURCE_EXCERPT,
) -> ContextSource:
    artifact = _artifact()
    return create_context_source(
        kind=kind,
        raw_artifact_id=artifact.artifact_id,
        location=_location() if kind is ContextSourceKind.SOURCE_EXCERPT else None,
        provenance=(
            ProvenanceLink(kind=ProvenanceKind.PLAN_STEP, id="inspect-auth"),
            ProvenanceLink(kind=ProvenanceKind.TOOL_CALL, id="call-read-auth"),
        ),
    )


def _item(
    *,
    content: str = "src/auth.py:1-2 defines authenticate.",
    priority: int = 80,
    pinned: bool = False,
) -> ContextItem:
    return create_context_item(
        source=_source(),
        section=ContextSection.EVIDENCE,
        verification_status=VerificationStatus.VERIFIED,
        content=content,
        estimated_tokens=12,
        priority=priority,
        pinned=pinned,
    )


def _selected(
    item: ContextItem,
    *,
    position: int = 0,
) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        item_id=item.item_id,
        outcome=ContextSelectionOutcome.SELECTED,
        reason=ContextSelectionReason.VERIFIED_EVIDENCE,
        estimated_tokens=item.estimated_tokens,
        selected_position=position,
    )


def test_context_vocabularies_are_closed() -> None:
    assert set(ContextSourceKind) == {
        ContextSourceKind.OBJECTIVE,
        ContextSourceKind.PLAN_STEP,
        ContextSourceKind.TOOL_RESULT,
        ContextSourceKind.SOURCE_EXCERPT,
        ContextSourceKind.EVALUATOR_REQUEST,
        ContextSourceKind.HUMAN_FEEDBACK,
        ContextSourceKind.CANDIDATE,
        ContextSourceKind.MEMORY,
        ContextSourceKind.SUMMARY,
    }
    assert set(VerificationStatus) == {
        VerificationStatus.VERIFIED,
        VerificationStatus.HUMAN_APPROVED,
        VerificationStatus.INFERRED,
        VerificationStatus.UNVERIFIED,
        VerificationStatus.REJECTED,
    }
    assert set(ContextSelectionOutcome) == {
        ContextSelectionOutcome.SELECTED,
        ContextSelectionOutcome.EXCLUDED,
    }


def test_raw_artifact_metadata_is_content_addressed_and_contains_no_content() -> None:
    first = _artifact()
    same = _artifact()
    changed = _artifact("different")

    assert first == same
    assert first.artifact_id != changed.artifact_id
    assert first.byte_count == len("def authenticate():\n    return True\n".encode())
    assert set(type(first).model_fields) == {
        "schema_version",
        "artifact_id",
        "kind",
        "media_type",
        "content_digest",
        "byte_count",
    }
    assert "content" not in first.model_dump(mode="json")


def test_raw_artifact_metadata_rejects_tampered_identity_and_media_type() -> None:
    artifact = _artifact()
    payload = artifact.model_dump(mode="json")

    with pytest.raises(ValidationError, match="artifact_id"):
        type(artifact).model_validate_json(
            json.dumps({**payload, "artifact_id": "f" * 64})
        )
    with pytest.raises(ValidationError, match="media_type"):
        create_raw_artifact_metadata(
            kind=RawArtifactKind.TOOL_OUTPUT,
            media_type="Text/Plain; charset=utf-8",
            content="result",
        )


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../secret.py", "src/../secret.py", "src\\auth.py", "./a.py"],
)
def test_repository_location_requires_normalized_relative_path(path: str) -> None:
    with pytest.raises(ValidationError, match="path"):
        _location(path)


def test_repository_location_requires_complete_ascending_line_range() -> None:
    values = _location().model_dump(mode="json")

    with pytest.raises(ValidationError, match="line_end"):
        RepositoryLocation.model_validate({**values, "line_start": None, "line_end": 2})
    with pytest.raises(ValidationError, match="line_end"):
        RepositoryLocation.model_validate({**values, "line_start": 3, "line_end": 2})


def test_context_source_has_canonical_provenance_and_stable_identity() -> None:
    first = _source()
    reordered = create_context_source(
        kind=ContextSourceKind.SOURCE_EXCERPT,
        raw_artifact_id=first.raw_artifact_id,
        location=first.location,
        provenance=tuple(reversed(first.provenance)),
    )

    assert first == reordered
    assert first.source_id == reordered.source_id
    assert first.provenance == tuple(
        sorted(first.provenance, key=lambda link: (link.kind.value, link.id))
    )

    payload = first.model_dump(mode="json")
    with pytest.raises(ValidationError, match="canonical"):
        ContextSource.model_validate_json(
            json.dumps({**payload, "provenance": list(reversed(payload["provenance"]))})
        )
    with pytest.raises(ValidationError, match="source_id"):
        ContextSource.model_validate_json(
            json.dumps({**payload, "source_id": "f" * 64})
        )


def test_source_excerpt_and_tool_result_require_raw_artifact_provenance() -> None:
    provenance = (ProvenanceLink(kind=ProvenanceKind.TOOL_CALL, id="call-1"),)

    with pytest.raises(ValidationError, match="raw_artifact_id"):
        create_context_source(
            kind=ContextSourceKind.TOOL_RESULT,
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match="location"):
        create_context_source(
            kind=ContextSourceKind.SOURCE_EXCERPT,
            raw_artifact_id=_artifact().artifact_id,
            provenance=provenance,
        )


def test_context_item_validates_content_digest_and_stable_identity() -> None:
    first = _item(priority=80, pinned=False)
    reranked = _item(priority=100, pinned=True)

    assert first.item_id == reranked.item_id
    assert first.content_digest == reranked.content_digest
    assert first.priority == 80
    assert reranked.pinned is True

    payload = first.model_dump(mode="json")
    with pytest.raises(ValidationError, match="content_digest"):
        ContextItem.model_validate_json(
            json.dumps({**payload, "content_digest": "f" * 64})
        )
    with pytest.raises(ValidationError, match="item_id"):
        ContextItem.model_validate_json(json.dumps({**payload, "item_id": "f" * 64}))


def test_context_item_preserves_source_whitespace() -> None:
    content = "  indented source line\n"
    item = _item(content=content)

    assert item.content == content


def test_context_item_rejects_unencodable_or_oversized_content() -> None:
    values = {
        "source": _source(),
        "section": ContextSection.EVIDENCE,
        "verification_status": VerificationStatus.VERIFIED,
        "estimated_tokens": 1,
        "priority": 50,
        "pinned": False,
    }

    with pytest.raises((UnicodeEncodeError, ValueError), match="UTF-8"):
        create_context_item(content="\ud800", **values)
    with pytest.raises(ValueError, match="byte limit"):
        create_context_item(content="x" * 262_145, **values)


def test_compressed_summary_cannot_claim_verified_source_status() -> None:
    summary_source = create_context_source(
        kind=ContextSourceKind.SUMMARY,
        provenance=(ProvenanceLink(kind=ProvenanceKind.RAW_ARTIFACT, id="a" * 64),),
    )

    with pytest.raises(ValidationError, match="verified"):
        create_context_item(
            source=summary_source,
            section=ContextSection.HISTORY,
            verification_status=VerificationStatus.VERIFIED,
            content="Compressed claim.",
            estimated_tokens=3,
            priority=20,
        )


def test_model_candidate_cannot_claim_verified_source_status() -> None:
    candidate_source = create_context_source(
        kind=ContextSourceKind.CANDIDATE,
        provenance=(ProvenanceLink(kind=ProvenanceKind.PLAN_STEP, id="step-1"),),
    )

    with pytest.raises(ValidationError, match="verified"):
        create_context_item(
            source=candidate_source,
            section=ContextSection.HISTORY,
            verification_status=VerificationStatus.VERIFIED,
            content="Model-generated claim.",
            estimated_tokens=3,
            priority=20,
        )


@pytest.mark.parametrize(
    ("outcome", "reason", "position"),
    [
        (
            ContextSelectionOutcome.SELECTED,
            ContextSelectionReason.BUDGET_EXCEEDED,
            0,
        ),
        (
            ContextSelectionOutcome.EXCLUDED,
            ContextSelectionReason.VERIFIED_EVIDENCE,
            None,
        ),
        (
            ContextSelectionOutcome.SELECTED,
            ContextSelectionReason.PINNED,
            None,
        ),
        (
            ContextSelectionOutcome.EXCLUDED,
            ContextSelectionReason.DUPLICATE,
            0,
        ),
    ],
)
def test_selection_decision_requires_consistent_reason_and_position(
    outcome: ContextSelectionOutcome,
    reason: ContextSelectionReason,
    position: int | None,
) -> None:
    with pytest.raises(ValidationError):
        ContextSelectionDecision(
            item_id="a" * 64,
            outcome=outcome,
            reason=reason,
            estimated_tokens=2,
            selected_position=position,
        )


def test_context_build_report_reconciles_every_candidate_and_budget() -> None:
    first = _item()
    second = _item(content="A lower-priority historical observation.")
    decisions = (
        _selected(first),
        ContextSelectionDecision(
            item_id=second.item_id,
            outcome=ContextSelectionOutcome.EXCLUDED,
            reason=ContextSelectionReason.BUDGET_EXCEEDED,
            estimated_tokens=second.estimated_tokens,
        ),
    )

    report = create_context_build_report(
        budget_tokens=first.estimated_tokens,
        decisions=decisions,
    )

    assert report.candidate_count == 2
    assert report.selected_count == 1
    assert report.selected_tokens == first.estimated_tokens
    assert report.excluded_count == 1
    assert (
        report.build_id
        == create_context_build_report(
            budget_tokens=first.estimated_tokens,
            decisions=decisions,
        ).build_id
    )

    payload = report.model_dump(mode="json")
    with pytest.raises(ValidationError, match="candidate_count"):
        ContextBuildReport.model_validate_json(
            json.dumps({**payload, "candidate_count": 3})
        )
    with pytest.raises(ValidationError, match="build_id"):
        ContextBuildReport.model_validate_json(
            json.dumps({**payload, "build_id": "f" * 64})
        )


def test_context_build_report_rejects_duplicate_items_and_noncanonical_positions() -> (
    None
):
    item = _item()

    with pytest.raises(ValidationError, match="unique"):
        create_context_build_report(
            budget_tokens=100,
            decisions=(_selected(item), _selected(item, position=1)),
        )

    other = _item(content="Another evidence item.")
    with pytest.raises(ValidationError, match="positions"):
        create_context_build_report(
            budget_tokens=100,
            decisions=(_selected(item), _selected(other, position=2)),
        )


def test_context_pack_matches_selected_order_and_hard_budget() -> None:
    first = _item(content="First verified excerpt.")
    second = _item(content="Second verified excerpt.")
    report = create_context_build_report(
        budget_tokens=30,
        decisions=(_selected(first), _selected(second, position=1)),
    )

    pack = create_context_pack(report=report, items=(first, second))

    assert pack.estimated_tokens == 24
    assert pack.items == (first, second)
    assert (
        pack.pack_id
        == create_context_pack(
            report=report,
            items=(first, second),
        ).pack_id
    )

    with pytest.raises(ValidationError, match="selected order"):
        create_context_pack(report=report, items=(second, first))

    with pytest.raises(ValidationError, match="budget"):
        create_context_build_report(
            budget_tokens=20,
            decisions=(_selected(first), _selected(second, position=1)),
        )


def test_context_contracts_round_trip_through_json_and_checkpoint() -> None:
    item = _item()
    report = create_context_build_report(
        budget_tokens=20,
        decisions=(_selected(item),),
    )
    pack = create_context_pack(report=report, items=(item,))

    assert type(pack).model_validate_json(pack.model_dump_json()) == pack

    serializer = RepoCompassCheckpointSerializer()
    encoded = serializer.dumps_typed({"context_pack": pack})
    restored = serializer.loads_typed(encoded)

    assert restored == {"context_pack": pack}
    assert isinstance(restored["context_pack"], type(pack))


def test_context_contracts_forbid_unknown_fields() -> None:
    payload = _item().model_dump(mode="json")

    with pytest.raises(ValidationError):
        ContextItem.model_validate_json(json.dumps({**payload, "raw_prompt": "secret"}))
