"""Tests for strict context-ablation benchmark case files."""

import json
from pathlib import Path

import pytest

from repocompass.agent.context import (
    ContextSection,
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
from repocompass.agent.context_builder import estimate_context_item_tokens
from repocompass.evaluation.case_file import (
    MAX_CONTEXT_ABLATION_CASE_FILE_BYTES,
    load_context_ablation_case_file,
)
from repocompass.evaluation.context_ablation import create_context_ablation_case


def make_case():
    objective_source = create_context_source(
        kind=ContextSourceKind.OBJECTIVE,
        provenance=(ProvenanceLink(kind=ProvenanceKind.OBJECTIVE, id="objective"),),
    )
    objective = create_context_item(
        source=objective_source,
        section=ContextSection.OBJECTIVE,
        verification_status=VerificationStatus.HUMAN_APPROVED,
        content="Locate the authentication entry point.",
        estimated_tokens=1,
        priority=100,
        pinned=True,
    )
    evidence_content = "src/auth.py: authenticate verifies credentials."
    artifact = create_raw_artifact_metadata(
        kind=RawArtifactKind.SOURCE_EXCERPT,
        media_type="text/plain",
        content=evidence_content,
    )
    evidence_source = create_context_source(
        kind=ContextSourceKind.SOURCE_EXCERPT,
        raw_artifact_id=artifact.artifact_id,
        location=RepositoryLocation(
            repository_id="a" * 64,
            commit_sha="b" * 40,
            path="src/auth.py",
            line_start=1,
            line_end=1,
            symbol="authenticate",
        ),
        provenance=(
            ProvenanceLink(kind=ProvenanceKind.PLAN_STEP, id="locate-auth"),
            ProvenanceLink(kind=ProvenanceKind.TOOL_CALL, id="read-auth"),
        ),
    )
    evidence = create_context_item(
        source=evidence_source,
        section=ContextSection.EVIDENCE,
        verification_status=VerificationStatus.VERIFIED,
        content=evidence_content,
        estimated_tokens=1,
        priority=100,
    )
    budget = estimate_context_item_tokens(objective) + estimate_context_item_tokens(
        evidence
    )
    return create_context_ablation_case(
        name="loader-case-v1",
        budget_tokens=budget,
        candidates=(evidence, objective),
        required_evidence_item_ids=(evidence.item_id,),
    )


def write_case(path: Path) -> None:
    path.write_text(make_case().model_dump_json(indent=2), encoding="utf-8")


def test_load_context_ablation_case_file_round_trips_canonical_case(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.json"
    write_case(path)

    loaded = load_context_ablation_case_file(path)

    assert loaded == make_case()


def test_load_context_ablation_case_file_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.json"
    compact = make_case().model_dump_json()
    path.write_text(
        compact.replace(
            '{"schema_version":1,',
            '{"schema_version":1,"schema_version":1,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_context_ablation_case_file(path)


def test_load_context_ablation_case_file_rejects_tampered_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.json"
    payload = make_case().model_dump(mode="json")
    payload["case_id"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid context ablation case file"):
        load_context_ablation_case_file(path)


def test_load_context_ablation_case_file_rejects_oversized_input_before_parse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.json"
    path.write_bytes(b"x" * (MAX_CONTEXT_ABLATION_CASE_FILE_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds"):
        load_context_ablation_case_file(path)
