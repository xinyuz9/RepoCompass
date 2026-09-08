"""Candidate-to-structure integration and provenance-boundary tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.agent.candidate import (
    CandidateRelation,
    CandidateResult,
    EvidenceItem,
)
from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.evaluation import validate_candidate
from repocompass.agent.structural_artifacts import (
    _resolve_graph_selector,
    build_investigation_structure,
)
from repocompass.structure import (
    EvidenceRelationKind,
    StructuralVerificationStatus,
)


def _write_fixture(root: Path) -> None:
    sources = {
        "pkg/core.py": (
            "def helper():\n"
            "    return True\n\n"
            "class Service:\n"
            "    def run(self):\n"
            "        return helper()\n"
        ),
        "pkg/api.py": (
            "from .core import Service\n\ndef build():\n    return Service().run()\n"
        ),
        "tests/test_api.py": (
            "from pkg.api import build\n\ndef test_build():\n    assert build()\n"
        ),
    }
    for relative_path, source in sources.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")


def _candidate(*, inferred_reference: bool = False) -> CandidateResult:
    relation = (
        CandidateRelation(
            source="pkg.api.build",
            relation=EvidenceRelationKind.REFERENCES,
            target="pkg.core.Service",
            supporting_evidence_indices=(0, 1),
            explanation=(
                "The validated API excerpt constructs Service, while lexical AST "
                "references do not resolve cross-file bindings."
            ),
        )
        if inferred_reference
        else CandidateRelation(
            source="pkg.api",
            relation=EvidenceRelationKind.IMPORTS,
            target="pkg.core",
            supporting_evidence_indices=(0, 1),
            explanation="The API import statement names the core module.",
        )
    )
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path="pkg/api.py",
                line_start=1,
                line_end=4,
                symbol="pkg.api" if not inferred_reference else "pkg.api.build",
                claim="The API imports and constructs Service.",
            ),
            EvidenceItem(
                path="pkg/core.py",
                line_start=4,
                line_end=6,
                symbol="pkg.core.Service",
                claim="Service is defined in the core module.",
            ),
        ],
        relations=[relation],
        summary="The API reaches the core Service.",
        unresolved_questions=[],
    )


def _build(root: Path, candidate: CandidateResult):
    validation = validate_candidate(root, candidate)
    assert validation.issues == ()
    return build_investigation_structure(
        repo_path=root,
        objective="Trace the API to its core implementation.",
        candidate=candidate,
        valid_evidence_indices=frozenset(
            excerpt.evidence_index for excerpt in validation.evidence_pack
        ),
    )


def test_candidate_relation_schema_rejects_unsupported_or_spoofed_facts() -> None:
    common = {
        "source": "pkg.api.build",
        "target": "pkg.core.Service",
        "supporting_evidence_indices": [0],
        "explanation": "Candidate-level relationship.",
    }
    with pytest.raises(ValidationError, match="relation"):
        CandidateRelation.model_validate({**common, "relation": "calls"})
    with pytest.raises(ValidationError, match="verification_status"):
        CandidateRelation.model_validate(
            {
                **common,
                "relation": "references",
                "verification_status": "verified",
            }
        )
    with pytest.raises(ValidationError, match="canonical"):
        CandidateRelation(
            source="pkg.api.build",
            relation=EvidenceRelationKind.REFERENCES,
            target="pkg.core.Service",
            supporting_evidence_indices=(1, 0),
            explanation="Out-of-order evidence is rejected.",
        )


def test_structural_artifact_promotes_only_ast_verified_existing_relation(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    result = _build(tmp_path, _candidate())

    assert result.issues == ()
    assert result.artifact is not None
    artifact = result.artifact
    assert artifact == _build(tmp_path, _candidate()).artifact
    assert artifact.indexed_paths == ("pkg/api.py", "pkg/core.py")
    assert {item.canonical_name for item in artifact.relevant_symbols} == {
        "pkg.api",
        "pkg.core.Service",
    }
    assert len(artifact.candidate_relation_assertion_ids) == 1
    assertion_id = artifact.candidate_relation_assertion_ids[0]
    edge = next(
        item for item in artifact.graph.edges if item.assertion_id == assertion_id
    )
    assert edge.relation is EvidenceRelationKind.IMPORTS
    assert edge.verification_status is StructuralVerificationStatus.VERIFIED
    lineage = artifact.reading_path.claim_lineages[0]
    assert lineage.supporting_assertion_ids == (assertion_id,)
    assert lineage.verification_status is StructuralVerificationStatus.VERIFIED
    assert lineage.provenance_complete is True
    assert lineage.explanation is None


def test_unresolved_cross_file_relation_stays_inferred_with_incomplete_lineage(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    result = _build(tmp_path, _candidate(inferred_reference=True))

    assert result.issues == ()
    assert result.artifact is not None
    assertion_id = result.artifact.candidate_relation_assertion_ids[0]
    edge = next(
        item
        for item in result.artifact.graph.edges
        if item.assertion_id == assertion_id
    )
    assert edge.relation is EvidenceRelationKind.REFERENCES
    assert edge.verification_status is StructuralVerificationStatus.INFERRED
    assert edge.evidence == ()
    lineage = result.artifact.reading_path.claim_lineages[0]
    assert lineage.verification_status is StructuralVerificationStatus.INFERRED
    assert lineage.provenance_complete is False
    assert lineage.evidence == ()
    assert (
        lineage.explanation
        == _candidate(inferred_reference=True).relations[0].explanation
    )


def test_structure_builder_reports_unresolvable_symbols_and_relation_endpoints(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="pkg/api.py",
                line_start=1,
                line_end=1,
                symbol="missing_symbol",
                claim="A missing symbol is claimed.",
            )
        ],
        relations=[
            CandidateRelation(
                source="missing.source",
                relation=EvidenceRelationKind.REFERENCES,
                target="missing.target",
                supporting_evidence_indices=(0,),
                explanation="This relationship cannot be resolved.",
            )
        ],
        summary="Unresolvable structure.",
        unresolved_questions=[],
    )

    result = _build(tmp_path, candidate)

    assert result.artifact is not None
    assert {issue.code for issue in result.issues} == {
        "relation_endpoint_not_found",
        "symbol_not_found",
    }
    assert result.artifact.candidate_relation_assertion_ids == ()
    assert result.artifact.reading_path.claim_lineages == ()


def test_relation_must_reference_deterministically_valid_evidence(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    candidate = _candidate().model_copy(
        update={
            "relations": [
                CandidateRelation(
                    source="pkg.api",
                    relation=EvidenceRelationKind.IMPORTS,
                    target="pkg.core",
                    supporting_evidence_indices=(2,),
                    explanation="The evidence index is outside the candidate.",
                )
            ]
        }
    )

    result = _build(tmp_path, candidate)

    assert result.artifact is not None
    assert [issue.code for issue in result.issues] == ["relation_evidence_invalid"]
    assert result.artifact.candidate_relation_assertion_ids == ()


@pytest.mark.parametrize("citation", ["target_only", "unrelated_source_range"])
def test_valid_but_unrelated_evidence_cannot_promote_an_existing_relation(
    tmp_path: Path, citation: str
) -> None:
    _write_fixture(tmp_path)
    candidate = _candidate()
    if citation == "target_only":
        relation = candidate.relations[0].model_copy(
            update={"supporting_evidence_indices": (1,)}
        )
        candidate = candidate.model_copy(update={"relations": [relation]})
    else:
        evidence = candidate.evidence[0].model_copy(
            update={"line_start": 4, "line_end": 4}
        )
        candidate = candidate.model_copy(
            update={"evidence": [evidence, candidate.evidence[1]]}
        )
    result = _build(tmp_path, candidate)
    assert result.artifact is not None
    assert [issue.code for issue in result.issues] == ["relation_evidence_mismatch"]
    assert result.artifact.candidate_relation_assertion_ids == ()
    assert result.artifact.reading_path.claim_lineages == ()
    # The underlying static edge still exists, but is not evidence for this
    # incorrectly cited candidate proposal.
    assert any(
        edge.relation is EvidenceRelationKind.IMPORTS
        and edge.verification_status is StructuralVerificationStatus.VERIFIED
        for edge in result.artifact.graph.edges
    )


def test_relation_single_line_source_occurrence_is_sufficient(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    candidate = _candidate()
    candidate = candidate.model_copy(
        update={
            "evidence": [
                candidate.evidence[0].model_copy(
                    update={"line_start": 1, "line_end": None}
                ),
                candidate.evidence[1],
            ],
            "relations": [
                candidate.relations[0].model_copy(
                    update={"supporting_evidence_indices": (0,)}
                )
            ],
        }
    )
    result = _build(tmp_path, candidate)
    assert result.issues == ()
    assert len(result.artifact.candidate_relation_assertion_ids) == 1


def test_package_self_import_reaches_structure_and_reading_path(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "pkg/__init__.py").write_text("from . import core\n")
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(path="pkg/__init__.py", line_start=1, claim="Imports core."),
            EvidenceItem(path="pkg/core.py", line_start=1, claim="Defines helper."),
        ],
        relations=[
            CandidateRelation(
                source="pkg/__init__.py",
                target="pkg.core",
                relation=EvidenceRelationKind.IMPORTS,
                supporting_evidence_indices=(0,),
                explanation="Initializer imports the independently indexed core module.",
            )
        ],
        summary="The package imports core.",
        unresolved_questions=[],
    )
    result = _build(tmp_path, candidate)
    assert result.issues == ()
    assert result.artifact is not None
    assert len(result.artifact.candidate_relation_assertion_ids) == 1
    assert result.artifact.reading_path.claim_lineages[0].provenance_complete


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("pkg/core.py", "pkg.core"),
        ("pkg/core.py:Service", "pkg.core.Service"),
        ("pkg/core.py:pkg.core.Service", "pkg.core.Service"),
        ("pkg/core.py:Service.run", None),
        ("pkg/core.py:pkg.core.Service.run", "pkg.core.Service.run"),
        ("pkg/core.py:run", "pkg.core.Service.run"),
        ("pkg/api.py:Service", None),
        ("pkg/missing.py:Service", None),
        ("../pkg/core.py:Service", None),
        ("/pkg/core.py:Service", None),
        ("pkg/./core.py:Service", None),
        ("pkg/core.py:", None),
    ],
)
def test_path_selectors_are_exact_and_confined_to_validated_nodes(
    tmp_path, selector, expected
):
    _write_fixture(tmp_path)
    graph = _build(tmp_path, _candidate()).artifact.graph
    nodes = _resolve_graph_selector(graph, selector)
    assert [node.canonical_name for node in nodes] == (
        [] if expected is None else [expected]
    )


def test_structure_artifact_round_trips_through_strict_checkpoint_serializer(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    result = _build(tmp_path, _candidate(inferred_reference=True))
    assert result.artifact is not None
    expected = {
        "candidate": _candidate(inferred_reference=True),
        "investigation_structure": result.artifact,
    }
    serializer = RepoCompassCheckpointSerializer()
    current: object = expected

    for _ in range(3):
        current = serializer.loads_typed(serializer.dumps_typed(current))

    assert current == expected
    assert isinstance(current, dict)
    restored = current["investigation_structure"]
    assert type(restored) is type(result.artifact)
    assert type(restored.graph) is type(result.artifact.graph)
    assert type(restored.reading_path) is type(result.artifact.reading_path)
