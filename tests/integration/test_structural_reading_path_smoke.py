"""Offline lexical-versus-structural investigation smoke."""

from pathlib import Path

from repocompass.agent.candidate import (
    CandidateRelation,
    CandidateResult,
    EvidenceItem,
)
from repocompass.agent.evaluation import validate_candidate
from repocompass.agent.structural_artifacts import build_investigation_structure
from repocompass.structure import (
    EvidenceRelationKind,
    ReadingStepKind,
    StructuralVerificationStatus,
)
from repocompass.tools.repository import search_text


def test_structural_path_adds_ordered_lineage_that_lexical_hits_cannot_express(
    tmp_path: Path,
) -> None:
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
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    lexical_hits = search_text(tmp_path, "Service", max_results=100)
    lexical_files = {hit["path"] for hit in lexical_hits}
    assert lexical_files == {"pkg/api.py", "pkg/core.py"}
    assert all(set(hit) == {"line", "path", "text"} for hit in lexical_hits)

    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="pkg/api.py",
                line_start=1,
                line_end=4,
                symbol="pkg.api",
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
        relations=[
            CandidateRelation(
                source="pkg.api",
                relation=EvidenceRelationKind.IMPORTS,
                target="pkg.core",
                supporting_evidence_indices=(0, 1),
                explanation="The API import statement names the core module.",
            )
        ],
        summary="The API imports the core implementation.",
        unresolved_questions=[],
    )
    validation = validate_candidate(tmp_path, candidate)
    result = build_investigation_structure(
        repo_path=tmp_path,
        objective="Trace the API to its core implementation and related tests.",
        candidate=candidate,
        valid_evidence_indices=frozenset(
            excerpt.evidence_index for excerpt in validation.evidence_pack
        ),
    )

    assert validation.issues == ()
    assert result.issues == ()
    assert result.artifact is not None
    artifact = result.artifact
    assertion_id = artifact.candidate_relation_assertion_ids[0]
    edge = next(
        item for item in artifact.graph.edges if item.assertion_id == assertion_id
    )
    assert edge.verification_status is StructuralVerificationStatus.VERIFIED
    assert edge.evidence
    assert artifact.reading_path.claim_lineages[0].provenance_complete is True
    assert artifact.reading_path.claim_lineages[0].source_locations

    ranks = {
        ReadingStepKind.FILE: 0,
        ReadingStepKind.SYMBOL: 1,
        ReadingStepKind.TEST: 2,
    }
    reading_ranks = [ranks[step.kind] for step in artifact.reading_path.steps]
    assert reading_ranks == sorted(reading_ranks)
    assert {item.canonical_name for item in artifact.relevant_symbols} == {
        "pkg.api",
        "pkg.core.Service",
    }
    assert (
        artifact.artifact_id
        == build_investigation_structure(
            repo_path=tmp_path,
            objective="Trace the API to its core implementation and related tests.",
            candidate=candidate,
            valid_evidence_indices=frozenset({0, 1}),
        ).artifact.artifact_id
    )
