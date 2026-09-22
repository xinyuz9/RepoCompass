"""Resource-bounded enrichment must not veto independently valid source text."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent import structural_artifacts as structure
from repocompass.agent.candidate import CandidateRelation, CandidateResult, EvidenceItem
from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.evaluation import SemanticAssessment, validate_candidate
from repocompass.agent.nodes import create_evaluate_candidate_node
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.structure import EvidenceRelationKind


@pytest.fixture
def candidate(tmp_path: Path) -> CandidateResult:
    for name in ("a", "b"):
        (tmp_path / f"{name}.py").write_text("def run():\n    return True\n")
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path=f"{name}.py", line_start=1, line_end=2, claim="Returns True."
            )
            for name in ("a", "b")
        ],
        summary="Both implementations return True.",
        unresolved_questions=[],
    )


def _build(tmp_path, candidate):
    validation = validate_candidate(tmp_path, candidate)
    assert not validation.issues
    return structure.build_investigation_structure(
        repo_path=tmp_path,
        objective="Find the return value.",
        candidate=candidate,
        valid_evidence_indices=frozenset(
            item.evidence_index for item in validation.evidence_pack
        ),
    )


@pytest.mark.parametrize(
    "limit,code",
    [
        ("MAX_CANDIDATE_STRUCTURE_RECORDS", "structure_record_limit_exceeded"),
        ("MAX_CANDIDATE_STRUCTURE_FILES", "structure_path_limit_exceeded"),
        ("MAX_GRAPH_SLICE_NODES", "structure_graph_limit_exceeded"),
        ("MAX_GRAPH_SLICE_SEEDS", "structure_seed_limit_exceeded"),
    ],
)
def test_source_only_resource_limits_are_visible_nonfatal_warnings(
    tmp_path, candidate, monkeypatch, limit, code
):
    monkeypatch.setattr(structure, limit, 1)
    result = _build(tmp_path, candidate)
    assert not result.issues
    assert code in {item.code for item in result.warnings}
    if result.artifact is not None:
        assert result.artifact.candidate_relation_assertion_ids == ()
        assert result.artifact.relevant_symbols == ()
        assert len(result.artifact.indexed_paths) <= 1


@pytest.mark.parametrize("claim_kind", ["symbol", "relation"])
def test_structural_claims_still_require_structure(
    tmp_path, candidate, monkeypatch, claim_kind
):
    monkeypatch.setattr(structure, "MAX_CANDIDATE_STRUCTURE_RECORDS", 1)
    if claim_kind == "symbol":
        candidate = candidate.model_copy(
            update={
                "evidence": [
                    candidate.evidence[0].model_copy(update={"symbol": "a.run"}),
                    candidate.evidence[1],
                ]
            }
        )
    else:
        candidate = candidate.model_copy(
            update={
                "relations": [
                    CandidateRelation(
                        source="a.run",
                        target="b.run",
                        relation=EvidenceRelationKind.REFERENCES,
                        supporting_evidence_indices=(0, 1),
                        explanation="Claims a structural relation.",
                    )
                ]
            }
        )
    result = _build(tmp_path, candidate)
    assert result.artifact is None
    assert "structure_record_limit_exceeded" in {item.code for item in result.issues}
    assert not result.warnings


def test_parse_failure_is_not_silently_downgraded(tmp_path, candidate):
    (tmp_path / "a.py").write_text("def broken(:\n    pass\n")
    result = _build(tmp_path, candidate)
    assert "structure_index_failed" in {item.code for item in result.issues}
    assert not result.warnings


def test_actual_default_record_limit_does_not_require_larger_budget(tmp_path):
    source = "".join(
        f"def function_{index}():\n    return True\n" for index in range(1000)
    )
    (tmp_path / "large.py").write_text(source)
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="large.py", line_start=1, line_end=2, claim="Returns True."
            )
        ],
        summary="The first function returns True.",
        unresolved_questions=[],
    )
    result = _build(tmp_path, candidate)
    assert result.artifact is None
    assert not result.issues
    assert [item.code for item in result.warnings] == [
        "structure_record_limit_exceeded"
    ]
    assert structure.MAX_CANDIDATE_STRUCTURE_RECORDS == 900


@pytest.mark.parametrize(
    "semantic_pass,invalid_source", [(True, False), (False, False), (True, True)]
)
def test_evaluator_persists_warning_without_bypassing_other_gates(
    tmp_path, candidate, monkeypatch, semantic_pass, invalid_source
):
    monkeypatch.setattr(structure, "MAX_CANDIDATE_STRUCTURE_RECORDS", 1)
    if invalid_source:
        candidate = candidate.model_copy(
            update={
                "evidence": [
                    candidate.evidence[0],
                    candidate.evidence[1].model_copy(update={"path": "../outside.py"}),
                ]
            }
        )
    assessment = SemanticAssessment(
        active_step_completed=semantic_pass,
        answers_objective=semantic_pass,
        evidence_supports_summary=semantic_pass,
        has_critical_omissions=not semantic_pass,
        missing_evidence=[],
        recommended_focus=[],
        summary="Scripted semantic assessment.",
    )
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(lambda _: assessment)
    update = create_evaluate_candidate_node(model)(
        {
            "repo_path": tmp_path,
            "objective": "Find the return value.",
            "investigation_plan": InvestigationPlan(
                steps=[
                    PlanStep(
                        id="inspect",
                        objective="Find the return value.",
                        expected_evidence=["A source range."],
                        status="active",
                    )
                ]
            ),
            "candidate_result": candidate,
            "context_budget_tokens": 16384,
        }
    )
    report = update["evaluation_report"]
    assert report.passed is (semantic_pass and not invalid_source)
    assert bool(report.deterministic_issues) is invalid_source
    assert {item.code for item in report.structural_warnings} == {
        "structure_record_limit_exceeded"
    }
    assert update["investigation_structure"] is None
    serializer = RepoCompassCheckpointSerializer()
    restored = serializer.loads_typed(serializer.dumps_typed(report))
    assert restored == report
    assert restored.structural_warnings == report.structural_warnings
