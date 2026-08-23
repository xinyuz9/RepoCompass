"""Unit tests for the local SQLite checkpoint lifecycle."""

from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.checkpoint import (
    RepoCompassCheckpointSerializer,
    open_sqlite_checkpointer,
)
from repocompass.agent.evaluation import (
    EvaluationReport,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep


def test_open_sqlite_checkpointer_creates_parent_and_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        checkpointer.setup()
        assert database_path.exists()

    assert database_path.is_file()


@pytest.mark.parametrize(
    "serializer",
    [
        JsonPlusSerializer(
            allowed_msgpack_modules=(
                CandidateResult,
                EvidenceItem,
                EvaluationReport,
                SemanticAssessment,
                ValidationIssue,
                InvestigationPlan,
                PlanStep,
            )
        ),
        RepoCompassCheckpointSerializer(),
    ],
    ids=["langgraph-narrow-allowlist", "repocompass-sqlite"],
)
def test_checkpoint_serializer_rehydrates_strict_models_for_repeated_writes(
    serializer: JsonPlusSerializer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="src/example.py",
                line_start=1,
                line_end=2,
                symbol="example",
                claim="The example is registered.",
            )
        ],
        summary="Example summary.",
        unresolved_questions=[],
    )
    evaluation_report = EvaluationReport(
        deterministic_issues=[
            ValidationIssue(
                code="example_issue",
                message="The example requires deterministic review.",
                path="src/example.py",
                evidence_index=0,
            )
        ],
        semantic_assessment=SemanticAssessment(
            active_step_completed=True,
            answers_objective=True,
            evidence_supports_summary=True,
            has_critical_omissions=False,
            missing_evidence=[],
            recommended_focus=[],
            summary="The candidate is supported.",
        ),
    )
    investigation_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="inspect-example",
                objective="Inspect the example.",
                expected_evidence=["Example registration."],
                status="active",
            )
        ]
    )
    expected = {
        "candidate_result": candidate,
        "evaluation_report": evaluation_report,
        "investigation_plan": investigation_plan,
    }
    current: object = expected

    for _ in range(3):
        current = serializer.loads_typed(serializer.dumps_typed(current))

    assert isinstance(current, dict)
    assert current == expected
    restored_candidate = current["candidate_result"]
    restored_report = current["evaluation_report"]
    restored_plan = current["investigation_plan"]
    assert isinstance(restored_candidate, CandidateResult)
    assert isinstance(restored_candidate.evidence[0], EvidenceItem)
    assert isinstance(restored_report, EvaluationReport)
    assert isinstance(
        restored_report.semantic_assessment,
        SemanticAssessment,
    )
    assert isinstance(restored_plan, InvestigationPlan)
    assert isinstance(restored_plan.active_step, PlanStep)
    assert "Blocked deserialization" not in caplog.text
