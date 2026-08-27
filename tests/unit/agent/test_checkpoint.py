"""Unit tests for the local SQLite checkpoint lifecycle."""

import warnings
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
from repocompass.structure.contracts import create_symbol_record
from repocompass.structure.python_index import build_python_file_index


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


def test_checkpoint_round_trip_canonicalizes_symbol_decorators_without_warning(
    tmp_path: Path,
) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "@registry.register\ndef example():\n    return True\n",
        encoding="utf-8",
    )
    index = build_python_file_index(
        tmp_path,
        Path("example.py"),
        repository_id="a" * 64,
        commit_sha="1" * 40,
    )
    indexed_symbol = next(item for item in index.symbols if item.name == "example")
    symbol = create_symbol_record(
        name=indexed_symbol.name,
        qualified_name=indexed_symbol.qualified_name,
        kind=indexed_symbol.kind,
        parent_symbol_id=indexed_symbol.parent_symbol_id,
        definition=indexed_symbol.definition,
        body=indexed_symbol.body,
        decorators=("registry.register",),
        is_test=indexed_symbol.is_test,
    )
    serializer = RepoCompassCheckpointSerializer()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        restored = serializer.loads_typed(serializer.dumps_typed(symbol))
        restored = serializer.loads_typed(serializer.dumps_typed(restored))

    assert restored == symbol
    assert restored.decorators == ("registry.register",)
    assert not any(
        "Pydantic serializer warnings" in str(item.message) for item in captured
    )
