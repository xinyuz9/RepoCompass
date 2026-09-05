"""Role-specific Gather stage and model-context boundary tests."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.context import (
    ContextSection,
    ContextSourceKind,
    RawArtifactKind,
    VerificationStatus,
)
from repocompass.agent.context_gathering import (
    gather_candidate_context,
    gather_evaluator_context,
    gather_explorer_context,
    gather_planner_context,
    gather_replanner_context,
    render_model_context,
)
from repocompass.agent.evaluation import (
    EvaluationReport,
    EvidenceExcerpt,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.usage import ModelRole


def _step() -> PlanStep:
    return PlanStep(
        id="locate-auth",
        objective="Locate authentication.",
        expected_evidence=["A source file and exact line range."],
        status="active",
    )


def test_exact_replayed_tool_evidence_is_idempotent_but_new_provenance_is_retained(
    tmp_path,
):
    message = ToolMessage(
        content="src/auth.py",
        name="list_tree",
        tool_call_id="stable-call",
        status="success",
    )

    def gather(messages):
        return gather_explorer_context(
            repo_path=tmp_path,
            objective="Locate authentication.",
            active_step=_step(),
            remaining_tool_calls=2,
            remaining_explorer_calls=2,
            messages=messages,
            evaluation_report=None,
            budget_tokens=8192,
        )

    once = gather((message,))
    replay = gather((message, message.model_copy()))
    assert replay == once
    other = gather(
        (message, message.model_copy(update={"tool_call_id": "different-call"}))
    )
    assert len(other.pack.items) == len(once.pack.items) + 1


def test_conflicting_context_identity_is_not_silently_deduplicated():
    from repocompass.agent.context_gathering import _build_gathered_context
    from repocompass.agent.context_builder import ContextCandidateError

    gathered = gather_planner_context(
        objective="Find entrypoint.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
        budget_tokens=8192,
    )
    item = gathered.pack.items[0]
    conflict = item.model_copy(update={"priority": 1})
    with pytest.raises(ValueError, match="context candidate conflicts"):
        _build_gathered_context(
            candidates=(item, conflict),
            raw_artifacts=gathered.raw_artifacts,
            budget_tokens=8192,
            reserve_evidence=False,
            reserve_feedback=False,
        )
    with pytest.raises(ContextCandidateError, match="too many candidates"):
        _build_gathered_context(
            candidates=(item,) * 1025,
            raw_artifacts=gathered.raw_artifacts,
            budget_tokens=8192,
            reserve_evidence=False,
            reserve_feedback=False,
        )


def _plan() -> InvestigationPlan:
    return InvestigationPlan(steps=[_step()])


def _candidate() -> CandidateResult:
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                symbol="authenticate",
                claim="src/auth.py defines authenticate.",
            )
        ],
        summary="Authentication is defined in src/auth.py.",
        unresolved_questions=[],
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=SemanticAssessment(
            active_step_completed=False,
            answers_objective=False,
            evidence_supports_summary=True,
            has_critical_omissions=True,
            missing_evidence=["The caller is missing."],
            recommended_focus=["Search for authenticate callers."],
            summary="The entry point is found but the call path is incomplete.",
        ),
    )


def _records(rendered: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in rendered.splitlines() if line]


def test_planner_gather_is_isolated_and_content_addressed() -> None:
    gathered = gather_planner_context(
        objective="Find authentication.",
        max_tool_calls_per_step=7,
        max_explorer_calls_per_step=3,
        budget_tokens=4_096,
    )

    assert len(gathered.pack.items) == 1
    item = gathered.pack.items[0]
    assert item.section is ContextSection.OBJECTIVE
    assert item.source.kind is ContextSourceKind.OBJECTIVE
    assert len(gathered.raw_artifacts) == 1
    artifact = gathered.raw_artifacts[0]
    assert artifact.metadata.kind is RawArtifactKind.HUMAN_INPUT
    assert artifact.metadata.artifact_id == item.source.raw_artifact_id
    assert b"Find authentication." in artifact.content

    rendered = render_model_context(ModelRole.PLANNER, gathered.pack)
    assert "repo_path" not in rendered
    assert "Available tool calls per plan step: 7" in rendered
    assert "Available Explorer invocations per plan step: 3" in rendered


def test_explorer_gather_uses_successful_tool_evidence_not_message_history(
    tmp_path: Path,
) -> None:
    messages = (
        HumanMessage(content="stale human history must not be replayed"),
        AIMessage(content="private explorer narrative must not be replayed"),
        ToolMessage(
            content=(
                "File: src/auth.py\nLines: 1-2\n"
                "1 | def authenticate():\n2 |     return True"
            ),
            tool_call_id="call-read-auth",
            name="read_code",
            status="success",
        ),
        ToolMessage(
            content="provider leaked internal failure details",
            tool_call_id="call-failed",
            name="search_text",
            status="error",
        ),
    )

    gathered = gather_explorer_context(
        repo_path=tmp_path,
        objective="Explain authentication.",
        active_step=_step(),
        remaining_tool_calls=5,
        remaining_explorer_calls=2,
        messages=messages,
        evaluation_report=None,
        budget_tokens=8_192,
    )
    rendered = render_model_context(ModelRole.EXPLORER, gathered.pack)

    assert "Explain authentication." in rendered
    assert "def authenticate" in rendered
    assert "private explorer narrative" not in rendered
    assert "stale human history" not in rendered
    assert "provider leaked" not in rendered
    evidence = next(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.SOURCE_EXCERPT
    )
    assert evidence.verification_status is VerificationStatus.VERIFIED
    assert evidence.source.location is not None
    assert evidence.source.location.path == "src/auth.py"
    assert evidence.source.location.line_start == 1
    assert evidence.source.location.line_end == 2
    assert any(
        artifact.metadata.artifact_id == evidence.source.raw_artifact_id
        and artifact.metadata.kind is RawArtifactKind.SOURCE_EXCERPT
        for artifact in gathered.raw_artifacts
    )


def test_explorer_gather_makes_evaluator_request_mandatory_and_auditable(
    tmp_path: Path,
) -> None:
    report = _report()

    gathered = gather_explorer_context(
        repo_path=tmp_path,
        objective="Explain authentication.",
        active_step=_step(),
        remaining_tool_calls=3,
        remaining_explorer_calls=1,
        messages=(),
        evaluation_report=report,
        budget_tokens=8_192,
    )

    feedback = next(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.EVALUATOR_REQUEST
    )
    assert "The caller is missing." in feedback.content
    assert "Search for authenticate callers." in feedback.content
    decision = next(
        decision
        for decision in gathered.pack.build_report.decisions
        if decision.item_id == feedback.item_id
    )
    assert decision.reason.value == "evaluator_request"


def test_candidate_gather_compresses_large_traceable_tool_output(
    tmp_path: Path,
) -> None:
    messages = (
        ToolMessage(
            content=("File: src/large.py\nLines: 1-400\n" + "1 | value = True\n" * 400),
            tool_call_id="call-large-read",
            name="read_code",
            status="success",
        ),
    )

    gathered = gather_candidate_context(
        repo_path=tmp_path,
        objective="Explain the large module.",
        active_step=_step(),
        step_stop_reason="step_explorer_budget_exhausted",
        messages=messages,
        budget_tokens=4_096,
    )

    assert gathered.pack.estimated_tokens <= 4_096
    summary = next(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.SUMMARY
    )
    assert summary.verification_status is VerificationStatus.UNVERIFIED
    assert summary.source.raw_artifact_id is not None
    assert "[TRUNCATED:" in summary.content
    assert any(
        artifact.metadata.artifact_id == summary.source.raw_artifact_id
        and len(artifact.content) > len(summary.content.encode("utf-8"))
        for artifact in gathered.raw_artifacts
    )


def test_evaluator_gather_contains_only_validated_evaluation_inputs(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    evidence = (
        EvidenceExcerpt(
            evidence_index=0,
            path="src/auth.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
            claim="src/auth.py defines authenticate.",
            content="1 | def authenticate():\n2 |     return True",
        ),
    )
    issues = (
        ValidationIssue(
            code="example_issue",
            message="Example deterministic finding.",
            path="src/auth.py",
            evidence_index=0,
        ),
    )

    gathered = gather_evaluator_context(
        repo_path=tmp_path,
        objective="Find authentication.",
        active_step=_step(),
        candidate_result=candidate,
        deterministic_issues=issues,
        evidence_pack=evidence,
        budget_tokens=12_000,
    )
    rendered = render_model_context(ModelRole.EVALUATOR, gathered.pack)

    assert candidate.summary in rendered
    assert "example_issue" in rendered
    assert "def authenticate" in rendered
    assert {item.source.kind for item in gathered.pack.items} == {
        ContextSourceKind.OBJECTIVE,
        ContextSourceKind.CANDIDATE,
        ContextSourceKind.EVALUATOR_REQUEST,
        ContextSourceKind.SOURCE_EXCERPT,
    }
    source = next(
        item
        for item in gathered.pack.items
        if item.source.kind is ContextSourceKind.SOURCE_EXCERPT
    )
    assert source.verification_status is VerificationStatus.VERIFIED
    assert source.source.location is not None
    assert source.source.location.path == "src/auth.py"


def test_replanner_gather_keeps_required_feedback_and_human_input() -> None:
    gathered = gather_replanner_context(
        objective="Explain authentication.",
        investigation_plan=_plan(),
        candidate_result=_candidate(),
        evaluation_report=_report(),
        human_feedback="Also inspect the session middleware.",
        plan_revision_count=1,
        max_tool_calls_per_step=8,
        max_explorer_calls_per_step=4,
        budget_tokens=16_000,
    )
    rendered = render_model_context(ModelRole.REPLANNER, gathered.pack)

    assert "Also inspect the session middleware." in rendered
    assert "Search for authenticate callers." in rendered
    assert _candidate().summary in rendered
    assert all(item.pinned for item in gathered.pack.items)
    assert {artifact.metadata.kind for artifact in gathered.raw_artifacts} >= {
        RawArtifactKind.HUMAN_INPUT,
        RawArtifactKind.MODEL_OUTPUT,
    }
