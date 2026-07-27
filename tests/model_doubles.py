"""Shared deterministic model doubles for workflow tests."""

from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import SemanticAssessment
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.result import FinalResult
from repocompass.models import WorkflowModels


def default_investigation_plan() -> InvestigationPlan:
    """Return a valid repository-agnostic plan for graph tests."""

    return InvestigationPlan(
        task_summary="Investigate the requested repository behavior.",
        success_criteria=["Answer the objective with repository evidence."],
        steps=[
            PlanStep(
                id="inspect-repository",
                objective="Collect repository evidence for the objective.",
                expected_evidence=["Relevant repository locations and code evidence."],
                status="active",
            )
        ],
        active_step_id="inspect-repository",
    )


def candidate_from_final_result(final_result: FinalResult) -> CandidateResult:
    """Build a structured candidate that preserves an expected public result."""

    fallback_path = (
        final_result.relevant_files[0] if final_result.relevant_files else "unresolved"
    )
    evidence = [
        EvidenceItem(
            path=(
                final_result.relevant_files[index]
                if index < len(final_result.relevant_files)
                else fallback_path
            ),
            line_start=1,
            claim=claim,
        )
        for index, claim in enumerate(final_result.evidence)
    ]
    return CandidateResult(
        relevant_files=final_result.relevant_files,
        evidence=evidence,
        summary=final_result.summary,
        unresolved_questions=final_result.unresolved_questions,
    )


def default_semantic_assessment() -> SemanticAssessment:
    """Return a deterministic accepting semantic assessment for graph tests."""

    return SemanticAssessment(
        answers_objective=True,
        evidence_supports_summary=True,
        has_critical_omissions=False,
        missing_evidence=[],
        recommended_focus=[],
        summary="The candidate is semantically sufficient.",
    )


def with_scripted_planner(
    explorer: BaseChatModel,
    *,
    plan: InvestigationPlan | None = None,
    semantic_assessment: SemanticAssessment | None = None,
) -> WorkflowModels:
    """Return workflow roles with deterministic planner and evaluator doubles."""

    resolved_plan = plan or default_investigation_plan()
    planner = MagicMock(spec=BaseChatModel)
    planner.with_structured_output.return_value = RunnableLambda(
        lambda _: resolved_plan
    )
    evaluator = MagicMock(spec=BaseChatModel)
    evaluator.with_structured_output.return_value = RunnableLambda(
        lambda _: semantic_assessment or default_semantic_assessment()
    )
    return WorkflowModels(
        explorer=explorer,
        planner=planner,
        evaluator=evaluator,
    )
