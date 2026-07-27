"""Shared deterministic model doubles for workflow tests."""

from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.contracts import InvestigationPlan, PlanStep
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


def with_scripted_planner(
    explorer: BaseChatModel,
    *,
    plan: InvestigationPlan | None = None,
) -> WorkflowModels:
    """Return workflow roles with an isolated deterministic planner."""

    resolved_plan = plan or default_investigation_plan()
    planner = MagicMock(spec=BaseChatModel)
    planner.with_structured_output.return_value = RunnableLambda(
        lambda _: resolved_plan
    )
    return WorkflowModels(
        explorer=explorer,
        planner=planner,
        evaluator=explorer,
    )
