"""Unit tests for deterministic M1.5 control and plan transitions."""

from typing import get_type_hints
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.context import ContextPack
from repocompass.agent.evaluation import (
    EvaluationReport,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.nodes import (
    ControllerInput,
    PlanAdvanceInput,
    ReplannerInput,
    advance_plan_step,
    create_revise_plan_node,
    decide_next_action,
    mark_investigation_accepted,
    mark_investigation_exhausted,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep, PlanProposal


def _plan(*, include_pending: bool = True) -> InvestigationPlan:
    steps = [
        PlanStep(
            id="locate",
            objective="Locate the relevant implementation.",
            expected_evidence=["A repository-relative source location."],
            status="active",
        )
    ]
    if include_pending:
        steps.append(
            PlanStep(
                id="explain",
                objective="Explain the verified implementation.",
                expected_evidence=["Evidence-backed behavior and relationships."],
                status="pending",
            )
        )
    return InvestigationPlan(steps=steps)


def _report(
    *,
    step_completed: bool,
    passed: bool = False,
) -> EvaluationReport:
    assessment = SemanticAssessment(
        active_step_completed=step_completed,
        answers_objective=passed,
        evidence_supports_summary=passed,
        has_critical_omissions=not passed,
        missing_evidence=[] if passed else ["More repository evidence is required."],
        recommended_focus=[] if passed else ["Inspect the missing implementation."],
        summary="Accepted." if passed else "Further investigation is required.",
    )
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=assessment,
    )


def _controller_state(
    *,
    report: EvaluationReport,
    plan: InvestigationPlan | None = None,
    step_stop_reason: str | None = None,
    explorer_calls: int = 1,
    max_explorer_calls: int = 3,
    revisions: int = 0,
    max_revisions: int = 2,
) -> ControllerInput:
    return {
        "investigation_plan": plan or _plan(),
        "evaluation_report": report,
        "step_stop_reason": step_stop_reason,
        "step_explorer_call_count": explorer_calls,
        "max_explorer_calls_per_step": max_explorer_calls,
        "plan_revision_count": revisions,
        "max_plan_revisions": max_revisions,
    }


@pytest.mark.parametrize(
    ("state", "expected_action"),
    [
        pytest.param(
            _controller_state(
                report=_report(step_completed=True, passed=True),
                plan=_plan(include_pending=False),
            ),
            "accepted",
            id="accepted-final-step-finalizes",
        ),
        pytest.param(
            _controller_state(report=_report(step_completed=True, passed=True)),
            "advance_step",
            id="overall-answer-cannot-bypass-pending-step",
        ),
        pytest.param(
            _controller_state(report=_report(step_completed=False)),
            "continue_step",
            id="incomplete-step-continues-with-budget",
        ),
        pytest.param(
            _controller_state(
                report=_report(step_completed=False, passed=True),
                plan=_plan(include_pending=False),
            ),
            "continue_step",
            id="overall-answer-cannot-bypass-incomplete-active-step",
        ),
        pytest.param(
            _controller_state(report=_report(step_completed=True)),
            "advance_step",
            id="completed-step-advances",
        ),
        pytest.param(
            _controller_state(
                report=_report(step_completed=False),
                step_stop_reason="step_explorer_budget_exhausted",
            ),
            "revise_plan",
            id="blocked-step-revises",
        ),
        pytest.param(
            _controller_state(
                report=_report(step_completed=True),
                plan=_plan(include_pending=False),
            ),
            "revise_plan",
            id="completed-plan-with-failed-answer-revises",
        ),
        pytest.param(
            _controller_state(
                report=_report(step_completed=False),
                step_stop_reason="step_explorer_budget_exhausted",
                revisions=2,
                max_revisions=2,
            ),
            "exhausted",
            id="revision-limit-exhausts",
        ),
    ],
)
def test_controller_decision_table(
    state: ControllerInput,
    expected_action: str,
) -> None:
    update = decide_next_action(state)

    assert update == {"next_action": expected_action}


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize(
    "code", ["path_not_found", "path_not_file", "relation_structure_unavailable"]
)
@pytest.mark.parametrize("budget", ["available", "revise", "exhausted"])
def test_deterministic_defects_override_semantic_step_approval(pending, code, budget):
    report = _report(step_completed=True, passed=True).model_copy(
        update={
            "deterministic_issues": [
                ValidationIssue(code=code, message="Invalid source.")
            ]
        }
    )
    plan = _plan(include_pending=pending)
    before = plan.model_dump_json()
    state = _controller_state(
        report=report,
        plan=plan,
        step_stop_reason=None
        if budget == "available"
        else "step_tool_call_budget_exhausted",
        revisions=2 if budget == "exhausted" else 0,
    )
    expected = {
        "available": "continue_step",
        "revise": "revise_plan",
        "exhausted": "exhausted",
    }[budget]
    assert decide_next_action(state) == {"next_action": expected}
    assert not report.passed
    assert plan.model_dump_json() == before


def test_controller_requires_plan_and_evaluation() -> None:
    missing_plan = _controller_state(report=_report(step_completed=False))
    missing_plan["investigation_plan"] = None
    with pytest.raises(ValueError, match="InvestigationPlan"):
        decide_next_action(missing_plan)

    missing_report = _controller_state(report=_report(step_completed=False))
    missing_report["evaluation_report"] = None
    with pytest.raises(ValueError, match="EvaluationReport"):
        decide_next_action(missing_report)


def test_terminal_markers_separate_run_status_from_step_stop_reason() -> None:
    assert mark_investigation_accepted({}) == {"termination_status": "accepted"}
    assert mark_investigation_exhausted({}) == {"termination_status": "exhausted"}


def test_advance_plan_step_completes_active_and_resets_step_state() -> None:
    original_plan = _plan()
    state: PlanAdvanceInput = {"investigation_plan": original_plan}

    update = advance_plan_step(state)

    revised_plan = update["investigation_plan"]
    assert revised_plan is not original_plan
    assert [step.status for step in original_plan.steps] == ["active", "pending"]
    assert revised_plan.active_step.id == "explain"
    assert [step.status for step in revised_plan.steps] == ["completed", "active"]
    assert update == {
        "investigation_plan": revised_plan,
        "step_tool_call_count": 0,
        "step_explorer_call_count": 0,
        "last_tool_batch_audits": [],
        "step_tool_audit_sequence_floor": None,
        "progress_snapshot": None,
        "consecutive_no_progress_rounds": 0,
        "step_stop_reason": None,
        "candidate_result": None,
        "evaluation_report": None,
        "investigation_structure": None,
        "next_action": None,
    }


def test_advance_plan_step_requires_a_pending_step() -> None:
    state: PlanAdvanceInput = {
        "investigation_plan": _plan(include_pending=False),
    }

    with pytest.raises(ValueError, match="pending"):
        advance_plan_step(state)


def test_advance_plan_step_requires_an_investigation_plan() -> None:
    state: PlanAdvanceInput = {"investigation_plan": None}

    with pytest.raises(ValueError, match="InvestigationPlan"):
        advance_plan_step(state)


def test_replanner_declares_narrow_input_schema() -> None:
    model = MagicMock(spec=BaseChatModel)

    node = create_revise_plan_node(model)

    assert get_type_hints(node)["state"] is ReplannerInput


def test_replanner_preserves_completed_steps_and_resets_execution_state() -> None:
    completed = PlanStep(
        id="locate",
        objective="Locate the relevant implementation.",
        expected_evidence=["A repository-relative source location."],
        status="completed",
    )
    current_plan = InvestigationPlan(
        steps=[
            completed,
            PlanStep(
                id="explain",
                objective="Explain the implementation.",
                expected_evidence=["Verified behavior."],
                status="active",
            ),
        ],
    )
    revised_plan = InvestigationPlan(
        steps=[
            completed,
            PlanStep(
                id="trace-callers",
                objective="Trace callers of the verified implementation.",
                expected_evidence=["Verified caller locations."],
                status="active",
            ),
        ],
    )
    candidate = CandidateResult(
        evidence=[],
        summary="The current plan did not collect enough evidence.",
        unresolved_questions=["Which callers use the implementation?"],
    )
    report = _report(step_completed=False)
    received_messages: list[list[object]] = []

    def return_revised_plan(messages: list[object]) -> InvestigationPlan:
        received_messages.append(list(messages))
        return revised_plan

    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(return_revised_plan)
    state: ReplannerInput = {
        "objective": "Explain the implementation.",
        "investigation_plan": current_plan,
        "candidate_result": candidate,
        "evaluation_report": report,
        "human_feedback": None,
        "max_tool_calls_per_step": 6,
        "max_explorer_calls_per_step": 3,
        "plan_revision_count": 0,
        "max_plan_revisions": 2,
        "context_budget_tokens": 16_384,
        "context_build_count": 0,
    }

    update = create_revise_plan_node(model)(state)

    assert isinstance(update["active_context_pack"], ContextPack)
    assert update["context_build_count"] == 1
    assert update == {
        "investigation_plan": revised_plan,
        "step_tool_call_count": 0,
        "step_explorer_call_count": 0,
        "last_tool_batch_audits": [],
        "step_tool_audit_sequence_floor": None,
        "progress_snapshot": None,
        "consecutive_no_progress_rounds": 0,
        "plan_revision_count": 1,
        "step_stop_reason": None,
        "candidate_result": None,
        "evaluation_report": None,
        "investigation_structure": None,
        "next_action": None,
        "human_review_payload": None,
        "human_decision_action": None,
        "human_feedback": None,
        "active_context_pack": update["active_context_pack"],
        "context_build_count": 1,
    }
    assert len(received_messages[0]) == 2
    prompt = str(received_messages[0][1].content)
    assert "None provided." not in prompt
    assert "role=replanner" in prompt
    assert current_plan.active_step.objective in prompt
    assert candidate.summary in prompt
    assert report.semantic_assessment.summary in prompt


def test_replanner_rejects_a_revision_that_changes_completed_steps() -> None:
    completed = PlanStep(
        id="locate",
        objective="Locate the implementation.",
        expected_evidence=["A source location."],
        status="completed",
    )
    current_plan = InvestigationPlan(
        steps=[
            completed,
            PlanStep(
                id="explain",
                objective="Explain the implementation.",
                expected_evidence=["Verified behavior."],
                status="active",
            ),
        ],
    )
    invalid_revision = InvestigationPlan(
        steps=[
            completed.model_copy(update={"objective": "Changed completed work."}),
            PlanStep(
                id="retry",
                objective="Retry the investigation.",
                expected_evidence=["Missing evidence."],
                status="active",
            ),
        ],
    )
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: invalid_revision
    )
    state: ReplannerInput = {
        "objective": "Explain the implementation.",
        "investigation_plan": current_plan,
        "candidate_result": CandidateResult(
            evidence=[],
            summary="Insufficient evidence.",
            unresolved_questions=["What implements the behavior?"],
        ),
        "evaluation_report": _report(step_completed=False),
        "human_feedback": None,
        "max_tool_calls_per_step": 4,
        "max_explorer_calls_per_step": 3,
        "plan_revision_count": 0,
        "max_plan_revisions": 2,
    }

    update = create_revise_plan_node(model)(state)
    assert update["termination_status"] == "exhausted"
    assert update["step_stop_reason"] == "invalid_plan_revision"
    assert "investigation_plan" not in update
    assert state["investigation_plan"] == current_plan


def test_replanner_rejects_exhausted_revision_budget_before_model_call() -> None:
    model = MagicMock(spec=BaseChatModel)
    state: ReplannerInput = {
        "objective": "Explain the implementation.",
        "investigation_plan": _plan(),
        "candidate_result": CandidateResult(
            evidence=[],
            summary="Insufficient evidence.",
            unresolved_questions=["What implements the behavior?"],
        ),
        "evaluation_report": _report(step_completed=False),
        "human_feedback": None,
        "max_tool_calls_per_step": 4,
        "max_explorer_calls_per_step": 3,
        "plan_revision_count": 2,
        "max_plan_revisions": 2,
    }

    with pytest.raises(ValueError, match="budget is exhausted"):
        create_revise_plan_node(model)(state)

    model.with_structured_output.assert_called_once_with(
        PlanProposal,
        method="function_calling",
        include_raw=True,
    )
    model.with_structured_output.return_value.invoke.assert_not_called()
