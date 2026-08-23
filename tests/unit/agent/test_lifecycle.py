"""Contract tests for M2.1 thread and human-review lifecycle schemas."""

import pytest
from pydantic import ValidationError

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    ContinueRequest,
    ExecutionStatus,
    HumanDecision,
    HumanDecisionAction,
    HumanResponseRequest,
    HumanReviewPolicy,
    HumanReviewRequest,
    TurnRecord,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import RunOutcome, RunRequest
from repocompass.agent.state import RepoCompassState, create_initial_state
from repocompass.agent.trace import RunTrace


def _final_result() -> FinalResult:
    return FinalResult(
        relevant_files=["src/auth.py"],
        evidence=["src/auth.py:10 defines authenticate."],
        summary="Authentication is implemented by authenticate.",
        unresolved_questions=[],
    )


def _candidate_result() -> CandidateResult:
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path="src/auth.py",
                line_start=10,
                symbol="authenticate",
                claim="authenticate validates credentials.",
            )
        ],
        summary="Authentication is implemented by authenticate.",
        unresolved_questions=[],
    )


def _evaluation_report() -> EvaluationReport:
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=SemanticAssessment(
            active_step_completed=True,
            answers_objective=True,
            evidence_supports_summary=True,
            has_critical_omissions=False,
            missing_evidence=[],
            recommended_focus=[],
            summary="The candidate is ready for human review.",
        ),
    )


def _review_request() -> HumanReviewRequest:
    return HumanReviewRequest(
        turn_number=1,
        candidate_result=_candidate_result(),
        evaluation_report=_evaluation_report(),
    )


def test_lifecycle_contract_fields_are_minimal() -> None:
    assert set(ContinueRequest.model_fields) == {"thread_id", "prompt"}
    assert set(HumanDecision.model_fields) == {"action", "feedback"}
    assert set(HumanResponseRequest.model_fields) == {
        "thread_id",
        "interrupt_id",
        "decision",
    }
    assert set(HumanReviewRequest.model_fields) == {
        "turn_number",
        "candidate_result",
        "evaluation_report",
        "allowed_actions",
    }
    assert set(TurnRecord.model_fields) == {
        "turn_number",
        "objective",
        "final_result",
        "termination_status",
    }


def test_lifecycle_enum_vocabularies_are_closed() -> None:
    assert set(HumanReviewPolicy) == {
        HumanReviewPolicy.NEVER,
        HumanReviewPolicy.BEFORE_FINALIZE,
    }
    assert set(HumanDecisionAction) == {
        HumanDecisionAction.APPROVE,
        HumanDecisionAction.REVISE,
        HumanDecisionAction.CANCEL,
    }
    assert set(ExecutionStatus) == {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.INTERRUPTED,
    }


def test_continue_request_normalizes_public_text() -> None:
    request = ContinueRequest(
        thread_id="  thread-123  ",
        prompt="  Explain how checkpointing is connected.  ",
    )

    assert request.thread_id == "thread-123"
    assert request.prompt == "Explain how checkpointing is connected."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thread_id", ""),
        ("thread_id", "   "),
        ("prompt", ""),
        ("prompt", "   "),
    ],
)
def test_continue_request_rejects_empty_text(field: str, value: str) -> None:
    values = {
        "thread_id": "thread-123",
        "prompt": "Continue the investigation.",
        field: value,
    }

    with pytest.raises(ValidationError):
        ContinueRequest.model_validate(values)


def test_continue_request_rejects_repository_override() -> None:
    with pytest.raises(ValidationError):
        ContinueRequest.model_validate(
            {
                "thread_id": "thread-123",
                "prompt": "Continue the investigation.",
                "repo_path": "/another/repository",
            }
        )


def test_revise_decision_requires_non_empty_feedback() -> None:
    with pytest.raises(ValidationError, match="feedback"):
        HumanDecision(
            action=HumanDecisionAction.REVISE,
            feedback="   ",
        )

    decision = HumanDecision(
        action=HumanDecisionAction.REVISE,
        feedback="  Inspect the middleware registration path.  ",
    )
    assert decision.feedback == "Inspect the middleware registration path."


@pytest.mark.parametrize(
    "action",
    [HumanDecisionAction.APPROVE, HumanDecisionAction.CANCEL],
)
def test_terminal_human_decisions_do_not_require_feedback(
    action: HumanDecisionAction,
) -> None:
    decision = HumanDecision(action=action)

    assert decision.feedback is None


def test_human_response_binds_decision_to_thread_and_interrupt() -> None:
    response = HumanResponseRequest(
        thread_id="thread-123",
        interrupt_id="interrupt-456",
        decision=HumanDecision(action=HumanDecisionAction.APPROVE),
    )

    assert response.thread_id == "thread-123"
    assert response.interrupt_id == "interrupt-456"
    assert response.decision.action is HumanDecisionAction.APPROVE


def test_human_review_request_has_deterministic_allowed_actions() -> None:
    review = _review_request()

    assert review.allowed_actions == tuple(HumanDecisionAction)
    assert review.model_dump(mode="json")["allowed_actions"] == [
        "approve",
        "revise",
        "cancel",
    ]


@pytest.mark.parametrize("turn_number", [0, -1])
def test_human_review_request_rejects_invalid_turn_number(
    turn_number: int,
) -> None:
    with pytest.raises(ValidationError):
        HumanReviewRequest(
            turn_number=turn_number,
            candidate_result=_candidate_result(),
            evaluation_report=_evaluation_report(),
        )


def test_turn_record_preserves_one_completed_turn() -> None:
    record = TurnRecord(
        turn_number=1,
        objective="Find authentication.",
        final_result=_final_result(),
        termination_status="accepted",
    )

    assert record.turn_number == 1
    assert record.final_result == _final_result()


def test_new_run_defaults_to_automatic_review_policy(tmp_path) -> None:
    request = RunRequest(
        repo_path=tmp_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
    )
    state = create_initial_state(
        tmp_path,
        request.objective,
        request.max_tool_calls_per_step,
        request.max_explorer_calls_per_step,
    )

    assert request.human_review_policy is HumanReviewPolicy.NEVER
    assert state["human_review_policy"] is HumanReviewPolicy.NEVER
    assert state["turn_number"] == 1
    assert state["turn_history"] == []


def test_thread_identity_remains_checkpoint_config_not_graph_state() -> None:
    assert "thread_id" not in RepoCompassState.__annotations__
    assert "run_id" not in RepoCompassState.__annotations__
    assert "turn_number" in RepoCompassState.__annotations__
    assert "turn_history" in RepoCompassState.__annotations__


def test_completed_outcome_preserves_existing_m1_contract() -> None:
    outcome = RunOutcome(
        thread_id="thread-123",
        final_result=_final_result(),
        termination_status="accepted",
        step_stop_reason=None,
        trace=RunTrace("attempt-1"),
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.human_review_request is None
    assert outcome.interrupt_id is None
    assert outcome.final_result == _final_result()


def test_interrupted_outcome_requires_review_and_omits_terminal_result() -> None:
    review = _review_request()
    outcome = RunOutcome(
        thread_id="thread-123",
        execution_status=ExecutionStatus.INTERRUPTED,
        human_review_request=review,
        interrupt_id="interrupt-456",
        final_result=None,
        termination_status=None,
        step_stop_reason=None,
        trace=RunTrace("attempt-1"),
    )

    assert outcome.human_review_request == review
    assert outcome.interrupt_id == "interrupt-456"
    assert outcome.final_result is None
    assert outcome.termination_status is None


@pytest.mark.parametrize(
    (
        "execution_status",
        "final_result",
        "termination_status",
        "review_request",
        "interrupt_id",
    ),
    [
        (ExecutionStatus.COMPLETED, None, None, None, None),
        (
            ExecutionStatus.COMPLETED,
            _final_result(),
            "accepted",
            _review_request(),
            None,
        ),
        (ExecutionStatus.INTERRUPTED, None, None, None, "interrupt-1"),
        (
            ExecutionStatus.INTERRUPTED,
            _final_result(),
            "accepted",
            _review_request(),
            "interrupt-1",
        ),
    ],
)
def test_run_outcome_rejects_inconsistent_lifecycle_fields(
    execution_status: ExecutionStatus,
    final_result: FinalResult | None,
    termination_status: str | None,
    review_request: HumanReviewRequest | None,
    interrupt_id: str | None,
) -> None:
    with pytest.raises(ValueError):
        RunOutcome(
            thread_id="thread-123",
            execution_status=execution_status,
            human_review_request=review_request,
            interrupt_id=interrupt_id,
            final_result=final_result,
            termination_status=termination_status,
            step_stop_reason=None,
            trace=RunTrace("attempt-1"),
        )


def test_interrupted_outcome_rejects_missing_interrupt_id() -> None:
    with pytest.raises(ValueError, match="interrupt ID"):
        RunOutcome(
            thread_id="thread-123",
            execution_status=ExecutionStatus.INTERRUPTED,
            human_review_request=_review_request(),
            interrupt_id=None,
            final_result=None,
            termination_status=None,
            step_stop_reason=None,
            trace=RunTrace("attempt-1"),
        )
