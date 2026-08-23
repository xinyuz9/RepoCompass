"""End-to-end tests for approve, revise, and cancel Human responses."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.lifecycle import (
    ContinueRequest,
    ExecutionStatus,
    HumanDecision,
    HumanDecisionAction,
    HumanResponseRequest,
    HumanReviewPolicy,
)
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.trace import TraceEventKind
from repocompass.models import WorkflowModels
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _result(path: str, summary: str) -> FinalResult:
    return FinalResult(
        relevant_files=[path],
        evidence=[f"{path}:1 supports the result."],
        summary=summary,
        unresolved_questions=[],
    )


def _models_for(result: FinalResult) -> WorkflowModels:
    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(result)
    )
    return with_scripted_planner(explorer)


def _models_that_must_not_run(result: FinalResult) -> WorkflowModels:
    def fail(_: object) -> object:
        raise AssertionError("approve/cancel must not invoke a model")

    models = _models_for(result)
    models.explorer.bind_tools.return_value = RunnableLambda(fail)
    models.explorer.with_structured_output.return_value = RunnableLambda(fail)
    models.planner.with_structured_output.return_value = RunnableLambda(fail)
    models.evaluator.with_structured_output.return_value = RunnableLambda(fail)
    return models


def _run_request(repo_path: Path) -> RunRequest:
    return RunRequest(
        repo_path=repo_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
        max_plan_revisions=2,
        human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
    )


def _response(
    *,
    thread_id: str,
    interrupt_id: str,
    action: HumanDecisionAction,
    feedback: str | None = None,
) -> HumanResponseRequest:
    return HumanResponseRequest(
        thread_id=thread_id,
        interrupt_id=interrupt_id,
        decision=HumanDecision(action=action, feedback=feedback),
    )


def test_approve_finalizes_without_model_calls_and_rejects_replay(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.py").write_text("first = True\n", encoding="utf-8")
    result = _result("first.py", "Approved result.")
    database_path = tmp_path / "state" / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        interrupted = AgentRunner(
            _models_for(result),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "approve-thread",
        ).run(_run_request(tmp_path))

    assert interrupted.interrupt_id is not None
    response = _response(
        thread_id=interrupted.thread_id,
        interrupt_id=interrupted.interrupt_id,
        action=HumanDecisionAction.APPROVE,
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            _models_that_must_not_run(result),
            checkpointer=checkpointer,
            run_id_factory=lambda: "approve-attempt",
        )
        wrong_thread_response = response.model_copy(
            update={"thread_id": "missing-thread"}
        )
        with pytest.raises(ValueError, match="does not exist"):
            runner.respond(wrong_thread_response)

        wrong_response = response.model_copy(update={"interrupt_id": "wrong-id"})
        with pytest.raises(ValueError, match="interrupt ID"):
            runner.respond(wrong_response)

        outcome = runner.respond(response)

        with pytest.raises(ValueError, match="not awaiting a Human response"):
            runner.respond(response)

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.termination_status == "accepted"
    assert outcome.final_result == result
    assert outcome.interrupt_id is None
    assert outcome.trace.events[0].payload["execution_mode"] == "respond"
    assert outcome.trace.events[0].payload["human_decision"] == {
        "action": "approve",
        "feedback_provided": False,
    }
    assert outcome.trace.events[-1].payload == {
        "thread_id": "approve-thread",
        "execution_mode": "respond",
        "termination_status": "accepted",
        "step_stop_reason": None,
    }
    assert [
        event.node
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ] == [
        "review_final_candidate",
        "mark_investigation_accepted",
        "finalize",
    ]


def test_cancel_publishes_best_effort_result_without_model_calls(
    tmp_path: Path,
) -> None:
    (tmp_path / "cancel.py").write_text("cancel = True\n", encoding="utf-8")
    result = _result("cancel.py", "Best effort before cancellation.")
    database_path = tmp_path / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            _models_for(result),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "cancel-thread",
        )
        interrupted = runner.run(_run_request(tmp_path))

    assert interrupted.interrupt_id is not None
    with open_sqlite_checkpointer(database_path) as checkpointer:
        outcome = AgentRunner(
            _models_that_must_not_run(result),
            checkpointer=checkpointer,
            run_id_factory=lambda: "cancel-attempt",
        ).respond(
            _response(
                thread_id=interrupted.thread_id,
                interrupt_id=interrupted.interrupt_id,
                action=HumanDecisionAction.CANCEL,
            )
        )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.termination_status == "cancelled"
    assert outcome.final_result == result
    node_names = [
        event.node
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert node_names == [
        "review_final_candidate",
        "mark_investigation_cancelled",
        "finalize",
    ]

    (tmp_path / "continued.py").write_text(
        "continued = True\n",
        encoding="utf-8",
    )
    continued_result = _result("continued.py", "Continued after cancellation.")
    continuation_models = _models_for(continued_result)
    with open_sqlite_checkpointer(database_path) as checkpointer:
        continued = AgentRunner(
            continuation_models,
            checkpointer=checkpointer,
            run_id_factory=lambda: "continued-attempt",
        ).continue_thread(
            ContinueRequest(
                thread_id=outcome.thread_id,
                prompt="Investigate a follow-up after cancellation.",
            )
        )
        snapshot = create_agent_graph(
            continuation_models,
            checkpointer=checkpointer,
        ).get_state({"configurable": {"thread_id": outcome.thread_id}})

    assert continued.execution_status is ExecutionStatus.INTERRUPTED
    assert continued.thread_id == outcome.thread_id
    assert continued.human_review_request is not None
    assert continued.human_review_request.candidate_result == (
        candidate_from_final_result(continued_result)
    )
    assert snapshot.values["turn_number"] == 2
    assert snapshot.values["turn_history"][0].termination_status == "cancelled"


def test_revise_passes_feedback_to_replanner_and_pauses_at_new_review(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.py").write_text("first = True\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("second = True\n", encoding="utf-8")
    first_result = _result("first.py", "Initial candidate.")
    second_result = _result("second.py", "Revised candidate.")
    database_path = tmp_path / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        interrupted = AgentRunner(
            _models_for(first_result),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "revise-thread",
        ).run(_run_request(tmp_path))

    assert interrupted.interrupt_id is not None
    revised_plan = InvestigationPlan(
        steps=[
            PlanStep(
                id="inspect-repository",
                objective="Inspect middleware registration.",
                expected_evidence=["Middleware registration source evidence."],
                status="active",
            )
        ]
    )
    captured_replanner_messages: list[list[object]] = []
    responder_models = _models_for(second_result)

    def revise_plan(messages: list[object]) -> InvestigationPlan:
        captured_replanner_messages.append(messages)
        return revised_plan

    responder_models.planner.with_structured_output.return_value = RunnableLambda(
        revise_plan
    )
    response = _response(
        thread_id=interrupted.thread_id,
        interrupt_id=interrupted.interrupt_id,
        action=HumanDecisionAction.REVISE,
        feedback="PRIVATE reviewer note: inspect middleware registration.",
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            responder_models,
            checkpointer=checkpointer,
            run_id_factory=lambda: "revise-attempt",
        )
        outcome = runner.respond(response)
        snapshot = create_agent_graph(
            responder_models,
            checkpointer=checkpointer,
        ).get_state({"configurable": {"thread_id": interrupted.thread_id}})

        with pytest.raises(ValueError, match="interrupt ID"):
            runner.respond(response)

    assert outcome.execution_status is ExecutionStatus.INTERRUPTED
    assert outcome.interrupt_id is not None
    assert outcome.interrupt_id != interrupted.interrupt_id
    assert outcome.human_review_request is not None
    assert outcome.human_review_request.candidate_result == candidate_from_final_result(
        second_result
    )
    assert snapshot.values["plan_revision_count"] == 1
    assert snapshot.values["investigation_plan"] == revised_plan
    assert len(captured_replanner_messages) == 1
    replanner_prompt = captured_replanner_messages[0][-1].content
    assert "PRIVATE reviewer note" in replanner_prompt
    node_names = [
        event.node
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert node_names[0] == "review_final_candidate"
    assert "revise_plan" in node_names
    assert "mark_investigation_accepted" not in node_names
    assert "finalize" not in node_names
    assert outcome.trace.events[0].payload["human_decision"] == {
        "action": "revise",
        "feedback_provided": True,
    }
    assert outcome.trace.events[-1].kind is TraceEventKind.RUN_INTERRUPTED
    assert outcome.trace.events[-1].payload == {
        "thread_id": "revise-thread",
        "execution_mode": "respond",
        "interrupt_id": outcome.interrupt_id,
        "turn_number": 1,
    }
    serialized_trace = "\n".join(
        event.model_dump_json() for event in outcome.trace.events
    )
    assert "PRIVATE reviewer note" not in serialized_trace
    review_event = next(
        event
        for event in outcome.trace.events
        if event.node == "review_final_candidate"
    )
    assert review_event.payload["human_feedback_provided"] is True
