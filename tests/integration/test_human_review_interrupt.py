"""Integration coverage for the persisted final-candidate review boundary."""

from pathlib import Path
from unittest.mock import MagicMock
from warnings import catch_warnings, simplefilter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.lifecycle import (
    ExecutionStatus,
    HumanDecisionAction,
    HumanReviewPolicy,
    decode_human_review_interrupt,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.trace import TraceEventKind
from repocompass.models import WorkflowModels
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


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
        raise AssertionError("reading an existing interrupt must not invoke a model")

    models = _models_for(result)
    models.explorer.bind_tools.return_value = RunnableLambda(fail)
    models.explorer.with_structured_output.return_value = RunnableLambda(fail)
    models.planner.with_structured_output.return_value = RunnableLambda(fail)
    models.evaluator.with_structured_output.return_value = RunnableLambda(fail)
    return models


def _result() -> FinalResult:
    return FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py:1 contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )


def _request(repo_path: Path) -> RunRequest:
    return RunRequest(
        repo_path=repo_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
        human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
    )


def test_runner_returns_structured_interrupt_before_terminal_nodes(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    result = _result()
    models = _models_for(result)
    database_path = tmp_path / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            models,
            checkpointer=checkpointer,
            run_id_factory=lambda: "review-attempt",
            thread_id_factory=lambda: "review-thread",
        )
        outcome = runner.run(_request(tmp_path))
        snapshot = create_agent_graph(models, checkpointer=checkpointer).get_state(
            {"configurable": {"thread_id": outcome.thread_id}}
        )

    assert outcome.execution_status is ExecutionStatus.INTERRUPTED
    assert outcome.thread_id == "review-thread"
    assert outcome.final_result is None
    assert outcome.termination_status is None
    assert outcome.interrupt_id is not None
    assert outcome.human_review_request is not None
    assert outcome.human_review_request.turn_number == 1
    assert outcome.human_review_request.candidate_result == candidate_from_final_result(
        result
    )
    assert outcome.human_review_request.allowed_actions == tuple(HumanDecisionAction)

    assert snapshot.next == ("review_final_candidate",)
    assert len(snapshot.interrupts) == 1
    assert outcome.interrupt_id == snapshot.interrupts[0].id
    assert (
        decode_human_review_interrupt(snapshot.interrupts[0].value)
        == outcome.human_review_request
    )
    assert snapshot.values["termination_status"] is None
    assert snapshot.values["final_result"] is None

    node_names = [
        event.node
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert node_names[-1] == "prepare_human_review"
    assert "mark_investigation_accepted" not in node_names
    assert "finalize" not in node_names
    assert all(
        event.kind is not TraceEventKind.RUN_FINISHED for event in outcome.trace.events
    )
    assert outcome.trace.events[-1].kind is TraceEventKind.RUN_INTERRUPTED
    assert outcome.trace.events[-1].payload == {
        "thread_id": "review-thread",
        "execution_mode": "start",
        "interrupt_id": outcome.interrupt_id,
        "turn_number": 1,
    }


def test_runner_reads_same_interrupt_after_reopening_sqlite_without_reexecution(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    result = _result()
    database_path = tmp_path / "state" / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        first_outcome = AgentRunner(
            _models_for(result),
            checkpointer=checkpointer,
            run_id_factory=lambda: "initial-attempt",
            thread_id_factory=lambda: "durable-review-thread",
        ).run(_request(tmp_path))

    with catch_warnings():
        simplefilter("error")
        with open_sqlite_checkpointer(database_path) as checkpointer:
            second_outcome = AgentRunner(
                _models_that_must_not_run(result),
                checkpointer=checkpointer,
                run_id_factory=lambda: "read-interrupt-attempt",
            ).resume(ResumeRequest(thread_id=first_outcome.thread_id))

    assert first_outcome.execution_status is ExecutionStatus.INTERRUPTED
    assert second_outcome.execution_status is ExecutionStatus.INTERRUPTED
    assert second_outcome.thread_id == "durable-review-thread"
    assert second_outcome.interrupt_id == first_outcome.interrupt_id
    assert second_outcome.human_review_request == first_outcome.human_review_request
    assert second_outcome.final_result is None
    assert second_outcome.termination_status is None
    assert second_outcome.trace.run_id == "read-interrupt-attempt"
    assert [event.kind for event in second_outcome.trace.events] == [
        TraceEventKind.RUN_STARTED,
        TraceEventKind.RUN_INTERRUPTED,
    ]
    assert second_outcome.trace.events[0].payload["execution_mode"] == "resume"
    assert second_outcome.trace.events[-1].payload == {
        "thread_id": "durable-review-thread",
        "execution_mode": "resume",
        "interrupt_id": first_outcome.interrupt_id,
        "turn_number": 1,
    }
