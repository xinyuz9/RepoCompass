"""Integration test for SQLite recovery across checkpointer connections."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.trace import TraceEventKind
from repocompass.tools.contracts import NoProgressReason, ToolDecisionReason
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def test_runner_resumes_unfinished_thread_after_reopening_sqlite(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    database_path = tmp_path / "state" / "checkpoints.sqlite"
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )

    crashing_explorer = MagicMock(spec=BaseChatModel)
    crashing_explorer.bind_tools.return_value = RunnableLambda(
        lambda _: (_ for _ in ()).throw(RuntimeError("simulated process failure"))
    )
    crashing_explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            with_scripted_planner(crashing_explorer),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "recoverable-thread",
        )
        request = RunRequest(
            repo_path=tmp_path,
            objective="Inspect the repository.",
            max_tool_calls_per_step=2,
            max_explorer_calls_per_step=2,
        )

        with pytest.raises(RuntimeError, match="simulated process failure"):
            runner.run(request)

    resumed_explorer = MagicMock(spec=BaseChatModel)
    resumed_explorer.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    resumed_explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    resumed_models = with_scripted_planner(resumed_explorer)
    resumed_models.planner.with_structured_output.return_value = RunnableLambda(
        lambda _: (_ for _ in ()).throw(
            AssertionError("resume must not execute the completed planner node")
        )
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            resumed_models,
            checkpointer=checkpointer,
            run_id_factory=lambda: "resume-attempt",
        )
        outcome = runner.resume(ResumeRequest(thread_id="recoverable-thread"))

    node_events = [
        event
        for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert outcome.final_result == final_result
    assert outcome.thread_id == "recoverable-thread"
    assert node_events[0].node == "agent"
    assert all(event.node != "plan_investigation" for event in node_events)


def test_runner_resume_preserves_no_progress_counter_across_sqlite_reopen(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    database_path = tmp_path / "state" / "no-progress.sqlite"
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    first_invalid_request = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_depth": "not-an-int"},
                id="invalid-before-crash",
                type="tool_call",
            )
        ],
    )
    initial_explorer_calls = 0

    def crash_after_first_stalled_round(_: object) -> AIMessage:
        nonlocal initial_explorer_calls
        initial_explorer_calls += 1
        if initial_explorer_calls == 1:
            return first_invalid_request
        raise RuntimeError("simulated failure after one stalled round")

    crashing_explorer = MagicMock(spec=BaseChatModel)
    crashing_explorer.bind_tools.return_value = RunnableLambda(
        crash_after_first_stalled_round
    )
    crashing_explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            with_scripted_planner(crashing_explorer),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "no-progress-thread",
        )
        request = RunRequest(
            repo_path=tmp_path,
            objective="Inspect the repository.",
            max_tool_calls_per_step=4,
            max_explorer_calls_per_step=4,
            max_no_progress_rounds=2,
        )

        with pytest.raises(
            RuntimeError,
            match="simulated failure after one stalled round",
        ):
            runner.run(request)

    resumed_invalid_request = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="list_tree",
                args={"max_depth": "still-not-an-int"},
                id="invalid-after-resume",
                type="tool_call",
            )
        ],
    )
    resumed_explorer_calls = 0

    def return_one_stalled_round(_: object) -> AIMessage:
        nonlocal resumed_explorer_calls
        resumed_explorer_calls += 1
        return resumed_invalid_request

    resumed_explorer = MagicMock(spec=BaseChatModel)
    resumed_explorer.bind_tools.return_value = RunnableLambda(return_one_stalled_round)
    resumed_explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    resumed_models = with_scripted_planner(resumed_explorer)
    resumed_models.planner.with_structured_output.return_value = RunnableLambda(
        lambda _: (_ for _ in ()).throw(
            AssertionError("resume must not execute the completed planner node")
        )
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            resumed_models,
            checkpointer=checkpointer,
            run_id_factory=lambda: "no-progress-resume-attempt",
        )
        outcome = runner.resume(ResumeRequest(thread_id="no-progress-thread"))

    assert outcome.final_result == final_result
    assert outcome.step_stop_reason == "step_no_progress"
    assert outcome.consecutive_no_progress_rounds == 2
    assert outcome.no_progress_decision is not None
    assert outcome.no_progress_decision.reason is NoProgressReason.NO_PROGRESS_LIMIT
    assert resumed_explorer_calls == 1
    assert [audit.decision.reason for audit in outcome.tool_audit_records] == [
        ToolDecisionReason.INVALID_ARGUMENTS
    ]
