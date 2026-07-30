"""Integration test for SQLite recovery across checkpointer connections."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.trace import TraceEventKind
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
        event for event in outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert outcome.final_result == final_result
    assert outcome.thread_id == "recoverable-thread"
    assert node_events[0].node == "agent"
    assert all(event.node != "plan_investigation" for event in node_events)
