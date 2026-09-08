"""Production runner events remain readable while executing and after failure."""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.trace import TraceEventKind
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.storage.raw_store import AttemptStatus, open_raw_run_store
from tests.model_doubles import with_scripted_planner
from tests.unit.agent.test_runner import _create_scripted_model
from repocompass.agent.result import FinalResult
from langchain_core.messages import AIMessage


def test_real_runner_public_journal_visible_during_call_and_after_exception(tmp_path):
    path = tmp_path / "raw.sqlite"
    with open_raw_run_store(path) as writer, open_raw_run_store(path) as reader:
        model = MagicMock(spec=BaseChatModel)

        def fail_after_reading_live_events(_):
            public = service.get_run("live-run")
            assert public.attempt.status is AttemptStatus.RUNNING
            assert [e.kind for e in public.trace_events] == [
                TraceEventKind.RUN_STARTED,
                TraceEventKind.NODE_COMPLETED,
            ]
            assert public.trace_events[1].node == "plan_investigation"
            assert str(tmp_path) not in public.model_dump_json()
            raise ValueError("api_key=do-not-expose provider-response secret")

        model.bind_tools.return_value = RunnableLambda(fail_after_reading_live_events)
        runner = AgentRunner(
            with_scripted_planner(model),
            checkpointer=InMemorySaver(),
            raw_run_store=writer,
            run_id_factory=lambda: "live-run",
            thread_id_factory=lambda: "live-thread",
        )
        service = DashboardQueryService(
            runner,
            raw_run_store=reader,
            trace_store=JsonlTraceStore(tmp_path / "no-jsonl"),
            repository_root=tmp_path,
        )
        with pytest.raises(ValueError, match="do-not-expose"):
            runner.run(
                RunRequest(
                    repo_path=tmp_path,
                    objective="Inspect source.",
                    max_tool_calls_per_step=2,
                    max_explorer_calls_per_step=2,
                )
            )
        public = service.get_run("live-run")
        assert public.attempt.status is AttemptStatus.FAILED
        assert public.trace_events[-1].kind is TraceEventKind.RUN_FAILED
        assert public.trace_events[-1].payload["phase"] == "graph_execution"
        assert (
            public.trace_events[-1].payload["last_completed_node"]
            == "plan_investigation"
        )
        assert "do-not-expose" not in public.model_dump_json()
        assert "provider-response" not in public.model_dump_json()
        assert not (tmp_path / "no-jsonl").exists()
    with open_raw_run_store(path) as store:
        assert store.list_trace_events("live-run")[-1].kind is TraceEventKind.RUN_FAILED


def test_success_journal_matches_returned_trace(tmp_path):
    (tmp_path / "example.py").write_text("value = True\n")
    final = FinalResult(
        relevant_files=["example.py"],
        evidence=["A definition exists."],
        summary="Defined here.",
        unresolved_questions=[],
    )
    model = _create_scripted_model([AIMessage(content="Enough evidence.")], final)
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        runner = AgentRunner(
            with_scripted_planner(model),
            checkpointer=InMemorySaver(),
            raw_run_store=store,
        )
        outcome = runner.run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect source.",
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=2,
            )
        )
        assert store.list_trace_events(outcome.trace.run_id) == outcome.trace.events
        assert store.get_attempt(outcome.trace.run_id).status is AttemptStatus.COMPLETED


def test_journal_failure_stops_before_model_and_preserves_original_error(
    tmp_path, monkeypatch
):
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        model = MagicMock(spec=BaseChatModel)

        def broken(_):
            raise OSError("original storage fault")

        monkeypatch.setattr(store, "append_trace_event", broken)
        runner = AgentRunner(
            with_scripted_planner(model),
            checkpointer=InMemorySaver(),
            raw_run_store=store,
            run_id_factory=lambda: "failed-journal",
        )
        with pytest.raises(OSError, match="original storage fault") as failure:
            runner.run(
                RunRequest(
                    repo_path=tmp_path,
                    objective="Inspect source.",
                    max_tool_calls_per_step=2,
                    max_explorer_calls_per_step=2,
                )
            )
        model.bind_tools.return_value.invoke.assert_not_called()
        assert store.get_attempt("failed-journal").status is AttemptStatus.FAILED
        assert store.list_trace_events("failed-journal") is None
        assert "Failed to persist" in failure.value.__notes__[0]
