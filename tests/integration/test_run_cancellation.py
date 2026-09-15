"""Real checkpoint/raw/control stores; deterministic models, no provider requests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.result import FinalResult
from repocompass.memory.store import open_memory_store
from repocompass.storage.cancellation import (
    CancellationRequest,
    open_cancellation_store,
)
from repocompass.storage.raw_store import open_raw_run_store
from tests.model_doubles import (
    candidate_from_final_result,
    default_investigation_plan,
    with_scripted_planner,
)


def test_cancellation_waits_for_current_node_preserves_usage_and_survives_restart(
    tmp_path: Path,
):
    (tmp_path / "example.py").write_text("value = 1\n")
    entered, release = Event(), Event()
    explorer = MagicMock(spec=BaseChatModel)

    def forbidden(_):
        raise AssertionError("no model may run after cancellation boundary")

    explorer.bind_tools.return_value = RunnableLambda(forbidden)
    explorer.with_structured_output.return_value = RunnableLambda(forbidden)
    models = with_scripted_planner(explorer)

    def plan(_):
        entered.set()
        assert release.wait(5)
        return {
            "parsed": default_investigation_plan(),
            "parsing_error": None,
            "raw": AIMessage(
                content="plan",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            ),
        }

    models.planner.with_structured_output.return_value = RunnableLambda(plan)
    database = tmp_path / "checkpoints.sqlite"
    with (
        open_sqlite_checkpointer(database) as checkpointer,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_cancellation_store(tmp_path / "cancel.sqlite") as control,
        open_memory_store(tmp_path / "memory.sqlite") as memory,
    ):
        runner = AgentRunner(
            models,
            checkpointer=checkpointer,
            raw_run_store=raw,
            cancellation_store=control,
            memory_store=memory,
            run_id_factory=lambda: "run-1",
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runner.run,
                RunRequest(
                    repo_path=tmp_path,
                    objective="Inspect source",
                    max_tool_calls_per_step=3,
                    max_explorer_calls_per_step=2,
                ),
            )
            try:
                assert entered.wait(5)
                row = control.get("run-1")
                command = CancellationRequest(
                    run_id=row.run_id,
                    thread_id=row.thread_id,
                    repository_id=row.repository_id,
                    commit_sha=row.commit_sha,
                    request_id="cancel-1",
                )
                assert control.request(command).state == "cancel_requested"
                assert not future.done()
            finally:
                release.set()
            outcome = future.result(timeout=10)
        assert outcome.termination_status == "cancelled"
        assert outcome.step_stop_reason == "user_cancel_requested"
        assert outcome.cancellation_request_id == "cancel-1"
        assert outcome.final_result.evidence == []
        assert outcome.repository_consolidation.reason == "not_accepted"
        assert outcome.repository_consolidation.created_ids == ()
        assert control.get("run-1").state == "cancelled"
        assert raw.get_attempt("run-1").status.value == "completed"
        assert outcome.trace.events[-1].payload["cancellation_request_id"] == "cancel-1"
        assert outcome.trace.events[-1].payload["termination_status"] == "cancelled"
        assert [e.node for e in outcome.trace.events if e.node] == [
            "plan_investigation"
        ]
        assert len(raw.list_model_invocations("run-1")) == 1
        usage = raw.list_model_invocations("run-1")[0].usage
        assert usage.input_tokens == 100 and usage.output_tokens == 20

    models.planner.with_structured_output.return_value = RunnableLambda(forbidden)
    with (
        open_sqlite_checkpointer(database) as checkpointer,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_cancellation_store(tmp_path / "cancel.sqlite") as control,
    ):
        reader = AgentRunner(
            models,
            checkpointer=checkpointer,
            raw_run_store=raw,
            cancellation_store=control,
            run_id_factory=lambda: "resume-1",
        )
        view = reader.inspect_thread(ResumeRequest(thread_id=outcome.thread_id))
        assert view.next_nodes == ()
        assert view.termination_status == "cancelled"
        resumed = reader.resume(ResumeRequest(thread_id=outcome.thread_id))
        assert resumed.termination_status == "cancelled"
        assert resumed.cancellation_request_id is None
        assert control.get("resume-1").state == "finished"
        assert control.get("run-1").state == "cancelled"


def _request_cancellation(control, run_id="edge-run"):
    row = control.get(run_id)
    return control.request(
        CancellationRequest(
            run_id=row.run_id,
            thread_id=row.thread_id,
            repository_id=row.repository_id,
            commit_sha=row.commit_sha,
            request_id="edge-cancel",
        )
    )


@pytest.mark.parametrize(
    "boundary", ["before_first_model", "before_tools", "node_failure", "after_finalize"]
)
def test_cancellation_boundaries_do_not_run_tools_or_overwrite_terminal_work(
    tmp_path, monkeypatch, boundary
):
    (tmp_path / "example.py").write_text("value = 1\n")
    final = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py defines value."],
        summary="The value is defined in example.py.",
        unresolved_questions=[],
    )
    explorer = MagicMock(spec=BaseChatModel)
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final)
    )
    models = with_scripted_planner(explorer)
    with (
        open_sqlite_checkpointer(tmp_path / "checkpoint.sqlite") as checkpoint,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_cancellation_store(tmp_path / "control.sqlite") as control,
    ):

        def explore(_):
            if boundary == "before_tools":
                _request_cancellation(control)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "example.py"},
                            "id": "never-execute",
                        }
                    ],
                )
            if boundary == "node_failure":
                _request_cancellation(control)
                raise RuntimeError("controlled model failure")
            return AIMessage(content="Enough evidence.")

        explorer.bind_tools.return_value = RunnableLambda(explore)
        runner = AgentRunner(
            models,
            checkpointer=checkpoint,
            raw_run_store=raw,
            cancellation_store=control,
            run_id_factory=lambda: "edge-run",
        )
        if boundary == "before_first_model":
            register = control.register

            def register_and_cancel(identity):
                register(identity)
                _request_cancellation(control)

            monkeypatch.setattr(control, "register", register_and_cancel)
        if boundary == "after_finalize":
            finish = runner._raw_run_recorder.finish_attempt

            def request_before_finish(*args, **kwargs):
                assert (
                    kwargs["terminal_event"].payload["termination_status"] == "accepted"
                )
                _request_cancellation(control)
                return finish(*args, **kwargs)

            monkeypatch.setattr(
                runner._raw_run_recorder, "finish_attempt", request_before_finish
            )
        request = RunRequest(
            repo_path=tmp_path,
            objective="Inspect source",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=2,
        )
        if boundary == "node_failure":
            with pytest.raises(RuntimeError, match="controlled model failure"):
                runner.run(request)
            assert raw.get_attempt("edge-run").status.value == "failed"
            assert control.get("edge-run").state == "finished"
            assert control.get("edge-run").request_id == "edge-cancel"
            return
        outcome = runner.run(request)
        assert outcome.tool_audit_records == ()
        if boundary == "after_finalize":
            assert outcome.termination_status == "accepted"
            assert outcome.cancellation_request_id is None
            assert outcome.final_result.evidence == final.evidence
            assert control.get("edge-run").state == "finished"
            assert control.get("edge-run").request_id == "edge-cancel"
        else:
            assert outcome.termination_status == "cancelled"
            assert outcome.cancellation_request_id == "edge-cancel"
            assert outcome.final_result.evidence == []
            assert control.get("edge-run").state == "cancelled"
            nodes = [event.node for event in outcome.trace.events if event.node]
            assert nodes == (
                []
                if boundary == "before_first_model"
                else ["plan_investigation", "agent"]
            )
            assert len(raw.list_model_invocations("edge-run")) == (
                0 if boundary == "before_first_model" else 2
            )
        assert (
            runner.inspect_thread(ResumeRequest(thread_id=outcome.thread_id)).next_nodes
            == ()
        )
