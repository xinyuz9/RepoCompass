"""Unit tests for Runner orchestration and domain-trace collection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.agent.trace import TraceEventKind
from repocompass.storage.raw_store import open_raw_run_store
from repocompass.tools.contracts import ToolDecisionReason
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _create_scripted_model(
    agent_responses: list[AIMessage],
    final_result: FinalResult,
) -> MagicMock:
    responses = iter(agent_responses)

    def next_agent_response(_: object) -> AIMessage:
        try:
            return next(responses)
        except StopIteration:
            return AIMessage(content="No additional repository evidence is available.")

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(next_agent_response)
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return model


def test_runner_returns_result_and_minimal_lifecycle_trace(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    agent_response = AIMessage(content="Enough evidence is available.")
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model([agent_response], final_result)
    runner = AgentRunner(
        with_scripted_planner(model),
        checkpointer=InMemorySaver(),
        run_id_factory=lambda: "fixed-run",
        thread_id_factory=lambda: "fixed-thread",
    )
    request = RunRequest(
        repo_path=tmp_path,
        objective="  Inspect the repository.  ",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
    )

    outcome = runner.run(request)
    events = outcome.trace.events
    thread_view = runner.inspect_thread(ResumeRequest(thread_id=outcome.thread_id))

    assert outcome.final_result is not None
    assert (
        outcome.final_result.model_copy(
            update={"relevant_symbols": [], "structure": None}
        )
        == final_result
    )
    assert outcome.final_result.structure is not None
    assert outcome.termination_status == "accepted"
    assert outcome.step_stop_reason is None
    assert outcome.thread_id == "fixed-thread"
    assert outcome.trace.run_id == "fixed-run"
    assert thread_view.thread_id == "fixed-thread"
    assert thread_view.execution_state is ThreadExecutionState.COMPLETED
    assert thread_view.final_result == outcome.final_result
    assert thread_view.evaluation_report is not None
    assert thread_view.context_pack is not None
    assert thread_view.structure == outcome.final_result.structure
    assert thread_view.pending_review is None
    assert thread_view.interrupt_id is None
    assert thread_view.next_nodes == ()
    assert "messages" not in type(thread_view).model_fields
    assert "repo_path" not in type(thread_view).model_fields
    assert [event.kind for event in events] == [
        TraceEventKind.RUN_STARTED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.RUN_FINISHED,
    ]
    assert [event.node for event in events] == [
        None,
        "plan_investigation",
        "agent",
        "build_candidate",
        "evaluate_candidate",
        "decide_next_action",
        "prepare_human_review",
        "review_final_candidate",
        "mark_investigation_accepted",
        "finalize",
        None,
    ]
    assert events[0].payload == {
        "thread_id": "fixed-thread",
        "execution_mode": "start",
        "turn_number": 1,
        "repo_path": str(tmp_path.resolve()),
        "objective": "Inspect the repository.",
        "max_tool_calls_per_step": 3,
        "max_explorer_calls_per_step": 2,
        "max_no_progress_rounds": 3,
        "max_plan_revisions": 2,
        "context_budget_tokens": 32_768,
    }
    assert (
        events[1].payload["investigation_plan"]["steps"][0]["objective"]
        == "Collect repository evidence for the objective."
    )
    assert events[2].payload["step_explorer_call_count"] == 1
    assert events[3].payload["candidate_result"] == candidate_from_final_result(
        final_result
    ).model_dump(mode="json")
    assert events[4].payload["evaluation_report"]["passed"] is True
    assert events[4].payload["evaluation_report"]["deterministic_issues"] == []
    structure_trace = events[4].payload["investigation_structure"]
    assert structure_trace["indexed_paths"] == ["example.py"]
    assert structure_trace["graph_node_count"] >= 2
    assert structure_trace["reading_step_count"] >= 1
    assert "nodes" not in structure_trace
    assert "edges" not in structure_trace
    assert events[5].payload == {"next_action": "accepted"}
    assert events[6].payload == {
        "human_review_payload": None,
        "human_decision_action": None,
        "human_feedback_provided": False,
    }
    assert events[7].payload == {
        "human_review_policy": "never",
        "human_decision_action": "approve",
        "human_feedback_provided": False,
    }
    assert events[8].payload == {"termination_status": "accepted"}
    final_trace = events[9].payload["final_result"]
    assert final_trace["relevant_files"] == final_result.relevant_files
    assert final_trace["evidence"] == final_result.evidence
    assert final_trace["summary"] == final_result.summary
    assert final_trace["unresolved_questions"] == []
    assert final_trace["structure"] == structure_trace
    assert events[-1].payload == {
        "thread_id": "fixed-thread",
        "execution_mode": "start",
        "termination_status": "accepted",
        "step_stop_reason": None,
    }
    assert all(event.model_dump_json() for event in events)


def test_runner_uses_trusted_explicit_thread_id_without_factory(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    thread_id_factory = MagicMock(side_effect=AssertionError("must not run"))
    runner = AgentRunner(
        with_scripted_planner(
            _create_scripted_model(
                [AIMessage(content="Enough evidence is available.")],
                final_result,
            )
        ),
        checkpointer=InMemorySaver(),
        thread_id_factory=thread_id_factory,
    )

    outcome = runner.run(
        RunRequest(
            thread_id="service-assigned-thread",
            repo_path=tmp_path,
            objective="Inspect the repository.",
            max_tool_calls_per_step=2,
            max_explorer_calls_per_step=2,
        )
    )

    assert outcome.thread_id == "service-assigned-thread"
    thread_id_factory.assert_not_called()


def test_runner_omits_hidden_reasoning_and_secret_metadata_from_trace(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    agent_response = AIMessage(
        content="Enough evidence is available.",
        additional_kwargs={
            "reasoning_content": "private chain of thought",
            "safe_marker": "retained",
        },
        response_metadata={
            "authorization": "Bearer private-token",
            "model_name": "scripted-model",
        },
    )
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    runner = AgentRunner(
        with_scripted_planner(_create_scripted_model([agent_response], final_result)),
        checkpointer=InMemorySaver(),
    )

    outcome = runner.run(
        RunRequest(
            repo_path=tmp_path,
            objective="Inspect the repository.",
            max_tool_calls_per_step=2,
            max_explorer_calls_per_step=2,
        )
    )

    agent_event = next(event for event in outcome.trace.events if event.node == "agent")
    message = agent_event.payload["messages"][0]
    assert message["additional_kwargs"] == {"safe_marker": "retained"}
    assert message["response_metadata"] == {"model_name": "scripted-model"}
    serialized_trace = "\n".join(event.model_dump_json() for event in outcome.trace)
    assert "private chain of thought" not in serialized_trace
    assert "private-token" not in serialized_trace


def test_runner_inspect_rejects_unknown_thread(tmp_path: Path) -> None:
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="No result should be produced.",
        unresolved_questions=[],
    )
    runner = AgentRunner(
        with_scripted_planner(_create_scripted_model([], final_result)),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="Checkpoint thread does not exist"):
        runner.inspect_thread(ResumeRequest(thread_id="missing-thread"))


def test_runner_records_tool_loop_updates_without_extra_event_kinds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    tool_request = AIMessage(content="", tool_calls=[tree_call])
    final_response = AIMessage(content="The tree contains auth.py.")
    final_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["The list_tree result contains auth.py."],
        summary="auth.py is relevant.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [tool_request, final_response],
        final_result,
    )
    runner = AgentRunner(
        with_scripted_planner(model),
        checkpointer=InMemorySaver(),
        run_id_factory=lambda: "tool-run",
    )
    request = RunRequest(
        repo_path=tmp_path,
        objective="Find authentication.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
    )

    outcome = runner.run(request)
    node_events = [
        event for event in outcome.trace if event.kind is TraceEventKind.NODE_COMPLETED
    ]

    assert [event.node for event in node_events] == [
        "plan_investigation",
        "agent",
        "tools",
        "assess_progress",
        "agent",
        "build_candidate",
        "evaluate_candidate",
        "decide_next_action",
        "prepare_human_review",
        "review_final_candidate",
        "mark_investigation_accepted",
        "finalize",
    ]
    first_agent_message = node_events[1].payload["messages"][0]
    assert first_agent_message["tool_calls"][0]["name"] == "list_tree"
    tool_event = node_events[2]
    progress_event = node_events[3]
    assert tool_event.payload["step_tool_call_count"] == 1
    assert tool_event.payload["step_stop_reason"] is None
    assert len(tool_event.payload["tool_audit_records"]) == 1
    assert tool_event.payload["tool_audit_records"][0]["run_id"] == "tool-run"
    tool_message = tool_event.payload["messages"][0]
    assert tool_message["tool_call_id"] == "call-tree"
    assert "auth.py" in tool_message["content"]
    assert progress_event.payload["consecutive_no_progress_rounds"] == 0
    assert progress_event.payload["step_stop_reason"] is None
    assert (
        progress_event.payload["no_progress_decision"]["reason"] == "progress_observed"
    )
    assert outcome.termination_status == "accepted"
    assert outcome.step_stop_reason is None
    assert len(outcome.tool_audit_records) == 1
    audit = outcome.tool_audit_records[0]
    assert audit.run_id == "tool-run"
    assert audit.call_id == "call-tree"
    assert audit.decision.reason is ToolDecisionReason.ALLOWED
    assert "auth.py" not in audit.model_dump_json()
    assert outcome.no_progress_decision is not None
    assert outcome.no_progress_decision.reason.value == "progress_observed"
    assert outcome.consecutive_no_progress_rounds == 0
    assert "unregistered type" not in caplog.text.casefold()


def test_runner_rejects_resume_after_repository_commit_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepoCompass Tests"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "example.py"
    source.write_text("value = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "initial"],
        cwd=tmp_path,
        check=True,
    )
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The implementation is in example.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [AIMessage(content="Enough evidence is available.")],
        final_result,
    )
    with open_raw_run_store(tmp_path / "raw.sqlite") as raw_store:
        runner = AgentRunner(
            with_scripted_planner(model),
            checkpointer=InMemorySaver(),
            raw_run_store=raw_store,
            run_id_factory=lambda: "snapshot-start",
            thread_id_factory=lambda: "snapshot-thread",
        )
        outcome = runner.run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect the repository.",
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=2,
            )
        )
        source.write_text("value = False\n", encoding="utf-8")
        subprocess.run(["git", "add", "example.py"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "changed"],
            cwd=tmp_path,
            check=True,
        )
        with pytest.raises(ValueError, match="snapshot changed"):
            runner.resume(ResumeRequest(thread_id=outcome.thread_id))


def test_runner_preserves_step_explorer_budget_step_stop_reason(
    tmp_path: Path,
) -> None:
    (tmp_path / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    final_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py defines authenticate."],
        summary="Authentication is implemented in auth.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [AIMessage(content="", tool_calls=[tree_call])],
        final_result,
    )
    runner = AgentRunner(
        with_scripted_planner(model),
        checkpointer=InMemorySaver(),
        run_id_factory=lambda: "budget-run",
    )
    request = RunRequest(
        repo_path=tmp_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=1,
    )

    outcome = runner.run(request)

    assert outcome.termination_status == "accepted"
    assert outcome.step_stop_reason == "step_explorer_budget_exhausted"
    assert outcome.trace.events[-1].payload == {
        "thread_id": outcome.thread_id,
        "execution_mode": "start",
        "termination_status": "accepted",
        "step_stop_reason": "step_explorer_budget_exhausted",
    }


def test_runner_rejects_file_as_repository_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-repository"
    file_path.write_text("not a directory", encoding="utf-8")
    runner = AgentRunner(
        with_scripted_planner(MagicMock(spec=BaseChatModel)),
        checkpointer=InMemorySaver(),
    )
    request = RunRequest(
        repo_path=file_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=1,
    )

    with pytest.raises(ValueError, match="not a directory"):
        runner.run(request)


def test_runner_rejects_resume_for_unknown_thread() -> None:
    runner = AgentRunner(
        with_scripted_planner(MagicMock(spec=BaseChatModel)),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint thread does not exist: missing-thread",
    ):
        runner.resume(ResumeRequest(thread_id="missing-thread"))


def test_runner_reads_completed_result_when_resuming_finished_thread(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    final_result = FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )
    model = _create_scripted_model(
        [AIMessage(content="Enough evidence is available.")],
        final_result,
    )
    run_ids = iter(["start-attempt", "resume-attempt"])
    runner = AgentRunner(
        with_scripted_planner(model),
        checkpointer=InMemorySaver(),
        run_id_factory=lambda: next(run_ids),
        thread_id_factory=lambda: "persisted-thread",
    )
    request = RunRequest(
        repo_path=tmp_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
    )

    initial_outcome = runner.run(request)
    resumed_outcome = runner.resume(ResumeRequest(thread_id=initial_outcome.thread_id))

    assert resumed_outcome.thread_id == "persisted-thread"
    assert resumed_outcome.trace.run_id == "resume-attempt"
    assert resumed_outcome.final_result is not None
    assert (
        resumed_outcome.final_result.model_copy(
            update={"relevant_symbols": [], "structure": None}
        )
        == final_result
    )
    assert resumed_outcome.final_result.structure is not None
    assert resumed_outcome.termination_status == "accepted"
    assert [event.kind for event in resumed_outcome.trace.events] == [
        TraceEventKind.RUN_STARTED,
        TraceEventKind.RUN_FINISHED,
    ]
    assert resumed_outcome.trace.events[0].payload["execution_mode"] == "resume"
    assert resumed_outcome.trace.events[0].payload["turn_number"] == 1
    assert resumed_outcome.trace.events[-1].payload == {
        "thread_id": "persisted-thread",
        "execution_mode": "resume",
        "termination_status": "accepted",
        "step_stop_reason": None,
    }


@pytest.mark.parametrize("value", ["0", "1.0", "-1", 1])
def test_run_request_rejects_noncanonical_attempt_cost_budget(value: object) -> None:
    with pytest.raises(ValidationError, match="cost budget"):
        RunRequest(
            repo_path=Path("."),
            objective="Inspect repository.",
            max_tool_calls_per_step=2,
            max_explorer_calls_per_step=2,
            max_attempt_cost_usd=value,  # type: ignore[arg-type]
        )


def test_runner_rejects_cost_budget_without_raw_store_and_pricing(
    tmp_path: Path,
) -> None:
    runner = AgentRunner(
        with_scripted_planner(MagicMock(spec=BaseChatModel)),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="raw store and pricing table"):
        runner.run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect repository.",
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=2,
                max_attempt_cost_usd="0.1",
            )
        )
