"""Unit tests for Runner orchestration and domain-trace collection."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableLambda

from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.trace import TraceEventKind


def _create_scripted_model(
    agent_responses: list[AIMessage],
    final_result: FinalResult,
) -> MagicMock:
    responses = iter(agent_responses)
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(lambda _: next(responses))
    model.with_structured_output.return_value = RunnableLambda(
        lambda _: final_result
    )
    return model


def test_runner_returns_result_and_minimal_lifecycle_trace(
    tmp_path: Path,
) -> None:
    agent_response = AIMessage(content="Enough evidence is available.")
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="No repository files were needed.",
        unresolved_questions=[],
    )
    model = _create_scripted_model([agent_response], final_result)
    runner = AgentRunner(model, run_id_factory=lambda: "fixed-run")
    request = RunRequest(
        repo_path=tmp_path,
        objective="  Inspect the repository.  ",
        max_tool_calls=3,
        max_iterations=2,
    )

    outcome = runner.run(request)
    events = outcome.trace.events

    assert outcome.final_result == final_result
    assert outcome.stop_reason is None
    assert outcome.trace.run_id == "fixed-run"
    assert [event.kind for event in events] == [
        TraceEventKind.RUN_STARTED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.NODE_COMPLETED,
        TraceEventKind.RUN_FINISHED,
    ]
    assert [event.node for event in events] == [
        None,
        "agent",
        "finalize",
        None,
    ]
    assert events[0].payload == {
        "repo_path": str(tmp_path.resolve()),
        "objective": "Inspect the repository.",
        "max_tool_calls": 3,
        "max_iterations": 2,
    }
    assert events[1].payload["iteration_count"] == 1
    assert events[2].payload["final_result"] == final_result.model_dump(
        mode="json"
    )
    assert events[-1].payload == {"stop_reason": None}
    assert all(event.model_dump_json() for event in events)


def test_runner_records_tool_loop_updates_without_extra_event_kinds(
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
    runner = AgentRunner(model, run_id_factory=lambda: "tool-run")
    request = RunRequest(
        repo_path=tmp_path,
        objective="Find authentication.",
        max_tool_calls=2,
        max_iterations=2,
    )

    outcome = runner.run(request)
    node_events = [
        event
        for event in outcome.trace
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]

    assert [event.node for event in node_events] == [
        "agent",
        "authorize_tools",
        "tools",
        "agent",
        "finalize",
    ]
    first_agent_message = node_events[0].payload["messages"][0]
    assert first_agent_message["tool_calls"][0]["name"] == "list_tree"
    assert node_events[1].payload == {"tool_call_count": 1}
    tool_message = node_events[2].payload["messages"][0]
    assert tool_message["tool_call_id"] == "call-tree"
    assert "auth.py" in tool_message["content"]
    assert outcome.stop_reason is None


def test_runner_preserves_iteration_budget_stop_reason(
    tmp_path: Path,
) -> None:
    tree_call = ToolCall(
        name="list_tree",
        args={"max_depth": 2, "max_entries": 10},
        id="call-tree",
        type="tool_call",
    )
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="The iteration budget was exhausted.",
        unresolved_questions=["Which file is relevant?"],
    )
    model = _create_scripted_model(
        [AIMessage(content="", tool_calls=[tree_call])],
        final_result,
    )
    runner = AgentRunner(model, run_id_factory=lambda: "budget-run")
    request = RunRequest(
        repo_path=tmp_path,
        objective="Inspect the repository.",
        max_tool_calls=2,
        max_iterations=1,
    )

    outcome = runner.run(request)

    assert outcome.stop_reason == "iteration_budget_exhausted"
    assert outcome.trace.events[-1].payload == {
        "stop_reason": "iteration_budget_exhausted"
    }


def test_runner_rejects_file_as_repository_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-repository"
    file_path.write_text("not a directory", encoding="utf-8")
    runner = AgentRunner(MagicMock(spec=BaseChatModel))
    request = RunRequest(
        repo_path=file_path,
        objective="Inspect the repository.",
        max_tool_calls=1,
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="not a directory"):
        runner.run(request)
