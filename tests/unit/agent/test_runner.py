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
        run_id_factory=lambda: "fixed-run",
    )
    request = RunRequest(
        repo_path=tmp_path,
        objective="  Inspect the repository.  ",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
    )

    outcome = runner.run(request)
    events = outcome.trace.events

    assert outcome.final_result == final_result
    assert outcome.termination_status == "accepted"
    assert outcome.step_stop_reason is None
    assert outcome.trace.run_id == "fixed-run"
    assert [event.kind for event in events] == [
        TraceEventKind.RUN_STARTED,
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
        "mark_investigation_accepted",
        "finalize",
        None,
    ]
    assert events[0].payload == {
        "repo_path": str(tmp_path.resolve()),
        "objective": "Inspect the repository.",
        "max_tool_calls_per_step": 3,
        "max_explorer_calls_per_step": 2,
        "max_plan_revisions": 2,
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
    assert events[5].payload == {"next_action": "accepted"}
    assert events[6].payload == {"termination_status": "accepted"}
    assert events[7].payload["final_result"] == final_result.model_dump(mode="json")
    assert events[-1].payload == {
        "termination_status": "accepted",
        "step_stop_reason": None,
    }
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
    runner = AgentRunner(
        with_scripted_planner(model),
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
        "authorize_tools",
        "tools",
        "agent",
        "build_candidate",
        "evaluate_candidate",
        "decide_next_action",
        "mark_investigation_accepted",
        "finalize",
    ]
    first_agent_message = node_events[1].payload["messages"][0]
    assert first_agent_message["tool_calls"][0]["name"] == "list_tree"
    assert node_events[2].payload == {"step_tool_call_count": 1}
    tool_message = node_events[3].payload["messages"][0]
    assert tool_message["tool_call_id"] == "call-tree"
    assert "auth.py" in tool_message["content"]
    assert outcome.termination_status == "accepted"
    assert outcome.step_stop_reason is None


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
        "termination_status": "accepted",
        "step_stop_reason": "step_explorer_budget_exhausted",
    }


def test_runner_rejects_file_as_repository_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-repository"
    file_path.write_text("not a directory", encoding="utf-8")
    runner = AgentRunner(with_scripted_planner(MagicMock(spec=BaseChatModel)))
    request = RunRequest(
        repo_path=file_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=1,
    )

    with pytest.raises(ValueError, match="not a directory"):
        runner.run(request)
