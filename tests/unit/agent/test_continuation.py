"""Tests for completed-thread continuation and turn-local state reset."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.graph import create_agent_graph
from repocompass.agent.lifecycle import ContinueRequest, TurnRecord
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.state import (
    create_continuation_update,
    create_initial_state,
)
from repocompass.agent.trace import TraceEventKind
from repocompass.models import WorkflowModels
from tests.model_doubles import (
    candidate_from_final_result,
    default_investigation_plan,
    with_scripted_planner,
)


def _result(filename: str, summary: str) -> FinalResult:
    return FinalResult(
        relevant_files=[filename],
        evidence=[f"{filename} supports the answer."],
        summary=summary,
        unresolved_questions=[],
    )


def _sequential_models(
    final_results: list[FinalResult],
) -> tuple[WorkflowModels, MagicMock]:
    agent_responses = iter(
        AIMessage(content=f"Evidence collected for {result.summary}")
        for result in final_results
    )
    candidates = iter(candidate_from_final_result(result) for result in final_results)
    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(lambda _: next(agent_responses))
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: next(candidates)
    )
    return with_scripted_planner(explorer), explorer


def test_create_continuation_update_archives_and_resets_turn_state(
    tmp_path: Path,
) -> None:
    first_result = _result("first.py", "First answer.")
    first_candidate = candidate_from_final_result(first_result)
    state = create_initial_state(
        tmp_path,
        "First objective",
        max_tool_calls_per_step=7,
        max_explorer_calls_per_step=5,
        max_plan_revisions=3,
    )
    state.update(
        {
            "messages": [
                *state["messages"],
                AIMessage(content="Old turn model output."),
            ],
            "investigation_plan": default_investigation_plan(),
            "step_tool_call_count": 4,
            "step_explorer_call_count": 3,
            "plan_revision_count": 2,
            "step_stop_reason": "step_explorer_budget_exhausted",
            "termination_status": "accepted",
            "candidate_result": first_candidate,
            "evaluation_report": None,
            "next_action": "accepted",
            "final_result": first_result,
        }
    )

    update = create_continuation_update(state, "  Second objective  ")

    assert update["repo_path"] == tmp_path.resolve()
    assert update["max_tool_calls_per_step"] == 7
    assert update["max_explorer_calls_per_step"] == 5
    assert update["max_plan_revisions"] == 3
    assert update["human_review_policy"] == state["human_review_policy"]
    assert update["objective"] == "Second objective"
    assert update["turn_number"] == 2
    assert update["turn_history"] == [
        TurnRecord(
            turn_number=1,
            objective="First objective",
            final_result=first_result,
            termination_status="accepted",
        )
    ]
    assert state["turn_history"] == []
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], HumanMessage)
    assert update["messages"][0].content == "Second objective"
    assert update["human_review_payload"] is None
    assert update["human_decision_action"] is None
    assert update["human_feedback"] is None
    assert update["investigation_plan"] is None
    assert update["step_tool_call_count"] == 0
    assert update["step_explorer_call_count"] == 0
    assert update["plan_revision_count"] == 0
    assert update["step_stop_reason"] is None
    assert update["termination_status"] is None
    assert update["candidate_result"] is None
    assert update["evaluation_report"] is None
    assert update["next_action"] is None
    assert update["final_result"] is None


def test_create_continuation_update_accepts_cancelled_turn(tmp_path: Path) -> None:
    result = _result("partial.py", "Best effort result.")
    state = create_initial_state(tmp_path, "Cancelled objective", 2, 2)
    state["termination_status"] = "cancelled"
    state["final_result"] = result

    update = create_continuation_update(state, "Follow-up objective")

    assert update["turn_history"] == [
        TurnRecord(
            turn_number=1,
            objective="Cancelled objective",
            final_result=result,
            termination_status="cancelled",
        )
    ]


@pytest.mark.parametrize(
    ("final_result", "termination_status"),
    [
        (None, None),
        (_result("partial.py", "Partial answer."), None),
    ],
)
def test_create_continuation_update_rejects_non_terminal_state(
    tmp_path: Path,
    final_result: FinalResult | None,
    termination_status: str | None,
) -> None:
    state = create_initial_state(tmp_path, "Objective", 2, 2)
    state["final_result"] = final_result
    state["termination_status"] = termination_status  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="completed turn"):
        create_continuation_update(state, "Next objective")


def test_runner_continues_completed_thread_from_planner_without_state_leakage(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.py").write_text("first = True\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("second = True\n", encoding="utf-8")
    first_result = _result("first.py", "First answer.")
    second_result = _result("second.py", "Second answer.")
    models, _ = _sequential_models([first_result, second_result])
    checkpointer = InMemorySaver()
    run_ids = iter(["first-run", "second-run"])
    runner = AgentRunner(
        models,
        checkpointer=checkpointer,
        run_id_factory=lambda: next(run_ids),
        thread_id_factory=lambda: "continued-thread",
    )

    first_outcome = runner.run(request=_run_request(tmp_path, "First objective"))
    second_outcome = runner.continue_thread(
        ContinueRequest(
            thread_id=first_outcome.thread_id,
            prompt="Second objective",
        )
    )

    assert second_outcome.thread_id == first_outcome.thread_id
    assert second_outcome.final_result == second_result
    assert second_outcome.trace.run_id == "second-run"
    assert second_outcome.trace.events[0].payload["execution_mode"] == "continue"
    node_events = [
        event
        for event in second_outcome.trace.events
        if event.kind is TraceEventKind.NODE_COMPLETED
    ]
    assert node_events[0].node == "plan_investigation"

    snapshot = create_agent_graph(models, checkpointer=checkpointer).get_state(
        {"configurable": {"thread_id": first_outcome.thread_id}}
    )
    assert snapshot.values["repo_path"] == tmp_path.resolve()
    assert snapshot.values["objective"] == "Second objective"
    assert snapshot.values["turn_number"] == 2
    assert snapshot.values["turn_history"] == [
        TurnRecord(
            turn_number=1,
            objective="First objective",
            final_result=first_result,
            termination_status="accepted",
        )
    ]
    assert snapshot.values["step_tool_call_count"] == 0
    assert snapshot.values["step_explorer_call_count"] == 1
    assert snapshot.values["plan_revision_count"] == 0
    assert snapshot.values["step_stop_reason"] is None
    assert snapshot.values["candidate_result"] == candidate_from_final_result(
        second_result
    )
    assert snapshot.values["evaluation_report"] is not None
    assert snapshot.values["final_result"] == second_result
    human_messages = [
        message.content
        for message in snapshot.values["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert human_messages == ["First objective", "Second objective"]


def test_runner_rejects_continuation_for_unknown_or_unfinished_thread(
    tmp_path: Path,
) -> None:
    models, _ = _sequential_models([_result("unused.py", "Unused.")])
    checkpointer = InMemorySaver()
    runner = AgentRunner(models, checkpointer=checkpointer)

    with pytest.raises(ValueError, match="does not exist"):
        runner.continue_thread(
            ContinueRequest(thread_id="missing-thread", prompt="Next objective")
        )

    graph = create_agent_graph(models, checkpointer=checkpointer)
    graph.update_state(
        {"configurable": {"thread_id": "unfinished-thread"}},
        create_initial_state(tmp_path, "Unfinished objective", 2, 2),
        as_node="__start__",
    )

    with pytest.raises(ValueError, match="not completed"):
        runner.continue_thread(
            ContinueRequest(thread_id="unfinished-thread", prompt="Too early")
        )


def test_continuation_does_not_modify_another_thread(tmp_path: Path) -> None:
    for filename in ("one.py", "two.py", "three.py"):
        (tmp_path / filename).write_text("value = True\n", encoding="utf-8")
    first_result = _result("one.py", "Thread one, turn one.")
    isolated_result = _result("two.py", "Thread two.")
    continued_result = _result("three.py", "Thread one, turn two.")
    models, _ = _sequential_models([first_result, isolated_result, continued_result])
    checkpointer = InMemorySaver()
    thread_ids = iter(["thread-one", "thread-two"])
    runner = AgentRunner(
        models,
        checkpointer=checkpointer,
        thread_id_factory=lambda: next(thread_ids),
    )

    first = runner.run(_run_request(tmp_path, "Thread one objective"))
    isolated = runner.run(_run_request(tmp_path, "Thread two objective"))
    runner.continue_thread(
        ContinueRequest(thread_id=first.thread_id, prompt="Thread one follow-up")
    )

    isolated_snapshot = create_agent_graph(
        models,
        checkpointer=checkpointer,
    ).get_state({"configurable": {"thread_id": isolated.thread_id}})
    assert isolated_snapshot.values["objective"] == "Thread two objective"
    assert isolated_snapshot.values["turn_number"] == 1
    assert isolated_snapshot.values["turn_history"] == []
    assert isolated_snapshot.values["final_result"] == isolated_result


def _run_request(repo_path: Path, objective: str) -> RunRequest:
    return RunRequest(
        repo_path=repo_path,
        objective=objective,
        max_tool_calls_per_step=4,
        max_explorer_calls_per_step=3,
        max_plan_revisions=2,
    )
