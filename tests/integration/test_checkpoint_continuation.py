"""SQLite integration coverage for completed-thread continuation."""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.lifecycle import ContinueRequest, TurnRecord
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunRequest
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


def test_runner_continues_completed_thread_after_reopening_sqlite(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.py").write_text("first = True\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("second = True\n", encoding="utf-8")
    database_path = tmp_path / "state" / "checkpoints.sqlite"
    first_result = FinalResult(
        relevant_files=["first.py"],
        evidence=["first.py supports the first answer."],
        summary="First answer.",
        unresolved_questions=[],
    )
    second_result = FinalResult(
        relevant_files=["second.py"],
        evidence=["second.py supports the second answer."],
        summary="Second answer.",
        unresolved_questions=[],
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        first_outcome = AgentRunner(
            _models_for(first_result),
            checkpointer=checkpointer,
            thread_id_factory=lambda: "sqlite-thread",
        ).run(
            RunRequest(
                repo_path=tmp_path,
                objective="First objective",
                max_tool_calls_per_step=4,
                max_explorer_calls_per_step=3,
            )
        )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        runner = AgentRunner(
            _models_for(second_result),
            checkpointer=checkpointer,
            run_id_factory=lambda: "continued-run",
        )
        second_outcome = runner.continue_thread(
            ContinueRequest(
                thread_id=first_outcome.thread_id,
                prompt="Second objective",
            )
        )

    assert second_outcome.thread_id == "sqlite-thread"
    assert second_outcome.trace.run_id == "continued-run"
    assert second_outcome.final_result == second_result

    with open_sqlite_checkpointer(database_path) as checkpointer:
        graph = create_agent_graph(
            _models_for(second_result),
            checkpointer=checkpointer,
        )
        snapshot = graph.get_state(
            {"configurable": {"thread_id": "sqlite-thread"}}
        )

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
