"""Cross-connection HITL, resume, and continuation episode lifecycle tests."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.lifecycle import (
    ContinueRequest,
    HumanDecision,
    HumanDecisionAction,
    HumanResponseRequest,
    HumanReviewPolicy,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.memory.consolidation import EpisodeConsolidationDisposition
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from repocompass.tools.gateway import ToolExecutionContext
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    (repository / "first.py").write_text("first = True\n", encoding="utf-8")
    (repository / "second.py").write_text("second = True\n", encoding="utf-8")
    _git(repository, "add", "first.py", "second.py")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository


def _result(path: str, summary: str) -> FinalResult:
    return FinalResult(
        relevant_files=[path],
        evidence=[f"{path}:1 supports the result."],
        summary=summary,
        unresolved_questions=[],
    )


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
        raise AssertionError("terminal respond/resume must not invoke a model")

    models = _models_for(result)
    models.explorer.bind_tools.return_value = RunnableLambda(fail)
    models.explorer.with_structured_output.return_value = RunnableLambda(fail)
    models.planner.with_structured_output.return_value = RunnableLambda(fail)
    models.evaluator.with_structured_output.return_value = RunnableLambda(fail)
    return models


def _response(outcome: object) -> HumanResponseRequest:
    assert hasattr(outcome, "thread_id")
    assert hasattr(outcome, "interrupt_id")
    interrupt_id = outcome.interrupt_id
    assert isinstance(interrupt_id, str)
    return HumanResponseRequest(
        thread_id=outcome.thread_id,
        interrupt_id=interrupt_id,
        decision=HumanDecision(action=HumanDecisionAction.APPROVE),
    )


def test_runner_consolidates_only_terminal_turns_across_process_boundaries(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    checkpoint_path = tmp_path / "state" / "checkpoints.sqlite"
    memory_path = tmp_path / "state" / "memory.sqlite"
    first_result = _result("first.py", "First terminal result.")
    second_result = _result("second.py", "Second terminal result.")
    request = RunRequest(
        repo_path=repository,
        objective="Inspect the first implementation.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
        human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
    )

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as memory_store,
    ):
        interrupted = AgentRunner(
            _models_for(first_result),
            checkpointer=checkpointer,
            memory_store=memory_store,
            episode_clock_factory=lambda: 1_000,
            run_id_factory=lambda: "start-attempt",
            thread_id_factory=lambda: "memory-thread",
        ).run(request)
        assert interrupted.episode_consolidation is not None
        assert (
            interrupted.episode_consolidation.disposition
            is EpisodeConsolidationDisposition.SKIPPED
        )
        assert (
            memory_store.list_episodic_memories(
                context=ToolExecutionContext(repository),
                max_results=10,
            )
            == ()
        )

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as memory_store,
    ):
        first_terminal = AgentRunner(
            _models_that_must_not_run(first_result),
            checkpointer=checkpointer,
            memory_store=memory_store,
            episode_clock_factory=lambda: 2_000,
            run_id_factory=lambda: "respond-attempt",
        ).respond(_response(interrupted))
        assert first_terminal.episode_consolidation is not None
        assert (
            first_terminal.episode_consolidation.disposition
            is EpisodeConsolidationDisposition.CREATED
        )
        first_episode = first_terminal.episode_consolidation.episode
        assert first_episode is not None
        assert first_episode.turn_number == 1
        assert first_episode.run_id == "respond-attempt"

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as memory_store,
    ):
        resumed = AgentRunner(
            _models_that_must_not_run(first_result),
            checkpointer=checkpointer,
            memory_store=memory_store,
            episode_clock_factory=lambda: 3_000,
            run_id_factory=lambda: "resume-attempt",
        ).resume(ResumeRequest(thread_id="memory-thread"))
        assert resumed.episode_consolidation is not None
        assert (
            resumed.episode_consolidation.disposition
            is EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED
        )
        assert resumed.episode_consolidation.episode == first_episode

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as memory_store,
    ):
        continued = AgentRunner(
            _models_for(second_result),
            checkpointer=checkpointer,
            memory_store=memory_store,
            episode_clock_factory=lambda: 4_000,
            run_id_factory=lambda: "continue-attempt",
        ).continue_thread(
            ContinueRequest(
                thread_id="memory-thread",
                prompt="Inspect the second implementation.",
            )
        )
        assert continued.episode_consolidation is not None
        assert (
            continued.episode_consolidation.disposition
            is EpisodeConsolidationDisposition.SKIPPED
        )
        assert (
            len(
                memory_store.list_episodic_memories(
                    context=ToolExecutionContext(repository),
                    max_results=10,
                )
            )
            == 1
        )

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as memory_store,
    ):
        second_terminal = AgentRunner(
            _models_that_must_not_run(second_result),
            checkpointer=checkpointer,
            memory_store=memory_store,
            episode_clock_factory=lambda: 5_000,
            run_id_factory=lambda: "second-respond-attempt",
        ).respond(_response(continued))
        assert second_terminal.episode_consolidation is not None
        assert (
            second_terminal.episode_consolidation.disposition
            is EpisodeConsolidationDisposition.CREATED
        )
        second_episode = second_terminal.episode_consolidation.episode
        assert second_episode is not None
        assert second_episode.turn_number == 2
        episodes = memory_store.list_episodic_memories(
            context=ToolExecutionContext(repository),
            max_results=10,
        )

    assert [episode.turn_number for episode in episodes] == [2, 1]
    assert first_episode.episode_id != second_episode.episode_id
    assert interrupted.trace.events[-1].payload["episode_consolidation"] == {
        "schema_version": 1,
        "disposition": "skipped",
        "reason": "attempt_interrupted",
        "episode_id": None,
    }
    assert (
        first_terminal.trace.events[-1].payload["episode_consolidation"]["episode_id"]
        == first_episode.episode_id
    )
