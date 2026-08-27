"""Cross-SQLite memory recall lifecycle and checkpoint idempotence tests."""

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.context import ContextMemoryFreshness, ContextSection
from repocompass.agent.graph import create_agent_graph
from repocompass.agent.lifecycle import (
    ContinueRequest,
    HumanDecision,
    HumanDecisionAction,
    HumanResponseRequest,
    HumanReviewPolicy,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryFactKind,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.agent.context import RepositoryLocation
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> tuple[Path, ToolExecutionContext]:
    repository = root / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    (repository / "auth.py").write_text(
        "def authenticate(request):\n    return request.user is not None\n",
        encoding="utf-8",
    )
    _git(repository, "add", "auth.py")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, ToolExecutionContext(repository)


def _memory(context: ToolExecutionContext):
    identity = resolve_memory_repository_identity(context)
    commit_sha = context.commit_sha
    assert commit_sha is not None
    source_path = context.repository_root / "auth.py"
    source = create_repository_memory_source(
        location=RepositoryLocation(
            repository_id=context.repository_id,
            commit_sha=commit_sha,
            path="auth.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
        ),
        source_digest=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    return create_repository_memory(
        identity=identity,
        commit_sha=commit_sha,
        kind=MemoryFactKind.ENTRY_POINT,
        title="Authentication entry point",
        content="auth.authenticate validates the request user.",
        verification_status=MemoryVerificationStatus.VERIFIED,
        write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
        sources=(source,),
        tags=("authentication",),
        created_at_ms=1_000,
        verified_at_ms=1_000,
    )


def _result() -> FinalResult:
    return FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py:1 supports the result."],
        summary="Authentication is implemented in auth.py.",
        unresolved_questions=[],
    )


def _models(captured_inputs: list[list[object]]) -> WorkflowModels:
    result = _result()

    def capture(messages: list[object]) -> AIMessage:
        captured_inputs.append(list(messages))
        return AIMessage(content="Enough evidence is available.")

    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(capture)
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(result)
    )
    return with_scripted_planner(explorer)


def _models_that_must_not_run() -> WorkflowModels:
    def fail(_: object) -> object:
        raise AssertionError("resume/respond must not invoke a model")

    models = _models([])
    models.explorer.bind_tools.return_value = RunnableLambda(fail)
    models.explorer.with_structured_output.return_value = RunnableLambda(fail)
    models.planner.with_structured_output.return_value = RunnableLambda(fail)
    models.evaluator.with_structured_output.return_value = RunnableLambda(fail)
    return models


def _approve(outcome: object) -> HumanResponseRequest:
    assert hasattr(outcome, "thread_id")
    assert hasattr(outcome, "interrupt_id")
    assert isinstance(outcome.interrupt_id, str)
    return HumanResponseRequest(
        thread_id=outcome.thread_id,
        interrupt_id=outcome.interrupt_id,
        decision=HumanDecision(action=HumanDecisionAction.APPROVE),
    )


def test_recall_runs_once_per_new_turn_not_resume_or_human_response(
    tmp_path: Path,
) -> None:
    repository, context = _repository(tmp_path)
    memory = _memory(context)
    checkpoint_path = tmp_path / "state" / "checkpoints.sqlite"
    memory_path = tmp_path / "state" / "memory.sqlite"
    captured_inputs: list[list[object]] = []
    request = RunRequest(
        repo_path=repository,
        objective="Explain authentication.",
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
        human_review_policy=HumanReviewPolicy.BEFORE_FINALIZE,
        memory_retrieval_enabled=True,
        max_repository_memory_results=3,
        max_episodic_memory_results=3,
    )

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as store,
    ):
        store.put_repository_memory(memory, context=context, run_id="seed-run")
        started = AgentRunner(
            _models(captured_inputs),
            checkpointer=checkpointer,
            memory_store=store,
            episode_clock_factory=lambda: 2_000,
            run_id_factory=lambda: "start-run",
            thread_id_factory=lambda: "memory-recall-thread",
        ).run(request)
        assert [event.node for event in started.trace.events].count(
            "recall_memory"
        ) == 1
        repository_events = store.list_memory_audit_events(memory.memory_id)
        assert [event.action for event in repository_events] == [
            MemoryAuditAction.CREATED,
            MemoryAuditAction.RETRIEVED,
        ]

        graph = create_agent_graph(
            _models([]),
            checkpointer=checkpointer,
            memory_store=store,
            memory_clock=lambda: 2_000,
        )
        snapshot = graph.get_state(
            {"configurable": {"thread_id": "memory-recall-thread"}}
        )
        report = snapshot.values["memory_recall_report"]
        items = snapshot.values["memory_context_items"]
        assert report.repository_result_count == 1
        assert len(items) == 1
        assert items[0].section is ContextSection.MEMORY
        assert items[0].source.memory.freshness is ContextMemoryFreshness.CURRENT

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as store,
    ):
        resumed = AgentRunner(
            _models_that_must_not_run(),
            checkpointer=checkpointer,
            memory_store=store,
            episode_clock_factory=lambda: 2_500,
            run_id_factory=lambda: "resume-run",
        ).resume(ResumeRequest(thread_id="memory-recall-thread"))
        assert all(event.node != "recall_memory" for event in resumed.trace.events)
        assert len(store.list_memory_audit_events(memory.memory_id)) == 2

        terminal = AgentRunner(
            _models_that_must_not_run(),
            checkpointer=checkpointer,
            memory_store=store,
            episode_clock_factory=lambda: 3_000,
            run_id_factory=lambda: "respond-run",
        ).respond(_approve(started))
        assert all(event.node != "recall_memory" for event in terminal.trace.events)
        first_episode = terminal.episode_consolidation.episode
        assert first_episode is not None
        assert len(store.list_memory_audit_events(memory.memory_id)) == 2

    with (
        open_sqlite_checkpointer(checkpoint_path) as checkpointer,
        open_memory_store(memory_path) as store,
    ):
        continued = AgentRunner(
            _models(captured_inputs),
            checkpointer=checkpointer,
            memory_store=store,
            episode_clock_factory=lambda: 4_000,
            run_id_factory=lambda: "continue-run",
        ).continue_thread(
            ContinueRequest(
                thread_id="memory-recall-thread",
                prompt="Recheck authentication using prior investigation history.",
            )
        )
        assert [event.node for event in continued.trace.events].count(
            "recall_memory"
        ) == 1
        assert len(store.list_memory_audit_events(memory.memory_id)) == 3
        episode_events = store.list_memory_audit_events(first_episode.episode_id)
        assert [event.action for event in episode_events] == [
            MemoryAuditAction.CREATED,
            MemoryAuditAction.RETRIEVED,
        ]

    assert len(captured_inputs) == 2
    rendered_inputs = "\n".join(
        str(message.content)
        for invocation in captured_inputs
        for message in invocation
        if hasattr(message, "content")
    )
    assert rendered_inputs.count(memory.content) == 2
    assert "observed_strategy_history_only_never_code_evidence" in rendered_inputs
