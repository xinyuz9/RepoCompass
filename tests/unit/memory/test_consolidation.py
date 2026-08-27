"""Deterministic terminal lifecycle to episodic-memory consolidation tests."""

import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    ExecutionStatus,
    HumanReviewRequest,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import RunOutcome
from repocompass.agent.trace import RunTrace
from repocompass.memory.consolidation import (
    EpisodeConsolidationDisposition,
    EpisodeConsolidationError,
    consolidate_terminal_episode,
)
from repocompass.memory.store import MemoryStoreConflictError, open_memory_store
from repocompass.tools.contracts import (
    ToolAuditRecord,
    ToolCallRequest,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolPolicyDecision,
    fingerprint_tool_call,
)
from repocompass.tools.gateway import ToolExecutionContext


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_repository(root: Path) -> tuple[Path, ToolExecutionContext]:
    repository = root / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    (repository / "auth.py").write_text(
        "def authenticate(user):\n    return user is not None\n",
        encoding="utf-8",
    )
    _git(repository, "add", "auth.py")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository, ToolExecutionContext(repository)


def _audit(
    context: ToolExecutionContext,
    *,
    run_id: str,
    sequence: int = 0,
    tool_name: str = "search_text",
) -> ToolAuditRecord:
    request = ToolCallRequest(
        call_id=f"call-{sequence}",
        tool_name=tool_name,
        arguments={"query": "authentication", "max_results": 10},
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
    )
    return ToolAuditRecord(
        sequence=sequence,
        run_id=run_id,
        call_id=request.call_id,
        tool_name=tool_name,
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
        fingerprint=fingerprint_tool_call(request),
        decision=ToolPolicyDecision(
            outcome=ToolDecisionOutcome.ALLOWED,
            reason=ToolDecisionReason.ALLOWED,
        ),
        latency_ms=2,
        output_bytes=32,
        result_digest="f" * 64,
    )


def _final_result(
    *,
    summary: str = "Authentication is implemented in auth.py.",
    relevant_files: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
) -> FinalResult:
    return FinalResult(
        relevant_files=["auth.py"] if relevant_files is None else relevant_files,
        evidence=["auth.py contains authenticate."],
        summary=summary,
        unresolved_questions=(
            ["Whether an external identity provider is configured."]
            if unresolved_questions is None
            else unresolved_questions
        ),
    )


def _completed_outcome(
    context: ToolExecutionContext,
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    termination_status: str = "accepted",
    final_result: FinalResult | None = None,
    audits: tuple[ToolAuditRecord, ...] | None = None,
) -> RunOutcome:
    return RunOutcome(
        thread_id=thread_id,
        final_result=_final_result() if final_result is None else final_result,
        termination_status=termination_status,  # type: ignore[arg-type]
        step_stop_reason=None,
        trace=RunTrace(run_id),
        tool_audit_records=(
            (_audit(context, run_id=run_id),) if audits is None else audits
        ),
    )


def _interrupted_outcome() -> RunOutcome:
    candidate = CandidateResult(
        evidence=[],
        summary="Candidate awaiting review.",
        unresolved_questions=[],
    )
    evaluation = EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=SemanticAssessment(
            active_step_completed=True,
            answers_objective=True,
            evidence_supports_summary=True,
            has_critical_omissions=False,
            missing_evidence=[],
            recommended_focus=[],
            summary="Ready for review.",
        ),
    )
    return RunOutcome(
        thread_id="thread-1",
        final_result=None,
        termination_status=None,
        step_stop_reason=None,
        trace=RunTrace("run-interrupted"),
        execution_status=ExecutionStatus.INTERRUPTED,
        human_review_request=HumanReviewRequest(
            turn_number=1,
            candidate_result=candidate,
            evaluation_report=evaluation,
        ),
        interrupt_id="interrupt-1",
    )


def test_completed_outcome_creates_sanitized_observed_episode(
    tmp_path: Path,
) -> None:
    repository, context = _create_repository(tmp_path)
    outcome = _completed_outcome(
        context,
        final_result=_final_result(
            summary=(
                f"Authentication is in {repository}/auth.py; api_key=super-secret-value"
            ),
            unresolved_questions=["password=hunter2 may configure the provider."],
        ),
    )

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        result = consolidate_terminal_episode(
            store,
            context=context,
            outcome=outcome,
            objective="Find authentication. api_key=objective-secret",
            turn_number=1,
            created_at_ms=1_000,
        )
        loaded = store.get_episodic_memory(result.episode.episode_id)  # type: ignore[union-attr]

    assert result.disposition is EpisodeConsolidationDisposition.CREATED
    assert result.episode is not None
    assert loaded == result.episode
    assert result.episode.termination_status == "accepted"
    assert result.episode.successful_paths == ("auth.py",)
    assert result.episode.query_sequence[0].startswith("search_text:allowed:")
    serialized = result.episode.model_dump_json()
    assert str(repository) not in serialized
    assert "super-secret-value" not in serialized
    assert "hunter2" not in serialized
    assert "objective-secret" not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize("termination_status", ["exhausted", "cancelled"])
def test_terminal_status_is_preserved_without_upgrading_verification(
    tmp_path: Path,
    termination_status: str,
) -> None:
    _, context = _create_repository(tmp_path)
    outcome = _completed_outcome(
        context,
        termination_status=termination_status,
    )

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        result = consolidate_terminal_episode(
            store,
            context=context,
            outcome=outcome,
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )

    assert result.episode is not None
    assert result.episode.termination_status == termination_status
    assert result.episode.verification_status.value == "observed"
    assert result.episode.write_authority.value == "terminal_consolidation"


def test_interrupted_attempt_is_skipped_without_write(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        result = consolidate_terminal_episode(
            store,
            context=context,
            outcome=_interrupted_outcome(),
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )
        assert (
            store.list_episodic_memories(
                context=context,
                max_results=10,
            )
            == ()
        )

    assert result.disposition is EpisodeConsolidationDisposition.SKIPPED
    assert result.episode is None


def test_completed_resume_returns_existing_logical_turn(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        first = consolidate_terminal_episode(
            store,
            context=context,
            outcome=_completed_outcome(context, run_id="run-1"),
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )

    with open_memory_store(database_path) as store:
        resumed = consolidate_terminal_episode(
            store,
            context=context,
            outcome=_completed_outcome(context, run_id="run-2"),
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=2_000,
        )
        episodes = store.list_episodic_memories(
            context=context,
            max_results=10,
        )

    assert first.disposition is EpisodeConsolidationDisposition.CREATED
    assert resumed.disposition is EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED
    assert resumed.episode == first.episode
    assert episodes == (first.episode,)


def test_same_turn_with_different_terminal_result_conflicts(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        consolidate_terminal_episode(
            store,
            context=context,
            outcome=_completed_outcome(context),
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )
        changed = _completed_outcome(
            context,
            run_id="run-2",
            final_result=_final_result(summary="A different terminal answer."),
        )
        with pytest.raises(MemoryStoreConflictError, match="terminal result"):
            consolidate_terminal_episode(
                store,
                context=context,
                outcome=changed,
                objective="Find authentication.",
                turn_number=1,
                created_at_ms=2_000,
            )


def test_continuation_turn_is_consolidated_as_a_distinct_episode(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        first = consolidate_terminal_episode(
            store,
            context=context,
            outcome=_completed_outcome(context, run_id="run-1"),
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )
        second = consolidate_terminal_episode(
            store,
            context=context,
            outcome=_completed_outcome(context, run_id="run-2"),
            objective="Find authentication tests.",
            turn_number=2,
            created_at_ms=2_000,
        )

    assert first.episode is not None
    assert second.episode is not None
    assert first.episode.episode_id != second.episode.episode_id
    assert second.episode.turn_number == 2


@pytest.mark.parametrize("mismatch", ["run", "repository", "commit"])
def test_tool_audit_must_match_terminal_attempt_and_repository(
    tmp_path: Path,
    mismatch: str,
) -> None:
    _, context = _create_repository(tmp_path)
    audit = _audit(context, run_id="run-1")
    if mismatch == "run":
        audit = audit.model_copy(update={"run_id": "other-run"})
    elif mismatch == "repository":
        audit = audit.model_copy(update={"repository_id": "a" * 64})
    else:
        audit = audit.model_copy(update={"commit_sha": "b" * 40})
    outcome = _completed_outcome(context, audits=(audit,))

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(EpisodeConsolidationError, match="audit"):
            consolidate_terminal_episode(
                store,
                context=context,
                outcome=outcome,
                objective="Find authentication.",
                turn_number=1,
                created_at_ms=1_000,
            )


def test_consolidation_enforces_sequence_text_and_path_bounds(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    audits = tuple(
        _audit(context, run_id="run-1", sequence=index) for index in range(70)
    )
    final = _final_result(
        summary="s" * 20_000,
        relevant_files=[f"src/file_{index}.py" for index in range(70)],
        unresolved_questions=["q" * 2_000 for _ in range(70)],
    )
    outcome = _completed_outcome(context, final_result=final, audits=audits)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        result = consolidate_terminal_episode(
            store,
            context=context,
            outcome=outcome,
            objective="Find authentication.",
            turn_number=1,
            created_at_ms=1_000,
        )

    assert result.episode is not None
    assert len(result.episode.summary) == 16_384
    assert len(result.episode.query_sequence) == 64
    assert len(result.episode.successful_paths) == 64
    assert len(result.episode.failed_hypotheses) == 64
    assert all(len(item) <= 1_024 for item in result.episode.failed_hypotheses)


def test_unsafe_terminal_path_fails_without_partial_write(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    outcome = _completed_outcome(
        context,
        final_result=_final_result(relevant_files=["../outside.py"]),
    )
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(EpisodeConsolidationError, match="path"):
            consolidate_terminal_episode(
                store,
                context=context,
                outcome=outcome,
                objective="Find authentication.",
                turn_number=1,
                created_at_ms=1_000,
            )
        assert (
            store.list_episodic_memories(
                context=context,
                max_results=10,
            )
            == ()
        )


def test_concurrent_terminal_retries_create_one_logical_episode(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path):
        pass

    def consolidate(index: int) -> EpisodeConsolidationDisposition:
        with open_memory_store(database_path) as store:
            result = consolidate_terminal_episode(
                store,
                context=context,
                outcome=_completed_outcome(context, run_id=f"run-{index}"),
                objective="Find authentication.",
                turn_number=1,
                created_at_ms=1_000 + index,
            )
            return result.disposition

    with ThreadPoolExecutor(max_workers=2) as executor:
        dispositions = set(executor.map(consolidate, range(2)))

    assert dispositions == {
        EpisodeConsolidationDisposition.CREATED,
        EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED,
    }
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM memory_episodes").fetchone()[0]
            == 1
        )
