"""Transactional storage tests for observed terminal investigation episodes."""

import hashlib
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repocompass.memory.contracts import (
    EpisodicMemory,
    MemoryAuditAction,
    create_episodic_memory,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import (
    MemoryStore,
    MemoryStoreConflictError,
    MemoryStoreCorruptionError,
    MemoryStoreIntegrityError,
    open_memory_store,
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


def _episode(
    context: ToolExecutionContext,
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    turn_number: int = 1,
    termination_status: str = "accepted",
    summary: str = "Authentication was localized to auth.py.",
    created_at_ms: int = 1_000,
) -> EpisodicMemory:
    return create_episodic_memory(
        identity=resolve_memory_repository_identity(context),
        commit_sha=context.commit_sha,
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        termination_status=termination_status,  # type: ignore[arg-type]
        objective_digest=hashlib.sha256(b"find authentication").hexdigest(),
        summary=summary,
        query_sequence=("search_text:allowed:0123456789abcdef",),
        successful_paths=("auth.py",),
        failed_hypotheses=("Authentication is implemented in cli.py.",),
        final_result_digest=hashlib.sha256(b"final-result").hexdigest(),
        created_at_ms=created_at_ms,
    )


def test_put_get_list_search_and_audit_episode(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        assert store.put_episodic_memory(episode, context=context)
        assert store.get_episodic_memory(episode.episode_id) == episode
        assert (
            store.get_episodic_memory_for_turn(
                episode.repository_key,
                episode.thread_id,
                episode.turn_number,
            )
            == episode
        )
        assert store.list_episodic_memories(
            context=context,
            max_results=5,
            source_run_id="run-1",
        ) == (episode,)
        assert store.search_episodic_memories(
            "authentication auth",
            context=context,
            max_results=5,
            audit_at_ms=1_500,
            run_id="retrieval-1",
        ) == (episode,)
        events = store.list_memory_audit_events(episode.episode_id)

    assert [event.action for event in events] == [
        MemoryAuditAction.CREATED,
        MemoryAuditAction.RETRIEVED,
    ]


def test_reopen_and_exact_retry_are_idempotent(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        assert store.put_episodic_memory(episode, context=context)

    with open_memory_store(database_path) as reopened:
        assert not reopened.put_episodic_memory(episode, context=context)
        assert reopened.get_episodic_memory(episode.episode_id) == episode
        assert len(reopened.list_memory_audit_events(episode.episode_id)) == 1


def test_logical_thread_turn_is_unique_across_attempt_ids(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    first = _episode(context, run_id="run-1")
    retry = _episode(context, run_id="run-2", created_at_ms=2_000)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_episodic_memory(first, context=context)
        with pytest.raises(MemoryStoreConflictError, match="thread turn"):
            store.put_episodic_memory(retry, context=context)
        assert store.get_episodic_memory(retry.episode_id) is None
        assert (
            store.get_episodic_memory_for_turn(
                first.repository_key,
                first.thread_id,
                first.turn_number,
            )
            == first
        )


def test_episode_write_rejects_untrusted_shape_identity_and_commit(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    _, other_context = _create_repository(tmp_path / "other")

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(TypeError, match="EpisodicMemory"):
            store.put_episodic_memory(  # type: ignore[arg-type]
                episode.model_dump(mode="json"),
                context=context,
            )
        with pytest.raises(MemoryStoreIntegrityError, match="identity"):
            store.put_episodic_memory(episode, context=other_context)
        wrong_commit = episode.model_copy(update={"commit_sha": "f" * 40})
        with pytest.raises(MemoryStoreIntegrityError, match="contract"):
            store.put_episodic_memory(wrong_commit, context=context)


def test_dirty_snapshot_can_be_recorded_only_as_observed_episode(
    tmp_path: Path,
) -> None:
    repository, context = _create_repository(tmp_path)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    episode = _episode(context, termination_status="exhausted")

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        assert store.put_episodic_memory(episode, context=context)
        loaded = store.get_episodic_memory(episode.episode_id)

    assert loaded is not None
    assert loaded.verification_status.value == "observed"
    assert loaded.write_authority.value == "terminal_consolidation"
    assert loaded.termination_status == "exhausted"


def test_listing_is_repository_scoped_and_stably_ordered(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path / "first")
    _, other_context = _create_repository(tmp_path / "second")
    older = _episode(context, turn_number=1, created_at_ms=1_000)
    newer = _episode(
        context,
        run_id="run-2",
        turn_number=2,
        created_at_ms=2_000,
    )
    other = _episode(other_context, thread_id="thread-2")

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_episodic_memory(older, context=context)
        store.put_episodic_memory(newer, context=context)
        store.put_episodic_memory(other, context=other_context)
        assert store.list_episodic_memories(
            context=context,
            max_results=10,
        ) == (newer, older)
        assert store.list_episodic_memories(
            context=other_context,
            max_results=10,
        ) == (other,)


@pytest.mark.parametrize("max_results", [0, 51, True])
def test_episode_list_and_search_enforce_result_bounds(
    tmp_path: Path,
    max_results: int,
) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(ValueError, match="max_results"):
            store.list_episodic_memories(
                context=context,
                max_results=max_results,
            )
        with pytest.raises(ValueError, match="max_results"):
            store.search_episodic_memories(
                "authentication",
                context=context,
                max_results=max_results,
                audit_at_ms=1_500,
            )


def test_corrupt_episode_record_or_index_fails_closed(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_episodic_memory(episode, context=context)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_entries SET record_json = '{}' WHERE memory_id = ?",
            (episode.episode_id,),
        )
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreCorruptionError, match="episode record"):
            store.get_episodic_memory(episode.episode_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_entries SET record_json = ? WHERE memory_id = ?",
            (episode.model_dump_json(), episode.episode_id),
        )
        connection.execute(
            "UPDATE memory_episodes SET objective_digest = ? WHERE episode_id = ?",
            ("f" * 64, episode.episode_id),
        )
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreCorruptionError, match="episode index"):
            store.get_episodic_memory(episode.episode_id)


def test_episode_index_failure_rolls_back_entry_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    database_path = tmp_path / "memory.sqlite"

    def fail_episode_index(store: MemoryStore, value: EpisodicMemory) -> None:
        del store, value
        raise sqlite3.IntegrityError("injected episode-index failure")

    monkeypatch.setattr(MemoryStore, "_insert_episode_index", fail_episode_index)
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreIntegrityError, match="storage integrity"):
            store.put_episodic_memory(episode, context=context)
        assert store.get_episodic_memory(episode.episode_id) is None
        assert store.list_memory_audit_events(episode.episode_id) == ()

    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM memory_entries").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM memory_episodes").fetchone()[0]
            == 0
        )


def test_concurrent_exact_episode_writers_are_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    episode = _episode(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path):
        pass

    def write_once() -> bool:
        with open_memory_store(database_path) as store:
            return store.put_episodic_memory(episode, context=context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(executor.map(lambda _: write_once(), range(2)))

    assert results == [False, True]
    with open_memory_store(database_path) as store:
        assert store.get_episodic_memory(episode.episode_id) == episode
        assert len(store.list_memory_audit_events(episode.episode_id)) == 1
