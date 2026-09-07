"""Repository-scoped inspection, deletion, tombstone, and retention tests."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryAuditReason,
    MemoryScope,
    create_memory_retention_policy,
)
from repocompass.memory.store import (
    MEMORY_STORE_SCHEMA_VERSION,
    MemoryStore,
    MemoryStoreConflictError,
    MemoryStoreIntegrityError,
    MemoryStoreMigrationError,
    open_memory_store,
)
from tests.unit.memory.test_episodic_store import _episode
from tests.unit.memory.test_store import _create_repository, _memory


def test_metadata_inspection_is_payload_free_and_repository_scoped(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    repository_memory = _memory(context)
    episode = _episode(context, created_at_ms=2_000)
    _, other_context = _create_repository(tmp_path / "other")

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(repository_memory, context=context)
        store.put_episodic_memory(episode, context=context)

        metadata = store.list_memory_metadata(context=context, max_results=10)
        repository_metadata = store.get_memory_metadata(
            repository_memory.memory_id,
            context=context,
        )

        assert (
            store.list_memory_metadata(
                context=other_context,
                max_results=10,
            )
            == ()
        )
        assert (
            store.get_memory_metadata(
                repository_memory.memory_id,
                context=other_context,
            )
            is None
        )

    assert [item.memory_scope for item in metadata] == [
        MemoryScope.EPISODIC,
        MemoryScope.REPOSITORY,
    ]
    assert repository_metadata is not None
    assert repository_metadata.source_count == 1
    assert repository_metadata.audit_event_count == 1
    serialized = "\n".join(item.model_dump_json() for item in metadata)
    assert "Authentication entry point" not in serialized
    assert "src/api.py" not in serialized
    assert "localized to" not in serialized


def test_explicit_delete_cleans_indexes_is_idempotent_and_blocks_resurrection(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context, run_id="write-1")
        tombstone = store.delete_memory(
            memory.memory_id,
            context=context,
            deleted_at_ms=2_000,
            run_id="delete-1",
        )
        retry = store.delete_memory(
            memory.memory_id,
            context=context,
            deleted_at_ms=9_999,
            run_id="delete-retry",
        )

        assert tombstone is not None
        assert retry == tombstone
        assert tombstone.reason is MemoryAuditReason.USER_REQUEST
        assert store.get_repository_memory(memory.memory_id) is None
        assert (
            store.search_repository_memories(
                "authentication",
                context=context,
                max_results=5,
                audit_at_ms=3_000,
            )
            == ()
        )
        assert store.list_memory_tombstones(
            context=context,
            max_results=5,
        ) == (tombstone,)
        assert (
            store.get_memory_tombstone(
                memory.memory_id,
                context=context,
            )
            == tombstone
        )
        events = store.list_memory_audit_events(memory.memory_id)
        with pytest.raises(MemoryStoreConflictError, match="resurrected"):
            store.put_repository_memory(memory, context=context)

    assert [event.action for event in events] == [
        MemoryAuditAction.CREATED,
        MemoryAuditAction.DELETED,
    ]
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM memory_entries").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM memory_sources").fetchone()[0] == 0
        )
        assert connection.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM memory_tombstones").fetchone()[0]
            == 1
        )


def test_delete_is_non_disclosing_across_repository_boundaries(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    _, other_context = _create_repository(tmp_path / "other")
    memory = _memory(context)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context)
        assert (
            store.delete_memory(
                memory.memory_id,
                context=other_context,
                deleted_at_ms=2_000,
            )
            is None
        )
        assert store.get_repository_memory(memory.memory_id) == memory
        assert (
            store.list_memory_tombstones(
                context=other_context,
                max_results=5,
            )
            == ()
        )


def test_delete_failure_rolls_back_audit_tombstone_and_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"

    def fail_tombstone_insert(store: MemoryStore, tombstone: object) -> None:
        del store, tombstone
        raise MemoryStoreIntegrityError("injected tombstone failure")

    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)
        monkeypatch.setattr(
            MemoryStore,
            "_insert_tombstone",
            fail_tombstone_insert,
        )
        with pytest.raises(MemoryStoreIntegrityError, match="injected"):
            store.delete_memory(
                memory.memory_id,
                context=context,
                deleted_at_ms=2_000,
            )
        assert store.get_repository_memory(memory.memory_id) == memory
        assert (
            store.list_memory_tombstones(
                context=context,
                max_results=5,
            )
            == ()
        )
        assert [
            event.action for event in store.list_memory_audit_events(memory.memory_id)
        ] == [MemoryAuditAction.CREATED]


def test_concurrent_delete_retries_converge_on_one_tombstone(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)

    def delete_once() -> object:
        with open_memory_store(database_path) as store:
            return store.delete_memory(
                memory.memory_id,
                context=context,
                deleted_at_ms=2_000,
                run_id="delete-1",
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        tombstones = tuple(executor.map(lambda _: delete_once(), range(2)))

    assert tombstones[0] is not None
    assert tombstones[0] == tombstones[1]
    with open_memory_store(database_path) as store:
        assert [
            event.action for event in store.list_memory_audit_events(memory.memory_id)
        ] == [MemoryAuditAction.CREATED, MemoryAuditAction.DELETED]


def test_retention_is_bounded_oldest_first_and_never_deletes_repository_facts(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    repository_memory = _memory(context)
    episodes = tuple(
        _episode(
            context,
            run_id=f"run-{turn_number}",
            thread_id="thread-retention",
            turn_number=turn_number,
            created_at_ms=turn_number * 1_000,
        )
        for turn_number in range(1, 5)
    )
    policy = create_memory_retention_policy(
        max_episodic_entries=1,
        max_deletions_per_run=2,
    )

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(repository_memory, context=context)
        for episode in episodes:
            store.put_episodic_memory(episode, context=context)
        first = store.apply_retention_policy(
            policy,
            context=context,
            evaluated_at_ms=5_000,
            run_id="retention-1",
        )
        assert first.candidate_count == 3
        assert first.deleted_memory_ids == tuple(
            episode.episode_id for episode in episodes[:2]
        )
        assert first.truncated
        assert store.get_repository_memory(repository_memory.memory_id) == (
            repository_memory
        )
        assert [
            episode.episode_id
            for episode in store.list_episodic_memories(
                context=context,
                max_results=10,
            )
        ] == [episodes[3].episode_id, episodes[2].episode_id]

        second = store.apply_retention_policy(
            policy,
            context=context,
            evaluated_at_ms=5_000,
            run_id="retention-2",
        )
        assert second.candidate_count == 1
        assert second.deleted_memory_ids == (episodes[2].episode_id,)
        assert not second.truncated
        assert store.list_episodic_memories(
            context=context,
            max_results=10,
        ) == (episodes[3],)
        assert all(
            tombstone.reason is MemoryAuditReason.RETENTION_POLICY
            for tombstone in store.list_memory_tombstones(
                context=context,
                max_results=10,
            )
        )


def test_retention_age_cutoff_and_policy_revalidation(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    episodes = (
        _episode(context, turn_number=1, created_at_ms=1_000),
        _episode(context, turn_number=2, created_at_ms=3_000),
        _episode(context, turn_number=3, created_at_ms=4_000),
    )
    policy = create_memory_retention_policy(
        max_episodic_entries=10,
        max_age_ms=1_500,
        max_deletions_per_run=10,
    )

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        for episode in episodes:
            store.put_episodic_memory(episode, context=context)
        report = store.apply_retention_policy(
            policy,
            context=context,
            evaluated_at_ms=5_000,
        )
        assert report.deleted_memory_ids == (
            episodes[0].episode_id,
            episodes[1].episode_id,
        )
        tampered = policy.model_copy(update={"max_episodic_entries": 0})
        with pytest.raises(MemoryStoreIntegrityError, match="revalidation"):
            store.apply_retention_policy(
                tampered,
                context=context,
                evaluated_at_ms=5_000,
            )


def test_v2_database_migrates_without_losing_live_memory(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)

    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE memory_tombstones")
        connection.execute("DROP TABLE memory_fact_keys")
        connection.execute(
            "UPDATE memory_schema SET value = '2' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()

    with open_memory_store(database_path) as migrated:
        assert migrated.schema_version == MEMORY_STORE_SCHEMA_VERSION
        assert migrated.get_repository_memory(memory.memory_id) == memory
        assert (
            migrated.list_memory_tombstones(
                context=context,
                max_results=5,
            )
            == ()
        )


def test_corrupt_tombstone_fails_closed_when_database_reopens(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)
        store.delete_memory(
            memory.memory_id,
            context=context,
            deleted_at_ms=2_000,
        )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_tombstones SET tombstone_json = '{}' WHERE memory_id = ?",
            (memory.memory_id,),
        )
        connection.commit()

    with pytest.raises(MemoryStoreMigrationError, match="tombstone"):
        with open_memory_store(database_path):
            pass
