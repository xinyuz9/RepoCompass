"""Transactional repository-memory write, retrieval, and corruption tests."""

import hashlib
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repocompass.agent.context import RepositoryLocation
from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryFactKind,
    MemoryFreshnessStatus,
    MemoryStalenessReason,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    RepositoryMemory,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import (
    MemoryStore,
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
    (repository / "src").mkdir()
    (repository / "src" / "api.py").write_text(
        "def authenticate(request):\n    return request.user is not None\n",
        encoding="utf-8",
    )
    _git(repository, "add", "src/api.py")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository, ToolExecutionContext(repository)


def _memory(
    context: ToolExecutionContext,
    *,
    source_digest: str | None = None,
    created_at_ms: int = 1_000,
) -> RepositoryMemory:
    identity = resolve_memory_repository_identity(context)
    commit_sha = context.commit_sha
    assert commit_sha is not None
    path = context.repository_root / "src" / "api.py"
    digest = source_digest or hashlib.sha256(path.read_bytes()).hexdigest()
    source = create_repository_memory_source(
        location=RepositoryLocation(
            repository_id=context.repository_id,
            commit_sha=commit_sha,
            path="src/api.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
        ),
        source_digest=digest,
        evidence_claim_digest=hashlib.sha256(b"auth entry point").hexdigest(),
    )
    return create_repository_memory(
        identity=identity,
        commit_sha=commit_sha,
        kind=MemoryFactKind.ENTRY_POINT,
        title="Authentication entry point",
        content="src.api.authenticate is the authentication entry point.",
        verification_status=MemoryVerificationStatus.VERIFIED,
        write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
        sources=(source,),
        tags=("authentication", "entry-point"),
        created_at_ms=created_at_ms,
        verified_at_ms=created_at_ms,
    )


def test_put_get_search_and_audit_repository_memory(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        assert store.put_repository_memory(
            memory,
            context=context,
            run_id="run-1",
        )
        assert store.get_repository_memory(memory.memory_id) == memory
        results = store.search_repository_memories(
            "authentication handler",
            context=context,
            max_results=5,
            audit_at_ms=1_500,
            run_id="run-2",
        )
        events = store.list_memory_audit_events(memory.memory_id)

    assert len(results) == 1
    assert results[0].memory == memory
    assert results[0].hit.memory_id == memory.memory_id
    assert results[0].hit.rank == 1
    assert results[0].hit.freshness is MemoryFreshnessStatus.CURRENT
    assert [event.action for event in events] == [
        MemoryAuditAction.CREATED,
        MemoryAuditAction.RETRIEVED,
    ]


def test_identical_retry_is_idempotent_with_one_created_audit(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        assert store.put_repository_memory(memory, context=context)
        assert not store.put_repository_memory(memory, context=context)
        events = store.list_memory_audit_events(memory.memory_id)

    assert len(events) == 1
    assert events[0].action is MemoryAuditAction.CREATED


def test_write_revalidates_source_digest_before_transaction(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context, source_digest="f" * 64)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreIntegrityError, match="source"):
            store.put_repository_memory(memory, context=context)
        assert store.get_repository_memory(memory.memory_id) is None


def test_write_rejects_model_shaped_dictionary_and_dirty_snapshot(
    tmp_path: Path,
) -> None:
    repository, context = _create_repository(tmp_path)
    memory = _memory(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(TypeError, match="RepositoryMemory"):
            store.put_repository_memory(  # type: ignore[arg-type]
                memory.model_dump(mode="json"),
                context=context,
            )
        (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        with pytest.raises(MemoryStoreIntegrityError, match="clean"):
            store.put_repository_memory(memory, context=context)


def test_write_rejects_repository_identity_or_commit_mismatch(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    other_root = tmp_path / "other"
    _, other_context = _create_repository(other_root)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(MemoryStoreIntegrityError, match="identity"):
            store.put_repository_memory(memory, context=other_context)

        wrong_commit = memory.model_copy(update={"commit_sha": "f" * 40})
        with pytest.raises(MemoryStoreIntegrityError, match="contract"):
            store.put_repository_memory(wrong_commit, context=context)


def test_source_symlink_is_rejected_without_following_target(tmp_path: Path) -> None:
    repository, context = _create_repository(tmp_path)
    memory = _memory(context)
    source = repository / "src" / "api.py"
    original = repository / "original.py"
    source.rename(original)
    source.symlink_to(original)

    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(MemoryStoreIntegrityError, match="source"):
            store.put_repository_memory(memory, context=context)


def test_search_marks_new_commit_as_revalidation_required(tmp_path: Path) -> None:
    repository, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)

    (repository / "src" / "api.py").write_text(
        "def authenticate(request):\n    return bool(request.user)\n",
        encoding="utf-8",
    )
    _git(repository, "add", "src/api.py")
    _git(repository, "commit", "--quiet", "-m", "change authentication")
    current_context = ToolExecutionContext(repository)
    with open_memory_store(database_path) as store:
        result = store.search_repository_memories(
            "authentication",
            context=current_context,
            max_results=5,
            audit_at_ms=2_000,
        )[0]

    assert result.hit.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED
    assert result.hit.staleness_reason is MemoryStalenessReason.COMMIT_MISMATCH


def test_search_marks_changed_or_missing_source_stale(tmp_path: Path) -> None:
    repository, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)

    source = repository / "src" / "api.py"
    source.write_text("changed = True\n", encoding="utf-8")
    with open_memory_store(database_path) as store:
        changed = store.search_repository_memories(
            "authentication",
            context=context,
            max_results=5,
            audit_at_ms=2_100,
        )[0]
    assert changed.hit.freshness is MemoryFreshnessStatus.STALE
    assert changed.hit.staleness_reason is MemoryStalenessReason.SOURCE_CHANGED

    source.unlink()
    with open_memory_store(database_path) as store:
        missing = store.search_repository_memories(
            "authentication",
            context=context,
            max_results=5,
            audit_at_ms=2_200,
        )[0]
    assert missing.hit.freshness is MemoryFreshnessStatus.STALE
    assert missing.hit.staleness_reason is MemoryStalenessReason.SOURCE_MISSING


def test_search_marks_unrelated_dirty_worktree_for_revalidation(
    tmp_path: Path,
) -> None:
    repository, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)
    (repository / "notes.txt").write_text("unrelated dirty file\n", encoding="utf-8")

    with open_memory_store(database_path) as store:
        result = store.search_repository_memories(
            "authentication",
            context=context,
            max_results=5,
            audit_at_ms=2_000,
        )[0]

    assert result.hit.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED
    assert result.hit.staleness_reason is MemoryStalenessReason.DIRTY_WORKTREE


def test_equivalent_remote_clone_requires_source_revalidation(
    tmp_path: Path,
) -> None:
    seed, _ = _create_repository(tmp_path / "seed-root")
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(seed), str(remote)],
        check=True,
        capture_output=True,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    subprocess.run(
        ["git", "clone", "--quiet", str(remote), str(first)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "clone", "--quiet", str(remote), str(second)],
        check=True,
        capture_output=True,
    )
    first_context = ToolExecutionContext(first)
    second_context = ToolExecutionContext(second)
    memory = _memory(first_context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=first_context)
        result = store.search_repository_memories(
            "authentication",
            context=second_context,
            max_results=5,
            audit_at_ms=2_000,
        )[0]

    assert result.hit.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED
    assert (
        result.hit.staleness_reason is MemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH
    )


@pytest.mark.parametrize(
    "query",
    ["", "   ", "\x00", "!@#$", "token " * 17, "x" * 513],
)
def test_search_rejects_unsafe_or_unbounded_plain_text_query(
    tmp_path: Path,
    query: str,
) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(ValueError, match="query"):
            store.search_repository_memories(
                query,
                context=context,
                max_results=5,
                audit_at_ms=2_000,
            )


def test_search_rejects_boolean_limit_and_does_not_expose_fts_syntax(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        with pytest.raises(ValueError, match="max_results"):
            store.search_repository_memories(
                "authentication OR *",
                context=context,
                max_results=True,  # type: ignore[arg-type]
                audit_at_ms=2_000,
            )
        assert (
            store.search_repository_memories(
                'authentication" OR secret*',
                context=context,
                max_results=5,
                audit_at_ms=2_000,
            )
            == ()
        )


def test_corrupt_entry_or_source_json_fails_closed_on_read(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path) as store:
        store.put_repository_memory(memory, context=context)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_entries SET record_json = '{}' WHERE memory_id = ?",
            (memory.memory_id,),
        )
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreCorruptionError, match="record"):
            store.get_repository_memory(memory.memory_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_entries SET record_json = ? WHERE memory_id = ?",
            (memory.model_dump_json(), memory.memory_id),
        )
        connection.execute(
            "UPDATE memory_sources SET source_json = '{}' WHERE memory_id = ?",
            (memory.memory_id,),
        )
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreCorruptionError, match="source"):
            store.get_repository_memory(memory.memory_id)


def test_source_insert_failure_rolls_back_entry_sources_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"

    def fail_source_insert(
        store: MemoryStore,
        memory_id: str,
        source: object,
    ) -> None:
        del store, memory_id, source
        raise sqlite3.IntegrityError("injected source failure")

    monkeypatch.setattr(
        MemoryStore,
        "_insert_repository_source",
        fail_source_insert,
    )
    with open_memory_store(database_path) as store:
        with pytest.raises(MemoryStoreIntegrityError, match="storage integrity"):
            store.put_repository_memory(memory, context=context)
        assert store.get_repository_memory(memory.memory_id) is None
        assert store.list_memory_audit_events(memory.memory_id) == ()

    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM memory_entries").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM memory_sources").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM memory_audit_events").fetchone()[0]
            == 0
        )


def test_reopen_preserves_record_and_idempotent_created_event(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"

    with open_memory_store(database_path) as store:
        assert store.put_repository_memory(memory, context=context)

    with open_memory_store(database_path) as reopened:
        assert reopened.get_repository_memory(memory.memory_id) == memory
        assert not reopened.put_repository_memory(memory, context=context)
        events = reopened.list_memory_audit_events(memory.memory_id)

    assert len(events) == 1
    assert events[0].action is MemoryAuditAction.CREATED


def test_concurrent_identical_writers_are_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path):
        pass

    def write_once() -> bool:
        with open_memory_store(database_path) as store:
            return store.put_repository_memory(memory, context=context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(executor.map(lambda _: write_once(), range(2)))

    assert results == [False, True]
    with open_memory_store(database_path) as store:
        assert store.get_repository_memory(memory.memory_id) == memory
        assert len(store.list_memory_audit_events(memory.memory_id)) == 1
