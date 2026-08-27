"""Versioned SQLite/FTS5 storage boundary for RepoCompass memory."""

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryAuditEvent,
    MemoryAuditReason,
    MemoryFreshnessStatus,
    MemoryRetrievalHit,
    MemoryStalenessReason,
    RepositoryMemory,
    RepositoryMemorySource,
    create_memory_audit_event,
    create_memory_retrieval_hit,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.revalidation import (
    classify_repository_memory_freshness,
)
from repocompass.tools.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN
from repocompass.tools.gateway import ToolExecutionContext

MEMORY_STORE_SCHEMA_VERSION = 1
_SQLITE_TIMEOUT_SECONDS = 10.0
_FTS_TOKENIZER = "unicode61 remove_diacritics 2"
_MAX_MEMORY_QUERY_CHARACTERS = 512
_MAX_MEMORY_QUERY_TOKENS = 16
_MAX_MEMORY_QUERY_TOKEN_CHARACTERS = 64
_MAX_MEMORY_RESULTS = 50
_MEMORY_QUERY_TIMEOUT_SECONDS = 0.25
_QUERY_TOKEN_PATTERN = re.compile(r"[\w]+(?:[.:-][\w]+)*", re.UNICODE)


class MemoryStoreError(RuntimeError):
    """Base error for local memory persistence failures."""


class MemoryStoreMigrationError(MemoryStoreError):
    """The memory database version or physical schema is not supported."""


class MemoryStoreConflictError(MemoryStoreError):
    """An immutable memory identity already contains different data."""


class MemoryStoreIntegrityError(MemoryStoreError):
    """A memory write or current repository snapshot violates policy."""


class MemoryStoreCorruptionError(MemoryStoreError):
    """Persisted memory JSON, mirrored columns, or provenance is corrupt."""


class MemoryStoreQueryError(MemoryStoreError):
    """A bounded FTS query failed or exceeded its local deadline."""


def _normalize_schema_sql(value: str) -> str:
    return " ".join(value.lower().split())


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE memory_schema (
        key TEXT NOT NULL PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    INSERT INTO memory_schema (key, value) VALUES
        ('schema_version', '1'),
        ('fts_tokenizer', 'unicode61 remove_diacritics 2')
    """,
    """
    CREATE TABLE memory_entries (
        memory_id TEXT NOT NULL PRIMARY KEY CHECK(length(memory_id) = 64),
        memory_scope TEXT NOT NULL
            CHECK(memory_scope IN ('repository', 'episodic')),
        repository_key TEXT NOT NULL CHECK(length(repository_key) = 64),
        source_repository_id TEXT NOT NULL
            CHECK(length(source_repository_id) = 64),
        commit_sha TEXT,
        verification_status TEXT NOT NULL
            CHECK(verification_status IN ('verified', 'human_approved', 'observed')),
        write_authority TEXT NOT NULL
            CHECK(write_authority IN (
                'validated_evidence', 'human_approval', 'terminal_consolidation'
            )),
        title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 512),
        content TEXT NOT NULL CHECK(length(content) BETWEEN 1 AND 16384),
        tags_text TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
        record_json TEXT NOT NULL,
        CHECK(
            (
                memory_scope = 'repository'
                AND commit_sha IS NOT NULL
                AND (
                    (
                        verification_status = 'verified'
                        AND write_authority = 'validated_evidence'
                    )
                    OR (
                        verification_status = 'human_approved'
                        AND write_authority = 'human_approval'
                    )
                )
            )
            OR (
                memory_scope = 'episodic'
                AND verification_status = 'observed'
                AND write_authority = 'terminal_consolidation'
            )
        )
    )
    """,
    """
    CREATE TABLE memory_sources (
        memory_id TEXT NOT NULL,
        source_id TEXT NOT NULL CHECK(length(source_id) = 64),
        repository_id TEXT NOT NULL CHECK(length(repository_id) = 64),
        commit_sha TEXT NOT NULL,
        path TEXT NOT NULL,
        source_digest TEXT NOT NULL CHECK(length(source_digest) = 64),
        source_json TEXT NOT NULL,
        PRIMARY KEY(memory_id, source_id),
        FOREIGN KEY(memory_id) REFERENCES memory_entries(memory_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE memory_audit_events (
        event_id TEXT NOT NULL PRIMARY KEY CHECK(length(event_id) = 64),
        memory_id TEXT NOT NULL CHECK(length(memory_id) = 64),
        repository_key TEXT NOT NULL CHECK(length(repository_key) = 64),
        action TEXT NOT NULL
            CHECK(action IN (
                'created', 'reverified', 'retrieved', 'marked_stale', 'deleted'
            )),
        created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
        event_json TEXT NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE memory_fts USING fts5(
        title,
        content,
        tags_text,
        content='memory_entries',
        content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER memory_entries_fts_insert AFTER INSERT ON memory_entries BEGIN
        INSERT INTO memory_fts(rowid, title, content, tags_text)
        VALUES (new.rowid, new.title, new.content, new.tags_text);
    END
    """,
    """
    CREATE TRIGGER memory_entries_fts_delete AFTER DELETE ON memory_entries BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, title, content, tags_text)
        VALUES ('delete', old.rowid, old.title, old.content, old.tags_text);
    END
    """,
    """
    CREATE TRIGGER memory_entries_fts_update AFTER UPDATE ON memory_entries BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, title, content, tags_text)
        VALUES ('delete', old.rowid, old.title, old.content, old.tags_text);
        INSERT INTO memory_fts(rowid, title, content, tags_text)
        VALUES (new.rowid, new.title, new.content, new.tags_text);
    END
    """,
    "CREATE INDEX idx_memory_entries_repository_commit "
    "ON memory_entries(repository_key, commit_sha, memory_scope)",
    "CREATE INDEX idx_memory_sources_repository_path "
    "ON memory_sources(repository_id, commit_sha, path)",
    "CREATE INDEX idx_memory_audit_target_time "
    "ON memory_audit_events(memory_id, created_at_ms)",
)

_REQUIRED_COLUMNS = {
    "memory_schema": ("key", "value"),
    "memory_entries": (
        "memory_id",
        "memory_scope",
        "repository_key",
        "source_repository_id",
        "commit_sha",
        "verification_status",
        "write_authority",
        "title",
        "content",
        "tags_text",
        "created_at_ms",
        "record_json",
    ),
    "memory_sources": (
        "memory_id",
        "source_id",
        "repository_id",
        "commit_sha",
        "path",
        "source_digest",
        "source_json",
    ),
    "memory_audit_events": (
        "event_id",
        "memory_id",
        "repository_key",
        "action",
        "created_at_ms",
        "event_json",
    ),
}
_REQUIRED_NOT_NULL = {
    "memory_schema": frozenset(_REQUIRED_COLUMNS["memory_schema"]),
    "memory_entries": frozenset(
        {
            "memory_id",
            "memory_scope",
            "repository_key",
            "source_repository_id",
            "verification_status",
            "write_authority",
            "title",
            "content",
            "tags_text",
            "created_at_ms",
            "record_json",
        }
    ),
    "memory_sources": frozenset(_REQUIRED_COLUMNS["memory_sources"]),
    "memory_audit_events": frozenset(_REQUIRED_COLUMNS["memory_audit_events"]),
}
_EXPECTED_TRIGGER_SQL = {
    "memory_entries_fts_insert": _normalize_schema_sql(_SCHEMA_STATEMENTS[6]),
    "memory_entries_fts_delete": _normalize_schema_sql(_SCHEMA_STATEMENTS[7]),
    "memory_entries_fts_update": _normalize_schema_sql(_SCHEMA_STATEMENTS[8]),
}
_ForeignKeyEdge = tuple[str, str, str, str]
_EXPECTED_FOREIGN_KEYS: dict[str, Counter[_ForeignKeyEdge]] = {
    "memory_schema": Counter(),
    "memory_entries": Counter(),
    "memory_sources": Counter(
        [("memory_entries", "memory_id", "memory_id", "CASCADE")]
    ),
    "memory_audit_events": Counter(),
}
_EXPECTED_METADATA = {
    "schema_version": str(MEMORY_STORE_SCHEMA_VERSION),
    "fts_tokenizer": _FTS_TOKENIZER,
}


@dataclass(frozen=True, slots=True)
class RepositoryMemorySearchResult:
    """Strict memory contents paired with current-snapshot trust metadata."""

    memory: RepositoryMemory
    hit: MemoryRetrievalHit


class MemoryStore:
    """Own one validated memory SQLite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = threading.RLock()

    @property
    def schema_version(self) -> int:
        """Return the physical memory schema version."""

        with self._lock:
            row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    @property
    def fts5_enabled(self) -> bool:
        """Report whether the validated FTS5 table is queryable."""

        with self._lock:
            self._connection.execute("SELECT count(*) FROM memory_fts").fetchone()
        return True

    def put_repository_memory(
        self,
        memory: RepositoryMemory,
        *,
        context: ToolExecutionContext,
        run_id: str | None = None,
    ) -> bool:
        """Validate and atomically persist one immutable repository fact."""

        memory = _revalidate_repository_memory_contract(memory)
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("context must be a ToolExecutionContext")
        _validate_optional_identifier(run_id, name="run_id")
        identity = resolve_memory_repository_identity(context)
        if (
            memory.repository_key != identity.repository_key
            or memory.source_repository_id != context.repository_id
        ):
            raise MemoryStoreIntegrityError(
                "repository memory identity does not match trusted context"
            )
        if context.commit_sha is None or memory.commit_sha != context.commit_sha:
            raise MemoryStoreIntegrityError(
                "repository memory commit does not match trusted context"
            )
        freshness, reason = classify_repository_memory_freshness(memory, context)
        if freshness is not MemoryFreshnessStatus.CURRENT:
            if reason is MemoryStalenessReason.DIRTY_WORKTREE:
                raise MemoryStoreIntegrityError(
                    "repository memory writes require a clean snapshot"
                )
            raise MemoryStoreIntegrityError(
                "repository memory source is not current and verified"
            )

        record_json = _canonical_model_json(memory)
        event = create_memory_audit_event(
            memory_id=memory.memory_id,
            repository_key=memory.repository_key,
            action=MemoryAuditAction.CREATED,
            reason=MemoryAuditReason.WRITE_POLICY_ALLOWED,
            created_at_ms=memory.created_at_ms,
            run_id=run_id,
        )
        inserted = False
        with self._transaction():
            existing_row = self._connection.execute(
                "SELECT * FROM memory_entries WHERE memory_id = ?",
                (memory.memory_id,),
            ).fetchone()
            if existing_row is None:
                try:
                    self._connection.execute(
                        """
                        INSERT INTO memory_entries (
                            memory_id, memory_scope, repository_key,
                            source_repository_id, commit_sha,
                            verification_status, write_authority, title,
                            content, tags_text, created_at_ms, record_json
                        ) VALUES (?, 'repository', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory.memory_id,
                            memory.repository_key,
                            memory.source_repository_id,
                            memory.commit_sha,
                            memory.verification_status.value,
                            memory.write_authority.value,
                            memory.title,
                            memory.content,
                            " ".join(memory.tags),
                            memory.created_at_ms,
                            record_json,
                        ),
                    )
                    for source in memory.sources:
                        self._insert_repository_source(memory.memory_id, source)
                except sqlite3.IntegrityError as error:
                    raise MemoryStoreIntegrityError(
                        "repository memory violates storage integrity"
                    ) from error
                inserted = True
            else:
                existing = _decode_repository_memory(
                    self._connection,
                    existing_row,
                )
                if existing != memory:
                    raise MemoryStoreConflictError(
                        "memory identity already exists with different contents"
                    )
            self._insert_audit_event(event)
        return inserted

    def get_repository_memory(self, memory_id: str) -> RepositoryMemory | None:
        """Strictly load one repository fact and every source row."""

        _validate_digest(memory_id, name="memory_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_entries WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            return _decode_repository_memory(self._connection, row)

    def search_repository_memories(
        self,
        query: str,
        *,
        context: ToolExecutionContext,
        max_results: int,
        audit_at_ms: int,
        run_id: str | None = None,
    ) -> tuple[RepositoryMemorySearchResult, ...]:
        """Run one bounded plain-text FTS query and classify every hit."""

        if not isinstance(context, ToolExecutionContext):
            raise TypeError("context must be a ToolExecutionContext")
        if (
            type(max_results) is not int
            or max_results < 1
            or max_results > _MAX_MEMORY_RESULTS
        ):
            raise ValueError("max_results is outside the memory query limit")
        if type(audit_at_ms) is not int or audit_at_ms < 0:
            raise ValueError("audit_at_ms must be a non-negative integer")
        _validate_optional_identifier(run_id, name="run_id")
        normalized_query, fts_query = _bounded_fts_query(query)
        query_digest = hashlib.sha256(
            b"repocompass:memory-query:v1\0" + normalized_query.encode("utf-8")
        ).hexdigest()
        identity = resolve_memory_repository_identity(context)
        with self._lock:
            rows = self._execute_bounded_search(
                fts_query=fts_query,
                repository_key=identity.repository_key,
                max_results=max_results,
            )
            memories = tuple(
                _decode_repository_memory(self._connection, row) for row in rows
            )

        results: list[RepositoryMemorySearchResult] = []
        events: list[MemoryAuditEvent] = []
        for rank, memory in enumerate(memories, start=1):
            freshness, reason = classify_repository_memory_freshness(memory, context)
            hit = create_memory_retrieval_hit(
                query_digest=query_digest,
                memory_id=memory.memory_id,
                repository_key=memory.repository_key,
                memory_source_repository_id=memory.source_repository_id,
                current_source_repository_id=context.repository_id,
                memory_commit_sha=memory.commit_sha,
                current_commit_sha=context.commit_sha,
                rank=rank,
                freshness=freshness,
                staleness_reason=reason,
                original_verification_status=memory.verification_status,
            )
            results.append(RepositoryMemorySearchResult(memory=memory, hit=hit))
            if freshness is MemoryFreshnessStatus.CURRENT:
                action = MemoryAuditAction.RETRIEVED
                audit_reason = MemoryAuditReason.CURRENT_SNAPSHOT_HIT
            elif freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED:
                action = MemoryAuditAction.RETRIEVED
                audit_reason = MemoryAuditReason.REVALIDATION_REQUIRED
            else:
                action = MemoryAuditAction.MARKED_STALE
                audit_reason = MemoryAuditReason.SOURCE_STALE
            events.append(
                create_memory_audit_event(
                    memory_id=memory.memory_id,
                    repository_key=memory.repository_key,
                    action=action,
                    reason=audit_reason,
                    created_at_ms=audit_at_ms,
                    run_id=run_id,
                )
            )
        if events:
            with self._transaction():
                for event in events:
                    self._insert_audit_event(event)
        return tuple(results)

    def list_memory_audit_events(
        self,
        memory_id: str,
    ) -> tuple[MemoryAuditEvent, ...]:
        """List one memory's audit trail without exposing raw actor identity."""

        _validate_digest(memory_id, name="memory_id")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_audit_events WHERE memory_id = ? "
                "ORDER BY created_at_ms, event_id",
                (memory_id,),
            ).fetchall()
        return tuple(_decode_audit_event(row) for row in rows)

    def _insert_repository_source(
        self,
        memory_id: str,
        source: RepositoryMemorySource,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO memory_sources (
                memory_id, source_id, repository_id, commit_sha,
                path, source_digest, source_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                source.source_id,
                source.location.repository_id,
                source.location.commit_sha,
                source.location.path,
                source.source_digest,
                _canonical_model_json(source),
            ),
        )

    def _insert_audit_event(self, event: MemoryAuditEvent) -> None:
        event_json = _canonical_model_json(event)
        existing = self._connection.execute(
            "SELECT event_json FROM memory_audit_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["event_json"]) != event_json:
                raise MemoryStoreCorruptionError(
                    "content-addressed memory audit event conflicts"
                )
            return
        self._connection.execute(
            """
            INSERT INTO memory_audit_events (
                event_id, memory_id, repository_key, action,
                created_at_ms, event_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.memory_id,
                event.repository_key,
                event.action.value,
                event.created_at_ms,
                event_json,
            ),
        )

    def _execute_bounded_search(
        self,
        *,
        fts_query: str,
        repository_key: str,
        max_results: int,
    ) -> tuple[sqlite3.Row, ...]:
        deadline = time.monotonic() + _MEMORY_QUERY_TIMEOUT_SECONDS
        self._connection.set_progress_handler(
            lambda: int(time.monotonic() > deadline),
            1_000,
        )
        try:
            rows = self._connection.execute(
                """
                SELECT memory_entries.*
                FROM memory_fts
                JOIN memory_entries ON memory_entries.rowid = memory_fts.rowid
                WHERE memory_fts MATCH ?
                  AND memory_entries.repository_key = ?
                  AND memory_entries.memory_scope = 'repository'
                ORDER BY bm25(memory_fts), memory_entries.memory_id
                LIMIT ?
                """,
                (fts_query, repository_key, max_results),
            ).fetchall()
        except sqlite3.OperationalError as error:
            raise MemoryStoreQueryError("bounded memory query failed") from error
        finally:
            self._connection.set_progress_handler(None, 0)
        return tuple(rows)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise


@contextmanager
def open_memory_store(database_path: Path) -> Iterator[MemoryStore]:
    """Open, migrate, validate, and close one local memory database."""

    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a pathlib.Path")
    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(
            str(resolved_path),
            timeout=_SQLITE_TIMEOUT_SECONDS,
            check_same_thread=False,
        )
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        _migrate_and_validate(connection)
        yield MemoryStore(connection)


def _migrate_and_validate(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > MEMORY_STORE_SCHEMA_VERSION:
            raise MemoryStoreMigrationError(
                "Memory Store schema is newer than this application"
            )
        if version == 0:
            if _user_schema_objects(connection):
                raise MemoryStoreMigrationError(
                    "unversioned nonempty memory database cannot be migrated safely"
                )
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {MEMORY_STORE_SCHEMA_VERSION}")
        elif version != MEMORY_STORE_SCHEMA_VERSION:
            raise MemoryStoreMigrationError(
                f"Unsupported Memory Store schema version: {version}"
            )
        connection.commit()
    except MemoryStoreMigrationError:
        connection.rollback()
        raise
    except sqlite3.DatabaseError as error:
        connection.rollback()
        raise MemoryStoreMigrationError(
            "Memory Store schema migration failed"
        ) from error
    _validate_schema(connection)


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = _schema_object_names(connection, "table")
    if not {*_REQUIRED_COLUMNS, "memory_fts"}.issubset(tables):
        raise MemoryStoreMigrationError(
            "Memory Store schema is missing required tables"
        )

    if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise MemoryStoreMigrationError("Memory Store foreign keys are disabled")
    for table, expected in _EXPECTED_FOREIGN_KEYS.items():
        actual = Counter(
            (
                str(row["table"]),
                str(row["from"]),
                str(row["to"]),
                str(row["on_delete"]).upper(),
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        if actual != expected:
            raise MemoryStoreMigrationError(
                f"Memory Store foreign key schema for {table} is incompatible"
            )

    for table, expected_columns in _REQUIRED_COLUMNS.items():
        column_rows = tuple(connection.execute(f"PRAGMA table_info({table})"))
        actual_columns = tuple(str(row["name"]) for row in column_rows)
        if actual_columns != expected_columns:
            raise MemoryStoreMigrationError(
                f"Memory Store schema for {table} is incompatible"
            )
        not_null_columns = {
            str(row["name"]) for row in column_rows if int(row["notnull"]) == 1
        }
        if not _REQUIRED_NOT_NULL[table].issubset(not_null_columns):
            raise MemoryStoreMigrationError(
                f"Memory Store nullability for {table} is incompatible"
            )

    trigger_sql = {
        str(row["name"]): _normalize_schema_sql(str(row["sql"]))
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    if trigger_sql != _EXPECTED_TRIGGER_SQL:
        raise MemoryStoreMigrationError(
            "Memory Store FTS trigger schema is incompatible"
        )

    metadata = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM memory_schema")
    }
    if metadata != _EXPECTED_METADATA:
        raise MemoryStoreMigrationError("Memory Store schema metadata is incompatible")

    fts_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
    ).fetchone()
    fts_sql = "" if fts_row is None else str(fts_row["sql"])
    normalized_fts_sql = " ".join(fts_sql.lower().split())
    if (
        "using fts5" not in normalized_fts_sql
        or "unicode61 remove_diacritics 2" not in normalized_fts_sql
        or "content='memory_entries'" not in normalized_fts_sql
    ):
        raise MemoryStoreMigrationError("Memory Store FTS5 schema is incompatible")
    try:
        connection.execute("SELECT count(*) FROM memory_fts").fetchone()
    except sqlite3.DatabaseError as error:
        raise MemoryStoreMigrationError(
            "Memory Store FTS5 index is unavailable"
        ) from error
    _validate_fts_integrity(connection)


def _validate_fts_integrity(connection: sqlite3.Connection) -> None:
    connection.execute("SAVEPOINT repocompass_memory_fts_integrity")
    try:
        connection.execute(
            "INSERT INTO memory_fts(memory_fts, rank) VALUES ('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as error:
        connection.execute("ROLLBACK TO repocompass_memory_fts_integrity")
        connection.execute("RELEASE repocompass_memory_fts_integrity")
        raise MemoryStoreMigrationError(
            "Memory Store FTS5 external-content integrity check failed"
        ) from error
    else:
        connection.execute("RELEASE repocompass_memory_fts_integrity")


def _revalidate_repository_memory_contract(value: object) -> RepositoryMemory:
    if not isinstance(value, RepositoryMemory):
        raise TypeError("memory must be a RepositoryMemory")
    try:
        return RepositoryMemory.model_validate_json(
            value.model_dump_json(exclude_computed_fields=True, warnings=False)
        )
    except (ValidationError, ValueError, TypeError) as error:
        raise MemoryStoreIntegrityError(
            "repository memory contract failed revalidation"
        ) from error


def _decode_repository_memory(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> RepositoryMemory:
    if str(row["memory_scope"]) != "repository":
        raise MemoryStoreCorruptionError("stored memory is not a repository record")
    try:
        memory = RepositoryMemory.model_validate_json(str(row["record_json"]))
    except (ValidationError, ValueError, TypeError) as error:
        raise MemoryStoreCorruptionError(
            "stored repository memory record is invalid"
        ) from error
    mirrored = (
        memory.memory_id,
        memory.repository_key,
        memory.source_repository_id,
        memory.commit_sha,
        memory.verification_status.value,
        memory.write_authority.value,
        memory.title,
        memory.content,
        " ".join(memory.tags),
        memory.created_at_ms,
    )
    stored = (
        str(row["memory_id"]),
        str(row["repository_key"]),
        str(row["source_repository_id"]),
        str(row["commit_sha"]),
        str(row["verification_status"]),
        str(row["write_authority"]),
        str(row["title"]),
        str(row["content"]),
        str(row["tags_text"]),
        int(row["created_at_ms"]),
    )
    if mirrored != stored:
        raise MemoryStoreCorruptionError(
            "stored repository memory mirrored columns disagree"
        )
    source_rows = connection.execute(
        "SELECT * FROM memory_sources WHERE memory_id = ? ORDER BY source_id",
        (memory.memory_id,),
    ).fetchall()
    sources = tuple(_decode_repository_source(source_row) for source_row in source_rows)
    if sources != memory.sources:
        raise MemoryStoreCorruptionError(
            "stored repository memory source set disagrees with record"
        )
    return memory


def _decode_repository_source(row: sqlite3.Row) -> RepositoryMemorySource:
    try:
        source = RepositoryMemorySource.model_validate_json(str(row["source_json"]))
    except (ValidationError, ValueError, TypeError) as error:
        raise MemoryStoreCorruptionError(
            "stored repository memory source is invalid"
        ) from error
    mirrored = (
        source.source_id,
        source.location.repository_id,
        source.location.commit_sha,
        source.location.path,
        source.source_digest,
    )
    stored = (
        str(row["source_id"]),
        str(row["repository_id"]),
        str(row["commit_sha"]),
        str(row["path"]),
        str(row["source_digest"]),
    )
    if mirrored != stored:
        raise MemoryStoreCorruptionError(
            "stored repository memory source columns disagree"
        )
    return source


def _decode_audit_event(row: sqlite3.Row) -> MemoryAuditEvent:
    try:
        event = MemoryAuditEvent.model_validate_json(str(row["event_json"]))
    except (ValidationError, ValueError, TypeError) as error:
        raise MemoryStoreCorruptionError(
            "stored memory audit event is invalid"
        ) from error
    mirrored = (
        event.event_id,
        event.memory_id,
        event.repository_key,
        event.action.value,
        event.created_at_ms,
    )
    stored = (
        str(row["event_id"]),
        str(row["memory_id"]),
        str(row["repository_key"]),
        str(row["action"]),
        int(row["created_at_ms"]),
    )
    if mirrored != stored:
        raise MemoryStoreCorruptionError("stored memory audit columns disagree")
    return event


def _bounded_fts_query(query: str) -> tuple[str, str]:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = " ".join(query.split())
    if (
        not normalized
        or len(normalized) > _MAX_MEMORY_QUERY_CHARACTERS
        or "\x00" in query
    ):
        raise ValueError("query is empty or exceeds the memory query limit")
    raw_tokens = _QUERY_TOKEN_PATTERN.findall(normalized.casefold())
    if (
        not raw_tokens
        or len(raw_tokens) > _MAX_MEMORY_QUERY_TOKENS
        or any(len(token) > _MAX_MEMORY_QUERY_TOKEN_CHARACTERS for token in raw_tokens)
    ):
        raise ValueError("query has no safe bounded search tokens")
    tokens = tuple(dict.fromkeys(raw_tokens))
    return " ".join(tokens), " OR ".join(f'"{token}"' for token in tokens)


def _canonical_model_json(model: object) -> str:
    if not hasattr(model, "model_dump"):
        raise TypeError("model must be a Pydantic domain object")
    payload = model.model_dump(mode="json", exclude_computed_fields=True)
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_digest(value: object, *, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(SHA256_PATTERN, value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _validate_optional_identifier(value: str | None, *, name: str) -> None:
    if value is not None and re.fullmatch(IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"{name} is not a canonical identifier")


def _user_schema_objects(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    return {
        (str(row["type"]), str(row["name"]))
        for row in connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'trigger', 'index')"
        )
    }


def _schema_object_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'",
            (object_type,),
        )
    }
