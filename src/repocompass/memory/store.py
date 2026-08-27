"""Versioned SQLite/FTS5 storage boundary for RepoCompass memory."""

import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

MEMORY_STORE_SCHEMA_VERSION = 1
_SQLITE_TIMEOUT_SECONDS = 10.0
_FTS_TOKENIZER = "unicode61 remove_diacritics 2"


class MemoryStoreError(RuntimeError):
    """Base error for local memory persistence failures."""


class MemoryStoreMigrationError(MemoryStoreError):
    """The memory database version or physical schema is not supported."""


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
