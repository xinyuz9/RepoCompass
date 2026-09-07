"""Versioned memory-store migration, FTS5, and schema-boundary tests."""

import sqlite3
from pathlib import Path

import pytest

from repocompass.memory.store import (
    MEMORY_STORE_SCHEMA_VERSION,
    MemoryStoreMigrationError,
    open_memory_store,
)

REQUIRED_TABLES = {
    "memory_schema",
    "memory_entries",
    "memory_sources",
    "memory_audit_events",
    "memory_episodes",
    "memory_tombstones",
    "memory_fts",
}
REQUIRED_TRIGGERS = {
    "memory_entries_fts_insert",
    "memory_entries_fts_delete",
    "memory_entries_fts_update",
}
REQUIRED_INDEXES = {
    "idx_memory_episodes_repository_thread_turn",
    "idx_memory_episodes_repository_run",
    "idx_memory_tombstones_repository_time",
}


def _user_version(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _schema_objects(database_path: Path, object_type: str) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?",
                (object_type,),
            )
            if not str(row[0]).startswith("sqlite_")
        }


def _foreign_key_edges(
    database_path: Path,
    table: str,
) -> list[tuple[str, str, str, str]]:
    with sqlite3.connect(database_path) as connection:
        return sorted(
            (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )


def test_fresh_memory_store_creates_exact_v3_schema_and_fts5(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "memory.sqlite"

    with open_memory_store(database_path) as store:
        assert store.schema_version == MEMORY_STORE_SCHEMA_VERSION
        assert store.fts5_enabled is True

    assert database_path.is_file()
    assert _user_version(database_path) == MEMORY_STORE_SCHEMA_VERSION
    assert REQUIRED_TABLES.issubset(_schema_objects(database_path, "table"))
    assert _schema_objects(database_path, "trigger") == REQUIRED_TRIGGERS
    assert REQUIRED_INDEXES.issubset(_schema_objects(database_path, "index"))
    assert _foreign_key_edges(database_path, "memory_sources") == [
        ("memory_entries", "memory_id", "memory_id", "CASCADE")
    ]
    assert _foreign_key_edges(database_path, "memory_episodes") == [
        ("memory_entries", "episode_id", "memory_id", "CASCADE")
    ]
    assert _foreign_key_edges(database_path, "memory_tombstones") == [
        ("memory_audit_events", "deletion_event_id", "event_id", "RESTRICT")
    ]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT value FROM memory_schema WHERE key = 'schema_version'"
        ).fetchone() == (str(MEMORY_STORE_SCHEMA_VERSION),)
        assert connection.execute(
            "SELECT value FROM memory_schema WHERE key = 'fts_tokenizer'"
        ).fetchone() == ("unicode61 remove_diacritics 2",)
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_fts_triggers_index_update_and_remove_memory_content(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path):
        pass

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO memory_entries (
                memory_id, memory_scope, repository_key, source_repository_id,
                commit_sha, verification_status, write_authority, title,
                content, tags_text, created_at_ms, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "a" * 64,
                "repository",
                "b" * 64,
                "c" * 64,
                "d" * 40,
                "verified",
                "validated_evidence",
                "Authentication entry point",
                "The request handler delegates to the auth service.",
                "api auth",
                1,
                "{}",
            ),
        )
        assert connection.execute(
            """
            SELECT memory_entries.memory_id
            FROM memory_fts
            JOIN memory_entries ON memory_entries.rowid = memory_fts.rowid
            WHERE memory_fts MATCH 'authentication'
            """
        ).fetchall() == [("a" * 64,)]

        connection.execute(
            "UPDATE memory_entries SET title = 'Request routing' WHERE memory_id = ?",
            ("a" * 64,),
        )
        assert (
            connection.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'authentication'"
            ).fetchall()
            == []
        )
        assert connection.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'routing'"
        ).fetchall()

        connection.execute(
            "DELETE FROM memory_entries WHERE memory_id = ?",
            ("a" * 64,),
        )
        assert (
            connection.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'routing'"
            ).fetchall()
            == []
        )


def test_reopening_current_memory_schema_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite"
    with open_memory_store(database_path):
        pass
    first_tables = _schema_objects(database_path, "table")
    first_triggers = _schema_objects(database_path, "trigger")

    with open_memory_store(database_path) as store:
        assert store.schema_version == MEMORY_STORE_SCHEMA_VERSION

    assert _schema_objects(database_path, "table") == first_tables
    assert _schema_objects(database_path, "trigger") == first_triggers


def _downgrade_empty_current_store_to_v1(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE memory_tombstones")
        connection.execute("DROP TABLE memory_fact_keys")
        connection.execute("DROP TABLE memory_episodes")
        connection.execute(
            "UPDATE memory_schema SET value = '1' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 1")


def test_v1_memory_store_migrates_to_current_without_losing_entries(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "v1.sqlite"
    with open_memory_store(database_path):
        pass
    _downgrade_empty_current_store_to_v1(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_entries (
                memory_id, memory_scope, repository_key, source_repository_id,
                commit_sha, verification_status, write_authority, title,
                content, tags_text, created_at_ms, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "a" * 64,
                "repository",
                "b" * 64,
                "c" * 64,
                "d" * 40,
                "verified",
                "validated_evidence",
                "Preserved memory",
                "Preserved content",
                "preserved",
                1,
                "{}",
            ),
        )

    with open_memory_store(database_path) as store:
        assert store.schema_version == MEMORY_STORE_SCHEMA_VERSION

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT title FROM memory_entries WHERE memory_id = ?",
            ("a" * 64,),
        ).fetchone() == ("Preserved memory",)
    assert "memory_episodes" in _schema_objects(database_path, "table")
    assert "memory_tombstones" in _schema_objects(database_path, "table")


def test_tampered_v1_schema_is_rejected_before_current_migration(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tampered-v1.sqlite"
    with open_memory_store(database_path):
        pass
    _downgrade_empty_current_store_to_v1(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER memory_entries_fts_update")

    with pytest.raises(MemoryStoreMigrationError, match="trigger"):
        with open_memory_store(database_path):
            pass

    assert _user_version(database_path) == 1
    assert "memory_episodes" not in _schema_objects(database_path, "table")


def test_v1_with_unindexed_episode_is_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "v1-episode.sqlite"
    with open_memory_store(database_path):
        pass
    _downgrade_empty_current_store_to_v1(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_entries (
                memory_id, memory_scope, repository_key, source_repository_id,
                commit_sha, verification_status, write_authority, title,
                content, tags_text, created_at_ms, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "a" * 64,
                "episodic",
                "b" * 64,
                "c" * 64,
                "d" * 40,
                "observed",
                "terminal_consolidation",
                "Legacy episode",
                "Cannot be safely indexed without validating JSON",
                "episodic",
                1,
                "{}",
            ),
        )

    with pytest.raises(MemoryStoreMigrationError, match="episodic"):
        with open_memory_store(database_path):
            pass

    assert _user_version(database_path) == 1
    assert "memory_episodes" not in _schema_objects(database_path, "table")
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT title FROM memory_entries WHERE memory_id = ?",
            ("a" * 64,),
        ).fetchone() == ("Legacy episode",)


def test_current_schema_with_missing_episode_unique_index_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing-episode-index.sqlite"
    with open_memory_store(database_path):
        pass
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX idx_memory_episodes_repository_thread_turn")

    with pytest.raises(MemoryStoreMigrationError, match="index"):
        with open_memory_store(database_path):
            pass


def test_current_schema_with_missing_tombstone_index_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing-tombstone-index.sqlite"
    with open_memory_store(database_path):
        pass
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX idx_memory_tombstones_repository_time")

    with pytest.raises(MemoryStoreMigrationError, match="index"):
        with open_memory_store(database_path):
            pass


def test_future_memory_schema_fails_closed_without_mutation(tmp_path: Path) -> None:
    database_path = tmp_path / "future.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(MemoryStoreMigrationError, match="newer"):
        with open_memory_store(database_path):
            pass

    assert _user_version(database_path) == 99
    assert _schema_objects(database_path, "table") == set()


def test_unversioned_nonempty_database_is_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE legacy_memory (value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_memory VALUES ('preserve-me')")

    with pytest.raises(MemoryStoreMigrationError, match="unversioned"):
        with open_memory_store(database_path):
            pass

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT value FROM legacy_memory").fetchone() == (
            "preserve-me",
        )
    assert _user_version(database_path) == 0


def test_current_version_with_missing_table_fails_schema_validation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE memory_schema (key TEXT PRIMARY KEY)")
        connection.execute(f"PRAGMA user_version = {MEMORY_STORE_SCHEMA_VERSION}")

    with pytest.raises(MemoryStoreMigrationError, match="missing"):
        with open_memory_store(database_path):
            pass


def test_current_schema_with_removed_foreign_key_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "foreign-key.sqlite"
    with open_memory_store(database_path):
        pass

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("ALTER TABLE memory_sources RENAME TO sources_with_fk")
        connection.execute(
            """
            CREATE TABLE memory_sources (
                memory_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                repository_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                path TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                source_json TEXT NOT NULL,
                PRIMARY KEY(memory_id, source_id)
            )
            """
        )
        connection.execute("DROP TABLE sources_with_fk")

    with pytest.raises(MemoryStoreMigrationError, match="foreign key"):
        with open_memory_store(database_path):
            pass


def test_current_schema_with_missing_fts_trigger_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "trigger.sqlite"
    with open_memory_store(database_path):
        pass
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER memory_entries_fts_update")

    with pytest.raises(MemoryStoreMigrationError, match="trigger"):
        with open_memory_store(database_path):
            pass


def test_current_schema_with_tampered_metadata_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite"
    with open_memory_store(database_path):
        pass
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_schema SET value = 'porter' WHERE key = 'fts_tokenizer'"
        )

    with pytest.raises(MemoryStoreMigrationError, match="metadata"):
        with open_memory_store(database_path):
            pass


def test_memory_store_requires_path_object(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        with open_memory_store(str(tmp_path / "memory.sqlite")):  # type: ignore[arg-type]
            pass


def test_same_name_noop_fts_trigger_fails_schema_validation(tmp_path: Path) -> None:
    database_path = tmp_path / "noop-trigger.sqlite"
    with open_memory_store(database_path):
        pass
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER memory_entries_fts_update")
        connection.execute(
            """
            CREATE TRIGGER memory_entries_fts_update
            AFTER UPDATE ON memory_entries BEGIN
                SELECT 1;
            END
            """
        )

    with pytest.raises(MemoryStoreMigrationError, match="trigger"):
        with open_memory_store(database_path):
            pass


def test_external_content_fts_drift_fails_integrity_validation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "fts-drift.sqlite"
    with open_memory_store(database_path):
        pass
    values = (
        "a" * 64,
        "repository",
        "b" * 64,
        "c" * 64,
        "d" * 40,
        "verified",
        "validated_evidence",
        "Authentication entry point",
        "The request handler delegates to the auth service.",
        "api auth",
        1,
        "{}",
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_entries (
                memory_id, memory_scope, repository_key, source_repository_id,
                commit_sha, verification_status, write_authority, title,
                content, tags_text, created_at_ms, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        row = connection.execute(
            "SELECT rowid, title, content, tags_text FROM memory_entries"
        ).fetchone()
        assert row is not None
        connection.execute(
            """
            INSERT INTO memory_fts(
                memory_fts, rowid, title, content, tags_text
            ) VALUES ('delete', ?, ?, ?, ?)
            """,
            row,
        )

    with pytest.raises(MemoryStoreMigrationError, match="integrity"):
        with open_memory_store(database_path):
            pass
