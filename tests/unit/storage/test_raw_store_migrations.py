"""Versioned Raw Run Store migration and schema validation tests."""

import sqlite3
from pathlib import Path

import pytest

from repocompass.storage.raw_store import (
    RAW_STORE_SCHEMA_VERSION,
    RawStoreMigrationError,
    open_raw_run_store,
)

EXPECTED_TABLES = {
    "attempts",
    "raw_artifacts",
    "attempt_artifacts",
    "context_builds",
    "model_invocations",
}


def _tables(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }


def _user_version(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _foreign_key_edges(
    database_path: Path,
    table: str,
) -> list[tuple[str, str, str, str]]:
    with sqlite3.connect(database_path) as connection:
        return sorted(
            (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )


def _column_is_not_null(database_path: Path, table: str, column: str) -> bool:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column and int(row[3]) == 1 for row in rows)


def test_fresh_store_creates_parent_and_exact_v1_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "raw.sqlite"

    with open_raw_run_store(database_path) as store:
        assert store.schema_version == RAW_STORE_SCHEMA_VERSION

    assert database_path.is_file()
    assert _user_version(database_path) == RAW_STORE_SCHEMA_VERSION
    assert _tables(database_path) == EXPECTED_TABLES
    assert _column_is_not_null(
        database_path,
        "model_invocations",
        "context_pack_id",
    )
    assert _foreign_key_edges(database_path, "attempt_artifacts") == [
        ("attempts", "run_id", "run_id", "CASCADE"),
        ("raw_artifacts", "artifact_id", "artifact_id", "CASCADE"),
    ]
    assert _foreign_key_edges(database_path, "context_builds") == [
        ("attempts", "run_id", "run_id", "CASCADE"),
    ]
    assert _foreign_key_edges(database_path, "model_invocations") == sorted(
        [
            ("attempts", "run_id", "run_id", "CASCADE"),
            ("attempts", "thread_id", "thread_id", "CASCADE"),
            ("attempts", "turn_number", "turn_number", "CASCADE"),
            ("attempt_artifacts", "run_id", "run_id", "CASCADE"),
            ("attempt_artifacts", "input_artifact_id", "artifact_id", "CASCADE"),
            ("attempt_artifacts", "run_id", "run_id", "CASCADE"),
            ("attempt_artifacts", "output_artifact_id", "artifact_id", "CASCADE"),
            ("context_builds", "run_id", "run_id", "CASCADE"),
            ("context_builds", "context_pack_id", "pack_id", "CASCADE"),
        ]
    )


def test_reopening_current_schema_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "raw.sqlite"

    with open_raw_run_store(database_path):
        pass
    first_schema = _tables(database_path)

    with open_raw_run_store(database_path) as store:
        assert store.schema_version == RAW_STORE_SCHEMA_VERSION

    assert _tables(database_path) == first_schema
    assert _user_version(database_path) == RAW_STORE_SCHEMA_VERSION


def test_future_schema_version_fails_closed_without_mutation(tmp_path: Path) -> None:
    database_path = tmp_path / "future.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RawStoreMigrationError, match="newer"):
        with open_raw_run_store(database_path):
            pass

    assert _user_version(database_path) == 99
    assert _tables(database_path) == set()


def test_unversioned_nonempty_database_fails_closed_without_dropping_tables(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE legacy_data (value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_data VALUES ('preserve-me')")

    with pytest.raises(RawStoreMigrationError, match="unversioned"):
        with open_raw_run_store(database_path):
            pass

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT value FROM legacy_data").fetchone() == (
            "preserve-me",
        )
    assert _user_version(database_path) == 0
    assert _tables(database_path) == {"legacy_data"}


def test_current_version_with_missing_table_fails_schema_validation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "broken.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE attempts (run_id TEXT PRIMARY KEY)")
        connection.execute(f"PRAGMA user_version = {RAW_STORE_SCHEMA_VERSION}")

    with pytest.raises(RawStoreMigrationError, match="schema"):
        with open_raw_run_store(database_path):
            pass

    assert _tables(database_path) == {"attempts"}


def test_current_schema_with_removed_foreign_key_fails_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "weakened.sqlite"
    with open_raw_run_store(database_path):
        pass

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "ALTER TABLE context_builds RENAME TO context_builds_with_fk"
        )
        connection.execute(
            """
            CREATE TABLE context_builds (
                run_id TEXT,
                build_id TEXT,
                pack_id TEXT,
                sequence INTEGER,
                budget_tokens INTEGER,
                selected_tokens INTEGER,
                candidate_count INTEGER,
                selected_count INTEGER,
                excluded_count INTEGER,
                pack_json TEXT
            )
            """
        )
        connection.execute("DROP TABLE context_builds_with_fk")

    with pytest.raises(RawStoreMigrationError, match="foreign key"):
        with open_raw_run_store(database_path):
            pass
