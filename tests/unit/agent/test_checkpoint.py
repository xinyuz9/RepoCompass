"""Unit tests for the local SQLite checkpoint lifecycle."""

from pathlib import Path

from repocompass.agent.checkpoint import open_sqlite_checkpointer


def test_open_sqlite_checkpointer_creates_parent_and_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        checkpointer.setup()
        assert database_path.exists()

    assert database_path.is_file()
