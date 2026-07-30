"""Local checkpoint persistence for RepoCompass CLI investigations.

This module owns only the SQLite connection lifecycle. Thread identity,
start/resume semantics, graph execution, and public CLI policy belong to the
Runner and CLI layers.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def open_sqlite_checkpointer(database_path: Path) -> Iterator[SqliteSaver]:
    """Open one local SQLite checkpointer and close it on context exit.

    Args:
        database_path: SQLite database file. Missing parent directories are
            created before the connection is opened.

    Yields:
        A synchronous LangGraph SQLite checkpointer bound to the resolved file.

    Side effects:
        Creates missing parent directories, opens the SQLite database, and
        closes its connection when the context exits.
    """

    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(resolved_path)) as checkpointer:
        yield checkpointer
