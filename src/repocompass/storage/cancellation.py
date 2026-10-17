"""Independent durable run control; requesting cancellation never takes the worker lock."""

from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import re
from threading import RLock
from time import time_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repocompass.tools.contracts import (
    CommitSha,
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    Sha256Digest,
)


class CancellationError(RuntimeError):
    """Safe cancellation control failure."""


class CancellationConflict(CancellationError):
    """Stale run identity or reused request key."""


class RunCancellationRequested(RuntimeError):
    def __init__(self, run_id: str, request_id: str):
        super().__init__("run cancellation requested")
        self.run_id = run_id
        self.request_id = request_id


class CancellationIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None


class CancellationRequest(CancellationIdentity):
    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)


class CancellationRecord(CancellationIdentity):
    schema_version: Literal[1] = 1
    state: Literal["running", "cancel_requested", "cancelled", "finished"] = "running"
    request_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    requested_at_ms: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    finished_at_ms: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def coherent_state(self):
        if (self.request_id is None) != (self.requested_at_ms is None):
            raise ValueError("cancellation request identity and time disagree")
        if self.state in {"cancel_requested", "cancelled"} and self.request_id is None:
            raise ValueError("cancellation state requires a request")
        if self.state == "running" and self.request_id is not None:
            raise ValueError("running control cannot contain a cancellation request")
        if (self.state in {"cancelled", "finished"}) != (
            self.finished_at_ms is not None
        ):
            raise ValueError("cancellation terminal state and time disagree")
        if (
            self.requested_at_ms is not None
            and self.finished_at_ms is not None
            and self.finished_at_ms < self.requested_at_ms
        ):
            raise ValueError("cancellation finish cannot precede its request")
        return self


class CancellationStore:
    """One enrolled worker attempt per row; atomic request/finish compare-and-set."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection
        self._lock = RLock()
        with self._transaction():
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise CancellationError("unsupported cancellation store version")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if version == 0:
                if tables:
                    raise CancellationError(
                        "cancellation requires a dedicated empty database"
                    )
                connection.execute("""CREATE TABLE run_cancellations (
                    run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                    repository_id TEXT NOT NULL, commit_sha TEXT,
                    state TEXT NOT NULL CHECK(state IN ('running','cancel_requested','cancelled','finished')),
                    request_id TEXT UNIQUE, requested_at_ms INTEGER, finished_at_ms INTEGER
                )""")
                connection.execute("PRAGMA user_version=1")
            elif tables != {"run_cancellations"}:
                raise CancellationError("incompatible cancellation store tables")
            columns = tuple(
                (row[1], row[2], row[3], row[5])
                for row in connection.execute("PRAGMA table_info(run_cancellations)")
            )
            if columns != (
                ("run_id", "TEXT", 0, 1),
                ("thread_id", "TEXT", 1, 0),
                ("repository_id", "TEXT", 1, 0),
                ("commit_sha", "TEXT", 0, 0),
                ("state", "TEXT", 1, 0),
                ("request_id", "TEXT", 0, 0),
                ("requested_at_ms", "INTEGER", 0, 0),
                ("finished_at_ms", "INTEGER", 0, 0),
            ):
                raise CancellationError("incompatible cancellation store columns")
            unique_keys = {
                tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                        (index[0],),
                    )
                )
                for index in connection.execute(
                    'SELECT name FROM pragma_index_list(?) WHERE "unique"=1 AND partial=0',
                    ("run_cancellations",),
                )
            }
            if unique_keys != {("run_id",), ("request_id",)}:
                raise CancellationError("incompatible cancellation store uniqueness")

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def get(self, run_id: str) -> CancellationRecord | None:
        # Validation before SQL, without requiring unrelated identity fields.
        if (
            not isinstance(run_id, str)
            or re.fullmatch(IDENTIFIER_PATTERN, run_id) is None
        ):
            raise ValueError("invalid cancellation run ID")
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id, thread_id, repository_id, commit_sha, state, request_id, requested_at_ms, finished_at_ms FROM run_cancellations WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return CancellationRecord(
                **dict(
                    zip(
                        (
                            "run_id",
                            "thread_id",
                            "repository_id",
                            "commit_sha",
                            "state",
                            "request_id",
                            "requested_at_ms",
                            "finished_at_ms",
                        ),
                        row,
                        strict=True,
                    )
                )
            )
        except ValueError:
            raise CancellationError("corrupt cancellation record") from None

    def register(self, identity: CancellationIdentity) -> None:
        identity = CancellationIdentity.model_validate(identity.model_dump())
        with self._transaction():
            try:
                self._connection.execute(
                    "INSERT INTO run_cancellations(run_id,thread_id,repository_id,commit_sha,state) VALUES (?,?,?,?,'running')",
                    (
                        identity.run_id,
                        identity.thread_id,
                        identity.repository_id,
                        identity.commit_sha,
                    ),
                )
            except sqlite3.IntegrityError:
                raise CancellationConflict("run is already enrolled") from None

    def request(self, request: CancellationRequest) -> CancellationRecord:
        request = CancellationRequest.model_validate(request.model_dump())
        with self._transaction():
            record = self.get(request.run_id)
            if record is None:
                raise CancellationConflict("run is not enrolled for cancellation")
            if any(
                getattr(record, key) != getattr(request, key)
                for key in CancellationIdentity.model_fields
            ):
                raise CancellationConflict(
                    "cancellation targets a different run identity"
                )
            if record.request_id == request.request_id:
                return record
            if record.state != "running":
                raise CancellationConflict(
                    "run already finished or cancellation already requested"
                )
            try:
                self._connection.execute(
                    "UPDATE run_cancellations SET state='cancel_requested',request_id=?,requested_at_ms=? WHERE run_id=? AND state='running'",
                    (request.request_id, time_ns() // 1_000_000, request.run_id),
                )
            except sqlite3.IntegrityError:
                raise CancellationConflict(
                    "cancellation request ID was reused"
                ) from None
            result = self.get(request.run_id)
            assert result is not None
            return result

    def check(self, run_id: str) -> None:
        record = self.get(run_id)
        if record is None or record.state not in {"running", "cancel_requested"}:
            raise CancellationError("worker has no active cancellation enrollment")
        if record.state == "cancel_requested":
            assert record.request_id is not None
            raise RunCancellationRequested(run_id, record.request_id)

    def finish(self, run_id: str, *, cancelled: bool) -> None:
        if type(cancelled) is not bool:
            raise TypeError("cancelled must be a bool")
        with self._transaction():
            record = self.get(run_id)
            if record is None:
                raise CancellationError("worker cancellation enrollment is missing")
            target = "cancelled" if cancelled else "finished"
            if record.state in {"cancelled", "finished"}:
                if record.state != target:
                    raise CancellationConflict(
                        "cancellation terminal outcome cannot change"
                    )
                return
            if cancelled and record.state != "cancel_requested":
                raise CancellationConflict(
                    "cannot acknowledge an unrequested cancellation"
                )
            self._connection.execute(
                "UPDATE run_cancellations SET state=?,finished_at_ms=? WHERE run_id=?",
                (
                    target,
                    max(time_ns() // 1_000_000, record.requested_at_ms or 0),
                    run_id,
                ),
            )


@contextmanager
def open_cancellation_store(path: Path):
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(str(resolved), check_same_thread=False, timeout=5)
    ) as connection:
        yield CancellationStore(connection)
