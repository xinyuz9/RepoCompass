"""SQLite idempotency ledger for public lifecycle mutations."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from time import time_ns

from pydantic import ValidationError

from repocompass.service.contracts import (
    InvestigationOperationKind,
    InvestigationOperationResult,
    REQUEST_ID_PATTERN,
)
from repocompass.storage.execution_lock import (
    execution_file_lock,
    ExecutionLockUnavailable,
)

SERVICE_OPERATION_STORE_SCHEMA_VERSION = 3
MAX_OPERATION_RESPONSE_BYTES = 33_554_432
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_PATTERN = re.compile(REQUEST_ID_PATTERN)


def _wall_clock_ms() -> int:
    return time_ns() // 1_000_000


class ServiceOperationStoreError(RuntimeError):
    """Base error for the durable public-operation ledger."""


class ServiceOperationMigrationError(ServiceOperationStoreError):
    """The database schema is newer than this application understands."""


class ServiceOperationConflictError(ServiceOperationStoreError):
    """A request ID was reused with incompatible immutable identity."""


class ServiceOperationCorruptionError(ServiceOperationStoreError):
    """Persisted operation state failed strict validation."""


class OperationReservationState(StrEnum):
    NEW = "new"
    PENDING = "pending"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class OperationReservation:
    request_id: str
    request_digest: str
    kind: InvestigationOperationKind
    thread_id: str
    reserved_at_ms: int
    state: OperationReservationState
    response: InvestigationOperationResult | None = None


class ServiceOperationStore:
    """Atomically reserve and complete caller-supplied idempotency keys."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        clock_ms: Callable[[], int] = _wall_clock_ms,
        execution_directory: Path | None = None,
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection")
        if not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._connection = connection
        self._lock = RLock()
        self._clock_ms = clock_ms
        self._execution_directory = execution_directory
        self._migrate()

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._connection.execute("PRAGMA user_version").fetchone()
        if row is None or type(row[0]) is not int:
            raise ServiceOperationCorruptionError("operation schema version is invalid")
        return row[0]

    def lookup(
        self,
        *,
        request_id: str,
        request_digest: str,
        kind: InvestigationOperationKind | str,
        thread_id: str,
    ) -> OperationReservation | None:
        """Resolve an existing exact operation without reserving a new key."""

        normalized_request_id = _validate_request_id(request_id)
        normalized_digest = _validate_digest(request_digest)
        normalized_kind = _normalize_kind(kind)
        normalized_thread_id = _validate_thread_id(thread_id)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT request_digest, operation_kind, thread_id,
                       reserved_at_ms, response_json
                FROM service_operations
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
        if row is None:
            return None
        (
            stored_digest,
            stored_kind,
            stored_thread_id,
            reserved_at_ms,
            response_json,
        ) = row
        if stored_digest != normalized_digest:
            raise ServiceOperationConflictError(
                "request_id was already used with a different payload"
            )
        if stored_kind != normalized_kind.value:
            raise ServiceOperationConflictError(
                "request_id was already used for a different operation kind"
            )
        if _validate_thread_id(stored_thread_id) != normalized_thread_id:
            raise ServiceOperationConflictError(
                "request_id was already used for a different thread"
            )
        if response_json is None:
            return OperationReservation(
                request_id=normalized_request_id,
                request_digest=normalized_digest,
                kind=normalized_kind,
                thread_id=normalized_thread_id,
                reserved_at_ms=_validate_timestamp(reserved_at_ms),
                state=OperationReservationState.PENDING,
            )
        response = _decode_response(response_json)
        if (
            response.request_id != normalized_request_id
            or response.thread_id != normalized_thread_id
        ):
            raise ServiceOperationCorruptionError(
                "persisted operation response identity does not reconcile"
            )
        return OperationReservation(
            request_id=normalized_request_id,
            request_digest=normalized_digest,
            kind=normalized_kind,
            thread_id=normalized_thread_id,
            reserved_at_ms=_validate_timestamp(reserved_at_ms),
            state=OperationReservationState.COMPLETED,
            response=response,
        )

    def get(self, request_id: str) -> OperationReservation | None:
        """Return one operation by opaque request identity without reserving it."""

        normalized_request_id = _validate_request_id(request_id)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT request_digest, operation_kind, thread_id,
                       reserved_at_ms, response_json
                FROM service_operations
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
        if row is None:
            return None
        request_digest, kind, thread_id, reserved_at_ms, response_json = row
        normalized_digest = _validate_digest(request_digest)
        normalized_kind = _normalize_kind(kind)
        normalized_thread_id = _validate_thread_id(thread_id)
        normalized_timestamp = _validate_timestamp(reserved_at_ms)
        if response_json is None:
            return OperationReservation(
                request_id=normalized_request_id,
                request_digest=normalized_digest,
                kind=normalized_kind,
                thread_id=normalized_thread_id,
                reserved_at_ms=normalized_timestamp,
                state=OperationReservationState.PENDING,
            )
        response = _decode_response(response_json)
        if (
            response.request_id != normalized_request_id
            or response.thread_id != normalized_thread_id
        ):
            raise ServiceOperationCorruptionError(
                "persisted operation response identity does not reconcile"
            )
        return OperationReservation(
            request_id=normalized_request_id,
            request_digest=normalized_digest,
            kind=normalized_kind,
            thread_id=normalized_thread_id,
            reserved_at_ms=normalized_timestamp,
            state=OperationReservationState.COMPLETED,
            response=response,
        )

    def reserve(
        self,
        *,
        request_id: str,
        request_digest: str,
        kind: InvestigationOperationKind | str,
        proposed_thread_id: str,
    ) -> OperationReservation:
        """Reserve one request or return its exact pending/completed identity."""

        normalized_request_id = _validate_request_id(request_id)
        normalized_digest = _validate_digest(request_digest)
        normalized_kind = _normalize_kind(kind)
        normalized_thread_id = _validate_thread_id(proposed_thread_id)

        with self._transaction():
            row = self._connection.execute(
                """
                SELECT request_digest, operation_kind, thread_id,
                       reserved_at_ms, response_json
                FROM service_operations
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
            if row is None:
                pending = self._connection.execute(
                    "SELECT 1 FROM service_operations WHERE thread_id = ? AND response_json IS NULL LIMIT 1",
                    (normalized_thread_id,),
                ).fetchone()
                if pending is not None:
                    raise ServiceOperationConflictError(
                        "thread has an unresolved lifecycle operation"
                    )
                reserved_at_ms = _validate_timestamp(self._clock_ms())
                self._connection.execute(
                    """
                    INSERT INTO service_operations (
                        request_id, request_digest, operation_kind,
                        thread_id, reserved_at_ms, response_json
                    ) VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        normalized_request_id,
                        normalized_digest,
                        normalized_kind.value,
                        normalized_thread_id,
                        reserved_at_ms,
                    ),
                )
                return OperationReservation(
                    request_id=normalized_request_id,
                    request_digest=normalized_digest,
                    kind=normalized_kind,
                    thread_id=normalized_thread_id,
                    reserved_at_ms=reserved_at_ms,
                    state=OperationReservationState.NEW,
                )

            (
                stored_digest,
                stored_kind,
                stored_thread_id,
                reserved_at_ms,
                response_json,
            ) = row
            if stored_digest != normalized_digest:
                raise ServiceOperationConflictError(
                    "request_id was already used with a different payload"
                )
            if stored_kind != normalized_kind.value:
                raise ServiceOperationConflictError(
                    "request_id was already used for a different operation kind"
                )
            stored_thread_id = _validate_thread_id(stored_thread_id)
            if response_json is None:
                return OperationReservation(
                    request_id=normalized_request_id,
                    request_digest=normalized_digest,
                    kind=normalized_kind,
                    thread_id=stored_thread_id,
                    reserved_at_ms=_validate_timestamp(reserved_at_ms),
                    state=OperationReservationState.PENDING,
                )
            response = _decode_response(response_json)
            if (
                response.request_id != normalized_request_id
                or response.thread_id != stored_thread_id
            ):
                raise ServiceOperationCorruptionError(
                    "persisted operation response identity does not reconcile"
                )
            return OperationReservation(
                request_id=normalized_request_id,
                request_digest=normalized_digest,
                kind=normalized_kind,
                thread_id=stored_thread_id,
                reserved_at_ms=_validate_timestamp(reserved_at_ms),
                state=OperationReservationState.COMPLETED,
                response=response,
            )

    @contextmanager
    def execution(self, request_id: str) -> Iterator[None]:
        """Enroll new execution once and hold its OS lock through completion."""
        request_id = _validate_request_id(request_id)
        if self._execution_directory is None:
            # In-memory/custom adapters remain usable but are not crash-recoverable.
            yield
            return
        with execution_file_lock(self._execution_directory, request_id, create=True):
            with self._transaction():
                reservation = self.get(request_id)
                if (
                    reservation is None
                    or reservation.state is not OperationReservationState.PENDING
                ):
                    raise ServiceOperationConflictError(
                        "execution needs a pending reservation"
                    )
                existing = self._connection.execute(
                    "SELECT 1 FROM service_executions WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if existing is not None:
                    raise ServiceOperationConflictError(
                        "execution already enrolled; reconcile explicitly"
                    )
                self._connection.execute(
                    "INSERT INTO service_executions(request_id, run_id) VALUES (?, NULL)",
                    (request_id,),
                )
            yield

    def bind_run(self, request_id: str, run_id: str) -> None:
        if self._execution_directory is None:
            return
        request_id = _validate_request_id(request_id)
        run_id = _validate_thread_id(run_id)
        with self._transaction():
            row = self._connection.execute(
                "SELECT run_id FROM service_executions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None or (row[0] is not None and row[0] != run_id):
                raise ServiceOperationConflictError("operation run binding conflicts")
            self._connection.execute(
                "UPDATE service_executions SET run_id = ? WHERE request_id = ?",
                (run_id, request_id),
            )

    def execution_run(self, request_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id FROM service_executions WHERE request_id = ?",
                (_validate_request_id(request_id),),
            ).fetchone()
        return None if row is None or row[0] is None else _validate_thread_id(row[0])

    @contextmanager
    def recovery(self, request_id: str) -> Iterator[str]:
        request_id = _validate_request_id(request_id)
        if self._execution_directory is None:
            raise ExecutionLockUnavailable("execution was not durably enrolled")
        with execution_file_lock(self._execution_directory, request_id, create=False):
            run_id = self.execution_run(request_id)
            if run_id is None:
                raise ExecutionLockUnavailable("operation has no exact run binding")
            yield run_id

    def recovery_availability(self, request_id: str) -> str:
        if self.execution_run(request_id) is None:
            return "unavailable"
        try:
            with self.recovery(request_id):
                return "reconcilable"
        except ExecutionLockUnavailable:
            return "active_or_unavailable"

    def complete(
        self,
        *,
        request_id: str,
        request_digest: str,
        response: InvestigationOperationResult,
    ) -> bool:
        """Persist one terminal service response exactly once."""

        normalized_request_id = _validate_request_id(request_id)
        normalized_digest = _validate_digest(request_digest)
        if not isinstance(response, InvestigationOperationResult):
            raise TypeError("response must be an InvestigationOperationResult")
        if response.request_id != normalized_request_id:
            raise ValueError("response request_id does not match reservation")
        canonical_response = response.model_copy(update={"idempotent_replay": False})
        encoded = canonical_response.model_dump_json(
            exclude_computed_fields=True,
            warnings=False,
        )
        if len(encoded.encode("utf-8")) > MAX_OPERATION_RESPONSE_BYTES:
            raise ValueError("operation response exceeds the storage byte limit")

        with self._transaction():
            row = self._connection.execute(
                """
                SELECT request_digest, thread_id, response_json
                FROM service_operations
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
            if row is None:
                raise ServiceOperationConflictError("operation reservation is missing")
            stored_digest, stored_thread_id, existing_response = row
            if stored_digest != normalized_digest:
                raise ServiceOperationConflictError(
                    "request_id was already used with a different payload"
                )
            if canonical_response.thread_id != stored_thread_id:
                raise ServiceOperationConflictError(
                    "operation response thread does not match reservation"
                )
            if existing_response is not None:
                existing = _decode_response(existing_response)
                if existing != canonical_response:
                    raise ServiceOperationConflictError(
                        "operation already has a different completed response"
                    )
                return False
            updated = self._connection.execute(
                """
                UPDATE service_operations
                SET response_json = ?
                WHERE request_id = ? AND response_json IS NULL
                """,
                (encoded, normalized_request_id),
            ).rowcount
            if updated != 1:
                raise ServiceOperationConflictError(
                    "operation completion lost an atomic update race"
                )
            return True

    def _migrate(self) -> None:
        # DDL, schema validation and version advancement commit together.
        with self._transaction():
            row = self._connection.execute("PRAGMA user_version").fetchone()
            version = 0 if row is None else row[0]
            if type(version) is not int or version < 0:
                raise ServiceOperationMigrationError(
                    "operation database schema version is invalid"
                )
            if version > SERVICE_OPERATION_STORE_SCHEMA_VERSION:
                raise ServiceOperationMigrationError(
                    "operation database schema is newer than this application"
                )
            if version == 0:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS service_operations (
                        request_id TEXT PRIMARY KEY NOT NULL,
                        request_digest TEXT NOT NULL,
                        operation_kind TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        reserved_at_ms INTEGER NOT NULL CHECK(reserved_at_ms >= 0),
                        response_json TEXT
                    ) STRICT
                    """
                )
                self._connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "service_operations_thread_request "
                    "ON service_operations(thread_id, request_id)"
                )
                version = 2
            if version == 1:
                self._connection.execute(
                    "ALTER TABLE service_operations ADD COLUMN reserved_at_ms "
                    "INTEGER NOT NULL DEFAULT 0 CHECK(reserved_at_ms >= 0)"
                )
                version = 2
            if version == 2:
                self._connection.execute(
                    "CREATE TABLE service_executions (request_id TEXT PRIMARY KEY NOT NULL REFERENCES service_operations(request_id), run_id TEXT) STRICT"
                )
                version = 3
            if version == SERVICE_OPERATION_STORE_SCHEMA_VERSION:
                self._validate_schema()
                self._connection.execute(
                    f"PRAGMA user_version = {SERVICE_OPERATION_STORE_SCHEMA_VERSION}"
                )

    def _validate_schema(self) -> None:
        expected = {
            "service_operations": {
                "request_id": ("TEXT", 1, 1),
                "request_digest": ("TEXT", 1, 0),
                "operation_kind": ("TEXT", 1, 0),
                "thread_id": ("TEXT", 1, 0),
                "reserved_at_ms": ("INTEGER", 1, 0),
                "response_json": ("TEXT", 0, 0),
            },
            "service_executions": {
                "request_id": ("TEXT", 1, 1),
                "run_id": ("TEXT", 0, 0),
            },
        }
        tables = {row[1]: row for row in self._connection.execute("PRAGMA table_list")}
        for name, columns in expected.items():
            actual = {
                row[1]: (row[2], row[3], row[5])
                for row in self._connection.execute(f"PRAGMA table_info({name})")
            }
            table = tables.get(name)
            if (
                actual != columns
                or table is None
                or table[2] != "table"
                or table[5] != 1
            ):
                raise ServiceOperationMigrationError(
                    "operation schema shape is invalid"
                )
        foreign_keys = self._connection.execute(
            "PRAGMA foreign_key_list(service_executions)"
        ).fetchall()
        if len(foreign_keys) != 1 or foreign_keys[0][2:5] != (
            "service_operations",
            "request_id",
            "request_id",
        ):
            raise ServiceOperationMigrationError(
                "execution ownership schema is invalid"
            )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()


def _validate_request_id(value: object) -> str:
    if not isinstance(value, str) or _REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("request_id is invalid")
    return value


def _validate_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("request_digest must be a SHA-256 digest")
    return value


def _validate_thread_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("thread_id must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > 256:
        raise ValueError("thread_id is invalid")
    return normalized


def _validate_timestamp(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
        raise ValueError("reserved_at_ms is invalid")
    return value


def _normalize_kind(
    value: InvestigationOperationKind | str,
) -> InvestigationOperationKind:
    try:
        return InvestigationOperationKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError("operation kind is invalid") from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_response(value: object) -> InvestigationOperationResult:
    if not isinstance(value, str):
        raise ServiceOperationCorruptionError(
            "persisted operation response is not text"
        )
    if len(value.encode("utf-8")) > MAX_OPERATION_RESPONSE_BYTES:
        raise ServiceOperationCorruptionError(
            "persisted operation response exceeds the byte limit"
        )
    try:
        payload = json.loads(value, object_pairs_hook=_unique_json_object)
        return InvestigationOperationResult.model_validate(payload, strict=False)
    except (json.JSONDecodeError, UnicodeError, ValidationError) as error:
        raise ServiceOperationCorruptionError(
            "persisted operation response is invalid"
        ) from error


@contextmanager
def open_service_operation_store(
    database_path: Path,
) -> Iterator[ServiceOperationStore]:
    """Open the local idempotency database with WAL-safe connection settings."""

    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a Path")
    resolved = database_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(resolved), check_same_thread=False)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        yield ServiceOperationStore(
            connection, execution_directory=resolved.with_name(resolved.name + ".locks")
        )
