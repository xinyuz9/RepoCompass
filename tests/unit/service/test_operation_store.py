"""Durable idempotency ledger tests for public investigation operations."""

import sqlite3
from pathlib import Path

import pytest

from repocompass.service.contracts import (
    InvestigationOperationKind,
    InvestigationOperationResult,
)
from repocompass.service.operation_store import (
    OperationReservationState,
    ServiceOperationConflictError,
    ServiceOperationMigrationError,
    ServiceOperationStore,
    open_service_operation_store,
)
from repocompass.agent.thread_view import ThreadExecutionState


def _result(request_id: str = "request-1") -> InvestigationOperationResult:
    return InvestigationOperationResult(
        request_id=request_id,
        thread_id="thread-1",
        execution_state=ThreadExecutionState.RESUMABLE,
        run_id=None,
        termination_status=None,
        step_stop_reason=None,
        next_nodes=("agent",),
        final_result=None,
        pending_review=None,
        interrupt_id=None,
        usage_cost_report=None,
        idempotent_replay=False,
        recovered_from_checkpoint=True,
    )


def test_operation_store_reserves_completes_and_replays() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    store = ServiceOperationStore(connection, clock_ms=lambda: 1_234)

    reserved = store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="thread-1",
    )
    pending = store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="ignored-thread",
    )

    assert reserved.state is OperationReservationState.NEW
    assert reserved.reserved_at_ms == 1_234
    assert pending.state is OperationReservationState.PENDING
    assert pending.thread_id == "thread-1"
    assert pending.reserved_at_ms == 1_234
    assert pending.response is None

    assert (
        store.complete(
            request_id="request-1",
            request_digest="a" * 64,
            response=_result(),
        )
        is True
    )
    assert (
        store.complete(
            request_id="request-1",
            request_digest="a" * 64,
            response=_result(),
        )
        is False
    )

    replay = store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="ignored-again",
    )
    assert replay.state is OperationReservationState.COMPLETED
    assert replay.response == _result()
    assert replay.reserved_at_ms == 1_234
    assert store.schema_version == 3


def test_operation_store_get_is_read_only_and_hides_missing_requests() -> None:
    store = ServiceOperationStore(sqlite3.connect(":memory:"), clock_ms=lambda: 2_000)

    assert store.get("request-missing") is None
    store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="thread-1",
    )
    pending = store.get("request-1")
    assert pending is not None
    assert pending.state is OperationReservationState.PENDING
    assert pending.reserved_at_ms == 2_000
    assert pending.response is None

    store.complete(
        request_id="request-1",
        request_digest="a" * 64,
        response=_result(),
    )
    completed = store.get("request-1")
    assert completed is not None
    assert completed.state is OperationReservationState.COMPLETED
    assert completed.response == _result()


def test_operation_store_rejects_request_id_reuse_with_changed_payload() -> None:
    store = ServiceOperationStore(sqlite3.connect(":memory:"))
    store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="thread-1",
    )

    with pytest.raises(ServiceOperationConflictError, match="different payload"):
        store.reserve(
            request_id="request-1",
            request_digest="b" * 64,
            kind=InvestigationOperationKind.START,
            proposed_thread_id="thread-2",
        )

    with pytest.raises(ServiceOperationConflictError, match="operation kind"):
        store.reserve(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.RESUME,
            proposed_thread_id="thread-1",
        )


def test_operation_lookup_is_non_reserving_and_requires_exact_identity() -> None:
    store = ServiceOperationStore(sqlite3.connect(":memory:"))

    assert (
        store.lookup(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.RESUME,
            thread_id="thread-1",
        )
        is None
    )
    reservation = store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.RESUME,
        proposed_thread_id="thread-1",
    )
    pending = store.lookup(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.RESUME,
        thread_id="thread-1",
    )

    assert reservation.state is OperationReservationState.NEW
    assert pending is not None
    assert pending.state is OperationReservationState.PENDING
    with pytest.raises(ServiceOperationConflictError, match="different payload"):
        store.lookup(
            request_id="request-1",
            request_digest="b" * 64,
            kind=InvestigationOperationKind.RESUME,
            thread_id="thread-1",
        )
    with pytest.raises(ServiceOperationConflictError, match="different thread"):
        store.lookup(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.RESUME,
            thread_id="thread-2",
        )

    store.complete(
        request_id="request-1",
        request_digest="a" * 64,
        response=_result(),
    )
    completed = store.lookup(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.RESUME,
        thread_id="thread-1",
    )
    assert completed is not None
    assert completed.state is OperationReservationState.COMPLETED
    assert completed.response == _result()


def test_operation_store_detects_corrupt_persisted_response() -> None:
    connection = sqlite3.connect(":memory:")
    store = ServiceOperationStore(connection)
    store.reserve(
        request_id="request-1",
        request_digest="a" * 64,
        kind=InvestigationOperationKind.START,
        proposed_thread_id="thread-1",
    )
    connection.execute(
        "UPDATE service_operations SET response_json = ? WHERE request_id = ?",
        ('{"request_id":"first","request_id":"second"}', "request-1"),
    )
    connection.commit()

    with pytest.raises(ValueError, match="duplicate JSON key"):
        store.reserve(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.START,
            proposed_thread_id="thread-1",
        )


def test_operation_store_replays_after_sqlite_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "service" / "operations.sqlite"
    with open_service_operation_store(database_path) as store:
        store.reserve(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.START,
            proposed_thread_id="thread-1",
        )
        store.complete(
            request_id="request-1",
            request_digest="a" * 64,
            response=_result(),
        )

    with open_service_operation_store(database_path) as reopened:
        replay = reopened.reserve(
            request_id="request-1",
            request_digest="a" * 64,
            kind=InvestigationOperationKind.START,
            proposed_thread_id="ignored-thread",
        )

    assert replay.state is OperationReservationState.COMPLETED
    assert replay.thread_id == "thread-1"
    assert replay.response == _result()


def test_operation_store_migrates_v1_reservations_with_safe_epoch() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE service_operations (
            request_id TEXT PRIMARY KEY NOT NULL,
            request_digest TEXT NOT NULL,
            operation_kind TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            response_json TEXT
        ) STRICT
        """
    )
    connection.execute(
        "INSERT INTO service_operations VALUES (?, ?, ?, ?, NULL)",
        ("request-legacy", "a" * 64, "start", "thread-legacy"),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()

    store = ServiceOperationStore(connection, clock_ms=lambda: 9_999)
    migrated = store.get("request-legacy")

    assert store.schema_version == 3
    assert migrated is not None
    assert migrated.reserved_at_ms == 0
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(service_operations)")
    }
    assert "reserved_at_ms" in columns


def test_operation_migration_rolls_back_on_invalid_legacy_schema() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE service_operations(request_id TEXT)")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    with pytest.raises(ServiceOperationMigrationError, match="schema shape"):
        ServiceOperationStore(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'service_executions'"
        ).fetchone()
        is None
    )


@pytest.mark.parametrize("version", [3, 4])
def test_operation_store_rejects_unrecognized_or_incomplete_schema(version) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(f"PRAGMA user_version = {version}")
    with pytest.raises(ServiceOperationMigrationError):
        ServiceOperationStore(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == version
