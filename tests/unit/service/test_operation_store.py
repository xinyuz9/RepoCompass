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
    store = ServiceOperationStore(connection)

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
    assert pending.state is OperationReservationState.PENDING
    assert pending.thread_id == "thread-1"
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
    assert store.schema_version == 1


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
