from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from repocompass.storage.cancellation import (
    CancellationConflict,
    CancellationError,
    CancellationIdentity,
    CancellationRequest,
    RunCancellationRequested,
    open_cancellation_store,
)
from repocompass.storage import cancellation


def identity(run="run-1", thread="thread-1"):
    return CancellationIdentity(run_id=run, thread_id=thread, repository_id="a" * 64)


def request(run="run-1", key="request-1", **kwargs):
    return CancellationRequest(
        **{**identity(run).model_dump(), **kwargs}, request_id=key
    )


def test_request_is_durable_idempotent_and_not_acknowledgement(tmp_path):
    database = tmp_path / "cancel.sqlite"
    with open_cancellation_store(database) as store:
        store.register(identity())
        store.check("run-1")
        first = store.request(request())
        assert first.state == "cancel_requested"
        assert first.finished_at_ms is None
        assert store.request(request()) == first
    with open_cancellation_store(database) as store:
        with pytest.raises(RunCancellationRequested) as caught:
            store.check("run-1")
        assert caught.value.request_id == "request-1"
        store.finish("run-1", cancelled=True)
        assert store.request(request()).state == "cancelled"
        with pytest.raises(CancellationConflict):
            store.finish("run-1", cancelled=False)


def test_wrong_identity_reused_keys_and_late_requests_fail_closed(tmp_path):
    with open_cancellation_store(tmp_path / "cancel.sqlite") as store:
        store.register(identity())
        store.register(identity("run-2"))
        for changed in [
            {"thread_id": "wrong"},
            {"repository_id": "b" * 64},
            {"commit_sha": "a" * 40},
        ]:
            with pytest.raises(CancellationConflict):
                store.request(request(**changed))
        store.request(request())
        with pytest.raises(CancellationConflict):
            store.request(request("run-2"))
        with pytest.raises(CancellationConflict):
            store.request(request(key="another"))
        with pytest.raises(CancellationConflict):
            store.finish("run-2", cancelled=True)
        store.finish("run-2", cancelled=False)
        with pytest.raises(CancellationConflict):
            store.request(request("run-2", key="late"))
        assert store.get("run-1").state == "cancel_requested"


def test_concurrent_connections_cannot_overwrite_first_request(tmp_path):
    database = tmp_path / "cancel.sqlite"
    with open_cancellation_store(database) as store:
        store.register(identity())

    def submit(key):
        with open_cancellation_store(database) as store:
            try:
                return store.request(request(key=key)).request_id
            except CancellationConflict:
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, ["one", "two"]))
    assert results.count("conflict") == 1
    with open_cancellation_store(database) as store:
        assert store.get("run-1").request_id in {"one", "two"}


@pytest.mark.parametrize("version", [0, 1, 2])
def test_foreign_or_future_database_is_rejected_without_mutation(tmp_path, version):
    database = tmp_path / "other.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE existing_data(value TEXT)")
        connection.execute("INSERT INTO existing_data VALUES ('keep')")
        connection.execute(f"PRAGMA user_version={version}")
    before = database.read_bytes()
    with pytest.raises(CancellationError), open_cancellation_store(database):
        pass
    assert database.read_bytes() == before


def test_incompatible_columns_and_corrupt_state_fail_closed(tmp_path):
    database = tmp_path / "cancel.sqlite"
    with open_cancellation_store(database) as store:
        store.register(identity())
        with pytest.raises(TypeError):
            store.finish("run-1", cancelled="yes")
        with pytest.raises(ValueError):
            store.get("../invalid")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE run_cancellations SET state='cancelled'")
    with open_cancellation_store(database) as store:
        with pytest.raises(CancellationError, match="corrupt"):
            store.get("run-1")
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE run_cancellations ADD COLUMN unexpected TEXT")
    before = database.read_bytes()
    with (
        pytest.raises(CancellationError, match="columns"),
        open_cancellation_store(database),
    ):
        pass
    assert database.read_bytes() == before


def test_wall_clock_rollback_does_not_lose_cancellation_ack(tmp_path, monkeypatch):
    ticks = iter([2_000_000_000, 1_000_000_000])
    monkeypatch.setattr(cancellation, "time_ns", lambda: next(ticks))
    with open_cancellation_store(tmp_path / "cancel.sqlite") as store:
        store.register(identity())
        store.request(request())
        store.finish("run-1", cancelled=True)
        record = store.get("run-1")
        assert record.state == "cancelled"
        assert record.requested_at_ms == record.finished_at_ms == 2000
