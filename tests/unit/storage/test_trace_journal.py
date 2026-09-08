"""Crash-visible committed prefixes and atomic immutable trace finalization."""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import repocompass.storage.raw_store as module
from repocompass.agent.trace import RunTrace, TraceEvent, TraceEventKind
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    RawStoreConflictError,
    RawStoreCorruptionError,
    RawStoreIntegrityError,
    RawStoreMigrationError,
    open_raw_run_store,
)


def _attempt():
    return RawRunAttempt(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        execution_mode=AttemptExecutionMode.START,
        repository_id="a" * 64,
        commit_sha="b" * 40,
        objective_digest="c" * 64,
        status=AttemptStatus.RUNNING,
        started_at_ms=1000,
    )


def _start(store):
    store.record_attempt(_attempt())
    trace = RunTrace("run-1")
    store.append_trace_event(trace.record(TraceEventKind.RUN_STARTED))
    return trace


def test_committed_prefix_visible_to_other_connection_and_survives_reopen(tmp_path):
    path = tmp_path / "raw.sqlite"
    with open_raw_run_store(path) as writer, open_raw_run_store(path) as reader:
        trace = _start(writer)
        assert reader.list_trace_events("run-1") == trace.events
        assert reader.get_attempt("run-1").status is AttemptStatus.RUNNING
        writer.append_trace_event(
            trace.record(TraceEventKind.NODE_COMPLETED, node="agent")
        )
        assert reader.list_trace_events("run-1") == trace.events
    with open_raw_run_store(path) as reopened:
        assert reopened.list_trace_events("run-1") == trace.events
        # Missing a terminal event does not fabricate cancellation or success.
        assert reopened.get_attempt("run-1").status is AttemptStatus.RUNNING
        assert reopened.list_trace_events("absent") is None


def test_abrupt_process_exit_keeps_committed_prefix(tmp_path):
    path = tmp_path / "abrupt.sqlite"
    script = """
import os,sys
from pathlib import Path
from repocompass.agent.trace import RunTrace,TraceEventKind
from repocompass.storage.raw_store import open_raw_run_store, RawRunAttempt
with open_raw_run_store(Path(sys.argv[1])) as store:
    store.record_attempt(RawRunAttempt.model_validate_json(
        '{"run_id":"run-1","thread_id":"thread-1","turn_number":1,'
        '"execution_mode":"start","repository_id":"' + 'a'*64 + '",'
        '"objective_digest":"' + 'c'*64 + '","status":"running","started_at_ms":1000}'
    ))
    trace=RunTrace('run-1')
    store.append_trace_event(trace.record(TraceEventKind.RUN_STARTED))
    store.append_trace_event(trace.record(TraceEventKind.NODE_COMPLETED,node='agent'))
    os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], timeout=15, capture_output=True
    )
    assert result.returncode == 17, result.stderr.decode()
    with open_raw_run_store(path) as store:
        assert [e.sequence for e in store.list_trace_events("run-1")] == [0, 1]
        assert store.get_attempt("run-1").status is AttemptStatus.RUNNING


@pytest.mark.parametrize(
    "status,kind",
    [
        (AttemptStatus.COMPLETED, TraceEventKind.RUN_FINISHED),
        (AttemptStatus.INTERRUPTED, TraceEventKind.RUN_INTERRUPTED),
        (AttemptStatus.FAILED, TraceEventKind.RUN_FAILED),
    ],
)
def test_terminal_event_is_atomic_idempotent_and_cannot_be_extended(
    tmp_path, status, kind
):
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        trace = _start(store)
        terminal = trace.record(kind)
        with pytest.raises(RawStoreIntegrityError, match="finish_attempt"):
            store.append_trace_event(terminal)
        for _ in range(2):
            store.finish_attempt(
                "run-1", status=status, finished_at_ms=1100, terminal_event=terminal
            )
        assert store.get_attempt("run-1").status is status
        assert store.list_trace_events("run-1") == trace.events
        with pytest.raises(RawStoreConflictError):
            store.append_trace_event(
                trace.record(TraceEventKind.NODE_COMPLETED, node="late")
            )


def test_failed_terminal_write_rolls_back_attempt_status(tmp_path):
    path = tmp_path / "raw.sqlite"
    with open_raw_run_store(path) as store:
        trace = _start(store)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TRIGGER fail_trace BEFORE INSERT ON trace_events BEGIN SELECT RAISE(ABORT, 'test-only'); END"
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.finish_attempt(
                "run-1",
                status=AttemptStatus.FAILED,
                finished_at_ms=1100,
                terminal_event=trace.record(TraceEventKind.RUN_FAILED),
            )
        assert store.get_attempt("run-1").status is AttemptStatus.RUNNING
        assert len(store.list_trace_events("run-1")) == 1


def test_order_identity_and_terminal_status_are_strict(tmp_path):
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        trace = _start(store)
        with pytest.raises(RawStoreConflictError):
            store.append_trace_event(
                TraceEvent(
                    run_id="run-1",
                    sequence=0,
                    kind=TraceEventKind.RUN_STARTED,
                    payload={"different": True},
                )
            )
        with pytest.raises(RawStoreIntegrityError, match="contiguous"):
            store.append_trace_event(
                TraceEvent(
                    run_id="run-1", sequence=3, kind=TraceEventKind.NODE_COMPLETED
                )
            )
        with pytest.raises(RawStoreIntegrityError, match="start exactly once"):
            store.append_trace_event(
                TraceEvent(run_id="run-1", sequence=1, kind=TraceEventKind.RUN_STARTED)
            )
        with pytest.raises(RawStoreIntegrityError, match="match attempt"):
            store.finish_attempt(
                "run-1",
                status=AttemptStatus.FAILED,
                finished_at_ms=1100,
                terminal_event=trace.record(TraceEventKind.RUN_FINISHED),
            )


def test_concurrent_duplicate_event_has_one_identity(tmp_path):
    path = tmp_path / "raw.sqlite"
    with open_raw_run_store(path) as store:
        trace = _start(store)
    event = trace.record(TraceEventKind.NODE_COMPLETED, node="agent")

    def append(_):
        with open_raw_run_store(path) as store:
            return store.append_trace_event(event)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(append, range(8))) == 1


@pytest.mark.parametrize(
    "bad", ["identity", "sequence", "duplicate_keys", "huge", "malformed"]
)
def test_corrupt_journal_fails_closed(tmp_path, bad, monkeypatch):
    path = tmp_path / "raw.sqlite"
    with open_raw_run_store(path) as store:
        trace = _start(store)
        raw = trace.events[0].model_dump_json()
        if bad == "identity":
            raw = raw.replace("run-1", "another-run")
        elif bad == "sequence":
            raw = raw.replace('"sequence":0', '"sequence":1')
        elif bad == "duplicate_keys":
            raw = raw.replace('"payload":{}', '"payload":{"x":1,"x":2}')
        elif bad == "huge":
            monkeypatch.setattr(module, "MAX_TRACE_BYTES", 16)
        else:
            raw = "not-json"
        with sqlite3.connect(path) as db:
            db.execute("UPDATE trace_events SET event_json = ?", (raw,))
        with pytest.raises(RawStoreCorruptionError):
            store.list_trace_events("run-1")


def test_trace_limits_fail_before_writing(tmp_path, monkeypatch):
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        trace = _start(store)
        event = trace.record(TraceEventKind.NODE_COMPLETED)
        monkeypatch.setattr(module, "MAX_TRACE_EVENTS", 1)
        with pytest.raises(RawStoreIntegrityError):
            store.append_trace_event(event)
        monkeypatch.setattr(module, "MAX_TRACE_EVENTS", 10000)
        monkeypatch.setattr(module, "MAX_TRACE_EVENT_BYTES", 1)
        with pytest.raises(RawStoreIntegrityError):
            store.append_trace_event(event)


def test_v1_migration_preserves_attempts_and_does_not_invent_history(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with open_raw_run_store(path) as store:
        store.record_attempt(_attempt())
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE trace_events")
        db.execute("PRAGMA user_version=1")
    with open_raw_run_store(path) as store:
        assert store.schema_version == 2
        assert store.get_attempt("run-1") == _attempt()
        assert store.list_trace_events("run-1") is None


def test_corrupt_v1_migration_is_atomic(tmp_path):
    path = tmp_path / "bad-legacy.sqlite"
    with open_raw_run_store(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE trace_events")
        db.execute("DROP TABLE model_invocations")
        db.execute("PRAGMA user_version=1")
    with pytest.raises(RawStoreMigrationError):
        with open_raw_run_store(path):
            pass
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='trace_events'"
        ).fetchall()
