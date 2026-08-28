"""Bounded JSONL dashboard trace reader tests."""

from pathlib import Path

import pytest

from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.dashboard import trace_store as trace_store_module
from repocompass.dashboard.trace_store import (
    JsonlTraceStore,
    TraceStoreCorruptionError,
)


def _write_trace(directory: Path, run_id: str = "run-1") -> RunTrace:
    trace = RunTrace(run_id)
    trace.record(
        TraceEventKind.RUN_STARTED,
        payload={"objective": "Inspect authentication."},
    )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="plan_investigation",
        payload={"step": 1},
    )
    JsonlTraceSink().write(trace, directory / f"{run_id}.jsonl")
    return trace


def test_trace_store_round_trips_known_run_and_reports_missing(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path)
    store = JsonlTraceStore(tmp_path)

    assert store.load("run-1") == trace.events
    assert store.load("missing-run") is None


def test_trace_store_rejects_traversal_and_inconsistent_identity(
    tmp_path: Path,
) -> None:
    store = JsonlTraceStore(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        store.load("../outside")

    path = tmp_path / "run-1.jsonl"
    path.write_text(
        '{"run_id":"other","sequence":0,"kind":"run_started",'
        '"node":null,"payload":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceStoreCorruptionError, match="identity"):
        store.load("run-1")


def test_trace_store_rejects_duplicate_keys_and_unterminated_lines(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"run_id":"duplicate","run_id":"duplicate","sequence":0,'
        '"kind":"run_started","node":null,"payload":{}}\n',
        encoding="utf-8",
    )
    store = JsonlTraceStore(tmp_path)
    with pytest.raises(TraceStoreCorruptionError, match="validation"):
        store.load("duplicate")

    unterminated = tmp_path / "unterminated.jsonl"
    unterminated.write_text(
        '{"run_id":"unterminated","sequence":0,"kind":"run_started",'
        '"node":null,"payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(TraceStoreCorruptionError, match="newline"):
        store.load("unterminated")


def test_trace_store_enforces_event_count_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_trace(tmp_path)
    monkeypatch.setattr(trace_store_module, "MAX_TRACE_EVENTS", 1)

    with pytest.raises(TraceStoreCorruptionError, match="event limit"):
        JsonlTraceStore(tmp_path).load("run-1")
