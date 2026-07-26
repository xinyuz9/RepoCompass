"""Unit tests for the in-memory RepoCompass domain trace."""

from pathlib import Path

import pytest

from repocompass.agent.trace import RunTrace, TraceEventKind


def test_run_trace_records_ordered_events_and_snapshots() -> None:
    trace = RunTrace("  test-run  ")
    source_payload = {"count": 1}

    started = trace.record(
        TraceEventKind.RUN_STARTED,
        payload=source_payload,
    )
    snapshot = trace.events
    completed = trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="agent",
        payload={"path": "src/example.py"},
    )

    source_payload["count"] = 99

    assert trace.run_id == "test-run"
    assert started.sequence == 0
    assert completed.sequence == 1
    assert started.payload == {"count": 1}
    assert snapshot == (started,)
    assert trace.events == (started, completed)
    assert tuple(trace) == trace.events
    assert len(trace) == 2


@pytest.mark.parametrize("run_id", ["", "   "])
def test_run_trace_rejects_empty_run_id(run_id: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RunTrace(run_id)


def test_run_trace_rejects_non_string_run_id() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        RunTrace(123)  # type: ignore[arg-type]


def test_trace_event_payload_is_json_serializable(tmp_path: Path) -> None:
    trace = RunTrace("json-run")
    event = trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="example",
        payload={
            "path": str(tmp_path),
            "values": [1, True, None],
        },
    )

    dumped = event.model_dump(mode="json")

    assert dumped["kind"] == "node_completed"
    assert dumped["payload"]["values"] == [1, True, None]
    assert event.model_dump_json()
