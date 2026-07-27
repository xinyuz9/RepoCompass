"""Unit tests for JSONL domain-trace persistence."""

import json
from pathlib import Path

import pytest

from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.agent.trace_sink import JsonlTraceSink


def test_jsonl_trace_sink_writes_one_event_per_line(tmp_path: Path) -> None:
    trace = RunTrace("sink-run")
    trace.record(
        TraceEventKind.RUN_STARTED,
        payload={"objective": "Inspect authentication."},
    )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="agent",
        payload={"step_explorer_call_count": 1},
    )
    destination = tmp_path / "nested" / "sink-run.jsonl"

    result = JsonlTraceSink().write(trace, destination)

    lines = result.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert result == destination.resolve()
    assert [event["sequence"] for event in events] == [0, 1]
    assert [event["kind"] for event in events] == [
        "run_started",
        "node_completed",
    ]
    assert events[1]["node"] == "agent"
    assert events[1]["payload"] == {"step_explorer_call_count": 1}


def test_jsonl_trace_sink_does_not_overwrite_existing_file(
    tmp_path: Path,
) -> None:
    trace = RunTrace("sink-run")
    trace.record(TraceEventKind.RUN_STARTED)
    destination = tmp_path / "sink-run.jsonl"
    destination.write_text("existing trace\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        JsonlTraceSink().write(trace, destination)

    assert destination.read_text(encoding="utf-8") == "existing trace\n"
