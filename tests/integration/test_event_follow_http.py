"""A real socket must deliver journal events before an attempt has finished."""

import socket
import threading
import time
from contextlib import ExitStack
from unittest.mock import MagicMock

import httpx
import uvicorn

from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.storage.raw_store import AttemptStatus, open_raw_run_store
from repocompass.tools.gateway import ToolExecutionContext
from tests.unit.dashboard.test_service import _attempt


def test_socket_follow_delivers_before_finalization_and_reconnects_read_only(tmp_path):
    context = ToolExecutionContext(tmp_path)
    with open_raw_run_store(tmp_path / "raw.sqlite") as raw:
        raw.record_attempt(_attempt(context, status=AttemptStatus.RUNNING))
        trace = RunTrace("run-1")
        raw.append_trace_event(
            trace.record(
                TraceEventKind.RUN_STARTED, payload={"api_key": "never-public"}
            )
        )
        runner = MagicMock()
        query = DashboardQueryService(
            runner,
            raw_run_store=raw,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=tmp_path,
        )
        # A subscription must never build the full context/cost projection.
        query.get_run = MagicMock(
            side_effect=AssertionError("full run should not be read")
        )
        app = create_dashboard_app(query)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            base = f"http://127.0.0.1:{listener.getsockname()[1]}"
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="error", lifespan="off")
            )
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            thread.start()
            try:
                deadline = time.monotonic() + 5
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert server.started
                path = "/api/v1/runs/run-1/events"
                with httpx.stream(
                    "GET", base + path + "?follow=true", timeout=5, trust_env=False
                ) as response:
                    assert response.status_code == 200
                    lines = response.iter_lines()
                    first = []
                    while True:
                        line = next(lines)
                        first.append(line)
                        if line == "":
                            break
                    assert "event: trace" in first and "id: 0" in first
                    assert "never-public" not in "".join(first)
                    assert raw.get_attempt("run-1").status is AttemptStatus.RUNNING
                    raw.append_trace_event(
                        trace.record(TraceEventKind.NODE_COMPLETED, node="agent")
                    )
                    # Reading the next event before writing terminal proves no response buffering.
                    while next(lines) != "id: 1":
                        pass
                    assert raw.get_attempt("run-1").status is AttemptStatus.RUNNING
                with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
                    replay = client.get(path, headers={"Last-Event-ID": "0"})
                    assert replay.text.count("event: trace") == 1
                    assert "id: 1" in replay.text
                    assert client.get(path + "?after_sequence=99").status_code == 409
                    raw.record_attempt(
                        _attempt(context, run_id="foreign", repository_id="f" * 64)
                    )
                    assert (
                        client.get(
                            "/api/v1/runs/foreign/events?follow=true"
                        ).status_code
                        == 404
                    )
                with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
                    with ExitStack() as subscriptions:
                        for _ in range(8):
                            response = subscriptions.enter_context(
                                client.stream("GET", path + "?follow=true")
                            )
                            assert response.status_code == 200
                        assert client.get(path + "?follow=true").status_code == 429
                    deadline = time.monotonic() + 5
                    while True:
                        # This bounded one-event connection finishes immediately.
                        reclaimed = client.get(path + "?follow=true&max_events=1")
                        if reclaimed.status_code == 200:
                            break
                        assert (
                            reclaimed.status_code == 429 and time.monotonic() < deadline
                        )
                        time.sleep(0.01)
                    assert '"reason": "event_limit"' in reclaimed.text
                runner.assert_not_called()
                query.get_run.assert_not_called()
                assert raw.get_attempt("run-1").status is AttemptStatus.RUNNING
            finally:
                server.should_exit = True
                thread.join(5)
                assert not thread.is_alive()
