"""Continuous SSE stays bounded, ordered, read-only and explicit on failure."""

import asyncio

import pytest

from repocompass.agent.trace import TraceEventKind
from repocompass.dashboard.contracts import DashboardAvailability, PublicTraceEvent
from repocompass.dashboard.events import EventPage, follow_events
from repocompass.storage.raw_store import AttemptStatus


def _page(sequences=(), status=AttemptStatus.RUNNING, truncated=False):
    return EventPage(
        DashboardAvailability.AVAILABLE,
        status,
        tuple(
            PublicTraceEvent(
                run_id="run-1",
                sequence=i,
                kind=TraceEventKind.NODE_COMPLETED,
                node="agent",
            )
            for i in sequences
        ),
        truncated,
    )


async def _connected():
    return False


def _collect(initial, reader, **kwargs):
    async def collect():
        return "".join(
            [
                part
                async for part in follow_events(
                    initial=initial,
                    read_page=reader,
                    disconnected=_connected,
                    after_sequence=-1,
                    max_events=10,
                    poll_seconds=0,
                    **kwargs,
                )
            ]
        )

    return asyncio.run(collect())


def test_follow_appends_committed_pages_once_then_stops_at_terminal():
    calls = []

    def read(cursor, limit):
        calls.append((cursor, limit))
        return _page([1, 2], AttemptStatus.COMPLETED)

    text = _collect(_page([0]), read)
    assert calls == [(0, 9)]
    assert text.count("event: trace") == 3
    assert text.count("id: 0\n") == 1
    assert ": heartbeat" in text
    assert '"reason": "terminal"' in text
    assert '"after_sequence": 2' in text


@pytest.mark.parametrize("kind", ["deadline", "byte_limit", "gap", "error"])
def test_follow_limits_and_errors_do_not_claim_completion(kind):
    def read(*_):
        raise RuntimeError("secret local credential/path")

    args = {"duration_seconds": 0} if kind == "deadline" else {}
    if kind == "byte_limit":
        args["max_bytes"] = 1
    text = _collect(_page([2 if kind == "gap" else 0]), read, **args)
    assert '"reason": "terminal"' not in text
    assert "secret" not in text
    assert {
        "deadline": "deadline",
        "byte_limit": "byte_limit",
        "gap": "event_gap",
        "error": "read_failed",
    }[kind] in text


def test_disconnect_does_not_read_or_emit_more_events():
    async def disconnected():
        return True

    def forbidden(*_):
        raise AssertionError("no read on disconnect")

    async def collect():
        return [
            x
            async for x in follow_events(
                initial=_page([0]),
                read_page=forbidden,
                disconnected=disconnected,
                after_sequence=-1,
                max_events=1,
            )
        ]

    assert asyncio.run(collect()) == []


def test_event_limit_keeps_last_cursor_for_explicit_reconnect():
    async def collect():
        return "".join(
            [
                x
                async for x in follow_events(
                    initial=_page([0], truncated=True),
                    read_page=lambda *_: pytest.fail("limit must stop"),
                    disconnected=_connected,
                    after_sequence=-1,
                    max_events=1,
                )
            ]
        )

    text = asyncio.run(collect())
    assert '"reason": "event_limit"' in text
    assert '"after_sequence": 0' in text
