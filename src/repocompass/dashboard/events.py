"""Bounded read-only SSE follow sessions over durable, sanitized event pages."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

from starlette.concurrency import run_in_threadpool

from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardEventStreamCursor,
    PublicTraceEvent,
)
from repocompass.storage.raw_store import AttemptStatus


@dataclass(frozen=True)
class EventPage:
    availability: DashboardAvailability
    status: AttemptStatus
    events: tuple[PublicTraceEvent, ...]
    truncated: bool


def encode_sse(*, event: str, data: str, event_id: str | None = None) -> str:
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.extend(f"data: {line}" for line in data.splitlines())
    return "\n".join(lines) + "\n\n"


async def follow_events(
    *,
    initial: EventPage,
    read_page: Callable[[int, int], EventPage],
    disconnected: Callable[[], Awaitable[bool]],
    after_sequence: int,
    max_events: int,
    duration_seconds: float = 30,
    poll_seconds: float = 1,
    max_bytes: int = 4 * 1024 * 1024,
    clock: Callable[[], float] = monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[str]:
    """No unbounded queue, implicit mutations, provider calls or hidden retries."""
    started = clock()
    cursor = after_sequence
    count = size = 0
    page = initial
    while True:
        if await disconnected():
            return
        for item in page.events:
            if item.sequence != cursor + 1:
                yield encode_sse(event="stream_error", data='{"code":"event_gap"}')
                return
            frame = encode_sse(
                event="trace", event_id=str(item.sequence), data=item.model_dump_json()
            )
            if size + len(frame.encode("utf-8")) > max_bytes:
                yield encode_sse(
                    event="stream_end",
                    data=json.dumps({"reason": "byte_limit", "after_sequence": cursor}),
                )
                return
            if await disconnected():
                return
            yield frame
            size += len(frame.encode("utf-8"))
            cursor = item.sequence
            count += 1
        snapshot = DashboardEventStreamCursor(
            availability=page.availability,
            after_sequence=after_sequence,
            emitted_count=count,
            next_after_sequence=cursor if count else None,
            truncated=page.truncated,
        )
        yield encode_sse(event="snapshot_end", data=snapshot.model_dump_json())
        reason = (
            "event_limit"
            if count >= max_events
            else "terminal"
            if page.status is not AttemptStatus.RUNNING
            else "deadline"
            if clock() - started >= duration_seconds
            else None
        )
        if reason:
            yield encode_sse(
                event="stream_end",
                data=json.dumps(
                    {
                        "reason": reason,
                        "after_sequence": cursor,
                        "status": page.status.value,
                    }
                ),
            )
            return
        yield ": heartbeat\n\n"
        await sleep(min(poll_seconds, max(0, duration_seconds - (clock() - started))))
        if await disconnected():
            return
        try:
            page = await run_in_threadpool(read_page, cursor, max_events - count)
        except Exception:
            # HTTP headers have already been sent; never expose local exception text.
            yield encode_sse(event="stream_error", data='{"code":"read_failed"}')
            return
