"""Attempt-local cancellation and deadlines, independent of graph state."""

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ModelCallPolicy:
    request_timeout_seconds: float = 60.0
    attempt_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        for value in (self.request_timeout_seconds, self.attempt_timeout_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("model timeouts must be positive finite seconds")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("model timeouts must be positive finite seconds")


class ModelCallTimeout(TimeoutError):
    """One provider request exceeded its transport or total deadline."""


class AttemptDeadlineExceeded(TimeoutError):
    """No further model/tool work may start in this execution attempt."""


@dataclass(frozen=True, slots=True)
class _ExecutionControl:
    check_cancellation: Callable[[], None]
    deadline: float


_execution: ContextVar[_ExecutionControl | None] = ContextVar(
    "repocompass_model_execution", default=None
)


@contextmanager
def model_execution_scope(
    check_cancellation: Callable[[], None], *, timeout_seconds: float
) -> Iterator[None]:
    token = _execution.set(
        _ExecutionControl(check_cancellation, monotonic() + timeout_seconds)
    )
    try:
        yield
    finally:
        _execution.reset(token)


def check_attempt_deadline() -> None:
    control = _execution.get()
    if control is not None and monotonic() >= control.deadline:
        raise AttemptDeadlineExceeded("execution attempt deadline exceeded")


async def await_model_request(
    invoke: Callable[[], Awaitable[T]], *, timeout_seconds: float
) -> T:
    """Cancel and join the actual async request before propagating control errors."""
    control = _execution.get()

    def check_control() -> None:
        if control is not None:
            control.check_cancellation()
        check_attempt_deadline()

    check_control()
    deadline = monotonic() + timeout_seconds
    task = asyncio.create_task(invoke())
    try:
        while True:
            done, _ = await asyncio.wait(
                {task}, timeout=min(0.1, max(0, deadline - monotonic()))
            )
            # Preserve an already received result/usage. A later cancellation is
            # handled at the next graph boundary instead of discarding that result.
            if done:
                return task.result()
            check_control()
            if monotonic() >= deadline:
                raise ModelCallTimeout("model request deadline exceeded")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
