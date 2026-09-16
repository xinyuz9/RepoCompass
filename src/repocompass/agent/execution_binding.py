"""Bind a public lifecycle request to its run before any model invocation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "run_observer", default=None
)


@contextmanager
def bind_run_observer(observer: Callable[[str], None]) -> Iterator[None]:
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def announce_run(run_id: str) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(run_id)
