"""In-memory domain-trace contracts for a RepoCompass investigation run.

This module defines trace data and collection interfaces only. Graph execution,
LangGraph update translation, persistence, and remote observability belong to
the Runner, ``JsonlTraceSink``, and LangSmith respectively.
"""

from collections.abc import Iterator, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TraceEventKind(StrEnum):
    """Finite event vocabulary for the minimal M0 domain trace."""

    RUN_STARTED = "run_started"
    NODE_COMPLETED = "node_completed"
    RUN_FINISHED = "run_finished"


class TraceEvent(BaseModel):
    """One ordered, JSON-compatible fact observed during a run.

    Attributes:
        run_id: Stable identifier shared by every event from the same run.
        sequence: Zero-based position assigned when the event is recorded.
        kind: Controlled event category used by tests and future sinks.
        node: Graph node associated with the event, when applicable.
        payload: Event-specific data containing JSON-compatible values only.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    kind: TraceEventKind
    node: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class RunTrace:
    """Own the ordered in-memory domain events for exactly one run.

    The class must not invoke the graph, inspect LangChain messages, write files,
    or send data to LangSmith. A future Runner will translate graph updates into
    calls to :meth:`record`.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty trace for ``run_id``.

        Args:
            run_id: Non-empty identifier supplied by the orchestration layer.

        Returns:
            None.

        Side effects:
            Initializes private in-memory storage only. It must perform no file,
            network, environment-variable, or graph operations.
        """

        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")

        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")

        self._run_id = normalized_run_id
        self._events: list[TraceEvent] = []

    @property
    def run_id(self) -> str:
        """Return the stable identifier for this trace.

        Returns:
            The non-empty run identifier passed to :meth:`__init__`.

        Side effects:
            None.
        """
        return self._run_id

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return an immutable snapshot of events in recording order.

        Returns:
            A tuple ordered by ascending ``TraceEvent.sequence``. Mutating the
            returned container must not mutate the trace's internal storage.

        Side effects:
            None.
        """
        return tuple(self._events)

    def record(
        self,
        kind: TraceEventKind,
        *,
        node: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> TraceEvent:
        """Validate, append, and return one event.

        Args:
            kind: Controlled category describing the observed fact.
            node: Related graph node, or ``None`` for run-level events.
            payload: Event-specific JSON-compatible values. ``None`` represents
                an empty payload. The implementation should defensively copy
                caller-owned mappings before storing them.

        Returns:
            The appended :class:`TraceEvent`, with ``run_id`` filled from this
            trace and ``sequence`` assigned from the next zero-based position.

        Side effects:
            Appends exactly one event to this trace's private in-memory storage.
            It must not write files, call the graph, or contact LangSmith.
        """

        event_payload = dict(payload) if payload is not None else {}

        event = TraceEvent(
            run_id=self._run_id,
            sequence=len(self._events),
            kind=kind,
            node=node,
            payload=event_payload,
        )

        self._events.append(event)
        return event

    def __iter__(self) -> Iterator[TraceEvent]:
        """Iterate over a stable snapshot in recording order.

        Returns:
            An iterator over the same ordered snapshot exposed by
            :attr:`events`.

        Side effects:
            None. Iteration must not consume or mutate stored events.
        """
        return iter(self.events)

    def __len__(self) -> int:
        """Return the number of recorded events.

        Returns:
            The current non-negative event count.

        Side effects:
            None.
        """
        return len(self._events)
