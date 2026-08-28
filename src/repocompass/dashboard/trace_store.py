"""Bounded, strict reader for append-only RepoCompass JSONL traces."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from repocompass.agent.trace import TraceEvent
from repocompass.tools.contracts import IDENTIFIER_PATTERN

MAX_TRACE_FILE_BYTES = 16 * 1024 * 1024
MAX_TRACE_EVENT_BYTES = 1024 * 1024
MAX_TRACE_EVENTS = 10_000
_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)


class TraceStoreError(RuntimeError):
    """Base error for safe dashboard trace reads."""


class TraceStoreCorruptionError(TraceStoreError):
    """A trace file violates its bounded, ordered JSONL contract."""


class JsonlTraceStore:
    """Load known run IDs from one operator-selected trace directory."""

    def __init__(self, directory: Path) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory must be a Path")
        self._directory = directory.expanduser().resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    def load(self, run_id: str) -> tuple[TraceEvent, ...] | None:
        """Return one strictly validated trace, or ``None`` when not persisted."""

        normalized_run_id = _validate_run_id(run_id)
        path = self._directory / f"{normalized_run_id}.jsonl"
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise TraceStoreError("trace metadata could not be read") from error
        if not path.is_file():
            raise TraceStoreCorruptionError("trace artifact is not a regular file")
        if metadata.st_size > MAX_TRACE_FILE_BYTES:
            raise TraceStoreCorruptionError("trace artifact exceeds the byte limit")

        events: list[TraceEvent] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line_number > MAX_TRACE_EVENTS:
                        raise TraceStoreCorruptionError(
                            "trace artifact exceeds the event limit"
                        )
                    if len(line.encode("utf-8")) > MAX_TRACE_EVENT_BYTES:
                        raise TraceStoreCorruptionError(
                            "trace event exceeds the byte limit"
                        )
                    if not line.endswith("\n"):
                        raise TraceStoreCorruptionError(
                            "trace event is not newline terminated"
                        )
                    try:
                        payload = json.loads(
                            line,
                            object_pairs_hook=_unique_json_object,
                        )
                        event = TraceEvent.model_validate(payload, strict=False)
                    except (
                        json.JSONDecodeError,
                        UnicodeError,
                        ValidationError,
                        ValueError,
                    ) as error:
                        raise TraceStoreCorruptionError(
                            "trace event failed strict validation"
                        ) from error
                    expected_sequence = len(events)
                    if (
                        event.run_id != normalized_run_id
                        or event.sequence != expected_sequence
                    ):
                        raise TraceStoreCorruptionError(
                            "trace event identity or sequence is inconsistent"
                        )
                    events.append(event)
        except TraceStoreError:
            raise
        except (OSError, UnicodeError) as error:
            raise TraceStoreError("trace artifact could not be read") from error
        return tuple(events)


def _validate_run_id(value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("run_id is invalid")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
