"""Local checkpoint persistence for RepoCompass CLI investigations.

This module owns the SQLite connection lifecycle and its narrow domain serde
boundary. Thread identity, graph execution, and public policy belong to the
Runner and CLI layers.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import HumanReviewPolicy, TurnRecord
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult


_CHECKPOINT_MODEL_TYPES: tuple[type[BaseModel], ...] = (
    CandidateResult,
    EvaluationReport,
    InvestigationPlan,
    FinalResult,
    TurnRecord,
)
_CHECKPOINT_ALLOWED_TYPES: tuple[type[object], ...] = (
    *_CHECKPOINT_MODEL_TYPES,
    HumanReviewPolicy,
)


def _rehydrate_checkpoint_value(value: object) -> object:
    """Revalidate strict domain models revived through msgpack fallback."""

    if isinstance(value, _CHECKPOINT_MODEL_TYPES):
        model_type = type(value)
        raw_fields = {
            key: _rehydrate_checkpoint_value(item)
            for key, item in value.__dict__.items()
        }
        return model_type.model_validate(raw_fields, strict=False)
    if isinstance(value, dict):
        return {key: _rehydrate_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_checkpoint_value(item) for item in value)

    return value


class RepoCompassCheckpointSerializer(JsonPlusSerializer):
    """Msgpack serializer with a narrow RepoCompass domain allowlist."""

    def __init__(self) -> None:
        super().__init__(
            pickle_fallback=False,
            allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES,
        )

    def loads_typed(self, data: tuple[str, bytes]) -> object:
        """Decode and restore nested strict Pydantic domain values."""

        return _rehydrate_checkpoint_value(super().loads_typed(data))


@contextmanager
def open_sqlite_checkpointer(database_path: Path) -> Iterator[SqliteSaver]:
    """Open one local SQLite checkpointer and close it on context exit.

    Args:
        database_path: SQLite database file. Missing parent directories are
            created before the connection is opened.

    Yields:
        A synchronous LangGraph SQLite checkpointer bound to the resolved file.

    Side effects:
        Creates missing parent directories, opens the SQLite database, and
        closes its connection when the context exits.
    """

    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(
        sqlite3.connect(
            str(resolved_path),
            check_same_thread=False,
        )
    ) as connection:
        yield SqliteSaver(
            connection,
            serde=RepoCompassCheckpointSerializer(),
        )
