"""Versioned transactional SQLite store for raw run artifacts and metadata."""

import json
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, model_validator

from repocompass.agent.context import (
    ContextPack,
    RawArtifactKind,
    RawArtifactMetadata,
    create_raw_artifact_metadata,
)
from repocompass.agent.cost import (
    AttemptUsageCostReport,
    PricingTable,
    build_attempt_usage_cost_report,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.usage import ModelInvocationRecord
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

RAW_STORE_SCHEMA_VERSION = 1
RAW_STORE_CONTRACT_SCHEMA_VERSION = 1
MAX_RAW_ARTIFACT_BYTES = 4 * 1024 * 1024
_SQLITE_TIMEOUT_SECONDS = 10.0
_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)


class RawStoreError(RuntimeError):
    """Base error for safe Raw Run Store failures."""


class RawStoreMigrationError(RawStoreError):
    """The database version or physical schema is not safely supported."""


class RawStoreConflictError(RawStoreError):
    """An immutable identity already exists with different metadata."""


class RawStoreIntegrityError(RawStoreError):
    """A write would break attempt, artifact, or provenance integrity."""


class RawStoreCorruptionError(RawStoreError):
    """Persisted JSON or duplicated columns fail strict revalidation."""


class AttemptExecutionMode(StrEnum):
    """Lifecycle entry point that produced one observable execution attempt."""

    START = "start"
    RESUME = "resume"
    CONTINUE = "continue"
    RESPOND = "respond"


class AttemptStatus(StrEnum):
    """Storage lifecycle of one execution attempt."""

    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


_TERMINAL_ATTEMPT_STATUSES = frozenset(
    {
        AttemptStatus.COMPLETED,
        AttemptStatus.INTERRUPTED,
        AttemptStatus.FAILED,
    }
)


class RawRunAttempt(CheckpointSafeModel):
    """Bounded attempt metadata; objective text remains a raw artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    schema_version: Literal[1] = RAW_STORE_CONTRACT_SCHEMA_VERSION
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    execution_mode: AttemptExecutionMode
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    objective_digest: Sha256Digest
    status: AttemptStatus
    started_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    finished_at_ms: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )

    @model_validator(mode="after")
    def require_canonical_lifecycle_timestamps(self) -> "RawRunAttempt":
        if self.status is AttemptStatus.RUNNING:
            if self.finished_at_ms is not None:
                raise ValueError("running attempt must not have finished_at_ms")
            return self
        if self.finished_at_ms is None:
            raise ValueError("terminal attempt requires finished_at_ms")
        if self.finished_at_ms < self.started_at_ms:
            raise ValueError("finished_at_ms must not precede started_at_ms")
        return self


@dataclass(frozen=True, slots=True)
class StoredRawArtifact:
    """Strict metadata and exact local bytes returned from the raw store."""

    metadata: RawArtifactMetadata
    content: bytes


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE attempts (
        run_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL CHECK(turn_number >= 1),
        execution_mode TEXT NOT NULL
            CHECK(execution_mode IN ('start', 'resume', 'continue', 'respond')),
        repository_id TEXT NOT NULL CHECK(length(repository_id) = 64),
        commit_sha TEXT,
        objective_digest TEXT NOT NULL CHECK(length(objective_digest) = 64),
        status TEXT NOT NULL
            CHECK(status IN ('running', 'completed', 'interrupted', 'failed')),
        started_at_ms INTEGER NOT NULL CHECK(started_at_ms >= 0),
        finished_at_ms INTEGER,
        record_json TEXT NOT NULL,
        UNIQUE(run_id, thread_id, turn_number),
        CHECK(
            (status = 'running' AND finished_at_ms IS NULL)
            OR (status != 'running' AND finished_at_ms >= started_at_ms)
        )
    )
    """,
    """
    CREATE TABLE raw_artifacts (
        artifact_id TEXT PRIMARY KEY CHECK(length(artifact_id) = 64),
        kind TEXT NOT NULL,
        media_type TEXT NOT NULL,
        content_digest TEXT NOT NULL CHECK(length(content_digest) = 64),
        byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
        metadata_json TEXT NOT NULL,
        content BLOB NOT NULL,
        CHECK(byte_count = length(content))
    )
    """,
    """
    CREATE TABLE attempt_artifacts (
        run_id TEXT NOT NULL,
        artifact_id TEXT NOT NULL,
        PRIMARY KEY(run_id, artifact_id),
        FOREIGN KEY(run_id) REFERENCES attempts(run_id) ON DELETE CASCADE,
        FOREIGN KEY(artifact_id) REFERENCES raw_artifacts(artifact_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE context_builds (
        run_id TEXT NOT NULL,
        build_id TEXT NOT NULL CHECK(length(build_id) = 64),
        pack_id TEXT NOT NULL CHECK(length(pack_id) = 64),
        sequence INTEGER NOT NULL CHECK(sequence >= 0),
        budget_tokens INTEGER NOT NULL CHECK(budget_tokens >= 1),
        selected_tokens INTEGER NOT NULL CHECK(selected_tokens >= 0),
        candidate_count INTEGER NOT NULL CHECK(candidate_count >= 1),
        selected_count INTEGER NOT NULL CHECK(selected_count >= 0),
        excluded_count INTEGER NOT NULL CHECK(excluded_count >= 0),
        pack_json TEXT NOT NULL,
        PRIMARY KEY(run_id, build_id),
        UNIQUE(run_id, pack_id),
        UNIQUE(run_id, sequence),
        FOREIGN KEY(run_id) REFERENCES attempts(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE model_invocations (
        invocation_id TEXT PRIMARY KEY CHECK(length(invocation_id) = 64),
        run_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL CHECK(turn_number >= 1),
        sequence INTEGER NOT NULL CHECK(sequence >= 0),
        node_name TEXT NOT NULL,
        role TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed')),
        latency_ms INTEGER NOT NULL CHECK(latency_ms >= 0),
        usage_availability TEXT NOT NULL
            CHECK(usage_availability IN ('reported', 'unavailable')),
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        cached_input_tokens INTEGER,
        reasoning_output_tokens INTEGER,
        plan_step_id TEXT,
        context_pack_id TEXT NOT NULL,
        input_artifact_id TEXT NOT NULL,
        output_artifact_id TEXT,
        error_code TEXT,
        record_json TEXT NOT NULL,
        UNIQUE(run_id, sequence),
        FOREIGN KEY(run_id, thread_id, turn_number)
            REFERENCES attempts(run_id, thread_id, turn_number) ON DELETE CASCADE,
        FOREIGN KEY(run_id, input_artifact_id)
            REFERENCES attempt_artifacts(run_id, artifact_id) ON DELETE CASCADE,
        FOREIGN KEY(run_id, output_artifact_id)
            REFERENCES attempt_artifacts(run_id, artifact_id) ON DELETE CASCADE,
        FOREIGN KEY(run_id, context_pack_id)
            REFERENCES context_builds(run_id, pack_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX idx_attempts_thread_turn ON attempts(thread_id, turn_number)",
    "CREATE INDEX idx_attempt_artifacts_artifact ON attempt_artifacts(artifact_id)",
    "CREATE INDEX idx_model_invocations_run_sequence "
    "ON model_invocations(run_id, sequence)",
)

_REQUIRED_COLUMNS = {
    "attempts": (
        "run_id",
        "thread_id",
        "turn_number",
        "execution_mode",
        "repository_id",
        "commit_sha",
        "objective_digest",
        "status",
        "started_at_ms",
        "finished_at_ms",
        "record_json",
    ),
    "raw_artifacts": (
        "artifact_id",
        "kind",
        "media_type",
        "content_digest",
        "byte_count",
        "metadata_json",
        "content",
    ),
    "attempt_artifacts": ("run_id", "artifact_id"),
    "context_builds": (
        "run_id",
        "build_id",
        "pack_id",
        "sequence",
        "budget_tokens",
        "selected_tokens",
        "candidate_count",
        "selected_count",
        "excluded_count",
        "pack_json",
    ),
    "model_invocations": (
        "invocation_id",
        "run_id",
        "thread_id",
        "turn_number",
        "sequence",
        "node_name",
        "role",
        "provider",
        "model",
        "status",
        "latency_ms",
        "usage_availability",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "plan_step_id",
        "context_pack_id",
        "input_artifact_id",
        "output_artifact_id",
        "error_code",
        "record_json",
    ),
}
_ForeignKeyEdge = tuple[str, str, str, str]
_EXPECTED_FOREIGN_KEYS: dict[str, Counter[_ForeignKeyEdge]] = {
    "attempts": Counter(),
    "raw_artifacts": Counter(),
    "attempt_artifacts": Counter(
        [
            ("attempts", "run_id", "run_id", "CASCADE"),
            ("raw_artifacts", "artifact_id", "artifact_id", "CASCADE"),
        ]
    ),
    "context_builds": Counter([("attempts", "run_id", "run_id", "CASCADE")]),
    "model_invocations": Counter(
        [
            ("attempts", "run_id", "run_id", "CASCADE"),
            ("attempts", "thread_id", "thread_id", "CASCADE"),
            ("attempts", "turn_number", "turn_number", "CASCADE"),
            ("attempt_artifacts", "run_id", "run_id", "CASCADE"),
            (
                "attempt_artifacts",
                "input_artifact_id",
                "artifact_id",
                "CASCADE",
            ),
            ("attempt_artifacts", "run_id", "run_id", "CASCADE"),
            (
                "attempt_artifacts",
                "output_artifact_id",
                "artifact_id",
                "CASCADE",
            ),
            ("context_builds", "run_id", "run_id", "CASCADE"),
            ("context_builds", "context_pack_id", "pack_id", "CASCADE"),
        ]
    ),
}
_REQUIRED_NOT_NULL_COLUMNS = {
    "model_invocations": frozenset({"context_pack_id"}),
}


class RawRunStore:
    """Own one validated SQLite connection and transactional storage API."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = threading.RLock()

    @property
    def schema_version(self) -> int:
        """Return the physical SQLite schema version."""

        with self._lock:
            row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def record_attempt(self, attempt: RawRunAttempt) -> bool:
        """Insert one immutable attempt, or accept an identical retry."""

        if not isinstance(attempt, RawRunAttempt):
            raise TypeError("attempt must be a RawRunAttempt")
        record_json = _canonical_model_json(attempt)
        with self._transaction():
            existing = self._connection.execute(
                "SELECT record_json FROM attempts WHERE run_id = ?",
                (attempt.run_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] == record_json:
                    return False
                raise RawStoreConflictError(
                    "attempt identity already exists with different metadata"
                )
            try:
                self._connection.execute(
                    """
                    INSERT INTO attempts (
                        run_id, thread_id, turn_number, execution_mode,
                        repository_id, commit_sha, objective_digest, status,
                        started_at_ms, finished_at_ms, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.run_id,
                        attempt.thread_id,
                        attempt.turn_number,
                        attempt.execution_mode.value,
                        attempt.repository_id,
                        attempt.commit_sha,
                        attempt.objective_digest,
                        attempt.status.value,
                        attempt.started_at_ms,
                        attempt.finished_at_ms,
                        record_json,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise RawStoreIntegrityError(
                    "attempt metadata violates Raw Run Store integrity"
                ) from error
        return True

    def get_attempt(self, run_id: str) -> RawRunAttempt | None:
        """Return one strictly decoded attempt without exposing raw artifacts."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else _decode_attempt_row(row)

    def list_attempts(
        self,
        *,
        max_results: int,
        thread_id: str | None = None,
    ) -> tuple[RawRunAttempt, ...]:
        """List bounded attempt metadata in stable reverse-chronological order."""

        if type(max_results) is not int or not 1 <= max_results <= 1_000:
            raise ValueError("max_results must be between 1 and 1000")
        if thread_id is not None and (
            not isinstance(thread_id, str)
            or _IDENTIFIER_RE.fullmatch(thread_id) is None
        ):
            raise ValueError("thread_id is invalid")
        predicates = "" if thread_id is None else "WHERE thread_id = ?"
        parameters: tuple[object, ...] = (
            (max_results,) if thread_id is None else (thread_id, max_results)
        )
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM attempts "
                f"{predicates} ORDER BY started_at_ms DESC, run_id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return tuple(_decode_attempt_row(row) for row in rows)

    def finish_attempt(
        self,
        run_id: str,
        *,
        status: AttemptStatus,
        finished_at_ms: int,
    ) -> RawRunAttempt:
        """Transition a running attempt once to a terminal status."""

        if status not in _TERMINAL_ATTEMPT_STATUSES:
            raise ValueError("finish_attempt requires a terminal status")
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RawStoreIntegrityError("attempt does not exist")
            current = _decode_attempt_row(row)
            if current.status is not AttemptStatus.RUNNING:
                if (
                    current.status is status
                    and current.finished_at_ms == finished_at_ms
                ):
                    return current
                raise RawStoreConflictError("attempt is already terminal")
            updated = RawRunAttempt.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "status": status,
                    "finished_at_ms": finished_at_ms,
                }
            )
            self._connection.execute(
                """
                UPDATE attempts
                SET status = ?, finished_at_ms = ?, record_json = ?
                WHERE run_id = ?
                """,
                (
                    updated.status.value,
                    updated.finished_at_ms,
                    _canonical_model_json(updated),
                    run_id,
                ),
            )
            return updated

    def put_artifact(
        self,
        run_id: str,
        metadata: RawArtifactMetadata,
        content: str | bytes,
    ) -> bool:
        """Verify, deduplicate, and link one raw artifact atomically."""

        if not isinstance(metadata, RawArtifactMetadata):
            raise TypeError("metadata must be RawArtifactMetadata")
        encoded = _encode_artifact_content(content)
        if len(encoded) > MAX_RAW_ARTIFACT_BYTES:
            raise RawStoreIntegrityError("raw artifact exceeds the storage byte limit")
        expected = create_raw_artifact_metadata(
            kind=metadata.kind,
            media_type=metadata.media_type,
            content=encoded,
        )
        if expected != metadata:
            raise RawStoreIntegrityError("raw artifact content does not match metadata")
        metadata_json = _canonical_model_json(metadata)
        inserted = False
        with self._transaction():
            self._require_attempt(run_id)
            existing = self._connection.execute(
                "SELECT metadata_json, content FROM raw_artifacts "
                "WHERE artifact_id = ?",
                (metadata.artifact_id,),
            ).fetchone()
            if existing is None:
                try:
                    self._connection.execute(
                        """
                        INSERT INTO raw_artifacts (
                            artifact_id, kind, media_type, content_digest,
                            byte_count, metadata_json, content
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            metadata.artifact_id,
                            metadata.kind.value,
                            metadata.media_type,
                            metadata.content_digest,
                            metadata.byte_count,
                            metadata_json,
                            encoded,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise RawStoreIntegrityError(
                        "raw artifact violates storage integrity"
                    ) from error
                inserted = True
            elif existing[0] != metadata_json or bytes(existing[1]) != encoded:
                raise RawStoreCorruptionError(
                    "content-addressed raw artifact has conflicting stored data"
                )
            try:
                self._connection.execute(
                    "INSERT OR IGNORE INTO attempt_artifacts (run_id, artifact_id) "
                    "VALUES (?, ?)",
                    (run_id, metadata.artifact_id),
                )
            except sqlite3.IntegrityError as error:
                raise RawStoreIntegrityError(
                    "raw artifact cannot be linked to attempt"
                ) from error
        return inserted

    def get_artifact(self, artifact_id: str) -> StoredRawArtifact | None:
        """Return exact local bytes after metadata and digest revalidation."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM raw_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        metadata = _decode_artifact_row(row)
        content = bytes(row["content"])
        expected = create_raw_artifact_metadata(
            kind=metadata.kind,
            media_type=metadata.media_type,
            content=content,
        )
        if expected != metadata:
            raise RawStoreCorruptionError("raw artifact bytes fail digest validation")
        return StoredRawArtifact(metadata=metadata, content=content)

    def list_artifacts(self, run_id: str) -> tuple[RawArtifactMetadata, ...]:
        """List metadata linked to one attempt without loading raw content."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT raw_artifacts.*
                FROM raw_artifacts
                JOIN attempt_artifacts USING (artifact_id)
                WHERE attempt_artifacts.run_id = ?
                ORDER BY raw_artifacts.artifact_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(_decode_artifact_row(row) for row in rows)

    def record_context_pack(
        self,
        run_id: str,
        *,
        sequence: int,
        pack: ContextPack,
    ) -> bool:
        """Persist a unique context snapshot, ordered by first-seen sequence.

        Identical snapshots can be used by later model invocations. Their
        per-call sequence, I/O, usage and cost remain in model_invocations.
        """

        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("context build sequence must be a non-negative integer")
        if not isinstance(pack, ContextPack):
            raise TypeError("pack must be a ContextPack")
        pack_json = _canonical_model_json(pack)
        raw_artifact_ids = {
            item.source.raw_artifact_id
            for item in pack.items
            if item.source.raw_artifact_id is not None
        }
        with self._transaction():
            self._require_attempt(run_id)
            for artifact_id in raw_artifact_ids:
                if not self._artifact_is_linked(run_id, artifact_id):
                    raise RawStoreIntegrityError(
                        "context pack references an artifact not linked to attempt"
                    )
            existing = self._connection.execute(
                """
                SELECT sequence, pack_json FROM context_builds
                WHERE run_id = ? AND (build_id = ? OR pack_id = ? OR sequence = ?)
                """,
                (
                    run_id,
                    pack.build_report.build_id,
                    pack.pack_id,
                    sequence,
                ),
            ).fetchall()
            occupied = self._connection.execute(
                "SELECT context_pack_id FROM model_invocations "
                "WHERE run_id = ? AND sequence = ?",
                (run_id, sequence),
            ).fetchone()
            if occupied is not None and occupied["context_pack_id"] != pack.pack_id:
                raise RawStoreConflictError(
                    "context sequence belongs to a different model invocation"
                )
            if existing:
                # Check every matching row: an identical pack must not mask a
                # different pack occupying the requested first-seen sequence.
                if all(row["pack_json"] == pack_json for row in existing):
                    return False
                raise RawStoreConflictError(
                    "context build identity or sequence already exists"
                )
            report = pack.build_report
            try:
                self._connection.execute(
                    """
                    INSERT INTO context_builds (
                        run_id, build_id, pack_id, sequence, budget_tokens,
                        selected_tokens, candidate_count, selected_count,
                        excluded_count, pack_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        report.build_id,
                        pack.pack_id,
                        sequence,
                        report.budget_tokens,
                        report.selected_tokens,
                        report.candidate_count,
                        report.selected_count,
                        report.excluded_count,
                        pack_json,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise RawStoreIntegrityError(
                    "context build violates Raw Run Store integrity"
                ) from error
        return True

    def get_context_pack(self, run_id: str, pack_id: str) -> ContextPack | None:
        """Return one strictly decoded active-context pack."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM context_builds WHERE run_id = ? AND pack_id = ?",
                (run_id, pack_id),
            ).fetchone()
        return None if row is None else _decode_context_row(row)

    def list_context_packs(self, run_id: str) -> tuple[ContextPack, ...]:
        """List unique context snapshots in first-seen invocation order."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM context_builds WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return tuple(_decode_context_row(row) for row in rows)

    def record_model_invocation(self, record: ModelInvocationRecord) -> bool:
        """Persist one invocation only after every provenance edge resolves."""

        if not isinstance(record, ModelInvocationRecord):
            raise TypeError("record must be a ModelInvocationRecord")
        record_json = _canonical_model_json(record)
        with self._transaction():
            attempt_row = self._connection.execute(
                "SELECT * FROM attempts WHERE run_id = ?",
                (record.run_id,),
            ).fetchone()
            if attempt_row is None:
                raise RawStoreIntegrityError("model invocation attempt does not exist")
            attempt = _decode_attempt_row(attempt_row)
            if (
                attempt.thread_id != record.thread_id
                or attempt.turn_number != record.turn_number
            ):
                raise RawStoreIntegrityError(
                    "model invocation does not match attempt identity"
                )
            input_metadata = self._linked_artifact_metadata(
                record.run_id,
                record.input_artifact_id,
            )
            if input_metadata is None:
                raise RawStoreIntegrityError(
                    "model invocation artifact provenance is incomplete"
                )
            if input_metadata.kind is not RawArtifactKind.MODEL_INPUT:
                raise RawStoreIntegrityError(
                    "model invocation input artifact must have kind MODEL_INPUT"
                )
            if record.output_artifact_id is not None:
                output_metadata = self._linked_artifact_metadata(
                    record.run_id,
                    record.output_artifact_id,
                )
                if output_metadata is None:
                    raise RawStoreIntegrityError(
                        "model invocation artifact provenance is incomplete"
                    )
                if output_metadata.kind is not RawArtifactKind.MODEL_OUTPUT:
                    raise RawStoreIntegrityError(
                        "model invocation output artifact must have kind MODEL_OUTPUT"
                    )
            context_exists = self._connection.execute(
                "SELECT 1 FROM context_builds WHERE run_id = ? AND pack_id = ?",
                (record.run_id, record.context_pack_id),
            ).fetchone()
            if context_exists is None:
                raise RawStoreIntegrityError(
                    "model invocation context provenance is incomplete"
                )
            existing = self._connection.execute(
                """
                SELECT record_json FROM model_invocations
                WHERE invocation_id = ? OR (run_id = ? AND sequence = ?)
                """,
                (record.invocation_id, record.run_id, record.sequence),
            ).fetchone()
            if existing is not None:
                if existing["record_json"] == record_json:
                    return False
                raise RawStoreConflictError(
                    "model invocation identity or sequence already exists"
                )
            usage = record.usage
            try:
                self._connection.execute(
                    """
                    INSERT INTO model_invocations (
                        invocation_id, run_id, thread_id, turn_number, sequence,
                        node_name, role, provider, model, status, latency_ms,
                        usage_availability, input_tokens, output_tokens,
                        total_tokens, cached_input_tokens, reasoning_output_tokens,
                        plan_step_id, context_pack_id, input_artifact_id,
                        output_artifact_id, error_code, record_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        record.invocation_id,
                        record.run_id,
                        record.thread_id,
                        record.turn_number,
                        record.sequence,
                        record.node_name,
                        record.role.value,
                        record.provider,
                        record.model,
                        record.status.value,
                        record.latency_ms,
                        usage.availability.value,
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.total_tokens,
                        usage.cached_input_tokens,
                        usage.reasoning_output_tokens,
                        record.plan_step_id,
                        record.context_pack_id,
                        record.input_artifact_id,
                        record.output_artifact_id,
                        record.error_code,
                        record_json,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise RawStoreIntegrityError(
                    "model invocation provenance violates storage integrity"
                ) from error
        return True

    def get_model_invocation(
        self,
        invocation_id: str,
    ) -> ModelInvocationRecord | None:
        """Return one strictly decoded model invocation record."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM model_invocations WHERE invocation_id = ?",
                (invocation_id,),
            ).fetchone()
        return None if row is None else _decode_invocation_row(row)

    def list_model_invocations(
        self,
        run_id: str,
    ) -> tuple[ModelInvocationRecord, ...]:
        """List one attempt's model calls in stable invocation order."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM model_invocations WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return tuple(_decode_invocation_row(row) for row in rows)

    def get_attempt_usage_cost_report(
        self,
        run_id: str,
        pricing_table: PricingTable,
        *,
        max_cost_usd: str | None = None,
    ) -> AttemptUsageCostReport | None:
        """Derive one immutable report from stored usage and explicit prices."""

        if not isinstance(pricing_table, PricingTable):
            raise TypeError("pricing_table must be a PricingTable")
        with self._lock:
            attempt_row = self._connection.execute(
                "SELECT * FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if attempt_row is None:
                return None
            invocation_rows = self._connection.execute(
                "SELECT * FROM model_invocations WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        attempt = _decode_attempt_row(attempt_row)
        return build_attempt_usage_cost_report(
            run_id=attempt.run_id,
            thread_id=attempt.thread_id,
            turn_number=attempt.turn_number,
            invocations=tuple(_decode_invocation_row(row) for row in invocation_rows),
            pricing_table=pricing_table,
            max_cost_usd=max_cost_usd,
        )

    def delete_attempt(self, run_id: str) -> bool:
        """Delete one attempt and collect only artifacts with no remaining links."""

        with self._transaction():
            exists = self._connection.execute(
                "SELECT 1 FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if exists is None:
                return False
            artifact_ids = tuple(
                row[0]
                for row in self._connection.execute(
                    "SELECT artifact_id FROM attempt_artifacts WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
            )
            self._connection.execute("DELETE FROM attempts WHERE run_id = ?", (run_id,))
            for artifact_id in artifact_ids:
                self._connection.execute(
                    """
                    DELETE FROM raw_artifacts
                    WHERE artifact_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM attempt_artifacts
                          WHERE attempt_artifacts.artifact_id = raw_artifacts.artifact_id
                      )
                    """,
                    (artifact_id,),
                )
        return True

    def _require_attempt(self, run_id: str) -> None:
        exists = self._connection.execute(
            "SELECT 1 FROM attempts WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if exists is None:
            raise RawStoreIntegrityError("attempt does not exist")

    def _linked_artifact_metadata(
        self,
        run_id: str,
        artifact_id: str,
    ) -> RawArtifactMetadata | None:
        row = self._connection.execute(
            """
            SELECT raw_artifacts.*
            FROM raw_artifacts
            JOIN attempt_artifacts USING (artifact_id)
            WHERE attempt_artifacts.run_id = ? AND raw_artifacts.artifact_id = ?
            """,
            (run_id, artifact_id),
        ).fetchone()
        if row is None:
            return None
        return _decode_artifact_row(row)

    def _artifact_is_linked(self, run_id: str, artifact_id: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM attempt_artifacts WHERE run_id = ? AND artifact_id = ?",
                (run_id, artifact_id),
            ).fetchone()
            is not None
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise


@contextmanager
def open_raw_run_store(database_path: Path) -> Iterator[RawRunStore]:
    """Open, migrate, validate, and close one trusted local Raw Run Store."""

    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a pathlib.Path")
    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(
            str(resolved_path),
            timeout=_SQLITE_TIMEOUT_SECONDS,
            check_same_thread=False,
        )
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        _migrate_and_validate(connection)
        yield RawRunStore(connection)


def _migrate_and_validate(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > RAW_STORE_SCHEMA_VERSION:
            raise RawStoreMigrationError(
                "Raw Run Store schema is newer than this application"
            )
        if version == 0:
            tables = _user_tables(connection)
            if tables:
                raise RawStoreMigrationError(
                    "unversioned nonempty database cannot be migrated safely"
                )
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {RAW_STORE_SCHEMA_VERSION}")
        elif version != RAW_STORE_SCHEMA_VERSION:
            raise RawStoreMigrationError(
                f"Unsupported Raw Run Store schema version: {version}"
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    _validate_schema(connection)


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = _user_tables(connection)
    required_tables = set(_REQUIRED_COLUMNS)
    if not required_tables.issubset(tables):
        raise RawStoreMigrationError("Raw Run Store schema is missing required tables")
    for table, expected_columns in _REQUIRED_COLUMNS.items():
        column_rows = tuple(connection.execute(f"PRAGMA table_info({table})"))
        actual_columns = tuple(str(row["name"]) for row in column_rows)
        if actual_columns != expected_columns:
            raise RawStoreMigrationError(
                f"Raw Run Store schema for {table} is incompatible"
            )
        required_not_null = _REQUIRED_NOT_NULL_COLUMNS.get(table, frozenset())
        not_null_columns = {
            str(row["name"]) for row in column_rows if int(row["notnull"]) == 1
        }
        if not required_not_null.issubset(not_null_columns):
            raise RawStoreMigrationError(
                f"Raw Run Store nullability for {table} is incompatible"
            )
    foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys != 1:
        raise RawStoreMigrationError("Raw Run Store foreign keys are disabled")
    for table, expected_foreign_keys in _EXPECTED_FOREIGN_KEYS.items():
        actual_foreign_keys = Counter(
            (
                str(row["table"]),
                str(row["from"]),
                str(row["to"]),
                str(row["on_delete"]).upper(),
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        if actual_foreign_keys != expected_foreign_keys:
            raise RawStoreMigrationError(
                f"Raw Run Store foreign key schema for {table} is incompatible"
            )


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _canonical_model_json(model: CheckpointSafeModel) -> str:
    return json.dumps(
        model.model_dump(mode="json", exclude_computed_fields=True),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_attempt_row(row: sqlite3.Row) -> RawRunAttempt:
    try:
        attempt = RawRunAttempt.model_validate_json(row["record_json"])
    except (ValidationError, ValueError, TypeError) as error:
        raise RawStoreCorruptionError("stored attempt record is invalid") from error
    mirrored = (
        attempt.run_id,
        attempt.thread_id,
        attempt.turn_number,
        attempt.execution_mode.value,
        attempt.repository_id,
        attempt.commit_sha,
        attempt.objective_digest,
        attempt.status.value,
        attempt.started_at_ms,
        attempt.finished_at_ms,
    )
    stored = tuple(
        row[name]
        for name in (
            "run_id",
            "thread_id",
            "turn_number",
            "execution_mode",
            "repository_id",
            "commit_sha",
            "objective_digest",
            "status",
            "started_at_ms",
            "finished_at_ms",
        )
    )
    if mirrored != stored:
        raise RawStoreCorruptionError("stored attempt columns disagree with record")
    return attempt


def _decode_artifact_row(row: sqlite3.Row) -> RawArtifactMetadata:
    try:
        metadata = RawArtifactMetadata.model_validate_json(row["metadata_json"])
    except (ValidationError, ValueError, TypeError) as error:
        raise RawStoreCorruptionError(
            "stored raw artifact metadata is invalid"
        ) from error
    mirrored = (
        metadata.artifact_id,
        metadata.kind.value,
        metadata.media_type,
        metadata.content_digest,
        metadata.byte_count,
    )
    stored = tuple(
        row[name]
        for name in (
            "artifact_id",
            "kind",
            "media_type",
            "content_digest",
            "byte_count",
        )
    )
    if mirrored != stored:
        raise RawStoreCorruptionError(
            "stored raw artifact columns disagree with metadata"
        )
    return metadata


def _decode_context_row(row: sqlite3.Row) -> ContextPack:
    try:
        pack = ContextPack.model_validate_json(row["pack_json"])
    except (ValidationError, ValueError, TypeError) as error:
        raise RawStoreCorruptionError("stored context pack is invalid") from error
    report = pack.build_report
    mirrored = (
        report.build_id,
        pack.pack_id,
        report.budget_tokens,
        report.selected_tokens,
        report.candidate_count,
        report.selected_count,
        report.excluded_count,
    )
    stored = tuple(
        row[name]
        for name in (
            "build_id",
            "pack_id",
            "budget_tokens",
            "selected_tokens",
            "candidate_count",
            "selected_count",
            "excluded_count",
        )
    )
    if mirrored != stored:
        raise RawStoreCorruptionError(
            "stored context columns disagree with context pack"
        )
    return pack


def _decode_invocation_row(row: sqlite3.Row) -> ModelInvocationRecord:
    try:
        record = ModelInvocationRecord.model_validate_json(row["record_json"])
    except (ValidationError, ValueError, TypeError) as error:
        raise RawStoreCorruptionError(
            "stored model invocation record is invalid"
        ) from error
    usage = record.usage
    mirrored = (
        record.invocation_id,
        record.run_id,
        record.thread_id,
        record.turn_number,
        record.sequence,
        record.node_name,
        record.role.value,
        record.provider,
        record.model,
        record.status.value,
        record.latency_ms,
        usage.availability.value,
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.cached_input_tokens,
        usage.reasoning_output_tokens,
        record.plan_step_id,
        record.context_pack_id,
        record.input_artifact_id,
        record.output_artifact_id,
        record.error_code,
    )
    stored = tuple(
        row[name]
        for name in (
            "invocation_id",
            "run_id",
            "thread_id",
            "turn_number",
            "sequence",
            "node_name",
            "role",
            "provider",
            "model",
            "status",
            "latency_ms",
            "usage_availability",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_output_tokens",
            "plan_step_id",
            "context_pack_id",
            "input_artifact_id",
            "output_artifact_id",
            "error_code",
        )
    )
    if mirrored != stored:
        raise RawStoreCorruptionError("stored invocation columns disagree with record")
    return record


def _encode_artifact_content(content: str | bytes) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        try:
            return content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise RawStoreIntegrityError(
                "raw artifact text must be valid UTF-8"
            ) from error
    raise TypeError("raw artifact content must be str or bytes")
