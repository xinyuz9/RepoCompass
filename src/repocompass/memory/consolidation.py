"""Deterministic terminal-attempt consolidation into observed local memory."""

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.lifecycle import ExecutionStatus
from repocompass.agent.result import FinalResult, revalidate_final_result
from repocompass.agent.runner import RunOutcome
from repocompass.memory.contracts import (
    MAX_MEMORY_CONTENT_CHARACTERS,
    MAX_MEMORY_ITEM_CHARACTERS,
    MAX_MEMORY_SEQUENCE_ITEMS,
    EpisodicMemory,
    create_episodic_memory,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import (
    MemoryStore,
    MemoryStoreConflictError,
)
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    ToolAuditRecord,
)
from repocompass.tools.gateway import ToolExecutionContext

EPISODE_CONSOLIDATION_SCHEMA_VERSION = 1
MAX_CONSOLIDATION_OBJECTIVE_CHARACTERS = 65_536
MAX_CONSOLIDATION_FINAL_RESULT_BYTES = 1_000_000
MAX_CONSOLIDATION_TOOL_AUDITS = 1_024
MAX_CONSOLIDATION_PATH_CHARACTERS = 4_096

_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"\b(api[_-]?key|access[_-]?token|authorization|password|refresh[_-]?token|secret)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bbearer\s+[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)


class EpisodeConsolidationDisposition(StrEnum):
    """Observable terminal-consolidation outcome."""

    CREATED = "created"
    ALREADY_CONSOLIDATED = "already_consolidated"
    SKIPPED = "skipped"


class EpisodeConsolidationReason(StrEnum):
    """Closed reason for a terminal-consolidation outcome."""

    TERMINAL_OUTCOME = "terminal_outcome"
    LOGICAL_TURN_EXISTS = "logical_turn_exists"
    ATTEMPT_INTERRUPTED = "attempt_interrupted"


class EpisodeConsolidationError(RuntimeError):
    """Trusted lifecycle input is unsafe or internally inconsistent."""


class EpisodeConsolidationResult(CheckpointSafeModel):
    """Strict result that never represents an interrupted attempt as stored."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1] = EPISODE_CONSOLIDATION_SCHEMA_VERSION
    disposition: EpisodeConsolidationDisposition
    reason: EpisodeConsolidationReason
    episode: EpisodicMemory | None = None

    @model_validator(mode="after")
    def require_canonical_outcome(self) -> "EpisodeConsolidationResult":
        allowed = {
            EpisodeConsolidationDisposition.CREATED: (
                EpisodeConsolidationReason.TERMINAL_OUTCOME
            ),
            EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED: (
                EpisodeConsolidationReason.LOGICAL_TURN_EXISTS
            ),
            EpisodeConsolidationDisposition.SKIPPED: (
                EpisodeConsolidationReason.ATTEMPT_INTERRUPTED
            ),
        }
        if self.reason is not allowed[self.disposition]:
            raise ValueError("episode consolidation disposition and reason disagree")
        if (self.episode is None) is not (
            self.disposition is EpisodeConsolidationDisposition.SKIPPED
        ):
            raise ValueError("episode consolidation payload is inconsistent")
        return self


def consolidate_terminal_episode(
    store: MemoryStore,
    *,
    context: ToolExecutionContext,
    outcome: RunOutcome,
    objective: str,
    turn_number: int,
    created_at_ms: int,
) -> EpisodeConsolidationResult:
    """Consolidate one completed attempt without exposing a model write tool."""

    _validate_consolidation_boundary(
        store=store,
        context=context,
        outcome=outcome,
        objective=objective,
        turn_number=turn_number,
        created_at_ms=created_at_ms,
    )
    normalized_objective = objective.strip()
    if outcome.execution_status is ExecutionStatus.INTERRUPTED:
        return EpisodeConsolidationResult(
            disposition=EpisodeConsolidationDisposition.SKIPPED,
            reason=EpisodeConsolidationReason.ATTEMPT_INTERRUPTED,
        )

    final_result = outcome.final_result
    termination_status = outcome.termination_status
    if final_result is None or termination_status is None:
        raise EpisodeConsolidationError(
            "completed outcome is missing terminal result fields"
        )
    final_result = revalidate_final_result(final_result)
    _validate_identifier(outcome.trace.run_id, name="run_id")
    _validate_identifier(outcome.thread_id, name="thread_id")
    _validate_tool_audits(outcome.tool_audit_records, context, outcome.trace.run_id)

    identity = resolve_memory_repository_identity(context)
    objective_digest = _namespaced_digest(
        b"repocompass:episode-objective:v1",
        normalized_objective.encode("utf-8"),
    )
    final_result_digest = _final_result_digest(final_result)
    existing = store.get_episodic_memory_for_turn(
        identity.repository_key,
        outcome.thread_id,
        turn_number,
    )
    if existing is not None:
        _require_matching_logical_turn(
            existing,
            context=context,
            termination_status=termination_status,
            objective_digest=objective_digest,
            final_result_digest=final_result_digest,
        )
        return EpisodeConsolidationResult(
            disposition=EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED,
            reason=EpisodeConsolidationReason.LOGICAL_TURN_EXISTS,
            episode=existing,
        )

    episode = create_episodic_memory(
        identity=identity,
        commit_sha=context.commit_sha,
        run_id=outcome.trace.run_id,
        thread_id=outcome.thread_id,
        turn_number=turn_number,
        termination_status=termination_status,
        objective_digest=objective_digest,
        summary=_sanitize_text(
            final_result.summary,
            context=context,
            max_characters=MAX_MEMORY_CONTENT_CHARACTERS,
            fallback="Investigation completed without a public summary.",
        ),
        query_sequence=_query_sequence(outcome.tool_audit_records),
        successful_paths=_successful_paths(final_result),
        failed_hypotheses=_failed_hypotheses(final_result, context=context),
        final_result_digest=final_result_digest,
        created_at_ms=created_at_ms,
    )
    try:
        inserted = store.put_episodic_memory(episode, context=context)
    except MemoryStoreConflictError:
        concurrent = store.get_episodic_memory_for_turn(
            identity.repository_key,
            outcome.thread_id,
            turn_number,
        )
        if concurrent is None:
            raise
        _require_matching_logical_turn(
            concurrent,
            context=context,
            termination_status=termination_status,
            objective_digest=objective_digest,
            final_result_digest=final_result_digest,
        )
        return EpisodeConsolidationResult(
            disposition=EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED,
            reason=EpisodeConsolidationReason.LOGICAL_TURN_EXISTS,
            episode=concurrent,
        )
    return EpisodeConsolidationResult(
        disposition=(
            EpisodeConsolidationDisposition.CREATED
            if inserted
            else EpisodeConsolidationDisposition.ALREADY_CONSOLIDATED
        ),
        reason=(
            EpisodeConsolidationReason.TERMINAL_OUTCOME
            if inserted
            else EpisodeConsolidationReason.LOGICAL_TURN_EXISTS
        ),
        episode=episode,
    )


def _validate_consolidation_boundary(
    *,
    store: object,
    context: object,
    outcome: object,
    objective: object,
    turn_number: object,
    created_at_ms: object,
) -> None:
    if not isinstance(store, MemoryStore):
        raise TypeError("store must be a MemoryStore")
    if not isinstance(context, ToolExecutionContext):
        raise TypeError("context must be a ToolExecutionContext")
    if not isinstance(outcome, RunOutcome):
        raise TypeError("outcome must be a RunOutcome")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be non-empty text")
    if len(objective) > MAX_CONSOLIDATION_OBJECTIVE_CHARACTERS:
        raise ValueError("objective exceeds the consolidation limit")
    if type(turn_number) is not int or turn_number < 1:
        raise ValueError("turn_number must be a positive integer")
    if (
        type(created_at_ms) is not int
        or created_at_ms < 0
        or created_at_ms > MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError("created_at_ms is outside the safe timestamp range")


def _validate_identifier(value: str, *, name: str) -> None:
    if re.fullmatch(IDENTIFIER_PATTERN, value) is None:
        raise EpisodeConsolidationError(f"{name} is not a canonical identifier")


def _validate_tool_audits(
    audits: tuple[ToolAuditRecord, ...],
    context: ToolExecutionContext,
    run_id: str,
) -> None:
    if len(audits) > MAX_CONSOLIDATION_TOOL_AUDITS:
        raise EpisodeConsolidationError("tool audit count exceeds the safe limit")
    sequences: set[int] = set()
    for audit in audits:
        if not isinstance(audit, ToolAuditRecord):
            raise EpisodeConsolidationError("tool audit record is invalid")
        if (
            audit.run_id != run_id
            or audit.repository_id != context.repository_id
            or audit.commit_sha != context.commit_sha
            or audit.sequence in sequences
        ):
            raise EpisodeConsolidationError(
                "tool audit does not match the terminal attempt snapshot"
            )
        sequences.add(audit.sequence)


def _query_sequence(
    audits: tuple[ToolAuditRecord, ...],
) -> tuple[str, ...]:
    ordered = sorted(audits, key=lambda item: (item.sequence, item.call_id))
    return tuple(
        f"{audit.tool_name}:{audit.decision.outcome.value}:"
        f"{audit.fingerprint.digest[:16]}"
        for audit in ordered[:MAX_MEMORY_SEQUENCE_ITEMS]
    )


def _successful_paths(final_result: FinalResult) -> tuple[str, ...]:
    candidates = list(final_result.relevant_files)
    if final_result.structure is not None:
        candidates.extend(final_result.structure.indexed_paths)
    paths: set[str] = set()
    for value in candidates:
        if not _is_safe_repository_path(value):
            raise EpisodeConsolidationError(
                "terminal result contains an unsafe repository path"
            )
        paths.add(value)
    return tuple(sorted(paths))[:MAX_MEMORY_SEQUENCE_ITEMS]


def _failed_hypotheses(
    final_result: FinalResult,
    *,
    context: ToolExecutionContext,
) -> tuple[str, ...]:
    return tuple(
        _sanitize_text(
            value,
            context=context,
            max_characters=MAX_MEMORY_ITEM_CHARACTERS,
            fallback="Unresolved investigation question.",
        )
        for value in final_result.unresolved_questions[:MAX_MEMORY_SEQUENCE_ITEMS]
    )


def _is_safe_repository_path(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_CONSOLIDATION_PATH_CHARACTERS
        or "\\" in value
        or "\x00" in value
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and value != "."
        and ".." not in path.parts
        and path.as_posix() == value
    )


def _sanitize_text(
    value: str,
    *,
    context: ToolExecutionContext,
    max_characters: int,
    fallback: str,
) -> str:
    repository_text = str(context.repository_root)
    sanitized = value.replace(repository_text, "<repository>")
    sanitized = _BEARER_PATTERN.sub("Bearer [REDACTED]", sanitized)
    sanitized = _SENSITIVE_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        sanitized,
    )
    sanitized = "".join(
        character
        if not unicodedata.category(character).startswith("C")
        or character in {"\n", "\t"}
        else " "
        for character in sanitized.replace("\r", "\n")
    ).strip()
    if not sanitized:
        sanitized = fallback
    return sanitized[:max_characters]


def _final_result_digest(final_result: FinalResult) -> str:
    payload = final_result.model_dump(
        mode="json",
        exclude_computed_fields=True,
        warnings=False,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_CONSOLIDATION_FINAL_RESULT_BYTES:
        raise EpisodeConsolidationError(
            "terminal result exceeds the consolidation size limit"
        )
    return _namespaced_digest(b"repocompass:episode-final-result:v1", encoded)


def _namespaced_digest(namespace: bytes, value: bytes) -> str:
    return hashlib.sha256(namespace + b"\0" + value).hexdigest()


def _require_matching_logical_turn(
    existing: EpisodicMemory,
    *,
    context: ToolExecutionContext,
    termination_status: str,
    objective_digest: str,
    final_result_digest: str,
) -> None:
    if (
        existing.source_repository_id != context.repository_id
        or existing.commit_sha != context.commit_sha
        or existing.termination_status != termination_status
        or existing.objective_digest != objective_digest
        or existing.final_result_digest != final_result_digest
    ):
        raise MemoryStoreConflictError(
            "existing thread turn terminal result does not match retry"
        )
