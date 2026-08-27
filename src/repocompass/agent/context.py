"""Traceable raw-artifact and active-context domain contracts.

The contracts in this module deliberately separate local raw content from the
bounded snippets selected for model context. They contain no storage or
selection algorithm; M3.3 storage and M3.4 context building consume these
stable, versioned records.
"""

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

CONTEXT_SCHEMA_VERSION = 1
MAX_CONTEXT_ITEM_BYTES = 262_144
MAX_CONTEXT_ITEMS = 1_024
MEDIA_TYPE_PATTERN = (
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/"
    r"[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$"
)


class RawArtifactKind(StrEnum):
    """Closed vocabulary for locally persisted raw payloads."""

    TOOL_OUTPUT = "tool_output"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    SOURCE_EXCERPT = "source_excerpt"
    HUMAN_INPUT = "human_input"
    CONTEXT_PACK = "context_pack"


class ContextSourceKind(StrEnum):
    """Origin category for one bounded active-context item."""

    OBJECTIVE = "objective"
    PLAN_STEP = "plan_step"
    TOOL_RESULT = "tool_result"
    SOURCE_EXCERPT = "source_excerpt"
    EVALUATOR_REQUEST = "evaluator_request"
    HUMAN_FEEDBACK = "human_feedback"
    CANDIDATE = "candidate"
    MEMORY = "memory"
    SUMMARY = "summary"


class ProvenanceKind(StrEnum):
    """Joinable upstream identity types used to reconstruct a reading path."""

    OBJECTIVE = "objective"
    PLAN_STEP = "plan_step"
    TOOL_CALL = "tool_call"
    RAW_ARTIFACT = "raw_artifact"
    EVALUATION = "evaluation"
    HUMAN_INPUT = "human_input"
    MEMORY = "memory"


class VerificationStatus(StrEnum):
    """Evidence confidence without treating model text as verified fact."""

    VERIFIED = "verified"
    HUMAN_APPROVED = "human_approved"
    INFERRED = "inferred"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class ContextSection(StrEnum):
    """Stable active-context partition names used by future budget policy."""

    OBJECTIVE = "objective"
    PLAN = "plan"
    FEEDBACK = "feedback"
    EVIDENCE = "evidence"
    MEMORY = "memory"
    HISTORY = "history"


class ContextMemoryScope(StrEnum):
    """Whether recalled context is a repository fact or observed episode."""

    REPOSITORY = "repository"
    EPISODIC = "episodic"


class ContextMemoryFreshness(StrEnum):
    """Trust posture attached to a memory candidate in active context."""

    CURRENT = "current"
    REVALIDATION_REQUIRED = "revalidation_required"
    STALE = "stale"
    OBSERVED = "observed"


class ContextMemoryStalenessReason(StrEnum):
    """Closed, dependency-neutral mirror of memory freshness reasons."""

    COMMIT_MISMATCH = "commit_mismatch"
    SOURCE_REPOSITORY_MISMATCH = "source_repository_mismatch"
    DIRTY_WORKTREE = "dirty_worktree"
    UNVERSIONED_SNAPSHOT = "unversioned_snapshot"
    SOURCE_CHANGED = "source_changed"
    SOURCE_MISSING = "source_missing"
    CORRUPT_RECORD = "corrupt_record"


class ContextSelectionOutcome(StrEnum):
    """Whether one candidate item entered the active context pack."""

    SELECTED = "selected"
    EXCLUDED = "excluded"


class ContextSelectionReason(StrEnum):
    """Closed, UI-safe explanation for one context selection decision."""

    REQUIRED_OBJECTIVE = "required_objective"
    PINNED = "pinned"
    EVALUATOR_REQUEST = "evaluator_request"
    VERIFIED_EVIDENCE = "verified_evidence"
    RELEVANCE = "relevance"
    RECENCY = "recency"
    COMPRESSED_FIT = "compressed_fit"
    DUPLICATE = "duplicate"
    BUDGET_EXCEEDED = "budget_exceeded"
    LOWER_PRIORITY = "lower_priority"
    INVALID_SOURCE = "invalid_source"


_SELECTED_REASONS = frozenset(
    {
        ContextSelectionReason.REQUIRED_OBJECTIVE,
        ContextSelectionReason.PINNED,
        ContextSelectionReason.EVALUATOR_REQUEST,
        ContextSelectionReason.VERIFIED_EVIDENCE,
        ContextSelectionReason.RELEVANCE,
        ContextSelectionReason.RECENCY,
        ContextSelectionReason.COMPRESSED_FIT,
    }
)
_EXCLUDED_REASONS = frozenset(
    {
        ContextSelectionReason.DUPLICATE,
        ContextSelectionReason.BUDGET_EXCEEDED,
        ContextSelectionReason.LOWER_PRIORITY,
        ContextSelectionReason.INVALID_SOURCE,
    }
)


class _ContextContractModel(CheckpointSafeModel):
    """Strict immutable base for context and provenance records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class RawArtifactMetadata(_ContextContractModel):
    """Content-addressed metadata that never embeds the raw payload itself."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    artifact_id: Sha256Digest
    kind: RawArtifactKind
    media_type: str = Field(pattern=MEDIA_TYPE_PATTERN)
    content_digest: Sha256Digest
    byte_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def require_content_addressed_identity(self) -> "RawArtifactMetadata":
        expected = _raw_artifact_id(
            kind=self.kind,
            media_type=self.media_type,
            content_digest=self.content_digest,
            byte_count=self.byte_count,
        )
        if self.artifact_id != expected:
            raise ValueError("artifact_id does not match raw artifact metadata")
        return self


class ContextMemoryMetadata(_ContextContractModel):
    """Content-safe identity and freshness metadata for one recalled memory."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    memory_id: Sha256Digest
    retrieval_id: Sha256Digest
    query_digest: Sha256Digest
    repository_key: Sha256Digest
    source_repository_id: Sha256Digest
    source_commit_sha: CommitSha | None = None
    scope: ContextMemoryScope
    freshness: ContextMemoryFreshness
    staleness_reason: ContextMemoryStalenessReason | None = None

    @model_validator(mode="after")
    def require_explicit_memory_trust(self) -> "ContextMemoryMetadata":
        if self.scope is ContextMemoryScope.EPISODIC:
            if (
                self.freshness is not ContextMemoryFreshness.OBSERVED
                or self.staleness_reason is not None
            ):
                raise ValueError("episodic context must remain observed history")
            return self
        if self.source_commit_sha is None:
            raise ValueError("repository memory requires a source commit")
        allowed_reasons = {
            ContextMemoryFreshness.CURRENT: {None},
            ContextMemoryFreshness.REVALIDATION_REQUIRED: {
                ContextMemoryStalenessReason.COMMIT_MISMATCH,
                ContextMemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH,
                ContextMemoryStalenessReason.DIRTY_WORKTREE,
                ContextMemoryStalenessReason.UNVERSIONED_SNAPSHOT,
            },
            ContextMemoryFreshness.STALE: {
                ContextMemoryStalenessReason.SOURCE_CHANGED,
                ContextMemoryStalenessReason.SOURCE_MISSING,
                ContextMemoryStalenessReason.CORRUPT_RECORD,
            },
        }
        if self.freshness is ContextMemoryFreshness.OBSERVED:
            raise ValueError("repository memory cannot be observed episode history")
        if self.staleness_reason not in allowed_reasons[self.freshness]:
            raise ValueError("memory freshness and staleness reason disagree")
        return self


class MemoryRecallDecision(_ContextContractModel):
    """One body-free retrieval decision retained in checkpoints and traces."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    retrieval_id: Sha256Digest
    memory_id: Sha256Digest
    scope: ContextMemoryScope
    freshness: ContextMemoryFreshness
    staleness_reason: ContextMemoryStalenessReason | None = None
    context_item_id: Sha256Digest | None = None

    @model_validator(mode="after")
    def require_compatible_selection(self) -> "MemoryRecallDecision":
        if self.scope is ContextMemoryScope.EPISODIC:
            if (
                self.freshness is not ContextMemoryFreshness.OBSERVED
                or self.staleness_reason is not None
                or self.context_item_id is None
            ):
                raise ValueError("episodic recall decision is inconsistent")
        elif self.freshness is ContextMemoryFreshness.OBSERVED:
            raise ValueError("repository recall cannot be observed")
        if self.freshness is ContextMemoryFreshness.STALE:
            if self.staleness_reason is None or self.context_item_id is not None:
                raise ValueError("stale memory must be excluded from context")
        elif self.context_item_id is None:
            raise ValueError("usable memory recall requires a context item")
        return self


class MemoryRecallReport(_ContextContractModel):
    """Body-free record proving one bounded recall ran for this turn."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    recall_id: Sha256Digest
    query_digest: Sha256Digest
    repository_result_count: int = Field(ge=0, le=50)
    episodic_result_count: int = Field(ge=0, le=50)
    decisions: tuple[MemoryRecallDecision, ...] = Field(default=(), max_length=100)
    context_item_ids: tuple[Sha256Digest, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def require_reconciled_content_addressed_report(self) -> "MemoryRecallReport":
        retrieval_ids = tuple(item.retrieval_id for item in self.decisions)
        if len(retrieval_ids) != len(set(retrieval_ids)):
            raise ValueError("memory recall retrieval IDs must be unique")
        repository_count = sum(
            item.scope is ContextMemoryScope.REPOSITORY for item in self.decisions
        )
        episodic_count = sum(
            item.scope is ContextMemoryScope.EPISODIC for item in self.decisions
        )
        if (
            repository_count != self.repository_result_count
            or episodic_count != self.episodic_result_count
        ):
            raise ValueError("memory recall result counts do not reconcile")
        expected_item_ids = tuple(
            item.context_item_id
            for item in self.decisions
            if item.context_item_id is not None
        )
        if self.context_item_ids != expected_item_ids:
            raise ValueError("memory recall context item IDs do not reconcile")
        expected_id = _memory_recall_id(
            query_digest=self.query_digest,
            decisions=self.decisions,
        )
        if self.recall_id != expected_id:
            raise ValueError("recall_id does not match memory decisions")
        return self


class RepositoryLocation(_ContextContractModel):
    """Normalized source location inside one immutable repository snapshot."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    path: str = Field(min_length=1, max_length=4_096)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    symbol: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("path")
    @classmethod
    def require_normalized_relative_path(cls, value: str) -> str:
        if "\\" in value or "\x00" in value:
            raise ValueError("path must be a normalized POSIX repository path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value == "."
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("path must be a normalized relative repository path")
        return value

    @model_validator(mode="after")
    def require_valid_line_range(self) -> "RepositoryLocation":
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must not precede line_start")
        return self


class ProvenanceLink(_ContextContractModel):
    """Typed edge to an upstream objective, step, call, artifact, or memory."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    kind: ProvenanceKind
    id: str = Field(pattern=IDENTIFIER_PATTERN)


class ContextSource(_ContextContractModel):
    """Canonical provenance and optional source location for a context item."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    source_id: Sha256Digest
    kind: ContextSourceKind
    raw_artifact_id: Sha256Digest | None = None
    location: RepositoryLocation | None = None
    provenance: tuple[ProvenanceLink, ...] = Field(min_length=1, max_length=64)
    memory: ContextMemoryMetadata | None = None

    @model_validator(mode="after")
    def require_canonical_traceable_source(self) -> "ContextSource":
        canonical = _canonical_provenance(self.provenance)
        if self.provenance != canonical:
            raise ValueError("provenance must be unique and in canonical order")
        if (
            self.kind
            in {ContextSourceKind.TOOL_RESULT, ContextSourceKind.SOURCE_EXCERPT}
            and self.raw_artifact_id is None
        ):
            raise ValueError(f"{self.kind.value} requires raw_artifact_id")
        if self.kind is ContextSourceKind.SOURCE_EXCERPT and self.location is None:
            raise ValueError("source_excerpt requires a repository location")
        if self.kind is ContextSourceKind.MEMORY:
            if self.memory is None or self.raw_artifact_id is not None:
                raise ValueError("memory source requires metadata and no raw artifact")
        elif self.memory is not None:
            raise ValueError("memory metadata requires a memory source")
        expected = _context_source_id(
            kind=self.kind,
            raw_artifact_id=self.raw_artifact_id,
            location=self.location,
            provenance=self.provenance,
            memory=self.memory,
        )
        if self.source_id != expected:
            raise ValueError("source_id does not match canonical source metadata")
        return self


class ContextItem(_ContextContractModel):
    """One bounded active-context candidate linked to immutable provenance."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    item_id: Sha256Digest
    source: ContextSource
    section: ContextSection
    verification_status: VerificationStatus
    content: str = Field(min_length=1)
    content_digest: Sha256Digest
    estimated_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    priority: int = Field(ge=0, le=100)
    pinned: bool = False

    @field_validator("content")
    @classmethod
    def require_bounded_utf8_content(cls, value: str) -> str:
        encoded = _encode_utf8(value, name="content")
        if len(encoded) > MAX_CONTEXT_ITEM_BYTES:
            raise ValueError("content exceeds the context-item byte limit")
        return value

    @model_validator(mode="after")
    def require_content_identity_and_verification_boundary(self) -> "ContextItem":
        expected_digest = hashlib.sha256(
            _encode_utf8(self.content, name="content")
        ).hexdigest()
        if self.content_digest != expected_digest:
            raise ValueError("content_digest does not match content")
        expected_id = _context_item_id(
            source_id=self.source.source_id,
            section=self.section,
            content_digest=self.content_digest,
        )
        if self.item_id != expected_id:
            raise ValueError("item_id does not match canonical context identity")
        if self.verification_status is VerificationStatus.VERIFIED and (
            self.source.kind is not ContextSourceKind.SOURCE_EXCERPT
        ):
            raise ValueError(
                "verified context requires source_excerpt repository provenance"
            )
        return self


class ContextSelectionDecision(_ContextContractModel):
    """One auditable inclusion or exclusion decision for a candidate item."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    item_id: Sha256Digest
    outcome: ContextSelectionOutcome
    reason: ContextSelectionReason
    estimated_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    selected_position: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_consistent_outcome_reason_and_position(
        self,
    ) -> "ContextSelectionDecision":
        if self.outcome is ContextSelectionOutcome.SELECTED:
            if self.reason not in _SELECTED_REASONS:
                raise ValueError("selected outcome has an exclusion reason")
            if self.selected_position is None:
                raise ValueError("selected outcome requires selected_position")
        else:
            if self.reason not in _EXCLUDED_REASONS:
                raise ValueError("excluded outcome has a selection reason")
            if self.selected_position is not None:
                raise ValueError("excluded outcome must not have selected_position")
        return self


class ContextBuildReport(_ContextContractModel):
    """Complete decision ledger for one hard-budget context build."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    build_id: Sha256Digest
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    candidate_count: int = Field(ge=1, le=MAX_CONTEXT_ITEMS)
    selected_count: int = Field(ge=0, le=MAX_CONTEXT_ITEMS)
    excluded_count: int = Field(ge=0, le=MAX_CONTEXT_ITEMS)
    selected_tokens: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    decisions: tuple[ContextSelectionDecision, ...] = Field(
        min_length=1,
        max_length=MAX_CONTEXT_ITEMS,
    )

    @model_validator(mode="after")
    def reconcile_decisions_and_budget(self) -> "ContextBuildReport":
        item_ids = [decision.item_id for decision in self.decisions]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("context decision item IDs must be unique")
        selected = [
            decision
            for decision in self.decisions
            if decision.outcome is ContextSelectionOutcome.SELECTED
        ]
        positions = sorted(
            decision.selected_position
            for decision in selected
            if decision.selected_position is not None
        )
        if positions != list(range(len(selected))):
            raise ValueError("selected positions must be contiguous and canonical")
        expected_selected_tokens = sum(
            decision.estimated_tokens for decision in selected
        )
        expected_counts = (
            len(self.decisions),
            len(selected),
            len(self.decisions) - len(selected),
        )
        actual_counts = (
            self.candidate_count,
            self.selected_count,
            self.excluded_count,
        )
        if actual_counts != expected_counts:
            raise ValueError("candidate_count and selection counts do not reconcile")
        if self.selected_tokens != expected_selected_tokens:
            raise ValueError("selected_tokens do not reconcile with decisions")
        if self.selected_tokens > self.budget_tokens:
            raise ValueError("selected context exceeds the hard token budget")
        expected_id = _context_build_id(self.budget_tokens, self.decisions)
        if self.build_id != expected_id:
            raise ValueError("build_id does not match context decisions")
        return self


class ContextPack(_ContextContractModel):
    """Ordered, budget-valid active context plus its complete build report."""

    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    pack_id: Sha256Digest
    build_report: ContextBuildReport
    estimated_tokens: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    items: tuple[ContextItem, ...] = Field(max_length=MAX_CONTEXT_ITEMS)

    @model_validator(mode="after")
    def reconcile_items_with_build_report(self) -> "ContextPack":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("context pack item IDs must be unique")
        selected = sorted(
            (
                decision
                for decision in self.build_report.decisions
                if decision.outcome is ContextSelectionOutcome.SELECTED
            ),
            key=lambda decision: (
                decision.selected_position
                if decision.selected_position is not None
                else -1
            ),
        )
        expected_ids = [decision.item_id for decision in selected]
        if item_ids != expected_ids:
            raise ValueError("context items do not match the selected order")
        for item, decision in zip(self.items, selected, strict=True):
            if item.estimated_tokens != decision.estimated_tokens:
                raise ValueError("context item token estimate differs from decision")
        expected_tokens = sum(item.estimated_tokens for item in self.items)
        if self.estimated_tokens != expected_tokens:
            raise ValueError("estimated_tokens do not reconcile with context items")
        if self.estimated_tokens > self.build_report.budget_tokens:
            raise ValueError("context pack exceeds the hard token budget")
        expected_id = _context_pack_id(
            self.build_report.build_id,
            tuple(item_ids),
        )
        if self.pack_id != expected_id:
            raise ValueError("pack_id does not match ordered context items")
        return self


def create_raw_artifact_metadata(
    *,
    kind: RawArtifactKind,
    media_type: str,
    content: str | bytes,
) -> RawArtifactMetadata:
    """Build content-addressed metadata without retaining the raw payload."""

    encoded = _encode_raw_content(content)
    content_digest = hashlib.sha256(encoded).hexdigest()
    byte_count = len(encoded)
    return RawArtifactMetadata(
        artifact_id=_raw_artifact_id(
            kind=kind,
            media_type=media_type,
            content_digest=content_digest,
            byte_count=byte_count,
        ),
        kind=kind,
        media_type=media_type,
        content_digest=content_digest,
        byte_count=byte_count,
    )


def create_context_source(
    *,
    kind: ContextSourceKind,
    provenance: tuple[ProvenanceLink, ...],
    raw_artifact_id: str | None = None,
    location: RepositoryLocation | None = None,
    memory: ContextMemoryMetadata | None = None,
) -> ContextSource:
    """Create a source with canonical provenance order and deterministic ID."""

    canonical = _canonical_provenance(provenance)
    source_id = _context_source_id(
        kind=kind,
        raw_artifact_id=raw_artifact_id,
        location=location,
        provenance=canonical,
        memory=memory,
    )
    return ContextSource(
        source_id=source_id,
        kind=kind,
        raw_artifact_id=raw_artifact_id,
        location=location,
        provenance=canonical,
        memory=memory,
    )


def create_memory_recall_report(
    *,
    query_digest: str,
    decisions: tuple[MemoryRecallDecision, ...],
) -> MemoryRecallReport:
    """Create one content-addressed, body-free recall report."""

    return MemoryRecallReport(
        recall_id=_memory_recall_id(
            query_digest=query_digest,
            decisions=decisions,
        ),
        query_digest=query_digest,
        repository_result_count=sum(
            item.scope is ContextMemoryScope.REPOSITORY for item in decisions
        ),
        episodic_result_count=sum(
            item.scope is ContextMemoryScope.EPISODIC for item in decisions
        ),
        decisions=decisions,
        context_item_ids=tuple(
            item.context_item_id
            for item in decisions
            if item.context_item_id is not None
        ),
    )


def create_context_item(
    *,
    source: ContextSource,
    section: ContextSection,
    verification_status: VerificationStatus,
    content: str,
    estimated_tokens: int,
    priority: int,
    pinned: bool = False,
) -> ContextItem:
    """Create one content-addressed bounded context item."""

    content_digest = hashlib.sha256(_encode_utf8(content, name="content")).hexdigest()
    return ContextItem(
        item_id=_context_item_id(
            source_id=source.source_id,
            section=section,
            content_digest=content_digest,
        ),
        source=source,
        section=section,
        verification_status=verification_status,
        content=content,
        content_digest=content_digest,
        estimated_tokens=estimated_tokens,
        priority=priority,
        pinned=pinned,
    )


def create_context_build_report(
    *,
    budget_tokens: int,
    decisions: tuple[ContextSelectionDecision, ...],
) -> ContextBuildReport:
    """Create and reconcile a complete context-selection decision ledger."""

    selected = tuple(
        decision
        for decision in decisions
        if decision.outcome is ContextSelectionOutcome.SELECTED
    )
    return ContextBuildReport(
        build_id=_context_build_id(budget_tokens, decisions),
        budget_tokens=budget_tokens,
        candidate_count=len(decisions),
        selected_count=len(selected),
        excluded_count=len(decisions) - len(selected),
        selected_tokens=sum(decision.estimated_tokens for decision in selected),
        decisions=decisions,
    )


def create_context_pack(
    *,
    report: ContextBuildReport,
    items: tuple[ContextItem, ...],
) -> ContextPack:
    """Create one ordered active-context pack tied to its decision report."""

    item_ids = tuple(item.item_id for item in items)
    return ContextPack(
        pack_id=_context_pack_id(report.build_id, item_ids),
        build_report=report,
        estimated_tokens=sum(item.estimated_tokens for item in items),
        items=items,
    )


def _canonical_json_hash(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()


def _raw_artifact_id(
    *,
    kind: RawArtifactKind,
    media_type: str,
    content_digest: str,
    byte_count: int,
) -> str:
    return _canonical_json_hash(
        b"repocompass:raw-artifact:v1",
        {
            "kind": kind.value,
            "media_type": media_type,
            "content_digest": content_digest,
            "byte_count": byte_count,
        },
    )


def _context_source_id(
    *,
    kind: ContextSourceKind,
    raw_artifact_id: str | None,
    location: RepositoryLocation | None,
    provenance: tuple[ProvenanceLink, ...],
    memory: ContextMemoryMetadata | None,
) -> str:
    payload = {
        "kind": kind.value,
        "raw_artifact_id": raw_artifact_id,
        "location": None if location is None else location.model_dump(mode="json"),
        "provenance": [link.model_dump(mode="json") for link in provenance],
    }
    if memory is not None:
        payload["memory"] = memory.model_dump(mode="json")
    return _canonical_json_hash(b"repocompass:context-source:v1", payload)


def _memory_recall_id(
    *,
    query_digest: str,
    decisions: tuple[MemoryRecallDecision, ...],
) -> str:
    return _canonical_json_hash(
        b"repocompass:memory-recall:v1",
        {
            "query_digest": query_digest,
            "decisions": [item.model_dump(mode="json") for item in decisions],
        },
    )


def _context_item_id(
    *,
    source_id: str,
    section: ContextSection,
    content_digest: str,
) -> str:
    return _canonical_json_hash(
        b"repocompass:context-item:v1",
        {
            "source_id": source_id,
            "section": section.value,
            "content_digest": content_digest,
        },
    )


def _context_build_id(
    budget_tokens: int,
    decisions: tuple[ContextSelectionDecision, ...],
) -> str:
    return _canonical_json_hash(
        b"repocompass:context-build:v1",
        {
            "budget_tokens": budget_tokens,
            "decisions": [decision.model_dump(mode="json") for decision in decisions],
        },
    )


def _context_pack_id(build_id: str, item_ids: tuple[str, ...]) -> str:
    return _canonical_json_hash(
        b"repocompass:context-pack:v1",
        {"build_id": build_id, "item_ids": item_ids},
    )


def _canonical_provenance(
    provenance: tuple[ProvenanceLink, ...],
) -> tuple[ProvenanceLink, ...]:
    return tuple(sorted(set(provenance), key=lambda link: (link.kind.value, link.id)))


def _encode_raw_content(content: str | bytes) -> bytes:
    if isinstance(content, str):
        return _encode_utf8(content, name="raw artifact content")
    if isinstance(content, bytes):
        return content
    raise TypeError("raw artifact content must be str or bytes")


def _encode_utf8(value: str, *, name: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
