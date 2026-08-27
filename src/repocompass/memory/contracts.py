"""Strict, content-addressed contracts for repository and episodic memory."""

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.context import RepositoryLocation
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

MEMORY_CONTRACT_SCHEMA_VERSION = 1
MAX_MEMORY_TITLE_CHARACTERS = 512
MAX_MEMORY_CONTENT_CHARACTERS = 16_384
MAX_MEMORY_SOURCES = 32
MAX_MEMORY_TAGS = 32
MAX_MEMORY_SEQUENCE_ITEMS = 64
MAX_MEMORY_ITEM_CHARACTERS = 1_024
_TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:-]{0,63}")


class MemoryRepositoryIdentityKind(StrEnum):
    """Safe source used to group repository clones without storing locators."""

    GIT_REMOTE = "git_remote"
    LOCAL_PATH = "local_path"


class MemoryFactKind(StrEnum):
    """Closed repository-fact vocabulary exposed to retrieval and the UI."""

    MODULE_RESPONSIBILITY = "module_responsibility"
    SYMBOL_RELATION = "symbol_relation"
    ENTRY_POINT = "entry_point"
    BUILD_COMMAND = "build_command"
    TEST_COMMAND = "test_command"
    ARCHITECTURE_FACT = "architecture_fact"


class MemoryVerificationStatus(StrEnum):
    """How a persisted memory record earned its current trust level."""

    VERIFIED = "verified"
    HUMAN_APPROVED = "human_approved"
    OBSERVED = "observed"


class MemoryWriteAuthority(StrEnum):
    """Closed write path; no value represents a model-controlled write."""

    VALIDATED_EVIDENCE = "validated_evidence"
    HUMAN_APPROVAL = "human_approval"
    TERMINAL_CONSOLIDATION = "terminal_consolidation"


class MemoryFreshnessStatus(StrEnum):
    """Whether a recalled record may be trusted in the current snapshot."""

    CURRENT = "current"
    REVALIDATION_REQUIRED = "revalidation_required"
    STALE = "stale"


class MemoryStalenessReason(StrEnum):
    """Closed explanation for downgrading a recalled memory."""

    COMMIT_MISMATCH = "commit_mismatch"
    SOURCE_REPOSITORY_MISMATCH = "source_repository_mismatch"
    DIRTY_WORKTREE = "dirty_worktree"
    UNVERSIONED_SNAPSHOT = "unversioned_snapshot"
    SOURCE_CHANGED = "source_changed"
    SOURCE_MISSING = "source_missing"
    CORRUPT_RECORD = "corrupt_record"


class MemoryAuditAction(StrEnum):
    """Append-only memory lifecycle action exposed to audit and UI layers."""

    CREATED = "created"
    REVERIFIED = "reverified"
    RETRIEVED = "retrieved"
    MARKED_STALE = "marked_stale"
    DELETED = "deleted"


class MemoryAuditReason(StrEnum):
    """Sanitized reason code for one memory audit action."""

    WRITE_POLICY_ALLOWED = "write_policy_allowed"
    CURRENT_SNAPSHOT_HIT = "current_snapshot_hit"
    EPISODIC_HISTORY_HIT = "episodic_history_hit"
    REVALIDATION_REQUIRED = "revalidation_required"
    SOURCE_STALE = "source_stale"
    USER_REQUEST = "user_request"
    RETENTION_POLICY = "retention_policy"
    CORRUPTION_DETECTED = "corruption_detected"


class _MemoryContractModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class MemoryRepositoryIdentity(_MemoryContractModel):
    """Stable repository key plus the exact local clone that supplied evidence."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    repository_key: Sha256Digest
    kind: MemoryRepositoryIdentityKind
    locator_digest: Sha256Digest
    scope: str = Field(min_length=1, max_length=4_096)
    source_repository_id: Sha256Digest

    @field_validator("scope")
    @classmethod
    def require_canonical_scope(cls, value: str) -> str:
        _validate_scope(value)
        return value

    @model_validator(mode="after")
    def require_content_addressed_key(self) -> "MemoryRepositoryIdentity":
        expected = _repository_key(
            kind=self.kind,
            locator_digest=self.locator_digest,
            scope=self.scope,
        )
        if self.repository_key != expected:
            raise ValueError("repository_key does not match repository identity")
        return self


class RepositoryMemorySource(_MemoryContractModel):
    """One immutable source anchor supporting a repository-memory fact."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    source_id: Sha256Digest
    location: RepositoryLocation
    source_digest: Sha256Digest
    evidence_claim_digest: Sha256Digest | None = None
    supporting_assertion_ids: tuple[Sha256Digest, ...] = Field(
        default=(),
        max_length=64,
    )

    @field_validator("supporting_assertion_ids")
    @classmethod
    def require_canonical_assertions(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("supporting assertion IDs must be canonical")
        return value

    @model_validator(mode="after")
    def require_versioned_content_addressed_source(self) -> "RepositoryMemorySource":
        if self.location.commit_sha is None:
            raise ValueError("repository memory source requires a commit SHA")
        expected = _repository_memory_source_id(
            location=self.location,
            source_digest=self.source_digest,
            evidence_claim_digest=self.evidence_claim_digest,
            supporting_assertion_ids=self.supporting_assertion_ids,
        )
        if self.source_id != expected:
            raise ValueError("source_id does not match repository memory source")
        return self


class RepositoryMemory(_MemoryContractModel):
    """Immutable, source-backed repository knowledge eligible for retrieval."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    memory_id: Sha256Digest
    repository_key: Sha256Digest
    source_repository_id: Sha256Digest
    commit_sha: CommitSha
    kind: MemoryFactKind
    title: str = Field(min_length=1, max_length=MAX_MEMORY_TITLE_CHARACTERS)
    content: str = Field(min_length=1, max_length=MAX_MEMORY_CONTENT_CHARACTERS)
    verification_status: MemoryVerificationStatus
    write_authority: MemoryWriteAuthority
    sources: tuple[RepositoryMemorySource, ...] = Field(
        min_length=1,
        max_length=MAX_MEMORY_SOURCES,
    )
    tags: tuple[str, ...] = Field(default=(), max_length=MAX_MEMORY_TAGS)
    created_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    verified_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    supersedes_memory_id: Sha256Digest | None = None

    @field_validator("title")
    @classmethod
    def require_canonical_title(cls, value: str) -> str:
        _validate_single_line(value, name="title")
        if value != " ".join(value.split()):
            raise ValueError("title must be canonical single-line text")
        return value

    @field_validator("content")
    @classmethod
    def require_safe_content(cls, value: str) -> str:
        _validate_bounded_text(value, name="content")
        if value != value.strip() or "\r" in value:
            raise ValueError("content must be canonical text")
        return value

    @field_validator("sources")
    @classmethod
    def require_canonical_sources(
        cls,
        value: tuple[RepositoryMemorySource, ...],
    ) -> tuple[RepositoryMemorySource, ...]:
        source_ids = tuple(item.source_id for item in value)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("sources must be unique and canonical")
        return value

    @field_validator("tags")
    @classmethod
    def require_canonical_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("tags must be unique and canonical")
        if any(_TAG_PATTERN.fullmatch(tag) is None for tag in value):
            raise ValueError("tags contain an invalid value")
        return value

    @model_validator(mode="after")
    def require_traceable_content_addressed_fact(self) -> "RepositoryMemory":
        expected_authority = {
            MemoryVerificationStatus.VERIFIED: (
                MemoryWriteAuthority.VALIDATED_EVIDENCE
            ),
            MemoryVerificationStatus.HUMAN_APPROVED: (
                MemoryWriteAuthority.HUMAN_APPROVAL
            ),
        }.get(self.verification_status)
        if expected_authority is None or self.write_authority is not expected_authority:
            raise ValueError("repository memory authority does not match status")
        if self.verified_at_ms < self.created_at_ms:
            raise ValueError("verified_at_ms must not precede created_at_ms")
        if any(
            source.location.repository_id != self.source_repository_id
            or source.location.commit_sha != self.commit_sha
            for source in self.sources
        ):
            raise ValueError("repository memory source snapshot does not match")
        expected = _repository_memory_id(
            repository_key=self.repository_key,
            source_repository_id=self.source_repository_id,
            commit_sha=self.commit_sha,
            kind=self.kind,
            title=self.title,
            content=self.content,
            verification_status=self.verification_status,
            write_authority=self.write_authority,
            sources=self.sources,
            tags=self.tags,
            created_at_ms=self.created_at_ms,
            verified_at_ms=self.verified_at_ms,
            supersedes_memory_id=self.supersedes_memory_id,
        )
        if self.memory_id != expected:
            raise ValueError("memory_id does not match repository memory contents")
        if self.supersedes_memory_id == self.memory_id:
            raise ValueError("repository memory cannot supersede itself")
        return self


class EpisodicMemory(_MemoryContractModel):
    """Immutable terminal investigation history, never a verified code fact."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    episode_id: Sha256Digest
    repository_key: Sha256Digest
    source_repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    termination_status: Literal["accepted", "exhausted", "cancelled"]
    objective_digest: Sha256Digest
    summary: str = Field(min_length=1, max_length=MAX_MEMORY_CONTENT_CHARACTERS)
    query_sequence: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_MEMORY_SEQUENCE_ITEMS,
    )
    successful_paths: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_MEMORY_SEQUENCE_ITEMS,
    )
    failed_hypotheses: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_MEMORY_SEQUENCE_ITEMS,
    )
    final_result_digest: Sha256Digest
    verification_status: Literal[MemoryVerificationStatus.OBSERVED] = (
        MemoryVerificationStatus.OBSERVED
    )
    write_authority: Literal[MemoryWriteAuthority.TERMINAL_CONSOLIDATION] = (
        MemoryWriteAuthority.TERMINAL_CONSOLIDATION
    )
    created_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)

    @field_validator("summary")
    @classmethod
    def require_safe_summary(cls, value: str) -> str:
        _validate_bounded_text(value, name="summary")
        if value != value.strip() or "\r" in value:
            raise ValueError("summary must be canonical text")
        return value

    @field_validator("query_sequence", "failed_hypotheses")
    @classmethod
    def require_safe_sequence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if len(item) > MAX_MEMORY_ITEM_CHARACTERS:
                raise ValueError("memory sequence item exceeds character limit")
            _validate_bounded_text(item, name="memory sequence item")
            if item != item.strip() or "\r" in item:
                raise ValueError("memory sequence item must be canonical")
        return value

    @field_validator("successful_paths")
    @classmethod
    def require_canonical_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("successful paths must be unique and canonical")
        for path in value:
            _validate_repository_path(path)
        return value

    @model_validator(mode="after")
    def require_content_addressed_episode(self) -> "EpisodicMemory":
        expected = _episodic_memory_id(
            repository_key=self.repository_key,
            source_repository_id=self.source_repository_id,
            commit_sha=self.commit_sha,
            run_id=self.run_id,
            thread_id=self.thread_id,
            turn_number=self.turn_number,
            termination_status=self.termination_status,
            objective_digest=self.objective_digest,
            summary=self.summary,
            query_sequence=self.query_sequence,
            successful_paths=self.successful_paths,
            failed_hypotheses=self.failed_hypotheses,
            final_result_digest=self.final_result_digest,
            created_at_ms=self.created_at_ms,
        )
        if self.episode_id != expected:
            raise ValueError("episode_id does not match episodic memory contents")
        return self


class MemoryRetrievalHit(_MemoryContractModel):
    """One ranked memory candidate with an explicit current-snapshot status."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    hit_id: Sha256Digest
    query_digest: Sha256Digest
    memory_id: Sha256Digest
    repository_key: Sha256Digest
    memory_source_repository_id: Sha256Digest
    current_source_repository_id: Sha256Digest
    memory_commit_sha: CommitSha
    current_commit_sha: CommitSha | None = None
    rank: int = Field(ge=1, le=1_000)
    freshness: MemoryFreshnessStatus
    staleness_reason: MemoryStalenessReason | None = None
    original_verification_status: MemoryVerificationStatus

    @model_validator(mode="after")
    def require_explicit_snapshot_trust(self) -> "MemoryRetrievalHit":
        if self.freshness is MemoryFreshnessStatus.CURRENT:
            if (
                self.staleness_reason is not None
                or self.current_commit_sha != self.memory_commit_sha
                or self.current_source_repository_id != self.memory_source_repository_id
            ):
                raise ValueError("current memory hit must match the exact snapshot")
        elif self.freshness is MemoryFreshnessStatus.REVALIDATION_REQUIRED:
            if self.staleness_reason not in {
                MemoryStalenessReason.COMMIT_MISMATCH,
                MemoryStalenessReason.SOURCE_REPOSITORY_MISMATCH,
                MemoryStalenessReason.DIRTY_WORKTREE,
                MemoryStalenessReason.UNVERSIONED_SNAPSHOT,
            }:
                raise ValueError("revalidation hit requires a candidate reason")
        elif self.staleness_reason not in {
            MemoryStalenessReason.SOURCE_CHANGED,
            MemoryStalenessReason.SOURCE_MISSING,
            MemoryStalenessReason.CORRUPT_RECORD,
        }:
            raise ValueError("stale hit requires a terminal stale reason")

        expected = _memory_retrieval_hit_id(
            query_digest=self.query_digest,
            memory_id=self.memory_id,
            repository_key=self.repository_key,
            memory_source_repository_id=self.memory_source_repository_id,
            current_source_repository_id=self.current_source_repository_id,
            memory_commit_sha=self.memory_commit_sha,
            current_commit_sha=self.current_commit_sha,
            rank=self.rank,
            freshness=self.freshness,
            staleness_reason=self.staleness_reason,
            original_verification_status=self.original_verification_status,
        )
        if self.hit_id != expected:
            raise ValueError("hit_id does not match memory retrieval contents")
        return self


class MemoryAuditEvent(_MemoryContractModel):
    """Content-addressed, sanitized memory lifecycle audit event."""

    schema_version: Literal[1] = MEMORY_CONTRACT_SCHEMA_VERSION
    event_id: Sha256Digest
    memory_id: Sha256Digest
    repository_key: Sha256Digest
    action: MemoryAuditAction
    reason: MemoryAuditReason
    created_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    run_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    actor_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def require_compatible_content_addressed_event(self) -> "MemoryAuditEvent":
        allowed_reasons = {
            MemoryAuditAction.CREATED: {MemoryAuditReason.WRITE_POLICY_ALLOWED},
            MemoryAuditAction.REVERIFIED: {MemoryAuditReason.WRITE_POLICY_ALLOWED},
            MemoryAuditAction.RETRIEVED: {
                MemoryAuditReason.CURRENT_SNAPSHOT_HIT,
                MemoryAuditReason.EPISODIC_HISTORY_HIT,
                MemoryAuditReason.REVALIDATION_REQUIRED,
            },
            MemoryAuditAction.MARKED_STALE: {
                MemoryAuditReason.SOURCE_STALE,
                MemoryAuditReason.CORRUPTION_DETECTED,
            },
            MemoryAuditAction.DELETED: {
                MemoryAuditReason.USER_REQUEST,
                MemoryAuditReason.RETENTION_POLICY,
            },
        }
        if self.reason not in allowed_reasons[self.action]:
            raise ValueError("memory audit action and reason are incompatible")
        expected = _memory_audit_event_id(
            memory_id=self.memory_id,
            repository_key=self.repository_key,
            action=self.action,
            reason=self.reason,
            created_at_ms=self.created_at_ms,
            run_id=self.run_id,
            actor_digest=self.actor_digest,
        )
        if self.event_id != expected:
            raise ValueError("event_id does not match memory audit event")
        return self


def create_memory_repository_identity(
    *,
    kind: MemoryRepositoryIdentityKind,
    locator_digest: str,
    scope: str,
    source_repository_id: str,
) -> MemoryRepositoryIdentity:
    """Create a clone-aware identity without persisting a raw remote or path."""

    repository_key = _repository_key(
        kind=kind,
        locator_digest=locator_digest,
        scope=scope,
    )
    return MemoryRepositoryIdentity(
        repository_key=repository_key,
        kind=kind,
        locator_digest=locator_digest,
        scope=scope,
        source_repository_id=source_repository_id,
    )


def create_repository_memory_source(
    *,
    location: RepositoryLocation,
    source_digest: str,
    evidence_claim_digest: str | None = None,
    supporting_assertion_ids: tuple[str, ...] = (),
) -> RepositoryMemorySource:
    """Create one source anchor whose identity covers every provenance field."""

    source_id = _repository_memory_source_id(
        location=location,
        source_digest=source_digest,
        evidence_claim_digest=evidence_claim_digest,
        supporting_assertion_ids=supporting_assertion_ids,
    )
    return RepositoryMemorySource(
        source_id=source_id,
        location=location,
        source_digest=source_digest,
        evidence_claim_digest=evidence_claim_digest,
        supporting_assertion_ids=supporting_assertion_ids,
    )


def create_repository_memory(
    *,
    identity: MemoryRepositoryIdentity,
    commit_sha: str,
    kind: MemoryFactKind,
    title: str,
    content: str,
    verification_status: MemoryVerificationStatus,
    write_authority: MemoryWriteAuthority,
    sources: tuple[RepositoryMemorySource, ...],
    tags: tuple[str, ...] = (),
    created_at_ms: int,
    verified_at_ms: int,
    supersedes_memory_id: str | None = None,
) -> RepositoryMemory:
    """Create one immutable repository-memory version."""

    memory_id = _repository_memory_id(
        repository_key=identity.repository_key,
        source_repository_id=identity.source_repository_id,
        commit_sha=commit_sha,
        kind=kind,
        title=title,
        content=content,
        verification_status=verification_status,
        write_authority=write_authority,
        sources=sources,
        tags=tags,
        created_at_ms=created_at_ms,
        verified_at_ms=verified_at_ms,
        supersedes_memory_id=supersedes_memory_id,
    )
    return RepositoryMemory(
        memory_id=memory_id,
        repository_key=identity.repository_key,
        source_repository_id=identity.source_repository_id,
        commit_sha=commit_sha,
        kind=kind,
        title=title,
        content=content,
        verification_status=verification_status,
        write_authority=write_authority,
        sources=sources,
        tags=tags,
        created_at_ms=created_at_ms,
        verified_at_ms=verified_at_ms,
        supersedes_memory_id=supersedes_memory_id,
    )


def create_episodic_memory(
    *,
    identity: MemoryRepositoryIdentity,
    commit_sha: str | None,
    run_id: str,
    thread_id: str,
    turn_number: int,
    termination_status: Literal["accepted", "exhausted", "cancelled"],
    objective_digest: str,
    summary: str,
    query_sequence: tuple[str, ...],
    successful_paths: tuple[str, ...],
    failed_hypotheses: tuple[str, ...],
    final_result_digest: str,
    created_at_ms: int,
) -> EpisodicMemory:
    """Create one model-inaccessible terminal investigation episode."""

    episode_id = _episodic_memory_id(
        repository_key=identity.repository_key,
        source_repository_id=identity.source_repository_id,
        commit_sha=commit_sha,
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        termination_status=termination_status,
        objective_digest=objective_digest,
        summary=summary,
        query_sequence=query_sequence,
        successful_paths=successful_paths,
        failed_hypotheses=failed_hypotheses,
        final_result_digest=final_result_digest,
        created_at_ms=created_at_ms,
    )
    return EpisodicMemory(
        episode_id=episode_id,
        repository_key=identity.repository_key,
        source_repository_id=identity.source_repository_id,
        commit_sha=commit_sha,
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        termination_status=termination_status,
        objective_digest=objective_digest,
        summary=summary,
        query_sequence=query_sequence,
        successful_paths=successful_paths,
        failed_hypotheses=failed_hypotheses,
        final_result_digest=final_result_digest,
        created_at_ms=created_at_ms,
    )


def create_memory_retrieval_hit(
    *,
    query_digest: str,
    memory_id: str,
    repository_key: str,
    memory_source_repository_id: str,
    current_source_repository_id: str,
    memory_commit_sha: str,
    current_commit_sha: str | None,
    rank: int,
    freshness: MemoryFreshnessStatus,
    staleness_reason: MemoryStalenessReason | None,
    original_verification_status: MemoryVerificationStatus,
) -> MemoryRetrievalHit:
    """Create one retrieval result without implicitly preserving verification."""

    hit_id = _memory_retrieval_hit_id(
        query_digest=query_digest,
        memory_id=memory_id,
        repository_key=repository_key,
        memory_source_repository_id=memory_source_repository_id,
        current_source_repository_id=current_source_repository_id,
        memory_commit_sha=memory_commit_sha,
        current_commit_sha=current_commit_sha,
        rank=rank,
        freshness=freshness,
        staleness_reason=staleness_reason,
        original_verification_status=original_verification_status,
    )
    return MemoryRetrievalHit(
        hit_id=hit_id,
        query_digest=query_digest,
        memory_id=memory_id,
        repository_key=repository_key,
        memory_source_repository_id=memory_source_repository_id,
        current_source_repository_id=current_source_repository_id,
        memory_commit_sha=memory_commit_sha,
        current_commit_sha=current_commit_sha,
        rank=rank,
        freshness=freshness,
        staleness_reason=staleness_reason,
        original_verification_status=original_verification_status,
    )


def create_memory_audit_event(
    *,
    memory_id: str,
    repository_key: str,
    action: MemoryAuditAction,
    reason: MemoryAuditReason,
    created_at_ms: int,
    run_id: str | None = None,
    actor_digest: str | None = None,
) -> MemoryAuditEvent:
    """Create one append-only audit event with only sanitized identities."""

    event_id = _memory_audit_event_id(
        memory_id=memory_id,
        repository_key=repository_key,
        action=action,
        reason=reason,
        created_at_ms=created_at_ms,
        run_id=run_id,
        actor_digest=actor_digest,
    )
    return MemoryAuditEvent(
        event_id=event_id,
        memory_id=memory_id,
        repository_key=repository_key,
        action=action,
        reason=reason,
        created_at_ms=created_at_ms,
        run_id=run_id,
        actor_digest=actor_digest,
    )


def _repository_key(
    *,
    kind: MemoryRepositoryIdentityKind,
    locator_digest: str,
    scope: str,
) -> str:
    return _canonical_hash(
        b"repocompass:memory-repository:v1",
        {
            "kind": kind.value,
            "locator_digest": locator_digest,
            "scope": scope,
        },
    )


def _repository_memory_source_id(
    *,
    location: RepositoryLocation,
    source_digest: str,
    evidence_claim_digest: str | None,
    supporting_assertion_ids: tuple[str, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:repository-memory-source:v1",
        {
            "location": location.model_dump(mode="json"),
            "source_digest": source_digest,
            "evidence_claim_digest": evidence_claim_digest,
            "supporting_assertion_ids": supporting_assertion_ids,
        },
    )


def _repository_memory_id(
    *,
    repository_key: str,
    source_repository_id: str,
    commit_sha: str,
    kind: MemoryFactKind,
    title: str,
    content: str,
    verification_status: MemoryVerificationStatus,
    write_authority: MemoryWriteAuthority,
    sources: tuple[RepositoryMemorySource, ...],
    tags: tuple[str, ...],
    created_at_ms: int,
    verified_at_ms: int,
    supersedes_memory_id: str | None,
) -> str:
    return _canonical_hash(
        b"repocompass:repository-memory:v1",
        {
            "repository_key": repository_key,
            "source_repository_id": source_repository_id,
            "commit_sha": commit_sha,
            "kind": kind.value,
            "title": title,
            "content": content,
            "verification_status": verification_status.value,
            "write_authority": write_authority.value,
            "sources": [item.model_dump(mode="json") for item in sources],
            "tags": tags,
            "created_at_ms": created_at_ms,
            "verified_at_ms": verified_at_ms,
            "supersedes_memory_id": supersedes_memory_id,
        },
    )


def _episodic_memory_id(
    *,
    repository_key: str,
    source_repository_id: str,
    commit_sha: str | None,
    run_id: str,
    thread_id: str,
    turn_number: int,
    termination_status: Literal["accepted", "exhausted", "cancelled"],
    objective_digest: str,
    summary: str,
    query_sequence: tuple[str, ...],
    successful_paths: tuple[str, ...],
    failed_hypotheses: tuple[str, ...],
    final_result_digest: str,
    created_at_ms: int,
) -> str:
    return _canonical_hash(
        b"repocompass:episodic-memory:v1",
        {
            "repository_key": repository_key,
            "source_repository_id": source_repository_id,
            "commit_sha": commit_sha,
            "run_id": run_id,
            "thread_id": thread_id,
            "turn_number": turn_number,
            "termination_status": termination_status,
            "objective_digest": objective_digest,
            "summary": summary,
            "query_sequence": query_sequence,
            "successful_paths": successful_paths,
            "failed_hypotheses": failed_hypotheses,
            "final_result_digest": final_result_digest,
            "created_at_ms": created_at_ms,
        },
    )


def _memory_retrieval_hit_id(
    *,
    query_digest: str,
    memory_id: str,
    repository_key: str,
    memory_source_repository_id: str,
    current_source_repository_id: str,
    memory_commit_sha: str,
    current_commit_sha: str | None,
    rank: int,
    freshness: MemoryFreshnessStatus,
    staleness_reason: MemoryStalenessReason | None,
    original_verification_status: MemoryVerificationStatus,
) -> str:
    return _canonical_hash(
        b"repocompass:memory-retrieval-hit:v1",
        {
            "query_digest": query_digest,
            "memory_id": memory_id,
            "repository_key": repository_key,
            "memory_source_repository_id": memory_source_repository_id,
            "current_source_repository_id": current_source_repository_id,
            "memory_commit_sha": memory_commit_sha,
            "current_commit_sha": current_commit_sha,
            "rank": rank,
            "freshness": freshness.value,
            "staleness_reason": (
                None if staleness_reason is None else staleness_reason.value
            ),
            "original_verification_status": original_verification_status.value,
        },
    )


def _memory_audit_event_id(
    *,
    memory_id: str,
    repository_key: str,
    action: MemoryAuditAction,
    reason: MemoryAuditReason,
    created_at_ms: int,
    run_id: str | None,
    actor_digest: str | None,
) -> str:
    return _canonical_hash(
        b"repocompass:memory-audit-event:v1",
        {
            "memory_id": memory_id,
            "repository_key": repository_key,
            "action": action.value,
            "reason": reason.value,
            "created_at_ms": created_at_ms,
            "run_id": run_id,
            "actor_digest": actor_digest,
        },
    )


def _validate_scope(value: str) -> None:
    if value == ".":
        return
    _validate_repository_path(value)


def _validate_repository_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("repository paths must be normalized POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("repository paths must be normalized POSIX paths")


def _validate_single_line(value: str, *, name: str) -> None:
    _validate_bounded_text(value, name=name)
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be single-line text")


def _validate_bounded_text(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    if any(
        character == "\x00"
        or (
            unicodedata.category(character).startswith("C")
            and character not in {"\n", "\t"}
        )
        for character in value
    ):
        raise ValueError(f"{name} contains unsafe control characters")


def _canonical_hash(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()
