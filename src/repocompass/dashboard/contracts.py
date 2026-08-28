"""Versioned, UI-safe projections for the local investigation dashboard."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from repocompass.agent.context import ContextPack
from repocompass.agent.cost import AttemptUsageCostReport
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.agent.trace import TraceEventKind
from repocompass.agent.usage import ModelInvocationRecord
from repocompass.memory.contracts import (
    MemoryAuditEvent,
    MemoryEntryMetadata,
    MemoryTombstone,
)
from repocompass.storage.raw_store import RawRunAttempt
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

DASHBOARD_SCHEMA_VERSION = 1


class DashboardAvailability(StrEnum):
    """Closed empty-state vocabulary used across dashboard resources."""

    AVAILABLE = "available"
    NOT_FOUND = "not_found"
    NOT_PERSISTED = "not_persisted"
    NOT_CONFIGURED = "not_configured"


class _DashboardModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class DashboardSystemInfo(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    product: Literal["RepoCompass"] = "RepoCompass"
    product_version: str = Field(pattern=IDENTIFIER_PATTERN)
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    pricing_configured: bool
    lifecycle_mutations_enabled: bool
    memory_mutations_enabled: bool


class DashboardRunSummary(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    attempt: RawRunAttempt
    trace_availability: DashboardAvailability
    cost_availability: DashboardAvailability

    @model_validator(mode="after")
    def restrict_availability_values(self) -> "DashboardRunSummary":
        if self.trace_availability not in {
            DashboardAvailability.AVAILABLE,
            DashboardAvailability.NOT_PERSISTED,
        }:
            raise ValueError("run trace availability is invalid")
        if self.cost_availability not in {
            DashboardAvailability.AVAILABLE,
            DashboardAvailability.NOT_CONFIGURED,
        }:
            raise ValueError("run cost availability is invalid")
        return self


class DashboardThreadSummary(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_availability: DashboardAvailability
    objective: str | None = Field(default=None, min_length=1, max_length=16_384)
    execution_state: ThreadExecutionState | None = None
    turn_number: int | None = Field(
        default=None,
        ge=1,
        le=MAX_SAFE_JSON_INTEGER,
    )
    termination_status: TerminationStatus | None = None
    latest_attempt: RawRunAttempt
    attempt_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def reconcile_checkpoint_availability(self) -> "DashboardThreadSummary":
        available = self.checkpoint_availability is DashboardAvailability.AVAILABLE
        if available != (
            self.objective is not None
            and self.execution_state is not None
            and self.turn_number is not None
        ):
            raise ValueError("thread summary checkpoint fields disagree")
        if not available and self.termination_status is not None:
            raise ValueError("unavailable checkpoint cannot expose termination")
        return self


class DashboardThreadCatalog(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    threads: tuple[DashboardThreadSummary, ...] = Field(max_length=100)
    scanned_attempt_count: int = Field(ge=0, le=1_000)
    truncated: bool


class PublicTraceEvent(_DashboardModel):
    """One domain event after recursive local-path/secret redaction."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    sequence: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    kind: TraceEventKind
    node: str | None = Field(default=None, min_length=1, max_length=128)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class PlanReplayFrame(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    sequence: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    node: str = Field(min_length=1, max_length=128)
    revision_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    plan: InvestigationPlan


class EvaluatorReplayFrame(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    sequence: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    node: str = Field(min_length=1, max_length=128)
    report: EvaluationReport


class DashboardThreadDetail(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    thread: ThreadInvestigationView
    attempts: tuple[DashboardRunSummary, ...] = Field(max_length=100)
    plan_timeline: tuple[PlanReplayFrame, ...] = Field(max_length=10_000)
    evaluator_timeline: tuple[EvaluatorReplayFrame, ...] = Field(max_length=10_000)

    @model_validator(mode="after")
    def reconcile_thread_attempts(self) -> "DashboardThreadDetail":
        if any(
            summary.attempt.thread_id != self.thread.thread_id
            for summary in self.attempts
        ):
            raise ValueError("thread detail contains an unrelated attempt")
        return self


class DashboardRunDetail(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    attempt: RawRunAttempt
    trace_availability: DashboardAvailability
    trace_events: tuple[PublicTraceEvent, ...] = Field(max_length=10_000)
    plan_timeline: tuple[PlanReplayFrame, ...] = Field(max_length=10_000)
    evaluator_timeline: tuple[EvaluatorReplayFrame, ...] = Field(max_length=10_000)
    context_packs: tuple[ContextPack, ...] = Field(max_length=10_000)
    model_invocations: tuple[ModelInvocationRecord, ...] = Field(max_length=10_000)
    cost_availability: DashboardAvailability
    usage_cost_report: AttemptUsageCostReport | None = None

    @model_validator(mode="after")
    def reconcile_optional_artifacts(self) -> "DashboardRunDetail":
        if self.trace_availability is DashboardAvailability.AVAILABLE:
            if any(event.run_id != self.attempt.run_id for event in self.trace_events):
                raise ValueError("run detail contains an unrelated trace event")
        elif (
            self.trace_availability is not DashboardAvailability.NOT_PERSISTED
            or self.trace_events
            or self.plan_timeline
            or self.evaluator_timeline
        ):
            raise ValueError("missing trace must have empty replay artifacts")
        cost_available = self.cost_availability is DashboardAvailability.AVAILABLE
        if cost_available != (self.usage_cost_report is not None):
            raise ValueError("run cost availability and report disagree")
        if not cost_available and (
            self.cost_availability is not DashboardAvailability.NOT_CONFIGURED
        ):
            raise ValueError("run cost availability is invalid")
        if any(
            invocation.run_id != self.attempt.run_id
            for invocation in self.model_invocations
        ):
            raise ValueError("run detail contains an unrelated invocation")
        return self


class DashboardMemoryCatalog(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    availability: DashboardAvailability
    entries: tuple[MemoryEntryMetadata, ...] = Field(max_length=1_000)
    tombstones: tuple[MemoryTombstone, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def reconcile_memory_availability(self) -> "DashboardMemoryCatalog":
        if self.availability not in {
            DashboardAvailability.AVAILABLE,
            DashboardAvailability.NOT_CONFIGURED,
        }:
            raise ValueError("memory availability is invalid")
        if self.availability is DashboardAvailability.NOT_CONFIGURED and (
            self.entries or self.tombstones
        ):
            raise ValueError("unconfigured memory cannot expose records")
        return self


class DashboardMemoryAudit(_DashboardModel):
    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    memory_id: Sha256Digest
    metadata: MemoryEntryMetadata | None = None
    tombstone: MemoryTombstone | None = None
    events: tuple[MemoryAuditEvent, ...] = Field(max_length=10_000)

    @model_validator(mode="after")
    def require_live_or_deleted_identity(self) -> "DashboardMemoryAudit":
        if (self.metadata is None) == (self.tombstone is None):
            raise ValueError("memory audit requires exactly one live/deleted identity")
        identity = self.metadata or self.tombstone
        assert identity is not None
        if identity.memory_id != self.memory_id:
            raise ValueError("memory audit identity disagrees")
        if any(event.memory_id != self.memory_id for event in self.events):
            raise ValueError("memory audit contains an unrelated event")
        return self
