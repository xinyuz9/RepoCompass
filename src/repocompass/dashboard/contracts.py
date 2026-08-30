"""Versioned, UI-safe projections for the local investigation dashboard."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from repocompass.agent.context import ContextPack
from repocompass.agent.cost import (
    AttemptUsageCostReport,
    validate_cost_budget_usd,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import HumanDecision, HumanReviewPolicy
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
from repocompass.service.contracts import (
    REQUEST_ID_PATTERN,
    InvestigationOperationKind,
    InvestigationOperationResult,
    InvestigationOperationState,
)
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
    workspace_name: str = Field(min_length=1, max_length=256)
    workspace_path: Literal["."] = "."
    memory_available: bool
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


class DashboardLifecycleOperationStatus(_DashboardModel):
    """Reconnect-safe progress projection for one lifecycle request."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    kind: InvestigationOperationKind
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    state: InvestigationOperationState
    reserved_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    latest_attempt: RawRunAttempt | None = None
    result: InvestigationOperationResult | None = None

    @model_validator(mode="after")
    def reconcile_operation_progress(self) -> "DashboardLifecycleOperationStatus":
        completed = self.state is InvestigationOperationState.COMPLETED
        if completed != (self.result is not None):
            raise ValueError("operation state and result disagree")
        if self.result is not None and (
            self.result.request_id != self.request_id
            or self.result.thread_id != self.thread_id
        ):
            raise ValueError("operation result identity does not reconcile")
        if self.latest_attempt is not None:
            if (
                self.latest_attempt.thread_id != self.thread_id
                or self.latest_attempt.execution_mode.value != self.kind.value
                or self.latest_attempt.started_at_ms < self.reserved_at_ms
            ):
                raise ValueError("operation attempt identity does not reconcile")
            if (
                self.result is not None
                and self.result.run_id is not None
                and self.latest_attempt.run_id != self.result.run_id
            ):
                raise ValueError("completed operation run identity does not reconcile")
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


class DashboardEventStreamCursor(_DashboardModel):
    """Terminal cursor emitted by one bounded SSE snapshot."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    availability: DashboardAvailability
    after_sequence: int = Field(ge=-1, le=MAX_SAFE_JSON_INTEGER)
    emitted_count: int = Field(ge=0, le=1_000)
    next_after_sequence: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )
    truncated: bool

    @model_validator(mode="after")
    def reconcile_stream_cursor(self) -> "DashboardEventStreamCursor":
        if self.availability not in {
            DashboardAvailability.AVAILABLE,
            DashboardAvailability.NOT_PERSISTED,
        }:
            raise ValueError("event stream availability is invalid")
        if self.availability is DashboardAvailability.NOT_PERSISTED and (
            self.emitted_count != 0
            or self.next_after_sequence is not None
            or self.truncated
        ):
            raise ValueError("missing trace cannot emit events")
        if (self.emitted_count == 0) != (self.next_after_sequence is None):
            raise ValueError("event stream cursor and count disagree")
        if self.truncated and self.next_after_sequence is None:
            raise ValueError("truncated event stream requires a cursor")
        if (
            self.next_after_sequence is not None
            and self.next_after_sequence <= self.after_sequence
        ):
            raise ValueError("event stream cursor did not advance")
        return self


class DashboardProblemDetail(_DashboardModel):
    """Sanitized RFC 9457-style error body used by every HTTP failure."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    type: str = Field(pattern=r"^urn:repocompass:problem:[a-z0-9-]+$")
    title: str = Field(min_length=1, max_length=128)
    status: int = Field(ge=400, le=599)
    detail: str = Field(min_length=1, max_length=1_024)
    instance: str = Field(min_length=1, max_length=2_048)


class DashboardStartCommand(_DashboardModel):
    """Start one goal inside the repository selected by the server operator."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    goal: str = Field(min_length=1, max_length=16_384)
    focus_path: str = Field(default=".", min_length=1, max_length=2_048)
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.BEFORE_FINALIZE
    memory_retrieval_enabled: bool = True
    max_attempt_cost_usd: str | None = None

    @field_validator("human_review_policy", mode="before")
    @classmethod
    def decode_json_review_policy(cls, value: object) -> HumanReviewPolicy:
        if isinstance(value, HumanReviewPolicy):
            return value
        try:
            return HumanReviewPolicy(value)
        except (TypeError, ValueError) as error:
            raise ValueError("human review policy is invalid") from error

    @field_validator("max_attempt_cost_usd", mode="before")
    @classmethod
    def require_canonical_cost_budget(cls, value: object) -> object:
        if value is None:
            return None
        try:
            return validate_cost_budget_usd(value)
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error


class DashboardResumeCommand(_DashboardModel):
    """Resume pending graph work against the exact visible repository snapshot."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    expected_repository_id: Sha256Digest
    expected_commit_sha: CommitSha | None = None


class DashboardContinueCommand(_DashboardModel):
    """Submit a new Human turn to one completed investigation."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    prompt: str = Field(min_length=1, max_length=16_384)
    focus_path: str = Field(default=".", min_length=1, max_length=2_048)
    expected_repository_id: Sha256Digest
    expected_commit_sha: CommitSha | None = None


class DashboardHumanResponseCommand(_DashboardModel):
    """One idempotent decision bound to an exact pending interrupt."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    interrupt_id: str = Field(pattern=IDENTIFIER_PATTERN, max_length=256)
    decision: HumanDecision

    @field_validator("decision", mode="before")
    @classmethod
    def decode_json_decision(cls, value: object) -> HumanDecision:
        if isinstance(value, HumanDecision):
            return value
        return HumanDecision.model_validate(value, strict=False)

    @model_validator(mode="after")
    def bound_revision_feedback(self) -> "DashboardHumanResponseCommand":
        if self.decision.feedback is not None and len(self.decision.feedback) > 16_384:
            raise ValueError("revision feedback exceeds the dashboard limit")
        return self


class DashboardMemoryDeleteCommand(_DashboardModel):
    """Explicit confirmation for one payload-free memory identity."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    confirm_memory_id: Sha256Digest

    @model_validator(mode="after")
    def bind_idempotency_to_memory(self) -> "DashboardMemoryDeleteCommand":
        if self.request_id != f"memory-delete:{self.confirm_memory_id}":
            raise ValueError("memory deletion request_id must bind the exact identity")
        return self


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


class DashboardMemoryDeleteResult(_DashboardModel):
    """Idempotent proof that an exact memory identity is now deleted."""

    schema_version: Literal[1] = DASHBOARD_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    memory_id: Sha256Digest
    tombstone: MemoryTombstone
    idempotent_replay: bool

    @model_validator(mode="after")
    def reconcile_deleted_identity(self) -> "DashboardMemoryDeleteResult":
        if self.tombstone.memory_id != self.memory_id:
            raise ValueError("memory deletion result identity disagrees")
        return self
