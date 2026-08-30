"""Stable application contracts shared by MCP and future HTTP/UI adapters."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.context import ContextPack, MemoryRecallReport
from repocompass.agent.cost import (
    AttemptUsageCostReport,
    validate_cost_budget_usd,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanReviewPolicy,
    HumanReviewRequest,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.runner import (
    DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_NO_PROGRESS_ROUNDS,
    DEFAULT_MAX_PLAN_REVISIONS,
    DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
)
from repocompass.agent.context_builder import DEFAULT_CONTEXT_BUDGET_TOKENS
from repocompass.agent.structural_artifacts import (
    InvestigationStructure,
    RelevantSymbol,
)
from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.structure.evidence_graph import (
    EvidenceGraphEdge,
    EvidenceGraphNode,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest

SERVICE_SCHEMA_VERSION = 1
REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class InvestigationOperationKind(StrEnum):
    """Lifecycle mutations admitted by the public service."""

    START = "start"
    RESUME = "resume"
    CONTINUE = "continue"
    RESPOND = "respond"


class InvestigationOperationState(StrEnum):
    """Durable completion state of one idempotent lifecycle request."""

    PENDING = "pending"
    COMPLETED = "completed"


class ArtifactAvailability(StrEnum):
    """Why a requested high-level artifact is or is not present."""

    AVAILABLE = "available"
    NOT_READY = "not_ready"
    NOT_PRODUCED = "not_produced"


class _ServiceModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class StartInvestigationRequest(_ServiceModel):
    """Trusted start request with all workflow budgets explicit and bounded."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    repo_path: Path
    objective: str = Field(min_length=1, max_length=16_384)
    max_tool_calls_per_step: int = Field(
        default=DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        ge=1,
        le=1_000,
    )
    max_explorer_calls_per_step: int = Field(
        default=DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
        ge=1,
        le=1_000,
    )
    max_no_progress_rounds: int = Field(
        default=DEFAULT_MAX_NO_PROGRESS_ROUNDS,
        ge=1,
        le=100,
    )
    max_plan_revisions: int = Field(
        default=DEFAULT_MAX_PLAN_REVISIONS,
        ge=1,
        le=100,
    )
    context_budget_tokens: int = Field(
        default=DEFAULT_CONTEXT_BUDGET_TOKENS,
        ge=1,
        le=10_000_000,
    )
    max_attempt_cost_usd: str | None = None
    human_review_policy: HumanReviewPolicy = HumanReviewPolicy.NEVER
    memory_retrieval_enabled: bool = False
    max_repository_memory_results: int = Field(
        default=DEFAULT_MAX_REPOSITORY_MEMORY_RESULTS,
        ge=1,
        le=10,
    )
    max_episodic_memory_results: int = Field(
        default=DEFAULT_MAX_EPISODIC_MEMORY_RESULTS,
        ge=1,
        le=10,
    )

    @field_validator("max_attempt_cost_usd", mode="before")
    @classmethod
    def require_canonical_cost_budget(cls, value: object) -> object:
        if value is None:
            return None
        try:
            return validate_cost_budget_usd(value)
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error


class ResumeInvestigationRequest(_ServiceModel):
    """Idempotent resume or exact Human-review response request."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    thread_id: str = Field(min_length=1, max_length=256)
    interrupt_id: str | None = Field(default=None, min_length=1, max_length=256)
    decision: HumanDecision | None = None

    @model_validator(mode="after")
    def require_complete_response_pair(self) -> "ResumeInvestigationRequest":
        if (self.interrupt_id is None) != (self.decision is None):
            raise ValueError("interrupt_id and decision must be supplied together")
        return self

    @property
    def operation_kind(self) -> InvestigationOperationKind:
        return (
            InvestigationOperationKind.RESUME
            if self.decision is None
            else InvestigationOperationKind.RESPOND
        )


class ContinueInvestigationRequest(_ServiceModel):
    """Idempotent new Human turn for one completed checkpoint thread."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    thread_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=16_384)

    @property
    def operation_kind(self) -> InvestigationOperationKind:
        return InvestigationOperationKind.CONTINUE


class ThreadArtifactRequest(_ServiceModel):
    """Read-only request for one checkpoint-bound public artifact."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str = Field(min_length=1, max_length=256)


class ExpandSymbolRequest(ThreadArtifactRequest):
    """Expand exactly one symbol already present in the bounded result graph."""

    entity_id: Sha256Digest | None = None
    canonical_name: str | None = Field(default=None, min_length=1, max_length=2_048)
    max_edges: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def require_exactly_one_selector(self) -> "ExpandSymbolRequest":
        if (self.entity_id is None) == (self.canonical_name is None):
            raise ValueError("exactly one symbol selector is required")
        return self


class InvestigationOperationResult(_ServiceModel):
    """One lifecycle attempt or safely recovered checkpoint state."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    thread_id: str = Field(min_length=1, max_length=256)
    execution_state: ThreadExecutionState
    run_id: str | None = Field(default=None, min_length=1, max_length=256)
    termination_status: TerminationStatus | None = None
    step_stop_reason: str | None = None
    next_nodes: tuple[str, ...] = Field(default=(), max_length=32)
    final_result: FinalResult | None = None
    pending_review: HumanReviewRequest | None = None
    interrupt_id: str | None = Field(default=None, min_length=1, max_length=256)
    usage_cost_report: AttemptUsageCostReport | None = None
    idempotent_replay: bool = False
    recovered_from_checkpoint: bool = False

    @model_validator(mode="after")
    def require_consistent_state(self) -> "InvestigationOperationResult":
        if self.execution_state is ThreadExecutionState.COMPLETED:
            if self.final_result is None or self.termination_status is None:
                raise ValueError("completed operation requires a final result")
            if self.next_nodes or self.pending_review is not None:
                raise ValueError("completed operation cannot remain pending")
        elif self.execution_state is ThreadExecutionState.INTERRUPTED:
            if (
                self.pending_review is None
                or self.interrupt_id is None
                or not self.next_nodes
            ):
                raise ValueError("interrupted operation requires a pending review")
            if self.final_result is not None or self.termination_status is not None:
                raise ValueError("interrupted operation cannot be terminal")
        elif self.execution_state is ThreadExecutionState.RESUMABLE:
            if not self.next_nodes:
                raise ValueError("resumable operation requires pending nodes")
            if self.final_result is not None or self.pending_review is not None:
                raise ValueError("resumable operation cannot publish a result")
        return self


class InvestigationOperationSnapshot(_ServiceModel):
    """Read-only durable identity and result of one lifecycle request."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    kind: InvestigationOperationKind
    thread_id: str = Field(min_length=1, max_length=256)
    state: InvestigationOperationState
    reserved_at_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    result: InvestigationOperationResult | None = None

    @model_validator(mode="after")
    def require_consistent_operation_state(self) -> "InvestigationOperationSnapshot":
        completed = self.state is InvestigationOperationState.COMPLETED
        if completed != (self.result is not None):
            raise ValueError("operation state and result disagree")
        if self.result is not None and (
            self.result.request_id != self.request_id
            or self.result.thread_id != self.thread_id
        ):
            raise ValueError("operation result identity does not reconcile")
        return self


class LocalizationResultPayload(_ServiceModel):
    """Localization-focused result without embedding the complete graph."""

    relevant_files: tuple[str, ...]
    relevant_symbols: tuple[str, ...]
    evidence: tuple[str, ...]
    summary: str
    unresolved_questions: tuple[str, ...]
    structure_artifact_id: Sha256Digest | None = None


class LocalizationResultView(_ServiceModel):
    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str
    execution_state: ThreadExecutionState
    availability: ArtifactAvailability
    result: LocalizationResultPayload | None = None

    @model_validator(mode="after")
    def reconcile_availability(self) -> "LocalizationResultView":
        if (self.availability is ArtifactAvailability.AVAILABLE) != (
            self.result is not None
        ):
            raise ValueError("localization availability and result disagree")
        return self


class CodeGraphView(_ServiceModel):
    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str
    execution_state: ThreadExecutionState
    availability: ArtifactAvailability
    structure: InvestigationStructure | None = None

    @model_validator(mode="after")
    def reconcile_availability(self) -> "CodeGraphView":
        if (self.availability is ArtifactAvailability.AVAILABLE) != (
            self.structure is not None
        ):
            raise ValueError("graph availability and structure disagree")
        return self


class ContextPackView(_ServiceModel):
    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str
    execution_state: ThreadExecutionState
    availability: ArtifactAvailability
    context_pack: ContextPack | None = None
    memory_recall_report: MemoryRecallReport | None = None

    @model_validator(mode="after")
    def reconcile_availability(self) -> "ContextPackView":
        if (self.availability is ArtifactAvailability.AVAILABLE) != (
            self.context_pack is not None
        ):
            raise ValueError("context availability and pack disagree")
        return self


class EvaluatorReportView(_ServiceModel):
    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str
    execution_state: ThreadExecutionState
    availability: ArtifactAvailability
    report: EvaluationReport | None = None
    plan_revision_count: int = Field(ge=0)
    max_plan_revisions: int = Field(ge=1)

    @model_validator(mode="after")
    def reconcile_availability(self) -> "EvaluatorReportView":
        if (self.availability is ArtifactAvailability.AVAILABLE) != (
            self.report is not None
        ):
            raise ValueError("evaluator availability and report disagree")
        return self


class ExpandedSymbolView(_ServiceModel):
    """Bounded one-hop expansion inside an existing task-specific graph."""

    schema_version: Literal[1] = SERVICE_SCHEMA_VERSION
    thread_id: str
    graph_id: Sha256Digest
    node: EvidenceGraphNode
    relevant_binding: RelevantSymbol | None = None
    incoming_edges: tuple[EvidenceGraphEdge, ...] = Field(default=(), max_length=100)
    outgoing_edges: tuple[EvidenceGraphEdge, ...] = Field(default=(), max_length=100)
    neighbor_nodes: tuple[EvidenceGraphNode, ...] = Field(default=(), max_length=200)
    truncated: bool = False

    @model_validator(mode="after")
    def require_closed_one_hop_expansion(self) -> "ExpandedSymbolView":
        incident = (*self.incoming_edges, *self.outgoing_edges)
        if len({edge.edge_id for edge in incident}) != len(incident):
            raise ValueError("symbol expansion contains duplicate edges")
        if any(
            edge.target_entity_id != self.node.entity_id for edge in self.incoming_edges
        ):
            raise ValueError("incoming edge direction is invalid")
        if any(
            edge.source_entity_id != self.node.entity_id for edge in self.outgoing_edges
        ):
            raise ValueError("outgoing edge direction is invalid")
        expected_neighbors = {edge.source_entity_id for edge in self.incoming_edges} | {
            edge.target_entity_id for edge in self.outgoing_edges
        }
        if {node.entity_id for node in self.neighbor_nodes} != expected_neighbors:
            raise ValueError("symbol expansion neighbors do not close selected edges")
        if self.relevant_binding is not None and (
            self.relevant_binding.entity_id != self.node.entity_id
        ):
            raise ValueError("relevant binding does not match expanded node")
        return self
