"""Safe, read-only projection of one persisted investigation thread.

The internal LangGraph state contains message history and trusted repository paths.
Neither belongs in application, MCP, or UI contracts.  This module maps a
``StateSnapshot`` to the narrow domain artifacts those adapters are allowed to read.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, TypeVar, cast

from langgraph.types import StateSnapshot
from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.context import ContextPack, MemoryRecallReport
from repocompass.agent.cost import AttemptCostBudgetSummary
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import (
    HumanReviewRequest,
    decode_human_review_interrupt,
)
from repocompass.agent.planning import InvestigationPlan
from repocompass.agent.result import FinalResult, revalidate_final_result
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.structural_artifacts import (
    InvestigationStructure,
    revalidate_investigation_structure,
)

THREAD_VIEW_SCHEMA_VERSION = 1
_TERMINATION_STATUSES = frozenset({"accepted", "exhausted", "cancelled"})


class ThreadExecutionState(StrEnum):
    """Externally useful lifecycle state without exposing graph internals."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    RESUMABLE = "resumable"


class ThreadInvestigationView(CheckpointSafeModel):
    """Validated public projection reconstructed from the latest checkpoint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal[1] = THREAD_VIEW_SCHEMA_VERSION
    thread_id: str = Field(min_length=1, max_length=256)
    execution_state: ThreadExecutionState
    turn_number: int = Field(ge=1)
    objective: str = Field(min_length=1)
    plan: InvestigationPlan | None = None
    plan_revision_count: int = Field(ge=0)
    max_plan_revisions: int = Field(ge=1)
    next_nodes: tuple[str, ...] = Field(default=(), max_length=32)
    termination_status: TerminationStatus | None = None
    step_stop_reason: str | None = None
    candidate_result: CandidateResult | None = None
    evaluation_report: EvaluationReport | None = None
    structure: InvestigationStructure | None = None
    context_pack: ContextPack | None = None
    memory_recall_report: MemoryRecallReport | None = None
    cost_budget: AttemptCostBudgetSummary | None = None
    final_result: FinalResult | None = None
    pending_review: HumanReviewRequest | None = None
    interrupt_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def require_unambiguous_lifecycle_state(self) -> "ThreadInvestigationView":
        if len(self.next_nodes) != len(set(self.next_nodes)):
            raise ValueError("next_nodes must be unique")
        if self.execution_state is ThreadExecutionState.COMPLETED:
            if self.final_result is None or self.termination_status is None:
                raise ValueError("completed thread view requires a terminal result")
            if self.next_nodes or self.pending_review is not None:
                raise ValueError("completed thread view cannot remain pending")
            if self.interrupt_id is not None:
                raise ValueError("completed thread view cannot expose an interrupt")
        elif self.execution_state is ThreadExecutionState.INTERRUPTED:
            if (
                self.pending_review is None
                or self.interrupt_id is None
                or not self.next_nodes
            ):
                raise ValueError("interrupted thread view requires one pending review")
            if self.final_result is not None or self.termination_status is not None:
                raise ValueError("interrupted thread view cannot be terminal")
        elif self.execution_state is ThreadExecutionState.RESUMABLE:
            if not self.next_nodes:
                raise ValueError("resumable thread view requires pending graph nodes")
            if self.pending_review is not None or self.interrupt_id is not None:
                raise ValueError("resumable thread view cannot contain a review")
            if self.final_result is not None or self.termination_status is not None:
                raise ValueError("resumable thread view cannot be terminal")
        else:  # pragma: no cover - strict enum validation owns this branch
            raise ValueError("unknown thread execution state")

        if self.final_result is not None and (
            self.structure != self.final_result.structure
        ):
            raise ValueError("thread structure must match the terminal result")
        return self


_ModelT = TypeVar("_ModelT", bound=CheckpointSafeModel)


def _optional_model(
    values: Mapping[str, object],
    key: str,
    model_type: type[_ModelT],
) -> _ModelT | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, model_type):
        raise TypeError(f"Checkpoint {key} is not a {model_type.__name__}")
    return value


def build_thread_investigation_view(
    *,
    thread_id: str,
    snapshot: StateSnapshot,
) -> ThreadInvestigationView:
    """Build a safe projection without returning messages or repository paths."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")
    if not isinstance(snapshot, StateSnapshot):
        raise TypeError("snapshot must be a StateSnapshot")
    values = snapshot.values
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"Checkpoint thread does not exist: {thread_id.strip()}")

    turn_number = values.get("turn_number")
    objective = values.get("objective")
    plan_revision_count = values.get("plan_revision_count")
    max_plan_revisions = values.get("max_plan_revisions")
    if type(turn_number) is not int or turn_number < 1:
        raise ValueError("Checkpoint turn_number is invalid")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("Checkpoint objective is invalid")
    if type(plan_revision_count) is not int or plan_revision_count < 0:
        raise ValueError("Checkpoint plan_revision_count is invalid")
    if type(max_plan_revisions) is not int or max_plan_revisions < 1:
        raise ValueError("Checkpoint max_plan_revisions is invalid")

    next_nodes = tuple(snapshot.next)
    if any(not isinstance(node, str) or not node for node in next_nodes):
        raise ValueError("Checkpoint next nodes are invalid")

    pending_review = None
    interrupt_id = None
    if snapshot.interrupts:
        if len(snapshot.interrupts) != 1:
            raise ValueError("Checkpoint has an unsupported number of interrupts")
        pending = snapshot.interrupts[0]
        if not isinstance(pending.id, str) or not pending.id.strip():
            raise ValueError("Checkpoint interrupt ID is invalid")
        interrupt_id = pending.id.strip()
        pending_review = decode_human_review_interrupt(pending.value)
        execution_state = ThreadExecutionState.INTERRUPTED
    elif next_nodes:
        execution_state = ThreadExecutionState.RESUMABLE
    else:
        execution_state = ThreadExecutionState.COMPLETED

    final_result = _optional_model(values, "final_result", FinalResult)
    if final_result is not None:
        final_result = revalidate_final_result(final_result)
    structure = _optional_model(
        values,
        "investigation_structure",
        InvestigationStructure,
    )
    if structure is not None:
        structure = revalidate_investigation_structure(structure)
    if final_result is not None:
        if structure is not None and structure != final_result.structure:
            raise ValueError("Checkpoint result and structure disagree")
        structure = final_result.structure

    termination_status = values.get("termination_status")
    if termination_status is not None:
        if termination_status not in _TERMINATION_STATUSES:
            raise ValueError("Checkpoint termination status is invalid")
        termination_status = cast(TerminationStatus, termination_status)
    step_stop_reason = values.get("step_stop_reason")
    if step_stop_reason is not None and not isinstance(step_stop_reason, str):
        raise TypeError("Checkpoint step stop reason is invalid")

    return ThreadInvestigationView(
        thread_id=thread_id.strip(),
        execution_state=execution_state,
        turn_number=turn_number,
        objective=objective.strip(),
        plan=_optional_model(values, "investigation_plan", InvestigationPlan),
        plan_revision_count=plan_revision_count,
        max_plan_revisions=max_plan_revisions,
        next_nodes=next_nodes,
        termination_status=termination_status,
        step_stop_reason=step_stop_reason,
        candidate_result=_optional_model(values, "candidate_result", CandidateResult),
        evaluation_report=_optional_model(
            values,
            "evaluation_report",
            EvaluationReport,
        ),
        structure=structure,
        context_pack=_optional_model(values, "active_context_pack", ContextPack),
        memory_recall_report=_optional_model(
            values,
            "memory_recall_report",
            MemoryRecallReport,
        ),
        cost_budget=_optional_model(
            values,
            "cost_budget",
            AttemptCostBudgetSummary,
        ),
        final_result=final_result,
        pending_review=pending_review,
        interrupt_id=interrupt_id,
    )
