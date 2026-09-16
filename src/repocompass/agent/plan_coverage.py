"""Evidence-linked review of pending work; never infer coverage from goal approval."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.context import ContextPack, ContextSourceKind, VerificationStatus
from repocompass.agent.planning import InvestigationPlan

if TYPE_CHECKING:
    from repocompass.agent.evaluation import EvidenceExcerpt


class _CoverageModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )


class RequirementCoverage(_CoverageModel):
    requirement_index: int = Field(ge=0)
    # An unresolved requirement can truthfully have no supporting excerpt.
    # Nonempty references are enforced by assess_plan_coverage for approval.
    evidence_indices: list[int] = Field(default_factory=list, max_length=16)
    explanation: str = Field(min_length=1, max_length=768)


class StepCoverage(_CoverageModel):
    step_id: str = Field(min_length=1, max_length=256)
    disposition: Literal["covered", "unresolved"]
    objective_evidence_indices: list[int] = Field(default_factory=list, max_length=16)
    requirements: list[RequirementCoverage] = Field(default_factory=list, max_length=32)
    # Missing diagnostic prose must not discard the entire negative assessment.
    # It remains mandatory for any code-issued coverage approval below.
    explanation: str | None = Field(default=None, min_length=1, max_length=768)


class PlanCoverageDecision(_CoverageModel):
    """Code-issued decision bound to the exact objective, plan and candidate."""

    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_step_ids: list[str] = Field(default_factory=list, max_length=5)
    reason: Literal[
        "all_pending_covered",
        "not_requested",
        "incomplete",
        "invalid_references",
        "evaluation_failed",
    ]


def coverage_identity(
    objective: str, plan: InvestigationPlan, candidate: CandidateResult
) -> str:
    encoded = json.dumps(
        [objective, plan.model_dump(mode="json"), candidate.model_dump(mode="json")],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_evidence_indices(
    pack: ContextPack, excerpts: tuple[EvidenceExcerpt, ...]
) -> frozenset[int]:
    """Only full validated excerpts actually visible to this evaluator count."""
    visible = set()
    expected = {excerpt.evidence_index: excerpt for excerpt in excerpts}
    for item in pack.items:
        if (
            item.source.kind is not ContextSourceKind.SOURCE_EXCERPT
            or item.verification_status is not VerificationStatus.VERIFIED
        ):
            continue
        try:
            content = json.loads(item.content)
            index = content.get("evidence_index") if isinstance(content, dict) else None
            excerpt = expected.get(index) if type(index) is int else None
            location = item.source.location
            if (
                excerpt is not None
                and content == excerpt.model_dump(mode="json")
                and location is not None
                and (
                    location.path,
                    location.line_start,
                    location.line_end,
                    location.symbol,
                )
                == (excerpt.path, excerpt.line_start, excerpt.line_end, excerpt.symbol)
            ):
                visible.add(index)
        except (TypeError, ValueError):
            continue
    return frozenset(visible)


def assess_plan_coverage(
    *,
    objective: str,
    plan: InvestigationPlan,
    candidate: CandidateResult,
    claims: list[StepCoverage],
    selected_indices: frozenset[int],
    evaluation_passed: bool,
) -> PlanCoverageDecision:
    identity = coverage_identity(objective, plan, candidate)

    def denied(reason):
        return PlanCoverageDecision(identity_digest=identity, reason=reason)

    pending = [step for step in plan.steps if step.status == "pending"]
    if not claims or not pending:
        return denied("not_requested")
    if not evaluation_passed:
        return denied("evaluation_failed")
    if len(claims) != len(pending) or {claim.step_id for claim in claims} != {
        step.id for step in pending
    }:
        return denied("incomplete")
    by_id = {claim.step_id: claim for claim in claims}

    def valid_indices(indices):
        return (
            bool(indices)
            and len(indices) == len(set(indices))
            and all(
                type(index) is int
                and 0 <= index < len(candidate.evidence)
                and index in selected_indices
                for index in indices
            )
        )

    for step in pending:
        claim = by_id[step.id]
        if claim.disposition != "covered" or not claim.explanation:
            return denied("incomplete")
        if not valid_indices(claim.objective_evidence_indices):
            return denied("invalid_references")
        if len(claim.requirements) != len(step.expected_evidence) or {
            item.requirement_index for item in claim.requirements
        } != set(range(len(step.expected_evidence))):
            return denied("incomplete")
        if any(not valid_indices(item.evidence_indices) for item in claim.requirements):
            return denied("invalid_references")
    return PlanCoverageDecision(
        identity_digest=identity,
        approved_step_ids=[step.id for step in pending],
        reason="all_pending_covered",
    )


def covers_pending_plan(
    decision: PlanCoverageDecision | None,
    *,
    objective: str | None,
    plan: InvestigationPlan,
    candidate: CandidateResult | None,
) -> bool:
    if decision is None or objective is None or candidate is None:
        return False
    pending = [step.id for step in plan.steps if step.status == "pending"]
    return (
        bool(pending)
        and decision.reason == "all_pending_covered"
        and decision.approved_step_ids == pending
        and decision.identity_digest == coverage_identity(objective, plan, candidate)
    )
