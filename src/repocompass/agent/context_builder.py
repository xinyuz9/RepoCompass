"""Deterministic hard-budget construction of traceable active context."""

import json
from collections import defaultdict
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.context import (
    MAX_CONTEXT_ITEM_BYTES,
    MAX_CONTEXT_ITEMS,
    ContextItem,
    ContextPack,
    ContextSection,
    ContextSelectionDecision,
    ContextSelectionOutcome,
    ContextSelectionReason,
    ContextSourceKind,
    VerificationStatus,
    create_context_build_report,
    create_context_item,
    create_context_pack,
    create_context_source,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER

CONTEXT_BUILDER_SCHEMA_VERSION = 1
DEFAULT_CONTEXT_BUDGET_TOKENS = 32_768
_TRUNCATION_MARKER = "\n[TRUNCATED: full content retained in raw artifact]"


class ContextBuilderError(RuntimeError):
    """Base failure raised before an unsafe or untraceable pack is produced."""


class ContextCandidateError(ContextBuilderError):
    """The supplied candidate collection has no canonical interpretation."""


class ContextBudgetError(ContextBuilderError):
    """Mandatory context cannot fit inside the configured hard budget."""


class _BuilderContractModel(CheckpointSafeModel):
    """Strict immutable base for builder policy values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ContextSectionReservation(_BuilderContractModel):
    """Best-effort capacity protected for one active-context section."""

    schema_version: Literal[1] = CONTEXT_BUILDER_SCHEMA_VERSION
    section: ContextSection
    reserved_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)


class ContextBudgetPolicy(_BuilderContractModel):
    """Versioned total budget, section reservations, and compression policy."""

    schema_version: Literal[1] = CONTEXT_BUILDER_SCHEMA_VERSION
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    section_reservations: tuple[ContextSectionReservation, ...] = Field(
        default=(),
        max_length=len(ContextSection),
    )
    compression_enabled: bool = False
    min_compressed_content_bytes: int = Field(
        default=64,
        ge=1,
        le=MAX_CONTEXT_ITEM_BYTES,
    )

    @model_validator(mode="after")
    def reconcile_section_reservations(self) -> "ContextBudgetPolicy":
        sections = [reservation.section for reservation in self.section_reservations]
        if len(sections) != len(set(sections)):
            raise ValueError("context reservation sections must be unique")
        if (
            sum(
                reservation.reserved_tokens for reservation in self.section_reservations
            )
            > self.budget_tokens
        ):
            raise ValueError("context reservations must not exceed the total budget")
        return self


_SECTION_ORDER = {
    ContextSection.OBJECTIVE: 0,
    ContextSection.PLAN: 1,
    ContextSection.FEEDBACK: 2,
    ContextSection.EVIDENCE: 3,
    ContextSection.MEMORY: 4,
    ContextSection.HISTORY: 5,
}
_VERIFICATION_ORDER = {
    VerificationStatus.VERIFIED: 0,
    VerificationStatus.HUMAN_APPROVED: 1,
    VerificationStatus.INFERRED: 2,
    VerificationStatus.UNVERIFIED: 3,
    VerificationStatus.REJECTED: 4,
}


def estimate_context_item_tokens(item: ContextItem) -> int:
    """Return a conservative provider-neutral upper bound for rendered tokens.

    The estimate is the exact UTF-8 byte length of the canonical JSONL record.
    Byte-level fallback tokenizers cannot emit more text tokens than input bytes,
    so this deliberately trades some capacity for a deterministic hard boundary.
    Provider-reported usage remains the source of truth for cost accounting.
    """

    if not isinstance(item, ContextItem):
        raise TypeError("item must be a ContextItem")
    return len(_render_context_item(item).encode("utf-8"))


def render_context_pack(pack: ContextPack) -> str:
    """Render selected items as stable JSONL with joinable provenance metadata."""

    if not isinstance(pack, ContextPack):
        raise TypeError("pack must be a ContextPack")
    rendered = "".join(_render_context_item(item) for item in pack.items)
    if len(rendered.encode("utf-8")) != pack.estimated_tokens:
        raise ContextCandidateError(
            "context pack token estimate does not match canonical rendering"
        )
    return rendered


def build_active_context(
    *,
    candidates: tuple[ContextItem, ...],
    policy: ContextBudgetPolicy,
) -> ContextPack:
    """Select, structure, and optionally truncate one active-context pack.

    Selection is independent of input ordering. Objective and pinned items are
    mandatory. Every unique input item receives an auditable decision; a
    traceable truncated derivative receives its own ``COMPRESSED_FIT`` decision.
    """

    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be a tuple of ContextItem values")
    if not isinstance(policy, ContextBudgetPolicy):
        raise TypeError("policy must be a ContextBudgetPolicy")
    if not candidates:
        raise ContextCandidateError("active context requires at least one candidate")
    if len(candidates) > MAX_CONTEXT_ITEMS:
        raise ContextCandidateError("active context has too many candidates")
    if not all(isinstance(item, ContextItem) for item in candidates):
        raise TypeError("candidates must contain only ContextItem values")

    candidate_ids = [item.item_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ContextCandidateError("candidate item IDs must be unique")

    ordered = tuple(
        sorted(
            (_normalize_token_estimate(item) for item in candidates),
            key=_selection_rank,
        )
    )
    unique_candidates, duplicate_ids = _deduplicate_candidates(ordered)
    mandatory = tuple(item for item in unique_candidates if _is_mandatory(item))
    mandatory_tokens = sum(item.estimated_tokens for item in mandatory)
    if mandatory_tokens > policy.budget_tokens:
        raise ContextBudgetError(
            "mandatory context exceeds the configured hard token budget"
        )

    selected: dict[str, ContextItem] = {}
    selection_reasons: dict[str, ContextSelectionReason] = {}
    remaining_tokens = policy.budget_tokens
    mandatory_tokens_by_section: dict[ContextSection, int] = defaultdict(int)
    for item in mandatory:
        selected[item.item_id] = item
        selection_reasons[item.item_id] = _selection_reason(item)
        remaining_tokens -= item.estimated_tokens
        mandatory_tokens_by_section[item.section] += item.estimated_tokens

    for reservation in sorted(
        policy.section_reservations,
        key=lambda value: _SECTION_ORDER[value.section],
    ):
        section_remaining = max(
            0,
            reservation.reserved_tokens
            - mandatory_tokens_by_section[reservation.section],
        )
        for item in unique_candidates:
            if (
                item.item_id in selected
                or item.section is not reservation.section
                or _is_mandatory(item)
            ):
                continue
            if (
                item.estimated_tokens <= section_remaining
                and item.estimated_tokens <= remaining_tokens
            ):
                selected[item.item_id] = item
                selection_reasons[item.item_id] = _selection_reason(item)
                section_remaining -= item.estimated_tokens
                remaining_tokens -= item.estimated_tokens

    compressed_by_original: dict[str, ContextItem] = {}
    for item in unique_candidates:
        if item.item_id in selected or _is_mandatory(item):
            continue
        if item.estimated_tokens <= remaining_tokens:
            selected[item.item_id] = item
            selection_reasons[item.item_id] = _selection_reason(item)
            remaining_tokens -= item.estimated_tokens
            continue
        if (
            not policy.compression_enabled
            or remaining_tokens < 1
            or len(ordered) + len(compressed_by_original) >= MAX_CONTEXT_ITEMS
        ):
            continue
        compressed = _compress_to_budget(
            item,
            budget_tokens=remaining_tokens,
            min_content_bytes=policy.min_compressed_content_bytes,
        )
        if compressed is None or compressed.item_id in candidate_ids:
            continue
        compressed_by_original[item.item_id] = compressed
        selected[compressed.item_id] = compressed
        selection_reasons[compressed.item_id] = ContextSelectionReason.COMPRESSED_FIT
        remaining_tokens -= compressed.estimated_tokens

    selected_items = tuple(sorted(selected.values(), key=_structure_rank))
    selected_positions = {
        item.item_id: position for position, item in enumerate(selected_items)
    }
    decisions: list[ContextSelectionDecision] = []
    for item in ordered:
        if item.item_id in duplicate_ids:
            decisions.append(_excluded_decision(item, ContextSelectionReason.DUPLICATE))
            continue
        if item.item_id in selected:
            decisions.append(
                _selected_decision(
                    item,
                    reason=selection_reasons[item.item_id],
                    position=selected_positions[item.item_id],
                )
            )
        else:
            reason = (
                ContextSelectionReason.BUDGET_EXCEEDED
                if item.estimated_tokens > policy.budget_tokens
                else ContextSelectionReason.LOWER_PRIORITY
            )
            decisions.append(_excluded_decision(item, reason))
        compressed = compressed_by_original.get(item.item_id)
        if compressed is not None:
            decisions.append(
                _selected_decision(
                    compressed,
                    reason=ContextSelectionReason.COMPRESSED_FIT,
                    position=selected_positions[compressed.item_id],
                )
            )

    report = create_context_build_report(
        budget_tokens=policy.budget_tokens,
        decisions=tuple(decisions),
    )
    pack = create_context_pack(report=report, items=selected_items)
    render_context_pack(pack)
    return pack


def _render_context_item(item: ContextItem) -> str:
    source = item.source.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    if source.get("memory") is None:
        source.pop("memory", None)
    payload = {
        "content": item.content,
        "item_id": item.item_id,
        "pinned": item.pinned,
        "section": item.section.value,
        "source": source,
        "verification_status": item.verification_status.value,
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _normalize_token_estimate(item: ContextItem) -> ContextItem:
    return create_context_item(
        source=item.source,
        section=item.section,
        verification_status=item.verification_status,
        content=item.content,
        estimated_tokens=estimate_context_item_tokens(item),
        priority=item.priority,
        pinned=item.pinned,
    )


def _deduplicate_candidates(
    ordered: tuple[ContextItem, ...],
) -> tuple[tuple[ContextItem, ...], frozenset[str]]:
    seen: set[tuple[str, str, str | None, str | None]] = set()
    unique: list[ContextItem] = []
    duplicate_ids: set[str] = set()
    for item in ordered:
        location = item.source.location
        key = (
            item.content_digest,
            item.source.kind.value,
            item.source.raw_artifact_id,
            None if location is None else location.model_dump_json(),
        )
        if key in seen:
            duplicate_ids.add(item.item_id)
        else:
            seen.add(key)
            unique.append(item)
    return tuple(unique), frozenset(duplicate_ids)


def _is_mandatory(item: ContextItem) -> bool:
    return item.pinned or item.source.kind in {
        ContextSourceKind.OBJECTIVE,
        ContextSourceKind.EVALUATOR_REQUEST,
    }


def _selection_reason(item: ContextItem) -> ContextSelectionReason:
    if item.source.kind is ContextSourceKind.OBJECTIVE:
        return ContextSelectionReason.REQUIRED_OBJECTIVE
    if item.pinned:
        return ContextSelectionReason.PINNED
    if item.source.kind is ContextSourceKind.EVALUATOR_REQUEST:
        return ContextSelectionReason.EVALUATOR_REQUEST
    if item.verification_status is VerificationStatus.VERIFIED:
        return ContextSelectionReason.VERIFIED_EVIDENCE
    return ContextSelectionReason.RELEVANCE


def _selection_rank(item: ContextItem) -> tuple[int, int, int, int, str]:
    if item.source.kind is ContextSourceKind.OBJECTIVE:
        source_rank = 0
    elif item.pinned:
        source_rank = 1
    elif item.source.kind is ContextSourceKind.EVALUATOR_REQUEST:
        source_rank = 2
    elif item.verification_status is VerificationStatus.VERIFIED:
        source_rank = 3
    else:
        source_rank = 4
    return (
        source_rank,
        _SECTION_ORDER[item.section],
        -item.priority,
        _VERIFICATION_ORDER[item.verification_status],
        item.item_id,
    )


def _structure_rank(item: ContextItem) -> tuple[int, tuple[int, int, int, int, str]]:
    return (_SECTION_ORDER[item.section], _selection_rank(item))


def _selected_decision(
    item: ContextItem,
    *,
    reason: ContextSelectionReason,
    position: int,
) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        item_id=item.item_id,
        outcome=ContextSelectionOutcome.SELECTED,
        reason=reason,
        estimated_tokens=item.estimated_tokens,
        selected_position=position,
    )


def _excluded_decision(
    item: ContextItem,
    reason: ContextSelectionReason,
) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        item_id=item.item_id,
        outcome=ContextSelectionOutcome.EXCLUDED,
        reason=reason,
        estimated_tokens=item.estimated_tokens,
        selected_position=None,
    )


def _compress_to_budget(
    item: ContextItem,
    *,
    budget_tokens: int,
    min_content_bytes: int,
) -> ContextItem | None:
    raw_artifact_id = item.source.raw_artifact_id
    if raw_artifact_id is None or len(item.content.encode("utf-8")) < min_content_bytes:
        return None
    summary_source = create_context_source(
        kind=ContextSourceKind.SUMMARY,
        raw_artifact_id=raw_artifact_id,
        location=item.source.location,
        provenance=item.source.provenance,
    )
    low = 0
    high = max(0, len(item.content) - 1)
    best: ContextItem | None = None
    while low <= high:
        midpoint = (low + high) // 2
        prefix = item.content[:midpoint]
        if len(prefix.encode("utf-8")) < min_content_bytes:
            low = midpoint + 1
            continue
        candidate = create_context_item(
            source=summary_source,
            section=item.section,
            verification_status=VerificationStatus.UNVERIFIED,
            content=prefix + _TRUNCATION_MARKER,
            estimated_tokens=1,
            priority=item.priority,
            pinned=False,
        )
        candidate = _normalize_token_estimate(candidate)
        if candidate.estimated_tokens <= budget_tokens:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best
