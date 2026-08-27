"""Deterministic memory on/off ablation using production context selection."""

import hashlib
import json
from decimal import Decimal, DecimalException, localcontext
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.context import (
    MAX_CONTEXT_ITEMS,
    ContextItem,
    ContextMemoryFreshness,
    ContextSection,
    ContextSourceKind,
    MemoryRecallReport,
    VerificationStatus,
)
from repocompass.agent.context_builder import ContextBudgetPolicy, build_active_context
from repocompass.agent.cost import (
    COST_DISCLAIMER,
    CostAvailability,
    CostUnavailableReason,
    PricingTable,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.usage import PROVIDER_PATTERN
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    Sha256Digest,
)

MEMORY_ABLATION_SCHEMA_VERSION = 1
PARTS_PER_MILLION = 1_000_000


class _MemoryAblationModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class MemoryAblationVariant(StrEnum):
    """Closed memory comparison variants in canonical report order."""

    MEMORY_OFF = "memory_off"
    MEMORY_ON = "memory_on"


class MemoryAblationCase(_MemoryAblationModel):
    """Traceable base context plus one real body-free recall result."""

    schema_version: Literal[1] = MEMORY_ABLATION_SCHEMA_VERSION
    case_id: Sha256Digest
    name: str = Field(pattern=IDENTIFIER_PATTERN)
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    base_candidates: tuple[ContextItem, ...] = Field(
        min_length=1,
        max_length=MAX_CONTEXT_ITEMS,
    )
    memory_candidates: tuple[ContextItem, ...] = Field(
        min_length=1,
        max_length=MAX_CONTEXT_ITEMS,
    )
    recall_report: MemoryRecallReport
    required_evidence_item_ids: tuple[Sha256Digest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_traceable_case(self) -> "MemoryAblationCase":
        if len(self.base_candidates) + len(self.memory_candidates) > MAX_CONTEXT_ITEMS:
            raise ValueError("memory ablation candidate count exceeds context limit")
        base_ids = tuple(item.item_id for item in self.base_candidates)
        memory_item_ids = tuple(item.item_id for item in self.memory_candidates)
        if base_ids != tuple(sorted(base_ids)):
            raise ValueError("base candidates must use canonical item-ID order")
        if memory_item_ids != tuple(sorted(memory_item_ids)):
            raise ValueError("memory candidates must use canonical item-ID order")
        all_ids = (*base_ids, *memory_item_ids)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("memory ablation candidate IDs must be unique")
        if any(
            item.source.kind is ContextSourceKind.MEMORY
            for item in self.base_candidates
        ):
            raise ValueError("base candidates cannot contain recalled memory")
        if any(
            item.source.kind is not ContextSourceKind.MEMORY
            or item.section is not ContextSection.MEMORY
            or item.source.memory is None
            or item.source.memory.freshness is ContextMemoryFreshness.STALE
            for item in self.memory_candidates
        ):
            raise ValueError("memory candidates must be usable production memory items")

        required_ids = self.required_evidence_item_ids
        if required_ids != tuple(sorted(set(required_ids))):
            raise ValueError("required evidence IDs must be unique and sorted")
        base_by_id = {item.item_id: item for item in self.base_candidates}
        if not set(required_ids).issubset(base_by_id):
            raise ValueError("required evidence must belong to the base context")
        for item_id in required_ids:
            item = base_by_id[item_id]
            if (
                item.source.kind is not ContextSourceKind.SOURCE_EXCERPT
                or item.source.raw_artifact_id is None
                or item.source.location is None
                or item.verification_status is not VerificationStatus.VERIFIED
            ):
                raise ValueError(
                    "required evidence must be a verified traceable source excerpt"
                )

        if set(memory_item_ids) != set(self.recall_report.context_item_ids):
            raise ValueError("memory candidates must match recall context item IDs")
        decisions_by_item = {
            decision.context_item_id: decision
            for decision in self.recall_report.decisions
            if decision.context_item_id is not None
        }
        memory_ids: set[str] = set()
        for item in self.memory_candidates:
            metadata = item.source.memory
            assert metadata is not None
            decision = decisions_by_item.get(item.item_id)
            if decision is None or (
                metadata.retrieval_id != decision.retrieval_id
                or metadata.memory_id != decision.memory_id
                or metadata.scope is not decision.scope
                or metadata.freshness is not decision.freshness
                or metadata.staleness_reason is not decision.staleness_reason
                or metadata.query_digest != self.recall_report.query_digest
            ):
                raise ValueError("memory candidate provenance disagrees with recall")
            if metadata.memory_id in memory_ids:
                raise ValueError("memory candidates must have unique memory IDs")
            memory_ids.add(metadata.memory_id)

        expected_id = _case_id(
            name=self.name,
            budget_tokens=self.budget_tokens,
            base_candidates=self.base_candidates,
            memory_candidates=self.memory_candidates,
            recall_report=self.recall_report,
            required_evidence_item_ids=required_ids,
        )
        if self.case_id != expected_id:
            raise ValueError("case_id does not match memory ablation contents")
        return self


class MemoryAblationVariantMetrics(_MemoryAblationModel):
    """One production Context Builder result with explicit recall accounting."""

    schema_version: Literal[1] = MEMORY_ABLATION_SCHEMA_VERSION
    variant: MemoryAblationVariant
    context_id: Sha256Digest
    build_id: Sha256Digest
    candidate_count: int = Field(ge=1, le=MAX_CONTEXT_ITEMS)
    selected_count: int = Field(ge=0, le=MAX_CONTEXT_ITEMS)
    selected_item_ids: tuple[Sha256Digest, ...]
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    context_tokens: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    required_evidence_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    retained_required_evidence_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    required_evidence_retention_ppm: int = Field(ge=0, le=PARTS_PER_MILLION)
    retrieval_result_count: int = Field(ge=0, le=100)
    repository_result_count: int = Field(ge=0, le=50)
    episodic_result_count: int = Field(ge=0, le=50)
    usable_memory_count: int = Field(ge=0, le=100)
    stale_excluded_count: int = Field(ge=0, le=100)
    selected_memory_count: int = Field(ge=0, le=100)
    memory_selection_ppm: int | None = Field(
        default=None,
        ge=0,
        le=PARTS_PER_MILLION,
    )
    cost_availability: CostAvailability
    estimated_input_cost_usd: str | None = None
    cost_unavailable_reason: CostUnavailableReason | None = None
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_input_cost_usd", mode="before")
    @classmethod
    def require_canonical_cost(cls, value: object) -> object:
        if value is not None and not _is_canonical_non_negative_decimal(value):
            raise ValueError("estimated input cost must be a canonical decimal")
        return value

    @model_validator(mode="after")
    def reconcile_metrics(self) -> "MemoryAblationVariantMetrics":
        if self.selected_count != len(self.selected_item_ids):
            raise ValueError("selected_count must match selected_item_ids")
        if len(self.selected_item_ids) != len(set(self.selected_item_ids)):
            raise ValueError("selected item IDs must be unique")
        if self.selected_count > self.candidate_count:
            raise ValueError("selected_count cannot exceed candidate_count")
        if self.context_tokens > self.budget_tokens:
            raise ValueError("memory ablation context exceeds its hard budget")
        if self.retained_required_evidence_count > self.required_evidence_count:
            raise ValueError("retained evidence cannot exceed required evidence")
        if self.required_evidence_retention_ppm != _coverage_ppm(
            self.retained_required_evidence_count,
            self.required_evidence_count,
        ):
            raise ValueError("required evidence retention is inconsistent")
        if self.retrieval_result_count != (
            self.repository_result_count + self.episodic_result_count
        ):
            raise ValueError("retrieval result counts do not reconcile")
        if self.usable_memory_count > self.retrieval_result_count:
            raise ValueError("usable memory cannot exceed retrieval results")
        if self.stale_excluded_count != (
            self.retrieval_result_count - self.usable_memory_count
        ):
            raise ValueError("stale exclusion count does not reconcile")
        if self.selected_memory_count > self.usable_memory_count:
            raise ValueError("selected memory cannot exceed usable memory")
        if self.variant is MemoryAblationVariant.MEMORY_OFF:
            if (
                any(
                    (
                        self.retrieval_result_count,
                        self.repository_result_count,
                        self.episodic_result_count,
                        self.usable_memory_count,
                        self.stale_excluded_count,
                        self.selected_memory_count,
                    )
                )
                or self.memory_selection_ppm is not None
            ):
                raise ValueError("memory-off metrics cannot claim recall activity")
        else:
            expected_selection = (
                None
                if self.usable_memory_count == 0
                else _coverage_ppm(
                    self.selected_memory_count,
                    self.usable_memory_count,
                )
            )
            if self.memory_selection_ppm != expected_selection:
                raise ValueError("memory selection coverage is inconsistent")
        if self.cost_availability is CostAvailability.ESTIMATED:
            if (
                self.estimated_input_cost_usd is None
                or self.cost_unavailable_reason is not None
            ):
                raise ValueError("estimated cost requires an amount and no reason")
        elif (
            self.estimated_input_cost_usd is not None
            or self.cost_unavailable_reason is None
        ):
            raise ValueError("unavailable cost requires a reason and no amount")
        return self


class MemoryAblationReport(_MemoryAblationModel):
    """Content-addressed deterministic memory on/off comparison artifact."""

    schema_version: Literal[1] = MEMORY_ABLATION_SCHEMA_VERSION
    report_id: Sha256Digest
    case_id: Sha256Digest
    case_name: str = Field(pattern=IDENTIFIER_PATTERN)
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    pricing_table_id: Sha256Digest
    pricing_version: str = Field(pattern=IDENTIFIER_PATTERN)
    pricing_effective_from: str
    variants: tuple[MemoryAblationVariantMetrics, MemoryAblationVariantMetrics]
    retrieval_result_delta: int = Field(ge=0, le=100)
    usable_memory_delta: int = Field(ge=0, le=100)
    selected_memory_delta: int = Field(ge=0, le=100)
    context_token_delta: int = Field(
        ge=-MAX_SAFE_JSON_INTEGER,
        le=MAX_SAFE_JSON_INTEGER,
    )
    required_evidence_retention_delta_ppm: int = Field(
        ge=-PARTS_PER_MILLION,
        le=PARTS_PER_MILLION,
    )
    estimated_input_cost_delta_usd: str | None = None
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_input_cost_delta_usd", mode="before")
    @classmethod
    def require_canonical_delta(cls, value: object) -> object:
        if value is not None and not _is_canonical_signed_decimal(value):
            raise ValueError("estimated cost delta must be a canonical decimal")
        return value

    @model_validator(mode="after")
    def require_content_addressed_report(self) -> "MemoryAblationReport":
        memory_off, memory_on = self.variants
        if (
            memory_off.variant is not MemoryAblationVariant.MEMORY_OFF
            or memory_on.variant is not MemoryAblationVariant.MEMORY_ON
        ):
            raise ValueError("memory variants must use canonical comparison order")
        if self.retrieval_result_delta != (
            memory_on.retrieval_result_count - memory_off.retrieval_result_count
        ):
            raise ValueError("retrieval result delta is inconsistent")
        if self.usable_memory_delta != (
            memory_on.usable_memory_count - memory_off.usable_memory_count
        ):
            raise ValueError("usable memory delta is inconsistent")
        if self.selected_memory_delta != (
            memory_on.selected_memory_count - memory_off.selected_memory_count
        ):
            raise ValueError("selected memory delta is inconsistent")
        if self.context_token_delta != (
            memory_on.context_tokens - memory_off.context_tokens
        ):
            raise ValueError("context token delta is inconsistent")
        if self.required_evidence_retention_delta_ppm != (
            memory_on.required_evidence_retention_ppm
            - memory_off.required_evidence_retention_ppm
        ):
            raise ValueError("required evidence retention delta is inconsistent")
        expected_cost_delta = _cost_delta(memory_off, memory_on)
        if self.estimated_input_cost_delta_usd != expected_cost_delta:
            raise ValueError("estimated input cost delta is inconsistent")
        expected_id = _report_id(
            case_id=self.case_id,
            case_name=self.case_name,
            provider=self.provider,
            model=self.model,
            pricing_table_id=self.pricing_table_id,
            pricing_version=self.pricing_version,
            pricing_effective_from=self.pricing_effective_from,
            variants=self.variants,
            retrieval_result_delta=self.retrieval_result_delta,
            usable_memory_delta=self.usable_memory_delta,
            selected_memory_delta=self.selected_memory_delta,
            context_token_delta=self.context_token_delta,
            required_evidence_retention_delta_ppm=(
                self.required_evidence_retention_delta_ppm
            ),
            estimated_input_cost_delta_usd=self.estimated_input_cost_delta_usd,
        )
        if self.report_id != expected_id:
            raise ValueError("report_id does not match memory ablation report")
        return self


def create_memory_ablation_case(
    *,
    name: str,
    budget_tokens: int,
    base_candidates: tuple[ContextItem, ...],
    memory_candidates: tuple[ContextItem, ...],
    recall_report: MemoryRecallReport,
    required_evidence_item_ids: tuple[str, ...],
) -> MemoryAblationCase:
    """Create one canonical, content-addressed memory ablation case."""

    if not isinstance(base_candidates, tuple) or not all(
        isinstance(item, ContextItem) for item in base_candidates
    ):
        raise TypeError("base_candidates must be a tuple of ContextItem values")
    if not isinstance(memory_candidates, tuple) or not all(
        isinstance(item, ContextItem) for item in memory_candidates
    ):
        raise TypeError("memory_candidates must be a tuple of ContextItem values")
    if not isinstance(recall_report, MemoryRecallReport):
        raise TypeError("recall_report must be a MemoryRecallReport")
    canonical_base = tuple(sorted(base_candidates, key=lambda item: item.item_id))
    canonical_memory = tuple(sorted(memory_candidates, key=lambda item: item.item_id))
    canonical_required = tuple(sorted(set(required_evidence_item_ids)))
    case_id = _case_id(
        name=name,
        budget_tokens=budget_tokens,
        base_candidates=canonical_base,
        memory_candidates=canonical_memory,
        recall_report=recall_report,
        required_evidence_item_ids=canonical_required,
    )
    return MemoryAblationCase(
        case_id=case_id,
        name=name,
        budget_tokens=budget_tokens,
        base_candidates=canonical_base,
        memory_candidates=canonical_memory,
        recall_report=recall_report,
        required_evidence_item_ids=canonical_required,
    )


def run_memory_ablation(
    case: MemoryAblationCase,
    *,
    pricing_table: PricingTable,
    provider: str,
    model: str,
) -> MemoryAblationReport:
    """Compare production context construction with recall disabled and enabled."""

    if not isinstance(case, MemoryAblationCase):
        raise TypeError("case must be a MemoryAblationCase")
    if not isinstance(pricing_table, PricingTable):
        raise TypeError("pricing_table must be a PricingTable")
    price = pricing_table.lookup(provider, model)
    price_rate = None if price is None else price.input_usd_per_million_tokens
    memory_off_pack = build_active_context(
        candidates=case.base_candidates,
        policy=ContextBudgetPolicy(budget_tokens=case.budget_tokens),
    )
    memory_on_candidates = tuple(
        sorted(
            (*case.base_candidates, *case.memory_candidates),
            key=lambda item: item.item_id,
        )
    )
    memory_on_pack = build_active_context(
        candidates=memory_on_candidates,
        policy=ContextBudgetPolicy(budget_tokens=case.budget_tokens),
    )
    memory_off = _variant_metrics(
        case,
        variant=MemoryAblationVariant.MEMORY_OFF,
        selected_items=memory_off_pack.items,
        context_id=memory_off_pack.pack_id,
        build_id=memory_off_pack.build_report.build_id,
        candidate_count=len(case.base_candidates),
        price_rate=price_rate,
    )
    memory_on = _variant_metrics(
        case,
        variant=MemoryAblationVariant.MEMORY_ON,
        selected_items=memory_on_pack.items,
        context_id=memory_on_pack.pack_id,
        build_id=memory_on_pack.build_report.build_id,
        candidate_count=len(memory_on_candidates),
        price_rate=price_rate,
    )
    variants = (memory_off, memory_on)
    retrieval_delta = memory_on.retrieval_result_count
    usable_delta = memory_on.usable_memory_count
    selected_delta = memory_on.selected_memory_count
    token_delta = memory_on.context_tokens - memory_off.context_tokens
    retention_delta = (
        memory_on.required_evidence_retention_ppm
        - memory_off.required_evidence_retention_ppm
    )
    cost_delta = _cost_delta(memory_off, memory_on)
    report_id = _report_id(
        case_id=case.case_id,
        case_name=case.name,
        provider=provider,
        model=model,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        variants=variants,
        retrieval_result_delta=retrieval_delta,
        usable_memory_delta=usable_delta,
        selected_memory_delta=selected_delta,
        context_token_delta=token_delta,
        required_evidence_retention_delta_ppm=retention_delta,
        estimated_input_cost_delta_usd=cost_delta,
    )
    return MemoryAblationReport(
        report_id=report_id,
        case_id=case.case_id,
        case_name=case.name,
        provider=provider,
        model=model,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        variants=variants,
        retrieval_result_delta=retrieval_delta,
        usable_memory_delta=usable_delta,
        selected_memory_delta=selected_delta,
        context_token_delta=token_delta,
        required_evidence_retention_delta_ppm=retention_delta,
        estimated_input_cost_delta_usd=cost_delta,
    )


def render_memory_ablation_summary(report: MemoryAblationReport) -> str:
    """Render a concise comparison only from the immutable report."""

    if not isinstance(report, MemoryAblationReport):
        raise TypeError("report must be a MemoryAblationReport")
    memory_off, memory_on = report.variants
    cost_delta = report.estimated_input_cost_delta_usd or "unavailable"
    return (
        f"Memory ablation {report.case_name}: retrieval results "
        f"{memory_on.retrieval_result_count}; usable {memory_on.usable_memory_count}; "
        f"selected {memory_on.selected_memory_count}; stale excluded "
        f"{memory_on.stale_excluded_count}; context token delta "
        f"{report.context_token_delta:+d}; required evidence retention "
        f"{memory_on.required_evidence_retention_ppm / 10_000:.2f}% "
        f"(delta {report.required_evidence_retention_delta_ppm:+d} ppm); "
        f"estimated input cost delta USD {cost_delta}; estimate, not provider invoice; "
        f"memory-off tokens {memory_off.context_tokens}, memory-on tokens "
        f"{memory_on.context_tokens}"
    )


class MemoryAblationReportSink:
    """Persist one immutable report without overwriting prior evidence."""

    def write(self, report: MemoryAblationReport, destination: Path) -> Path:
        if not isinstance(report, MemoryAblationReport):
            raise TypeError("report must be a MemoryAblationReport")
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2))
            stream.write("\n")
        return resolved


def _variant_metrics(
    case: MemoryAblationCase,
    *,
    variant: MemoryAblationVariant,
    selected_items: tuple[ContextItem, ...],
    context_id: str,
    build_id: str,
    candidate_count: int,
    price_rate: str | None,
) -> MemoryAblationVariantMetrics:
    selected_ids = tuple(item.item_id for item in selected_items)
    retained_required = len(
        set(case.required_evidence_item_ids).intersection(selected_ids)
    )
    if variant is MemoryAblationVariant.MEMORY_OFF:
        retrieval_count = 0
        repository_count = 0
        episodic_count = 0
        usable_count = 0
        stale_count = 0
        selected_memory_count = 0
        selection_ppm = None
    else:
        recall = case.recall_report
        retrieval_count = len(recall.decisions)
        repository_count = recall.repository_result_count
        episodic_count = recall.episodic_result_count
        usable_count = len(case.memory_candidates)
        stale_count = retrieval_count - usable_count
        memory_item_ids = {item.item_id for item in case.memory_candidates}
        selected_memory_count = len(memory_item_ids.intersection(selected_ids))
        selection_ppm = _coverage_ppm(selected_memory_count, usable_count)
    context_tokens = sum(item.estimated_tokens for item in selected_items)
    if price_rate is None:
        cost_availability = CostAvailability.UNAVAILABLE
        estimated_cost = None
        cost_reason = CostUnavailableReason.PRICE_NOT_FOUND
    else:
        cost_availability = CostAvailability.ESTIMATED
        estimated_cost = _input_cost(context_tokens, price_rate)
        cost_reason = None
    return MemoryAblationVariantMetrics(
        variant=variant,
        context_id=context_id,
        build_id=build_id,
        candidate_count=candidate_count,
        selected_count=len(selected_items),
        selected_item_ids=selected_ids,
        budget_tokens=case.budget_tokens,
        context_tokens=context_tokens,
        required_evidence_count=len(case.required_evidence_item_ids),
        retained_required_evidence_count=retained_required,
        required_evidence_retention_ppm=_coverage_ppm(
            retained_required,
            len(case.required_evidence_item_ids),
        ),
        retrieval_result_count=retrieval_count,
        repository_result_count=repository_count,
        episodic_result_count=episodic_count,
        usable_memory_count=usable_count,
        stale_excluded_count=stale_count,
        selected_memory_count=selected_memory_count,
        memory_selection_ppm=selection_ppm,
        cost_availability=cost_availability,
        estimated_input_cost_usd=estimated_cost,
        cost_unavailable_reason=cost_reason,
    )


def _coverage_ppm(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return PARTS_PER_MILLION
    return numerator * PARTS_PER_MILLION // denominator


def _input_cost(tokens: int, rate: str) -> str:
    with localcontext() as context:
        context.prec = 128
        amount = Decimal(tokens) * Decimal(rate) / Decimal(PARTS_PER_MILLION)
    return _non_negative_decimal_string(amount)


def _cost_delta(
    memory_off: MemoryAblationVariantMetrics,
    memory_on: MemoryAblationVariantMetrics,
) -> str | None:
    if (
        memory_off.estimated_input_cost_usd is None
        or memory_on.estimated_input_cost_usd is None
    ):
        return None
    return _signed_decimal_string(
        Decimal(memory_on.estimated_input_cost_usd)
        - Decimal(memory_off.estimated_input_cost_usd)
    )


def _is_canonical_non_negative_decimal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return _non_negative_decimal_string(Decimal(value)) == value
    except (DecimalException, ValueError):
        return False


def _is_canonical_signed_decimal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return _signed_decimal_string(Decimal(value)) == value
    except (DecimalException, ValueError):
        return False


def _non_negative_decimal_string(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("decimal amount must be finite and non-negative")
    return _render_decimal(value)


def _signed_decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("decimal delta must be finite")
    return _render_decimal(value)


def _render_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _case_id(
    *,
    name: str,
    budget_tokens: int,
    base_candidates: tuple[ContextItem, ...],
    memory_candidates: tuple[ContextItem, ...],
    recall_report: MemoryRecallReport,
    required_evidence_item_ids: tuple[str, ...],
) -> str:
    return _digest(
        b"repocompass:memory-ablation-case:v1\0",
        {
            "base_candidates": [
                item.model_dump(mode="json", exclude_computed_fields=True)
                for item in base_candidates
            ],
            "budget_tokens": budget_tokens,
            "memory_candidates": [
                item.model_dump(mode="json", exclude_computed_fields=True)
                for item in memory_candidates
            ],
            "name": name,
            "recall_report": recall_report.model_dump(
                mode="json",
                exclude_computed_fields=True,
            ),
            "required_evidence_item_ids": list(required_evidence_item_ids),
            "schema_version": MEMORY_ABLATION_SCHEMA_VERSION,
        },
    )


def _report_id(
    *,
    case_id: str,
    case_name: str,
    provider: str,
    model: str,
    pricing_table_id: str,
    pricing_version: str,
    pricing_effective_from: str,
    variants: tuple[MemoryAblationVariantMetrics, MemoryAblationVariantMetrics],
    retrieval_result_delta: int,
    usable_memory_delta: int,
    selected_memory_delta: int,
    context_token_delta: int,
    required_evidence_retention_delta_ppm: int,
    estimated_input_cost_delta_usd: str | None,
) -> str:
    return _digest(
        b"repocompass:memory-ablation-report:v1\0",
        {
            "case_id": case_id,
            "case_name": case_name,
            "context_token_delta": context_token_delta,
            "disclaimer": COST_DISCLAIMER,
            "estimated_input_cost_delta_usd": estimated_input_cost_delta_usd,
            "model": model,
            "pricing_effective_from": pricing_effective_from,
            "pricing_table_id": pricing_table_id,
            "pricing_version": pricing_version,
            "provider": provider,
            "required_evidence_retention_delta_ppm": (
                required_evidence_retention_delta_ppm
            ),
            "retrieval_result_delta": retrieval_result_delta,
            "schema_version": MEMORY_ABLATION_SCHEMA_VERSION,
            "selected_memory_delta": selected_memory_delta,
            "usable_memory_delta": usable_memory_delta,
            "variants": [
                item.model_dump(mode="json", exclude_computed_fields=True)
                for item in variants
            ],
        },
    )


def _digest(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(namespace + encoded).hexdigest()
