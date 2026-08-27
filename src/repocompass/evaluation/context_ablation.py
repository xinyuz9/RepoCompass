"""Deterministic Active Context Builder ablation harness."""

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal, DecimalException, localcontext
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.context import (
    ContextItem,
    ContextSection,
    ContextSourceKind,
    VerificationStatus,
)
from repocompass.agent.context_builder import (
    ContextBudgetPolicy,
    build_active_context,
    estimate_context_item_tokens,
)
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

CONTEXT_ABLATION_SCHEMA_VERSION = 1
PARTS_PER_MILLION = 1_000_000


class _AblationModel(CheckpointSafeModel):
    """Strict immutable base for portable benchmark artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class AblationVariant(StrEnum):
    """Closed comparison variants for the context benchmark."""

    ACTIVE_CONTEXT = "active_context"
    ALL_CONTEXT_BASELINE = "all_context_baseline"


class LatencyAvailability(StrEnum):
    """Distinguish an absent observation from a measured zero."""

    NOT_MEASURED = "not_measured"
    OBSERVED = "observed"


class ContextAblationCase(_AblationModel):
    """Content-addressed candidate set and evidence-retention objective."""

    schema_version: Literal[1] = CONTEXT_ABLATION_SCHEMA_VERSION
    case_id: Sha256Digest
    name: str = Field(pattern=IDENTIFIER_PATTERN)
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    candidates: tuple[ContextItem, ...] = Field(min_length=1)
    required_evidence_item_ids: tuple[Sha256Digest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_canonical_traceable_case(self) -> "ContextAblationCase":
        candidate_ids = tuple(item.item_id for item in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ValueError("ablation candidates must use canonical item-ID order")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("ablation candidate item IDs must be unique")
        required_ids = self.required_evidence_item_ids
        if required_ids != tuple(sorted(set(required_ids))):
            raise ValueError("required evidence IDs must be unique and sorted")
        candidates_by_id = {item.item_id: item for item in self.candidates}
        if not set(required_ids).issubset(candidates_by_id):
            raise ValueError("required evidence must belong to the candidate set")
        for item_id in required_ids:
            item = candidates_by_id[item_id]
            if (
                item.source.kind is not ContextSourceKind.SOURCE_EXCERPT
                or item.source.raw_artifact_id is None
                or item.source.location is None
                or item.verification_status is not VerificationStatus.VERIFIED
            ):
                raise ValueError(
                    "required evidence must be a verified traceable source excerpt"
                )
        expected_id = _case_id(
            name=self.name,
            budget_tokens=self.budget_tokens,
            candidates=self.candidates,
            required_evidence_item_ids=required_ids,
        )
        if self.case_id != expected_id:
            raise ValueError("case_id does not match ablation case contents")
        return self


class ContextAblationVariantMetrics(_AblationModel):
    """Bounded aggregate for one context-selection strategy."""

    schema_version: Literal[1] = CONTEXT_ABLATION_SCHEMA_VERSION
    variant: AblationVariant
    context_id: Sha256Digest
    build_id: Sha256Digest | None = None
    candidate_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    selected_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    selected_item_ids: tuple[Sha256Digest, ...]
    budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    selected_tokens: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    overflow_tokens: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    required_evidence_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    retained_required_evidence_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    required_evidence_retention_ppm: int = Field(ge=0, le=PARTS_PER_MILLION)
    evidence_item_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    traceable_evidence_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    evidence_provenance_coverage_ppm: int = Field(ge=0, le=PARTS_PER_MILLION)
    cost_availability: CostAvailability
    estimated_input_cost_usd: str | None = None
    cost_unavailable_reason: CostUnavailableReason | None = None
    latency_availability: LatencyAvailability
    latency_ms: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_input_cost_usd", mode="before")
    @classmethod
    def require_canonical_cost(cls, value: object) -> object:
        if value is not None and not _is_canonical_decimal(value):
            raise ValueError("estimated input cost must be a canonical decimal")
        return value

    @model_validator(mode="after")
    def reconcile_metrics(self) -> "ContextAblationVariantMetrics":
        if self.selected_count != len(self.selected_item_ids):
            raise ValueError("selected_count must match selected_item_ids")
        if len(self.selected_item_ids) != len(set(self.selected_item_ids)):
            raise ValueError("selected item IDs must be unique")
        if self.selected_count > self.candidate_count:
            raise ValueError("selected_count cannot exceed candidate_count")
        expected_overflow = max(0, self.selected_tokens - self.budget_tokens)
        if self.overflow_tokens != expected_overflow:
            raise ValueError("overflow_tokens does not match selected token total")
        if self.retained_required_evidence_count > self.required_evidence_count:
            raise ValueError("retained evidence cannot exceed required evidence")
        if self.required_evidence_retention_ppm != _coverage_ppm(
            self.retained_required_evidence_count,
            self.required_evidence_count,
        ):
            raise ValueError("required evidence retention is inconsistent")
        if self.traceable_evidence_count > self.evidence_item_count:
            raise ValueError("traceable evidence cannot exceed selected evidence")
        if self.evidence_provenance_coverage_ppm != _coverage_ppm(
            self.traceable_evidence_count,
            self.evidence_item_count,
        ):
            raise ValueError("evidence provenance coverage is inconsistent")
        if self.variant is AblationVariant.ACTIVE_CONTEXT:
            if self.build_id is None or self.overflow_tokens != 0:
                raise ValueError("active context requires a build ID and no overflow")
        elif self.build_id is not None:
            raise ValueError("all-context baseline must not claim a builder ID")
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
        if self.latency_availability is LatencyAvailability.OBSERVED:
            if self.latency_ms is None:
                raise ValueError("observed latency requires latency_ms")
        elif self.latency_ms is not None:
            raise ValueError("unmeasured latency must not contain latency_ms")
        return self


class ContextAblationReport(_AblationModel):
    """Content-addressed two-variant context benchmark artifact."""

    schema_version: Literal[1] = CONTEXT_ABLATION_SCHEMA_VERSION
    report_id: Sha256Digest
    case_id: Sha256Digest
    case_name: str = Field(pattern=IDENTIFIER_PATTERN)
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    pricing_table_id: Sha256Digest
    pricing_version: str = Field(pattern=IDENTIFIER_PATTERN)
    pricing_effective_from: str
    variants: tuple[ContextAblationVariantMetrics, ContextAblationVariantMetrics]
    token_reduction: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    required_evidence_retention_delta_ppm: int = Field(
        ge=-PARTS_PER_MILLION,
        le=PARTS_PER_MILLION,
    )
    estimated_input_cost_savings_usd: str | None = None
    observed_latency_delta_ms: int | None = Field(
        default=None,
        ge=-MAX_SAFE_JSON_INTEGER,
        le=MAX_SAFE_JSON_INTEGER,
    )
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_input_cost_savings_usd", mode="before")
    @classmethod
    def require_canonical_savings(cls, value: object) -> object:
        if value is not None and not _is_canonical_decimal(value):
            raise ValueError("estimated savings must be a canonical decimal")
        return value

    @model_validator(mode="after")
    def require_content_addressed_report(self) -> "ContextAblationReport":
        active, baseline = self.variants
        if (
            active.variant is not AblationVariant.ACTIVE_CONTEXT
            or baseline.variant is not AblationVariant.ALL_CONTEXT_BASELINE
        ):
            raise ValueError("ablation variants must use canonical comparison order")
        if self.token_reduction != baseline.selected_tokens - active.selected_tokens:
            raise ValueError("token reduction does not match variant metrics")
        expected_retention_delta = (
            active.required_evidence_retention_ppm
            - baseline.required_evidence_retention_ppm
        )
        if self.required_evidence_retention_delta_ppm != expected_retention_delta:
            raise ValueError("retention delta does not match variant metrics")
        expected_savings = _cost_savings(active, baseline)
        if self.estimated_input_cost_savings_usd != expected_savings:
            raise ValueError("cost savings do not match variant metrics")
        expected_latency_delta = _latency_delta(active, baseline)
        if self.observed_latency_delta_ms != expected_latency_delta:
            raise ValueError("latency delta does not match variant metrics")
        expected_id = _report_id(
            case_id=self.case_id,
            case_name=self.case_name,
            provider=self.provider,
            model=self.model,
            pricing_table_id=self.pricing_table_id,
            pricing_version=self.pricing_version,
            pricing_effective_from=self.pricing_effective_from,
            variants=self.variants,
            token_reduction=self.token_reduction,
            required_evidence_retention_delta_ppm=(
                self.required_evidence_retention_delta_ppm
            ),
            estimated_input_cost_savings_usd=(self.estimated_input_cost_savings_usd),
            observed_latency_delta_ms=self.observed_latency_delta_ms,
        )
        if self.report_id != expected_id:
            raise ValueError("report_id does not match ablation report contents")
        return self


def create_context_ablation_case(
    *,
    name: str,
    budget_tokens: int,
    candidates: tuple[ContextItem, ...],
    required_evidence_item_ids: tuple[str, ...],
) -> ContextAblationCase:
    """Create one canonical content-addressed benchmark case."""

    if not isinstance(candidates, tuple) or not all(
        isinstance(item, ContextItem) for item in candidates
    ):
        raise TypeError("candidates must be a tuple of ContextItem values")
    canonical_candidates = tuple(sorted(candidates, key=lambda item: item.item_id))
    canonical_required_ids = tuple(sorted(set(required_evidence_item_ids)))
    case_id = _case_id(
        name=name,
        budget_tokens=budget_tokens,
        candidates=canonical_candidates,
        required_evidence_item_ids=canonical_required_ids,
    )
    return ContextAblationCase(
        case_id=case_id,
        name=name,
        budget_tokens=budget_tokens,
        candidates=canonical_candidates,
        required_evidence_item_ids=canonical_required_ids,
    )


def run_context_ablation(
    case: ContextAblationCase,
    *,
    pricing_table: PricingTable,
    provider: str,
    model: str,
    observed_latency_ms: Mapping[AblationVariant, int] | None = None,
) -> ContextAblationReport:
    """Compare the production builder with an unbounded all-context baseline."""

    if not isinstance(case, ContextAblationCase):
        raise TypeError("case must be a ContextAblationCase")
    if not isinstance(pricing_table, PricingTable):
        raise TypeError("pricing_table must be a PricingTable")
    latency = _normalize_latency_observations(observed_latency_ms)
    active_pack = build_active_context(
        candidates=case.candidates,
        policy=ContextBudgetPolicy(budget_tokens=case.budget_tokens),
    )
    price = pricing_table.lookup(provider, model)
    active = _variant_metrics(
        case,
        variant=AblationVariant.ACTIVE_CONTEXT,
        selected_items=active_pack.items,
        context_id=active_pack.pack_id,
        build_id=active_pack.build_report.build_id,
        price_rate=(None if price is None else price.input_usd_per_million_tokens),
        latency_ms=latency.get(AblationVariant.ACTIVE_CONTEXT),
    )
    baseline_items = case.candidates
    baseline = _variant_metrics(
        case,
        variant=AblationVariant.ALL_CONTEXT_BASELINE,
        selected_items=baseline_items,
        context_id=_baseline_context_id(baseline_items),
        build_id=None,
        price_rate=(None if price is None else price.input_usd_per_million_tokens),
        latency_ms=latency.get(AblationVariant.ALL_CONTEXT_BASELINE),
    )
    variants = (active, baseline)
    token_reduction = baseline.selected_tokens - active.selected_tokens
    if token_reduction < 0:
        raise ValueError("active context cannot exceed the all-context baseline")
    retention_delta = (
        active.required_evidence_retention_ppm
        - baseline.required_evidence_retention_ppm
    )
    savings = _cost_savings(active, baseline)
    latency_delta = _latency_delta(active, baseline)
    report_id = _report_id(
        case_id=case.case_id,
        case_name=case.name,
        provider=provider,
        model=model,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        variants=variants,
        token_reduction=token_reduction,
        required_evidence_retention_delta_ppm=retention_delta,
        estimated_input_cost_savings_usd=savings,
        observed_latency_delta_ms=latency_delta,
    )
    return ContextAblationReport(
        report_id=report_id,
        case_id=case.case_id,
        case_name=case.name,
        provider=provider,
        model=model,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        variants=variants,
        token_reduction=token_reduction,
        required_evidence_retention_delta_ppm=retention_delta,
        estimated_input_cost_savings_usd=savings,
        observed_latency_delta_ms=latency_delta,
    )


def render_context_ablation_summary(report: ContextAblationReport) -> str:
    """Render a compact human-readable view from the immutable report."""

    if not isinstance(report, ContextAblationReport):
        raise TypeError("report must be a ContextAblationReport")
    active, baseline = report.variants
    savings = report.estimated_input_cost_savings_usd or "unavailable"
    return (
        f"Context ablation {report.case_name}: active_context selected "
        f"{active.selected_tokens} tokens; all_context_baseline selected "
        f"{baseline.selected_tokens}; token reduction {report.token_reduction}; "
        f"required evidence retention "
        f"{active.required_evidence_retention_ppm / 10_000:.2f}%; "
        f"estimated input cost savings USD {savings}; estimate, not provider invoice"
    )


class ContextAblationReportSink:
    """Persist one immutable report without overwriting existing evidence."""

    def write(self, report: ContextAblationReport, destination: Path) -> Path:
        if not isinstance(report, ContextAblationReport):
            raise TypeError("report must be a ContextAblationReport")
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2))
            stream.write("\n")
        return resolved


def _variant_metrics(
    case: ContextAblationCase,
    *,
    variant: AblationVariant,
    selected_items: tuple[ContextItem, ...],
    context_id: str,
    build_id: str | None,
    price_rate: str | None,
    latency_ms: int | None,
) -> ContextAblationVariantMetrics:
    selected_ids = tuple(item.item_id for item in selected_items)
    selected_tokens = sum(estimate_context_item_tokens(item) for item in selected_items)
    retained_required = len(
        set(case.required_evidence_item_ids).intersection(selected_ids)
    )
    evidence_items = tuple(
        item for item in selected_items if item.section is ContextSection.EVIDENCE
    )
    traceable_evidence = sum(
        item.source.raw_artifact_id is not None and item.source.location is not None
        for item in evidence_items
    )
    if price_rate is None:
        cost_availability = CostAvailability.UNAVAILABLE
        estimated_cost = None
        cost_reason = CostUnavailableReason.PRICE_NOT_FOUND
    else:
        cost_availability = CostAvailability.ESTIMATED
        estimated_cost = _input_cost(selected_tokens, price_rate)
        cost_reason = None
    return ContextAblationVariantMetrics(
        variant=variant,
        context_id=context_id,
        build_id=build_id,
        candidate_count=len(case.candidates),
        selected_count=len(selected_items),
        selected_item_ids=selected_ids,
        budget_tokens=case.budget_tokens,
        selected_tokens=selected_tokens,
        overflow_tokens=max(0, selected_tokens - case.budget_tokens),
        required_evidence_count=len(case.required_evidence_item_ids),
        retained_required_evidence_count=retained_required,
        required_evidence_retention_ppm=_coverage_ppm(
            retained_required,
            len(case.required_evidence_item_ids),
        ),
        evidence_item_count=len(evidence_items),
        traceable_evidence_count=traceable_evidence,
        evidence_provenance_coverage_ppm=_coverage_ppm(
            traceable_evidence,
            len(evidence_items),
        ),
        cost_availability=cost_availability,
        estimated_input_cost_usd=estimated_cost,
        cost_unavailable_reason=cost_reason,
        latency_availability=(
            LatencyAvailability.NOT_MEASURED
            if latency_ms is None
            else LatencyAvailability.OBSERVED
        ),
        latency_ms=latency_ms,
    )


def _normalize_latency_observations(
    observations: Mapping[AblationVariant, int] | None,
) -> dict[AblationVariant, int]:
    if observations is None:
        return {}
    normalized: dict[AblationVariant, int] = {}
    for variant, value in observations.items():
        if not isinstance(variant, AblationVariant):
            raise TypeError("latency observation keys must be AblationVariant values")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("latency observations must be non-negative integers")
        normalized[variant] = value
    return normalized


def _coverage_ppm(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return PARTS_PER_MILLION
    return numerator * PARTS_PER_MILLION // denominator


def _input_cost(tokens: int, rate: str) -> str:
    with localcontext() as context:
        context.prec = 128
        amount = Decimal(tokens) * Decimal(rate) / Decimal(PARTS_PER_MILLION)
    return _decimal_string(amount)


def _cost_savings(
    active: ContextAblationVariantMetrics,
    baseline: ContextAblationVariantMetrics,
) -> str | None:
    if (
        active.estimated_input_cost_usd is None
        or baseline.estimated_input_cost_usd is None
    ):
        return None
    return _decimal_string(
        Decimal(baseline.estimated_input_cost_usd)
        - Decimal(active.estimated_input_cost_usd)
    )


def _latency_delta(
    active: ContextAblationVariantMetrics,
    baseline: ContextAblationVariantMetrics,
) -> int | None:
    if active.latency_ms is None or baseline.latency_ms is None:
        return None
    return baseline.latency_ms - active.latency_ms


def _is_canonical_decimal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return _decimal_string(Decimal(value)) == value
    except (DecimalException, ValueError):
        return False


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("decimal amount must be finite and non-negative")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _case_id(
    *,
    name: str,
    budget_tokens: int,
    candidates: tuple[ContextItem, ...],
    required_evidence_item_ids: tuple[str, ...],
) -> str:
    return _digest(
        b"repocompass:context-ablation-case:v1\0",
        {
            "budget_tokens": budget_tokens,
            "candidates": [
                item.model_dump(mode="json", exclude_computed_fields=True)
                for item in candidates
            ],
            "name": name,
            "required_evidence_item_ids": list(required_evidence_item_ids),
            "schema_version": CONTEXT_ABLATION_SCHEMA_VERSION,
        },
    )


def _baseline_context_id(items: tuple[ContextItem, ...]) -> str:
    return _digest(
        b"repocompass:all-context-baseline:v1\0",
        {
            "item_ids": [item.item_id for item in items],
            "token_estimates": [estimate_context_item_tokens(item) for item in items],
        },
    )


def _report_id(**payload: object) -> str:
    return _digest(
        b"repocompass:context-ablation-report:v1\0",
        {
            "schema_version": CONTEXT_ABLATION_SCHEMA_VERSION,
            **{
                key: (
                    [item.model_dump(mode="json") for item in value]
                    if key == "variants"
                    else value
                )
                for key, value in payload.items()
            },
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
