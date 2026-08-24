"""Versioned pricing and deterministic model-usage cost reports."""

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.usage import (
    ModelInvocationRecord,
    ModelInvocationStatus,
    PROVIDER_PATTERN,
    UsageAvailability,
)
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    Sha256Digest,
)

COST_SCHEMA_VERSION = 1
PRICING_UNIT_TOKENS = 1_000_000
COST_DISCLAIMER = "estimate_not_invoice"
_MAX_PRICE_ENTRIES = 10_000
_MAX_RATE_LENGTH = 64
_MAX_COST_LENGTH = 128
_CANONICAL_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")


class _CostContractModel(CheckpointSafeModel):
    """Strict immutable base for pricing and cost-report contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ModelPrice(_CostContractModel):
    """One exact provider/model price in USD per million tokens."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    input_usd_per_million_tokens: str = Field(max_length=_MAX_RATE_LENGTH)
    output_usd_per_million_tokens: str = Field(max_length=_MAX_RATE_LENGTH)
    cached_input_usd_per_million_tokens: str = Field(
        default="same_as_input",
        max_length=_MAX_RATE_LENGTH,
    )
    reasoning_output_usd_per_million_tokens: str = Field(
        default="same_as_output",
        max_length=_MAX_RATE_LENGTH,
    )

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        mode="before",
    )
    @classmethod
    def require_canonical_base_rate(cls, value: object) -> object:
        _require_canonical_decimal(value, label="rate")
        return value

    @field_validator("cached_input_usd_per_million_tokens", mode="before")
    @classmethod
    def require_canonical_cached_rate(cls, value: object) -> object:
        if value != "same_as_input":
            _require_canonical_decimal(value, label="rate")
        return value

    @field_validator("reasoning_output_usd_per_million_tokens", mode="before")
    @classmethod
    def require_canonical_reasoning_rate(cls, value: object) -> object:
        if value != "same_as_output":
            _require_canonical_decimal(value, label="rate")
        return value


class PricingTable(_CostContractModel):
    """Content-addressed immutable price snapshot with exact-key lookup."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    table_id: Sha256Digest
    version: str = Field(pattern=IDENTIFIER_PATTERN)
    effective_from: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    currency: Literal["USD"] = "USD"
    unit_tokens: Literal[1_000_000] = PRICING_UNIT_TOKENS
    entries: tuple[ModelPrice, ...] = Field(
        min_length=1,
        max_length=_MAX_PRICE_ENTRIES,
    )

    @field_validator("effective_from", mode="before")
    @classmethod
    def require_real_iso_date(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("effective_from must be an ISO date string")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("effective_from must be a real ISO date") from error
        if parsed.isoformat() != value:
            raise ValueError("effective_from must be a canonical ISO date")
        return value

    @model_validator(mode="after")
    def require_canonical_snapshot(self) -> "PricingTable":
        keys = tuple((entry.provider, entry.model) for entry in self.entries)
        if len(keys) != len(set(keys)):
            raise ValueError("pricing table contains a duplicate provider/model key")
        if keys != tuple(sorted(keys)):
            raise ValueError("pricing table entries must use canonical sorted order")
        expected_id = _pricing_table_id(
            version=self.version,
            effective_from=self.effective_from,
            entries=self.entries,
        )
        if self.table_id != expected_id:
            raise ValueError("table_id does not match pricing snapshot contents")
        return self

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        """Return an exact case-sensitive provider/model price, if present."""

        return next(
            (
                entry
                for entry in self.entries
                if entry.provider == provider and entry.model == model
            ),
            None,
        )


class CostAvailability(StrEnum):
    """Whether one invocation has a defensible price estimate."""

    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class CostUnavailableReason(StrEnum):
    """Closed reasons why an invocation or aggregate cannot be fully priced."""

    USAGE_UNAVAILABLE = "usage_unavailable"
    INPUT_TOKENS_UNAVAILABLE = "input_tokens_unavailable"
    OUTPUT_TOKENS_UNAVAILABLE = "output_tokens_unavailable"
    PRICE_NOT_FOUND = "price_not_found"
    CACHED_INPUT_TOKENS_UNAVAILABLE = "cached_input_tokens_unavailable"
    REASONING_OUTPUT_TOKENS_UNAVAILABLE = "reasoning_output_tokens_unavailable"
    NO_INVOCATIONS = "no_invocations"


class InvocationCostEstimate(_CostContractModel):
    """Exact decimal estimate for one immutable model invocation."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    invocation_id: Sha256Digest
    pricing_table_id: Sha256Digest
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    availability: CostAvailability
    input_cost_usd: str | None = Field(default=None, max_length=_MAX_COST_LENGTH)
    output_cost_usd: str | None = Field(default=None, max_length=_MAX_COST_LENGTH)
    estimated_cost_usd: str | None = Field(default=None, max_length=_MAX_COST_LENGTH)
    unavailable_reason: CostUnavailableReason | None = None
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator(
        "input_cost_usd",
        "output_cost_usd",
        "estimated_cost_usd",
        mode="before",
    )
    @classmethod
    def require_canonical_cost(cls, value: object) -> object:
        if value is not None:
            _require_canonical_decimal(value, label="cost")
        return value

    @model_validator(mode="after")
    def reconcile_availability(self) -> "InvocationCostEstimate":
        costs = (
            self.input_cost_usd,
            self.output_cost_usd,
            self.estimated_cost_usd,
        )
        if self.availability is CostAvailability.UNAVAILABLE:
            if any(cost is not None for cost in costs):
                raise ValueError("unavailable cost estimate must not contain amounts")
            if self.unavailable_reason is None:
                raise ValueError("unavailable cost estimate requires a reason")
            return self
        if any(cost is None for cost in costs):
            raise ValueError("estimated cost requires input, output, and total amounts")
        if self.unavailable_reason is not None:
            raise ValueError("estimated cost must not contain an unavailable reason")
        if Decimal(self.input_cost_usd or "0") + Decimal(
            self.output_cost_usd or "0"
        ) != Decimal(self.estimated_cost_usd or "0"):
            raise ValueError("estimated cost total must equal input plus output cost")
        return self


class MetricCoverage(StrEnum):
    """Coverage state for one token metric in an aggregate."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class UsageMetricSummary(_CostContractModel):
    """One summed token dimension with explicit contributor coverage."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    coverage: MetricCoverage
    value: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    contributing_invocations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    total_invocations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def require_truthful_coverage(self) -> "UsageMetricSummary":
        if self.contributing_invocations > self.total_invocations:
            raise ValueError("metric contributors cannot exceed total invocations")
        expected = _metric_coverage(
            self.contributing_invocations,
            self.total_invocations,
        )
        if self.coverage is not expected:
            raise ValueError("metric coverage does not match contributor counts")
        if (self.value is None) != (self.contributing_invocations == 0):
            raise ValueError("metric value presence must match contributor coverage")
        return self


class CostCoverage(StrEnum):
    """Coverage state for an aggregate cost estimate."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class CostSummary(_CostContractModel):
    """Summed estimated cost and the invocations excluded from that sum."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    coverage: CostCoverage
    estimated_cost_usd: str | None = Field(default=None, max_length=_MAX_COST_LENGTH)
    estimated_invocations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    unavailable_invocations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    unavailable_reasons: tuple[CostUnavailableReason, ...] = ()
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_cost_usd", mode="before")
    @classmethod
    def require_canonical_total(cls, value: object) -> object:
        if value is not None:
            _require_canonical_decimal(value, label="cost")
        return value

    @model_validator(mode="after")
    def require_truthful_coverage(self) -> "CostSummary":
        total = self.estimated_invocations + self.unavailable_invocations
        if total > MAX_SAFE_JSON_INTEGER:
            raise ValueError("cost invocation total exceeds the safe JSON integer")
        expected = _cost_coverage(self.estimated_invocations, total)
        if self.coverage is not expected:
            raise ValueError("cost coverage does not match invocation counts")
        if (self.estimated_cost_usd is None) != (self.estimated_invocations == 0):
            raise ValueError("cost amount presence must match estimated invocations")
        reasons = tuple(
            sorted(set(self.unavailable_reasons), key=lambda item: item.value)
        )
        if reasons != self.unavailable_reasons:
            raise ValueError("unavailable cost reasons must be unique and sorted")
        if total == 0:
            if reasons != (CostUnavailableReason.NO_INVOCATIONS,):
                raise ValueError("empty cost coverage requires no_invocations reason")
            return self
        if CostUnavailableReason.NO_INVOCATIONS in reasons:
            raise ValueError("nonempty cost coverage cannot use no_invocations reason")
        if self.unavailable_invocations == 0 and reasons:
            raise ValueError("complete cost coverage must not contain reasons")
        if self.unavailable_invocations > 0 and not reasons:
            raise ValueError("unavailable cost coverage requires at least one reason")
        return self


class BreakdownDimension(StrEnum):
    """Stable grouping dimensions exposed by an attempt report."""

    ATTEMPT = "attempt"
    ROLE = "role"
    NODE = "node"
    MODEL = "model"


class UsageCostBreakdown(_CostContractModel):
    """Usage, latency, status, and cost totals for one deterministic group."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    dimension: BreakdownDimension
    key: str = Field(min_length=1, max_length=512)
    invocation_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    succeeded_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    failed_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    latency_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    input_tokens: UsageMetricSummary
    output_tokens: UsageMetricSummary
    total_tokens: UsageMetricSummary
    cached_input_tokens: UsageMetricSummary
    reasoning_output_tokens: UsageMetricSummary
    cost: CostSummary

    @model_validator(mode="after")
    def reconcile_status_counts(self) -> "UsageCostBreakdown":
        if self.succeeded_count + self.failed_count != self.invocation_count:
            raise ValueError("breakdown status counts must equal invocation_count")
        metric_totals = (
            self.input_tokens.total_invocations,
            self.output_tokens.total_invocations,
            self.total_tokens.total_invocations,
            self.cached_input_tokens.total_invocations,
            self.reasoning_output_tokens.total_invocations,
        )
        if any(total != self.invocation_count for total in metric_totals):
            raise ValueError("breakdown metric totals must equal invocation_count")
        if (
            self.cost.estimated_invocations + self.cost.unavailable_invocations
            != self.invocation_count
        ):
            raise ValueError("breakdown cost totals must equal invocation_count")
        return self


class AttemptUsageCostReport(_CostContractModel):
    """Immutable report derived from invocation records and one price snapshot."""

    schema_version: Literal[1] = COST_SCHEMA_VERSION
    report_id: Sha256Digest
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    pricing_table_id: Sha256Digest
    pricing_version: str = Field(pattern=IDENTIFIER_PATTERN)
    pricing_effective_from: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    currency: Literal["USD"] = "USD"
    unit_tokens: Literal[1_000_000] = PRICING_UNIT_TOKENS
    invocation_costs: tuple[InvocationCostEstimate, ...]
    breakdowns: tuple[UsageCostBreakdown, ...] = Field(min_length=1)
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @model_validator(mode="after")
    def require_content_addressed_report(self) -> "AttemptUsageCostReport":
        if self.breakdowns[0].dimension is not BreakdownDimension.ATTEMPT:
            raise ValueError("first report breakdown must be the attempt total")
        expected_id = _attempt_report_id(
            run_id=self.run_id,
            thread_id=self.thread_id,
            turn_number=self.turn_number,
            pricing_table_id=self.pricing_table_id,
            pricing_version=self.pricing_version,
            pricing_effective_from=self.pricing_effective_from,
            invocation_costs=self.invocation_costs,
            breakdowns=self.breakdowns,
        )
        if self.report_id != expected_id:
            raise ValueError("report_id does not match usage and cost report contents")
        return self


def create_pricing_table(
    *,
    version: str,
    effective_from: str,
    entries: Iterable[ModelPrice],
) -> PricingTable:
    """Create a sorted, duplicate-checked, content-addressed price snapshot."""

    candidate_entries = tuple(entries)
    if any(not isinstance(item, ModelPrice) for item in candidate_entries):
        raise TypeError("entries must contain ModelPrice values")
    canonical_entries = tuple(
        sorted(candidate_entries, key=lambda item: (item.provider, item.model))
    )
    table_id = _pricing_table_id(
        version=version,
        effective_from=effective_from,
        entries=canonical_entries,
    )
    return PricingTable(
        table_id=table_id,
        version=version,
        effective_from=effective_from,
        entries=canonical_entries,
    )


def estimate_invocation_cost(
    invocation: ModelInvocationRecord,
    pricing_table: PricingTable,
) -> InvocationCostEstimate:
    """Estimate one invocation exactly, or return a specific unavailable reason."""

    if not isinstance(invocation, ModelInvocationRecord):
        raise TypeError("invocation must be a ModelInvocationRecord")
    if not isinstance(pricing_table, PricingTable):
        raise TypeError("pricing_table must be a PricingTable")
    usage = invocation.usage
    if usage.availability is UsageAvailability.UNAVAILABLE:
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.USAGE_UNAVAILABLE,
        )
    price = pricing_table.lookup(invocation.provider, invocation.model)
    if price is None:
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.PRICE_NOT_FOUND,
        )
    if usage.input_tokens is None:
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.INPUT_TOKENS_UNAVAILABLE,
        )
    if usage.output_tokens is None:
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.OUTPUT_TOKENS_UNAVAILABLE,
        )
    if (
        price.cached_input_usd_per_million_tokens != "same_as_input"
        and usage.cached_input_tokens is None
    ):
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.CACHED_INPUT_TOKENS_UNAVAILABLE,
        )
    if (
        price.reasoning_output_usd_per_million_tokens != "same_as_output"
        and usage.reasoning_output_tokens is None
    ):
        return _unavailable_estimate(
            invocation,
            pricing_table,
            CostUnavailableReason.REASONING_OUTPUT_TOKENS_UNAVAILABLE,
        )

    with localcontext() as context:
        context.prec = 128
        input_cost = _tier_cost(
            total_tokens=usage.input_tokens,
            specialized_tokens=usage.cached_input_tokens,
            base_rate=price.input_usd_per_million_tokens,
            specialized_rate=price.cached_input_usd_per_million_tokens,
            same_as_base="same_as_input",
        )
        output_cost = _tier_cost(
            total_tokens=usage.output_tokens,
            specialized_tokens=usage.reasoning_output_tokens,
            base_rate=price.output_usd_per_million_tokens,
            specialized_rate=price.reasoning_output_usd_per_million_tokens,
            same_as_base="same_as_output",
        )
        total_cost = input_cost + output_cost
    return InvocationCostEstimate(
        invocation_id=invocation.invocation_id,
        pricing_table_id=pricing_table.table_id,
        provider=invocation.provider,
        model=invocation.model,
        availability=CostAvailability.ESTIMATED,
        input_cost_usd=_decimal_string(input_cost),
        output_cost_usd=_decimal_string(output_cost),
        estimated_cost_usd=_decimal_string(total_cost),
    )


def build_attempt_usage_cost_report(
    *,
    run_id: str,
    thread_id: str,
    turn_number: int,
    invocations: Iterable[ModelInvocationRecord],
    pricing_table: PricingTable,
) -> AttemptUsageCostReport:
    """Build stable attempt, role, node, and model usage/cost breakdowns."""

    if not isinstance(pricing_table, PricingTable):
        raise TypeError("pricing_table must be a PricingTable")
    candidate_records = tuple(invocations)
    if any(not isinstance(item, ModelInvocationRecord) for item in candidate_records):
        raise TypeError("invocations must contain ModelInvocationRecord values")
    records = tuple(sorted(candidate_records, key=lambda item: item.sequence))
    if any(
        item.run_id != run_id
        or item.thread_id != thread_id
        or item.turn_number != turn_number
        for item in records
    ):
        raise ValueError("model invocation does not match attempt identity")
    sequences = tuple(item.sequence for item in records)
    if len(sequences) != len(set(sequences)):
        raise ValueError("model invocation sequence must be unique within an attempt")

    invocation_costs = tuple(
        estimate_invocation_cost(item, pricing_table) for item in records
    )
    costs_by_id = {item.invocation_id: item for item in invocation_costs}
    breakdowns = [
        _build_breakdown(BreakdownDimension.ATTEMPT, run_id, records, costs_by_id)
    ]
    breakdowns.extend(
        _group_breakdowns(
            BreakdownDimension.ROLE,
            records,
            costs_by_id,
            key=lambda item: item.role.value,
        )
    )
    breakdowns.extend(
        _group_breakdowns(
            BreakdownDimension.NODE,
            records,
            costs_by_id,
            key=lambda item: item.node_name,
        )
    )
    breakdowns.extend(
        _group_breakdowns(
            BreakdownDimension.MODEL,
            records,
            costs_by_id,
            key=lambda item: f"{item.provider}:{item.model}",
        )
    )
    canonical_breakdowns = tuple(breakdowns)
    report_id = _attempt_report_id(
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        invocation_costs=invocation_costs,
        breakdowns=canonical_breakdowns,
    )
    return AttemptUsageCostReport(
        report_id=report_id,
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        pricing_table_id=pricing_table.table_id,
        pricing_version=pricing_table.version,
        pricing_effective_from=pricing_table.effective_from,
        invocation_costs=invocation_costs,
        breakdowns=canonical_breakdowns,
    )


def _group_breakdowns(
    dimension: BreakdownDimension,
    records: tuple[ModelInvocationRecord, ...],
    costs_by_id: dict[str, InvocationCostEstimate],
    *,
    key: Callable[[ModelInvocationRecord], str],
) -> tuple[UsageCostBreakdown, ...]:
    grouped: dict[str, list[ModelInvocationRecord]] = {}
    for record in records:
        group_key = key(record)
        grouped.setdefault(group_key, []).append(record)
    return tuple(
        _build_breakdown(
            dimension,
            group_key,
            tuple(grouped[group_key]),
            costs_by_id,
        )
        for group_key in sorted(grouped)
    )


def _build_breakdown(
    dimension: BreakdownDimension,
    key: str,
    records: tuple[ModelInvocationRecord, ...],
    costs_by_id: dict[str, InvocationCostEstimate],
) -> UsageCostBreakdown:
    costs = tuple(costs_by_id[item.invocation_id] for item in records)
    return UsageCostBreakdown(
        dimension=dimension,
        key=key,
        invocation_count=len(records),
        succeeded_count=sum(
            item.status is ModelInvocationStatus.SUCCEEDED for item in records
        ),
        failed_count=sum(
            item.status is ModelInvocationStatus.FAILED for item in records
        ),
        latency_ms=_checked_integer_sum(item.latency_ms for item in records),
        input_tokens=_usage_metric(records, "input_tokens"),
        output_tokens=_usage_metric(records, "output_tokens"),
        total_tokens=_usage_metric(records, "total_tokens"),
        cached_input_tokens=_usage_metric(records, "cached_input_tokens"),
        reasoning_output_tokens=_usage_metric(records, "reasoning_output_tokens"),
        cost=_cost_summary(costs),
    )


def _usage_metric(
    records: Sequence[ModelInvocationRecord],
    attribute: str,
) -> UsageMetricSummary:
    values = tuple(
        value
        for item in records
        if (value := getattr(item.usage, attribute)) is not None
    )
    contributors = len(values)
    return UsageMetricSummary(
        coverage=_metric_coverage(contributors, len(records)),
        value=_checked_integer_sum(values) if values else None,
        contributing_invocations=contributors,
        total_invocations=len(records),
    )


def _cost_summary(costs: Sequence[InvocationCostEstimate]) -> CostSummary:
    estimated = tuple(
        item for item in costs if item.availability is CostAvailability.ESTIMATED
    )
    unavailable = tuple(
        item for item in costs if item.availability is CostAvailability.UNAVAILABLE
    )
    reasons = (
        (CostUnavailableReason.NO_INVOCATIONS,)
        if not costs
        else tuple(
            sorted(
                {
                    item.unavailable_reason
                    for item in unavailable
                    if item.unavailable_reason is not None
                },
                key=lambda item: item.value,
            )
        )
    )
    with localcontext() as context:
        context.prec = 128
        amount = sum(
            (Decimal(item.estimated_cost_usd or "0") for item in estimated),
            start=Decimal(0),
        )
    return CostSummary(
        coverage=_cost_coverage(len(estimated), len(costs)),
        estimated_cost_usd=_decimal_string(amount) if estimated else None,
        estimated_invocations=len(estimated),
        unavailable_invocations=len(unavailable),
        unavailable_reasons=reasons,
    )


def _unavailable_estimate(
    invocation: ModelInvocationRecord,
    pricing_table: PricingTable,
    reason: CostUnavailableReason,
) -> InvocationCostEstimate:
    return InvocationCostEstimate(
        invocation_id=invocation.invocation_id,
        pricing_table_id=pricing_table.table_id,
        provider=invocation.provider,
        model=invocation.model,
        availability=CostAvailability.UNAVAILABLE,
        unavailable_reason=reason,
    )


def _tier_cost(
    *,
    total_tokens: int,
    specialized_tokens: int | None,
    base_rate: str,
    specialized_rate: str,
    same_as_base: str,
) -> Decimal:
    base = Decimal(base_rate)
    if specialized_rate == same_as_base:
        return Decimal(total_tokens) * base / Decimal(PRICING_UNIT_TOKENS)
    specialized_count = specialized_tokens or 0
    return (
        Decimal(total_tokens - specialized_count) * base
        + Decimal(specialized_count) * Decimal(specialized_rate)
    ) / Decimal(PRICING_UNIT_TOKENS)


def _metric_coverage(contributors: int, total: int) -> MetricCoverage:
    if contributors == 0:
        return MetricCoverage.UNAVAILABLE
    if contributors == total:
        return MetricCoverage.COMPLETE
    return MetricCoverage.PARTIAL


def _cost_coverage(estimated: int, total: int) -> CostCoverage:
    if estimated == 0:
        return CostCoverage.UNAVAILABLE
    if estimated == total:
        return CostCoverage.COMPLETE
    return CostCoverage.PARTIAL


def _checked_integer_sum(values: Iterable[int]) -> int:
    total = sum(values)
    if total > MAX_SAFE_JSON_INTEGER:
        raise ValueError("aggregate exceeds the maximum safe JSON integer")
    return total


def _require_canonical_decimal(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not _CANONICAL_DECIMAL_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a canonical non-negative decimal string")
    decimal_value = Decimal(value)
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError(f"{label} must be a finite non-negative decimal")


def _decimal_string(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _pricing_table_id(
    *,
    version: str,
    effective_from: str,
    entries: Sequence[ModelPrice],
) -> str:
    return _content_digest(
        b"repocompass:pricing-table:v1\0",
        {
            "version": version,
            "effective_from": effective_from,
            "currency": "USD",
            "unit_tokens": PRICING_UNIT_TOKENS,
            "entries": [item.model_dump(mode="json") for item in entries],
        },
    )


def _attempt_report_id(
    *,
    run_id: str,
    thread_id: str,
    turn_number: int,
    pricing_table_id: str,
    pricing_version: str,
    pricing_effective_from: str,
    invocation_costs: Sequence[InvocationCostEstimate],
    breakdowns: Sequence[UsageCostBreakdown],
) -> str:
    return _content_digest(
        b"repocompass:attempt-usage-cost-report:v1\0",
        {
            "run_id": run_id,
            "thread_id": thread_id,
            "turn_number": turn_number,
            "pricing_table_id": pricing_table_id,
            "pricing_version": pricing_version,
            "pricing_effective_from": pricing_effective_from,
            "currency": "USD",
            "unit_tokens": PRICING_UNIT_TOKENS,
            "invocation_costs": [
                item.model_dump(mode="json") for item in invocation_costs
            ],
            "breakdowns": [item.model_dump(mode="json") for item in breakdowns],
            "disclaimer": COST_DISCLAIMER,
        },
    )


def _content_digest(prefix: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(prefix + encoded).hexdigest()
