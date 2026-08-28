"""Deterministic case-balanced statistics for localization evaluations."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.cost import CostAvailability
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation.localization_contracts import (
    LOCALIZATION_SCOPE_DISCLAIMER,
    LocalizationDatasetKind,
    LocalizationMetricAvailability,
    LocalizationRunObservation,
    LocalizationRunStatus,
    LocalizationVariant,
)
from repocompass.evaluation.localization_harness import LocalizationObservationSet
from repocompass.evaluation.localization_metrics import (
    PARTS_PER_MILLION,
    LocalizationCaseMetrics,
    LocalizationEvaluationReport,
    LocalizationRate,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest

ANALYSIS_SCHEMA_VERSION = 1
MIN_BOOTSTRAP_RESAMPLES = 100
MAX_BOOTSTRAP_RESAMPLES = 100_000


class LocalizationAnalysisProvenance(StrEnum):
    """Closed claims about how observations were produced."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    LIVE_PROVIDER = "live_provider"
    EXTERNAL_IMPORT = "external_import"


class LocalizationAnalysisMetric(StrEnum):
    """Case-level metrics included in confidence analysis."""

    FILE_HIT_AT_1 = "file_hit_at_1"
    FILE_HIT_AT_3 = "file_hit_at_3"
    FILE_HIT_AT_5 = "file_hit_at_5"
    FILE_RECALL_AT_1 = "file_recall_at_1"
    FILE_RECALL_AT_3 = "file_recall_at_3"
    FILE_RECALL_AT_5 = "file_recall_at_5"
    MEAN_RECIPROCAL_RANK = "mean_reciprocal_rank"
    SYMBOL_HIT_AT_1 = "symbol_hit_at_1"
    SYMBOL_HIT_AT_3 = "symbol_hit_at_3"
    SYMBOL_HIT_AT_5 = "symbol_hit_at_5"
    EVIDENCE_VALIDITY = "evidence_validity"
    UNSUPPORTED_RELATION_RATE = "unsupported_relation_rate"


class LocalizationAnalysisUnavailableReason(StrEnum):
    NO_ELIGIBLE_CASES = "no_eligible_cases"
    INSUFFICIENT_CASES = "insufficient_cases"


class LocalizationCaseErrorKind(StrEnum):
    RUN_NOT_ACCEPTED = "run_not_accepted"
    NO_FILE_HIT_AT_5 = "no_file_hit_at_5"
    PARTIAL_FILE_RECALL_AT_5 = "partial_file_recall_at_5"
    NO_SYMBOL_HIT_AT_5 = "no_symbol_hit_at_5"
    QUALITY_UNAVAILABLE = "quality_unavailable"
    NO_EVIDENCE = "no_evidence"
    INVALID_EVIDENCE = "invalid_evidence"
    UNSUPPORTED_RELATION = "unsupported_relation"


class LocalizationResourceMetric(StrEnum):
    CONTEXT_LINES = "context_lines"
    CONTEXT_TOKENS = "context_tokens"
    TOOL_CALLS = "tool_calls"
    LATENCY_MS = "latency_ms"
    EVALUATOR_ITERATIONS = "evaluator_iterations"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"


class _AnalysisModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LocalizationBootstrapConfig(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    config_id: Sha256Digest
    confidence_level_ppm: int = Field(ge=500_000, lt=PARTS_PER_MILLION)
    resamples: int = Field(
        ge=MIN_BOOTSTRAP_RESAMPLES,
        le=MAX_BOOTSTRAP_RESAMPLES,
    )
    random_seed: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    method: Literal["case_resampling_percentile"] = "case_resampling_percentile"

    @model_validator(mode="after")
    def require_identity(self) -> "LocalizationBootstrapConfig":
        expected = _model_identity(self, "localization_bootstrap_config", "config_id")
        if self.config_id != expected:
            raise ValueError("config_id does not match bootstrap configuration")
        return self


class LocalizationMetricConfidenceInterval(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    variant: LocalizationVariant
    metric: LocalizationAnalysisMetric
    total_case_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    eligible_case_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    point_estimate_ppm: int | None = Field(default=None, ge=0, le=PARTS_PER_MILLION)
    lower_bound_ppm: int | None = Field(default=None, ge=0, le=PARTS_PER_MILLION)
    upper_bound_ppm: int | None = Field(default=None, ge=0, le=PARTS_PER_MILLION)
    unavailable_reason: LocalizationAnalysisUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile(self) -> "LocalizationMetricConfidenceInterval":
        if self.eligible_case_count > self.total_case_count:
            raise ValueError("eligible cases cannot exceed total cases")
        _validate_interval_fields(
            count=self.eligible_case_count,
            point=self.point_estimate_ppm,
            lower=self.lower_bound_ppm,
            upper=self.upper_bound_ppm,
            reason=self.unavailable_reason,
        )
        return self


class LocalizationPairedVariantDelta(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    reference_variant: LocalizationVariant
    candidate_variant: LocalizationVariant
    metric: LocalizationAnalysisMetric
    paired_case_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    point_delta_ppm: int | None = Field(
        default=None, ge=-PARTS_PER_MILLION, le=PARTS_PER_MILLION
    )
    lower_bound_ppm: int | None = Field(
        default=None, ge=-PARTS_PER_MILLION, le=PARTS_PER_MILLION
    )
    upper_bound_ppm: int | None = Field(
        default=None, ge=-PARTS_PER_MILLION, le=PARTS_PER_MILLION
    )
    unavailable_reason: LocalizationAnalysisUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile(self) -> "LocalizationPairedVariantDelta":
        if self.reference_variant is self.candidate_variant:
            raise ValueError("paired delta variants must differ")
        _validate_interval_fields(
            count=self.paired_case_count,
            point=self.point_delta_ppm,
            lower=self.lower_bound_ppm,
            upper=self.upper_bound_ppm,
            reason=self.unavailable_reason,
        )
        return self


class LocalizationCaseErrorAnalysis(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    observation_id: Sha256Digest
    case_id: Sha256Digest
    task_id: Sha256Digest
    variant: LocalizationVariant
    repeat_index: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    run_status: LocalizationRunStatus
    error_kinds: tuple[LocalizationCaseErrorKind, ...]

    @model_validator(mode="after")
    def require_canonical_errors(self) -> "LocalizationCaseErrorAnalysis":
        expected = tuple(
            kind for kind in LocalizationCaseErrorKind if kind in self.error_kinds
        )
        if self.error_kinds != expected or len(expected) != len(set(expected)):
            raise ValueError("case error kinds must be unique and canonical")
        return self


class LocalizationResourceAggregate(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    variant: LocalizationVariant
    metric: LocalizationResourceMetric
    total_observations: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    observed_observations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    total_value: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    mean_numerator: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    mean_denominator: int | None = Field(default=None, ge=1, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def reconcile(self) -> "LocalizationResourceAggregate":
        if self.observed_observations > self.total_observations:
            raise ValueError("observed resources cannot exceed total observations")
        values = (self.total_value, self.mean_numerator, self.mean_denominator)
        if self.observed_observations == 0:
            if values != (None, None, None):
                raise ValueError("unobserved resource must not contain values")
        elif (
            self.total_value is None
            or self.mean_numerator != self.total_value
            or self.mean_denominator != self.observed_observations
        ):
            raise ValueError("resource mean must retain its exact total/count ratio")
        return self


class LocalizationCostAggregate(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    variant: LocalizationVariant
    total_observations: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    estimated_observations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    unavailable_observations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    total_estimated_cost_usd: str | None = Field(default=None, max_length=128)
    disclaimer: Literal["estimate_not_invoice"] = "estimate_not_invoice"

    @model_validator(mode="after")
    def reconcile(self) -> "LocalizationCostAggregate":
        if (
            self.estimated_observations + self.unavailable_observations
            != self.total_observations
        ):
            raise ValueError("cost coverage must account for every observation")
        if self.estimated_observations == 0:
            if self.total_estimated_cost_usd is not None:
                raise ValueError("unavailable cost aggregate must not contain a total")
        elif self.total_estimated_cost_usd is None:
            raise ValueError("estimated cost aggregate requires an exact total")
        if self.total_estimated_cost_usd is not None:
            try:
                parsed = Decimal(self.total_estimated_cost_usd)
            except InvalidOperation as error:
                raise ValueError("estimated cost total must be decimal") from error
            if (
                parsed < 0
                or _canonical_decimal(parsed) != self.total_estimated_cost_usd
            ):
                raise ValueError("estimated cost total must be canonical")
        return self


class LocalizationAnalysisReport(_AnalysisModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    analysis_id: Sha256Digest
    source_report_id: Sha256Digest
    observation_set_id: Sha256Digest
    dataset_manifest_id: Sha256Digest
    dataset_name: str
    dataset_version: str
    dataset_kind: LocalizationDatasetKind
    budget_profile_id: Sha256Digest
    pricing_table_id: Sha256Digest | None = None
    provenance: LocalizationAnalysisProvenance
    bootstrap_config: LocalizationBootstrapConfig
    reference_variant: LocalizationVariant
    metric_intervals: tuple[LocalizationMetricConfidenceInterval, ...] = Field(
        min_length=1
    )
    paired_deltas: tuple[LocalizationPairedVariantDelta, ...]
    case_errors: tuple[LocalizationCaseErrorAnalysis, ...] = Field(min_length=1)
    resource_aggregates: tuple[LocalizationResourceAggregate, ...] = Field(min_length=1)
    cost_aggregates: tuple[LocalizationCostAggregate, ...] = Field(min_length=1)
    evaluation_scope: Literal["localization_only"] = "localization_only"
    disclaimer: Literal["swe_bench_derived_localization_not_resolved_rate"] = (
        LOCALIZATION_SCOPE_DISCLAIMER
    )

    @model_validator(mode="after")
    def require_identity_and_order(self) -> "LocalizationAnalysisReport":
        synthetic = self.dataset_kind is LocalizationDatasetKind.SYNTHETIC_FIXTURE
        if synthetic != (
            self.provenance is LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE
        ):
            raise ValueError("synthetic provenance must match synthetic dataset kind")
        if self.metric_intervals != tuple(
            sorted(self.metric_intervals, key=_interval_sort_key)
        ):
            raise ValueError("metric intervals must use canonical order")
        if self.paired_deltas != tuple(sorted(self.paired_deltas, key=_delta_sort_key)):
            raise ValueError("paired deltas must use canonical order")
        if any(
            item.reference_variant is not self.reference_variant
            for item in self.paired_deltas
        ):
            raise ValueError("paired deltas must use the reference variant")
        if self.case_errors != tuple(
            sorted(self.case_errors, key=_case_error_sort_key)
        ):
            raise ValueError("case errors must use canonical order")
        if self.resource_aggregates != tuple(
            sorted(self.resource_aggregates, key=_resource_sort_key)
        ):
            raise ValueError("resource aggregates must use canonical order")
        if self.cost_aggregates != tuple(
            sorted(self.cost_aggregates, key=lambda item: _variant_index(item.variant))
        ):
            raise ValueError("cost aggregates must use canonical order")
        expected = _model_identity(self, "localization_analysis_report", "analysis_id")
        if self.analysis_id != expected:
            raise ValueError("analysis_id does not match localization analysis")
        return self


class LocalizationAnalysisReportSink:
    """Persist one immutable machine-readable analysis."""

    def write(self, report: LocalizationAnalysisReport, destination: Path) -> Path:
        if not isinstance(report, LocalizationAnalysisReport):
            raise TypeError("report must be a LocalizationAnalysisReport")
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2, exclude_computed_fields=True))
            stream.write("\n")
        return resolved


def create_localization_bootstrap_config(
    *,
    confidence_level_ppm: int = 950_000,
    resamples: int = 10_000,
    random_seed: int = 202_608_28,
) -> LocalizationBootstrapConfig:
    fields = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "confidence_level_ppm": confidence_level_ppm,
        "resamples": resamples,
        "random_seed": random_seed,
        "method": "case_resampling_percentile",
    }
    return LocalizationBootstrapConfig(
        config_id=_identity("localization_bootstrap_config", fields),
        **fields,
    )


def analyze_localization_evaluation(
    report: LocalizationEvaluationReport,
    observation_set: LocalizationObservationSet,
    *,
    provenance: LocalizationAnalysisProvenance,
    bootstrap_config: LocalizationBootstrapConfig,
    reference_variant: LocalizationVariant,
) -> LocalizationAnalysisReport:
    """Build deterministic case-balanced quality and resource analysis."""

    _validate_inputs(
        report,
        observation_set,
        provenance,
        bootstrap_config,
        reference_variant,
    )
    variants = tuple(item.variant for item in report.aggregates)
    scores = _case_balanced_scores(report.case_metrics)
    intervals = tuple(
        _metric_interval(
            variant,
            metric,
            report.case_count,
            scores[(variant, metric)],
            bootstrap_config,
        )
        for variant in variants
        for metric in LocalizationAnalysisMetric
    )
    deltas = tuple(
        _paired_delta(
            reference_variant,
            candidate,
            metric,
            scores[(reference_variant, metric)],
            scores[(candidate, metric)],
            bootstrap_config,
        )
        for candidate in variants
        if candidate is not reference_variant
        for metric in LocalizationAnalysisMetric
    )
    metric_lookup = {item.observation_id: item for item in report.case_metrics}
    case_errors = tuple(
        _case_error(metric_lookup[item.observation_id], item)
        for item in observation_set.observations
    )
    resources = tuple(
        _resource_aggregate(variant, metric, observation_set.observations)
        for variant in variants
        for metric in LocalizationResourceMetric
    )
    costs = tuple(
        _cost_aggregate(variant, observation_set.observations) for variant in variants
    )
    fields = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_report_id": report.report_id,
        "observation_set_id": observation_set.set_id,
        "dataset_manifest_id": report.dataset_manifest_id,
        "dataset_name": report.dataset_name,
        "dataset_version": report.dataset_version,
        "dataset_kind": report.dataset_kind,
        "budget_profile_id": report.budget_profile_id,
        "pricing_table_id": report.pricing_table_id,
        "provenance": provenance,
        "bootstrap_config": bootstrap_config,
        "reference_variant": reference_variant,
        "metric_intervals": intervals,
        "paired_deltas": deltas,
        "case_errors": case_errors,
        "resource_aggregates": resources,
        "cost_aggregates": costs,
        "evaluation_scope": "localization_only",
        "disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
    }
    return LocalizationAnalysisReport(
        analysis_id=_identity("localization_analysis_report", fields),
        **fields,
    )


def render_localization_analysis_summary(
    report: LocalizationAnalysisReport,
) -> str:
    """Render a deterministic provenance-explicit Markdown summary."""

    if not isinstance(report, LocalizationAnalysisReport):
        raise TypeError("report must be a LocalizationAnalysisReport")
    provenance_note = {
        LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE: (
            "Synthetic fixture: validates harness behavior, not model quality."
        ),
        LocalizationAnalysisProvenance.DETERMINISTIC_REPLAY: (
            "Deterministic replay: validates reproducibility, not live-provider quality."
        ),
        LocalizationAnalysisProvenance.LIVE_PROVIDER: (
            "Live-provider observations; cost remains an estimate, not an invoice."
        ),
        LocalizationAnalysisProvenance.EXTERNAL_IMPORT: (
            "External import: audit the source provenance before comparison."
        ),
    }[report.provenance]
    lines = [
        "# RepoCompass localization analysis",
        "",
        f"Analysis {report.analysis_id} of report {report.source_report_id}.",
        "",
        provenance_note,
        "",
        (
            "Scope: localization-only, SWE-bench-derived evidence. This is not a "
            "SWE-bench resolved rate."
        ),
        "",
        (
            f"Intervals: {_format_ppm(report.bootstrap_config.confidence_level_ppm)} "
            f"case-resampling percentile bootstrap, "
            f"{report.bootstrap_config.resamples} resamples, "
            f"seed {report.bootstrap_config.random_seed}."
        ),
        "",
        "| Variant | File Hit@1 | File Hit@5 | Recall@5 | MRR | Symbol Hit@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected = (
        LocalizationAnalysisMetric.FILE_HIT_AT_1,
        LocalizationAnalysisMetric.FILE_HIT_AT_5,
        LocalizationAnalysisMetric.FILE_RECALL_AT_5,
        LocalizationAnalysisMetric.MEAN_RECIPROCAL_RANK,
        LocalizationAnalysisMetric.SYMBOL_HIT_AT_5,
    )
    for variant in LocalizationVariant:
        lookup = {
            item.metric: item
            for item in report.metric_intervals
            if item.variant is variant
        }
        if lookup:
            lines.append(
                "| "
                + " | ".join(
                    (
                        variant.value,
                        *(_format_interval(lookup[metric]) for metric in selected),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "| Variant | Context tokens | Tool calls | Latency ms | Input tokens | Output tokens | Estimated cost USD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for cost in report.cost_aggregates:
        resources = {
            item.metric: item
            for item in report.resource_aggregates
            if item.variant is cost.variant
        }
        lines.append(
            "| "
            + " | ".join(
                (
                    cost.variant.value,
                    _format_resource(
                        resources[LocalizationResourceMetric.CONTEXT_TOKENS]
                    ),
                    _format_resource(resources[LocalizationResourceMetric.TOOL_CALLS]),
                    _format_resource(resources[LocalizationResourceMetric.LATENCY_MS]),
                    _format_resource(
                        resources[LocalizationResourceMetric.INPUT_TOKENS]
                    ),
                    _format_resource(
                        resources[LocalizationResourceMetric.OUTPUT_TOKENS]
                    ),
                    cost.total_estimated_cost_usd or "n/a",
                )
            )
            + " |"
        )
    flagged = sum(bool(item.error_kinds) for item in report.case_errors)
    lines.extend(
        (
            "",
            (
                f"Deterministic error taxonomy flags {flagged}/"
                f"{len(report.case_errors)} logical observations."
            ),
            "",
            (
                "Estimated costs appear only when provider usage and matching "
                "pricing were available; missing cost is never treated as zero."
            ),
        )
    )
    return "\n".join(lines)


def _validate_inputs(
    report: LocalizationEvaluationReport,
    observation_set: LocalizationObservationSet,
    provenance: LocalizationAnalysisProvenance,
    bootstrap_config: LocalizationBootstrapConfig,
    reference_variant: LocalizationVariant,
) -> None:
    if not isinstance(report, LocalizationEvaluationReport):
        raise TypeError("report must be a LocalizationEvaluationReport")
    if not isinstance(observation_set, LocalizationObservationSet):
        raise TypeError("observation_set must be a LocalizationObservationSet")
    if not isinstance(provenance, LocalizationAnalysisProvenance):
        raise TypeError("provenance must be a LocalizationAnalysisProvenance")
    if not isinstance(bootstrap_config, LocalizationBootstrapConfig):
        raise TypeError("bootstrap_config must be a LocalizationBootstrapConfig")
    if not isinstance(reference_variant, LocalizationVariant):
        raise TypeError("reference_variant must be a LocalizationVariant")
    if report.dataset_manifest_id != observation_set.dataset_manifest_id:
        raise ValueError("analysis inputs must share one dataset manifest")
    if report.budget_profile_id != observation_set.budget_profile.profile_id:
        raise ValueError("analysis inputs must share one budget profile")
    if report.pricing_table_id != observation_set.pricing_table_id:
        raise ValueError("analysis inputs must share one pricing table")
    if report.observation_ids != tuple(
        item.observation_id for item in observation_set.observations
    ):
        raise ValueError("source report and observation set must contain exact run IDs")
    synthetic = report.dataset_kind is LocalizationDatasetKind.SYNTHETIC_FIXTURE
    if synthetic != (provenance is LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE):
        raise ValueError("synthetic provenance must match synthetic dataset kind")
    if reference_variant not in tuple(item.variant for item in report.aggregates):
        raise ValueError("reference variant is absent from source report")


def _format_interval(item: LocalizationMetricConfidenceInterval) -> str:
    if item.point_estimate_ppm is None:
        return "n/a"
    point = _format_ppm(item.point_estimate_ppm)
    if item.lower_bound_ppm is None or item.upper_bound_ppm is None:
        return f"{point} (CI n/a)"
    return (
        f"{point} [{_format_ppm(item.lower_bound_ppm)}, "
        f"{_format_ppm(item.upper_bound_ppm)}]"
    )


def _format_resource(item: LocalizationResourceAggregate) -> str:
    if item.mean_numerator is None or item.mean_denominator is None:
        return "n/a"
    mean = Decimal(item.mean_numerator) / Decimal(item.mean_denominator)
    return f"{mean:.2f} ({item.observed_observations}/{item.total_observations})"


def _format_ppm(value: int) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    return f"{sign}{absolute // 10_000}.{absolute % 10_000:04d}%"


def _case_balanced_scores(
    metrics: tuple[LocalizationCaseMetrics, ...],
) -> dict[
    tuple[LocalizationVariant, LocalizationAnalysisMetric],
    dict[str, int],
]:
    repeats: dict[
        tuple[LocalizationVariant, LocalizationAnalysisMetric, str],
        list[int],
    ] = defaultdict(list)
    for item in metrics:
        for metric in LocalizationAnalysisMetric:
            rate = _metric_rate(item, metric)
            if rate.availability is LocalizationMetricAvailability.OBSERVED:
                assert rate.value_ppm is not None
                repeats[(item.variant, metric, item.case_id)].append(rate.value_ppm)
    result: dict[
        tuple[LocalizationVariant, LocalizationAnalysisMetric],
        dict[str, int],
    ] = defaultdict(dict)
    for (variant, metric, case_id), values in repeats.items():
        result[(variant, metric)][case_id] = _integer_mean(values)
    return result


def _metric_interval(
    variant: LocalizationVariant,
    metric: LocalizationAnalysisMetric,
    total_cases: int,
    case_values: dict[str, int],
    config: LocalizationBootstrapConfig,
) -> LocalizationMetricConfidenceInterval:
    values = tuple(case_values[key] for key in sorted(case_values))
    point, lower, upper, reason = _estimate(
        values, config, f"interval:{variant.value}:{metric.value}"
    )
    return LocalizationMetricConfidenceInterval(
        variant=variant,
        metric=metric,
        total_case_count=total_cases,
        eligible_case_count=len(values),
        point_estimate_ppm=point,
        lower_bound_ppm=lower,
        upper_bound_ppm=upper,
        unavailable_reason=reason,
    )


def _paired_delta(
    reference: LocalizationVariant,
    candidate: LocalizationVariant,
    metric: LocalizationAnalysisMetric,
    reference_values: dict[str, int],
    candidate_values: dict[str, int],
    config: LocalizationBootstrapConfig,
) -> LocalizationPairedVariantDelta:
    case_ids = tuple(sorted(set(reference_values).intersection(candidate_values)))
    values = tuple(
        candidate_values[case_id] - reference_values[case_id] for case_id in case_ids
    )
    point, lower, upper, reason = _estimate(
        values,
        config,
        f"delta:{reference.value}:{candidate.value}:{metric.value}",
    )
    return LocalizationPairedVariantDelta(
        reference_variant=reference,
        candidate_variant=candidate,
        metric=metric,
        paired_case_count=len(values),
        point_delta_ppm=point,
        lower_bound_ppm=lower,
        upper_bound_ppm=upper,
        unavailable_reason=reason,
    )


def _estimate(
    values: tuple[int, ...],
    config: LocalizationBootstrapConfig,
    stream_key: str,
) -> tuple[
    int | None,
    int | None,
    int | None,
    LocalizationAnalysisUnavailableReason | None,
]:
    if not values:
        return None, None, None, LocalizationAnalysisUnavailableReason.NO_ELIGIBLE_CASES
    point = _integer_mean(values)
    if len(values) == 1:
        return (
            point,
            None,
            None,
            LocalizationAnalysisUnavailableReason.INSUFFICIENT_CASES,
        )
    if len(set(values)) == 1:
        return point, point, point, None
    generator = random.Random(_stream_seed(config.random_seed, stream_key))
    size = len(values)
    samples = sorted(
        _integer_mean(tuple(values[generator.randrange(size)] for _ in values))
        for _ in range(config.resamples)
    )
    tail = (PARTS_PER_MILLION - config.confidence_level_ppm) // 2
    lower_index = tail * (config.resamples - 1) // PARTS_PER_MILLION
    upper_index = min(
        config.resamples - 1,
        ((PARTS_PER_MILLION - tail) * (config.resamples - 1) + PARTS_PER_MILLION - 1)
        // PARTS_PER_MILLION,
    )
    return point, samples[lower_index], samples[upper_index], None


def _case_error(
    metrics: LocalizationCaseMetrics,
    observation: LocalizationRunObservation,
) -> LocalizationCaseErrorAnalysis:
    kinds: list[LocalizationCaseErrorKind] = []
    if observation.status is not LocalizationRunStatus.ACCEPTED:
        kinds.append(LocalizationCaseErrorKind.RUN_NOT_ACCEPTED)
    if metrics.file_hit_at_5.value_ppm == 0:
        kinds.append(LocalizationCaseErrorKind.NO_FILE_HIT_AT_5)
    recall = metrics.file_recall_at_5.value_ppm
    if recall is not None and 0 < recall < PARTS_PER_MILLION:
        kinds.append(LocalizationCaseErrorKind.PARTIAL_FILE_RECALL_AT_5)
    if (
        metrics.symbol_hit_at_5.availability is LocalizationMetricAvailability.OBSERVED
        and metrics.symbol_hit_at_5.value_ppm == 0
    ):
        kinds.append(LocalizationCaseErrorKind.NO_SYMBOL_HIT_AT_5)
    if metrics.prediction_quality_unavailable_reason is not None:
        kinds.append(LocalizationCaseErrorKind.QUALITY_UNAVAILABLE)
    else:
        if metrics.evidence_count == 0:
            kinds.append(LocalizationCaseErrorKind.NO_EVIDENCE)
        elif (
            metrics.evidence_validity.availability
            is LocalizationMetricAvailability.OBSERVED
            and metrics.evidence_validity.value_ppm != PARTS_PER_MILLION
        ):
            kinds.append(LocalizationCaseErrorKind.INVALID_EVIDENCE)
        if (metrics.unsupported_relation_count or 0) > 0:
            kinds.append(LocalizationCaseErrorKind.UNSUPPORTED_RELATION)
    return LocalizationCaseErrorAnalysis(
        observation_id=metrics.observation_id,
        case_id=metrics.case_id,
        task_id=metrics.task_id,
        variant=metrics.variant,
        repeat_index=metrics.repeat_index,
        run_status=observation.status,
        error_kinds=tuple(kinds),
    )


def _resource_aggregate(
    variant: LocalizationVariant,
    metric: LocalizationResourceMetric,
    observations: tuple[LocalizationRunObservation, ...],
) -> LocalizationResourceAggregate:
    matching = tuple(item for item in observations if item.variant is variant)
    values = tuple(
        value
        for item in matching
        if getattr(item, metric.value).availability
        is LocalizationMetricAvailability.OBSERVED
        for value in (getattr(item, metric.value).value,)
        if value is not None
    )
    total = sum(values) if values else None
    return LocalizationResourceAggregate(
        variant=variant,
        metric=metric,
        total_observations=len(matching),
        observed_observations=len(values),
        total_value=total,
        mean_numerator=total,
        mean_denominator=len(values) if values else None,
    )


def _cost_aggregate(
    variant: LocalizationVariant,
    observations: tuple[LocalizationRunObservation, ...],
) -> LocalizationCostAggregate:
    matching = tuple(item for item in observations if item.variant is variant)
    estimated = tuple(
        item
        for item in matching
        if item.cost.availability is CostAvailability.ESTIMATED
    )
    total = None
    if estimated:
        total = _canonical_decimal(
            sum(
                (Decimal(item.cost.estimated_cost_usd or "0") for item in estimated),
                Decimal(0),
            )
        )
    return LocalizationCostAggregate(
        variant=variant,
        total_observations=len(matching),
        estimated_observations=len(estimated),
        unavailable_observations=len(matching) - len(estimated),
        total_estimated_cost_usd=total,
    )


def _metric_rate(
    metrics: LocalizationCaseMetrics,
    metric: LocalizationAnalysisMetric,
) -> LocalizationRate:
    attribute = (
        "reciprocal_rank"
        if metric is LocalizationAnalysisMetric.MEAN_RECIPROCAL_RANK
        else metric.value
    )
    return getattr(metrics, attribute)


def _validate_interval_fields(
    *,
    count: int,
    point: int | None,
    lower: int | None,
    upper: int | None,
    reason: LocalizationAnalysisUnavailableReason | None,
) -> None:
    if count == 0:
        valid = (
            point is None
            and lower is None
            and upper is None
            and reason is LocalizationAnalysisUnavailableReason.NO_ELIGIBLE_CASES
        )
    elif count == 1:
        valid = (
            point is not None
            and lower is None
            and upper is None
            and reason is LocalizationAnalysisUnavailableReason.INSUFFICIENT_CASES
        )
    else:
        valid = (
            point is not None
            and lower is not None
            and upper is not None
            and reason is None
        )
    if not valid:
        raise ValueError("statistical interval availability is inconsistent")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("confidence interval bounds are reversed")


def _integer_mean(values: tuple[int, ...] | list[int]) -> int:
    if not values:
        raise ValueError("cannot average an empty sequence")
    total = sum(values)
    return total // len(values) if total >= 0 else -((-total) // len(values))


def _canonical_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _stream_seed(seed: int, key: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8],
        "big",
    )


def _interval_sort_key(
    item: LocalizationMetricConfidenceInterval,
) -> tuple[int, int]:
    return _variant_index(item.variant), tuple(LocalizationAnalysisMetric).index(
        item.metric
    )


def _delta_sort_key(item: LocalizationPairedVariantDelta) -> tuple[int, int]:
    return _variant_index(item.candidate_variant), tuple(
        LocalizationAnalysisMetric
    ).index(item.metric)


def _case_error_sort_key(
    item: LocalizationCaseErrorAnalysis,
) -> tuple[int, str, int, str]:
    return (
        _variant_index(item.variant),
        item.case_id,
        item.repeat_index,
        item.observation_id,
    )


def _resource_sort_key(item: LocalizationResourceAggregate) -> tuple[int, int]:
    return _variant_index(item.variant), tuple(LocalizationResourceMetric).index(
        item.metric
    )


def _variant_index(variant: LocalizationVariant) -> int:
    return tuple(LocalizationVariant).index(variant)


def _jsonable(value: object) -> object:
    if isinstance(value, CheckpointSafeModel):
        return value.model_dump(mode="json", exclude_computed_fields=True)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _identity(kind: str, fields: object) -> str:
    encoded = json.dumps(
        {"kind": kind, "content": _jsonable(fields)},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_identity(model: CheckpointSafeModel, kind: str, id_field: str) -> str:
    return _identity(
        kind,
        model.model_dump(
            mode="json",
            exclude={id_field},
            exclude_computed_fields=True,
        ),
    )
