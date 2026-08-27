"""Deterministic localization metrics and macro/micro aggregation."""

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation.localization_contracts import (
    LOCALIZATION_SCHEMA_VERSION,
    LOCALIZATION_SCOPE_DISCLAIMER,
    LocalizationCase,
    LocalizationDatasetKind,
    LocalizationDatasetManifest,
    LocalizationMetricAvailability,
    LocalizationMetricUnavailableReason,
    LocalizationRunObservation,
    LocalizationVariant,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest

PARTS_PER_MILLION = 1_000_000


class LocalizationScoreUnavailableReason(StrEnum):
    """Closed denominator and coverage reasons for unavailable scores."""

    NO_GOLD_SYMBOLS = "no_gold_symbols"
    NO_EVIDENCE = "no_evidence"
    NO_RELATIONS = "no_relations"
    OBSERVATION_UNAVAILABLE = "observation_unavailable"
    NO_ELIGIBLE_OBSERVATIONS = "no_eligible_observations"
    NOT_APPLICABLE = "not_applicable"


class _LocalizationMetricModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LocalizationRate(_LocalizationMetricModel):
    """One exact bounded ratio expressed as integer parts per million."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    availability: LocalizationMetricAvailability
    numerator: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    denominator: int | None = Field(default=None, ge=1, le=MAX_SAFE_JSON_INTEGER)
    value_ppm: int | None = Field(default=None, ge=0, le=PARTS_PER_MILLION)
    unavailable_reason: LocalizationScoreUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile_ratio(self) -> "LocalizationRate":
        values = (self.numerator, self.denominator, self.value_ppm)
        if self.availability is LocalizationMetricAvailability.UNAVAILABLE:
            if any(value is not None for value in values):
                raise ValueError("unavailable score must not contain ratio values")
            if self.unavailable_reason is None:
                raise ValueError("unavailable score requires a closed reason")
            return self
        if any(value is None for value in values):
            raise ValueError("observed score requires numerator, denominator, and ppm")
        if self.unavailable_reason is not None:
            raise ValueError("observed score must not contain an unavailable reason")
        assert self.numerator is not None
        assert self.denominator is not None
        assert self.value_ppm is not None
        if self.numerator > self.denominator:
            raise ValueError("score numerator cannot exceed denominator")
        if self.value_ppm != _ppm(self.numerator, self.denominator):
            raise ValueError("score ppm does not match numerator and denominator")
        return self


class LocalizationScalarScore(_LocalizationMetricModel):
    """One aggregate scalar score with explicit unavailable semantics."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    availability: LocalizationMetricAvailability
    value_ppm: int | None = Field(default=None, ge=0, le=PARTS_PER_MILLION)
    unavailable_reason: LocalizationScoreUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile_score(self) -> "LocalizationScalarScore":
        if self.availability is LocalizationMetricAvailability.OBSERVED:
            if self.value_ppm is None or self.unavailable_reason is not None:
                raise ValueError("observed aggregate score requires only a value")
        elif self.value_ppm is not None or self.unavailable_reason is None:
            raise ValueError("unavailable aggregate score requires only a reason")
        return self


class LocalizationCaseMetrics(_LocalizationMetricModel):
    """Recomputable metrics for one immutable case/run observation pair."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    metrics_id: Sha256Digest
    observation_id: Sha256Digest
    case_id: Sha256Digest
    task_id: Sha256Digest
    variant: LocalizationVariant
    repeat_index: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    gold_file_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    gold_symbol_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    matched_file_ranks: tuple[int, ...]
    matched_symbol_ranks: tuple[int, ...]
    prediction_quality_unavailable_reason: (
        LocalizationMetricUnavailableReason | None
    ) = None
    evidence_count: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    valid_evidence_count: int | None = Field(
        default=None, ge=0, le=MAX_SAFE_JSON_INTEGER
    )
    relation_count: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    unsupported_relation_count: int | None = Field(
        default=None, ge=0, le=MAX_SAFE_JSON_INTEGER
    )
    file_hit_at_1: LocalizationRate
    file_hit_at_3: LocalizationRate
    file_hit_at_5: LocalizationRate
    file_recall_at_1: LocalizationRate
    file_recall_at_3: LocalizationRate
    file_recall_at_5: LocalizationRate
    reciprocal_rank: LocalizationRate
    symbol_hit_at_1: LocalizationRate
    symbol_hit_at_3: LocalizationRate
    symbol_hit_at_5: LocalizationRate
    evidence_validity: LocalizationRate
    unsupported_relation_rate: LocalizationRate

    @field_validator("matched_file_ranks", "matched_symbol_ranks")
    @classmethod
    def require_canonical_ranks(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(rank) is not int or rank < 1 for rank in value):
            raise ValueError("matched ranks must be positive integers")
        if value != tuple(sorted(set(value))):
            raise ValueError("matched ranks must be unique and sorted")
        return value

    @model_validator(mode="after")
    def require_recomputable_content_addressed_metrics(
        self,
    ) -> "LocalizationCaseMetrics":
        if len(self.matched_file_ranks) > self.gold_file_count:
            raise ValueError("matched file ranks cannot exceed gold files")
        if len(self.matched_symbol_ranks) > self.gold_symbol_count:
            raise ValueError("matched symbol ranks cannot exceed gold symbols")
        counts = (
            self.evidence_count,
            self.valid_evidence_count,
            self.relation_count,
            self.unsupported_relation_count,
        )
        if self.prediction_quality_unavailable_reason is None:
            if any(value is None for value in counts):
                raise ValueError("observed quality requires all source counts")
            assert self.evidence_count is not None
            assert self.valid_evidence_count is not None
            assert self.relation_count is not None
            assert self.unsupported_relation_count is not None
            if self.valid_evidence_count > self.evidence_count:
                raise ValueError("valid evidence cannot exceed total evidence")
            if self.unsupported_relation_count > self.relation_count:
                raise ValueError("unsupported relations cannot exceed total relations")
        elif any(value is not None for value in counts):
            raise ValueError("unavailable quality must not contain source counts")

        expected = _derive_case_rates(
            gold_file_count=self.gold_file_count,
            gold_symbol_count=self.gold_symbol_count,
            matched_file_ranks=self.matched_file_ranks,
            matched_symbol_ranks=self.matched_symbol_ranks,
            quality_unavailable_reason=self.prediction_quality_unavailable_reason,
            evidence_count=self.evidence_count,
            valid_evidence_count=self.valid_evidence_count,
            relation_count=self.relation_count,
            unsupported_relation_count=self.unsupported_relation_count,
        )
        actual = _case_rate_fields(self)
        if actual != expected:
            raise ValueError(
                "file metrics, symbol metrics, or quality metrics are inconsistent"
            )
        expected_id = _case_metrics_id(
            observation_id=self.observation_id,
            case_id=self.case_id,
            task_id=self.task_id,
            variant=self.variant,
            repeat_index=self.repeat_index,
            gold_file_count=self.gold_file_count,
            gold_symbol_count=self.gold_symbol_count,
            matched_file_ranks=self.matched_file_ranks,
            matched_symbol_ranks=self.matched_symbol_ranks,
            quality_unavailable_reason=self.prediction_quality_unavailable_reason,
            evidence_count=self.evidence_count,
            valid_evidence_count=self.valid_evidence_count,
            relation_count=self.relation_count,
            unsupported_relation_count=self.unsupported_relation_count,
            rates=actual,
        )
        if self.metrics_id != expected_id:
            raise ValueError("metrics_id does not match per-case metrics")
        return self


class LocalizationAggregateRate(_LocalizationMetricModel):
    """Macro score and optional pooled micro ratio for one variant metric."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    total_observations: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    eligible_observations: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    score_sum_ppm: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    macro: LocalizationScalarScore
    micro_numerator: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    micro_denominator: int | None = Field(default=None, ge=1, le=MAX_SAFE_JSON_INTEGER)
    micro: LocalizationScalarScore

    @model_validator(mode="after")
    def reconcile_aggregate(self) -> "LocalizationAggregateRate":
        if self.eligible_observations > self.total_observations:
            raise ValueError("eligible observations cannot exceed total observations")
        maximum_sum = self.eligible_observations * PARTS_PER_MILLION
        if self.score_sum_ppm > maximum_sum:
            raise ValueError("aggregate score sum exceeds eligible observation bound")
        if self.eligible_observations == 0:
            if self.score_sum_ppm != 0 or self.macro != _unavailable_scalar(
                LocalizationScoreUnavailableReason.NO_ELIGIBLE_OBSERVATIONS
            ):
                raise ValueError("empty aggregate must keep macro unavailable")
        elif self.macro != _observed_scalar(
            self.score_sum_ppm // self.eligible_observations
        ):
            raise ValueError("macro score does not match eligible score sum")

        pooled = (self.micro_numerator, self.micro_denominator)
        if pooled == (None, None):
            reason = (
                LocalizationScoreUnavailableReason.NO_ELIGIBLE_OBSERVATIONS
                if self.eligible_observations == 0
                else LocalizationScoreUnavailableReason.NOT_APPLICABLE
            )
            if self.micro != _unavailable_scalar(reason):
                raise ValueError("unpooled micro score has an invalid reason")
        elif any(value is None for value in pooled):
            raise ValueError("micro numerator and denominator must be paired")
        else:
            assert self.micro_numerator is not None
            assert self.micro_denominator is not None
            if self.micro_numerator > self.micro_denominator:
                raise ValueError("micro numerator cannot exceed denominator")
            if self.micro != _observed_scalar(
                _ppm(self.micro_numerator, self.micro_denominator)
            ):
                raise ValueError("micro score does not match pooled ratio")
        return self


class LocalizationVariantAggregate(_LocalizationMetricModel):
    """All deterministic quality aggregates for one architecture variant."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    aggregate_id: Sha256Digest
    variant: LocalizationVariant
    observation_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    unique_case_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    file_hit_at_1: LocalizationAggregateRate
    file_hit_at_3: LocalizationAggregateRate
    file_hit_at_5: LocalizationAggregateRate
    file_recall_at_1: LocalizationAggregateRate
    file_recall_at_3: LocalizationAggregateRate
    file_recall_at_5: LocalizationAggregateRate
    mean_reciprocal_rank: LocalizationAggregateRate
    symbol_hit_at_1: LocalizationAggregateRate
    symbol_hit_at_3: LocalizationAggregateRate
    symbol_hit_at_5: LocalizationAggregateRate
    evidence_validity: LocalizationAggregateRate
    unsupported_relation_rate: LocalizationAggregateRate

    @model_validator(mode="after")
    def require_content_addressed_aggregate(self) -> "LocalizationVariantAggregate":
        if self.unique_case_count > self.observation_count:
            raise ValueError("unique case count cannot exceed observations")
        rates = _aggregate_rate_fields(self)
        if any(rate.total_observations != self.observation_count for rate in rates):
            raise ValueError("aggregate rate totals must match observation_count")
        expected = _variant_aggregate_id(
            variant=self.variant,
            observation_count=self.observation_count,
            unique_case_count=self.unique_case_count,
            rates=rates,
        )
        if self.aggregate_id != expected:
            raise ValueError("aggregate_id does not match variant metrics")
        return self


class LocalizationEvaluationReport(_LocalizationMetricModel):
    """Content-addressed localization-only report for comparable observations."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    report_id: Sha256Digest
    dataset_manifest_id: Sha256Digest
    dataset_name: str
    dataset_version: str
    dataset_kind: LocalizationDatasetKind
    budget_profile_id: Sha256Digest
    pricing_table_id: Sha256Digest | None = None
    case_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    observation_count: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    observation_ids: tuple[Sha256Digest, ...]
    case_metrics: tuple[LocalizationCaseMetrics, ...]
    aggregates: tuple[LocalizationVariantAggregate, ...]
    evaluation_scope: Literal["localization_only"] = "localization_only"
    disclaimer: Literal["swe_bench_derived_localization_not_resolved_rate"] = (
        LOCALIZATION_SCOPE_DISCLAIMER
    )

    @model_validator(mode="after")
    def require_recomputable_content_addressed_report(
        self,
    ) -> "LocalizationEvaluationReport":
        if self.observation_count != len(self.observation_ids):
            raise ValueError("observation_count must match observation IDs")
        if self.observation_count != len(self.case_metrics):
            raise ValueError("observation_count must match case metrics")
        if self.observation_ids != tuple(
            metric.observation_id for metric in self.case_metrics
        ):
            raise ValueError("observation IDs must align with sorted case metrics")
        if len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("observation IDs must be unique")
        expected_order = tuple(sorted(self.case_metrics, key=_case_metric_sort_key))
        if self.case_metrics != expected_order:
            raise ValueError(
                "case metrics must use canonical variant/case/repeat order"
            )
        expected_aggregates = _aggregate_all_variants(self.case_metrics)
        if self.aggregates != expected_aggregates:
            raise ValueError("report aggregates do not match per-case metrics")
        if self.aggregates != tuple(
            sorted(self.aggregates, key=lambda item: _variant_index(item.variant))
        ):
            raise ValueError("report aggregates must use canonical variant order")
        expected_id = _report_id(
            dataset_manifest_id=self.dataset_manifest_id,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            dataset_kind=self.dataset_kind,
            budget_profile_id=self.budget_profile_id,
            pricing_table_id=self.pricing_table_id,
            case_count=self.case_count,
            observation_ids=self.observation_ids,
            case_metrics=self.case_metrics,
            aggregates=self.aggregates,
        )
        if self.report_id != expected_id:
            raise ValueError("report_id does not match localization evaluation report")
        return self


def evaluate_localization_case(
    case: LocalizationCase,
    observation: LocalizationRunObservation,
) -> LocalizationCaseMetrics:
    """Evaluate one ranked prediction against hidden labels without patch access."""

    if not isinstance(case, LocalizationCase):
        raise TypeError("case must be a LocalizationCase")
    if not isinstance(observation, LocalizationRunObservation):
        raise TypeError("observation must be a LocalizationRunObservation")
    if observation.case_id != case.case_id or observation.task_id != case.task.task_id:
        raise ValueError("observation is not bound to the supplied case and task")

    file_ranks = {
        path: rank for rank, path in enumerate(observation.prediction.ranked_files, 1)
    }
    matched_file_ranks = tuple(
        sorted(
            file_ranks[path] for path in case.labels.gold_files if path in file_ranks
        )
    )
    symbol_ranks = {
        (symbol.path, symbol.qualified_name): rank
        for rank, symbol in enumerate(observation.prediction.ranked_symbols, 1)
    }
    matched_symbol_ranks = tuple(
        sorted(
            symbol_ranks[(symbol.path, symbol.qualified_name)]
            for symbol in case.labels.gold_symbols
            if (symbol.path, symbol.qualified_name) in symbol_ranks
        )
    )
    prediction = observation.prediction
    rates = _derive_case_rates(
        gold_file_count=len(case.labels.gold_files),
        gold_symbol_count=len(case.labels.gold_symbols),
        matched_file_ranks=matched_file_ranks,
        matched_symbol_ranks=matched_symbol_ranks,
        quality_unavailable_reason=prediction.quality_unavailable_reason,
        evidence_count=prediction.evidence_count,
        valid_evidence_count=prediction.valid_evidence_count,
        relation_count=prediction.relation_count,
        unsupported_relation_count=prediction.unsupported_relation_count,
    )
    inputs: dict[str, object] = {
        "observation_id": observation.observation_id,
        "case_id": case.case_id,
        "task_id": case.task.task_id,
        "variant": observation.variant,
        "repeat_index": observation.repeat_index,
        "gold_file_count": len(case.labels.gold_files),
        "gold_symbol_count": len(case.labels.gold_symbols),
        "matched_file_ranks": matched_file_ranks,
        "matched_symbol_ranks": matched_symbol_ranks,
        "prediction_quality_unavailable_reason": (
            prediction.quality_unavailable_reason
        ),
        "evidence_count": prediction.evidence_count,
        "valid_evidence_count": prediction.valid_evidence_count,
        "relation_count": prediction.relation_count,
        "unsupported_relation_count": prediction.unsupported_relation_count,
        **rates,
    }
    metrics_id = _case_metrics_id(
        observation_id=observation.observation_id,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=observation.variant,
        repeat_index=observation.repeat_index,
        gold_file_count=len(case.labels.gold_files),
        gold_symbol_count=len(case.labels.gold_symbols),
        matched_file_ranks=matched_file_ranks,
        matched_symbol_ranks=matched_symbol_ranks,
        quality_unavailable_reason=prediction.quality_unavailable_reason,
        evidence_count=prediction.evidence_count,
        valid_evidence_count=prediction.valid_evidence_count,
        relation_count=prediction.relation_count,
        unsupported_relation_count=prediction.unsupported_relation_count,
        rates=rates,
    )
    return LocalizationCaseMetrics(metrics_id=metrics_id, **inputs)


def evaluate_localization_dataset(
    manifest: LocalizationDatasetManifest,
    observations: tuple[LocalizationRunObservation, ...],
) -> LocalizationEvaluationReport:
    """Evaluate comparable observations and reject mixed dataset/budget/pricing."""

    if not isinstance(manifest, LocalizationDatasetManifest):
        raise TypeError("manifest must be a LocalizationDatasetManifest")
    if (
        not isinstance(observations, tuple)
        or not observations
        or not all(
            isinstance(item, LocalizationRunObservation) for item in observations
        )
    ):
        raise TypeError("observations must be a nonempty tuple of run observations")
    if any(item.dataset_manifest_id != manifest.manifest_id for item in observations):
        raise ValueError("every observation must reference the exact dataset manifest")
    budget_ids = {item.budget_profile_id for item in observations}
    if len(budget_ids) != 1:
        raise ValueError("comparison observations must share one budget profile")
    pricing_ids = {item.cost.pricing_table_id for item in observations}
    if len(pricing_ids) != 1:
        raise ValueError("comparison observations must share one pricing table")
    observation_ids = [item.observation_id for item in observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("observation IDs must be unique")
    logical_keys = [
        (item.case_id, item.variant, item.repeat_index) for item in observations
    ]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("each logical observation may occur only once")

    cases = {case.case_id: case for case in manifest.cases}
    metrics = []
    for observation in observations:
        case = cases.get(observation.case_id)
        if case is None:
            raise ValueError("observation references a case outside the manifest")
        metrics.append(evaluate_localization_case(case, observation))
    canonical_metrics = tuple(sorted(metrics, key=_case_metric_sort_key))
    canonical_observation_ids = tuple(
        metric.observation_id for metric in canonical_metrics
    )
    aggregates = _aggregate_all_variants(canonical_metrics)
    budget_profile_id = next(iter(budget_ids))
    pricing_table_id = next(iter(pricing_ids))
    report_id = _report_id(
        dataset_manifest_id=manifest.manifest_id,
        dataset_name=manifest.name,
        dataset_version=manifest.version,
        dataset_kind=manifest.kind,
        budget_profile_id=budget_profile_id,
        pricing_table_id=pricing_table_id,
        case_count=len(manifest.cases),
        observation_ids=canonical_observation_ids,
        case_metrics=canonical_metrics,
        aggregates=aggregates,
    )
    return LocalizationEvaluationReport(
        report_id=report_id,
        dataset_manifest_id=manifest.manifest_id,
        dataset_name=manifest.name,
        dataset_version=manifest.version,
        dataset_kind=manifest.kind,
        budget_profile_id=budget_profile_id,
        pricing_table_id=pricing_table_id,
        case_count=len(manifest.cases),
        observation_count=len(canonical_metrics),
        observation_ids=canonical_observation_ids,
        case_metrics=canonical_metrics,
        aggregates=aggregates,
    )


def _derive_case_rates(
    *,
    gold_file_count: int,
    gold_symbol_count: int,
    matched_file_ranks: tuple[int, ...],
    matched_symbol_ranks: tuple[int, ...],
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None,
    evidence_count: int | None,
    valid_evidence_count: int | None,
    relation_count: int | None,
    unsupported_relation_count: int | None,
) -> dict[str, LocalizationRate]:
    rates: dict[str, LocalizationRate] = {}
    for cutoff in (1, 3, 5):
        file_matches = sum(rank <= cutoff for rank in matched_file_ranks)
        rates[f"file_hit_at_{cutoff}"] = _observed_rate(int(file_matches > 0), 1)
        rates[f"file_recall_at_{cutoff}"] = _observed_rate(
            file_matches, gold_file_count
        )
    first_file_rank = min(matched_file_ranks, default=None)
    rates["reciprocal_rank"] = (
        _observed_rate(0, 1)
        if first_file_rank is None
        else _observed_rate(1, first_file_rank)
    )
    for cutoff in (1, 3, 5):
        rates[f"symbol_hit_at_{cutoff}"] = (
            _unavailable_rate(LocalizationScoreUnavailableReason.NO_GOLD_SYMBOLS)
            if gold_symbol_count == 0
            else _observed_rate(
                int(any(rank <= cutoff for rank in matched_symbol_ranks)), 1
            )
        )
    if quality_unavailable_reason is not None:
        rates["evidence_validity"] = _unavailable_rate(
            LocalizationScoreUnavailableReason.OBSERVATION_UNAVAILABLE
        )
        rates["unsupported_relation_rate"] = _unavailable_rate(
            LocalizationScoreUnavailableReason.OBSERVATION_UNAVAILABLE
        )
        return rates
    assert evidence_count is not None
    assert valid_evidence_count is not None
    assert relation_count is not None
    assert unsupported_relation_count is not None
    rates["evidence_validity"] = (
        _unavailable_rate(LocalizationScoreUnavailableReason.NO_EVIDENCE)
        if evidence_count == 0
        else _observed_rate(valid_evidence_count, evidence_count)
    )
    rates["unsupported_relation_rate"] = (
        _unavailable_rate(LocalizationScoreUnavailableReason.NO_RELATIONS)
        if relation_count == 0
        else _observed_rate(unsupported_relation_count, relation_count)
    )
    return rates


def _aggregate_all_variants(
    case_metrics: tuple[LocalizationCaseMetrics, ...],
) -> tuple[LocalizationVariantAggregate, ...]:
    variants = tuple(
        variant
        for variant in LocalizationVariant
        if any(metric.variant is variant for metric in case_metrics)
    )
    return tuple(
        _aggregate_variant(
            variant,
            tuple(metric for metric in case_metrics if metric.variant is variant),
        )
        for variant in variants
    )


def _aggregate_variant(
    variant: LocalizationVariant,
    metrics: tuple[LocalizationCaseMetrics, ...],
) -> LocalizationVariantAggregate:
    rate_names = (
        "file_hit_at_1",
        "file_hit_at_3",
        "file_hit_at_5",
        "file_recall_at_1",
        "file_recall_at_3",
        "file_recall_at_5",
        "reciprocal_rank",
        "symbol_hit_at_1",
        "symbol_hit_at_3",
        "symbol_hit_at_5",
        "evidence_validity",
        "unsupported_relation_rate",
    )
    aggregates: dict[str, LocalizationAggregateRate] = {}
    for name in rate_names:
        rates = tuple(getattr(metric, name) for metric in metrics)
        aggregates["mean_reciprocal_rank" if name == "reciprocal_rank" else name] = (
            _aggregate_rate(rates, pool=name != "reciprocal_rank")
        )
    rates_tuple = tuple(aggregates[name] for name in _aggregate_output_names())
    unique_case_count = len({metric.case_id for metric in metrics})
    aggregate_id = _variant_aggregate_id(
        variant=variant,
        observation_count=len(metrics),
        unique_case_count=unique_case_count,
        rates=rates_tuple,
    )
    return LocalizationVariantAggregate(
        aggregate_id=aggregate_id,
        variant=variant,
        observation_count=len(metrics),
        unique_case_count=unique_case_count,
        **aggregates,
    )


def _aggregate_rate(
    rates: tuple[LocalizationRate, ...], *, pool: bool
) -> LocalizationAggregateRate:
    observed = tuple(
        rate
        for rate in rates
        if rate.availability is LocalizationMetricAvailability.OBSERVED
    )
    score_sum = sum(rate.value_ppm or 0 for rate in observed)
    macro = (
        _unavailable_scalar(LocalizationScoreUnavailableReason.NO_ELIGIBLE_OBSERVATIONS)
        if not observed
        else _observed_scalar(score_sum // len(observed))
    )
    if observed and pool:
        micro_numerator = sum(rate.numerator or 0 for rate in observed)
        micro_denominator = sum(rate.denominator or 0 for rate in observed)
        micro = _observed_scalar(_ppm(micro_numerator, micro_denominator))
    else:
        micro_numerator = None
        micro_denominator = None
        reason = (
            LocalizationScoreUnavailableReason.NO_ELIGIBLE_OBSERVATIONS
            if not observed
            else LocalizationScoreUnavailableReason.NOT_APPLICABLE
        )
        micro = _unavailable_scalar(reason)
    return LocalizationAggregateRate(
        total_observations=len(rates),
        eligible_observations=len(observed),
        score_sum_ppm=score_sum,
        macro=macro,
        micro_numerator=micro_numerator,
        micro_denominator=micro_denominator,
        micro=micro,
    )


def _observed_rate(numerator: int, denominator: int) -> LocalizationRate:
    return LocalizationRate(
        availability=LocalizationMetricAvailability.OBSERVED,
        numerator=numerator,
        denominator=denominator,
        value_ppm=_ppm(numerator, denominator),
    )


def _unavailable_rate(reason: LocalizationScoreUnavailableReason) -> LocalizationRate:
    return LocalizationRate(
        availability=LocalizationMetricAvailability.UNAVAILABLE,
        unavailable_reason=reason,
    )


def _observed_scalar(value_ppm: int) -> LocalizationScalarScore:
    return LocalizationScalarScore(
        availability=LocalizationMetricAvailability.OBSERVED,
        value_ppm=value_ppm,
    )


def _unavailable_scalar(
    reason: LocalizationScoreUnavailableReason,
) -> LocalizationScalarScore:
    return LocalizationScalarScore(
        availability=LocalizationMetricAvailability.UNAVAILABLE,
        unavailable_reason=reason,
    )


def _ppm(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("score denominator must be positive")
    return numerator * PARTS_PER_MILLION // denominator


def _case_rate_fields(
    metrics: LocalizationCaseMetrics,
) -> dict[str, LocalizationRate]:
    return {
        "file_hit_at_1": metrics.file_hit_at_1,
        "file_hit_at_3": metrics.file_hit_at_3,
        "file_hit_at_5": metrics.file_hit_at_5,
        "file_recall_at_1": metrics.file_recall_at_1,
        "file_recall_at_3": metrics.file_recall_at_3,
        "file_recall_at_5": metrics.file_recall_at_5,
        "reciprocal_rank": metrics.reciprocal_rank,
        "symbol_hit_at_1": metrics.symbol_hit_at_1,
        "symbol_hit_at_3": metrics.symbol_hit_at_3,
        "symbol_hit_at_5": metrics.symbol_hit_at_5,
        "evidence_validity": metrics.evidence_validity,
        "unsupported_relation_rate": metrics.unsupported_relation_rate,
    }


def _aggregate_output_names() -> tuple[str, ...]:
    return (
        "file_hit_at_1",
        "file_hit_at_3",
        "file_hit_at_5",
        "file_recall_at_1",
        "file_recall_at_3",
        "file_recall_at_5",
        "mean_reciprocal_rank",
        "symbol_hit_at_1",
        "symbol_hit_at_3",
        "symbol_hit_at_5",
        "evidence_validity",
        "unsupported_relation_rate",
    )


def _aggregate_rate_fields(
    aggregate: LocalizationVariantAggregate,
) -> tuple[LocalizationAggregateRate, ...]:
    return tuple(getattr(aggregate, name) for name in _aggregate_output_names())


def _case_metric_sort_key(
    metric: LocalizationCaseMetrics,
) -> tuple[int, str, int, str]:
    return (
        _variant_index(metric.variant),
        metric.case_id,
        metric.repeat_index,
        metric.observation_id,
    )


def _variant_index(variant: LocalizationVariant) -> int:
    return tuple(LocalizationVariant).index(variant)


def _case_metrics_id(
    *,
    observation_id: str,
    case_id: str,
    task_id: str,
    variant: LocalizationVariant,
    repeat_index: int,
    gold_file_count: int,
    gold_symbol_count: int,
    matched_file_ranks: tuple[int, ...],
    matched_symbol_ranks: tuple[int, ...],
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None,
    evidence_count: int | None,
    valid_evidence_count: int | None,
    relation_count: int | None,
    unsupported_relation_count: int | None,
    rates: dict[str, LocalizationRate],
) -> str:
    return _content_digest(
        "localization_case_metrics",
        {
            "observation_id": observation_id,
            "case_id": case_id,
            "task_id": task_id,
            "variant": variant.value,
            "repeat_index": repeat_index,
            "gold_file_count": gold_file_count,
            "gold_symbol_count": gold_symbol_count,
            "matched_file_ranks": list(matched_file_ranks),
            "matched_symbol_ranks": list(matched_symbol_ranks),
            "quality_unavailable_reason": (
                None
                if quality_unavailable_reason is None
                else quality_unavailable_reason.value
            ),
            "evidence_count": evidence_count,
            "valid_evidence_count": valid_evidence_count,
            "relation_count": relation_count,
            "unsupported_relation_count": unsupported_relation_count,
            "rates": {name: _model_json(rate) for name, rate in rates.items()},
        },
    )


def _variant_aggregate_id(
    *,
    variant: LocalizationVariant,
    observation_count: int,
    unique_case_count: int,
    rates: tuple[LocalizationAggregateRate, ...],
) -> str:
    return _content_digest(
        "localization_variant_aggregate",
        {
            "variant": variant.value,
            "observation_count": observation_count,
            "unique_case_count": unique_case_count,
            "rates": [_model_json(rate) for rate in rates],
        },
    )


def _report_id(
    *,
    dataset_manifest_id: str,
    dataset_name: str,
    dataset_version: str,
    dataset_kind: LocalizationDatasetKind,
    budget_profile_id: str,
    pricing_table_id: str | None,
    case_count: int,
    observation_ids: tuple[str, ...],
    case_metrics: tuple[LocalizationCaseMetrics, ...],
    aggregates: tuple[LocalizationVariantAggregate, ...],
) -> str:
    return _content_digest(
        "localization_evaluation_report",
        {
            "dataset_manifest_id": dataset_manifest_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_kind": dataset_kind.value,
            "budget_profile_id": budget_profile_id,
            "pricing_table_id": pricing_table_id,
            "case_count": case_count,
            "observation_ids": list(observation_ids),
            "case_metric_ids": [metric.metrics_id for metric in case_metrics],
            "aggregate_ids": [aggregate.aggregate_id for aggregate in aggregates],
            "evaluation_scope": "localization_only",
            "disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
        },
    )


def _content_digest(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {
            "kind": kind,
            "schema_version": LOCALIZATION_SCHEMA_VERSION,
            "payload": payload,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_json(value: _LocalizationMetricModel) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_computed_fields=True)
