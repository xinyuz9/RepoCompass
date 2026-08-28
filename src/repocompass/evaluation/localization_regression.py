"""Confidence-bound non-inferiority gates for localization reports."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation.localization_analysis import (
    LocalizationAnalysisMetric,
    LocalizationBootstrapConfig,
    _case_balanced_scores,
    _estimate,
    _identity,
    _variant_index,
)
from repocompass.evaluation.localization_contracts import (
    LOCALIZATION_SCOPE_DISCLAIMER,
    LocalizationDatasetKind,
    LocalizationVariant,
)
from repocompass.evaluation.localization_metrics import (
    PARTS_PER_MILLION,
    LocalizationEvaluationReport,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest

REGRESSION_SCHEMA_VERSION = 1
DEFAULT_REGRESSION_METRICS = (
    LocalizationAnalysisMetric.FILE_HIT_AT_1,
    LocalizationAnalysisMetric.FILE_HIT_AT_5,
    LocalizationAnalysisMetric.FILE_RECALL_AT_5,
    LocalizationAnalysisMetric.MEAN_RECIPROCAL_RANK,
    LocalizationAnalysisMetric.SYMBOL_HIT_AT_5,
    LocalizationAnalysisMetric.EVIDENCE_VALIDITY,
)


class LocalizationMetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class LocalizationRegressionUnavailableReason(StrEnum):
    NO_PAIRED_CASES = "no_paired_cases"
    INSUFFICIENT_PAIRED_CASES = "insufficient_paired_cases"


class _RegressionModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LocalizationRegressionRule(_RegressionModel):
    schema_version: Literal[1] = REGRESSION_SCHEMA_VERSION
    variant: LocalizationVariant
    metric: LocalizationAnalysisMetric
    direction: LocalizationMetricDirection
    maximum_regression_ppm: int = Field(ge=0, le=PARTS_PER_MILLION)
    minimum_paired_cases: int = Field(ge=2, le=MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def require_metric_direction(self) -> "LocalizationRegressionRule":
        if self.direction is not _metric_direction(self.metric):
            raise ValueError("regression direction does not match metric semantics")
        return self


class LocalizationRegressionPolicy(_RegressionModel):
    schema_version: Literal[1] = REGRESSION_SCHEMA_VERSION
    policy_id: Sha256Digest
    rules: tuple[LocalizationRegressionRule, ...] = Field(min_length=1)
    decision_rule: Literal["upper_confidence_bound_non_inferiority"] = (
        "upper_confidence_bound_non_inferiority"
    )

    @model_validator(mode="after")
    def require_identity_and_order(self) -> "LocalizationRegressionPolicy":
        expected_order = tuple(sorted(self.rules, key=_rule_sort_key))
        if self.rules != expected_order:
            raise ValueError("regression rules must use canonical order")
        identities = tuple((item.variant, item.metric) for item in self.rules)
        if len(identities) != len(set(identities)):
            raise ValueError("regression rules must be unique by variant and metric")
        expected = _model_identity(
            self,
            "localization_regression_policy",
            "policy_id",
        )
        if self.policy_id != expected:
            raise ValueError("policy_id does not match regression policy")
        return self


class LocalizationRegressionCheck(_RegressionModel):
    schema_version: Literal[1] = REGRESSION_SCHEMA_VERSION
    variant: LocalizationVariant
    metric: LocalizationAnalysisMetric
    direction: LocalizationMetricDirection
    paired_case_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    minimum_paired_cases: int = Field(ge=2, le=MAX_SAFE_JSON_INTEGER)
    maximum_regression_ppm: int = Field(ge=0, le=PARTS_PER_MILLION)
    point_regression_ppm: int | None = Field(
        default=None,
        ge=-PARTS_PER_MILLION,
        le=PARTS_PER_MILLION,
    )
    lower_bound_ppm: int | None = Field(
        default=None,
        ge=-PARTS_PER_MILLION,
        le=PARTS_PER_MILLION,
    )
    upper_bound_ppm: int | None = Field(
        default=None,
        ge=-PARTS_PER_MILLION,
        le=PARTS_PER_MILLION,
    )
    passed: bool
    unavailable_reason: LocalizationRegressionUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile(self) -> "LocalizationRegressionCheck":
        if self.direction is not _metric_direction(self.metric):
            raise ValueError("check direction does not match metric semantics")
        if self.paired_case_count == 0:
            valid_values = (
                self.point_regression_ppm is None
                and self.lower_bound_ppm is None
                and self.upper_bound_ppm is None
                and self.unavailable_reason
                is LocalizationRegressionUnavailableReason.NO_PAIRED_CASES
            )
        elif self.paired_case_count == 1:
            valid_values = (
                self.point_regression_ppm is not None
                and self.lower_bound_ppm is None
                and self.upper_bound_ppm is None
                and self.unavailable_reason
                is LocalizationRegressionUnavailableReason.INSUFFICIENT_PAIRED_CASES
            )
        else:
            valid_values = (
                self.point_regression_ppm is not None
                and self.lower_bound_ppm is not None
                and self.upper_bound_ppm is not None
            )
            expected_reason = (
                LocalizationRegressionUnavailableReason.INSUFFICIENT_PAIRED_CASES
                if self.paired_case_count < self.minimum_paired_cases
                else None
            )
            valid_values = valid_values and self.unavailable_reason is expected_reason
        if not valid_values:
            raise ValueError("regression check availability is inconsistent")
        if (
            self.lower_bound_ppm is not None
            and self.upper_bound_ppm is not None
            and self.lower_bound_ppm > self.upper_bound_ppm
        ):
            raise ValueError("regression confidence bounds are reversed")
        expected_pass = (
            self.unavailable_reason is None
            and self.upper_bound_ppm is not None
            and self.upper_bound_ppm <= self.maximum_regression_ppm
        )
        if self.passed is not expected_pass:
            raise ValueError("regression check passed value is inconsistent")
        return self


class LocalizationRegressionReport(_RegressionModel):
    schema_version: Literal[1] = REGRESSION_SCHEMA_VERSION
    regression_id: Sha256Digest
    baseline_report_id: Sha256Digest
    candidate_report_id: Sha256Digest
    dataset_manifest_id: Sha256Digest
    dataset_name: str
    dataset_version: str
    dataset_kind: LocalizationDatasetKind
    budget_profile_id: Sha256Digest
    pricing_table_id: Sha256Digest | None = None
    policy: LocalizationRegressionPolicy
    bootstrap_config: LocalizationBootstrapConfig
    checks: tuple[LocalizationRegressionCheck, ...] = Field(min_length=1)
    passed: bool
    evaluation_scope: Literal["localization_only"] = "localization_only"
    disclaimer: Literal["swe_bench_derived_localization_not_resolved_rate"] = (
        LOCALIZATION_SCOPE_DISCLAIMER
    )

    @model_validator(mode="after")
    def require_identity_and_result(self) -> "LocalizationRegressionReport":
        if self.checks != tuple(sorted(self.checks, key=_check_sort_key)):
            raise ValueError("regression checks must use canonical order")
        if self.passed is not all(item.passed for item in self.checks):
            raise ValueError("regression report passed value is inconsistent")
        expected = _model_identity(
            self,
            "localization_regression_report",
            "regression_id",
        )
        if self.regression_id != expected:
            raise ValueError("regression_id does not match regression report")
        return self


class LocalizationRegressionReportSink:
    """Persist one immutable machine-readable regression decision."""

    def write(self, report: LocalizationRegressionReport, destination: Path) -> Path:
        if not isinstance(report, LocalizationRegressionReport):
            raise TypeError("report must be a LocalizationRegressionReport")
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2, exclude_computed_fields=True))
            stream.write("\n")
        return resolved


def create_localization_regression_rule(
    *,
    variant: LocalizationVariant,
    metric: LocalizationAnalysisMetric,
    maximum_regression_ppm: int,
    minimum_paired_cases: int,
) -> LocalizationRegressionRule:
    return LocalizationRegressionRule(
        variant=variant,
        metric=metric,
        direction=_metric_direction(metric),
        maximum_regression_ppm=maximum_regression_ppm,
        minimum_paired_cases=minimum_paired_cases,
    )


def create_localization_regression_policy(
    *,
    rules: tuple[LocalizationRegressionRule, ...],
) -> LocalizationRegressionPolicy:
    if (
        not isinstance(rules, tuple)
        or not rules
        or not all(isinstance(item, LocalizationRegressionRule) for item in rules)
    ):
        raise TypeError("rules must be a nonempty tuple of regression rules")
    canonical = tuple(sorted(rules, key=_rule_sort_key))
    fields = {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "rules": canonical,
        "decision_rule": "upper_confidence_bound_non_inferiority",
    }
    return LocalizationRegressionPolicy(
        policy_id=_identity("localization_regression_policy", fields),
        **fields,
    )


def create_default_localization_regression_policy(
    *,
    maximum_regression_ppm: int = 0,
    minimum_paired_cases: int = 10,
) -> LocalizationRegressionPolicy:
    return create_localization_regression_policy(
        rules=tuple(
            create_localization_regression_rule(
                variant=variant,
                metric=metric,
                maximum_regression_ppm=maximum_regression_ppm,
                minimum_paired_cases=minimum_paired_cases,
            )
            for variant in LocalizationVariant
            for metric in DEFAULT_REGRESSION_METRICS
        )
    )


def evaluate_localization_regression(
    baseline: LocalizationEvaluationReport,
    candidate: LocalizationEvaluationReport,
    *,
    policy: LocalizationRegressionPolicy,
    bootstrap_config: LocalizationBootstrapConfig,
) -> LocalizationRegressionReport:
    """Evaluate a case-paired confidence-bound non-inferiority policy."""

    _validate_inputs(baseline, candidate, policy, bootstrap_config)
    baseline_scores = _case_balanced_scores(baseline.case_metrics)
    candidate_scores = _case_balanced_scores(candidate.case_metrics)
    checks = tuple(
        _evaluate_rule(
            rule,
            baseline_scores.get((rule.variant, rule.metric), {}),
            candidate_scores.get((rule.variant, rule.metric), {}),
            baseline,
            candidate,
            bootstrap_config,
        )
        for rule in policy.rules
    )
    fields = {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "baseline_report_id": baseline.report_id,
        "candidate_report_id": candidate.report_id,
        "dataset_manifest_id": baseline.dataset_manifest_id,
        "dataset_name": baseline.dataset_name,
        "dataset_version": baseline.dataset_version,
        "dataset_kind": baseline.dataset_kind,
        "budget_profile_id": baseline.budget_profile_id,
        "pricing_table_id": baseline.pricing_table_id,
        "policy": policy,
        "bootstrap_config": bootstrap_config,
        "checks": checks,
        "passed": all(item.passed for item in checks),
        "evaluation_scope": "localization_only",
        "disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
    }
    return LocalizationRegressionReport(
        regression_id=_identity("localization_regression_report", fields),
        **fields,
    )


def render_localization_regression_summary(
    report: LocalizationRegressionReport,
) -> str:
    if not isinstance(report, LocalizationRegressionReport):
        raise TypeError("report must be a LocalizationRegressionReport")
    lines = [
        "# RepoCompass localization regression gate",
        "",
        f"Decision: {'PASS' if report.passed else 'FAIL'}",
        "",
        (
            "Rule: the upper confidence bound of case-paired regression must not "
            "exceed the configured non-inferiority threshold."
        ),
        "",
        "| Variant | Metric | Cases | Regression | Upper bound | Threshold | Result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report.checks:
        lines.append(
            "| "
            + " | ".join(
                (
                    item.variant.value,
                    item.metric.value,
                    str(item.paired_case_count),
                    _format_ppm(item.point_regression_ppm),
                    _format_ppm(item.upper_bound_ppm),
                    _format_ppm(item.maximum_regression_ppm),
                    "PASS" if item.passed else "FAIL",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Positive regression values are worse. Missing or insufficient paired "
            "cases fail closed.",
        )
    )
    return "\n".join(lines)


def _evaluate_rule(
    rule: LocalizationRegressionRule,
    baseline_values: dict[str, int],
    candidate_values: dict[str, int],
    baseline: LocalizationEvaluationReport,
    candidate: LocalizationEvaluationReport,
    config: LocalizationBootstrapConfig,
) -> LocalizationRegressionCheck:
    case_ids = tuple(sorted(set(baseline_values).intersection(candidate_values)))
    if rule.direction is LocalizationMetricDirection.HIGHER_IS_BETTER:
        regressions = tuple(
            baseline_values[case_id] - candidate_values[case_id] for case_id in case_ids
        )
    else:
        regressions = tuple(
            candidate_values[case_id] - baseline_values[case_id] for case_id in case_ids
        )
    point, lower, upper, _ = _estimate(
        regressions,
        config,
        (
            f"regression:{baseline.report_id}:{candidate.report_id}:"
            f"{rule.variant.value}:{rule.metric.value}"
        ),
    )
    if not regressions:
        reason = LocalizationRegressionUnavailableReason.NO_PAIRED_CASES
    elif len(regressions) < rule.minimum_paired_cases:
        reason = LocalizationRegressionUnavailableReason.INSUFFICIENT_PAIRED_CASES
    else:
        reason = None
    passed = (
        reason is None and upper is not None and upper <= rule.maximum_regression_ppm
    )
    return LocalizationRegressionCheck(
        variant=rule.variant,
        metric=rule.metric,
        direction=rule.direction,
        paired_case_count=len(regressions),
        minimum_paired_cases=rule.minimum_paired_cases,
        maximum_regression_ppm=rule.maximum_regression_ppm,
        point_regression_ppm=point,
        lower_bound_ppm=lower,
        upper_bound_ppm=upper,
        passed=passed,
        unavailable_reason=reason,
    )


def _validate_inputs(
    baseline: LocalizationEvaluationReport,
    candidate: LocalizationEvaluationReport,
    policy: LocalizationRegressionPolicy,
    bootstrap_config: LocalizationBootstrapConfig,
) -> None:
    if not isinstance(baseline, LocalizationEvaluationReport):
        raise TypeError("baseline must be a LocalizationEvaluationReport")
    if not isinstance(candidate, LocalizationEvaluationReport):
        raise TypeError("candidate must be a LocalizationEvaluationReport")
    if not isinstance(policy, LocalizationRegressionPolicy):
        raise TypeError("policy must be a LocalizationRegressionPolicy")
    if not isinstance(bootstrap_config, LocalizationBootstrapConfig):
        raise TypeError("bootstrap_config must be a LocalizationBootstrapConfig")
    if baseline.dataset_manifest_id != candidate.dataset_manifest_id:
        raise ValueError("reports must share one dataset manifest")
    if baseline.budget_profile_id != candidate.budget_profile_id:
        raise ValueError("reports must share one budget profile")
    if baseline.pricing_table_id != candidate.pricing_table_id:
        raise ValueError("reports must share one pricing table")
    if baseline.case_count != candidate.case_count:
        raise ValueError("reports must share one manifest case count")


def _metric_direction(
    metric: LocalizationAnalysisMetric,
) -> LocalizationMetricDirection:
    if metric is LocalizationAnalysisMetric.UNSUPPORTED_RELATION_RATE:
        return LocalizationMetricDirection.LOWER_IS_BETTER
    return LocalizationMetricDirection.HIGHER_IS_BETTER


def _rule_sort_key(rule: LocalizationRegressionRule) -> tuple[int, int]:
    return _variant_index(rule.variant), tuple(LocalizationAnalysisMetric).index(
        rule.metric
    )


def _check_sort_key(check: LocalizationRegressionCheck) -> tuple[int, int]:
    return _variant_index(check.variant), tuple(LocalizationAnalysisMetric).index(
        check.metric
    )


def _format_ppm(value: int | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    return f"{sign}{absolute // 10_000}.{absolute % 10_000:04d}%"


def _model_identity(model: CheckpointSafeModel, kind: str, id_field: str) -> str:
    return _identity(
        kind,
        model.model_dump(
            mode="json",
            exclude={id_field},
            exclude_computed_fields=True,
        ),
    )
