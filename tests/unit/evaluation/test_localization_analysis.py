"""Case-balanced localization statistics, error taxonomy, and resource tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.generate_localization_fixture import build_artifacts
from repocompass.evaluation import (
    LocalizationRunStatus,
    LocalizationVariant,
    create_localization_prediction,
    create_localization_run_observation,
    create_localization_observation_set,
    evaluate_localization_dataset,
)
from repocompass.evaluation.localization_analysis import (
    LocalizationAnalysisMetric,
    LocalizationAnalysisProvenance,
    LocalizationAnalysisReport,
    LocalizationAnalysisUnavailableReason,
    LocalizationCaseErrorKind,
    LocalizationResourceMetric,
    analyze_localization_evaluation,
    create_localization_bootstrap_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _find_interval(analysis, variant, metric):
    return next(
        item
        for item in analysis.metric_intervals
        if item.variant is variant and item.metric is metric
    )


def _find_resource(analysis, variant, metric):
    return next(
        item
        for item in analysis.resource_aggregates
        if item.variant is variant and item.metric is metric
    )


def _analysis():
    _, observation_set, report, _ = build_artifacts()
    config = create_localization_bootstrap_config(
        confidence_level_ppm=950_000,
        resamples=500,
        random_seed=17,
    )
    return analyze_localization_evaluation(
        report,
        observation_set,
        provenance=LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE,
        bootstrap_config=config,
        reference_variant=LocalizationVariant.TOOL_WRAPPER,
    )


def test_analysis_is_deterministic_case_balanced_and_paired() -> None:
    first = _analysis()
    second = _analysis()

    assert first == second
    assert first.analysis_id == (
        "09f404a259a3e3f1041f785f57bedcbfe6c61f54ac76c52b63a0805ee6089c15"
    )
    assert len(first.metric_intervals) == 60
    assert len(first.paired_deltas) == 48
    assert first.bootstrap_config.resamples == 500
    assert first.reference_variant is LocalizationVariant.TOOL_WRAPPER

    evaluator_hit = _find_interval(
        first,
        LocalizationVariant.EVALUATOR_LOOP,
        LocalizationAnalysisMetric.FILE_HIT_AT_1,
    )
    assert evaluator_hit.total_case_count == 2
    assert evaluator_hit.eligible_case_count == 2
    assert evaluator_hit.point_estimate_ppm == 500_000
    assert evaluator_hit.lower_bound_ppm == 0
    assert evaluator_hit.upper_bound_ppm == 1_000_000

    full_delta = next(
        item
        for item in first.paired_deltas
        if item.candidate_variant is LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER
        and item.metric is LocalizationAnalysisMetric.FILE_HIT_AT_1
    )
    assert full_delta.reference_variant is LocalizationVariant.TOOL_WRAPPER
    assert full_delta.paired_case_count == 2
    assert full_delta.point_delta_ppm == 1_000_000
    assert full_delta.lower_bound_ppm == 1_000_000
    assert full_delta.upper_bound_ppm == 1_000_000


def test_case_balance_prevents_repeats_from_overweighting_one_case() -> None:
    manifest, observation_set, _, _ = build_artifacts()
    tool_runs = tuple(
        item
        for item in observation_set.observations
        if item.variant is LocalizationVariant.TOOL_WRAPPER
    )
    repeated = tool_runs[0]
    case = next(item for item in manifest.cases if item.case_id == repeated.case_id)
    perfect_prediction = create_localization_prediction(
        ranked_files=case.labels.gold_files,
        ranked_symbols=case.labels.gold_symbols,
        evidence_count=1,
        valid_evidence_count=1,
        relation_count=0,
        unsupported_relation_count=0,
    )
    perfect_repeat = create_localization_run_observation(
        dataset_manifest_id=repeated.dataset_manifest_id,
        case_id=repeated.case_id,
        task_id=repeated.task_id,
        variant=repeated.variant,
        budget_profile_id=repeated.budget_profile_id,
        run_id="fixture-repeat-perfect",
        repeat_index=1,
        status=LocalizationRunStatus.ACCEPTED,
        prediction=perfect_prediction,
        context_lines=repeated.context_lines,
        context_tokens=repeated.context_tokens,
        tool_calls=repeated.tool_calls,
        latency_ms=repeated.latency_ms,
        evaluator_iterations=repeated.evaluator_iterations,
        input_tokens=repeated.input_tokens,
        output_tokens=repeated.output_tokens,
        cost=repeated.cost,
    )
    repeated_set = create_localization_observation_set(
        dataset_manifest_id=manifest.manifest_id,
        budget_profile=observation_set.budget_profile,
        observations=(*tool_runs, perfect_repeat),
    )
    report = evaluate_localization_dataset(manifest, repeated_set.observations)
    analysis = analyze_localization_evaluation(
        report,
        repeated_set,
        provenance=LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE,
        bootstrap_config=create_localization_bootstrap_config(
            confidence_level_ppm=950_000,
            resamples=100,
            random_seed=1,
        ),
        reference_variant=LocalizationVariant.TOOL_WRAPPER,
    )

    interval = _find_interval(
        analysis,
        LocalizationVariant.TOOL_WRAPPER,
        LocalizationAnalysisMetric.FILE_HIT_AT_1,
    )
    assert report.aggregates[0].file_hit_at_1.macro.value_ppm == 333_333
    assert interval.point_estimate_ppm == 250_000


def test_analysis_preserves_case_error_taxonomy_without_guessing() -> None:
    analysis = _analysis()
    tool_errors = tuple(
        item
        for item in analysis.case_errors
        if item.variant is LocalizationVariant.TOOL_WRAPPER
    )
    assert len(tool_errors) == 2

    assert {item.error_kinds for item in tool_errors} == {
        (
            LocalizationCaseErrorKind.PARTIAL_FILE_RECALL_AT_5,
            LocalizationCaseErrorKind.NO_SYMBOL_HIT_AT_5,
            LocalizationCaseErrorKind.INVALID_EVIDENCE,
        ),
        (
            LocalizationCaseErrorKind.NO_FILE_HIT_AT_5,
            LocalizationCaseErrorKind.NO_SYMBOL_HIT_AT_5,
            LocalizationCaseErrorKind.INVALID_EVIDENCE,
        ),
    }


def test_analysis_aggregates_operational_metrics_and_cost_coverage() -> None:
    analysis = _analysis()
    context_tokens = _find_resource(
        analysis,
        LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
        LocalizationResourceMetric.CONTEXT_TOKENS,
    )
    assert context_tokens.total_observations == 2
    assert context_tokens.observed_observations == 2
    assert context_tokens.total_value == 6_400
    assert context_tokens.mean_numerator == 6_400
    assert context_tokens.mean_denominator == 2

    assert len(analysis.cost_aggregates) == 5
    assert all(item.estimated_observations == 0 for item in analysis.cost_aggregates)
    assert all(
        item.total_estimated_cost_usd is None for item in analysis.cost_aggregates
    )


def test_bootstrap_marks_one_case_interval_as_insufficient() -> None:
    _, observation_set, _, _ = build_artifacts()
    observation = observation_set.observations[0]
    single_set = create_localization_observation_set(
        dataset_manifest_id=observation_set.dataset_manifest_id,
        budget_profile=observation_set.budget_profile,
        observations=(observation,),
    )
    manifest, _, _, _ = build_artifacts()
    report = evaluate_localization_dataset(manifest, single_set.observations)
    analysis = analyze_localization_evaluation(
        report,
        single_set,
        provenance=LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE,
        bootstrap_config=create_localization_bootstrap_config(
            confidence_level_ppm=950_000,
            resamples=100,
            random_seed=1,
        ),
        reference_variant=LocalizationVariant.TOOL_WRAPPER,
    )

    interval = _find_interval(
        analysis,
        LocalizationVariant.TOOL_WRAPPER,
        LocalizationAnalysisMetric.FILE_HIT_AT_1,
    )
    assert interval.point_estimate_ppm == 0
    assert interval.lower_bound_ppm is None
    assert interval.upper_bound_ppm is None
    assert interval.unavailable_reason is (
        LocalizationAnalysisUnavailableReason.INSUFFICIENT_CASES
    )


def test_analysis_rejects_false_provenance_and_identity_tampering() -> None:
    manifest, observation_set, report, _ = build_artifacts()
    config = create_localization_bootstrap_config(
        confidence_level_ppm=950_000,
        resamples=100,
        random_seed=1,
    )
    assert manifest.kind.value == "synthetic_fixture"
    with pytest.raises(ValueError, match="synthetic"):
        analyze_localization_evaluation(
            report,
            observation_set,
            provenance=LocalizationAnalysisProvenance.DETERMINISTIC_REPLAY,
            bootstrap_config=config,
            reference_variant=LocalizationVariant.TOOL_WRAPPER,
        )

    analysis = _analysis()
    tampered = analysis.model_copy(update={"source_report_id": "f" * 64})
    with pytest.raises(ValidationError, match="analysis_id|source report"):
        LocalizationAnalysisReport.model_validate_json(tampered.model_dump_json())
