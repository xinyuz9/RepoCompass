"""Confidence-bound localization regression gate tests."""

import pytest
from pydantic import ValidationError

from benchmarks.generate_localization_fixture import build_artifacts
from repocompass.evaluation import (
    LocalizationRunStatus,
    LocalizationVariant,
    create_localization_observation_set,
    create_localization_prediction,
    create_localization_run_observation,
    evaluate_localization_dataset,
)
from repocompass.evaluation.localization_analysis import (
    LocalizationAnalysisMetric,
    create_localization_bootstrap_config,
)
from repocompass.evaluation.localization_regression import (
    LocalizationRegressionReport,
    create_default_localization_regression_policy,
    create_localization_regression_policy,
    create_localization_regression_rule,
    evaluate_localization_regression,
)


def _regressed_report():
    manifest, observation_set, baseline, _ = build_artifacts()
    target = next(
        item
        for item in observation_set.observations
        if item.variant is LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER
    )
    prediction = create_localization_prediction(
        ranked_files=(),
        ranked_symbols=(),
        evidence_count=target.prediction.evidence_count,
        valid_evidence_count=target.prediction.valid_evidence_count,
        relation_count=target.prediction.relation_count,
        unsupported_relation_count=target.prediction.unsupported_relation_count,
    )
    replacement = create_localization_run_observation(
        dataset_manifest_id=target.dataset_manifest_id,
        case_id=target.case_id,
        task_id=target.task_id,
        variant=target.variant,
        budget_profile_id=target.budget_profile_id,
        run_id="candidate-regressed",
        repeat_index=target.repeat_index,
        status=LocalizationRunStatus.ACCEPTED,
        prediction=prediction,
        context_lines=target.context_lines,
        context_tokens=target.context_tokens,
        tool_calls=target.tool_calls,
        latency_ms=target.latency_ms,
        evaluator_iterations=target.evaluator_iterations,
        input_tokens=target.input_tokens,
        output_tokens=target.output_tokens,
        cost=target.cost,
    )
    candidate_set = create_localization_observation_set(
        dataset_manifest_id=observation_set.dataset_manifest_id,
        budget_profile=observation_set.budget_profile,
        observations=tuple(
            replacement if item.observation_id == target.observation_id else item
            for item in observation_set.observations
        ),
    )
    candidate = evaluate_localization_dataset(
        manifest,
        candidate_set.observations,
    )
    return baseline, candidate


def _config():
    return create_localization_bootstrap_config(
        confidence_level_ppm=950_000,
        resamples=500,
        random_seed=29,
    )


def test_identical_report_passes_default_non_inferiority_policy() -> None:
    _, _, report, _ = build_artifacts()
    policy = create_default_localization_regression_policy(
        maximum_regression_ppm=0,
        minimum_paired_cases=2,
    )

    result = evaluate_localization_regression(
        report,
        report,
        policy=policy,
        bootstrap_config=_config(),
    )

    assert result.passed is True
    assert len(result.checks) == 30
    assert all(item.passed for item in result.checks)
    assert all(item.point_regression_ppm == 0 for item in result.checks)
    assert all(item.upper_bound_ppm == 0 for item in result.checks)


def test_upper_confidence_bound_fails_material_file_regression() -> None:
    baseline, candidate = _regressed_report()
    policy = create_localization_regression_policy(
        rules=(
            create_localization_regression_rule(
                variant=LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
                metric=LocalizationAnalysisMetric.FILE_HIT_AT_1,
                maximum_regression_ppm=0,
                minimum_paired_cases=2,
            ),
        )
    )

    result = evaluate_localization_regression(
        baseline,
        candidate,
        policy=policy,
        bootstrap_config=_config(),
    )
    check = result.checks[0]

    assert result.passed is False
    assert check.point_regression_ppm == 500_000
    assert check.lower_bound_ppm == 0
    assert check.upper_bound_ppm == 1_000_000
    assert check.passed is False


def test_regression_gate_rejects_identity_mismatch_and_policy_tampering() -> None:
    _, _, report, _ = build_artifacts()
    policy = create_default_localization_regression_policy(
        maximum_regression_ppm=0,
        minimum_paired_cases=2,
    )
    wrong_dataset = report.model_copy(update={"dataset_manifest_id": "f" * 64})
    with pytest.raises(ValueError, match="dataset manifest"):
        evaluate_localization_regression(
            report,
            wrong_dataset,
            policy=policy,
            bootstrap_config=_config(),
        )

    tampered = policy.model_copy(update={"rules": policy.rules[:1]})
    with pytest.raises(ValidationError, match="policy_id"):
        type(policy).model_validate_json(tampered.model_dump_json())

    result = evaluate_localization_regression(
        report,
        report,
        policy=policy,
        bootstrap_config=_config(),
    )
    changed = result.model_copy(update={"passed": False})
    with pytest.raises(ValidationError, match="passed|regression_id"):
        LocalizationRegressionReport.model_validate_json(changed.model_dump_json())
