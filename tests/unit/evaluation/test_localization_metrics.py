"""Deterministic per-case and aggregate localization metric tests."""

import pytest
from pydantic import ValidationError

from repocompass.agent.cost import CostUnavailableReason
from repocompass.evaluation import (
    LocalizationDatasetKind,
    LocalizationLabelSource,
    LocalizationMetricUnavailableReason,
    LocalizationRunStatus,
    LocalizationScoreUnavailableReason,
    LocalizationSymbol,
    LocalizationVariant,
    create_localization_budget_profile,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_prediction,
    create_localization_run_observation,
    create_localization_task,
    create_unavailable_cost_observation,
    evaluate_localization_case,
    evaluate_localization_dataset,
    observed_localization_metric,
)


def _case(
    *,
    instance_id: str,
    gold_files: tuple[str, ...],
    gold_symbols: tuple[LocalizationSymbol, ...] = (),
):
    task = create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id=instance_id,
        repository="example/fixture",
        base_commit="a" * 40,
        objective=f"Localize {instance_id}.",
    )
    labels = create_localization_labels(
        task_id=task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="b" * 64,
        gold_files=gold_files,
        gold_symbols=gold_symbols,
    )
    return create_localization_case(task=task, labels=labels)


def _budget_id(*, max_tool_calls: int = 8) -> str:
    return create_localization_budget_profile(
        max_tool_calls_per_step=max_tool_calls,
        max_explorer_calls_per_step=3,
        max_no_progress_rounds=2,
        max_plan_revisions=2,
        context_budget_tokens=6_000,
        max_attempt_cost_usd="0.5",
        max_repository_memory_results=5,
        max_episodic_memory_results=3,
    ).profile_id


def _observation(
    *,
    manifest_id: str,
    case,
    ranked_files: tuple[str, ...],
    ranked_symbols: tuple[LocalizationSymbol, ...] = (),
    evidence_count: int | None = 4,
    valid_evidence_count: int | None = 3,
    relation_count: int | None = 4,
    unsupported_relation_count: int | None = 1,
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None = None,
    variant: LocalizationVariant = LocalizationVariant.EVALUATOR_LOOP,
    budget_profile_id: str | None = None,
    pricing_table_id: str = "d" * 64,
    repeat_index: int = 0,
    run_id: str | None = None,
):
    prediction = create_localization_prediction(
        ranked_files=ranked_files,
        ranked_symbols=ranked_symbols,
        evidence_count=evidence_count,
        valid_evidence_count=valid_evidence_count,
        relation_count=relation_count,
        unsupported_relation_count=unsupported_relation_count,
        quality_unavailable_reason=quality_unavailable_reason,
    )
    metric = observed_localization_metric
    return create_localization_run_observation(
        dataset_manifest_id=manifest_id,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=variant,
        budget_profile_id=budget_profile_id or _budget_id(),
        run_id=run_id or f"run-{case.task.instance_id}-{variant.value}",
        repeat_index=repeat_index,
        status=LocalizationRunStatus.ACCEPTED,
        prediction=prediction,
        context_lines=metric(40),
        context_tokens=metric(1_000),
        tool_calls=metric(6),
        latency_ms=metric(500),
        evaluator_iterations=metric(2),
        input_tokens=metric(1_200),
        output_tokens=metric(300),
        cost=create_unavailable_cost_observation(
            pricing_table_id=pricing_table_id,
            reason=CostUnavailableReason.PRICE_NOT_FOUND,
        ),
    )


def test_per_case_file_symbol_quality_and_relation_metrics_are_exact() -> None:
    symbols = (
        LocalizationSymbol(path="src/a.py", qualified_name="a.target"),
        LocalizationSymbol(path="src/z.py", qualified_name="Z.target"),
    )
    case = _case(
        instance_id="fixture__metrics-1",
        gold_files=("src/a.py", "src/z.py"),
        gold_symbols=symbols,
    )
    manifest = create_localization_dataset_manifest(
        name="metrics-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )
    observation = _observation(
        manifest_id=manifest.manifest_id,
        case=case,
        ranked_files=(
            "src/noise.py",
            "src/z.py",
            "src/other.py",
            "src/a.py",
        ),
        ranked_symbols=(
            LocalizationSymbol(path="src/noise.py", qualified_name="noise"),
            symbols[1],
            symbols[0],
        ),
    )

    metrics = evaluate_localization_case(case, observation)

    assert metrics.matched_file_ranks == (2, 4)
    assert metrics.file_hit_at_1.value_ppm == 0
    assert metrics.file_hit_at_3.value_ppm == 1_000_000
    assert metrics.file_hit_at_5.value_ppm == 1_000_000
    assert metrics.file_recall_at_1.value_ppm == 0
    assert metrics.file_recall_at_3.value_ppm == 500_000
    assert metrics.file_recall_at_5.value_ppm == 1_000_000
    assert metrics.reciprocal_rank.value_ppm == 500_000
    assert metrics.symbol_hit_at_1.value_ppm == 0
    assert metrics.symbol_hit_at_3.value_ppm == 1_000_000
    assert metrics.symbol_hit_at_5.value_ppm == 1_000_000
    assert metrics.evidence_validity.value_ppm == 750_000
    assert metrics.unsupported_relation_rate.value_ppm == 250_000


def test_empty_optional_denominators_and_unknown_quality_are_not_reported_as_zero() -> (
    None
):
    case = _case(
        instance_id="fixture__empty-denominators",
        gold_files=("src/a.py",),
    )
    manifest = create_localization_dataset_manifest(
        name="empty-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )
    empty = _observation(
        manifest_id=manifest.manifest_id,
        case=case,
        ranked_files=(),
        evidence_count=0,
        valid_evidence_count=0,
        relation_count=0,
        unsupported_relation_count=0,
    )
    empty_metrics = evaluate_localization_case(case, empty)

    assert empty_metrics.file_hit_at_5.value_ppm == 0
    assert empty_metrics.symbol_hit_at_5.value_ppm is None
    assert empty_metrics.symbol_hit_at_5.unavailable_reason is (
        LocalizationScoreUnavailableReason.NO_GOLD_SYMBOLS
    )
    assert empty_metrics.evidence_validity.value_ppm is None
    assert empty_metrics.evidence_validity.unavailable_reason is (
        LocalizationScoreUnavailableReason.NO_EVIDENCE
    )
    assert empty_metrics.unsupported_relation_rate.value_ppm is None
    assert empty_metrics.unsupported_relation_rate.unavailable_reason is (
        LocalizationScoreUnavailableReason.NO_RELATIONS
    )

    unknown = _observation(
        manifest_id=manifest.manifest_id,
        case=case,
        ranked_files=(),
        evidence_count=None,
        valid_evidence_count=None,
        relation_count=None,
        unsupported_relation_count=None,
        quality_unavailable_reason=(LocalizationMetricUnavailableReason.NOT_COLLECTED),
        repeat_index=1,
    )
    unknown_metrics = evaluate_localization_case(case, unknown)
    assert unknown_metrics.evidence_validity.unavailable_reason is (
        LocalizationScoreUnavailableReason.OBSERVATION_UNAVAILABLE
    )
    assert unknown_metrics.unsupported_relation_rate.unavailable_reason is (
        LocalizationScoreUnavailableReason.OBSERVATION_UNAVAILABLE
    )


def test_dataset_report_computes_macro_micro_and_mrr_without_case_size_bias() -> None:
    first = _case(
        instance_id="fixture__aggregate-1",
        gold_files=("src/a.py", "src/b.py"),
    )
    second = _case(
        instance_id="fixture__aggregate-2",
        gold_files=("src/c.py",),
    )
    manifest = create_localization_dataset_manifest(
        name="aggregate-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(second, first),
    )
    first_observation = _observation(
        manifest_id=manifest.manifest_id,
        case=first,
        ranked_files=("src/noise.py", "src/a.py"),
    )
    second_observation = _observation(
        manifest_id=manifest.manifest_id,
        case=second,
        ranked_files=("src/c.py",),
    )

    report = evaluate_localization_dataset(
        manifest, (second_observation, first_observation)
    )
    aggregate = report.aggregates[0]

    assert report.case_count == 2
    assert report.observation_count == 2
    assert aggregate.file_recall_at_3.macro.value_ppm == 750_000
    assert aggregate.file_recall_at_3.micro.value_ppm == 666_666
    assert aggregate.file_recall_at_3.micro_numerator == 2
    assert aggregate.file_recall_at_3.micro_denominator == 3
    assert aggregate.mean_reciprocal_rank.macro.value_ppm == 750_000
    assert aggregate.mean_reciprocal_rank.micro.value_ppm is None
    assert aggregate.mean_reciprocal_rank.micro.unavailable_reason is (
        LocalizationScoreUnavailableReason.NOT_APPLICABLE
    )


def test_dataset_report_rejects_mixed_identity_and_duplicate_logical_runs() -> None:
    first = _case(instance_id="fixture__identity-1", gold_files=("src/a.py",))
    second = _case(instance_id="fixture__identity-2", gold_files=("src/b.py",))
    manifest = create_localization_dataset_manifest(
        name="identity-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(first, second),
    )
    one = _observation(
        manifest_id=manifest.manifest_id,
        case=first,
        ranked_files=("src/a.py",),
    )
    mixed_budget = _observation(
        manifest_id=manifest.manifest_id,
        case=second,
        ranked_files=("src/b.py",),
        budget_profile_id=_budget_id(max_tool_calls=9),
    )
    with pytest.raises(ValueError, match="budget profile"):
        evaluate_localization_dataset(manifest, (one, mixed_budget))

    mixed_pricing = _observation(
        manifest_id=manifest.manifest_id,
        case=second,
        ranked_files=("src/b.py",),
        pricing_table_id="e" * 64,
    )
    with pytest.raises(ValueError, match="pricing table"):
        evaluate_localization_dataset(manifest, (one, mixed_pricing))

    duplicate_logical_run = _observation(
        manifest_id=manifest.manifest_id,
        case=first,
        ranked_files=("src/noise.py",),
        run_id="different-run-id",
    )
    with pytest.raises(ValueError, match="logical observation"):
        evaluate_localization_dataset(manifest, (one, duplicate_logical_run))

    wrong_manifest = _observation(
        manifest_id="f" * 64,
        case=second,
        ranked_files=("src/b.py",),
    )
    with pytest.raises(ValueError, match="manifest"):
        evaluate_localization_dataset(manifest, (one, wrong_manifest))


def test_metrics_and_report_reject_internal_or_content_identity_tampering() -> None:
    case = _case(instance_id="fixture__tamper", gold_files=("src/a.py",))
    manifest = create_localization_dataset_manifest(
        name="tamper-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )
    observation = _observation(
        manifest_id=manifest.manifest_id,
        case=case,
        ranked_files=("src/a.py",),
    )
    report = evaluate_localization_dataset(manifest, (observation,))

    metrics = report.case_metrics[0]
    tampered_metrics = metrics.model_copy(update={"matched_file_ranks": ()})
    with pytest.raises(ValidationError, match="file metrics"):
        type(metrics).model_validate_json(tampered_metrics.model_dump_json())

    tampered_report = report.model_copy(update={"observation_count": 2})
    with pytest.raises(ValidationError, match="observation_count|report_id"):
        type(report).model_validate_json(tampered_report.model_dump_json())
