"""Strict localization dataset and run-observation contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.agent.cost import CostUnavailableReason
from repocompass.evaluation import (
    MAX_LOCALIZATION_DATASET_FILE_BYTES,
    LocalizationCase,
    LocalizationDatasetKind,
    LocalizationDatasetManifest,
    LocalizationLabelSource,
    LocalizationLabels,
    LocalizationMetricUnavailableReason,
    LocalizationRunObservation,
    LocalizationRunStatus,
    LocalizationSymbol,
    LocalizationTask,
    LocalizationVariant,
    create_localization_budget_profile,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_prediction,
    create_localization_run_observation,
    create_localization_task,
    create_unavailable_cost_observation,
    load_localization_dataset_file,
    observed_localization_metric,
    unavailable_localization_metric,
)


def _task(*, instance_id: str = "fixture__issue-1") -> LocalizationTask:
    return create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id=instance_id,
        repository="example/repocompass-fixture",
        base_commit="a" * 40,
        objective="Find the files and symbols responsible for retry routing.",
    )


def _labels(task: LocalizationTask) -> LocalizationLabels:
    return create_localization_labels(
        task_id=task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="b" * 64,
        gold_files=("src/pkg/z.py", "src/pkg/a.py"),
        gold_symbols=(
            LocalizationSymbol(path="src/pkg/z.py", qualified_name="Router.retry"),
            LocalizationSymbol(path="src/pkg/a.py", qualified_name="route_request"),
        ),
    )


def _case(*, instance_id: str = "fixture__issue-1") -> LocalizationCase:
    task = _task(instance_id=instance_id)
    return create_localization_case(task=task, labels=_labels(task))


def _observation(case: LocalizationCase) -> LocalizationRunObservation:
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=8,
        max_explorer_calls_per_step=3,
        max_no_progress_rounds=2,
        max_plan_revisions=2,
        context_budget_tokens=6_000,
        max_attempt_cost_usd="0.5",
        max_repository_memory_results=5,
        max_episodic_memory_results=3,
    )
    prediction = create_localization_prediction(
        ranked_files=("src/pkg/a.py", "src/pkg/z.py"),
        ranked_symbols=(
            LocalizationSymbol(path="src/pkg/a.py", qualified_name="route_request"),
        ),
        evidence_count=2,
        valid_evidence_count=2,
        relation_count=1,
        unsupported_relation_count=0,
    )
    return create_localization_run_observation(
        dataset_manifest_id="c" * 64,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
        budget_profile_id=budget.profile_id,
        run_id="fixture-run-1",
        repeat_index=0,
        status=LocalizationRunStatus.ACCEPTED,
        prediction=prediction,
        context_lines=observed_localization_metric(48),
        context_tokens=observed_localization_metric(1_220),
        tool_calls=observed_localization_metric(6),
        latency_ms=observed_localization_metric(340),
        evaluator_iterations=observed_localization_metric(2),
        input_tokens=unavailable_localization_metric(
            LocalizationMetricUnavailableReason.PROVIDER_NOT_REPORTED
        ),
        output_tokens=unavailable_localization_metric(
            LocalizationMetricUnavailableReason.PROVIDER_NOT_REPORTED
        ),
        cost=create_unavailable_cost_observation(
            pricing_table_id="d" * 64,
            reason=CostUnavailableReason.NO_INVOCATIONS,
        ),
    )


def test_task_and_hidden_labels_have_stable_separate_identities() -> None:
    case = _case()
    labels_reordered = create_localization_labels(
        task_id=case.task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="b" * 64,
        gold_files=tuple(reversed(case.labels.gold_files)),
        gold_symbols=tuple(reversed(case.labels.gold_symbols)),
    )

    assert labels_reordered == case.labels
    assert case.labels.gold_files == ("src/pkg/a.py", "src/pkg/z.py")
    assert tuple(item.qualified_name for item in case.labels.gold_symbols) == (
        "route_request",
        "Router.retry",
    )

    manifest = create_localization_dataset_manifest(
        name="fixture-localization",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )
    agent_payload = json.dumps(
        [task.model_dump(mode="json") for task in manifest.agent_tasks()],
        sort_keys=True,
    )
    assert "gold_files" not in agent_payload
    assert "gold_symbols" not in agent_payload
    assert case.labels.source_artifact_digest not in agent_payload


def test_content_ids_reject_task_label_case_manifest_and_observation_tampering() -> (
    None
):
    case = _case()
    manifest = create_localization_dataset_manifest(
        name="fixture-localization",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )
    observation = _observation(case)

    tampered_task = case.task.model_copy(update={"objective": "changed"})
    with pytest.raises(ValidationError, match="task_id"):
        LocalizationTask.model_validate_json(tampered_task.model_dump_json())

    tampered_labels = case.labels.model_copy(
        update={"source_artifact_digest": "e" * 64}
    )
    with pytest.raises(ValidationError, match="label_id"):
        LocalizationLabels.model_validate_json(tampered_labels.model_dump_json())

    tampered_case = case.model_copy(update={"labels": tampered_labels})
    with pytest.raises(ValidationError, match="labels|case_id"):
        LocalizationCase.model_validate_json(tampered_case.model_dump_json())

    tampered_manifest = manifest.model_copy(update={"version": "v2"})
    with pytest.raises(ValidationError, match="manifest_id"):
        LocalizationDatasetManifest.model_validate_json(
            tampered_manifest.model_dump_json()
        )

    tampered_observation = observation.model_copy(update={"repeat_index": 1})
    with pytest.raises(ValidationError, match="observation_id"):
        LocalizationRunObservation.model_validate_json(
            tampered_observation.model_dump_json()
        )


@pytest.mark.parametrize(
    "path",
    ("/src/a.py", "src/../a.py", "src\\a.py", "src//a.py", "./src/a.py"),
)
def test_repository_paths_fail_closed_in_labels_and_predictions(path: str) -> None:
    task = _task()
    with pytest.raises(ValidationError, match="normalized repository-relative"):
        create_localization_labels(
            task_id=task.task_id,
            source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
            source_artifact_digest="b" * 64,
            gold_files=(path,),
            gold_symbols=(),
        )

    with pytest.raises(ValidationError, match="normalized repository-relative"):
        create_localization_prediction(
            ranked_files=(path,),
            ranked_symbols=(),
            evidence_count=0,
            valid_evidence_count=0,
            relation_count=0,
            unsupported_relation_count=0,
        )


def test_canonical_duplicates_and_cross_file_symbols_are_rejected() -> None:
    task = _task()
    with pytest.raises(ValidationError, match="unique"):
        LocalizationLabels(
            label_id="0" * 64,
            task_id=task.task_id,
            source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
            source_artifact_digest="b" * 64,
            gold_files=("src/a.py", "src/a.py"),
            gold_symbols=(),
        )

    with pytest.raises(ValidationError, match="gold file"):
        create_localization_labels(
            task_id=task.task_id,
            source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
            source_artifact_digest="b" * 64,
            gold_files=("src/a.py",),
            gold_symbols=(
                LocalizationSymbol(path="src/b.py", qualified_name="outside"),
            ),
        )

    with pytest.raises(ValidationError, match="unique"):
        create_localization_prediction(
            ranked_files=("src/a.py", "src/a.py"),
            ranked_symbols=(),
            evidence_count=0,
            valid_evidence_count=0,
            relation_count=0,
            unsupported_relation_count=0,
        )


def test_observation_keeps_unavailable_metrics_unknown_and_reconciles_quality() -> None:
    case = _case()
    observation = _observation(case)

    assert observation.input_tokens.value is None
    assert observation.input_tokens.unavailable_reason is (
        LocalizationMetricUnavailableReason.PROVIDER_NOT_REPORTED
    )
    assert observation.cost.estimated_cost_usd is None
    assert observation.prediction.valid_evidence_count == 2

    inconsistent = observation.prediction.model_copy(update={"valid_evidence_count": 3})
    with pytest.raises(ValidationError, match="valid evidence"):
        type(observation.prediction).model_validate_json(inconsistent.model_dump_json())


def test_manifest_loader_is_bounded_duplicate_key_free_and_canonical(
    tmp_path: Path,
) -> None:
    first = _case(instance_id="fixture__issue-1")
    second = _case(instance_id="fixture__issue-2")
    manifest = create_localization_dataset_manifest(
        name="fixture-localization",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(second, first),
    )
    assert tuple(case.task.instance_id for case in manifest.cases) == (
        "fixture__issue-1",
        "fixture__issue-2",
    )

    valid = tmp_path / "manifest.json"
    valid.write_text(
        manifest.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )
    assert load_localization_dataset_file(valid) == manifest

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_localization_dataset_file(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_LOCALIZATION_DATASET_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_localization_dataset_file(oversized)


def test_variant_vocabulary_matches_the_five_planned_comparisons() -> None:
    assert tuple(LocalizationVariant) == (
        LocalizationVariant.TOOL_WRAPPER,
        LocalizationVariant.EVALUATOR_LOOP,
        LocalizationVariant.EVALUATOR_LOOP_SYMBOL_TOOLS,
        LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
        LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY,
    )
