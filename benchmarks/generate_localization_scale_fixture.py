"""Generate a 40-case synthetic scale fixture for statistical pipeline closure."""

from __future__ import annotations

import argparse
import hashlib
from decimal import Decimal
from pathlib import Path

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation import (
    LocalizationAnalysisProvenance,
    LocalizationDatasetKind,
    LocalizationLabelSource,
    LocalizationRunStatus,
    LocalizationSymbol,
    LocalizationVariant,
    analyze_localization_evaluation,
    create_default_localization_regression_policy,
    create_estimated_cost_observation,
    create_localization_bootstrap_config,
    create_localization_budget_profile,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_observation_set,
    create_localization_prediction,
    create_localization_run_observation,
    create_localization_task,
    evaluate_localization_dataset,
    evaluate_localization_regression,
    observed_localization_metric,
    render_localization_analysis_summary,
    render_localization_regression_summary,
    render_localization_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_COUNT = 40
PRICING_TABLE_ID = hashlib.sha256(b"synthetic-scale-pricing-v1").hexdigest()
DEFAULT_CASE_OUTPUT = PROJECT_ROOT / "benchmarks/cases/localization-scale-40-v1.json"
DEFAULT_OBSERVATION_OUTPUT = (
    PROJECT_ROOT / "benchmarks/observations/localization-scale-40-v1.json"
)
DEFAULT_REPORT_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.example.json"
)
DEFAULT_REPORT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.example.md"
)
DEFAULT_ANALYSIS_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.analysis.json"
)
DEFAULT_ANALYSIS_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.analysis.md"
)
DEFAULT_REGRESSION_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.regression.json"
)
DEFAULT_REGRESSION_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.regression.md"
)

SUCCESS_COUNTS = {
    LocalizationVariant.TOOL_WRAPPER: 12,
    LocalizationVariant.EVALUATOR_LOOP: 22,
    LocalizationVariant.EVALUATOR_LOOP_SYMBOL_TOOLS: 30,
    LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER: 35,
    LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY: 37,
}
CONTEXT_LINES = (430, 395, 345, 205, 220)
CONTEXT_TOKENS = (7_700, 7_050, 6_150, 3_450, 3_700)
TOOL_CALLS = (12, 11, 9, 8, 6)
LATENCY_MS = (920, 1_170, 1_090, 990, 930)
EVALUATOR_ITERATIONS = (0, 2, 2, 1, 1)
INPUT_TOKENS = (8_100, 9_200, 8_500, 5_150, 5_450)
OUTPUT_TOKENS = (720, 1_020, 920, 770, 720)


def build_scale_artifacts() -> tuple[
    CheckpointSafeModel,
    CheckpointSafeModel,
    CheckpointSafeModel,
    str,
    CheckpointSafeModel,
    str,
    CheckpointSafeModel,
    str,
]:
    """Build 40 cases, 200 observations, reports, and summaries."""

    cases = tuple(_case(index) for index in range(CASE_COUNT))
    manifest = create_localization_dataset_manifest(
        name="localization-scale-fixture-40",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=cases,
    )
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=12,
        max_explorer_calls_per_step=4,
        max_no_progress_rounds=2,
        max_plan_revisions=3,
        context_budget_tokens=8_000,
        max_attempt_cost_usd="0.25",
        max_repository_memory_results=5,
        max_episodic_memory_results=3,
    )
    observations = tuple(
        _observation(
            manifest_id=manifest.manifest_id,
            budget_profile_id=budget.profile_id,
            case=case,
            case_index=case_index,
            variant=variant,
        )
        for variant in LocalizationVariant
        for case_index, case in enumerate(cases)
    )
    observation_set = create_localization_observation_set(
        dataset_manifest_id=manifest.manifest_id,
        budget_profile=budget,
        observations=observations,
    )
    report = evaluate_localization_dataset(manifest, observation_set.observations)
    config = create_localization_bootstrap_config(
        confidence_level_ppm=950_000,
        resamples=2_000,
        random_seed=202_608_28,
    )
    analysis = analyze_localization_evaluation(
        report,
        observation_set,
        provenance=LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE,
        bootstrap_config=config,
        reference_variant=LocalizationVariant.TOOL_WRAPPER,
    )
    regression = evaluate_localization_regression(
        report,
        report,
        policy=create_default_localization_regression_policy(
            maximum_regression_ppm=0,
            minimum_paired_cases=CASE_COUNT,
        ),
        bootstrap_config=config,
    )
    return (
        manifest,
        observation_set,
        report,
        render_localization_summary(report),
        analysis,
        render_localization_analysis_summary(analysis),
        regression,
        render_localization_regression_summary(regression),
    )


def _case(index: int):
    primary_path = f"src/domain_{index:02d}.py"
    gold_files = (primary_path,)
    symbols = (
        LocalizationSymbol(
            path=primary_path,
            qualified_name=f"src.domain_{index:02d}.target",
        ),
    )
    if index % 5 == 0:
        shared_path = f"src/shared_{index % 7:02d}.py"
        gold_files = (*gold_files, shared_path)
        symbols = (
            *symbols,
            LocalizationSymbol(
                path=shared_path,
                qualified_name=f"src.shared_{index % 7:02d}.helper_{index:02d}",
            ),
        )
    task = create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id=f"fixture__scale-{index:03d}",
        repository="example/repocompass-scale-fixture",
        base_commit=hashlib.sha1(
            f"synthetic-scale-commit-{index}".encode(),
            usedforsecurity=False,
        ).hexdigest(),
        objective=(
            f"Synthetic scale case {index:02d}: localize the deterministic "
            "domain ownership path."
        ),
    )
    labels = create_localization_labels(
        task_id=task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest=hashlib.sha256(
            f"synthetic-scale-label-{index}".encode()
        ).hexdigest(),
        gold_files=gold_files,
        gold_symbols=symbols,
    )
    return create_localization_case(task=task, labels=labels)


def _observation(
    *,
    manifest_id: str,
    budget_profile_id: str,
    case,
    case_index: int,
    variant: LocalizationVariant,
):
    variant_index = tuple(LocalizationVariant).index(variant)
    score = (case_index * 17 + 13) % CASE_COUNT
    hit = score < SUCCESS_COUNTS[variant]
    noise_path = f"src/noise_{case_index % 9:02d}.py"
    complete_recall = variant_index >= 2 or case_index % 2 == 0
    rank_first = (case_index + variant_index) % 5 != 0
    if hit:
        selected_gold = (
            case.labels.gold_files if complete_recall else case.labels.gold_files[:1]
        )
        ranked_files = selected_gold if rank_first else (noise_path, *selected_gold)
    else:
        ranked_files = (noise_path,)

    if variant_index == 0 or not hit:
        ranked_symbols = ()
    elif variant_index == 1:
        ranked_symbols = case.labels.gold_symbols[:1] if case_index % 2 == 0 else ()
    elif variant_index == 2:
        ranked_symbols = case.labels.gold_symbols[:1]
    else:
        ranked_symbols = case.labels.gold_symbols

    evidence_count = 2 + variant_index + case_index % 3
    evidence_penalty = max(0, 2 - variant_index) + int(not hit)
    valid_evidence_count = max(0, evidence_count - evidence_penalty)
    relation_count = 0 if variant_index == 0 else variant_index + case_index % 2
    if variant_index == 1:
        unsupported_relation_count = int(case_index % 4 == 0)
    elif variant_index == 2:
        unsupported_relation_count = int(case_index % 11 == 0)
    else:
        unsupported_relation_count = 0

    input_tokens = INPUT_TOKENS[variant_index] + case_index * 7
    output_tokens = OUTPUT_TOKENS[variant_index] + case_index % 17
    estimated_cost = (
        Decimal(input_tokens) + Decimal(output_tokens) * Decimal(2)
    ) / Decimal(1_000_000)
    run_id = f"scale-{variant.value}-{case_index:03d}"
    metric = observed_localization_metric
    return create_localization_run_observation(
        dataset_manifest_id=manifest_id,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=variant,
        budget_profile_id=budget_profile_id,
        run_id=run_id,
        repeat_index=0,
        status=LocalizationRunStatus.ACCEPTED,
        prediction=create_localization_prediction(
            ranked_files=ranked_files,
            ranked_symbols=ranked_symbols,
            evidence_count=evidence_count,
            valid_evidence_count=valid_evidence_count,
            relation_count=relation_count,
            unsupported_relation_count=unsupported_relation_count,
        ),
        context_lines=metric(CONTEXT_LINES[variant_index] + case_index % 13),
        context_tokens=metric(CONTEXT_TOKENS[variant_index] + case_index * 11),
        tool_calls=metric(TOOL_CALLS[variant_index] + case_index % 2),
        latency_ms=metric(LATENCY_MS[variant_index] + case_index * 3),
        evaluator_iterations=metric(EVALUATOR_ITERATIONS[variant_index]),
        input_tokens=metric(input_tokens),
        output_tokens=metric(output_tokens),
        cost=create_estimated_cost_observation(
            pricing_table_id=PRICING_TABLE_ID,
            usage_cost_report_id=hashlib.sha256(run_id.encode()).hexdigest(),
            estimated_cost_usd=_canonical_decimal(estimated_cost),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASE_OUTPUT)
    parser.add_argument(
        "--observation-output",
        type=Path,
        default=DEFAULT_OBSERVATION_OUTPUT,
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--report-summary-output",
        type=Path,
        default=DEFAULT_REPORT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=DEFAULT_ANALYSIS_OUTPUT,
    )
    parser.add_argument(
        "--analysis-summary-output",
        type=Path,
        default=DEFAULT_ANALYSIS_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--regression-output",
        type=Path,
        default=DEFAULT_REGRESSION_OUTPUT,
    )
    parser.add_argument(
        "--regression-summary-output",
        type=Path,
        default=DEFAULT_REGRESSION_SUMMARY_OUTPUT,
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    (
        manifest,
        observations,
        report,
        report_summary,
        analysis,
        analysis_summary,
        regression,
        regression_summary,
    ) = build_scale_artifacts()
    _write_json(manifest, arguments.case_output, force=arguments.force)
    _write_json(
        observations,
        arguments.observation_output,
        force=arguments.force,
    )
    _write_json(report, arguments.report_output, force=arguments.force)
    _write_text(
        report_summary,
        arguments.report_summary_output,
        force=arguments.force,
    )
    _write_json(analysis, arguments.analysis_output, force=arguments.force)
    _write_text(
        analysis_summary,
        arguments.analysis_summary_output,
        force=arguments.force,
    )
    _write_json(regression, arguments.regression_output, force=arguments.force)
    _write_text(
        regression_summary,
        arguments.regression_summary_output,
        force=arguments.force,
    )
    return 0


def _write_json(value: CheckpointSafeModel, path: Path, *, force: bool) -> None:
    mode = "w" if force else "x"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(value.model_dump_json(indent=2, exclude_computed_fields=True))
        stream.write("\n")


def _write_text(value: str, path: Path, *, force: bool) -> None:
    mode = "w" if force else "x"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.write("\n")


def _canonical_decimal(value: Decimal) -> str:
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


if __name__ == "__main__":
    raise SystemExit(main())
