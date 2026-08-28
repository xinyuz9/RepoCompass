"""Regenerate the deterministic five-variant localization fixture."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from repocompass.agent.cost import CostUnavailableReason
from repocompass.evaluation import (
    LocalizationDatasetKind,
    LocalizationLabelSource,
    LocalizationRunStatus,
    LocalizationSymbol,
    LocalizationVariant,
    create_localization_budget_profile,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_observation_set,
    create_localization_prediction,
    create_localization_run_observation,
    create_localization_task,
    create_unavailable_cost_observation,
    evaluate_localization_dataset,
    observed_localization_metric,
    render_localization_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_OUTPUT = PROJECT_ROOT / "benchmarks/cases/localization-fixture-v1.json"
DEFAULT_OBSERVATION_OUTPUT = (
    PROJECT_ROOT / "benchmarks/observations/localization-fixture-v1.json"
)
DEFAULT_REPORT_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.json"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.md"
)
PRICING_TABLE_ID = "d" * 64


@dataclass(frozen=True, slots=True)
class _VariantFixture:
    files_by_case: tuple[tuple[str, ...], tuple[str, ...]]
    symbols_by_case: tuple[
        tuple[LocalizationSymbol, ...], tuple[LocalizationSymbol, ...]
    ]
    valid_evidence: tuple[int, int]
    evidence: tuple[int, int]
    unsupported_relations: tuple[int, int]
    relations: tuple[int, int]
    context_lines: int
    context_tokens: int
    tool_calls: int
    latency_ms: int
    evaluator_iterations: int
    input_tokens: int
    output_tokens: int


def _symbol(path: str, name: str) -> LocalizationSymbol:
    return LocalizationSymbol(path=path, qualified_name=name)


ROUTER_SYMBOL = _symbol("src/router.py", "src.router.Router.retry")
CONTROLLER_SYMBOL = _symbol("src/controller.py", "src.controller.route_request")
COST_SYMBOL = _symbol("src/cost.py", "src.cost.build_cost_report")
NOISE_SYMBOL = _symbol("src/noise.py", "src.noise.unrelated")


VARIANT_FIXTURES = {
    LocalizationVariant.TOOL_WRAPPER: _VariantFixture(
        files_by_case=(("src/noise.py", "src/router.py"), ("src/noise.py",)),
        symbols_by_case=((), ()),
        valid_evidence=(1, 0),
        evidence=(2, 1),
        unsupported_relations=(0, 0),
        relations=(0, 0),
        context_lines=420,
        context_tokens=7_500,
        tool_calls=12,
        latency_ms=900,
        evaluator_iterations=0,
        input_tokens=8_000,
        output_tokens=700,
    ),
    LocalizationVariant.EVALUATOR_LOOP: _VariantFixture(
        files_by_case=(
            ("src/router.py", "src/noise.py", "src/controller.py"),
            ("src/noise.py", "src/cost.py"),
        ),
        symbols_by_case=((ROUTER_SYMBOL,), ()),
        valid_evidence=(2, 1),
        evidence=(3, 2),
        unsupported_relations=(1, 1),
        relations=(2, 1),
        context_lines=390,
        context_tokens=6_900,
        tool_calls=11,
        latency_ms=1_150,
        evaluator_iterations=2,
        input_tokens=9_100,
        output_tokens=1_000,
    ),
    LocalizationVariant.EVALUATOR_LOOP_SYMBOL_TOOLS: _VariantFixture(
        files_by_case=(
            ("src/router.py", "src/controller.py"),
            ("src/cost.py", "src/noise.py"),
        ),
        symbols_by_case=(
            (ROUTER_SYMBOL, CONTROLLER_SYMBOL),
            (COST_SYMBOL,),
        ),
        valid_evidence=(3, 2),
        evidence=(3, 2),
        unsupported_relations=(0, 0),
        relations=(2, 1),
        context_lines=350,
        context_tokens=6_100,
        tool_calls=9,
        latency_ms=1_080,
        evaluator_iterations=2,
        input_tokens=8_400,
        output_tokens=900,
    ),
    LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER: _VariantFixture(
        files_by_case=(
            ("src/router.py", "src/controller.py"),
            ("src/cost.py",),
        ),
        symbols_by_case=(
            (ROUTER_SYMBOL, CONTROLLER_SYMBOL),
            (COST_SYMBOL,),
        ),
        valid_evidence=(3, 2),
        evidence=(3, 2),
        unsupported_relations=(0, 0),
        relations=(2, 1),
        context_lines=180,
        context_tokens=3_200,
        tool_calls=8,
        latency_ms=980,
        evaluator_iterations=1,
        input_tokens=5_000,
        output_tokens=750,
    ),
    LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY: _VariantFixture(
        files_by_case=(
            ("src/router.py", "src/controller.py"),
            ("src/cost.py",),
        ),
        symbols_by_case=(
            (ROUTER_SYMBOL, CONTROLLER_SYMBOL),
            (COST_SYMBOL,),
        ),
        valid_evidence=(3, 2),
        evidence=(3, 2),
        unsupported_relations=(0, 0),
        relations=(2, 1),
        context_lines=205,
        context_tokens=3_650,
        tool_calls=6,
        latency_ms=920,
        evaluator_iterations=1,
        input_tokens=5_350,
        output_tokens=700,
    ),
}


def _cases():
    first_task = create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id="fixture__routing-1",
        repository="example/repocompass-fixture",
        base_commit="a" * 40,
        objective="Locate retry routing and controller ownership.",
    )
    first_labels = create_localization_labels(
        task_id=first_task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="b" * 64,
        gold_files=("src/router.py", "src/controller.py"),
        gold_symbols=(ROUTER_SYMBOL, CONTROLLER_SYMBOL),
    )
    second_task = create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id="fixture__cost-2",
        repository="example/repocompass-fixture",
        base_commit="a" * 40,
        objective="Locate the attempt cost aggregation entry point.",
    )
    second_labels = create_localization_labels(
        task_id=second_task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="c" * 64,
        gold_files=("src/cost.py",),
        gold_symbols=(COST_SYMBOL,),
    )
    return (
        create_localization_case(task=first_task, labels=first_labels),
        create_localization_case(task=second_task, labels=second_labels),
    )


def build_artifacts():
    """Build the manifest, five-variant observations, report, and summary."""

    cases = _cases()
    manifest = create_localization_dataset_manifest(
        name="localization-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=cases,
    )
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
    observations = []
    for variant, fixture in VARIANT_FIXTURES.items():
        for case_index, case in enumerate(cases):
            prediction = create_localization_prediction(
                ranked_files=fixture.files_by_case[case_index],
                ranked_symbols=fixture.symbols_by_case[case_index],
                evidence_count=fixture.evidence[case_index],
                valid_evidence_count=fixture.valid_evidence[case_index],
                relation_count=fixture.relations[case_index],
                unsupported_relation_count=(fixture.unsupported_relations[case_index]),
            )
            metric = observed_localization_metric
            observations.append(
                create_localization_run_observation(
                    dataset_manifest_id=manifest.manifest_id,
                    case_id=case.case_id,
                    task_id=case.task.task_id,
                    variant=variant,
                    budget_profile_id=budget.profile_id,
                    run_id=f"fixture-{variant.value}-{case_index + 1}",
                    repeat_index=0,
                    status=LocalizationRunStatus.ACCEPTED,
                    prediction=prediction,
                    context_lines=metric(fixture.context_lines),
                    context_tokens=metric(fixture.context_tokens),
                    tool_calls=metric(fixture.tool_calls),
                    latency_ms=metric(fixture.latency_ms),
                    evaluator_iterations=metric(fixture.evaluator_iterations),
                    input_tokens=metric(fixture.input_tokens),
                    output_tokens=metric(fixture.output_tokens),
                    cost=create_unavailable_cost_observation(
                        pricing_table_id=PRICING_TABLE_ID,
                        reason=CostUnavailableReason.PRICE_NOT_FOUND,
                    ),
                )
            )
    observation_set = create_localization_observation_set(
        dataset_manifest_id=manifest.manifest_id,
        budget_profile=budget,
        observations=tuple(observations),
    )
    report = evaluate_localization_dataset(manifest, observation_set.observations)
    summary = render_localization_summary(report)
    return manifest, observation_set, report, summary


def _write_json(value, path: Path, *, force: bool) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASE_OUTPUT)
    parser.add_argument(
        "--observation-output", type=Path, default=DEFAULT_OBSERVATION_OUTPUT
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    manifest, observation_set, report, summary = build_artifacts()
    _write_json(manifest, arguments.case_output, force=arguments.force)
    _write_json(observation_set, arguments.observation_output, force=arguments.force)
    _write_json(report, arguments.report_output, force=arguments.force)
    _write_text(summary, arguments.summary_output, force=arguments.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
