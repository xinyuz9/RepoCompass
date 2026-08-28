"""Regenerate deterministic localization analysis and regression goldens."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from benchmarks.generate_localization_fixture import build_artifacts
else:
    from generate_localization_fixture import build_artifacts
from repocompass.evaluation import (
    LocalizationAnalysisProvenance,
    LocalizationAnalysisReport,
    LocalizationRegressionReport,
    LocalizationVariant,
    analyze_localization_evaluation,
    create_default_localization_regression_policy,
    create_localization_bootstrap_config,
    evaluate_localization_regression,
    render_localization_analysis_summary,
    render_localization_regression_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.analysis.json"
)
DEFAULT_ANALYSIS_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.analysis.md"
)
DEFAULT_REGRESSION_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.regression.json"
)
DEFAULT_REGRESSION_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.regression.md"
)


def build_analysis_artifacts() -> tuple[
    LocalizationAnalysisReport,
    str,
    LocalizationRegressionReport,
    str,
]:
    """Build deterministic statistical and regression artifacts."""

    _, observations, report, _ = build_artifacts()
    config = create_localization_bootstrap_config(
        confidence_level_ppm=950_000,
        resamples=10_000,
        random_seed=202_608_28,
    )
    analysis = analyze_localization_evaluation(
        report,
        observations,
        provenance=LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE,
        bootstrap_config=config,
        reference_variant=LocalizationVariant.TOOL_WRAPPER,
    )
    regression = evaluate_localization_regression(
        report,
        report,
        policy=create_default_localization_regression_policy(
            maximum_regression_ppm=0,
            minimum_paired_cases=2,
        ),
        bootstrap_config=config,
    )
    return (
        analysis,
        render_localization_analysis_summary(analysis),
        regression,
        render_localization_regression_summary(regression),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
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
    analysis, analysis_summary, regression, regression_summary = (
        build_analysis_artifacts()
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


def _write_json(
    value: LocalizationAnalysisReport | LocalizationRegressionReport,
    path: Path,
    *,
    force: bool,
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
