"""Installable statistical analysis and regression-gate CLIs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from repocompass.evaluation.localization_analysis import (
    LocalizationAnalysisProvenance,
    LocalizationAnalysisReportSink,
    analyze_localization_evaluation,
    create_localization_bootstrap_config,
    render_localization_analysis_summary,
)
from repocompass.evaluation.localization_contracts import LocalizationVariant
from repocompass.evaluation.localization_harness import (
    load_localization_evaluation_report_file,
    load_localization_observation_set_file,
)
from repocompass.evaluation.localization_regression import (
    LocalizationRegressionReportSink,
    create_default_localization_regression_policy,
    evaluate_localization_regression,
    render_localization_regression_summary,
)


def create_analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-localization-analyze",
        description=(
            "Add case-balanced bootstrap intervals, paired deltas, error taxonomy, "
            "and resource/cost coverage to one localization report."
        ),
    )
    parser.add_argument("report_file", type=Path)
    parser.add_argument("observation_file", type=Path)
    parser.add_argument(
        "--provenance",
        type=LocalizationAnalysisProvenance,
        choices=tuple(LocalizationAnalysisProvenance),
        required=True,
    )
    parser.add_argument(
        "--reference-variant",
        type=LocalizationVariant,
        choices=tuple(LocalizationVariant),
        default=LocalizationVariant.TOOL_WRAPPER,
    )
    _add_bootstrap_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def create_regression_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-localization-regression",
        description=(
            "Fail closed when the upper confidence bound of case-paired regression "
            "exceeds a non-inferiority threshold."
        ),
    )
    parser.add_argument("baseline_report_file", type=Path)
    parser.add_argument("candidate_report_file", type=Path)
    parser.add_argument(
        "--maximum-regression-ppm",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--minimum-paired-cases",
        type=int,
        default=10,
    )
    _add_bootstrap_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def analysis_main(argv: Sequence[str] | None = None) -> int:
    arguments = create_analysis_parser().parse_args(argv)
    try:
        source = load_localization_evaluation_report_file(arguments.report_file)
        observations = load_localization_observation_set_file(
            arguments.observation_file
        )
        config = create_localization_bootstrap_config(
            confidence_level_ppm=arguments.confidence_level_ppm,
            resamples=arguments.resamples,
            random_seed=arguments.random_seed,
        )
        report = analyze_localization_evaluation(
            source,
            observations,
            provenance=arguments.provenance,
            bootstrap_config=config,
            reference_variant=arguments.reference_variant,
        )
        report_path = LocalizationAnalysisReportSink().write(
            report,
            arguments.output,
        )
    except FileExistsError:
        print(
            "repocompass-localization-analyze: error: output already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(
            f"repocompass-localization-analyze: error: {error}",
            file=sys.stderr,
        )
        return 1
    print(report.model_dump_json(indent=2, exclude_computed_fields=True))
    print(render_localization_analysis_summary(report), file=sys.stderr)
    print(f"Analysis: {report_path}", file=sys.stderr)
    return 0


def regression_main(argv: Sequence[str] | None = None) -> int:
    arguments = create_regression_parser().parse_args(argv)
    try:
        baseline = load_localization_evaluation_report_file(
            arguments.baseline_report_file
        )
        candidate = load_localization_evaluation_report_file(
            arguments.candidate_report_file
        )
        config = create_localization_bootstrap_config(
            confidence_level_ppm=arguments.confidence_level_ppm,
            resamples=arguments.resamples,
            random_seed=arguments.random_seed,
        )
        policy = create_default_localization_regression_policy(
            maximum_regression_ppm=arguments.maximum_regression_ppm,
            minimum_paired_cases=arguments.minimum_paired_cases,
        )
        report = evaluate_localization_regression(
            baseline,
            candidate,
            policy=policy,
            bootstrap_config=config,
        )
        report_path = LocalizationRegressionReportSink().write(
            report,
            arguments.output,
        )
    except FileExistsError:
        print(
            "repocompass-localization-regression: error: output already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(
            f"repocompass-localization-regression: error: {error}",
            file=sys.stderr,
        )
        return 1
    print(report.model_dump_json(indent=2, exclude_computed_fields=True))
    print(render_localization_regression_summary(report), file=sys.stderr)
    print(f"Regression report: {report_path}", file=sys.stderr)
    return 0 if report.passed else 2


def _add_bootstrap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confidence-level-ppm",
        type=int,
        default=950_000,
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=202_608_28,
    )
