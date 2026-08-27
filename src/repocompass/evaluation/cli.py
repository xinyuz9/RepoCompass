"""Command-line entry point for deterministic context-ablation benchmarks."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from repocompass.evaluation.case_file import load_context_ablation_case_file
from repocompass.evaluation.context_ablation import (
    AblationVariant,
    ContextAblationReportSink,
    render_context_ablation_summary,
    run_context_ablation,
)
from repocompass.pricing import load_pricing_table_file


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone reproducible benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="repocompass-benchmark",
        description=(
            "Compare the production Active Context Builder with an unbounded "
            "all-context baseline without making model API calls."
        ),
    )
    parser.add_argument(
        "case_file",
        type=Path,
        help="strict content-addressed context-ablation case JSON",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        required=True,
        help="strict versioned pricing JSON used for input-cost estimates",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="provider identity used to select one pricing entry",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model identity used to select one pricing entry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable report path; existing files are never overwritten",
    )
    parser.add_argument(
        "--active-latency-ms",
        type=_non_negative_int,
        help="optional externally observed active-context latency",
    )
    parser.add_argument(
        "--baseline-latency-ms",
        type=_non_negative_int,
        help="optional externally observed all-context latency",
    )
    return parser


def _validate_latency_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> dict[AblationVariant, int] | None:
    active_latency = arguments.active_latency_ms
    baseline_latency = arguments.baseline_latency_ms
    if (active_latency is None) != (baseline_latency is None):
        parser.error("latency observations must be supplied together")
    if active_latency is None:
        return None
    return {
        AblationVariant.ACTIVE_CONTEXT: active_latency,
        AblationVariant.ALL_CONTEXT_BASELINE: baseline_latency,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic ablation and publish its immutable report."""

    parser = create_parser()
    arguments = parser.parse_args(argv)
    observed_latency_ms = _validate_latency_arguments(parser, arguments)
    try:
        case = load_context_ablation_case_file(arguments.case_file)
        pricing_table = load_pricing_table_file(arguments.pricing_file)
        report = run_context_ablation(
            case,
            pricing_table=pricing_table,
            provider=arguments.provider,
            model=arguments.model,
            observed_latency_ms=observed_latency_ms,
        )
        report_path = ContextAblationReportSink().write(report, arguments.output)
    except FileExistsError:
        print(
            f"repocompass-benchmark: error: report already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"repocompass-benchmark: error: {error}", file=sys.stderr)
        return 1

    print(report.model_dump_json(indent=2))
    print(render_context_ablation_summary(report), file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
