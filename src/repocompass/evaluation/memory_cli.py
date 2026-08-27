"""Installable CLI for deterministic memory on/off ablation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from repocompass.evaluation.memory_ablation import (
    MemoryAblationReportSink,
    render_memory_ablation_summary,
    run_memory_ablation,
)
from repocompass.evaluation.memory_case_file import (
    load_memory_ablation_case_file,
)
from repocompass.pricing import load_pricing_table_file


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone reproducible memory benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="repocompass-memory-benchmark",
        description=(
            "Compare production active context with commit-aware recall disabled "
            "and enabled without making model API calls."
        ),
    )
    parser.add_argument(
        "case_file",
        type=Path,
        help="strict content-addressed memory-ablation case JSON",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic memory ablation and publish its report."""

    arguments = create_parser().parse_args(argv)
    try:
        case = load_memory_ablation_case_file(arguments.case_file)
        pricing_table = load_pricing_table_file(arguments.pricing_file)
        report = run_memory_ablation(
            case,
            pricing_table=pricing_table,
            provider=arguments.provider,
            model=arguments.model,
        )
        report_path = MemoryAblationReportSink().write(report, arguments.output)
    except FileExistsError:
        print(
            "repocompass-memory-benchmark: error: report already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"repocompass-memory-benchmark: error: {error}", file=sys.stderr)
        return 1

    print(report.model_dump_json(indent=2))
    print(render_memory_ablation_summary(report), file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
