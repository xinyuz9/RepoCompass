"""Installable deterministic localization-evaluation CLI."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from repocompass.evaluation.localization_case_file import (
    load_localization_dataset_file,
)
from repocompass.evaluation.localization_harness import (
    LocalizationEvaluationReportSink,
    load_localization_observation_set_file,
    render_localization_summary,
)
from repocompass.evaluation.localization_metrics import (
    evaluate_localization_dataset,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone localization benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="repocompass-localization-benchmark",
        description=(
            "Score frozen ranked file/symbol observations against hidden labels. "
            "This reports localization metrics, never SWE-bench resolved rate."
        ),
    )
    parser.add_argument(
        "dataset_file",
        type=Path,
        help="strict content-addressed localization dataset manifest",
    )
    parser.add_argument(
        "observation_file",
        type=Path,
        help="strict comparable localization observation set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new immutable report path; existing files are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one frozen observation set and publish its report."""

    arguments = create_parser().parse_args(argv)
    try:
        manifest = load_localization_dataset_file(arguments.dataset_file)
        observation_set = load_localization_observation_set_file(
            arguments.observation_file
        )
        report = evaluate_localization_dataset(
            manifest,
            observation_set.observations,
        )
        report_path = LocalizationEvaluationReportSink().write(
            report,
            arguments.output,
        )
    except FileExistsError:
        print(
            "repocompass-localization-benchmark: error: report already exists: "
            f"{arguments.output.resolve()}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"repocompass-localization-benchmark: error: {error}", file=sys.stderr)
        return 1

    print(report.model_dump_json(indent=2, exclude_computed_fields=True))
    print(render_localization_summary(report), file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
