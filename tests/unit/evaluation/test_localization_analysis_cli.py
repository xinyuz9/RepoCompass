"""Installed localization analysis and regression CLI tests."""

import json
from pathlib import Path

from repocompass.evaluation import (
    MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES,
    load_localization_evaluation_report_file,
)
from repocompass.evaluation.localization_analysis_cli import (
    analysis_main,
    regression_main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.json"
OBSERVATION_PATH = PROJECT_ROOT / "benchmarks/observations/localization-fixture-v1.json"


def test_analysis_cli_writes_immutable_provenance_explicit_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "analysis.json"
    arguments = [
        str(REPORT_PATH),
        str(OBSERVATION_PATH),
        "--provenance",
        "synthetic_fixture",
        "--resamples",
        "100",
        "--random-seed",
        "17",
        "--output",
        str(output),
    ]

    assert analysis_main(arguments) == 0
    first = capsys.readouterr()
    payload = json.loads(first.out)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert payload["provenance"] == "synthetic_fixture"
    assert "not model quality" in first.err
    assert "not a SWE-bench resolved rate" in first.err

    assert analysis_main(arguments) == 1
    second = capsys.readouterr()
    assert second.out == ""
    assert "output already exists" in second.err


def test_regression_cli_returns_zero_for_identical_non_inferior_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "regression.json"
    arguments = [
        str(REPORT_PATH),
        str(REPORT_PATH),
        "--minimum-paired-cases",
        "2",
        "--resamples",
        "100",
        "--random-seed",
        "17",
        "--output",
        str(output),
    ]

    assert regression_main(arguments) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["passed"] is True
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert "Decision: PASS" in captured.err


def test_evaluation_report_loader_is_bounded_duplicate_free_and_exact(
    tmp_path: Path,
) -> None:
    loaded = load_localization_evaluation_report_file(REPORT_PATH)
    assert loaded.report_id == (
        "01dd3fbd5ee267186c01313fa0c67450e85b3564449d4c8c4d91f0eb22ce55ef"
    )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    try:
        load_localization_evaluation_report_file(duplicate)
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate report key was accepted")

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES + 1))
    try:
        load_localization_evaluation_report_file(oversized)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("oversized report was accepted")
