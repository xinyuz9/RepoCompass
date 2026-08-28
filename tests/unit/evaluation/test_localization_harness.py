"""Localization observation-set, golden fixture, sink, and CLI tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.generate_localization_fixture import build_artifacts
from repocompass.evaluation import (
    MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES,
    LocalizationEvaluationReport,
    LocalizationEvaluationReportSink,
    LocalizationObservationSet,
    LocalizationVariant,
    evaluate_localization_dataset,
    load_localization_dataset_file,
    load_localization_observation_set_file,
    render_localization_summary,
)
from repocompass.evaluation import localization_cli

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_PATH = PROJECT_ROOT / "benchmarks/cases/localization-fixture-v1.json"
OBSERVATION_PATH = PROJECT_ROOT / "benchmarks/observations/localization-fixture-v1.json"
RESULT_PATH = PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.json"
SUMMARY_PATH = PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.md"


def _report() -> LocalizationEvaluationReport:
    manifest = load_localization_dataset_file(CASE_PATH)
    observation_set = load_localization_observation_set_file(OBSERVATION_PATH)
    return evaluate_localization_dataset(manifest, observation_set.observations)


def test_bundled_fixture_and_report_are_exactly_reproducible() -> None:
    manifest = load_localization_dataset_file(CASE_PATH)
    observation_set = load_localization_observation_set_file(OBSERVATION_PATH)
    report = _report()
    golden = LocalizationEvaluationReport.model_validate_json(
        RESULT_PATH.read_text(encoding="utf-8")
    )
    built_manifest, built_observations, built_report, summary = build_artifacts()

    assert manifest == built_manifest
    assert observation_set == built_observations
    assert report == built_report == golden
    assert render_localization_summary(report) == summary
    assert SUMMARY_PATH.read_text(encoding="utf-8") == summary + "\n"
    assert len(manifest.cases) == 2
    assert len(observation_set.observations) == 10
    assert tuple(aggregate.variant for aggregate in report.aggregates) == tuple(
        LocalizationVariant
    )
    assert report.report_id == (
        "01dd3fbd5ee267186c01313fa0c67450e85b3564449d4c8c4d91f0eb22ce55ef"
    )


def test_observation_set_rejects_mixed_identity_duplicate_and_tampering() -> None:
    observation_set = load_localization_observation_set_file(OBSERVATION_PATH)
    first = observation_set.observations[0]

    mixed = first.model_copy(update={"dataset_manifest_id": "f" * 64})
    with pytest.raises(ValidationError, match="observation_id"):
        type(first).model_validate_json(mixed.model_dump_json())

    tampered = observation_set.model_copy(update={"observations": (first,)})
    with pytest.raises(ValidationError, match="set_id|observation"):
        LocalizationObservationSet.model_validate_json(tampered.model_dump_json())


def test_observation_loader_is_bounded_and_duplicate_key_free(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_localization_observation_set_file(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_localization_observation_set_file(oversized)


def test_report_sink_is_immutable(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "nested" / "report.json"

    written = LocalizationEvaluationReportSink().write(report, destination)

    assert (
        LocalizationEvaluationReport.model_validate_json(
            written.read_text(encoding="utf-8")
        )
        == report
    )
    with pytest.raises(FileExistsError):
        LocalizationEvaluationReportSink().write(report, destination)


def test_localization_cli_writes_golden_report_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "report.json"
    arguments = [
        str(CASE_PATH),
        str(OBSERVATION_PATH),
        "--output",
        str(destination),
    ]

    assert localization_cli.main(arguments) == 0
    first = capsys.readouterr()
    assert json.loads(first.out) == _report().model_dump(mode="json")
    assert "localization-only" in first.err
    assert "resolved rate" in first.err
    assert destination.read_text(encoding="utf-8") == (
        RESULT_PATH.read_text(encoding="utf-8")
    )

    assert localization_cli.main(arguments) == 1
    second = capsys.readouterr()
    assert second.out == ""
    assert "report already exists" in second.err
