"""Tests for the reproducible context-ablation benchmark CLI."""

import json
from pathlib import Path

import pytest

from repocompass.evaluation import cli
from repocompass.agent.cost import CostAvailability
from repocompass.evaluation.context_ablation import (
    AblationVariant,
    ContextAblationReport,
)
from tests.unit.evaluation.test_case_file import write_case

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _arguments(tmp_path: Path) -> list[str]:
    case_path = tmp_path / "case.json"
    write_case(case_path)
    return [
        str(case_path),
        "--pricing-file",
        str(PROJECT_ROOT / "examples/pricing.example.json"),
        "--provider",
        "openai",
        "--model",
        "example-openai-model",
        "--output",
        str(tmp_path / "report.json"),
    ]


def test_main_writes_machine_report_and_human_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        *_arguments(tmp_path),
        "--active-latency-ms",
        "3",
        "--baseline-latency-ms",
        "7",
    ]

    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    report_path = tmp_path / "report.json"
    from_stdout = ContextAblationReport.model_validate_json(captured.out)
    from_file = ContextAblationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    metrics = {item.variant: item for item in from_file.variants}
    assert exit_code == 0
    assert from_stdout == from_file
    assert metrics[AblationVariant.ACTIVE_CONTEXT].latency_ms == 3
    assert metrics[AblationVariant.ALL_CONTEXT_BASELINE].latency_ms == 7
    assert from_file.observed_latency_delta_ms == 4
    assert all(
        item.cost_availability is CostAvailability.ESTIMATED
        for item in from_file.variants
    )
    assert from_file.estimated_input_cost_savings_usd is not None
    assert "Context ablation loader-case-v1" in captured.err
    assert "estimate, not provider invoice" in captured.err
    assert f"Report: {report_path.resolve()}" in captured.err


def test_main_refuses_to_overwrite_existing_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = _arguments(tmp_path)

    assert cli.main(arguments) == 0
    first_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "already exists" in captured.err
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == (
        first_payload
    )


def test_main_rejects_unpaired_latency_observation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([*_arguments(tmp_path), "--active-latency-ms", "3"])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "latency observations must be supplied together" in captured.err


def test_main_reports_invalid_case_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = _arguments(tmp_path)
    case_path = Path(arguments[0])
    case_path.write_text("{}", encoding="utf-8")

    exit_code = cli.main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Invalid context ablation case file" in captured.err
    assert not (tmp_path / "report.json").exists()
