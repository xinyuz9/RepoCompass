"""Installed-entry-point smoke test for the deterministic benchmark CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

from repocompass.evaluation import (
    ContextAblationReport,
    LocalizationAnalysisReport,
    LocalizationRegressionReport,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_installed_benchmark_entry_point_publishes_report(tmp_path: Path) -> None:
    executable_name = (
        "repocompass-benchmark.exe" if os.name == "nt" else "repocompass-benchmark"
    )
    executable = Path(sys.executable).with_name(executable_name)
    output = tmp_path / "subprocess-report.json"

    completed = subprocess.run(
        [
            str(executable),
            str(PROJECT_ROOT / "benchmarks/cases/context-pressure-v1.json"),
            "--pricing-file",
            str(PROJECT_ROOT / "examples/pricing.example.json"),
            "--provider",
            "openai",
            "--model",
            "example-openai-model",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = ContextAblationReport.model_validate_json(completed.stdout)
    file_report = ContextAblationReport.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    assert stdout_report == file_report
    assert json.loads(completed.stdout)["report_id"] == (
        "61d958a8c22ab41f40d875b7a850f5b79213a31bcfef48b2f827947b23a0d27f"
    )
    assert "Context ablation context-pressure-v1" in completed.stderr
    assert f"Report: {output.resolve()}" in completed.stderr


def test_installed_localization_statistical_entry_points(
    tmp_path: Path,
) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    executable_root = Path(sys.executable).parent
    source_report = (
        PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.example.json"
    )
    observations = PROJECT_ROOT / "benchmarks/observations/localization-fixture-v1.json"
    analysis_output = tmp_path / "analysis.json"
    analysis = subprocess.run(
        [
            str(executable_root / f"repocompass-localization-analyze{suffix}"),
            str(source_report),
            str(observations),
            "--provenance",
            "synthetic_fixture",
            "--resamples",
            "100",
            "--random-seed",
            "17",
            "--output",
            str(analysis_output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert analysis.returncode == 0, analysis.stderr
    assert LocalizationAnalysisReport.model_validate_json(analysis.stdout) == (
        LocalizationAnalysisReport.model_validate_json(
            analysis_output.read_text(encoding="utf-8")
        )
    )
    assert "not model quality" in analysis.stderr

    regression_output = tmp_path / "regression.json"
    regression = subprocess.run(
        [
            str(executable_root / f"repocompass-localization-regression{suffix}"),
            str(source_report),
            str(source_report),
            "--minimum-paired-cases",
            "2",
            "--resamples",
            "100",
            "--random-seed",
            "17",
            "--output",
            str(regression_output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert regression.returncode == 0, regression.stderr
    parsed = LocalizationRegressionReport.model_validate_json(regression.stdout)
    assert parsed.passed is True
    assert parsed == LocalizationRegressionReport.model_validate_json(
        regression_output.read_text(encoding="utf-8")
    )
    assert "Decision: PASS" in regression.stderr
