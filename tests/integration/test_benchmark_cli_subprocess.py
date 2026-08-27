"""Installed-entry-point smoke test for the deterministic benchmark CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

from repocompass.evaluation import ContextAblationReport

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
        "d64e740822df166dd568ce132508a0d2b320ddcc2a7b471ad13ae6ce022ae745"
    )
    assert "Context ablation context-pressure-v1" in completed.stderr
    assert f"Report: {output.resolve()}" in completed.stderr
