"""Deterministic memory on/off contracts, CLI, fixture, and golden tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.generate_memory_ablation_fixture import build_case
from repocompass.evaluation import (
    MAX_MEMORY_ABLATION_CASE_FILE_BYTES,
    MemoryAblationCase,
    MemoryAblationReport,
    MemoryAblationReportSink,
    MemoryAblationVariant,
    load_memory_ablation_case_file,
    render_memory_ablation_summary,
    run_memory_ablation,
)
from repocompass.evaluation import memory_cli
from repocompass.pricing import load_pricing_table_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_PATH = PROJECT_ROOT / "benchmarks/cases/memory-recall-v1.json"
RESULT_PATH = PROJECT_ROOT / "benchmarks/results/memory-recall-v1.example.json"
PRICING_PATH = PROJECT_ROOT / "examples/pricing.example.json"


def _report() -> MemoryAblationReport:
    return run_memory_ablation(
        load_memory_ablation_case_file(CASE_PATH),
        pricing_table=load_pricing_table_file(PRICING_PATH),
        provider="openai",
        model="example-openai-model",
    )


def test_bundled_memory_ablation_reproduces_golden_report() -> None:
    case = load_memory_ablation_case_file(CASE_PATH)
    report = _report()
    golden = MemoryAblationReport.model_validate_json(
        RESULT_PATH.read_text(encoding="utf-8")
    )
    memory_off, memory_on = report.variants

    assert case == build_case()
    assert report == golden
    assert report.report_id == (
        "fdad7ec73736c30cfd77fb63b59ad16bba896a3414d1322216e8f9ffdcd88820"
    )
    assert memory_off.variant is MemoryAblationVariant.MEMORY_OFF
    assert memory_on.variant is MemoryAblationVariant.MEMORY_ON
    assert memory_on.retrieval_result_count == 3
    assert memory_on.usable_memory_count == 2
    assert memory_on.selected_memory_count == 2
    assert memory_on.stale_excluded_count == 1
    assert memory_on.repository_result_count == 2
    assert memory_on.episodic_result_count == 1
    assert memory_on.required_evidence_retention_ppm == 1_000_000
    assert report.context_token_delta == 2_623
    assert report.required_evidence_retention_delta_ppm == 0
    assert report.estimated_input_cost_delta_usd == "0.002623"
    assert "retrieval results 3; usable 2; selected 2" in (
        render_memory_ablation_summary(report)
    )


def test_case_and_report_reject_content_identity_or_metric_tampering() -> None:
    case = load_memory_ablation_case_file(CASE_PATH)
    tampered_case = case.model_copy(update={"budget_tokens": 4_999})
    with pytest.raises(ValidationError, match="case_id"):
        MemoryAblationCase.model_validate_json(tampered_case.model_dump_json())

    report = _report()
    tampered_report = report.model_copy(
        update={"context_token_delta": report.context_token_delta + 1}
    )
    with pytest.raises(ValidationError, match="token delta"):
        MemoryAblationReport.model_validate_json(tampered_report.model_dump_json())


def test_case_loader_rejects_duplicate_keys_and_oversized_input(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_memory_ablation_case_file(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_MEMORY_ABLATION_CASE_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_memory_ablation_case_file(oversized)


def test_report_sink_is_immutable(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "nested" / "report.json"

    written = MemoryAblationReportSink().write(report, destination)

    assert (
        MemoryAblationReport.model_validate_json(written.read_text(encoding="utf-8"))
        == report
    )
    with pytest.raises(FileExistsError):
        MemoryAblationReportSink().write(report, destination)


def test_memory_benchmark_cli_writes_exact_report_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "report.json"
    arguments = [
        str(CASE_PATH),
        "--pricing-file",
        str(PRICING_PATH),
        "--provider",
        "openai",
        "--model",
        "example-openai-model",
        "--output",
        str(destination),
    ]

    assert memory_cli.main(arguments) == 0
    first = capsys.readouterr()
    assert json.loads(first.out) == _report().model_dump(mode="json")
    assert "context token delta +2623" in first.err
    assert destination.is_file()

    assert memory_cli.main(arguments) == 1
    second = capsys.readouterr()
    assert second.out == ""
    assert "report already exists" in second.err


def test_missing_price_stays_unavailable_instead_of_becoming_zero() -> None:
    report = run_memory_ablation(
        load_memory_ablation_case_file(CASE_PATH),
        pricing_table=load_pricing_table_file(PRICING_PATH),
        provider="unknown-provider",
        model="unknown-model",
    )

    assert report.estimated_input_cost_delta_usd is None
    assert all(item.estimated_input_cost_usd is None for item in report.variants)
