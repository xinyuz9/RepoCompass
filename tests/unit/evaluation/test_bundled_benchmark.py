"""Regression lock for the bundled reproducible context benchmark."""

from decimal import Decimal
from pathlib import Path

from repocompass.agent.cost import CostAvailability
from repocompass.evaluation import (
    AblationVariant,
    ContextAblationReport,
    load_context_ablation_case_file,
    run_context_ablation,
)
from repocompass.pricing import load_pricing_table_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_PATH = PROJECT_ROOT / "benchmarks/cases/context-pressure-v1.json"
RESULT_PATH = PROJECT_ROOT / "benchmarks/results/context-pressure-v1.example.json"
PRICING_PATH = PROJECT_ROOT / "examples/pricing.example.json"


def test_bundled_context_pressure_result_is_exactly_reproducible() -> None:
    case = load_context_ablation_case_file(CASE_PATH)
    pricing = load_pricing_table_file(PRICING_PATH)
    committed = ContextAblationReport.model_validate_json(
        RESULT_PATH.read_text(encoding="utf-8")
    )

    reproduced = run_context_ablation(
        case,
        pricing_table=pricing,
        provider="openai",
        model="example-openai-model",
    )

    assert reproduced == committed
    assert committed.report_id == (
        "61d958a8c22ab41f40d875b7a850f5b79213a31bcfef48b2f827947b23a0d27f"
    )
    metrics = {item.variant: item for item in committed.variants}
    active = metrics[AblationVariant.ACTIVE_CONTEXT]
    baseline = metrics[AblationVariant.ALL_CONTEXT_BASELINE]
    assert active.selected_tokens == 2_330
    assert baseline.selected_tokens == 18_272
    assert committed.token_reduction == 15_942
    assert active.retained_required_evidence_count == 2
    assert active.required_evidence_retention_ppm == 1_000_000
    assert active.evidence_provenance_coverage_ppm == 1_000_000
    assert baseline.evidence_provenance_coverage_ppm == 666_666
    assert all(
        item.cost_availability is CostAvailability.ESTIMATED
        for item in committed.variants
    )
    assert Decimal(committed.estimated_input_cost_savings_usd) == Decimal("0.015942")
    assert committed.observed_latency_delta_ms is None
