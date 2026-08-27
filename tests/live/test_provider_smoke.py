"""Opt-in smoke test against the configured real provider API."""

import os
from pathlib import Path

import pytest

from repocompass.agent.cost import CostAvailability
from repocompass.evaluation.live_smoke import (
    LiveProviderSmokeReportSink,
    run_live_provider_smoke,
)
from repocompass.models.config import load_model_settings
from repocompass.pricing import load_pricing_table_file

pytestmark = pytest.mark.live_api


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.fail(f"{name} is required for an enabled live API smoke test")
    return value


def test_configured_provider_returns_usage_within_explicit_cost_cap() -> None:
    if os.getenv("REPOCOMPASS_RUN_LIVE_TESTS", "").strip() != "1":
        pytest.skip("set REPOCOMPASS_RUN_LIVE_TESTS=1 to spend provider credits")
    pricing_path = Path(_required_environment("REPOCOMPASS_LIVE_PRICING_FILE"))
    max_cost_usd = _required_environment("REPOCOMPASS_LIVE_MAX_COST_USD")
    output_path = Path(_required_environment("REPOCOMPASS_LIVE_REPORT_FILE"))

    report = run_live_provider_smoke(
        settings=load_model_settings(),
        pricing_table=load_pricing_table_file(pricing_path),
        max_cost_usd=max_cost_usd,
        max_output_tokens=32,
    )
    written = LiveProviderSmokeReportSink().write(report, output_path)

    assert report.response_marker_matched
    assert report.cost.availability is CostAvailability.ESTIMATED
    assert report.cost.estimated_cost_usd is not None
    assert written.is_file()
    print(report.model_dump_json(indent=2))
