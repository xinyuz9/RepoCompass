"""Tests for the bounded real-provider smoke harness."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from repocompass.agent.cost import CostAvailability
from repocompass.evaluation import live_smoke
from repocompass.evaluation.live_smoke import (
    LIVE_SMOKE_INPUT_TOKEN_CEILING,
    LiveProviderSmokeReport,
    LiveProviderSmokeReportSink,
    LiveProviderSmokeError,
    run_live_provider_smoke,
)
from repocompass.models.config import DeepSeekModelSettings
from repocompass.pricing import load_pricing_table_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRICING_PATH = PROJECT_ROOT / "examples/pricing.deepseek-2026-08-27.json"


def _settings() -> DeepSeekModelSettings:
    return DeepSeekModelSettings(
        model="deepseek-v4-pro",
        api_key="unit-test-secret",
        base_url="https://api.deepseek.com",
    )


def _response() -> AIMessage:
    return AIMessage(
        content="REPOCOMPASS_LIVE_OK",
        usage_metadata={
            "input_tokens": 20,
            "output_tokens": 3,
            "total_tokens": 23,
            "input_token_details": {"cache_read": 0},
        },
    )


def _clock(values: tuple[int, int]):
    ticks: Iterator[int] = iter(values)
    return lambda: next(ticks)


def test_run_live_provider_smoke_preflights_and_reports_bounded_call() -> None:
    model = MagicMock()
    bounded_model = MagicMock()
    bounded_model.invoke.return_value = _response()
    model.bind.return_value = bounded_model
    model_factory = MagicMock(return_value=model)

    report = run_live_provider_smoke(
        settings=_settings(),
        pricing_table=load_pricing_table_file(PRICING_PATH),
        max_cost_usd="0.002",
        max_output_tokens=32,
        model_factory=model_factory,
        monotonic_ns=_clock((1_000_000_000, 1_123_000_000)),
    )

    model_factory.assert_called_once_with(_settings())
    model.bind.assert_called_once_with(max_tokens=32)
    assert bounded_model.invoke.call_count == 1
    assert report.provider == "deepseek"
    assert report.model == "deepseek-v4-pro"
    assert report.max_input_tokens_assumed == LIVE_SMOKE_INPUT_TOKEN_CEILING
    assert report.max_output_tokens == 32
    assert report.preflight_max_cost_usd == "0.0018096"
    assert report.max_cost_usd == "0.002"
    assert report.invocation.latency_ms == 123
    assert report.invocation.usage.input_tokens == 20
    assert report.invocation.usage.output_tokens == 3
    assert report.cost.availability is CostAvailability.ESTIMATED
    assert report.cost.estimated_cost_usd == "0.00001131"
    assert report.response_marker_matched
    assert "REPOCOMPASS_LIVE_OK" not in report.model_dump_json()
    assert (
        LiveProviderSmokeReport.model_validate_json(report.model_dump_json()) == report
    )


def test_run_live_provider_smoke_rejects_unsafe_cap_before_model_creation() -> None:
    model_factory = MagicMock()

    with pytest.raises(ValueError, match="preflight maximum cost"):
        run_live_provider_smoke(
            settings=_settings(),
            pricing_table=load_pricing_table_file(PRICING_PATH),
            max_cost_usd="0.001",
            max_output_tokens=32,
            model_factory=model_factory,
        )

    model_factory.assert_not_called()


def test_run_live_provider_smoke_requires_exact_model_price_before_call() -> None:
    model_factory = MagicMock()
    settings = DeepSeekModelSettings(
        model="unpriced-model",
        api_key="unit-test-secret",
        base_url="https://api.deepseek.com",
    )

    with pytest.raises(ValueError, match="Pricing snapshot has no exact entry"):
        run_live_provider_smoke(
            settings=settings,
            pricing_table=load_pricing_table_file(PRICING_PATH),
            max_cost_usd="1",
            max_output_tokens=32,
            model_factory=model_factory,
        )

    model_factory.assert_not_called()


def test_live_report_is_content_addressed_and_sink_never_overwrites(
    tmp_path: Path,
) -> None:
    model = MagicMock()
    model.bind.return_value.invoke.return_value = _response()
    report = run_live_provider_smoke(
        settings=_settings(),
        pricing_table=load_pricing_table_file(PRICING_PATH),
        max_cost_usd="0.002",
        max_output_tokens=32,
        model_factory=MagicMock(return_value=model),
        monotonic_ns=_clock((0, 1_000_000)),
    )
    destination = tmp_path / "nested" / "live-smoke.json"

    written = LiveProviderSmokeReportSink().write(report, destination)

    assert written == destination.resolve()
    assert json.loads(written.read_text(encoding="utf-8")) == report.model_dump(
        mode="json"
    )
    with pytest.raises(FileExistsError):
        LiveProviderSmokeReportSink().write(report, destination)
    payload = report.model_dump(mode="json")
    payload["report_id"] = "f" * 64
    with pytest.raises(ValidationError, match="report_id"):
        LiveProviderSmokeReport.model_validate_json(json.dumps(payload))


def test_live_cli_requires_environment_marker_before_loading_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("REPOCOMPASS_RUN_LIVE_TESTS", raising=False)
    settings_loader = MagicMock(side_effect=AssertionError("must not load settings"))
    monkeypatch.setattr(live_smoke, "load_model_settings", settings_loader)

    exit_code = live_smoke.main(
        [
            "--acknowledge-live-api-costs",
            "--pricing-file",
            str(PRICING_PATH),
            "--max-cost-usd",
            "0.002",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "REPOCOMPASS_RUN_LIVE_TESTS=1" in captured.err
    settings_loader.assert_not_called()


def test_live_cli_requires_explicit_cost_acknowledgement_before_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("REPOCOMPASS_RUN_LIVE_TESTS", "1")
    settings_loader = MagicMock(side_effect=AssertionError("must not load settings"))
    monkeypatch.setattr(live_smoke, "load_model_settings", settings_loader)

    with pytest.raises(SystemExit) as error:
        live_smoke.main(
            [
                "--pricing-file",
                str(PRICING_PATH),
                "--max-cost-usd",
                "0.002",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "--acknowledge-live-api-costs is required" in captured.err
    settings_loader.assert_not_called()


def test_provider_failure_is_sanitized_without_raw_response_text() -> None:
    model = MagicMock()
    model.bind.return_value.invoke.side_effect = RuntimeError(
        "sensitive provider response body"
    )

    with pytest.raises(LiveProviderSmokeError, match="RuntimeError") as error:
        run_live_provider_smoke(
            settings=_settings(),
            pricing_table=load_pricing_table_file(PRICING_PATH),
            max_cost_usd="0.002",
            max_output_tokens=32,
            model_factory=MagicMock(return_value=model),
        )

    assert "sensitive provider response body" not in str(error.value)


def test_live_cli_reserves_output_before_loading_credentials_or_calling_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("REPOCOMPASS_RUN_LIVE_TESTS", "1")
    destination = tmp_path / "existing.json"
    destination.write_text("preserve-me", encoding="utf-8")
    settings_loader = MagicMock(side_effect=AssertionError("must not load settings"))
    monkeypatch.setattr(live_smoke, "load_model_settings", settings_loader)

    exit_code = live_smoke.main(
        [
            "--acknowledge-live-api-costs",
            "--pricing-file",
            str(PRICING_PATH),
            "--max-cost-usd",
            "0.002",
            "--output",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "report already exists" in captured.err
    assert destination.read_text(encoding="utf-8") == "preserve-me"
    settings_loader.assert_not_called()


def test_bundled_real_provider_report_is_valid_sanitized_evidence() -> None:
    path = (
        PROJECT_ROOT / "benchmarks/results/deepseek-v4-pro-live-smoke-2026-08-27.json"
    )
    raw = path.read_text(encoding="utf-8")

    report = LiveProviderSmokeReport.model_validate_json(raw)

    assert report.report_id == (
        "f4467a495f45ac43147039249bea25e993decc81e4a1dae81983e0dc1de936aa"
    )
    assert report.provider == "deepseek"
    assert report.model == "deepseek-v4-pro"
    assert report.invocation.usage.input_tokens == 29
    assert report.invocation.usage.output_tokens == 7
    assert report.invocation.latency_ms == 1_230
    assert report.cost.estimated_cost_usd == "0.000018705"
    assert report.max_cost_usd == "0.002"
    assert report.response_marker_matched
    assert "REPOCOMPASS_LIVE_OK" not in raw
    assert "REPOCOMPASS_API_KEY" not in raw
