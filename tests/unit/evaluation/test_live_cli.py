"""Live API boundary tests using an in-process HTTP mock, never paid calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from repocompass.agent.cost import ModelPrice
from repocompass.evaluation.live_budget import (
    BudgetAdmissionError,
    BudgetCallback,
    SQLiteSpendLedger,
)
from repocompass.evaluation.live_cli import (
    ProviderConnectivityError,
    check_provider_connectivity,
    consecutive_new_failures,
    main,
    persist,
    public_error_metadata,
)
from repocompass.evaluation.localization_contracts import LocalizationRunStatus


def test_restart_does_not_stop_on_reused_failures():
    failed = SimpleNamespace(status=LocalizationRunStatus.FAILED)
    assert consecutive_new_failures(0, failed, False) == 0
    assert consecutive_new_failures(0, failed, True) == 1
    assert consecutive_new_failures(1, failed, True) == 2
    assert consecutive_new_failures(1, SimpleNamespace(status=None), True) == 0


def test_error_projection_does_not_leak_exception_payload():
    error = RuntimeError("secret-key https://user:password@provider.test")
    error.__cause__ = ValueError("private prompt")
    assert public_error_metadata(error) == {
        "error_code": "RuntimeError",
        "cause_code": "ValueError",
    }


@pytest.mark.parametrize("status", [200, 401, 404, 503])
def test_connectivity_probe_is_credential_free_and_does_not_redirect(
    monkeypatch, status
):
    calls = []
    original = httpx.Client

    def transport(request):
        calls.append(request)
        return httpx.Response(status)

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(transport), **kwargs)

    monkeypatch.setattr("repocompass.evaluation.live_cli.httpx.Client", client)
    if status >= 500:
        with pytest.raises(ProviderConnectivityError):
            check_provider_connectivity("https://api.deepseek.com/v1")
    else:
        check_provider_connectivity("https://api.deepseek.com/v1")
    assert len(calls) == 1
    assert calls[0].method == "HEAD"
    assert "authorization" not in calls[0].headers


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.deepseek.com",
        "https://other.invalid",
        "https://secret@api.deepseek.com",
        "https://api.deepseek.com:8443",
        "https://api.deepseek.com?key=secret",
    ],
)
def test_connectivity_probe_rejects_unreviewed_authority(endpoint):
    with pytest.raises(ValueError):
        check_provider_connectivity(endpoint)


def test_connectivity_failure_is_sanitized(monkeypatch):
    def failed(*args, **kwargs):
        raise httpx.ConnectError("private network detail")

    monkeypatch.setattr(httpx.Client, "head", failed)
    with pytest.raises(
        ProviderConnectivityError, match="connectivity failed"
    ) as caught:
        check_provider_connectivity("https://api.deepseek.com")
    assert caught.value.__cause__ is None


def test_opt_in_is_required_before_any_source_or_model_work(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "repocompass.evaluation.live_cli.load_model_settings",
        lambda: pytest.fail("must not load provider"),
    )
    assert (
        main(
            [
                "run",
                "--cache-root",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    assert "--allow-paid-api" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_immutable_artifacts_cannot_be_relabelled(tmp_path):
    path = tmp_path / "config.json"
    persist(path, {"model": "first"})
    persist(path, {"model": "first"})
    with pytest.raises(ValueError, match="identity differs"):
        persist(path, {"model": "second"})


@pytest.mark.parametrize("limit,expected_calls", [("1", 1), ("0.000001", 0)])
def test_real_langchain_callback_runs_before_http_request(
    tmp_path: Path, limit, expected_calls
):
    calls = []

    def transport(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    price = ModelPrice(
        provider="deepseek",
        model="test",
        input_usd_per_million_tokens="1",
        output_usd_per_million_tokens="2",
    )
    with SQLiteSpendLedger(
        tmp_path / "budget.sqlite", limit_usd=limit, pricing_id="a" * 64
    ) as ledger:
        callback = BudgetCallback(ledger, price, max_output_tokens=128)
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = ChatOpenAI(
                model="test",
                api_key="test-only",
                base_url="https://test.invalid/v1",
                use_responses_api=False,
                max_tokens=128,
                max_retries=0,
                http_client=client,
                callbacks=[callback],
            )
            assert model.root_client.max_retries == 0
            if expected_calls:
                assert model.invoke([HumanMessage(content="Trace.")]).content == "Done"
                assert ledger.summary()["settled_invocations"] == 1
                assert ledger.summary()["charged_or_reserved_usd"] == "0.000014"
            else:
                with pytest.raises(BudgetAdmissionError):
                    model.invoke([HumanMessage(content="Trace.")])
            assert len(calls) == expected_calls
