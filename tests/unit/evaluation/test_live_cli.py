"""Live API boundary tests using an in-process HTTP mock, never paid calls."""

import json
from pathlib import Path

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
from repocompass.evaluation.live_cli import main, persist


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
