"""No-provider regression tests for durable pre-call spending admission."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from repocompass.agent.cost import ModelPrice
from repocompass.evaluation.live_budget import (
    BudgetAdmissionError,
    BudgetCallback,
    SQLiteSpendLedger,
)


def test_reopen_keeps_unknown_reservations_and_cannot_raise_limit(tmp_path: Path):
    path = tmp_path / "budget.sqlite"
    with SQLiteSpendLedger(path, limit_usd="1", pricing_id="a" * 64) as ledger:
        ledger.reserve("call-1", Decimal("0.7"))
    with SQLiteSpendLedger(path, limit_usd="1", pricing_id="a" * 64) as ledger:
        assert ledger.summary()["charged_or_reserved_usd"] == "0.7"
        with pytest.raises(BudgetAdmissionError):
            ledger.reserve("call-2", Decimal("0.4"))
        ledger.settle("call-1", Decimal("0.2"))
        ledger.reserve("call-2", Decimal("0.4"))
        with pytest.raises(BudgetAdmissionError):
            ledger.reserve("call-2", Decimal("0.4"))
    with pytest.raises(ValueError, match="identity"):
        SQLiteSpendLedger(path, limit_usd="5", pricing_id="a" * 64)


def test_concurrent_connections_cannot_spend_the_same_remaining_budget(tmp_path: Path):
    path = tmp_path / "budget.sqlite"
    with SQLiteSpendLedger(path, limit_usd="1", pricing_id="a" * 64):
        pass

    def reserve(index):
        with SQLiteSpendLedger(path, limit_usd="1", pricing_id="a" * 64) as ledger:
            try:
                ledger.reserve(str(index), Decimal("0.6"))
            except BudgetAdmissionError:
                return False
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, range(2))) == 1


def test_callback_settles_reported_tokens_but_keeps_error_reservation(tmp_path: Path):
    price = ModelPrice(
        provider="deepseek",
        model="test",
        input_usd_per_million_tokens="1",
        output_usd_per_million_tokens="2",
    )
    with SQLiteSpendLedger(
        tmp_path / "budget.sqlite", limit_usd="1", pricing_id="a" * 64
    ) as ledger:
        callback = BudgetCallback(ledger, price, max_output_tokens=128)
        run_id = uuid4()
        callback.on_chat_model_start(
            {}, [[HumanMessage(content="Trace the code.")]], run_id=run_id
        )
        reserved = Decimal(ledger.summary()["charged_or_reserved_usd"])
        message = AIMessage(
            content="Done",
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )
        callback.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id
        )
        assert Decimal(ledger.summary()["charged_or_reserved_usd"]) == Decimal(
            "0.000014"
        )
        assert reserved > Decimal("0.000014")
        error_id = uuid4()
        callback.on_chat_model_start(
            {}, [[HumanMessage(content="Trace again.")]], run_id=error_id
        )
        before = ledger.summary()["charged_or_reserved_usd"]
        callback.on_llm_error(RuntimeError("private token"), run_id=error_id)
        assert ledger.summary()["charged_or_reserved_usd"] == before
        assert "private token" not in str(ledger.summary())


def test_callback_denies_before_provider_and_never_refunds_missing_usage(
    tmp_path: Path,
):
    price = ModelPrice(
        provider="deepseek",
        model="test",
        input_usd_per_million_tokens="1",
        output_usd_per_million_tokens="2",
    )
    with SQLiteSpendLedger(
        tmp_path / "tiny.sqlite", limit_usd="0.000001", pricing_id="a" * 64
    ) as ledger:
        callback = BudgetCallback(ledger, price, max_output_tokens=128)
        assert callback.raise_error is True
        with pytest.raises(BudgetAdmissionError):
            callback.on_chat_model_start(
                {}, [[HumanMessage(content="Hi")]], run_id=uuid4()
            )
        assert ledger.summary()["invocations"] == 0


def test_overshoot_blocks_future_calls_and_is_not_hidden(tmp_path: Path):
    with SQLiteSpendLedger(
        tmp_path / "budget.sqlite", limit_usd="1", pricing_id="a" * 64
    ) as ledger:
        ledger.reserve("call", Decimal("0.2"))
        with pytest.raises(BudgetAdmissionError):
            ledger.settle("call", Decimal("0.3"))
        assert ledger.summary()["blocked"] is True
        assert ledger.summary()["charged_or_reserved_usd"] == "0.3"
        with pytest.raises(BudgetAdmissionError):
            ledger.reserve("next", Decimal("0.1"))
