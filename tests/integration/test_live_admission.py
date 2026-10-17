"""Real SDK + production graph denial checks using an in-process HTTP mock."""

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.lifecycle import HumanReviewPolicy
from repocompass.evaluation.live_budget import BudgetCallback, SQLiteSpendLedger
from repocompass.evaluation.live_cli import bounded_model
from repocompass.models.config import OpenAIModelSettings
from repocompass.models.factory import WorkflowModels
from repocompass.pricing import load_pricing_table_file
from repocompass.storage.raw_store import open_raw_run_store
from pathlib import Path


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_graph_stops_before_paid_http_without_fake_failed_usage(tmp_path, mode):
    prices = load_pricing_table_file(
        Path("examples/pricing.openai-standard-2026-09-05.json")
    )
    requests = []

    def unexpected(request):
        requests.append(request)
        raise AssertionError("a denied request must never reach HTTP")

    repo = tmp_path / "repo"
    repo.mkdir()
    with (
        SQLiteSpendLedger(
            tmp_path / "budget.sqlite", limit_usd="5", pricing_id=prices.table_id
        ) as ledger,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        httpx.Client(transport=httpx.MockTransport(unexpected)) as client,
    ):
        ledger.begin_attempt("denied-attempt", limit_usd="0.000001")
        guard = BudgetCallback(
            ledger,
            prices.lookup("openai", "gpt-5.4-mini"),
            max_output_tokens=128,
            attempt_id="denied-attempt",
        )
        model = bounded_model(
            OpenAIModelSettings(model="gpt-5.4-mini", api_key="test-only"),
            guard,
            http_client=client,
        )
        runner = AgentRunner(
            WorkflowModels.from_single_model(model),
            raw_run_store=raw,
            checkpointer=InMemorySaver(),
            pricing_table=prices,
            run_id_factory=lambda: "denied-attempt",
        )
        outcome = runner.run(
            RunRequest(
                repo_path=repo,
                objective="Find the public entrypoint.",
                investigation_mode=mode,
                max_tool_calls_per_step=4,
                max_explorer_calls_per_step=4,
                max_attempt_cost_usd="0.000001",
                human_review_policy=HumanReviewPolicy.NEVER,
            )
        )
        assert outcome.termination_status == "exhausted"
        assert outcome.step_stop_reason == "attempt_cost_reservation_denied"
        assert "reservation" in outcome.final_result.summary
        assert requests == []
        assert raw.list_model_invocations(outcome.trace.run_id) == ()
        assert ledger.summary()["invocations"] == 0
        assert (
            ledger.attempt_summary("denied-attempt")["charged_or_reserved_usd"] == "0"
        )
