"""Prewarm validity and non-billable target gating; no network is used."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repocompass.evaluation import live_cli
from repocompass.evaluation.live_budget import SQLiteSpendLedger
from repocompass.evaluation.localization_case_file import load_localization_dataset_file
from repocompass.evaluation.localization_contracts import (
    LocalizationRunStatus,
    LocalizationVariant,
    create_localization_budget_profile,
)
from repocompass.models.config import OpenAIModelSettings
from repocompass.memory.fact_consolidation import RepositoryConsolidationResult
from repocompass.pricing import load_pricing_table_file


@pytest.mark.parametrize(
    "status,files,evidence,episode_status,ready",
    [
        ("accepted", ["public.py"], ["source"], "accepted", True),
        ("exhausted", ["public.py"], ["source"], "exhausted", False),
        ("cancelled", ["public.py"], ["source"], "accepted", False),
        ("accepted", [], ["source"], "accepted", False),
        ("accepted", ["public.py"], [], "accepted", False),
        ("accepted", ["public.py"], ["source"], None, False),
    ],
)
def test_prewarm_requires_accepted_nonempty_episode(
    tmp_path, status, files, evidence, episode_status, ready
):
    def episodes(**kwargs):
        assert kwargs["source_run_id"] == "warm-only"
        return (
            [SimpleNamespace(episode_id="ep", termination_status=episode_status)]
            if episode_status
            else []
        )

    memory = SimpleNamespace(
        list_episodic_memories=episodes, list_memory_metadata=lambda **kwargs: []
    )
    outcome = SimpleNamespace(
        termination_status=status,
        trace=SimpleNamespace(run_id="warm-only"),
        final_result=SimpleNamespace(relevant_files=files, evidence=evidence),
    )
    result = live_cli.prewarm_readiness(outcome, memory, tmp_path)
    assert result["ready"] is ready
    assert result["repository_memory_count"] == 0
    assert result["memory_kind"] == "observed_episode_not_verified_repository_fact"


@pytest.mark.parametrize("flag", ["--attempt-limit-usd", "--warmup-limit-usd"])
def test_attempt_limit_cannot_exceed_batch_before_work(tmp_path, capsys, flag):
    assert (
        live_cli.main(
            [
                "run",
                "--cache-root",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "out"),
                "--allow-paid-api",
                "--limit-usd",
                "5",
                flag,
                "5.01",
            ]
        )
        == 1
    )
    assert "must not exceed" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_failed_prewarm_never_starts_target_and_retains_receipt(tmp_path, monkeypatch):
    manifest = load_localization_dataset_file(
        Path("benchmarks/cases/swe-bench-lite-localization-10-v1.json")
    )
    prices = load_pricing_table_file(
        Path("examples/pricing.openai-standard-2026-09-05.json")
    )
    case = manifest.cases[0]
    variant = LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY
    args = live_cli.parser().parse_args(
        [
            "run",
            "--cache-root",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "out"),
            "--warm-memory",
        ]
    )
    calls = []

    class FakeRunner:
        def __init__(self, *args, run_id_factory, **kwargs):
            self.run_id = run_id_factory()

        def run(self, request):
            calls.append(self.run_id)
            return SimpleNamespace(
                termination_status="exhausted",
                trace=SimpleNamespace(run_id=self.run_id),
                final_result=None,
                step_stop_reason="attempt_cost_reservation_denied",
                repository_consolidation=RepositoryConsolidationResult(
                    reason="not_accepted"
                ),
            )

    monkeypatch.setattr(live_cli, "AgentRunner", FakeRunner)
    monkeypatch.setattr(live_cli.JsonlTraceSink, "write", lambda *args: None)
    monkeypatch.setattr(live_cli, "validate_snapshot", lambda *args: None)
    monkeypatch.setattr(
        live_cli, "create_localization_run_request", lambda **kwargs: object()
    )
    checkout = tmp_path / "repo"
    checkout.mkdir()
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=6,
        max_explorer_calls_per_step=5,
        max_no_progress_rounds=2,
        max_plan_revisions=1,
        context_budget_tokens=32768,
        max_attempt_cost_usd="0.1",
        max_repository_memory_results=3,
        max_episodic_memory_results=3,
    )
    with SQLiteSpendLedger(
        tmp_path / "budget.sqlite", limit_usd="5", pricing_id=prices.table_id
    ) as ledger:
        observation, created = live_cli._cell(
            args.output,
            checkout,
            case,
            variant,
            0,
            manifest,
            budget,
            prices,
            OpenAIModelSettings(model="gpt-5.4-mini", api_key="test-only"),
            ledger,
            args,
        )
        assert created
        assert observation.status is LocalizationRunStatus.FAILED
        assert len(calls) == 1
        assert calls[0].startswith("warmup-")
        assert ledger.summary()["invocations"] == 0
    receipt = json.loads(next(args.output.glob("cells/*/receipt.json")).read_text())
    assert receipt["error_code"] == "PrewarmNotReadyError"
    assert receipt["target_started"] is False
    assert receipt["attempt_budget"] is None
    assert receipt["target_usage_cost_report"] is None
    assert receipt["target_repository_consolidation"] is None
    assert receipt["warmup"]["ready"] is False
    assert receipt["warmup"]["state"] == "exhausted"
    assert receipt["warmup"]["attempt_budget"]["limit_usd"] == "0.1"
    assert receipt["warmup"]["repository_consolidation"] == {
        "schema_version": 1,
        "policy": "accepted_static_python_v1",
        "reason": "not_accepted",
        "created_ids": [],
        "existing_ids": [],
        "tombstoned_ids": [],
        "truncated": False,
    }
