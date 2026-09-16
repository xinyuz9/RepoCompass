"""Live API boundary tests using an in-process HTTP mock, never paid calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from repocompass.agent.cost import ModelPrice
from repocompass.evaluation import live_cli
from repocompass.evaluation.live_budget import (
    BudgetAdmissionError,
    BudgetCallback,
    SQLiteSpendLedger,
)
from repocompass.evaluation.live_cli import (
    ProviderConnectivityError,
    bounded_model,
    budget_path,
    check_provider_connectivity,
    consecutive_new_failures,
    main,
    persist,
    pilot_settings,
    parser,
    public_error_metadata,
)
from repocompass.evaluation.localization_contracts import LocalizationRunStatus
from repocompass.models.config import OpenAIModelSettings, DeepSeekModelSettings
from repocompass.pricing import load_pricing_table_file


@pytest.mark.parametrize("budget", [0, 1023, 131073])
def test_invalid_context_budget_fails_before_source_or_provider_work(
    tmp_path, capsys, budget
):
    assert (
        main(
            [
                "run",
                "--cache-root",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "out"),
                "--allow-paid-api",
                "--context-budget-tokens",
                str(budget),
            ]
        )
        == 1
    )
    assert "context budget" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_context_budget_is_explicit_without_relabelling_legacy_default(tmp_path):
    base = [
        "run",
        "--cache-root",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "out"),
    ]
    assert parser().parse_args(base).context_budget_tokens == 8192
    assert (
        parser()
        .parse_args([*base, "--context-budget-tokens", "32768"])
        .context_budget_tokens
        == 32768
    )


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


@pytest.mark.parametrize("status", [200, 401, 404, 302, 407, 421, 429, 503])
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
    if status not in (200, 401, 403, 404):
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
        "https://api.deepseek.com/v1/arbitrary/path",
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


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize("legacy_version", [3, 4, 5, 6, 7, 8, 9, 10])
def test_memory_policy_changes_protocol_before_any_network_or_spend(
    tmp_path, monkeypatch, provider, legacy_version
):
    args = parser().parse_args(
        [
            "run",
            "--provider",
            provider,
            "--cache-root",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "out"),
            "--case-count",
            "1",
            "--allow-paid-api",
        ]
    )
    settings = (
        OpenAIModelSettings(model="gpt-5.4-mini", api_key="test-only")
        if provider == "openai"
        else DeepSeekModelSettings(
            model="deepseek-v4-flash",
            api_key="test-only",
            base_url="https://api.deepseek.com",
        )
    )
    monkeypatch.setattr(live_cli, "pilot_settings", lambda *args: settings)
    monkeypatch.setattr(
        live_cli, "prepare_snapshot", lambda *args, **kwargs: tmp_path / "checkout"
    )
    probes = []

    class OfflineStop(Exception):
        pass

    def no_network(*args, **kwargs):
        probes.append(True)
        raise OfflineStop()

    monkeypatch.setattr(live_cli, "check_provider_connectivity", no_network)
    with pytest.raises(OfflineStop):
        live_cli.execute(args)
    path = args.output / "config.json"
    config = json.loads(path.read_text())
    assert config["schema_version"] == 11
    assert (
        config["revision_rejection_policy"] == "terminal_exhausted_preserve_history_v1"
    )
    assert config["repository_memory_policy"] == "accepted_static_python_v1"
    assert config["memory_store_schema_version"] == 4
    assert config["prewarm_validation"] == "accepted_nonempty_episode_v1"
    assert (
        config["structure_resolution_policy"] == "self_import_placeholder_exact_path_v2"
    )
    assert (
        config["candidate_evidence_policy"]
        == "source_occurrence_discovery_separation_v2"
    )
    assert (
        config["workflow_prompt_policy"] == "evidence_linked_remaining_plan_coverage_v6"
    )
    assert (
        config["candidate_output_policy"]
        == "selected_ast_or_unbound_source_observations_v2"
    )
    assert (
        config["plan_coverage_policy"]
        == "full_plan_selected_excerpts_identity_bound_v1"
    )
    assert config["step_completion_policy"] == "deterministic_and_semantic_gate_v2"
    assert config["validation_feedback_policy"] == "bounded_pinned_defects_v1"
    assert config["gateway_feedback_policy"] == "last_batch_body_free_pinned_v1"
    assert config["plan_revision_policy"] == "controller_owned_completion_v1"
    assert config["trace_persistence_policy"] == "transactional_node_events_v1"
    assert config["raw_store_schema_version"] == 2
    assert config["context_snapshot_policy"] == "content_dedup_per_invocation_usage_v2"
    assert not (args.output / "budget.sqlite").exists()

    # Simulate a previous experiment, not permission to rewrite real artifacts.
    config["schema_version"] = legacy_version
    config.pop("candidate_output_policy")
    config.pop("plan_coverage_policy")
    config.pop("gateway_feedback_policy")
    config.pop("plan_revision_policy")
    config.pop("context_snapshot_policy")
    if legacy_version < 5:
        config.pop("structure_resolution_policy")
        config.pop("candidate_evidence_policy")
        config.pop("workflow_prompt_policy")
    if legacy_version == 3:
        config.pop("repository_memory_policy")
        config.pop("memory_store_schema_version")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity differs"):
        live_cli.execute(args)
    assert probes == [True]
    assert json.loads(path.read_text()) == config
    assert not (args.output / "budget.sqlite").exists()


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


@pytest.mark.parametrize("status", [200, 401, 302, 403, 404, 421, 503])
def test_openai_probe_uses_unauthenticated_models_endpoint(monkeypatch, status):
    requests = []
    original = httpx.Client

    def transport(request):
        requests.append(request)
        return httpx.Response(status)

    monkeypatch.setattr(
        "repocompass.evaluation.live_cli.httpx.Client",
        lambda **kw: original(transport=httpx.MockTransport(transport), **kw),
    )
    if status in (200, 401):
        check_provider_connectivity("https://api.openai.com/v1", provider="openai")
    else:
        with pytest.raises(ProviderConnectivityError):
            check_provider_connectivity("https://api.openai.com/v1", provider="openai")
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "https://api.openai.com/v1/models"
    assert "authorization" not in requests[0].headers


def test_openai_never_reuses_deepseek_key_or_endpoint(monkeypatch):
    monkeypatch.setattr(
        "repocompass.evaluation.live_cli.load_dotenv", lambda **kw: None
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "deepseek-only")
    monkeypatch.setenv("REPOCOMPASS_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("REPOCOMPASS_BASE_URL", "https://api.deepseek.com")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        pilot_settings("openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    settings = pilot_settings("openai", None)
    assert settings.api_key == "openai-only"
    assert settings.model == "gpt-5.4-mini"
    assert "openai-only" not in repr(settings)
    with pytest.raises(ValueError, match="reviewed"):
        pilot_settings("openai", "unreviewed-model")
    with pytest.raises(ValueError):
        check_provider_connectivity("https://api.deepseek.com", provider="openai")


def test_shared_budget_must_exist_and_stay_outside_source_cache(tmp_path):
    args = SimpleNamespace(budget_db=None, allow_pricing_extension=True)
    with pytest.raises(ValueError, match="requires"):
        budget_path(args, tmp_path / "out", tmp_path / "cache")
    args.budget_db = tmp_path / "typo.sqlite"
    with pytest.raises(ValueError, match="existing"):
        budget_path(args, tmp_path / "out", tmp_path / "cache")
    assert not args.budget_db.exists()
    cache = tmp_path / "cache"
    cache.mkdir()
    args.budget_db = cache / "budget.sqlite"
    args.budget_db.touch()
    with pytest.raises(ValueError, match="source cache"):
        budget_path(args, tmp_path / "out", cache)
    args.budget_db = tmp_path / "empty.sqlite"
    args.budget_db.touch()
    with pytest.raises(ValueError, match="initialized"):
        budget_path(args, tmp_path / "out", cache)
    with SQLiteSpendLedger(args.budget_db, limit_usd="5", pricing_id="a" * 64):
        pass
    assert budget_path(args, tmp_path / "out", cache) == args.budget_db


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize("limit,expected_calls", [("1", 1), ("0.000001", 0)])
def test_bounded_provider_model_has_correct_wire_contract(
    tmp_path, provider, limit, expected_calls, monkeypatch
):
    requests = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unreviewed.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-ambient-key")
    settings = (
        OpenAIModelSettings(model="gpt-5.4-mini", api_key="openai-only")
        if provider == "openai"
        else DeepSeekModelSettings(
            model="deepseek-v4-flash",
            api_key="deepseek-only",
            base_url="https://api.deepseek.com",
        )
    )
    prices = load_pricing_table_file(
        Path(
            "examples/pricing.openai-standard-2026-09-05.json"
            if provider == "openai"
            else "examples/pricing.deepseek-peak-2026-09-05.json"
        )
    )

    def transport(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": settings.model,
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

    with SQLiteSpendLedger(
        tmp_path / "budget.sqlite", limit_usd=limit, pricing_id=prices.table_id
    ) as ledger:
        guard = BudgetCallback(
            ledger, prices.lookup(provider, settings.model), max_output_tokens=128
        )
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            model = bounded_model(settings, guard, http_client=client)
            assert model.root_client.max_retries == 0
            if expected_calls:
                assert model.invoke([HumanMessage(content="Trace")]).content == "Done"
                assert ledger.summary()["settled_invocations"] == 1
            else:
                with pytest.raises(BudgetAdmissionError):
                    model.invoke([HumanMessage(content="Trace")])
            assert len(requests) == expected_calls
            if not requests:
                return
            request = requests[0]
            assert request.url.host == (
                "api.openai.com" if provider == "openai" else "api.deepseek.com"
            )
            assert request.headers["authorization"] == f"Bearer {settings.api_key}"
            payload = json.loads(request.content)
            assert payload["max_completion_tokens"] == 128
            assert payload["n"] == 1
            if provider == "openai":
                assert payload["reasoning_effort"] == "none"
                assert payload["service_tier"] == "default"
                assert payload["store"] is False
                assert "thinking" not in payload
                assert "temperature" not in payload
                assert ledger.summary()["charged_or_reserved_usd"] == "0.0000165"
            else:
                assert payload["thinking"] == {"type": "disabled"}
                assert "reasoning_effort" not in payload
