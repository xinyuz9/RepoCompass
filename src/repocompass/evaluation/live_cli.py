"""Opt-in real localization pilot with durable cells and a shared spending ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.context_builder import DEFAULT_CONTEXT_BUDGET_TOKENS
from repocompass.agent.cost import CostUnavailableReason, validate_cost_budget_usd
from repocompass.memory.contracts import MemoryScope
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.agent.runner import AgentRunner, RunRequest
from repocompass.agent.lifecycle import HumanReviewPolicy
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.evaluation.checkout import prepare_snapshot, validate_snapshot
from repocompass.evaluation.live_budget import BudgetCallback, SQLiteSpendLedger
from repocompass.evaluation.localization_adapter import (
    adapt_localization_run_outcome,
    create_localization_run_request,
    localization_architecture_profile,
)
from repocompass.evaluation.localization_case_file import load_localization_dataset_file
from repocompass.evaluation.localization_contracts import (
    LocalizationRunObservation,
    LocalizationRunStatus,
    LocalizationVariant,
    LocalizationMetricUnavailableReason,
    create_localization_budget_profile,
    create_localization_dataset_manifest,
    create_localization_run_observation,
    create_localization_prediction,
    create_unavailable_cost_observation,
    observed_localization_metric,
    unavailable_localization_metric,
)
from repocompass.evaluation.localization_harness import (
    create_localization_observation_set,
    render_localization_summary,
)
from repocompass.evaluation.localization_metrics import evaluate_localization_dataset
from repocompass.memory.store import MEMORY_STORE_SCHEMA_VERSION, open_memory_store
from repocompass.models.config import (
    DeepSeekModelSettings,
    OpenAIModelSettings,
    load_model_settings,
)
from repocompass.models.factory import WorkflowModels
from repocompass.pricing import load_pricing_table_file
from repocompass.storage.raw_store import open_raw_run_store

DEFAULT_VARIANTS = (
    "tool_wrapper",
    "evaluator_loop_context_builder",
    "evaluator_loop_context_builder_memory",
)
WARMUP_OBJECTIVE = (
    "Identify the repository's package public entrypoints and test configuration. "
    "Use only repository evidence. Do not diagnose or solve any issue."
)


class ProviderConnectivityError(RuntimeError):
    """A credential-free connectivity probe failed before paid admission."""


def check_provider_connectivity(base_url: str, *, provider: str = "deepseek") -> None:
    """Fail before reserving a cell when the configured TLS route is broken."""
    endpoint = urlsplit(base_url)
    hosts = {"deepseek": "api.deepseek.com", "openai": "api.openai.com"}
    if (
        provider not in hosts
        or endpoint.scheme != "https"
        or endpoint.hostname != hosts.get(provider)
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.port not in (None, 443)
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in ("", "/", "/v1", "/v1/")
    ):
        raise ValueError(
            "provider endpoint must be the direct credential-free HTTPS origin"
        )
    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            # An unauthenticated 401 proves routing, not credentials/model access.
            if provider == "openai":
                response = client.get("https://api.openai.com/v1/models")
                accepted = (200, 401)
            else:
                response = client.head("https://api.deepseek.com")
                accepted = (200, 401, 403, 404)
            if response.status_code not in accepted:
                raise ProviderConnectivityError("provider route is unavailable")
    except httpx.HTTPError:
        raise ProviderConnectivityError(
            "provider TLS/HTTP connectivity failed"
        ) from None


def pilot_settings(provider: str, model: str | None):
    """Never reinterpret a DeepSeek generic key as an OpenAI credential."""
    if provider == "openai":
        load_dotenv(override=False)
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "OpenAI pilot requires OPENAI_API_KEY; do not reuse the DeepSeek key"
            )
        selected = model or "gpt-5.4-mini"
        if selected != "gpt-5.4-mini":
            raise ValueError("OpenAI pilot currently admits only reviewed gpt-5.4-mini")
        return OpenAIModelSettings(model=selected, api_key=key)
    settings = load_model_settings()
    if not isinstance(settings, DeepSeekModelSettings):
        raise ValueError(
            "DeepSeek pilot requires explicitly configured DeepSeek settings"
        )
    return replace(settings, model=model or "deepseek-v4-flash")


def provider_endpoint(settings) -> str:
    return (
        "https://api.openai.com/v1"
        if isinstance(settings, OpenAIModelSettings)
        else settings.base_url
    )


def bounded_model(settings, guard: BudgetCallback, *, http_client=None):
    """Only supported text parameters; no SDK retries or hidden provider fallback."""
    options = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": provider_endpoint(settings),
        "use_responses_api": False,
        "metadata": {
            "repocompass_provider": settings.provider.value,
            "repocompass_model": settings.model,
        },
        "max_retries": 0,
        "timeout": 45.0,
        "callbacks": [guard],
        "cache": False,
        "disable_streaming": True,
        "n": 1,
    }
    if http_client is not None:
        options["http_client"] = http_client
    if isinstance(settings, OpenAIModelSettings):
        options.update(
            max_completion_tokens=guard.max_output_tokens,
            reasoning_effort="none",
            service_tier="default",
            store=False,
        )
    else:
        options.update(
            max_tokens=guard.max_output_tokens,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return ChatOpenAI(**options)


def attempt_model(settings, price, ledger, args, *, run_id: str, limit_usd: str):
    ledger.begin_attempt(run_id, limit_usd=limit_usd)
    guard = BudgetCallback(
        ledger, price, max_output_tokens=args.max_output_tokens, attempt_id=run_id
    )
    return bounded_model(settings, guard)


def prewarm_readiness(outcome, memory, checkout: Path) -> dict:
    """Validate an accepted label-free episode; never call it a repository fact."""
    context = ToolExecutionContext(checkout)
    episodes = memory.list_episodic_memories(
        context=context, source_run_id=outcome.trace.run_id, max_results=1
    )
    accepted_ids = [
        episode.episode_id
        for episode in episodes
        if episode.termination_status == "accepted"
    ]
    repository_entries = memory.list_memory_metadata(
        context=context, memory_scope=MemoryScope.REPOSITORY, max_results=50
    )
    ready = bool(
        outcome.termination_status == "accepted"
        and outcome.final_result is not None
        and outcome.final_result.relevant_files
        and outcome.final_result.evidence
        and accepted_ids
    )
    return {
        "ready": ready,
        "criterion": "accepted_nonempty_episode_v1",
        "accepted_episode_ids": accepted_ids,
        "repository_memory_count": len(repository_entries),
        "repository_memory_query_limit": 50,
        "memory_kind": "observed_episode_not_verified_repository_fact",
    }


class PrewarmNotReadyError(RuntimeError):
    """The target run was not started because its prewarm contract was not met."""


def budget_path(args, root: Path, cache: Path) -> Path:
    """An explicitly shared ledger must already exist; a typo must not reset funds."""
    if args.allow_pricing_extension and args.budget_db is None:
        raise ValueError("--allow-pricing-extension requires an existing --budget-db")
    if args.budget_db is None:
        return root / "budget.sqlite"
    selected = args.budget_db.expanduser()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("--budget-db must name an existing non-symlink ledger")
    selected = selected.resolve()
    if selected.is_relative_to(cache):
        raise ValueError("shared budget must remain outside Agent-visible source cache")
    try:
        with closing(sqlite3.connect(f"{selected.as_uri()}?mode=ro", uri=True)) as db:
            rows = db.execute(
                "SELECT id,ceiling,pricing,blocked FROM budget"
            ).fetchall()
            if len(rows) != 1 or rows[0][0] != 1:
                raise ValueError("shared budget has no valid singleton budget identity")
            db.execute("SELECT id,reserved,charged,status FROM calls LIMIT 0")
    except sqlite3.Error:
        raise ValueError("--budget-db is not an initialized spending ledger") from None
    return selected


def public_error_metadata(error: Exception) -> dict:
    """Do not copy provider messages, URLs, request headers or response bodies."""
    result = {"error_code": type(error).__name__}
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    if error.__cause__ is not None:
        result["cause_code"] = type(error.__cause__).__name__
    return result


def consecutive_new_failures(previous: int, observation, created: bool) -> int:
    """Retained failures remain in reports but do not prevent advancing a restart."""
    if not created:
        return previous
    return previous + 1 if observation.status is LocalizationRunStatus.FAILED else 0


def persist(path: Path, value) -> None:
    """Create immutable artifacts; retries must match, never silently replace."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("artifact must not be a symlink")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                f"existing artifact identity differs: {path.name}"
            ) from None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("prepare", "run"))
    result.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/cases/swe-bench-lite-localization-10-v1.json"),
    )
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--allow-network",
        action="store_true",
        help="prepare GitHub source snapshots only",
    )
    result.add_argument("--allow-paid-api", action="store_true")
    result.add_argument(
        "--provider", choices=("deepseek", "openai"), default="deepseek"
    )
    result.add_argument("--pricing-file", type=Path)
    result.add_argument("--model")
    result.add_argument(
        "--budget-db", type=Path, help="existing shared ledger; never resets funds"
    )
    result.add_argument(
        "--allow-pricing-extension",
        action="store_true",
        help="register this reviewed pricing identity in the existing shared budget",
    )
    result.add_argument("--limit-usd", default="5")
    result.add_argument("--attempt-limit-usd", default="0.1")
    result.add_argument("--warmup-limit-usd", default="0.1")
    result.add_argument("--case-count", type=int, default=10)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument(
        "--max-cells",
        type=int,
        default=100,
        help="pause after this many new cells; resume never reruns completed cells",
    )
    result.add_argument("--max-output-tokens", type=int, default=2048)
    result.add_argument(
        "--context-budget-tokens",
        type=int,
        default=8192,
        help="hard context UTF-8 byte upper bound, not reported model tokens; "
        f"production default is {DEFAULT_CONTEXT_BUDGET_TOKENS}; changes require a new output",
    )
    result.add_argument(
        "--variants",
        nargs="+",
        choices=[v.value for v in LocalizationVariant],
        default=list(DEFAULT_VARIANTS),
    )
    result.add_argument(
        "--warm-memory",
        action="store_true",
        help="independent label-free prewarm, isolated per case/repeat",
    )
    return result


def _failed_observation(
    manifest, case, variant, budget, pricing_id, run_id, repeat, duration
):
    unavailable = unavailable_localization_metric(
        LocalizationMetricUnavailableReason.RUN_FAILED
    )
    return create_localization_run_observation(
        dataset_manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=variant,
        budget_profile_id=budget.profile_id,
        run_id=run_id,
        repeat_index=repeat,
        status=LocalizationRunStatus.FAILED,
        prediction=create_localization_prediction(
            ranked_files=(),
            ranked_symbols=(),
            quality_unavailable_reason=LocalizationMetricUnavailableReason.RUN_FAILED,
        ),
        context_lines=unavailable,
        context_tokens=unavailable,
        tool_calls=unavailable,
        latency_ms=observed_localization_metric(duration),
        evaluator_iterations=unavailable,
        input_tokens=unavailable,
        output_tokens=unavailable,
        cost=create_unavailable_cost_observation(
            pricing_table_id=pricing_id, reason=CostUnavailableReason.USAGE_UNAVAILABLE
        ),
    )


def _cell(
    root,
    checkout,
    case,
    variant,
    repeat,
    manifest,
    budget,
    prices,
    settings,
    ledger,
    args,
):
    directory = root / "cells" / f"{case.task.instance_id}-{variant.value}-{repeat}"
    observation_path = directory / "observation.json"
    if observation_path.exists():
        return LocalizationRunObservation.model_validate_json(
            observation_path.read_text()
        ), False
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"pilot-{uuid4()}"
    # An interrupted paid call is never silently retried under a new identity.
    with (directory / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "task_id": case.task.task_id,
                "variant": variant.value,
                "repeat": repeat,
                "run_id": run_id,
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    price = prices.lookup(settings.provider.value, settings.model)
    if price is None:
        raise ValueError("provider/model has no reviewed price")
    warmup = {
        "requested": args.warm_memory,
        "state": "not_applicable",
        "ready": False,
        "repository_consolidation": None,
    }
    target_started = False
    target_repository_consolidation = None
    with (
        open_sqlite_checkpointer(directory / "checkpoints.sqlite") as checkpointer,
        open_raw_run_store(directory / "raw.sqlite") as raw,
        open_memory_store(directory / "memory.sqlite") as memory,
    ):
        is_memory = variant is LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY
        if is_memory and args.warm_memory:
            warmup["state"] = "started"
            warm_started = monotonic()
            warm_before = ledger.summary()["charged_or_reserved_usd"]
            warm_run_id = f"warmup-{uuid4()}"
            warmup["run_id"] = warm_run_id
            try:
                model = attempt_model(
                    settings,
                    price,
                    ledger,
                    args,
                    run_id=warm_run_id,
                    limit_usd=args.warmup_limit_usd,
                )
                runner = AgentRunner(
                    WorkflowModels.from_single_model(model),
                    checkpointer=checkpointer,
                    raw_run_store=raw,
                    memory_store=memory,
                    pricing_table=prices,
                    architecture_profile=localization_architecture_profile(variant),
                    run_id_factory=lambda: warm_run_id,
                )
                warm = runner.run(
                    RunRequest(
                        repo_path=checkout,
                        objective=WARMUP_OBJECTIVE,
                        max_tool_calls_per_step=4,
                        max_explorer_calls_per_step=4,
                        max_no_progress_rounds=2,
                        max_plan_revisions=1,
                        context_budget_tokens=args.context_budget_tokens,
                        max_attempt_cost_usd=args.warmup_limit_usd,
                        human_review_policy=HumanReviewPolicy.NEVER,
                        memory_retrieval_enabled=True,
                    )
                )
                JsonlTraceSink().write(
                    warm.trace, directory / "traces" / f"{warm.trace.run_id}.jsonl"
                )
                warmup.update(state=warm.termination_status, run_id=warm.trace.run_id)
                warmup.update(prewarm_readiness(warm, memory, checkout))
                warmup["stop_reason"] = warm.step_stop_reason
                if warm.repository_consolidation is not None:
                    warmup["repository_consolidation"] = (
                        warm.repository_consolidation.model_dump(mode="json")
                    )
            except Exception as error:
                warmup.update(state="failed", **public_error_metadata(error))
            warmup.update(
                latency_ms=int((monotonic() - warm_started) * 1000),
                batch_reserved_before_usd=warm_before,
                batch_reserved_after_usd=ledger.summary()["charged_or_reserved_usd"],
                attempt_budget=ledger.attempt_summary(warm_run_id),
            )
        persist(directory / "warmup.json", warmup)
        request = create_localization_run_request(
            task=case.task,
            repository_path=checkout,
            budget_profile=budget,
            variant=variant,
        )
        error_info = {"error_code": None}
        started = monotonic()
        try:
            if is_memory and args.warm_memory and not warmup["ready"]:
                raise PrewarmNotReadyError(
                    "label-free memory prewarm was not accepted; target not started"
                )
            model = attempt_model(
                settings,
                price,
                ledger,
                args,
                run_id=run_id,
                limit_usd=args.attempt_limit_usd,
            )
            runner = AgentRunner(
                WorkflowModels.from_single_model(model),
                checkpointer=checkpointer,
                raw_run_store=raw,
                memory_store=memory if is_memory else None,
                pricing_table=prices,
                architecture_profile=localization_architecture_profile(variant),
                run_id_factory=lambda: run_id,
            )
            target_started = True
            outcome = runner.run(request)
            if outcome.repository_consolidation is not None:
                target_repository_consolidation = (
                    outcome.repository_consolidation.model_dump(mode="json")
                )
            error_info["stop_reason"] = outcome.step_stop_reason
            observation = adapt_localization_run_outcome(
                dataset_manifest_id=manifest.manifest_id,
                case=case,
                variant=variant,
                budget_profile=budget,
                outcome=outcome,
                repeat_index=repeat,
                latency_ms=int((monotonic() - started) * 1000),
                pricing_table_id=prices.table_id,
            )
            JsonlTraceSink().write(
                outcome.trace, directory / "traces" / f"{outcome.trace.run_id}.jsonl"
            )
        except Exception as error:
            error_info = public_error_metadata(error)
            observation = _failed_observation(
                manifest,
                case,
                variant,
                budget,
                prices.table_id,
                run_id,
                repeat,
                int((monotonic() - started) * 1000),
            )
        validate_snapshot(case.task, checkout)
        usage_report = raw.get_attempt_usage_cost_report(run_id, prices)
        persist(directory / "observation.json", observation.model_dump(mode="json"))
        persist(
            directory / "receipt.json",
            {
                "provenance": "live_provider",
                **error_info,
                "run_id": run_id,
                "target_started": target_started,
                "target_repository_consolidation": target_repository_consolidation,
                "attempt_budget": ledger.attempt_summary(run_id)
                if target_started
                else None,
                "target_usage_cost_report": (
                    usage_report.model_dump(mode="json")
                    if usage_report is not None
                    else None
                ),
                "warmup": warmup,
                "budget": ledger.summary(),
                "telemetry_note": "Failure costs remain reserved in the batch ledger; never interpreted as zero.",
            },
        )
    return observation, True


def execute(args) -> dict:
    if args.action == "run" and not args.allow_paid_api:
        raise ValueError("run requires --allow-paid-api; no provider was contacted")
    if (
        not 1 <= args.case_count <= 40
        or not 1 <= args.repeats <= 10
        or args.max_cells < 1
    ):
        raise ValueError("case/repeat/cell limits are outside the supported range")
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("variants must be unique")
    if not 64 <= args.max_output_tokens <= 8192:
        raise ValueError("provider output cap must be between 64 and 8192")
    if not 1024 <= args.context_budget_tokens <= 131072:
        raise ValueError("context budget must be between 1024 and 131072")
    validate_cost_budget_usd(args.limit_usd)
    validate_cost_budget_usd(args.attempt_limit_usd)
    validate_cost_budget_usd(args.warmup_limit_usd)
    if max(Decimal(args.attempt_limit_usd), Decimal(args.warmup_limit_usd)) > Decimal(
        args.limit_usd
    ):
        raise ValueError("attempt and warmup limits must not exceed the batch limit")
    original = load_localization_dataset_file(args.dataset)
    if args.case_count > len(original.cases):
        raise ValueError("case count exceeds the frozen manifest")
    manifest = (
        original
        if args.case_count == len(original.cases)
        else create_localization_dataset_manifest(
            name=original.name + f"-pilot-{args.case_count}",
            version=original.version,
            kind=original.kind,
            cases=original.cases[: args.case_count],
        )
    )
    root = args.output.expanduser().resolve()
    # Never put hidden labels, raw responses or memories inside Agent-visible checkouts.
    cache = args.cache_root.expanduser().resolve()
    if root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("source cache and evaluation artifacts must be separate roots")
    paths = {}
    for case in manifest.cases:
        paths[case.task.task_id] = prepare_snapshot(
            case.task,
            cache,
            allow_network=args.action == "prepare" and args.allow_network,
        )
        print(json.dumps({"prepared": case.task.instance_id}), flush=True)
    persist(root / "manifest.json", manifest.model_dump(mode="json"))
    persist(
        root / "sources.json",
        {
            "upstream_manifest_id": original.manifest_id,
            "checkouts": {key: str(value) for key, value in paths.items()},
        },
    )
    if args.action == "prepare":
        return {"prepared_cases": len(paths), "provider_calls": 0}
    settings = pilot_settings(args.provider, args.model)
    pricing_file = args.pricing_file or Path(
        "examples/pricing.openai-standard-2026-09-05.json"
        if args.provider == "openai"
        else "examples/pricing.deepseek-peak-2026-09-05.json"
    )
    prices = load_pricing_table_file(pricing_file)
    ledger_path = budget_path(args, root, cache)
    if prices.lookup(settings.provider.value, settings.model) is None:
        raise ValueError("selected model has no reviewed pricing entry")
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=6,
        max_explorer_calls_per_step=5,
        max_no_progress_rounds=2,
        max_plan_revisions=1,
        context_budget_tokens=args.context_budget_tokens,
        max_attempt_cost_usd=args.attempt_limit_usd,
        max_repository_memory_results=3,
        max_episodic_memory_results=3,
    )
    variants = tuple(v for v in LocalizationVariant if v.value in args.variants)
    config = {
        "schema_version": 8,
        "provenance": "live_provider",
        "manifest_id": manifest.manifest_id,
        "budget_profile": budget.model_dump(mode="json"),
        "pricing": prices.model_dump(mode="json"),
        "model": settings.model,
        "variants": [v.value for v in variants],
        "repeats": args.repeats,
        "max_output_tokens": args.max_output_tokens,
        "limit_usd": args.limit_usd,
        "warmup_objective": WARMUP_OBJECTIVE if args.warm_memory else None,
        "memory_isolation": "fresh-per-case-variant-repeat; no target-answer reuse",
        "repository_memory_policy": "accepted_static_python_v1",
        "memory_store_schema_version": MEMORY_STORE_SCHEMA_VERSION,
        "structure_resolution_policy": "self_import_placeholder_exact_path_v2",
        "candidate_evidence_policy": "source_occurrence_discovery_separation_v2",
        "workflow_prompt_policy": "source_backed_validation_feedback_v4",
        "gateway_feedback_policy": "last_batch_body_free_pinned_v1",
        "plan_revision_policy": "controller_owned_completion_v1",
        "step_completion_policy": "deterministic_and_semantic_gate_v2",
        "validation_feedback_policy": "bounded_pinned_defects_v1",
        "trace_persistence_policy": "transactional_node_events_v1",
        "raw_store_schema_version": 2,
        "context_snapshot_policy": "content_dedup_per_invocation_usage_v2",
        "provider_retries": 0,
        "price_basis": "peak_ceiling_not_invoice",
        "admission_policy": "atomic_batch_and_attempt_v1",
        "prewarm_validation": "accepted_nonempty_episode_v1",
        "warmup_limit_usd": args.warmup_limit_usd,
        "progress_policy": "observed_result_novelty_v2",
    }
    if args.provider == "openai" or args.budget_db is not None:
        config.update(
            provider=args.provider,
            endpoint=provider_endpoint(settings),
            budget_db=str(ledger_path),
            price_basis="standard_cache_miss_ceiling_not_invoice"
            if args.provider == "openai"
            else "peak_ceiling_not_invoice",
        )
        if args.provider == "openai":
            config.update(reasoning_effort="none", service_tier="default", store=False)
    persist(root / "config.json", config)
    check_provider_connectivity(provider_endpoint(settings), provider=args.provider)
    observations = []
    new_cells = 0
    failures = 0
    stopped = None
    with SQLiteSpendLedger(
        ledger_path,
        limit_usd=args.limit_usd,
        pricing_id=prices.table_id,
        allow_pricing_extension=args.allow_pricing_extension,
    ) as ledger:
        # Case-balanced order permits a useful paired pilot before a bounded pause.
        for case in manifest.cases:
            for repeat in range(args.repeats):
                for variant in variants:
                    if (
                        new_cells >= args.max_cells
                        or failures >= 2
                        or ledger.summary()["blocked"]
                    ):
                        stopped = (
                            "cell_limit_or_consecutive_failures_or_budget_boundary"
                        )
                        break
                    observation, created = _cell(
                        root,
                        paths[case.task.task_id],
                        case,
                        variant,
                        repeat,
                        manifest,
                        budget,
                        prices,
                        settings,
                        ledger,
                        args,
                    )
                    observations.append(observation)
                    new_cells += int(created)
                    failures = consecutive_new_failures(failures, observation, created)
                    print(
                        json.dumps(
                            {
                                "instance": case.task.instance_id,
                                "variant": variant.value,
                                "repeat": repeat,
                                "status": observation.status.value,
                                "reused": not created,
                                "budget": ledger.summary(),
                            }
                        ),
                        flush=True,
                    )
                if stopped:
                    break
            if stopped:
                break
        spending = ledger.summary()
    if observations:
        bundle = create_localization_observation_set(
            dataset_manifest_id=manifest.manifest_id,
            budget_profile=budget,
            observations=tuple(observations),
        )
        report = evaluate_localization_dataset(manifest, bundle.observations)
        suffix = hashlib.sha256(bundle.set_id.encode()).hexdigest()[:12]
        persist(root / f"observations-{suffix}.json", bundle.model_dump(mode="json"))
        persist(
            root / f"report-{suffix}.json",
            report.model_dump(mode="json", exclude_computed_fields=True),
        )
        summary_path = root / f"report-{suffix}.md"
        if not summary_path.exists():
            with summary_path.open("x", encoding="utf-8") as stream:
                stream.write(render_localization_summary(report))
    expected = len(manifest.cases) * len(variants) * args.repeats
    result = {
        "complete": len(observations) == expected,
        "expected_cells": expected,
        "observed_cells": len(observations),
        "new_cells": new_cells,
        "stopped": stopped,
        "spending": spending,
    }
    progress_id = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()[:12]
    persist(
        root / f"progress-{len(observations)}-{new_cells}-{progress_id}.json", result
    )
    return result


def main(argv=None) -> int:
    try:
        result = execute(parser().parse_args(argv))
    except (ValueError, OSError, RuntimeError) as error:
        # Do not leak provider response bodies, endpoint credentials, or prompts.
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "detail": str(error)
                    if isinstance(error, ValueError)
                    else "Operation failed; inspect local receipts before retrying.",
                }
            ),
            flush=True,
        )
        return 1
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
