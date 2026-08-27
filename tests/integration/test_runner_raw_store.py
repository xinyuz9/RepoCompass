"""Runner integration with durable raw context and model provenance."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.cost import (
    CostBudgetReason,
    CostBudgetStatus,
    ModelPrice,
    create_pricing_table,
)
from repocompass.agent.lifecycle import ExecutionStatus, HumanReviewPolicy
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.usage import ModelInvocationStatus, ModelRole
from repocompass.models import WorkflowModels
from repocompass.storage.raw_store import (
    AttemptStatus,
    RawRunStore,
    open_raw_run_store,
)
from tests.model_doubles import (
    candidate_from_final_result,
    default_investigation_plan,
    default_semantic_assessment,
    with_scripted_planner,
)


def _envelope(parsed: object, *, input_tokens: int, output_tokens: int) -> dict:
    return {
        "raw": AIMessage(
            content="structured provider response",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        ),
        "parsed": parsed,
        "parsing_error": None,
    }


def _result() -> FinalResult:
    return FinalResult(
        relevant_files=["example.py"],
        evidence=["example.py contains the relevant implementation."],
        summary="The relevant implementation is in example.py.",
        unresolved_questions=[],
    )


def _models(
    result: FinalResult,
    *,
    explorer_error: Exception | None = None,
) -> WorkflowModels:
    explorer = MagicMock(spec=BaseChatModel)
    explorer.metadata = {
        "repocompass_provider": "openai",
        "repocompass_model": "gpt-integration",
    }

    def explore(_: object) -> AIMessage:
        if explorer_error is not None:
            raise explorer_error
        return AIMessage(
            content="Enough evidence is available.",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
            },
        )

    explorer.bind_tools.return_value = RunnableLambda(explore)
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: _envelope(
            candidate_from_final_result(result), input_tokens=30, output_tokens=6
        )
    )
    models = with_scripted_planner(explorer)
    models.planner.with_structured_output.return_value = RunnableLambda(
        lambda _: _envelope(
            default_investigation_plan(), input_tokens=10, output_tokens=2
        )
    )
    models.evaluator.with_structured_output.return_value = RunnableLambda(
        lambda _: _envelope(
            default_semantic_assessment(), input_tokens=40, output_tokens=8
        )
    )
    for model in (models.planner, models.evaluator):
        model.metadata = {
            "repocompass_provider": "openai",
            "repocompass_model": "gpt-integration",
        }
    return models


def _request(repo_path: Path) -> RunRequest:
    return RunRequest(
        repo_path=repo_path,
        objective="Inspect the repository.",
        max_tool_calls_per_step=2,
        max_explorer_calls_per_step=2,
    )


def _runner(models: WorkflowModels, store: RawRunStore) -> AgentRunner:
    return AgentRunner(
        models,
        checkpointer=InMemorySaver(),
        raw_run_store=store,
        run_id_factory=lambda: "raw-run",
        thread_id_factory=lambda: "raw-thread",
    )


def test_runner_persists_complete_model_provenance(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        outcome = _runner(_models(_result()), store).run(_request(tmp_path))

        attempt = store.get_attempt("raw-run")
        assert attempt is not None
        assert attempt.status is AttemptStatus.COMPLETED
        assert attempt.thread_id == outcome.thread_id == "raw-thread"
        invocations = store.list_model_invocations("raw-run")
        assert [invocation.sequence for invocation in invocations] == [0, 1, 2, 3]
        assert [invocation.role for invocation in invocations] == [
            ModelRole.PLANNER,
            ModelRole.EXPLORER,
            ModelRole.CANDIDATE_BUILDER,
            ModelRole.EVALUATOR,
        ]
        assert [invocation.node_name for invocation in invocations] == [
            "plan_investigation",
            "agent",
            "build_candidate",
            "evaluate_candidate",
        ]
        assert all(
            invocation.status is ModelInvocationStatus.SUCCEEDED
            for invocation in invocations
        )
        assert all(invocation.provider == "openai" for invocation in invocations)
        assert all(invocation.model == "gpt-integration" for invocation in invocations)
        assert all(
            invocation.usage.availability.value == "reported"
            for invocation in invocations
        )
        assert [invocation.usage.total_tokens for invocation in invocations] == [
            12,
            24,
            36,
            48,
        ]
        assert len(store.list_context_packs("raw-run")) == len(invocations)
        for invocation in invocations:
            assert (
                store.get_context_pack("raw-run", invocation.context_pack_id)
                is not None
            )
            assert store.get_artifact(invocation.input_artifact_id) is not None
            assert invocation.output_artifact_id is not None
            assert store.get_artifact(invocation.output_artifact_id) is not None

        explorer_output = store.get_artifact(invocations[1].output_artifact_id or "")
        assert explorer_output is not None
        assert b"Enough evidence is available." in explorer_output.content


def test_runner_marks_human_review_attempt_interrupted(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    request = _request(tmp_path).model_copy(
        update={"human_review_policy": HumanReviewPolicy.BEFORE_FINALIZE}
    )

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        outcome = _runner(_models(_result()), store).run(request)

        assert outcome.execution_status is ExecutionStatus.INTERRUPTED
        assert outcome.interrupt_id is not None
        attempt = store.get_attempt("raw-run")
        assert attempt is not None
        assert attempt.status is AttemptStatus.INTERRUPTED
        assert len(store.list_model_invocations("raw-run")) == 4


def test_runner_marks_attempt_failed_after_provider_error(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    provider_error = RuntimeError("private provider failure detail")

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        runner = _runner(_models(_result(), explorer_error=provider_error), store)

        with pytest.raises(RuntimeError, match="private provider failure detail"):
            runner.run(_request(tmp_path))

        attempt = store.get_attempt("raw-run")
        assert attempt is not None
        assert attempt.status is AttemptStatus.FAILED
        invocations = store.list_model_invocations("raw-run")
        assert [invocation.status for invocation in invocations] == [
            ModelInvocationStatus.SUCCEEDED,
            ModelInvocationStatus.FAILED,
        ]
        assert invocations[-1].error_code == "model_call_failed"
        assert invocations[-1].output_artifact_id is None
        assert (
            "private provider failure detail" not in invocations[-1].model_dump_json()
        )


def test_graph_closes_deterministically_after_planner_cost_overshoot(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    pricing = create_pricing_table(
        version="integration-v1",
        effective_from="2026-08-24",
        entries=(
            ModelPrice(
                provider="openai",
                model="gpt-integration",
                input_usd_per_million_tokens="10",
                output_usd_per_million_tokens="10",
            ),
        ),
    )

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        outcome = AgentRunner(
            _models(_result()),
            checkpointer=InMemorySaver(),
            raw_run_store=store,
            pricing_table=pricing,
            run_id_factory=lambda: "cost-run",
            thread_id_factory=lambda: "cost-thread",
        ).run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect the repository.",
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=2,
                max_attempt_cost_usd="0.0001",
            )
        )

        assert outcome.termination_status == "exhausted"
        assert outcome.step_stop_reason == "attempt_cost_budget_exhausted"
        assert outcome.final_result is not None
        assert outcome.final_result.relevant_files == []
        assert "cost budget" in outcome.final_result.summary
        assert outcome.usage_cost_report is not None
        assert outcome.usage_cost_report.budget is not None
        assert outcome.usage_cost_report.budget.status is CostBudgetStatus.OVERSHOT
        assert (
            outcome.usage_cost_report.budget.reason
            is CostBudgetReason.POST_RESPONSE_OVERSHOOT
        )
        assert outcome.usage_cost_report.budget.overshoot_usd == "0.00002"
        assert [item.role for item in store.list_model_invocations("cost-run")] == [
            ModelRole.PLANNER
        ]
        attempt = store.get_attempt("cost-run")
        assert attempt is not None
        assert attempt.status is AttemptStatus.COMPLETED
        assert [
            event.node for event in outcome.trace.events if event.node is not None
        ] == [
            "plan_investigation",
            "agent",
            "finalize_cost_budget_exhausted",
        ]


def test_resume_rejects_a_different_cost_budget_pricing_snapshot(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.py").write_text("value = True\n", encoding="utf-8")
    original_pricing = create_pricing_table(
        version="original-v1",
        effective_from="2026-08-24",
        entries=(
            ModelPrice(
                provider="openai",
                model="gpt-integration",
                input_usd_per_million_tokens="10",
                output_usd_per_million_tokens="10",
            ),
        ),
    )
    replacement_pricing = create_pricing_table(
        version="replacement-v1",
        effective_from="2026-08-24",
        entries=(
            ModelPrice(
                provider="openai",
                model="gpt-integration",
                input_usd_per_million_tokens="20",
                output_usd_per_million_tokens="20",
            ),
        ),
    )
    checkpointer = InMemorySaver()

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        initial = AgentRunner(
            _models(_result()),
            checkpointer=checkpointer,
            raw_run_store=store,
            pricing_table=original_pricing,
            run_id_factory=lambda: "priced-start",
            thread_id_factory=lambda: "priced-thread",
        ).run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect the repository.",
                max_tool_calls_per_step=2,
                max_explorer_calls_per_step=2,
                max_attempt_cost_usd="1",
            )
        )

        resumed_runner = AgentRunner(
            _models(_result()),
            checkpointer=checkpointer,
            raw_run_store=store,
            pricing_table=replacement_pricing,
            run_id_factory=lambda: "priced-resume",
        )
        with pytest.raises(ValueError, match="pricing snapshot"):
            resumed_runner.resume(ResumeRequest(thread_id=initial.thread_id))

        assert store.get_attempt("priced-resume") is None
