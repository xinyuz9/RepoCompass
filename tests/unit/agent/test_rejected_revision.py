"""Rejected completion-history edits terminate without implicit paid retries."""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.evaluation import SemanticAssessment
from repocompass.agent.planning import InvestigationPlan, PlanStep, PlanProposal
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.agent.trace import TraceEventKind
from repocompass.agent.usage import ModelInvocationStatus
from repocompass.models import WorkflowModels
from repocompass.storage.raw_store import AttemptStatus, open_raw_run_store


def _step(identifier, status):
    return PlanStep(
        id=identifier,
        objective="Inspect source.",
        expected_evidence=["Source lines."],
        status=status,
    )


def _models(*, provider_failure=False):
    plan = InvestigationPlan(steps=[_step("inspect", "active")])
    rejected = InvestigationPlan(
        steps=[_step("inspect", "completed"), _step("retry", "active")]
    )
    responses = iter([plan, rejected])
    calls = []
    model = MagicMock(spec=BaseChatModel)

    def raw():
        return AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
            },
        )

    def explorer(_):
        calls.append("explorer")
        return raw()

    def structured(schema, **_):
        def invoke(_messages):
            calls.append(schema.__name__)
            if schema is PlanProposal:
                value = next(responses)
                if value == rejected and provider_failure:
                    raise ValueError("provider fault must not become a successful stop")
            elif schema is EvidenceOnlyCandidateResult:
                value = EvidenceOnlyCandidateResult(
                    evidence=[],
                    summary="unvalidated candidate narrative",
                    unresolved_questions=["Source required."],
                )
            else:
                assert schema is SemanticAssessment
                value = SemanticAssessment(
                    active_step_completed=False,
                    answers_objective=False,
                    evidence_supports_summary=False,
                    has_critical_omissions=True,
                    missing_evidence=["Read source."],
                    recommended_focus=["Inspect source."],
                    summary="Not supported.",
                )
            return {"raw": raw(), "parsed": value, "parsing_error": None}

        return RunnableLambda(invoke)

    model.bind_tools.return_value = RunnableLambda(explorer)
    model.with_structured_output.side_effect = structured
    return WorkflowModels.from_single_model(model), calls, plan


def test_rejected_revision_finishes_checkpoint_and_preserves_failed_usage(tmp_path):
    models, calls, original_plan = _models()
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        runner = AgentRunner(models, checkpointer=InMemorySaver(), raw_run_store=store)
        outcome = runner.run(
            RunRequest(
                repo_path=tmp_path,
                objective="Inspect source.",
                max_tool_calls_per_step=1,
                max_explorer_calls_per_step=1,
                max_plan_revisions=1,
            )
        )
        assert outcome.termination_status == "exhausted"
        assert outcome.step_stop_reason == "invalid_plan_revision"
        assert outcome.final_result.evidence == []
        assert "unvalidated candidate narrative" not in outcome.final_result.summary
        view = runner.inspect_thread(ResumeRequest(thread_id=outcome.thread_id))
        assert view.execution_state is ThreadExecutionState.COMPLETED
        assert view.next_nodes == ()
        assert view.plan == original_plan
        assert view.evaluation_report is not None and not view.evaluation_report.passed
        assert calls == [
            "PlanProposal",
            "explorer",
            "EvidenceOnlyCandidateResult",
            "SemanticAssessment",
            "PlanProposal",
        ]
        invocations = store.list_model_invocations(outcome.trace.run_id)
        assert len(invocations) == 5
        rejected = invocations[-1]
        assert rejected.status is ModelInvocationStatus.FAILED
        assert rejected.error_code == "invalid_model_response"
        assert rejected.usage.input_tokens == 100 and rejected.usage.output_tokens == 25
        assert rejected.output_artifact_id is not None
        assert store.get_attempt(outcome.trace.run_id).status is AttemptStatus.COMPLETED
        assert store.list_trace_events(outcome.trace.run_id) == outcome.trace.events
        assert outcome.trace.events[-1].kind is TraceEventKind.RUN_FINISHED
        assert outcome.trace.events[-1].payload["termination_status"] == "exhausted"
        # Even an explicit resume of the terminal checkpoint cannot repeat the
        # rejected planner call. It may emit an empty metadata-only attempt.
        resumed = runner.resume(ResumeRequest(thread_id=outcome.thread_id))
        assert resumed.termination_status == "exhausted"
        assert len(calls) == 5
        assert store.list_model_invocations(resumed.trace.run_id) == ()


def test_provider_value_error_is_not_swallowed_as_revision_rejection(tmp_path):
    models, _, _ = _models(provider_failure=True)
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        runner = AgentRunner(
            models,
            checkpointer=InMemorySaver(),
            raw_run_store=store,
            run_id_factory=lambda: "provider-failure",
        )
        with pytest.raises(ValueError, match="provider fault"):
            runner.run(
                RunRequest(
                    repo_path=tmp_path,
                    objective="Inspect source.",
                    max_tool_calls_per_step=1,
                    max_explorer_calls_per_step=1,
                    max_plan_revisions=1,
                )
            )
        assert store.get_attempt("provider-failure").status is AttemptStatus.FAILED
        assert (
            store.list_trace_events("provider-failure")[-1].kind
            is TraceEventKind.RUN_FAILED
        )
