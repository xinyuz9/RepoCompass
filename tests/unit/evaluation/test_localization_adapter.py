"""Production RunOutcome adapter and five-variant runner tests."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.cost import ModelPrice, build_attempt_usage_cost_report
from repocompass.agent.evaluation import (
    EvaluationReport,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import AgentRunner, RunOutcome, RunRequest
from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.agent.usage import (
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    UsageRecord,
    create_model_invocation_record,
)
from repocompass.evaluation import (
    LocalizationBudgetProfile,
    LocalizationDatasetKind,
    LocalizationLabelSource,
    LocalizationMetricAvailability,
    LocalizationMetricUnavailableReason,
    LocalizationRunStatus,
    LocalizationVariant,
    create_localization_budget_profile,
    create_localization_case,
    create_localization_dataset_manifest,
    create_localization_labels,
    create_localization_task,
)
from repocompass.evaluation.localization_adapter import (
    AgentRunnerVariantExecutor,
    LocalizationBenchmarkRunner,
    LocalizationVariantExecutor,
    adapt_localization_run_outcome,
    create_localization_run_request,
    localization_architecture_profile,
)
from repocompass.tools.contracts import (
    ToolAuditRecord,
    ToolCallFingerprint,
    ToolDecisionOutcome,
    ToolDecisionReason,
    ToolPolicyDecision,
)
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from tests.model_doubles import candidate_from_final_result, with_scripted_planner


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    repository = tmp_path / "checkout"
    repository.mkdir()
    commands = (
        ("git", "init", "--quiet"),
        ("git", "config", "user.email", "tests@example.com"),
        ("git", "config", "user.name", "RepoCompass Tests"),
    )
    for command in commands:
        subprocess.run(
            command,
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ("git", "add", "module.py"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ("git", "commit", "--quiet", "-m", "fixture"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return repository


def _budget() -> LocalizationBudgetProfile:
    return create_localization_budget_profile(
        max_tool_calls_per_step=8,
        max_explorer_calls_per_step=3,
        max_no_progress_rounds=2,
        max_plan_revisions=2,
        context_budget_tokens=6_000,
        max_attempt_cost_usd="0.5",
        max_repository_memory_results=5,
        max_episodic_memory_results=3,
    )


def _manifest(checkout: Path):
    commit_sha = ToolExecutionContext(checkout).commit_sha
    assert commit_sha is not None
    task = create_localization_task(
        dataset_kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        instance_id="production-adapter-1",
        repository="example/repocompass",
        base_commit=commit_sha,
        objective="Locate the routing implementation without seeing labels.",
    )
    labels = create_localization_labels(
        task_id=task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="a" * 64,
        gold_files=("src/repocompass/agent/routing.py",),
        gold_symbols=(),
    )
    case = create_localization_case(task=task, labels=labels)
    return create_localization_dataset_manifest(
        name="production-adapter-fixture",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(case,),
    )


def _usage_report(run_id: str, thread_id: str):
    usage = UsageRecord(
        availability=UsageAvailability.REPORTED,
        input_tokens=1_000,
        output_tokens=100,
        total_tokens=1_100,
        cached_input_tokens=None,
        reasoning_output_tokens=None,
    )
    invocation = create_model_invocation_record(
        run_id=run_id,
        thread_id=thread_id,
        turn_number=1,
        sequence=0,
        node_name="agent",
        role=ModelRole.EXPLORER,
        provider="openai",
        model="gpt-test",
        status=ModelInvocationStatus.SUCCEEDED,
        latency_ms=17,
        usage=usage,
        input_artifact_id="b" * 64,
        output_artifact_id="c" * 64,
        context_pack_id="d" * 64,
        error_code=None,
    )
    from repocompass.agent.cost import create_pricing_table

    table = create_pricing_table(
        version="adapter-test-v1",
        effective_from="2026-08-28",
        entries=(
            ModelPrice(
                provider="openai",
                model="gpt-test",
                input_usd_per_million_tokens="2.5",
                output_usd_per_million_tokens="10",
            ),
        ),
    )
    report = build_attempt_usage_cost_report(
        run_id=run_id,
        thread_id=thread_id,
        turn_number=1,
        invocations=(invocation,),
        pricing_table=table,
        max_cost_usd="0.5",
    )
    return table.table_id, report


def _audit(run_id: str, sequence: int) -> ToolAuditRecord:
    return ToolAuditRecord(
        sequence=sequence,
        run_id=run_id,
        call_id=f"call-{sequence}",
        tool_name="search_text",
        repository_id="e" * 64,
        commit_sha=None,
        fingerprint=ToolCallFingerprint(digest=f"{sequence + 1:064x}"),
        decision=ToolPolicyDecision(
            outcome=ToolDecisionOutcome.ALLOWED,
            reason=ToolDecisionReason.ALLOWED,
        ),
        latency_ms=3,
        output_bytes=20,
        result_digest="f" * 64,
    )


def _outcome(run_id: str) -> tuple[str, RunOutcome]:
    thread_id = f"thread-{run_id}"
    pricing_table_id, usage_report = _usage_report(run_id, thread_id)
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="src/repocompass/agent/routing.py",
                line_start=1,
                line_end=4,
                symbol="route_controller",
                claim="The controller owns the route decision.",
            ),
            EvidenceItem(
                path="missing.py",
                line_start=8,
                line_end=9,
                claim="This location is invalid.",
            ),
        ],
        summary="Routing is decided by deterministic controller code.",
        unresolved_questions=[],
    )
    assessment = SemanticAssessment(
        active_step_completed=True,
        answers_objective=True,
        evidence_supports_summary=True,
        has_critical_omissions=False,
        missing_evidence=[],
        recommended_focus=[],
        summary="Enough evidence.",
    )
    evaluation = EvaluationReport(
        deterministic_issues=[
            ValidationIssue(
                code="path_not_found",
                message="Candidate path does not exist.",
                path="missing.py",
                evidence_index=1,
            )
        ],
        semantic_assessment=assessment,
    )
    trace = RunTrace(run_id)
    trace.record(TraceEventKind.RUN_STARTED)
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="agent",
        payload={
            "active_context_pack": {
                "estimated_tokens": 120,
                "items": [
                    {
                        "location": {
                            "path": "src/repocompass/agent/routing.py",
                            "line_start": 2,
                            "line_end": 5,
                        }
                    },
                    {
                        "location": {
                            "path": "src/repocompass/agent/routing.py",
                            "line_start": 4,
                            "line_end": 7,
                        }
                    },
                    {"location": None},
                ],
            }
        },
    )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="build_candidate",
        payload={
            "candidate_result": candidate.model_dump(mode="json"),
            "active_context_pack": {
                "estimated_tokens": 80,
                "items": [
                    {
                        "location": {
                            "path": "src/repocompass/agent/result.py",
                            "line_start": 10,
                            "line_end": 12,
                        }
                    }
                ],
            },
        },
    )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="evaluate_candidate",
        payload={
            "evaluation_report": evaluation.model_dump(
                mode="json", exclude_computed_fields=True
            )
        },
    )
    trace.record(TraceEventKind.RUN_FINISHED)
    return pricing_table_id, RunOutcome(
        thread_id=thread_id,
        final_result=FinalResult(
            relevant_files=[
                "src/repocompass/agent/routing.py",
                "src\\repocompass\\agent\\routing.py",
                "../escape.py",
                "src/repocompass/agent/result.py",
            ],
            evidence=["first", "second"],
            summary="Routing is decided by deterministic controller code.",
            unresolved_questions=[],
        ),
        termination_status="accepted",
        step_stop_reason=None,
        trace=trace,
        tool_audit_records=(_audit(run_id, 0), _audit(run_id, 1)),
        usage_cost_report=usage_report,
    )


def test_completed_outcome_maps_ranked_output_quality_and_resource_metrics(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    case = manifest.cases[0]
    pricing_table_id, outcome = _outcome("adapter-run")

    observation = adapt_localization_run_outcome(
        dataset_manifest_id=manifest.manifest_id,
        case=case,
        variant=LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
        budget_profile=_budget(),
        outcome=outcome,
        repeat_index=0,
        latency_ms=41,
        pricing_table_id=pricing_table_id,
    )

    assert observation.status is LocalizationRunStatus.ACCEPTED
    assert observation.prediction.ranked_files == (
        "src/repocompass/agent/routing.py",
        "src/repocompass/agent/result.py",
    )
    assert observation.prediction.evidence_count == 2
    assert observation.prediction.valid_evidence_count == 1
    assert observation.prediction.relation_count == 0
    assert observation.prediction.unsupported_relation_count == 0
    assert observation.context_lines.value == 9
    assert observation.context_tokens.value == 200
    assert observation.tool_calls.value == 2
    assert observation.latency_ms.value == 41
    assert observation.evaluator_iterations.value == 1
    assert observation.input_tokens.value == 1_000
    assert observation.output_tokens.value == 100
    assert observation.cost.estimated_cost_usd == "0.0035"
    assert observation.cost.usage_cost_report_id == outcome.usage_cost_report.report_id


def test_run_request_is_label_free_and_enables_memory_only_for_memory_variant(
    checkout: Path,
) -> None:
    task = _manifest(checkout).cases[0].task
    budget = _budget()

    request = create_localization_run_request(
        task=task,
        repository_path=checkout,
        budget_profile=budget,
        variant=LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY,
    )
    control = create_localization_run_request(
        task=task,
        repository_path=checkout,
        budget_profile=budget,
        variant=LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER,
    )

    assert isinstance(request, RunRequest)
    assert request.objective == task.objective
    assert request.memory_retrieval_enabled is True
    assert control.memory_retrieval_enabled is False
    assert request.max_tool_calls_per_step == budget.max_tool_calls_per_step
    assert "labels" not in request.model_dump(mode="json")
    assert "gold" not in request.model_dump_json()


class _Executor(LocalizationVariantExecutor):
    def __init__(
        self,
        variant: LocalizationVariant,
        *,
        error: Exception | None = None,
    ) -> None:
        self.variant = variant
        self.architecture_profile = localization_architecture_profile(variant)
        self.error = error
        self.requests: list[RunRequest] = []

    def run(self, request: RunRequest) -> RunOutcome:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return _outcome(f"run-{self.variant.value}")[1]


def _executors() -> dict[LocalizationVariant, _Executor]:
    return {variant: _Executor(variant) for variant in LocalizationVariant}


def _workflow_models(final_result: FinalResult) -> WorkflowModels:
    explorer = MagicMock(spec=BaseChatModel)
    explorer.bind_tools.return_value = RunnableLambda(
        lambda _: AIMessage(content="Enough evidence is available.")
    )
    explorer.with_structured_output.return_value = RunnableLambda(
        lambda _: candidate_from_final_result(final_result)
    )
    return with_scripted_planner(explorer)


def test_five_variant_runner_preflights_identity_and_uses_one_budget(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    executors = _executors()
    pricing_table_id = _outcome("pricing-probe")[0]
    runner = LocalizationBenchmarkRunner(
        manifest=manifest,
        budget_profile=_budget(),
        pricing_table_id=pricing_table_id,
        executors=executors,
    )

    observation_set = runner.run(
        repository_paths={manifest.cases[0].task.task_id: checkout},
        repeat_index=2,
    )

    assert len(observation_set.observations) == 5
    assert tuple(item.variant for item in observation_set.observations) == tuple(
        LocalizationVariant
    )
    assert {item.budget_profile_id for item in observation_set.observations} == {
        _budget().profile_id
    }
    assert all(len(executor.requests) == 1 for executor in executors.values())
    assert (
        executors[LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY]
        .requests[0]
        .memory_retrieval_enabled
    )
    assert all(
        not executor.requests[0].memory_retrieval_enabled
        for variant, executor in executors.items()
        if variant is not LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY
    )


def test_real_agent_runners_execute_complete_five_variant_matrix(
    checkout: Path,
    tmp_path: Path,
) -> None:
    manifest = _manifest(checkout)
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=3,
        max_explorer_calls_per_step=2,
        max_no_progress_rounds=2,
        max_plan_revisions=1,
        context_budget_tokens=6_000,
        max_attempt_cost_usd=None,
        max_repository_memory_results=2,
        max_episodic_memory_results=2,
    )
    final_result = FinalResult(
        relevant_files=["module.py"],
        evidence=["module.py contains the requested implementation."],
        summary="The implementation is in module.py.",
        unresolved_questions=[],
    )
    executors: dict[LocalizationVariant, AgentRunnerVariantExecutor] = {}
    with open_memory_store(tmp_path / "memory.sqlite3") as memory_store:
        for variant in LocalizationVariant:
            runner = AgentRunner(
                models=_workflow_models(final_result),
                checkpointer=InMemorySaver(),
                architecture_profile=localization_architecture_profile(variant),
                memory_store=(
                    memory_store
                    if variant
                    is LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY
                    else None
                ),
                run_id_factory=lambda value=variant.value: f"matrix-{value}",
                thread_id_factory=lambda value=variant.value: f"thread-{value}",
            )
            executors[variant] = AgentRunnerVariantExecutor(
                variant=variant,
                runner=runner,
            )
        observation_set = LocalizationBenchmarkRunner(
            manifest=manifest,
            budget_profile=budget,
            pricing_table_id="9" * 64,
            executors=executors,
        ).run(repository_paths={manifest.cases[0].task.task_id: checkout})

    assert len(observation_set.observations) == 5
    assert [item.status for item in observation_set.observations] == [
        LocalizationRunStatus.ACCEPTED
    ] * 5
    assert observation_set.observations[0].evaluator_iterations.value == 0
    assert all(
        item.evaluator_iterations.value == 1
        for item in observation_set.observations[1:]
    )
    assert observation_set.observations[0].prediction.quality_unavailable_reason is (
        LocalizationMetricUnavailableReason.NOT_COLLECTED
    )
    assert all(
        item.prediction.quality_availability is LocalizationMetricAvailability.OBSERVED
        for item in observation_set.observations[1:]
    )


def test_runner_records_failed_and_timed_out_runs_without_fabricated_metrics(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    executors = _executors()
    executors[LocalizationVariant.TOOL_WRAPPER] = _Executor(
        LocalizationVariant.TOOL_WRAPPER,
        error=RuntimeError("secret provider body"),
    )
    executors[LocalizationVariant.EVALUATOR_LOOP] = _Executor(
        LocalizationVariant.EVALUATOR_LOOP,
        error=TimeoutError("secret timeout body"),
    )
    pricing_table_id = _outcome("pricing-probe-2")[0]
    failure_ids = iter(("failed-run", "timed-out-run"))
    runner = LocalizationBenchmarkRunner(
        manifest=manifest,
        budget_profile=_budget(),
        pricing_table_id=pricing_table_id,
        executors=executors,
        failed_run_id_factory=lambda: next(failure_ids),
    )

    observation_set = runner.run(
        repository_paths={manifest.cases[0].task.task_id: checkout}
    )
    failed, timed_out = observation_set.observations[:2]

    assert failed.status is LocalizationRunStatus.FAILED
    assert timed_out.status is LocalizationRunStatus.TIMED_OUT
    for item in (failed, timed_out):
        assert item.prediction.quality_unavailable_reason is (
            LocalizationMetricUnavailableReason.RUN_FAILED
        )
        assert item.context_tokens.availability is (
            LocalizationMetricAvailability.UNAVAILABLE
        )
        assert item.latency_ms.availability is LocalizationMetricAvailability.OBSERVED
        assert item.cost.estimated_cost_usd is None
        assert "secret" not in item.model_dump_json()


def test_runner_rejects_missing_mislabeled_or_reused_variant_executors(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    pricing_table_id = _outcome("pricing-probe-3")[0]
    missing = _executors()
    missing.pop(LocalizationVariant.TOOL_WRAPPER)
    with pytest.raises(ValueError, match="exactly five"):
        LocalizationBenchmarkRunner(
            manifest=manifest,
            budget_profile=_budget(),
            pricing_table_id=pricing_table_id,
            executors=missing,
        )

    mislabeled = _executors()
    mislabeled[
        LocalizationVariant.TOOL_WRAPPER
    ].variant = LocalizationVariant.EVALUATOR_LOOP
    with pytest.raises(ValueError, match="label"):
        LocalizationBenchmarkRunner(
            manifest=manifest,
            budget_profile=_budget(),
            pricing_table_id=pricing_table_id,
            executors=mislabeled,
        )

    wrong_profile = _executors()
    wrong_profile[
        LocalizationVariant.TOOL_WRAPPER
    ].architecture_profile = localization_architecture_profile(
        LocalizationVariant.EVALUATOR_LOOP
    )
    with pytest.raises(ValueError, match="architecture profile"):
        LocalizationBenchmarkRunner(
            manifest=manifest,
            budget_profile=_budget(),
            pricing_table_id=pricing_table_id,
            executors=wrong_profile,
        )

    shared = _Executor(LocalizationVariant.TOOL_WRAPPER)
    reused: dict[LocalizationVariant, LocalizationVariantExecutor] = {
        variant: shared for variant in LocalizationVariant
    }
    with pytest.raises(ValueError, match="distinct"):
        LocalizationBenchmarkRunner(
            manifest=manifest,
            budget_profile=_budget(),
            pricing_table_id=pricing_table_id,
            executors=reused,
        )


def test_runner_rejects_checkout_mismatch_before_any_model_call(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    original = manifest.cases[0]
    mismatched_task = create_localization_task(
        dataset_kind=original.task.dataset_kind,
        instance_id="production-adapter-mismatch",
        repository=original.task.repository,
        base_commit="f" * 40,
        objective=original.task.objective,
    )
    labels = create_localization_labels(
        task_id=mismatched_task.task_id,
        source=LocalizationLabelSource.SYNTHETIC_FIXTURE,
        source_artifact_digest="a" * 64,
        gold_files=original.labels.gold_files,
        gold_symbols=(),
    )
    mismatch_manifest = create_localization_dataset_manifest(
        name="checkout-mismatch",
        version="v1",
        kind=LocalizationDatasetKind.SYNTHETIC_FIXTURE,
        cases=(create_localization_case(task=mismatched_task, labels=labels),),
    )
    executors = _executors()
    runner = LocalizationBenchmarkRunner(
        manifest=mismatch_manifest,
        budget_profile=_budget(),
        pricing_table_id=_outcome("pricing-probe-4")[0],
        executors=executors,
    )

    with pytest.raises(ValueError, match="base commit"):
        runner.run(repository_paths={mismatched_task.task_id: checkout})
    assert all(not executor.requests for executor in executors.values())


def test_runner_rejects_dirty_checkout_before_any_model_call(
    checkout: Path,
) -> None:
    manifest = _manifest(checkout)
    executors = _executors()
    (checkout / "untracked.txt").write_text("mutation\n", encoding="utf-8")
    runner = LocalizationBenchmarkRunner(
        manifest=manifest,
        budget_profile=_budget(),
        pricing_table_id=_outcome("pricing-probe-5")[0],
        executors=executors,
    )

    with pytest.raises(ValueError, match="clean immutable"):
        runner.run(repository_paths={manifest.cases[0].task.task_id: checkout})
    assert all(not executor.requests for executor in executors.values())
