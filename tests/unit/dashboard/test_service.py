"""Repository-bound dashboard query and redaction tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from repocompass.agent.evaluation import (
    EvaluationReport,
    SemanticAssessment,
    ValidationIssue,
)
from repocompass.agent.conversation import PublicConversationPage
from repocompass.agent.planning import InvestigationPlan, PlanStep
from repocompass.agent.runner import ResumeRequest
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.dashboard.contracts import DashboardAvailability
from repocompass.dashboard.service import (
    DashboardNotFoundError,
    DashboardQueryService,
)
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.service.contracts import InvestigationOperationKind
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    open_raw_run_store,
)
from repocompass.tools.gateway import ToolExecutionContext


class _FakeRunner:
    def __init__(self, views: dict[str, ThreadInvestigationView]) -> None:
        self._views = views
        self.inspected: list[str] = []

    def inspect_thread(self, request: ResumeRequest) -> ThreadInvestigationView:
        self.inspected.append(request.thread_id)
        view = self._views.get(request.thread_id)
        if view is None:
            raise ValueError(f"Checkpoint thread does not exist: {request.thread_id}")
        return view


def _view(thread_id: str = "thread-1") -> ThreadInvestigationView:
    return ThreadInvestigationView(
        thread_id=thread_id,
        execution_state=ThreadExecutionState.RESUMABLE,
        turn_number=1,
        objective="Locate the authentication flow.",
        plan_revision_count=0,
        max_plan_revisions=2,
        next_nodes=("plan_investigation",),
    )


def test_public_conversation_checks_repository_before_checkpoint_access(tmp_path):
    context = ToolExecutionContext(tmp_path)
    calls = []
    runner = _FakeRunner({"thread-1": _view()})

    def inspect(request, **kwargs):
        calls.append((request.thread_id, kwargs))
        return PublicConversationPage(
            thread_id=request.thread_id, latest_turn_number=1, turns=()
        )

    runner.inspect_conversation = inspect
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        store.record_attempt(_attempt(context))
        store.record_attempt(
            _attempt(
                context,
                run_id="foreign-run",
                thread_id="foreign-thread",
                repository_id="f" * 64,
            )
        )
        service = DashboardQueryService(
            runner,
            raw_run_store=store,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=tmp_path,
        )
        with pytest.raises(DashboardNotFoundError):
            service.get_conversation("foreign-thread")
        assert calls == []
        assert service.get_conversation("thread-1", before_turn=1, limit=1).turns == ()
        assert calls == [("thread-1", {"before_turn": 1, "limit": 1})]


def _attempt(
    context: ToolExecutionContext,
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    started_at_ms: int = 1_000,
    repository_id: str | None = None,
    execution_mode: AttemptExecutionMode = AttemptExecutionMode.START,
    status: AttemptStatus = AttemptStatus.COMPLETED,
) -> RawRunAttempt:
    return RawRunAttempt(
        run_id=run_id,
        thread_id=thread_id,
        turn_number=1,
        execution_mode=execution_mode,
        repository_id=repository_id or context.repository_id,
        commit_sha=context.commit_sha,
        objective_digest=hashlib.sha256(b"objective").hexdigest(),
        status=status,
        started_at_ms=started_at_ms,
        finished_at_ms=(
            None if status is AttemptStatus.RUNNING else started_at_ms + 100
        ),
    )


def _plan() -> InvestigationPlan:
    return InvestigationPlan(
        steps=[
            PlanStep(
                id="inspect-auth",
                objective="Inspect authentication entry points.",
                expected_evidence=["A repository-relative source location."],
                status="active",
            )
        ]
    )


def _evaluation() -> EvaluationReport:
    return EvaluationReport(
        deterministic_issues=[],
        semantic_assessment=SemanticAssessment(
            active_step_completed=True,
            answers_objective=True,
            evidence_supports_summary=True,
            has_critical_omissions=False,
            missing_evidence=[],
            recommended_focus=[],
            summary="The evidence is sufficient.",
        ),
    )


def _write_trace(
    trace_directory: Path, repository_root: Path, *, structural_warning: bool = False
) -> None:
    trace = RunTrace("run-1")
    trace.record(
        TraceEventKind.RUN_STARTED,
        payload={
            "objective": "Locate the authentication flow.",
            "repo_path": str(repository_root),
            "nested": {
                "api-key": "must-not-leak",
                "path": f"{repository_root}/src/auth.py",
            },
            "messages": ["must-not-leak"],
        },
    )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="plan_investigation",
        payload={"investigation_plan": _plan().model_dump(mode="json")},
    )
    evaluation = _evaluation()
    if structural_warning:
        evaluation = evaluation.model_copy(
            update={
                "structural_warnings": [
                    ValidationIssue(
                        code="structure_record_limit_exceeded",
                        message="Optional AST record cap reached.",
                        path="src/auth.py",
                        evidence_index=0,
                    )
                ]
            }
        )
    trace.record(
        TraceEventKind.NODE_COMPLETED,
        node="evaluate_candidate",
        payload={"evaluation_report": evaluation.model_dump(mode="json")},
    )
    JsonlTraceSink().write(trace, trace_directory / "run-1.jsonl")


def test_dashboard_catalog_is_repository_bound_and_has_explicit_empty_states(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    runner = _FakeRunner({"thread-1": _view()})
    trace_directory = tmp_path / "traces"

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        own_attempt = _attempt(context)
        other_attempt = _attempt(
            context,
            run_id="run-other",
            thread_id="thread-other",
            started_at_ms=2_000,
            repository_id="f" * 64,
        )
        store.record_attempt(own_attempt)
        store.record_attempt(other_attempt)
        service = DashboardQueryService(
            runner,
            raw_run_store=store,
            trace_store=JsonlTraceStore(trace_directory),
            repository_root=tmp_path,
        )

        catalog = service.list_threads()
        detail = service.get_run("run-1")
        memory = service.list_memory()
        with pytest.raises(DashboardNotFoundError, match="run"):
            service.get_run("run-other")

    assert [thread.thread_id for thread in catalog.threads] == ["thread-1"]
    assert catalog.threads[0].checkpoint_availability is (
        DashboardAvailability.AVAILABLE
    )
    assert detail.trace_availability is DashboardAvailability.NOT_PERSISTED
    assert detail.trace_events == ()
    assert detail.cost_availability is DashboardAvailability.NOT_CONFIGURED
    assert memory.availability is DashboardAvailability.NOT_CONFIGURED


@pytest.mark.parametrize("structural_warning", [False, True])
def test_dashboard_run_redacts_roots_and_replays_plan_and_evaluator(
    tmp_path: Path,
    structural_warning: bool,
) -> None:
    context = ToolExecutionContext(tmp_path)
    trace_directory = tmp_path / "traces"
    _write_trace(
        trace_directory, tmp_path.resolve(), structural_warning=structural_warning
    )
    runner = _FakeRunner({"thread-1": _view()})

    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        store.record_attempt(_attempt(context))
        service = DashboardQueryService(
            runner,
            raw_run_store=store,
            trace_store=JsonlTraceStore(trace_directory),
            repository_root=tmp_path,
        )
        run = service.get_run("run-1")
        thread = service.get_thread("thread-1")

    serialized = run.model_dump_json()
    assert str(tmp_path.resolve()) not in serialized
    assert "must-not-leak" not in serialized
    assert "repo_path" not in serialized
    assert run.trace_events[0].payload["nested"] == {"path": "<repository>/src/auth.py"}
    assert run.plan_timeline[0].plan == _plan()
    assert run.evaluator_timeline[0].report.passed is True
    assert (
        bool(run.evaluator_timeline[0].report.structural_warnings) is structural_warning
    )
    assert thread.plan_timeline == run.plan_timeline
    assert thread.evaluator_timeline == run.evaluator_timeline


def test_dashboard_catalog_reports_missing_checkpoint_without_failing(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    runner = _FakeRunner({})
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        store.record_attempt(_attempt(context))
        service = DashboardQueryService(
            runner,
            raw_run_store=store,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=tmp_path,
        )
        catalog = service.list_threads()

    assert catalog.threads[0].checkpoint_availability is (
        DashboardAvailability.NOT_FOUND
    )
    assert catalog.threads[0].objective is None


def test_dashboard_finds_only_attempts_inside_an_operation_window(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(tmp_path)
    runner = _FakeRunner({"thread-1": _view()})
    with open_raw_run_store(tmp_path / "raw.sqlite") as store:
        store.record_attempt(_attempt(context, run_id="run-old", started_at_ms=900))
        current = _attempt(
            context,
            run_id="run-current",
            started_at_ms=1_100,
            execution_mode=AttemptExecutionMode.RESUME,
            status=AttemptStatus.RUNNING,
        )
        store.record_attempt(current)
        service = DashboardQueryService(
            runner,
            raw_run_store=store,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=tmp_path,
        )

        pending = service.find_lifecycle_attempt(
            thread_id="thread-1",
            kind=InvestigationOperationKind.RESUME,
            reserved_at_ms=1_000,
            run_id=None,
        )
        exact = service.find_lifecycle_attempt(
            thread_id="thread-1",
            kind=InvestigationOperationKind.RESUME,
            reserved_at_ms=1_000,
            run_id="run-current",
        )
        stale = service.find_lifecycle_attempt(
            thread_id="thread-1",
            kind=InvestigationOperationKind.START,
            reserved_at_ms=1_000,
            run_id="run-old",
        )

    assert pending == current
    assert exact == current
    assert stale is None
