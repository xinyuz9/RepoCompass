"""Explicit dashboard mutation-boundary tests."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.context import RepositoryLocation
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanDecisionAction,
    HumanReviewRequest,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.dashboard.commands import (
    DashboardCommandConflictError,
    DashboardCommandService,
    DashboardCommandUnavailableError,
)
from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardContinueCommand,
    DashboardResumeCommand,
    DashboardRunSummary,
    DashboardStartCommand,
    DashboardHumanResponseCommand,
    DashboardMemoryAudit,
    DashboardMemoryDeleteCommand,
    DashboardThreadDetail,
)
from repocompass.memory.contracts import (
    MemoryFactKind,
    MemoryVerificationStatus,
    MemoryWriteAuthority,
    RepositoryMemory,
    create_repository_memory,
    create_repository_memory_source,
)
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import MemoryStore, open_memory_store
from repocompass.service.contracts import (
    ContinueInvestigationRequest,
    InvestigationOperationResult,
    ResumeInvestigationRequest,
    StartInvestigationRequest,
)
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
)
from repocompass.tools.gateway import ToolExecutionContext


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_repository(root: Path) -> tuple[Path, ToolExecutionContext]:
    repository = root / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    (repository / "src").mkdir()
    (repository / "src" / "api.py").write_text(
        "def authenticate(request):\n    return request.user is not None\n",
        encoding="utf-8",
    )
    _git(repository, "add", "src/api.py")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository, ToolExecutionContext(repository)


def _memory(context: ToolExecutionContext) -> RepositoryMemory:
    identity = resolve_memory_repository_identity(context)
    commit_sha = context.commit_sha
    assert commit_sha is not None
    source_path = context.repository_root / "src" / "api.py"
    source = create_repository_memory_source(
        location=RepositoryLocation(
            repository_id=context.repository_id,
            commit_sha=commit_sha,
            path="src/api.py",
            line_start=1,
            line_end=2,
            symbol="authenticate",
        ),
        source_digest=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        evidence_claim_digest=hashlib.sha256(b"auth entry point").hexdigest(),
    )
    return create_repository_memory(
        identity=identity,
        commit_sha=commit_sha,
        kind=MemoryFactKind.ENTRY_POINT,
        title="Authentication entry point",
        content="src.api.authenticate is the authentication entry point.",
        verification_status=MemoryVerificationStatus.VERIFIED,
        write_authority=MemoryWriteAuthority.VALIDATED_EVIDENCE,
        sources=(source,),
        tags=("authentication", "entry-point"),
        created_at_ms=1_000,
        verified_at_ms=1_000,
    )


def _candidate() -> CandidateResult:
    return CandidateResult(
        evidence=[
            EvidenceItem(
                path="src/api.py",
                line_start=1,
                line_end=2,
                symbol="authenticate",
                claim="authenticate owns the authentication decision.",
            )
        ],
        summary="Authentication is implemented by authenticate.",
        unresolved_questions=[],
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
            summary="The evidence is ready for Human review.",
        ),
    )


def _thread_detail(context: ToolExecutionContext) -> DashboardThreadDetail:
    candidate = _candidate()
    evaluation = _evaluation()
    review = HumanReviewRequest(
        turn_number=1,
        candidate_result=candidate,
        evaluation_report=evaluation,
        allowed_actions=(
            HumanDecisionAction.APPROVE,
            HumanDecisionAction.CANCEL,
        ),
    )
    return DashboardThreadDetail(
        thread=ThreadInvestigationView(
            thread_id="thread-1",
            execution_state=ThreadExecutionState.INTERRUPTED,
            turn_number=1,
            objective="Locate authentication.",
            plan=None,
            plan_revision_count=2,
            max_plan_revisions=2,
            next_nodes=("human_review",),
            termination_status=None,
            step_stop_reason=None,
            candidate_result=candidate,
            evaluation_report=evaluation,
            structure=None,
            context_pack=None,
            memory_recall_report=None,
            cost_budget=None,
            final_result=None,
            pending_review=review,
            interrupt_id="interrupt-1",
        ),
        attempts=(
            DashboardRunSummary(
                attempt=RawRunAttempt(
                    run_id="run-1",
                    thread_id="thread-1",
                    turn_number=1,
                    execution_mode=AttemptExecutionMode.START,
                    repository_id=context.repository_id,
                    commit_sha=context.commit_sha,
                    objective_digest="c" * 64,
                    status=AttemptStatus.INTERRUPTED,
                    started_at_ms=1_000,
                    finished_at_ms=2_000,
                ),
                trace_availability=DashboardAvailability.NOT_PERSISTED,
                cost_availability=DashboardAvailability.NOT_CONFIGURED,
            ),
        ),
        plan_timeline=(),
        evaluator_timeline=(),
    )


def _operation_result(
    request_id: str, thread_id: str = "thread-1"
) -> InvestigationOperationResult:
    return InvestigationOperationResult(
        request_id=request_id,
        thread_id=thread_id,
        execution_state=ThreadExecutionState.COMPLETED,
        run_id="response-run",
        termination_status="accepted",
        step_stop_reason=None,
        next_nodes=(),
        final_result=FinalResult(
            relevant_files=["src/api.py"],
            evidence=["src/api.py:1-2 owns authentication."],
            summary="Authentication is implemented by authenticate.",
            unresolved_questions=[],
        ),
        pending_review=None,
        interrupt_id=None,
        usage_cost_report=None,
        idempotent_replay=False,
        recovered_from_checkpoint=False,
    )


class _Query:
    def __init__(
        self,
        context: ToolExecutionContext,
        *,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.repository_context = context
        self.memory_store = memory_store
        self.thread_detail = _thread_detail(context)

    def get_thread(self, thread_id: str) -> DashboardThreadDetail:
        assert thread_id == "thread-1"
        return self.thread_detail

    def get_memory_audit(self, memory_id: str) -> DashboardMemoryAudit:
        assert self.memory_store is not None
        metadata = self.memory_store.get_memory_metadata(
            memory_id,
            context=self.repository_context,
        )
        tombstone = self.memory_store.get_memory_tombstone(
            memory_id,
            context=self.repository_context,
        )
        return DashboardMemoryAudit(
            memory_id=memory_id,
            metadata=metadata,
            tombstone=tombstone,
            events=self.memory_store.list_memory_audit_events(memory_id),
        )


class _Lifecycle:
    def __init__(self) -> None:
        self.replay_result: InvestigationOperationResult | None = None
        self.continue_replay_result: InvestigationOperationResult | None = None
        self.start_requests: list[StartInvestigationRequest] = []
        self.replay_requests: list[ResumeInvestigationRequest] = []
        self.response_requests: list[ResumeInvestigationRequest] = []
        self.continue_requests: list[ContinueInvestigationRequest] = []

    def start_investigation(
        self, request: StartInvestigationRequest
    ) -> InvestigationOperationResult:
        self.start_requests.append(request)
        return _operation_result(request.request_id, "thread-started")

    def replay_resume_investigation(
        self,
        request: ResumeInvestigationRequest,
    ) -> InvestigationOperationResult | None:
        self.replay_requests.append(request)
        return self.replay_result

    def resume_investigation(
        self,
        request: ResumeInvestigationRequest,
    ) -> InvestigationOperationResult:
        self.response_requests.append(request)
        return _operation_result(request.request_id)

    def replay_continue_investigation(
        self, request: ContinueInvestigationRequest
    ) -> InvestigationOperationResult | None:
        return self.continue_replay_result

    def continue_investigation(
        self, request: ContinueInvestigationRequest
    ) -> InvestigationOperationResult:
        self.continue_requests.append(request)
        return _operation_result(request.request_id, request.thread_id)


def test_human_response_is_bound_to_interrupt_action_and_idempotent_replay(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    lifecycle = _Lifecycle()
    service = DashboardCommandService(
        _Query(context),
        lifecycle_service=lifecycle,
    )
    command = DashboardHumanResponseCommand(
        request_id="review-response-1",
        interrupt_id="interrupt-1",
        decision=HumanDecision(
            action=HumanDecisionAction.APPROVE,
            feedback=None,
        ),
    )

    result = service.respond_to_review("thread-1", command)

    assert result.request_id == command.request_id
    assert len(lifecycle.replay_requests) == 1
    assert len(lifecycle.response_requests) == 1
    assert lifecycle.response_requests[0].interrupt_id == "interrupt-1"

    lifecycle.replay_result = result.model_copy(update={"idempotent_replay": True})
    replay = service.respond_to_review("thread-1", command)
    assert replay.idempotent_replay is True
    assert len(lifecycle.response_requests) == 1

    stale = command.model_copy(update={"interrupt_id": "interrupt-stale"})
    lifecycle.replay_result = None
    with pytest.raises(DashboardCommandConflictError, match="interrupt_id"):
        service.respond_to_review("thread-1", stale)

    disallowed = command.model_copy(
        update={
            "request_id": "review-response-2",
            "decision": HumanDecision(
                action=HumanDecisionAction.REVISE,
                feedback="Inspect the call sites.",
            ),
        }
    )
    with pytest.raises(DashboardCommandConflictError, match="not allowed"):
        service.respond_to_review("thread-1", disallowed)
    assert len(lifecycle.response_requests) == 1


def test_goal_resume_and_follow_up_are_bound_to_repository_snapshot(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    query = _Query(context)
    lifecycle = _Lifecycle()
    service = DashboardCommandService(query, lifecycle_service=lifecycle)

    started = service.start_investigation(
        DashboardStartCommand(
            request_id="dashboard-start-1",
            goal="Trace authentication.",
            focus_path="src/api.py",
            memory_retrieval_enabled=True,
        )
    )
    assert started.thread_id == "thread-started"
    start_request = lifecycle.start_requests[0]
    assert start_request.repo_path == context.repository_root
    assert start_request.objective.endswith("Focus this investigation on `src/api.py`.")
    assert start_request.memory_retrieval_enabled is True

    query.thread_detail = query.thread_detail.model_copy(
        update={
            "thread": query.thread_detail.thread.model_copy(
                update={
                    "execution_state": ThreadExecutionState.RESUMABLE,
                    "next_nodes": ("agent",),
                    "pending_review": None,
                    "interrupt_id": None,
                }
            )
        }
    )
    service.resume_thread(
        "thread-1",
        DashboardResumeCommand(
            request_id="dashboard-resume-1",
            expected_repository_id=context.repository_id,
            expected_commit_sha=context.commit_sha,
        ),
    )
    assert lifecycle.response_requests[-1].decision is None

    query.thread_detail = query.thread_detail.model_copy(
        update={
            "thread": query.thread_detail.thread.model_copy(
                update={
                    "execution_state": ThreadExecutionState.COMPLETED,
                    "next_nodes": (),
                }
            )
        }
    )
    service.continue_thread(
        "thread-1",
        DashboardContinueCommand(
            request_id="dashboard-continue-1",
            prompt="Now trace rejection.",
            focus_path="src",
            expected_repository_id=context.repository_id,
            expected_commit_sha=context.commit_sha,
        ),
    )
    assert lifecycle.continue_requests[0].prompt.endswith(
        "Focus this investigation on `src`."
    )

    with pytest.raises(DashboardCommandConflictError, match="stale"):
        service.continue_thread(
            "thread-1",
            DashboardContinueCommand(
                request_id="dashboard-continue-stale",
                prompt="Use stale code.",
                expected_repository_id="f" * 64,
                expected_commit_sha=context.commit_sha,
            ),
        )
    with pytest.raises(DashboardCommandConflictError, match="escapes"):
        service.start_investigation(
            DashboardStartCommand(
                request_id="dashboard-start-escape",
                goal="Escape.",
                focus_path="../outside",
            )
        )


def test_disabled_lifecycle_command_fails_closed(tmp_path: Path) -> None:
    _, context = _create_repository(tmp_path)
    service = DashboardCommandService(_Query(context))
    command = DashboardHumanResponseCommand(
        request_id="review-response-disabled",
        interrupt_id="interrupt-1",
        decision=HumanDecision(
            action=HumanDecisionAction.CANCEL,
            feedback=None,
        ),
    )

    with pytest.raises(DashboardCommandUnavailableError, match="not enabled"):
        service.respond_to_review("thread-1", command)


def test_memory_delete_requires_full_identity_and_replays_tombstone(
    tmp_path: Path,
) -> None:
    _, context = _create_repository(tmp_path)
    memory = _memory(context)
    with open_memory_store(tmp_path / "memory.sqlite") as store:
        store.put_repository_memory(memory, context=context)
        query = _Query(context, memory_store=store)
        service = DashboardCommandService(
            query, memory_store=store, clock_ms=lambda: 2_000
        )

        mismatched = DashboardMemoryDeleteCommand(
            request_id=f"memory-delete:{'f' * 64}",
            confirm_memory_id="f" * 64,
        )
        with pytest.raises(DashboardCommandConflictError, match="confirmation"):
            service.delete_memory(memory.memory_id, mismatched)
        assert store.get_repository_memory(memory.memory_id) == memory

        command = DashboardMemoryDeleteCommand(
            request_id=f"memory-delete:{memory.memory_id}",
            confirm_memory_id=memory.memory_id,
        )
        deleted = service.delete_memory(memory.memory_id, command)
        replay = service.delete_memory(memory.memory_id, command)
        events = store.list_memory_audit_events(memory.memory_id)

    assert deleted.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert replay.tombstone == deleted.tombstone
    assert len(events) == 2
    assert events[-1].actor_digest is not None
    assert len(events[-1].actor_digest or "") == 64
    serialized = "\n".join(event.model_dump_json() for event in events)
    assert f"memory-delete:{memory.memory_id}" not in serialized
    assert "Authentication entry point" not in serialized
