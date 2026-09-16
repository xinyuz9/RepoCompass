"""Application-service tests for lifecycle and high-level artifact boundaries."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.lifecycle import (
    ExecutionStatus,
    HumanDecision,
    HumanDecisionAction,
)
from repocompass.agent.result import FinalResult
from repocompass.agent.runner import RunOutcome
from repocompass.agent.structural_artifacts import build_investigation_structure
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.agent.trace import RunTrace
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.service.contracts import (
    ArtifactAvailability,
    ContinueInvestigationRequest,
    ExpandSymbolRequest,
    InvestigationOperationState,
    ResumeInvestigationRequest,
    StartInvestigationRequest,
    ThreadArtifactRequest,
)
from repocompass.service.investigation import (
    InvestigationOperationPendingError,
    InvestigationArtifactUnavailableError,
    InvestigationService,
)
from repocompass.service.operation_store import ServiceOperationStore
from repocompass.service.operation_store import ServiceOperationConflictError


def _final_result() -> FinalResult:
    return FinalResult(
        relevant_files=["src/auth.py"],
        evidence=["src/auth.py:10-14 validates credentials."],
        summary="Authentication is owned by src/auth.py.",
        unresolved_questions=[],
        relevant_symbols=["src.auth.authenticate"],
    )


def _completed_view(
    thread_id: str = "thread-1",
    final_result: FinalResult | None = None,
) -> ThreadInvestigationView:
    final_result = final_result or _final_result()
    return ThreadInvestigationView(
        thread_id=thread_id,
        execution_state=ThreadExecutionState.COMPLETED,
        turn_number=1,
        objective="Locate authentication.",
        plan=None,
        plan_revision_count=0,
        max_plan_revisions=2,
        next_nodes=(),
        termination_status="accepted",
        step_stop_reason=None,
        candidate_result=None,
        evaluation_report=None,
        structure=final_result.structure,
        context_pack=None,
        memory_recall_report=None,
        cost_budget=None,
        final_result=final_result,
        pending_review=None,
        interrupt_id=None,
    )


class _FakeRunner:
    def __init__(self) -> None:
        self.run_requests = []
        self.resume_requests = []
        self.response_requests = []
        self.continue_requests = []
        self.views = {"thread-1": _completed_view()}

    def run(self, request):
        self.run_requests.append(request)
        result = _final_result()
        self.views[request.thread_id] = _completed_view(request.thread_id)
        return RunOutcome(
            thread_id=request.thread_id,
            final_result=result,
            termination_status="accepted",
            step_stop_reason=None,
            trace=RunTrace("run-1"),
            execution_status=ExecutionStatus.COMPLETED,
        )

    def resume(self, request):
        self.resume_requests.append(request)
        result = _final_result()
        return RunOutcome(
            thread_id=request.thread_id,
            final_result=result,
            termination_status="accepted",
            step_stop_reason=None,
            trace=RunTrace("resume-run"),
        )

    def respond(self, request):
        self.response_requests.append(request)
        result = _final_result()
        return RunOutcome(
            thread_id=request.thread_id,
            final_result=result,
            termination_status="accepted",
            step_stop_reason=None,
            trace=RunTrace("respond-run"),
        )

    def continue_thread(self, request):
        self.continue_requests.append(request)
        result = _final_result()
        self.views[request.thread_id] = _completed_view(request.thread_id)
        return RunOutcome(
            thread_id=request.thread_id,
            final_result=result,
            termination_status="accepted",
            step_stop_reason=None,
            trace=RunTrace("continue-run"),
        )

    def inspect_thread(self, request):
        try:
            return self.views[request.thread_id]
        except KeyError as error:
            raise ValueError(
                f"Checkpoint thread does not exist: {request.thread_id}"
            ) from error


def _service(runner: _FakeRunner) -> InvestigationService:
    return InvestigationService(
        runner,
        operation_store=ServiceOperationStore(
            sqlite3.connect(":memory:", check_same_thread=False)
        ),
        thread_id_factory=lambda: "thread-generated",
    )


def test_start_is_durably_idempotent_and_preserves_all_budgets(tmp_path: Path) -> None:
    runner = _FakeRunner()
    service = _service(runner)
    request = StartInvestigationRequest(
        request_id="client-start-1",
        repo_path=tmp_path,
        objective="Locate authentication.",
        max_tool_calls_per_step=7,
        max_explorer_calls_per_step=5,
        max_no_progress_rounds=2,
        max_plan_revisions=3,
        context_budget_tokens=4096,
        max_attempt_cost_usd=None,
        memory_retrieval_enabled=True,
        max_repository_memory_results=4,
        max_episodic_memory_results=3,
    )

    first = service.start_investigation(request)
    replay = service.start_investigation(request)
    operation = service.get_operation(request.request_id)

    assert first.thread_id == "thread-generated"
    assert first.run_id == "run-1"
    assert first.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert replay.model_copy(update={"idempotent_replay": False}) == first
    assert operation is not None
    assert operation.state is InvestigationOperationState.COMPLETED
    assert operation.thread_id == first.thread_id
    assert operation.result == first
    assert service.get_operation("missing-request") is None
    assert len(runner.run_requests) == 1
    run_request = runner.run_requests[0]
    assert run_request.thread_id == "thread-generated"
    assert run_request.max_tool_calls_per_step == 7
    assert run_request.max_explorer_calls_per_step == 5
    assert run_request.max_no_progress_rounds == 2
    assert run_request.max_plan_revisions == 3
    assert run_request.context_budget_tokens == 4096
    assert run_request.memory_retrieval_enabled is True
    assert run_request.max_repository_memory_results == 4
    assert run_request.max_episodic_memory_results == 3

    changed = request.model_copy(update={"objective": "A changed objective."})
    with pytest.raises(ServiceOperationConflictError, match="different payload"):
        service.start_investigation(changed)


def test_service_persists_each_attempt_trace_once(tmp_path: Path) -> None:
    runner = _FakeRunner()
    trace_directory = tmp_path / "traces"
    service = InvestigationService(
        runner,
        operation_store=ServiceOperationStore(sqlite3.connect(":memory:")),
        thread_id_factory=lambda: "thread-generated",
        trace_sink=JsonlTraceSink(),
        trace_directory=trace_directory,
    )
    request = StartInvestigationRequest(
        request_id="trace-start",
        repo_path=tmp_path,
        objective="Locate authentication.",
    )

    first = service.start_investigation(request)
    replay = service.start_investigation(request)

    assert first.run_id == "run-1"
    assert replay.idempotent_replay is True
    assert tuple(path.name for path in trace_directory.iterdir()) == ("run-1.jsonl",)
    assert (trace_directory / "run-1.jsonl").read_text(encoding="utf-8") == ""


def test_service_rejects_trace_identifier_path_traversal(tmp_path: Path) -> None:
    class UnsafeTraceRunner(_FakeRunner):
        def run(self, request):
            return replace(super().run(request), trace=RunTrace("../escape"))

    trace_directory = tmp_path / "traces"
    service = InvestigationService(
        UnsafeTraceRunner(),
        operation_store=ServiceOperationStore(sqlite3.connect(":memory:")),
        thread_id_factory=lambda: "thread-generated",
        trace_sink=JsonlTraceSink(),
        trace_directory=trace_directory,
    )

    with pytest.raises(ValueError, match="invalid run identifier"):
        service.start_investigation(
            StartInvestigationRequest(
                request_id="unsafe-trace",
                repo_path=tmp_path,
                objective="Locate authentication.",
            )
        )

    assert not (tmp_path / "escape.jsonl").exists()


def test_resume_tool_preserves_resume_versus_human_response_semantics() -> None:
    runner = _FakeRunner()
    service = _service(runner)

    resumed = service.resume_investigation(
        ResumeInvestigationRequest(
            request_id="resume-1",
            thread_id="thread-1",
        )
    )
    responded = service.resume_investigation(
        ResumeInvestigationRequest(
            request_id="respond-1",
            thread_id="thread-1",
            interrupt_id="interrupt-1",
            decision=HumanDecision(
                action=HumanDecisionAction.APPROVE,
                feedback=None,
            ),
        )
    )

    assert resumed.run_id == "resume-run"
    assert responded.run_id == "respond-run"
    assert len(runner.resume_requests) == 1
    assert len(runner.response_requests) == 1
    assert runner.response_requests[0].interrupt_id == "interrupt-1"


def test_resume_replay_lookup_never_reserves_and_executes_once() -> None:
    runner = _FakeRunner()
    service = _service(runner)
    request = ResumeInvestigationRequest(
        request_id="respond-replay-1",
        thread_id="thread-1",
        interrupt_id="interrupt-1",
        decision=HumanDecision(
            action=HumanDecisionAction.APPROVE,
            feedback=None,
        ),
    )

    assert service.replay_resume_investigation(request) is None
    first = service.resume_investigation(request)
    replay = service.replay_resume_investigation(request)

    assert first.idempotent_replay is False
    assert replay is not None
    assert replay.idempotent_replay is True
    assert replay.model_copy(update={"idempotent_replay": False}) == first
    assert len(runner.response_requests) == 1

    changed = request.model_copy(
        update={
            "decision": HumanDecision(
                action=HumanDecisionAction.CANCEL,
                feedback=None,
            )
        }
    )
    with pytest.raises(ServiceOperationConflictError, match="different payload"):
        service.replay_resume_investigation(changed)


def test_continue_is_durably_idempotent_and_preserves_follow_up_prompt() -> None:
    runner = _FakeRunner()
    service = _service(runner)
    request = ContinueInvestigationRequest(
        request_id="continue-1",
        thread_id="thread-1",
        prompt="Now trace the failure path.",
    )

    assert service.replay_continue_investigation(request) is None
    first = service.continue_investigation(request)
    replay = service.continue_investigation(request)

    assert first.run_id == "continue-run"
    assert replay.idempotent_replay is True
    assert len(runner.continue_requests) == 1
    assert runner.continue_requests[0].prompt == "Now trace the failure path."

    changed = request.model_copy(update={"prompt": "Different follow-up."})
    with pytest.raises(ServiceOperationConflictError, match="different payload"):
        service.continue_investigation(changed)


def test_high_level_getters_never_offer_internal_repository_tools() -> None:
    runner = _FakeRunner()
    service = _service(runner)
    request = ThreadArtifactRequest(thread_id="thread-1")

    localization = service.get_localization_result(request)
    graph = service.get_code_graph(request)
    context = service.get_context_pack(request)
    evaluator = service.get_evaluator_report(request)

    assert localization.availability is ArtifactAvailability.AVAILABLE
    assert localization.result is not None
    assert localization.result.relevant_files == ("src/auth.py",)
    assert localization.result.structure_artifact_id is None
    assert graph.availability is ArtifactAvailability.NOT_PRODUCED
    assert context.availability is ArtifactAvailability.NOT_PRODUCED
    assert evaluator.availability is ArtifactAvailability.NOT_PRODUCED
    assert not hasattr(service, "list_tree")
    assert not hasattr(service, "search_text")
    assert not hasattr(service, "read_code")

    with pytest.raises(
        InvestigationArtifactUnavailableError,
        match="code graph",
    ):
        service.expand_symbol(
            ExpandSymbolRequest(
                thread_id="thread-1",
                canonical_name="src.auth.authenticate",
            )
        )


def test_pending_start_does_not_mistake_any_checkpoint_for_its_outcome(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner()
    store = ServiceOperationStore(sqlite3.connect(":memory:"))
    service = InvestigationService(
        runner,
        operation_store=store,
        thread_id_factory=lambda: "unused-thread",
    )
    request = StartInvestigationRequest(
        request_id="recover-1",
        repo_path=tmp_path,
        objective="Locate authentication.",
    )
    digest = service.request_digest(request)
    store.reserve(
        request_id=request.request_id,
        request_digest=digest,
        kind="start",
        proposed_thread_id="thread-1",
    )

    with pytest.raises(InvestigationOperationPendingError, match="reconciliation"):
        service.start_investigation(request)
    assert store.get(request.request_id).state.value == "pending"
    assert not runner.run_requests


def test_concurrent_duplicate_start_executes_runner_once(tmp_path: Path) -> None:
    runner = _FakeRunner()
    service = _service(runner)
    request = StartInvestigationRequest(
        request_id="concurrent-start",
        repo_path=tmp_path,
        objective="Locate authentication.",
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(
            executor.map(
                lambda _: service.start_investigation(request),
                range(8),
            )
        )

    assert len(runner.run_requests) == 1
    assert {result.thread_id for result in results} == {"thread-generated"}
    assert sum(not result.idempotent_replay for result in results) == 1


def test_expand_symbol_is_bounded_to_existing_task_graph(tmp_path: Path) -> None:
    source = "def authenticate(user):\n    return bool(user)\n"
    (tmp_path / "auth.py").write_text(source, encoding="utf-8")
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py",
                line_start=1,
                line_end=2,
                symbol="authenticate",
                claim="authenticate validates the supplied identity.",
            )
        ],
        summary="authenticate owns authentication.",
        unresolved_questions=[],
    )
    built = build_investigation_structure(
        repo_path=tmp_path,
        objective="Locate authentication.",
        candidate=candidate,
        valid_evidence_indices=frozenset({0}),
    )
    assert built.artifact is not None
    structure = built.artifact
    binding = structure.relevant_symbols[0]
    final_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py:1-2 owns authentication."],
        summary="Authentication is owned by authenticate.",
        unresolved_questions=[],
        relevant_symbols=[binding.canonical_name],
        structure=structure,
    )
    runner = _FakeRunner()
    runner.views["graph-thread"] = _completed_view(
        "graph-thread",
        final_result,
    )
    service = _service(runner)

    graph = service.get_code_graph(ThreadArtifactRequest(thread_id="graph-thread"))
    expanded = service.expand_symbol(
        ExpandSymbolRequest(
            thread_id="graph-thread",
            entity_id=binding.entity_id,
            max_edges=1,
        )
    )

    assert graph.availability is ArtifactAvailability.AVAILABLE
    assert graph.structure == structure
    assert expanded.graph_id == structure.graph.graph_id
    assert expanded.node.entity_id == binding.entity_id
    assert expanded.relevant_binding == binding
    assert len(expanded.incoming_edges) + len(expanded.outgoing_edges) <= 1
    assert all(
        edge.source_entity_id == binding.entity_id
        or edge.target_entity_id == binding.entity_id
        for edge in (*expanded.incoming_edges, *expanded.outgoing_edges)
    )
