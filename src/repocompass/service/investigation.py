"""High-level investigation service shared by MCP and future API adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Protocol
from uuid import uuid4

from repocompass.agent.lifecycle import ContinueRequest, HumanResponseRequest
from repocompass.agent.execution_binding import bind_run_observer
from repocompass.agent.runner import (
    ResumeRequest,
    RunOutcome,
    RunRequest,
)
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.agent.trace import RunTrace, TraceEvent, TraceEventKind
from repocompass.agent.cost import PricingTable
from repocompass.storage.raw_store import RawRunStore, AttemptStatus
from repocompass.storage.execution_lock import ExecutionLockUnavailable
from repocompass.structure.evidence_graph import EvidenceNodeKind
from repocompass.service.contracts import (
    ArtifactAvailability,
    CodeGraphView,
    ContinueInvestigationRequest,
    ContextPackView,
    EvaluatorReportView,
    ExpandedSymbolView,
    ExpandSymbolRequest,
    InvestigationOperationKind,
    InvestigationOperationResult,
    InvestigationOperationSnapshot,
    InvestigationOperationState,
    LocalizationResultPayload,
    LocalizationResultView,
    ResumeInvestigationRequest,
    ReconcileInvestigationRequest,
    StartInvestigationRequest,
    ThreadArtifactRequest,
)
from repocompass.service.operation_store import (
    OperationReservationState,
    ServiceOperationStore,
)


class InvestigationServiceError(RuntimeError):
    """Safe application error that adapters may expose without internals."""


class InvestigationOperationPendingError(InvestigationServiceError):
    """A prior mutation is uncertain and requires explicit recovery."""


class InvestigationArtifactUnavailableError(InvestigationServiceError):
    """A high-level artifact does not exist for the selected thread state."""


class InvestigationSymbolNotFoundError(InvestigationServiceError):
    """The selector did not identify a symbol in the task-specific graph."""


class InvestigationSymbolAmbiguousError(InvestigationServiceError):
    """A canonical-name selector matched more than one graph symbol."""


class InvestigationTraceSink(Protocol):
    """Minimal persistence boundary for one completed attempt trace."""

    def write(self, trace: RunTrace, destination: Path) -> Path: ...


class InvestigationRunner(Protocol):
    """Runner methods consumed by the application service."""

    def run(self, request: RunRequest) -> RunOutcome: ...

    def resume(self, request: ResumeRequest) -> RunOutcome: ...

    def respond(self, request: HumanResponseRequest) -> RunOutcome: ...

    def continue_thread(self, request: ContinueRequest) -> RunOutcome: ...

    def inspect_thread(self, request: ResumeRequest) -> ThreadInvestigationView: ...


class InvestigationService:
    """Serialize lifecycle mutations and expose only bounded domain artifacts."""

    def __init__(
        self,
        runner: InvestigationRunner,
        *,
        operation_store: ServiceOperationStore,
        thread_id_factory: Callable[[], str] = lambda: str(uuid4()),
        trace_sink: InvestigationTraceSink | None = None,
        trace_directory: Path | None = None,
        raw_run_store: RawRunStore | None = None,
        pricing_table: PricingTable | None = None,
    ) -> None:
        if not all(
            callable(getattr(runner, method, None))
            for method in (
                "run",
                "resume",
                "respond",
                "continue_thread",
                "inspect_thread",
            )
        ):
            raise TypeError("runner does not implement the investigation boundary")
        if not isinstance(operation_store, ServiceOperationStore):
            raise TypeError("operation_store must be a ServiceOperationStore")
        if not callable(thread_id_factory):
            raise TypeError("thread_id_factory must be callable")
        if (trace_sink is None) != (trace_directory is None):
            raise ValueError(
                "trace_sink and trace_directory must be configured together"
            )
        if trace_sink is not None and not callable(getattr(trace_sink, "write", None)):
            raise TypeError("trace_sink must implement write")
        if trace_directory is not None and not isinstance(trace_directory, Path):
            raise TypeError("trace_directory must be a Path")
        self._runner = runner
        self._operation_store = operation_store
        self._thread_id_factory = thread_id_factory
        self._trace_sink = trace_sink
        self._trace_directory = trace_directory
        self._mutation_lock = RLock()
        self._raw_run_store = raw_run_store
        self._pricing_table = pricing_table

    @staticmethod
    def request_digest(
        request: (
            StartInvestigationRequest
            | ResumeInvestigationRequest
            | ContinueInvestigationRequest
        ),
    ) -> str:
        """Return the canonical payload digest excluding its idempotency key."""

        if not isinstance(
            request,
            (
                StartInvestigationRequest,
                ResumeInvestigationRequest,
                ContinueInvestigationRequest,
            ),
        ):
            raise TypeError("request is not a lifecycle service contract")
        payload = request.model_dump(
            mode="json",
            exclude={"request_id"},
            exclude_computed_fields=True,
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(b"repocompass:service-request:v1\0" + encoded).hexdigest()

    def get_operation(self, request_id: str) -> InvestigationOperationSnapshot | None:
        """Return one durable lifecycle request without exposing its payload digest."""

        reservation = self._operation_store.get(request_id)
        if reservation is None:
            return None
        completed = reservation.state is OperationReservationState.COMPLETED
        return InvestigationOperationSnapshot(
            request_id=reservation.request_id,
            kind=reservation.kind,
            thread_id=reservation.thread_id,
            state=(
                InvestigationOperationState.COMPLETED
                if completed
                else InvestigationOperationState.PENDING
            ),
            reserved_at_ms=reservation.reserved_at_ms,
            result=reservation.response,
            bound_run_id=self._operation_store.execution_run(request_id),
            recovery_status=(
                "not_needed"
                if completed
                else "unavailable"
                if self._raw_run_store is None
                else self._operation_store.recovery_availability(request_id)
            ),
        )

    def reconcile_operation(
        self, request: ReconcileInvestigationRequest
    ) -> InvestigationOperationResult:
        """Explicit, run-bound local reconciliation. Never re-invokes the runner."""
        if self._raw_run_store is None or not all(
            callable(getattr(self._runner, name, None))
            for name in ("reconcile_persisted_artifacts", "recovery_guard")
        ):
            raise InvestigationOperationPendingError(
                "local reconciliation is unavailable"
            )
        try:
            with ExitStack() as guards:
                run_id = guards.enter_context(
                    self._operation_store.recovery(request.request_id)
                )
                reservation = self._operation_store.get(request.request_id)
                if reservation is None or run_id != request.expected_run_id:
                    raise InvestigationOperationPendingError(
                        "operation run binding changed"
                    )
                guards.enter_context(self._runner.recovery_guard(reservation.thread_id))
                raw = self._raw_run_store
                attempt = raw.get_attempt(run_id)
                if attempt is None or (
                    attempt.thread_id != reservation.thread_id
                    or attempt.execution_mode.value != reservation.kind.value
                    or attempt.repository_id != request.expected_repository_id
                    or attempt.commit_sha != request.expected_commit_sha
                ):
                    raise InvestigationOperationPendingError(
                        "operation and raw identity do not reconcile"
                    )
                if reservation.response is not None:
                    return reservation.response.model_copy(
                        update={"idempotent_replay": True}
                    )
                latest = raw.list_attempts(
                    thread_id=reservation.thread_id, max_results=1
                )
                if not latest or latest[0].run_id != run_id:
                    raise InvestigationOperationPendingError(
                        "a later attempt requires manual inspection"
                    )
                events = raw.list_trace_events(run_id)
                if not events:
                    raise InvestigationOperationPendingError(
                        "committed attempt journal is unavailable"
                    )
                view, episode, repository = self._runner.reconcile_persisted_artifacts(
                    attempt=attempt
                )
                if view.turn_number != attempt.turn_number:
                    raise InvestigationOperationPendingError(
                        "checkpoint turn does not reconcile"
                    )
                if attempt.status is AttemptStatus.RUNNING:
                    raw.finish_attempt(
                        run_id,
                        status=AttemptStatus.FAILED,
                        finished_at_ms=max(
                            time_ns() // 1_000_000, attempt.started_at_ms
                        ),
                        terminal_event=TraceEvent(
                            run_id=run_id,
                            sequence=len(events),
                            kind=TraceEventKind.RUN_FAILED,
                            payload={
                                "error_code": "orphan_reconciled",
                                "phase": "local_recovery",
                                "request_id": request.request_id,
                                "checkpoint_state": view.execution_state.value,
                                "model_replayed": False,
                                "unknown_usage_preserved": True,
                                "episode_consolidation": None
                                if episode is None
                                else episode.disposition.value,
                                "repository_consolidation": None
                                if repository is None
                                else repository.reason,
                            },
                        ),
                    )
                response = self._operation_from_view(
                    request_id=request.request_id,
                    view=view,
                    outcome=None,
                    recovered=True,
                )
                response = response.model_copy(
                    update={
                        "run_id": run_id,
                        "recovery_note": "orphan_reconciled_without_model_replay",
                        "usage_cost_report": None
                        if self._pricing_table is None
                        else raw.get_attempt_usage_cost_report(
                            run_id, self._pricing_table
                        ),
                    }
                )
                self._operation_store.complete(
                    request_id=request.request_id,
                    request_digest=reservation.request_digest,
                    response=response,
                )
                return response
        except ExecutionLockUnavailable as error:
            raise InvestigationOperationPendingError(str(error)) from error

    def start_investigation(
        self,
        request: StartInvestigationRequest,
    ) -> InvestigationOperationResult:
        """Start exactly one budget-bound investigation per request ID."""

        if not isinstance(request, StartInvestigationRequest):
            raise TypeError("request must be a StartInvestigationRequest")
        digest = self.request_digest(request)
        with self._mutation_lock:
            proposed_thread_id = self._normalize_generated_thread_id(
                self._thread_id_factory()
            )
            reservation = self._operation_store.reserve(
                request_id=request.request_id,
                request_digest=digest,
                kind=InvestigationOperationKind.START,
                proposed_thread_id=proposed_thread_id,
            )
            replay = self._resolve_existing_reservation(
                reservation=reservation,
                request_digest=digest,
            )
            if replay is not None:
                return replay

            return self._execute_reserved(
                request_id=request.request_id,
                request_digest=digest,
                invoke=lambda: self._runner.run(
                    RunRequest(
                        thread_id=reservation.thread_id,
                        repo_path=request.repo_path,
                        objective=request.objective,
                        max_tool_calls_per_step=request.max_tool_calls_per_step,
                        max_explorer_calls_per_step=(
                            request.max_explorer_calls_per_step
                        ),
                        max_no_progress_rounds=request.max_no_progress_rounds,
                        max_plan_revisions=request.max_plan_revisions,
                        context_budget_tokens=request.context_budget_tokens,
                        max_attempt_cost_usd=request.max_attempt_cost_usd,
                        human_review_policy=request.human_review_policy,
                        memory_retrieval_enabled=request.memory_retrieval_enabled,
                        max_repository_memory_results=(
                            request.max_repository_memory_results
                        ),
                        max_episodic_memory_results=(
                            request.max_episodic_memory_results
                        ),
                    )
                ),
            )

    def replay_resume_investigation(
        self,
        request: ResumeInvestigationRequest,
    ) -> InvestigationOperationResult | None:
        """Return an exact existing resume/respond result without reserving a key."""

        if not isinstance(request, ResumeInvestigationRequest):
            raise TypeError("request must be a ResumeInvestigationRequest")
        digest = self.request_digest(request)
        with self._mutation_lock:
            reservation = self._operation_store.lookup(
                request_id=request.request_id,
                request_digest=digest,
                kind=request.operation_kind,
                thread_id=request.thread_id,
            )
            if reservation is None:
                return None
            return self._resolve_existing_reservation(
                reservation=reservation,
                request_digest=digest,
            )

    def resume_investigation(
        self,
        request: ResumeInvestigationRequest,
    ) -> InvestigationOperationResult:
        """Resume graph work or consume one exact pending Human decision."""

        if not isinstance(request, ResumeInvestigationRequest):
            raise TypeError("request must be a ResumeInvestigationRequest")
        digest = self.request_digest(request)
        with self._mutation_lock:
            reservation = self._operation_store.reserve(
                request_id=request.request_id,
                request_digest=digest,
                kind=request.operation_kind,
                proposed_thread_id=request.thread_id,
            )
            replay = self._resolve_existing_reservation(
                reservation=reservation,
                request_digest=digest,
            )
            if replay is not None:
                return replay

            def invoke():
                if request.decision is None:
                    return self._runner.resume(
                        ResumeRequest(thread_id=reservation.thread_id)
                    )
                assert request.interrupt_id is not None
                return self._runner.respond(
                    HumanResponseRequest(
                        thread_id=reservation.thread_id,
                        interrupt_id=request.interrupt_id,
                        decision=request.decision,
                    )
                )

            return self._execute_reserved(
                request_id=request.request_id,
                request_digest=digest,
                invoke=invoke,
            )

    def replay_continue_investigation(
        self,
        request: ContinueInvestigationRequest,
    ) -> InvestigationOperationResult | None:
        """Return an exact existing follow-up result without reserving a key."""

        if not isinstance(request, ContinueInvestigationRequest):
            raise TypeError("request must be a ContinueInvestigationRequest")
        digest = self.request_digest(request)
        with self._mutation_lock:
            reservation = self._operation_store.lookup(
                request_id=request.request_id,
                request_digest=digest,
                kind=request.operation_kind,
                thread_id=request.thread_id,
            )
            if reservation is None:
                return None
            return self._resolve_existing_reservation(
                reservation=reservation,
                request_digest=digest,
            )

    def continue_investigation(
        self,
        request: ContinueInvestigationRequest,
    ) -> InvestigationOperationResult:
        """Start one new Human turn on a completed checkpoint thread."""

        if not isinstance(request, ContinueInvestigationRequest):
            raise TypeError("request must be a ContinueInvestigationRequest")
        digest = self.request_digest(request)
        with self._mutation_lock:
            reservation = self._operation_store.reserve(
                request_id=request.request_id,
                request_digest=digest,
                kind=request.operation_kind,
                proposed_thread_id=request.thread_id,
            )
            replay = self._resolve_existing_reservation(
                reservation=reservation,
                request_digest=digest,
            )
            if replay is not None:
                return replay

            return self._execute_reserved(
                request_id=request.request_id,
                request_digest=digest,
                invoke=lambda: self._runner.continue_thread(
                    ContinueRequest(
                        thread_id=reservation.thread_id,
                        prompt=request.prompt,
                    )
                ),
            )

    def get_localization_result(
        self,
        request: ThreadArtifactRequest,
    ) -> LocalizationResultView:
        view = self._inspect(request)
        final = view.final_result
        payload = (
            None
            if final is None
            else LocalizationResultPayload(
                relevant_files=tuple(final.relevant_files),
                relevant_symbols=tuple(final.relevant_symbols),
                evidence=tuple(final.evidence),
                summary=final.summary,
                unresolved_questions=tuple(final.unresolved_questions),
                structure_artifact_id=(
                    None if final.structure is None else final.structure.artifact_id
                ),
            )
        )
        return LocalizationResultView(
            thread_id=view.thread_id,
            execution_state=view.execution_state,
            availability=self._availability(view, payload),
            result=payload,
        )

    def get_code_graph(self, request: ThreadArtifactRequest) -> CodeGraphView:
        view = self._inspect(request)
        return CodeGraphView(
            thread_id=view.thread_id,
            execution_state=view.execution_state,
            availability=self._availability(view, view.structure),
            structure=view.structure,
        )

    def get_context_pack(self, request: ThreadArtifactRequest) -> ContextPackView:
        view = self._inspect(request)
        return ContextPackView(
            thread_id=view.thread_id,
            execution_state=view.execution_state,
            availability=self._availability(view, view.context_pack),
            context_pack=view.context_pack,
            memory_recall_report=view.memory_recall_report,
        )

    def get_evaluator_report(
        self,
        request: ThreadArtifactRequest,
    ) -> EvaluatorReportView:
        view = self._inspect(request)
        return EvaluatorReportView(
            thread_id=view.thread_id,
            execution_state=view.execution_state,
            availability=self._availability(view, view.evaluation_report),
            report=view.evaluation_report,
            plan_revision_count=view.plan_revision_count,
            max_plan_revisions=view.max_plan_revisions,
        )

    def expand_symbol(self, request: ExpandSymbolRequest) -> ExpandedSymbolView:
        """Expand one graph symbol without invoking repository tools."""

        if not isinstance(request, ExpandSymbolRequest):
            raise TypeError("request must be an ExpandSymbolRequest")
        view = self._runner.inspect_thread(ResumeRequest(thread_id=request.thread_id))
        structure = view.structure
        if structure is None:
            raise InvestigationArtifactUnavailableError(
                "code graph is unavailable for this investigation"
            )
        symbol_kinds = {
            EvidenceNodeKind.CLASS,
            EvidenceNodeKind.FUNCTION,
            EvidenceNodeKind.METHOD,
            EvidenceNodeKind.TEST,
        }
        matches = [
            node
            for node in structure.graph.nodes
            if node.kind in symbol_kinds
            and (
                node.entity_id == request.entity_id
                if request.entity_id is not None
                else node.canonical_name == request.canonical_name
            )
        ]
        if not matches:
            raise InvestigationSymbolNotFoundError(
                "symbol is not present in the task-specific code graph"
            )
        if len(matches) > 1:
            raise InvestigationSymbolAmbiguousError(
                "canonical symbol name is ambiguous; use entity_id"
            )
        selected = matches[0]
        incident = sorted(
            (
                edge
                for edge in structure.graph.edges
                if selected.entity_id in (edge.source_entity_id, edge.target_entity_id)
            ),
            key=lambda edge: edge.edge_id,
        )
        chosen = incident[: request.max_edges]
        incoming = tuple(
            edge for edge in chosen if edge.target_entity_id == selected.entity_id
        )
        outgoing = tuple(
            edge for edge in chosen if edge.source_entity_id == selected.entity_id
        )
        neighbor_ids = {edge.source_entity_id for edge in incoming} | {
            edge.target_entity_id for edge in outgoing
        }
        neighbors = tuple(
            sorted(
                (
                    node
                    for node in structure.graph.nodes
                    if node.entity_id in neighbor_ids
                ),
                key=lambda node: node.node_id,
            )
        )
        binding = next(
            (
                item
                for item in structure.relevant_symbols
                if item.entity_id == selected.entity_id
            ),
            None,
        )
        return ExpandedSymbolView(
            thread_id=view.thread_id,
            graph_id=structure.graph.graph_id,
            node=selected,
            relevant_binding=binding,
            incoming_edges=incoming,
            outgoing_edges=outgoing,
            neighbor_nodes=neighbors,
            truncated=len(chosen) < len(incident),
        )

    def _inspect(self, request: ThreadArtifactRequest) -> ThreadInvestigationView:
        if not isinstance(request, ThreadArtifactRequest):
            raise TypeError("request must be a ThreadArtifactRequest")
        return self._runner.inspect_thread(ResumeRequest(thread_id=request.thread_id))

    def _resolve_existing_reservation(
        self,
        *,
        reservation,
        request_digest: str,
    ) -> InvestigationOperationResult | None:
        if reservation.state is OperationReservationState.COMPLETED:
            assert reservation.response is not None
            return reservation.response.model_copy(update={"idempotent_replay": True})
        if reservation.state is OperationReservationState.NEW:
            return None

        raise InvestigationOperationPendingError(
            "pending lifecycle operation requires explicit run-bound reconciliation"
        )

    def _execute_reserved(
        self,
        *,
        request_id: str,
        request_digest: str,
        invoke: Callable[[], RunOutcome],
    ) -> InvestigationOperationResult:
        try:
            with (
                self._operation_store.execution(request_id),
                bind_run_observer(
                    lambda run_id: self._operation_store.bind_run(request_id, run_id)
                ),
            ):
                outcome = invoke()
                return self._complete_outcome(
                    request_id=request_id,
                    request_digest=request_digest,
                    outcome=outcome,
                )
        except ExecutionLockUnavailable as error:
            raise InvestigationOperationPendingError(str(error)) from error

    def _complete_outcome(
        self,
        *,
        request_id: str,
        request_digest: str,
        outcome: RunOutcome,
    ) -> InvestigationOperationResult:
        view = self._runner.inspect_thread(ResumeRequest(thread_id=outcome.thread_id))
        response = self._operation_from_view(
            request_id=request_id,
            view=view,
            outcome=outcome,
            recovered=False,
        )
        if self._trace_sink is not None:
            assert self._trace_directory is not None
            self._trace_sink.write(
                outcome.trace,
                self._trace_directory / self._trace_filename(outcome.trace.run_id),
            )
        self._operation_store.complete(
            request_id=request_id,
            request_digest=request_digest,
            response=response,
        )
        return response

    @staticmethod
    def _operation_from_view(
        *,
        request_id: str,
        view: ThreadInvestigationView,
        outcome: RunOutcome | None,
        recovered: bool,
    ) -> InvestigationOperationResult:
        if outcome is not None:
            if outcome.thread_id != view.thread_id:
                raise ValueError("Runner outcome and checkpoint thread disagree")
            if outcome.final_result != view.final_result:
                raise ValueError("Runner outcome and checkpoint result disagree")
            if outcome.interrupt_id != view.interrupt_id:
                raise ValueError("Runner outcome and checkpoint interrupt disagree")
        return InvestigationOperationResult(
            request_id=request_id,
            thread_id=view.thread_id,
            execution_state=view.execution_state,
            run_id=None if outcome is None else outcome.trace.run_id,
            termination_status=view.termination_status,
            step_stop_reason=view.step_stop_reason,
            next_nodes=view.next_nodes,
            final_result=view.final_result,
            pending_review=view.pending_review,
            interrupt_id=view.interrupt_id,
            usage_cost_report=(None if outcome is None else outcome.usage_cost_report),
            idempotent_replay=False,
            recovered_from_checkpoint=recovered,
        )

    @staticmethod
    def _availability(
        view: ThreadInvestigationView,
        artifact: object | None,
    ) -> ArtifactAvailability:
        if artifact is not None:
            return ArtifactAvailability.AVAILABLE
        if view.execution_state is ThreadExecutionState.COMPLETED:
            return ArtifactAvailability.NOT_PRODUCED
        return ArtifactAvailability.NOT_READY

    @staticmethod
    def _normalize_generated_thread_id(value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("thread_id_factory must return a string")
        normalized = value.strip()
        if not normalized or len(normalized) > 256:
            raise ValueError("thread_id_factory returned an invalid identifier")
        return normalized

    @staticmethod
    def _trace_filename(run_id: str) -> str:
        """Map an internal run identifier to one traversal-safe filename."""

        if (
            not run_id
            or len(run_id) > 256
            or run_id in {".", ".."}
            or "/" in run_id
            or "\\" in run_id
            or "\x00" in run_id
        ):
            raise ValueError("Runner trace returned an invalid run identifier")
        return f"{run_id}.jsonl"
