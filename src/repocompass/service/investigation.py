"""High-level investigation service shared by MCP and future API adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import uuid4

from repocompass.agent.lifecycle import HumanResponseRequest
from repocompass.agent.runner import (
    ResumeRequest,
    RunOutcome,
    RunRequest,
)
from repocompass.agent.thread_view import (
    ThreadExecutionState,
    ThreadInvestigationView,
)
from repocompass.agent.trace import RunTrace
from repocompass.structure.evidence_graph import EvidenceNodeKind
from repocompass.service.contracts import (
    ArtifactAvailability,
    CodeGraphView,
    ContextPackView,
    EvaluatorReportView,
    ExpandedSymbolView,
    ExpandSymbolRequest,
    InvestigationOperationKind,
    InvestigationOperationResult,
    LocalizationResultPayload,
    LocalizationResultView,
    ResumeInvestigationRequest,
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
    ) -> None:
        if not all(
            callable(getattr(runner, method, None))
            for method in ("run", "resume", "respond", "inspect_thread")
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

    @staticmethod
    def request_digest(
        request: StartInvestigationRequest | ResumeInvestigationRequest,
    ) -> str:
        """Return the canonical payload digest excluding its idempotency key."""

        if not isinstance(
            request,
            (StartInvestigationRequest, ResumeInvestigationRequest),
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

            outcome = self._runner.run(
                RunRequest(
                    thread_id=reservation.thread_id,
                    repo_path=request.repo_path,
                    objective=request.objective,
                    max_tool_calls_per_step=request.max_tool_calls_per_step,
                    max_explorer_calls_per_step=(request.max_explorer_calls_per_step),
                    max_no_progress_rounds=request.max_no_progress_rounds,
                    max_plan_revisions=request.max_plan_revisions,
                    context_budget_tokens=request.context_budget_tokens,
                    max_attempt_cost_usd=request.max_attempt_cost_usd,
                    human_review_policy=request.human_review_policy,
                    memory_retrieval_enabled=request.memory_retrieval_enabled,
                    max_repository_memory_results=(
                        request.max_repository_memory_results
                    ),
                    max_episodic_memory_results=(request.max_episodic_memory_results),
                )
            )
            return self._complete_outcome(
                request_id=request.request_id,
                request_digest=digest,
                outcome=outcome,
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

            if request.decision is None:
                outcome = self._runner.resume(
                    ResumeRequest(thread_id=reservation.thread_id)
                )
            else:
                assert request.interrupt_id is not None
                outcome = self._runner.respond(
                    HumanResponseRequest(
                        thread_id=reservation.thread_id,
                        interrupt_id=request.interrupt_id,
                        decision=request.decision,
                    )
                )
            return self._complete_outcome(
                request_id=request.request_id,
                request_digest=digest,
                outcome=outcome,
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

        try:
            view = self._runner.inspect_thread(
                ResumeRequest(thread_id=reservation.thread_id)
            )
        except ValueError as error:
            expected = f"Checkpoint thread does not exist: {reservation.thread_id}"
            if (
                str(error) == expected
                and reservation.kind is InvestigationOperationKind.START
            ):
                return None
            if str(error) == expected:
                raise InvestigationOperationPendingError(
                    "reserved lifecycle operation has no recoverable checkpoint"
                ) from error
            raise
        recovered = self._operation_from_view(
            request_id=reservation.request_id,
            view=view,
            outcome=None,
            recovered=True,
        )
        self._operation_store.complete(
            request_id=reservation.request_id,
            request_digest=request_digest,
            response=recovered,
        )
        return recovered.model_copy(update={"idempotent_replay": True})

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
