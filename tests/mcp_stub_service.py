"""Deterministic high-level service stub for MCP protocol tests."""

from repocompass.agent.result import FinalResult
from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.service.contracts import (
    ArtifactAvailability,
    CodeGraphView,
    ContextPackView,
    EvaluatorReportView,
    InvestigationOperationResult,
    LocalizationResultPayload,
    LocalizationResultView,
)
from repocompass.service.investigation import InvestigationArtifactUnavailableError


class StubInvestigationService:
    def __init__(self) -> None:
        self.start_requests = []
        self.resume_requests = []

    @staticmethod
    def _final_result() -> FinalResult:
        return FinalResult(
            relevant_files=["src/auth.py"],
            evidence=["src/auth.py:1-2 owns authentication."],
            summary="Authentication is owned by src/auth.py.",
            unresolved_questions=[],
            relevant_symbols=["src.auth.authenticate"],
        )

    def start_investigation(self, request):
        self.start_requests.append(request)
        return InvestigationOperationResult(
            request_id=request.request_id,
            thread_id="mcp-thread",
            execution_state=ThreadExecutionState.COMPLETED,
            run_id="mcp-run",
            termination_status="accepted",
            step_stop_reason=None,
            next_nodes=(),
            final_result=self._final_result(),
            pending_review=None,
            interrupt_id=None,
            usage_cost_report=None,
            idempotent_replay=False,
            recovered_from_checkpoint=False,
        )

    def resume_investigation(self, request):
        self.resume_requests.append(request)
        return InvestigationOperationResult(
            request_id=request.request_id,
            thread_id=request.thread_id,
            execution_state=ThreadExecutionState.COMPLETED,
            run_id="mcp-resume-run",
            termination_status="accepted",
            step_stop_reason=None,
            next_nodes=(),
            final_result=self._final_result(),
            pending_review=None,
            interrupt_id=None,
            usage_cost_report=None,
            idempotent_replay=False,
            recovered_from_checkpoint=False,
        )

    def get_localization_result(self, request):
        return LocalizationResultView(
            thread_id=request.thread_id,
            execution_state=ThreadExecutionState.COMPLETED,
            availability=ArtifactAvailability.AVAILABLE,
            result=LocalizationResultPayload(
                relevant_files=("src/auth.py",),
                relevant_symbols=("src.auth.authenticate",),
                evidence=("src/auth.py:1-2 owns authentication.",),
                summary="Authentication is owned by src/auth.py.",
                unresolved_questions=(),
                structure_artifact_id=None,
            ),
        )

    def get_code_graph(self, request):
        return CodeGraphView(
            thread_id=request.thread_id,
            execution_state=ThreadExecutionState.COMPLETED,
            availability=ArtifactAvailability.NOT_PRODUCED,
            structure=None,
        )

    def expand_symbol(self, request):
        raise InvestigationArtifactUnavailableError(
            "code graph is unavailable for this investigation"
        )

    def get_context_pack(self, request):
        return ContextPackView(
            thread_id=request.thread_id,
            execution_state=ThreadExecutionState.COMPLETED,
            availability=ArtifactAvailability.NOT_PRODUCED,
            context_pack=None,
            memory_recall_report=None,
        )

    def get_evaluator_report(self, request):
        return EvaluatorReportView(
            thread_id=request.thread_id,
            execution_state=ThreadExecutionState.COMPLETED,
            availability=ArtifactAvailability.NOT_PRODUCED,
            report=None,
            plan_revision_count=0,
            max_plan_revisions=2,
        )
