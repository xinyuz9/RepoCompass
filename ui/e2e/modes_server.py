"""Real HTTP/SQLite mode fixture, with deterministic models and no credentials."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import uvicorn
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.candidate_output import EvidenceOnlyCandidateResult
from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.evaluation import SemanticAssessment
from repocompass.agent.planning import InvestigationPlan, PlanProposal, PlanStep
from repocompass.agent.runner import AgentRunner
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from repocompass.service import InvestigationService, open_service_operation_store
from repocompass.storage.raw_store import open_raw_run_store


class ScriptedModel:
    def bind_tools(self, *_args, **_kwargs):
        return RunnableLambda(lambda _: AIMessage(content="Source located by fixture."))

    def with_structured_output(self, schema, **_kwargs):
        values = {
            PlanProposal: InvestigationPlan(
                steps=[
                    PlanStep(
                        id="locate",
                        objective="Locate source",
                        expected_evidence=["example.py"],
                        status="active",
                    )
                ]
            ),
            CandidateResult: CandidateResult(
                evidence=[
                    EvidenceItem(
                        path="example.py",
                        line_start=1,
                        line_end=2,
                        claim="answer returns 42.",
                    )
                ],
                summary="The fixture answer is in example.py.",
                unresolved_questions=[],
            ),
            SemanticAssessment: SemanticAssessment(
                active_step_completed=True,
                answers_objective=True,
                evidence_supports_summary=True,
                has_critical_omissions=False,
                missing_evidence=[],
                recommended_focus=[],
                summary="Scripted review.",
            ),
        }
        # The source-only contract is normalized by the production candidate node.
        return RunnableLambda(
            lambda _: values[
                CandidateResult if schema is EvidenceOnlyCandidateResult else schema
            ]
        )


def main():
    with ExitStack() as stack:
        root = Path(
            stack.enter_context(TemporaryDirectory(prefix="repocompass-modes-e2e-"))
        )
        repo = root / "repo"
        repo.mkdir()
        (repo / "example.py").write_text("def answer():\n    return 42\n")
        checkpoint = stack.enter_context(
            open_sqlite_checkpointer(root / "checkpoint.sqlite")
        )
        raw = stack.enter_context(open_raw_run_store(root / "raw.sqlite"))
        memory = stack.enter_context(open_memory_store(root / "memory.sqlite"))
        operations = stack.enter_context(
            open_service_operation_store(root / "operations.sqlite")
        )
        runner = AgentRunner(
            WorkflowModels.from_single_model(cast(BaseChatModel, ScriptedModel())),
            checkpointer=checkpoint,
            raw_run_store=raw,
            memory_store=memory,
        )
        query = DashboardQueryService(
            runner,
            raw_run_store=raw,
            trace_store=JsonlTraceStore(root / "traces"),
            repository_root=repo,
            memory_store=memory,
            lifecycle_mutations_enabled=True,
        )
        service = InvestigationService(
            runner,
            operation_store=operations,
            raw_run_store=raw,
            trace_sink=JsonlTraceSink(),
            trace_directory=root / "traces",
        )
        app = create_dashboard_app(
            query, commands=DashboardCommandService(query, lifecycle_service=service)
        )
        uvicorn.run(
            app, host="127.0.0.1", port=8881, log_level="warning", proxy_headers=False
        )


if __name__ == "__main__":
    main()
