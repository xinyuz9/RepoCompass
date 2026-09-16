"""Ephemeral, loopback-only mutation fixture. Never loads credentials or provider models."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import cast

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
import uvicorn

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.planning import InvestigationPlan, PlanStep, PlanProposal
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.cancellation import DashboardCancellationService
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.models import WorkflowModels
from repocompass.service import InvestigationService, open_service_operation_store
from repocompass.storage.cancellation import open_cancellation_store
from repocompass.storage.raw_store import open_raw_run_store


class ControlledModel:
    def __init__(self):
        self.releases: dict[str, Event] = {}
        self.lock = Lock()

    def with_structured_output(self, schema, **_):
        return RunnableLambda(self.plan if schema is PlanProposal else self.forbidden)

    def bind_tools(self, *_args, **_kwargs):
        return RunnableLambda(self.forbidden)

    def forbidden(self, _):
        raise AssertionError("Cancellation fixture must never reach another model")

    def plan(self, _, config: RunnableConfig):
        run_id = config["configurable"]["repocompass_run_id"]
        with self.lock:
            release = self.releases.setdefault(run_id, Event())
        if not release.wait(45):
            raise RuntimeError(
                "fixture node timed out waiting for explicit test release"
            )
        return {
            "parsed": InvestigationPlan(
                steps=[
                    PlanStep(
                        id="inspect",
                        objective="Inspect the fixture repository",
                        expected_evidence=["example.py source"],
                        status="active",
                    )
                ]
            ),
            "parsing_error": None,
            "raw": AIMessage(
                content="scripted plan",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            ),
        }


def main():
    with ExitStack() as stack:
        root = Path(
            stack.enter_context(
                TemporaryDirectory(prefix="repocompass-cancellation-e2e-")
            )
        )
        repo = root / "cancellation-fixture"
        repo.mkdir()
        (repo / "example.py").write_text("value = 1\n", encoding="utf-8")
        checkpoint = stack.enter_context(
            open_sqlite_checkpointer(root / "checkpoint.sqlite")
        )
        raw = stack.enter_context(open_raw_run_store(root / "raw.sqlite"))
        control = stack.enter_context(open_cancellation_store(root / "control.sqlite"))
        operations = stack.enter_context(
            open_service_operation_store(root / "operations.sqlite")
        )
        model = ControlledModel()
        runner = AgentRunner(
            WorkflowModels.from_single_model(cast(BaseChatModel, model)),
            checkpointer=checkpoint,
            raw_run_store=raw,
            cancellation_store=control,
        )
        query = DashboardQueryService(
            runner,
            raw_run_store=raw,
            trace_store=JsonlTraceStore(root / "traces"),
            repository_root=repo,
            lifecycle_mutations_enabled=True,
            cancellation_enabled=True,
        )
        commands = DashboardCommandService(
            query,
            lifecycle_service=InvestigationService(runner, operation_store=operations),
        )
        app = create_dashboard_app(
            query,
            commands=commands,
            cancellation=DashboardCancellationService(control, raw, repo),
        )

        @app.post("/__test__/release-node/{run_id}")
        def release_node(run_id: str):
            record = control.get(run_id)
            with model.lock:
                release = model.releases.get(run_id)
            if record is None or record.state != "cancel_requested" or release is None:
                raise HTTPException(
                    409, "fixture requires a persisted cancellation request"
                )
            release.set()
            return {"released": True}

        uvicorn.run(
            app, host="127.0.0.1", port=8879, log_level="warning", proxy_headers=False
        )


if __name__ == "__main__":
    main()
