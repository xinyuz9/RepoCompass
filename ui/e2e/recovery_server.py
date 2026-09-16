"""Loopback-only crash-recovery browser fixture. No credentials or provider calls."""

import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from unittest.mock import MagicMock

import uvicorn
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from repocompass.service import InvestigationService, open_service_operation_store
from repocompass.storage.cancellation import open_cancellation_store
from repocompass.storage.raw_store import open_raw_run_store


def main():
    with ExitStack() as stack:
        root = Path(
            stack.enter_context(TemporaryDirectory(prefix="repocompass-recovery-e2e-"))
        )
        repo = root / "repo"
        repo.mkdir()
        (repo / "config.py").write_text("FLAG = True\n")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "config.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        checkpoint = stack.enter_context(
            open_sqlite_checkpointer(root / "checkpoints.sqlite")
        )
        raw = stack.enter_context(open_raw_run_store(root / "raw.sqlite"))
        memory = stack.enter_context(open_memory_store(root / "memory.sqlite"))
        operations = stack.enter_context(
            open_service_operation_store(root / "operations.sqlite")
        )
        cancellation = stack.enter_context(
            open_cancellation_store(root / "cancellations.sqlite")
        )

        def forbidden(_):
            raise AssertionError("the recovery UI must not execute models")

        model = MagicMock(spec=BaseChatModel)
        model.bind_tools.return_value = RunnableLambda(forbidden)
        model.with_structured_output.return_value = RunnableLambda(forbidden)
        runner = AgentRunner(
            WorkflowModels.from_single_model(model),
            checkpointer=checkpoint,
            raw_run_store=raw,
            memory_store=memory,
            cancellation_store=cancellation,
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
            runner, operation_store=operations, raw_run_store=raw
        )
        app = create_dashboard_app(
            query, commands=DashboardCommandService(query, lifecycle_service=service)
        )

        @app.post("/__test__/orphan")
        def orphan():
            suffix = uuid4().hex
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.recovery_worker",
                    str(root),
                    "node",
                    suffix,
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                timeout=20,
            )
            assert result.returncode == 19, result.stderr.decode()
            return {
                "request_id": "orphan-request-" + suffix,
                "thread_id": "orphan-thread-" + suffix,
                "run_id": "orphan-run-" + suffix,
            }

        uvicorn.run(
            app, host="127.0.0.1", port=8880, log_level="warning", proxy_headers=False
        )


if __name__ == "__main__":
    main()
