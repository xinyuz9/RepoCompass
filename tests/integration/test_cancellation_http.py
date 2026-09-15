"""Cancellation must get through while both lifecycle command locks are held."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.cancellation import DashboardCancellationService
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.service import InvestigationService, open_service_operation_store
from repocompass.storage.cancellation import open_cancellation_store
from repocompass.storage.raw_store import open_raw_run_store
from tests.model_doubles import default_investigation_plan, with_scripted_planner


def test_http_cancel_does_not_wait_for_long_start_and_rejects_unsafe_targets(tmp_path):
    (tmp_path / "example.py").write_text("value = 1\n")
    entered, release = Event(), Event()
    explorer = MagicMock(spec=BaseChatModel)

    def forbidden(_):
        raise AssertionError("next model must not run")

    explorer.bind_tools.return_value = RunnableLambda(forbidden)
    explorer.with_structured_output.return_value = RunnableLambda(forbidden)
    models = with_scripted_planner(explorer)

    def plan(_):
        entered.set()
        assert release.wait(10)
        return default_investigation_plan()

    models.planner.with_structured_output.return_value = RunnableLambda(plan)
    with (
        open_sqlite_checkpointer(tmp_path / "checkpoint.sqlite") as checkpoint,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_service_operation_store(tmp_path / "operations.sqlite") as operations,
        open_cancellation_store(tmp_path / "cancel.sqlite") as control,
    ):
        runner = AgentRunner(
            models,
            checkpointer=checkpoint,
            raw_run_store=raw,
            cancellation_store=control,
            run_id_factory=lambda: "http-run",
        )
        query = DashboardQueryService(
            runner,
            raw_run_store=raw,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=tmp_path,
            lifecycle_mutations_enabled=True,
            cancellation_enabled=True,
        )
        lifecycle = InvestigationService(runner, operation_store=operations)
        commands = DashboardCommandService(query, lifecycle_service=lifecycle)
        app = create_dashboard_app(
            query,
            commands=commands,
            cancellation=DashboardCancellationService(control, raw, tmp_path),
        )
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
            system = client.get("/api/v1/system").json()
            assert system["cancellation_enabled"] is True
            future = pool.submit(
                client.post,
                "/api/v1/investigations",
                headers={"Idempotency-Key": "start-1"},
                json={
                    "request_id": "start-1",
                    "goal": "Inspect source",
                    "expected_repository_id": system["repository_id"],
                    "expected_commit_sha": system["commit_sha"],
                    "memory_retrieval_enabled": False,
                },
            )
            try:
                assert entered.wait(5)
                row = control.get("http-run")
                command = {
                    "schema_version": 1,
                    "request_id": "cancel-1",
                    "run_id": "http-run",
                    "thread_id": row.thread_id,
                    "repository_id": row.repository_id,
                    "commit_sha": row.commit_sha,
                }
                url = "/api/v1/runs/http-run/cancellation"
                headers = {"Idempotency-Key": "cancel-1"}
                assert client.get(url).json()["state"] == "running"
                assert (
                    client.post(
                        url,
                        headers={**headers, "Origin": "https://evil.example"},
                        json=command,
                    ).status_code
                    == 403
                )
                assert (
                    client.post(
                        url, headers={"Idempotency-Key": "wrong"}, json=command
                    ).status_code
                    == 409
                )
                assert (
                    client.post(
                        url, headers=headers, json={**command, "thread_id": "wrong"}
                    ).status_code
                    == 409
                )
                assert (
                    client.post(
                        url,
                        headers=headers,
                        json={**command, "repository_id": "f" * 64},
                    ).status_code
                    == 409
                )
                assert (
                    client.get("/api/v1/runs/missing/cancellation").status_code == 404
                )
                raw.record_attempt(
                    raw.get_attempt("http-run").model_copy(
                        update={"run_id": "foreign", "repository_id": "f" * 64}
                    )
                )
                assert (
                    client.get("/api/v1/runs/foreign/cancellation").status_code == 404
                )
                response = client.post(url, headers=headers, json=command)
                assert response.status_code == 200
                assert response.json()["state"] == "cancel_requested"
                assert (
                    client.post(url, headers=headers, json=command).json()
                    == response.json()
                )
                assert not future.done()
            finally:
                release.set()
            result = future.result(timeout=10)
            assert result.status_code == 200, result.text
            assert result.json()["termination_status"] == "cancelled"
            assert client.get(url).json()["state"] == "cancelled"
            assert (
                client.post(url, headers=headers, json=command).json()["state"]
                == "cancelled"
            )
            late = {**command, "request_id": "late"}
            assert (
                client.post(
                    url, headers={"Idempotency-Key": "late"}, json=late
                ).status_code
                == 409
            )
