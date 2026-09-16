"""Run-bound crash recovery through production HTTP with no model re-invocation."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.commands import DashboardCommandService
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.memory.store import open_memory_store
from repocompass.models import WorkflowModels
from repocompass.service import InvestigationService, open_service_operation_store
from repocompass.storage.cancellation import open_cancellation_store
from repocompass.storage.raw_store import open_raw_run_store
from repocompass.tools.gateway import ToolExecutionContext


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("partial_commit", [False, True])
def test_reconcile_dead_worker_repairs_stores_without_replaying_models(
    tmp_path, terminal, partial_commit, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.py").write_text("FLAG = True\n")
    # Memory identity requires a real, local Git repository; nothing is pushed.
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
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.recovery_worker",
            str(tmp_path),
            "terminal" if terminal else "node",
        ],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 19, result.stderr.decode()
    context = ToolExecutionContext(repo)
    identity = resolve_memory_repository_identity(context)
    calls = []

    def forbidden(_):
        calls.append("model")
        raise AssertionError("recovery must not invoke models")

    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(forbidden)
    model.with_structured_output.return_value = RunnableLambda(forbidden)
    with (
        open_sqlite_checkpointer(tmp_path / "checkpoints.sqlite") as checkpoint,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_memory_store(tmp_path / "memory.sqlite") as memory,
        open_service_operation_store(tmp_path / "operations.sqlite") as operations,
        open_cancellation_store(tmp_path / "cancellations.sqlite") as cancellation,
    ):
        runner = AgentRunner(
            WorkflowModels.from_single_model(model),
            checkpointer=checkpoint,
            raw_run_store=raw,
            memory_store=memory,
            cancellation_store=cancellation,
        )
        service = InvestigationService(
            runner, operation_store=operations, raw_run_store=raw
        )
        query = DashboardQueryService(
            runner,
            raw_run_store=raw,
            trace_store=JsonlTraceStore(tmp_path / "traces"),
            repository_root=repo,
            lifecycle_mutations_enabled=True,
        )
        commands = DashboardCommandService(query, lifecycle_service=service)
        client = TestClient(
            create_dashboard_app(query, commands=commands),
            raise_server_exceptions=False,
        )
        path = "/api/v1/operations/orphan-request/reconcile"
        command = {
            "request_id": "orphan-request",
            "expected_run_id": "orphan-run",
            "expected_repository_id": context.repository_id,
            "expected_commit_sha": context.commit_sha,
        }
        headers = {"Idempotency-Key": "orphan-request"}
        assert service.get_operation("orphan-request").recovery_status == "reconcilable"
        assert raw.get_attempt("orphan-run").status.value == "running"
        assert (
            memory.get_episodic_memory_for_turn(
                identity.repository_key, "orphan-thread", 1
            )
            is None
        )
        assert (
            client.post(
                path,
                json=command,
                headers={**headers, "Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                path, json=command, headers={"Idempotency-Key": "other"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                path,
                json={**command, "expected_repository_id": "f" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                path, json={**command, "expected_run_id": "different"}, headers=headers
            ).status_code
            == 503
        )
        with operations.recovery("orphan-request"):
            assert client.post(path, json=command, headers=headers).status_code == 503
        with runner.recovery_guard("orphan-thread"):
            assert client.post(path, json=command, headers=headers).status_code == 503
        import repocompass.agent.runner as runner_module

        snapshot = runner_module.create_agent_graph(
            runner._models, checkpointer=checkpoint
        ).get_state({"configurable": {"thread_id": "orphan-thread"}})
        for invalid in (
            snapshot._replace(metadata={"repocompass_run_id": "another-run"}),
            snapshot._replace(values={**snapshot.values, "objective": "Wrong goal"}),
            snapshot._replace(values={**snapshot.values, "turn_number": 99}),
        ):
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    runner_module,
                    "create_agent_graph",
                    lambda *args, **kwargs: SimpleNamespace(
                        get_state=lambda _: invalid
                    ),
                )
                assert (
                    client.post(path, json=command, headers=headers).status_code == 409
                )
            assert raw.get_attempt("orphan-run").status.value == "running"
            assert operations.get("orphan-request").state.value == "pending"
            assert (
                memory.get_episodic_memory_for_turn(
                    identity.repository_key, "orphan-thread", 1
                )
                is None
            )
        complete = operations.complete
        if partial_commit:

            def fail(**_):
                raise RuntimeError("injected ledger failure")

            monkeypatch.setattr(operations, "complete", fail)
            assert client.post(path, json=command, headers=headers).status_code == 500
            assert raw.get_attempt("orphan-run").status.value == "failed"
            assert operations.get("orphan-request").state.value == "pending"
            monkeypatch.setattr(operations, "complete", complete)
        response = client.post(path, json=command, headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["recovery_note"] == "orphan_reconciled_without_model_replay"
        assert data["execution_state"] == ("completed" if terminal else "resumable")
        assert raw.get_attempt("orphan-run").status.value == "failed"
        assert cancellation.get("orphan-run").state == "finished"
        events = raw.list_trace_events("orphan-run")
        assert events[-1].payload["error_code"] == "orphan_reconciled"
        assert events[-1].payload["unknown_usage_preserved"] is True
        episode = memory.get_episodic_memory_for_turn(
            identity.repository_key, "orphan-thread", 1
        )
        assert (episode is not None) == terminal
        if terminal:
            assert episode.termination_status == "accepted"
        replay = client.post(path, json=command, headers=headers)
        assert replay.status_code == 200 and replay.json()["idempotent_replay"] is True
        assert raw.list_trace_events("orphan-run") == events
        assert (
            memory.get_episodic_memory_for_turn(
                identity.repository_key, "orphan-thread", 1
            )
            == episode
        )
        assert calls == []
