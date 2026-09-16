"""Production demo checkpoint/source/memory -> read-only baseline-to-head API."""

from fastapi.testclient import TestClient

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.demo import (
    DEMO_THREAD_ID,
    DEMO_RUN_ID,
    create_dashboard_demo,
)
from repocompass.dashboard.runtime import create_inspection_models
from repocompass.dashboard.service import DashboardQueryService
from repocompass.dashboard.trace_store import JsonlTraceStore
from repocompass.memory.store import open_memory_store
from repocompass.storage.raw_store import open_raw_run_store
from tests.unit.structure.test_drift import commit


def test_drift_api_invalidates_evidence_without_writes_or_source_disclosure(tmp_path):
    workspace = create_dashboard_demo(tmp_path / "demo")
    with (
        open_sqlite_checkpointer(workspace.checkpoint_database) as checkpoint,
        open_raw_run_store(workspace.raw_store_database) as raw,
        open_memory_store(workspace.memory_database) as memory,
    ):
        service = DashboardQueryService(
            AgentRunner(
                create_inspection_models(), checkpointer=checkpoint, raw_run_store=raw
            ),
            raw_run_store=raw,
            trace_store=JsonlTraceStore(workspace.trace_directory),
            repository_root=workspace.repository_path,
            memory_store=memory,
        )
        client = TestClient(create_dashboard_app(service))
        events = raw.list_trace_events(DEMO_RUN_ID)
        calls = raw.list_model_invocations(DEMO_RUN_ID)
        memory_before = service.list_memory()
        response = client.get(f"/api/v1/threads/{DEMO_THREAD_ID}/drift")
        assert response.status_code == 200, response.text
        before = response.json()
        assert before["recheck_paths"] == []
        assert before["provider_calls"] == 0
        assert (
            client.get(
                f"/api/v1/threads/{DEMO_THREAD_ID}/drift?expected_artifact_id="
                + "f" * 64
            ).status_code
            == 409
        )
        assert any(s["kind"] == "memory" for s in before["subjects"])
        assert client.get("/api/v1/threads/foreign-thread/drift").status_code == 404
        path = workspace.repository_path / "pkg/policy.py"
        path.write_text(path.read_text() + "\n# changed source\n")
        commit(workspace.repository_path)
        changed = client.get(f"/api/v1/threads/{DEMO_THREAD_ID}/drift")
        assert changed.status_code == 200, changed.text
        report = changed.json()
        assert report["baseline_commit"] != report["current_commit"]
        assert "pkg/policy.py" in report["recheck_paths"]
        assert any(s["status"] == "evidence_invalidated" for s in report["subjects"])
        assert str(workspace.repository_path) not in changed.text
        assert "class ProposalVerifier" not in changed.text
        assert raw.list_trace_events(DEMO_RUN_ID) == events
        assert raw.list_model_invocations(DEMO_RUN_ID) == calls
        assert service.list_memory() == memory_before
