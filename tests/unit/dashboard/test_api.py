"""Local FastAPI dashboard contract, error, host, and SSE tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from repocompass.agent.trace import TraceEventKind
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardMemoryCatalog,
    DashboardRunDetail,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    PublicTraceEvent,
)
from repocompass.dashboard.service import DashboardNotFoundError
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    RawStoreError,
)


def _attempt(run_id: str = "run-1") -> RawRunAttempt:
    return RawRunAttempt(
        run_id=run_id,
        thread_id="thread-1",
        turn_number=1,
        execution_mode=AttemptExecutionMode.START,
        repository_id="a" * 64,
        commit_sha="b" * 40,
        objective_digest="c" * 64,
        status=AttemptStatus.COMPLETED,
        started_at_ms=1_000,
        finished_at_ms=2_000,
    )


def _run_detail(
    *,
    run_id: str = "run-1",
    persisted: bool = True,
) -> DashboardRunDetail:
    events = (
        tuple(
            PublicTraceEvent(
                run_id=run_id,
                sequence=sequence,
                kind=(
                    TraceEventKind.RUN_STARTED
                    if sequence == 0
                    else TraceEventKind.NODE_COMPLETED
                ),
                node=None if sequence == 0 else f"node-{sequence}",
                payload={"sequence": sequence, "path": "src/auth.py"},
            )
            for sequence in range(3)
        )
        if persisted
        else ()
    )
    return DashboardRunDetail(
        attempt=_attempt(run_id),
        trace_availability=(
            DashboardAvailability.AVAILABLE
            if persisted
            else DashboardAvailability.NOT_PERSISTED
        ),
        trace_events=events,
        plan_timeline=(),
        evaluator_timeline=(),
        context_packs=(),
        model_invocations=(),
        cost_availability=DashboardAvailability.NOT_CONFIGURED,
    )


class _FakeDashboardService:
    def __init__(self) -> None:
        self.fail_system = False
        self.fail_threads = False

    def system_info(self) -> DashboardSystemInfo:
        if self.fail_system:
            raise RuntimeError("private path: /home/operator/repository")
        return DashboardSystemInfo(
            product_version="0.1.0",
            repository_id="a" * 64,
            commit_sha="b" * 40,
            pricing_configured=False,
            lifecycle_mutations_enabled=False,
            memory_mutations_enabled=False,
        )

    def list_threads(self, *, max_results: int = 50) -> DashboardThreadCatalog:
        if self.fail_threads:
            raise RawStoreError("private sqlite detail")
        assert 1 <= max_results <= 100
        return DashboardThreadCatalog(
            threads=(),
            scanned_attempt_count=0,
            truncated=False,
        )

    def get_thread(self, thread_id: str):
        raise DashboardNotFoundError(f"missing thread {thread_id}")

    def get_run(self, run_id: str) -> DashboardRunDetail:
        if run_id == "run-1":
            return _run_detail()
        if run_id == "empty-run":
            return _run_detail(run_id=run_id, persisted=False)
        raise DashboardNotFoundError(f"missing run {run_id}")

    def list_memory(self, *, max_results: int = 50, memory_scope=None):
        assert 1 <= max_results <= 50
        assert memory_scope is None
        return DashboardMemoryCatalog(
            availability=DashboardAvailability.NOT_CONFIGURED,
            entries=(),
            tombstones=(),
        )

    def get_memory_audit(self, memory_id: str):
        raise DashboardNotFoundError(f"missing memory {memory_id}")


def _client(service: _FakeDashboardService) -> TestClient:
    return TestClient(
        create_dashboard_app(service),
        raise_server_exceptions=False,
    )


def test_read_api_returns_versioned_empty_states_and_security_headers() -> None:
    client = _client(_FakeDashboardService())

    system = client.get("/api/v1/system")
    threads = client.get("/api/v1/threads", params={"max_results": 10})
    memory = client.get("/api/v1/memory")

    assert system.status_code == 200
    assert system.json()["schema_version"] == 1
    assert "repository_root" not in system.text
    assert system.headers["cache-control"] == "no-store"
    assert system.headers["x-content-type-options"] == "nosniff"
    assert threads.json()["threads"] == []
    assert memory.json()["availability"] == "not_configured"


def test_every_http_failure_uses_sanitized_problem_detail() -> None:
    service = _FakeDashboardService()
    client = _client(service)

    invalid_host = client.get(
        "/api/v1/system",
        headers={"host": "dashboard.attacker.example"},
    )
    invalid_query = client.get("/api/v1/threads", params={"max_results": 0})
    missing_route = client.get("/api/v1/not-a-route")
    missing_run = client.get("/api/v1/runs/missing")
    wrong_method = client.post("/api/v1/system")
    service.fail_threads = True
    storage_failure = client.get("/api/v1/threads")
    service.fail_system = True
    unexpected = client.get("/api/v1/system")

    expected = (
        (invalid_host, 400),
        (invalid_query, 422),
        (missing_route, 404),
        (missing_run, 404),
        (wrong_method, 405),
        (storage_failure, 503),
        (unexpected, 500),
    )
    for response, status in expected:
        assert response.status_code == status
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["schema_version"] == 1
        assert body["status"] == status
        assert body["type"].startswith("urn:repocompass:problem:")
        assert "/home/operator" not in response.text
        assert "private sqlite detail" not in response.text


def test_sse_is_cursor_bounded_and_reports_snapshot_completion() -> None:
    client = _client(_FakeDashboardService())

    first = client.get(
        "/api/v1/runs/run-1/events",
        params={"max_events": 2},
    )
    remaining = client.get(
        "/api/v1/runs/run-1/events",
        params={"after_sequence": 1, "max_events": 2},
    )

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    assert first.headers["x-accel-buffering"] == "no"
    assert first.text.count("event: trace") == 2
    assert "id: 0" in first.text
    assert "id: 1" in first.text
    assert '"truncated":true' in first.text
    assert '"next_after_sequence":1' in first.text
    assert remaining.text.count("event: trace") == 1
    assert "id: 2" in remaining.text
    assert '"truncated":false' in remaining.text


def test_sse_missing_trace_is_an_explicit_terminal_empty_state() -> None:
    response = _client(_FakeDashboardService()).get("/api/v1/runs/empty-run/events")

    assert response.status_code == 200
    assert response.headers["x-repocompass-trace-availability"] == "not_persisted"
    assert "event: trace" not in response.text
    assert response.text.count("event: snapshot_end") == 1
    assert '"availability":"not_persisted"' in response.text
    assert '"emitted_count":0' in response.text


def test_openapi_contains_only_read_dashboard_paths() -> None:
    document = _client(_FakeDashboardService()).get("/api/v1/openapi.json").json()

    assert document["info"]["title"] == "RepoCompass Local Investigation API"
    assert document["paths"]
    for operations in document["paths"].values():
        assert set(operations).issubset({"get", "parameters"})
    serialized = str(document)
    assert "search_text" not in serialized
    assert "read_code" not in serialized
