"""Local FastAPI dashboard contract, error, host, and SSE tests."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.agent.trace import TraceEventKind
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardHumanResponseCommand,
    DashboardMemoryCatalog,
    DashboardMemoryDeleteCommand,
    DashboardMemoryDeleteResult,
    DashboardRunDetail,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    PublicTraceEvent,
)
from repocompass.dashboard.service import DashboardNotFoundError
from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryAuditReason,
    MemoryScope,
    create_memory_audit_event,
    create_memory_tombstone,
)
from repocompass.service.contracts import InvestigationOperationResult
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
    def __init__(
        self,
        *,
        lifecycle_mutations_enabled: bool = False,
        memory_mutations_enabled: bool = False,
    ) -> None:
        self.fail_system = False
        self.fail_threads = False
        self.lifecycle_mutations_enabled = lifecycle_mutations_enabled
        self.memory_mutations_enabled = memory_mutations_enabled

    def system_info(self) -> DashboardSystemInfo:
        if self.fail_system:
            raise RuntimeError("private path: /home/operator/repository")
        return DashboardSystemInfo(
            product_version="0.1.0",
            repository_id="a" * 64,
            commit_sha="b" * 40,
            pricing_configured=False,
            lifecycle_mutations_enabled=self.lifecycle_mutations_enabled,
            memory_mutations_enabled=self.memory_mutations_enabled,
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


class _FakeDashboardCommands:
    lifecycle_mutations_enabled = True
    memory_mutations_enabled = True

    def __init__(self) -> None:
        self.response_calls: list[tuple[str, DashboardHumanResponseCommand]] = []
        self.delete_calls: list[tuple[str, DashboardMemoryDeleteCommand]] = []

    def respond_to_review(
        self,
        thread_id: str,
        command: DashboardHumanResponseCommand,
    ) -> InvestigationOperationResult:
        self.response_calls.append((thread_id, command))
        return InvestigationOperationResult(
            request_id=command.request_id,
            thread_id=thread_id,
            execution_state=ThreadExecutionState.RESUMABLE,
            run_id="response-run",
            termination_status=None,
            step_stop_reason=None,
            next_nodes=("agent",),
            final_result=None,
            pending_review=None,
            interrupt_id=None,
            usage_cost_report=None,
            idempotent_replay=False,
            recovered_from_checkpoint=False,
        )

    def delete_memory(
        self,
        memory_id: str,
        command: DashboardMemoryDeleteCommand,
    ) -> DashboardMemoryDeleteResult:
        self.delete_calls.append((memory_id, command))
        event = create_memory_audit_event(
            memory_id=memory_id,
            repository_key="a" * 64,
            action=MemoryAuditAction.DELETED,
            reason=MemoryAuditReason.USER_REQUEST,
            created_at_ms=2_000,
        )
        tombstone = create_memory_tombstone(
            memory_id=memory_id,
            repository_key="a" * 64,
            memory_scope=MemoryScope.REPOSITORY,
            deletion_event_id=event.event_id,
            reason=MemoryAuditReason.USER_REQUEST,
            deleted_at_ms=2_000,
        )
        return DashboardMemoryDeleteResult(
            request_id=command.request_id,
            memory_id=memory_id,
            tombstone=tombstone,
            idempotent_replay=False,
        )


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
    content_security_policy = system.headers["content-security-policy"]
    assert "script-src 'self'" in content_security_policy
    assert "style-src-elem 'self'" in content_security_policy
    assert "style-src-attr 'unsafe-inline'" in content_security_policy
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


def test_mutation_routes_require_matching_capabilities() -> None:
    commands = _FakeDashboardCommands()

    with pytest.raises(ValueError, match="capabilities require"):
        create_dashboard_app(_FakeDashboardService(lifecycle_mutations_enabled=True))
    with pytest.raises(ValueError, match="do not reconcile"):
        create_dashboard_app(_FakeDashboardService(), commands=commands)


def test_explicit_mutations_require_idempotency_and_same_origin() -> None:
    service = _FakeDashboardService(
        lifecycle_mutations_enabled=True,
        memory_mutations_enabled=True,
    )
    commands = _FakeDashboardCommands()
    client = TestClient(
        create_dashboard_app(service, commands=commands),
        raise_server_exceptions=False,
    )
    response_body = {
        "request_id": "review-command-1",
        "interrupt_id": "interrupt-1",
        "decision": {"action": "approve", "feedback": None},
    }

    missing_key = client.post(
        "/api/v1/threads/thread-1/responses",
        json=response_body,
    )
    mismatched_key = client.post(
        "/api/v1/threads/thread-1/responses",
        json=response_body,
        headers={"Idempotency-Key": "review-command-other"},
    )
    cross_site = client.post(
        "/api/v1/threads/thread-1/responses",
        json=response_body,
        headers={
            "Idempotency-Key": "review-command-1",
            "Origin": "http://attacker.example",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    accepted = client.post(
        "/api/v1/threads/thread-1/responses",
        json=response_body,
        headers={
            "Idempotency-Key": "review-command-1",
            "Origin": "http://testserver",
            "Sec-Fetch-Site": "same-origin",
        },
    )

    assert missing_key.status_code == 422
    assert mismatched_key.status_code == 409
    assert cross_site.status_code == 403
    assert cross_site.json()["type"].endswith("cross-site-mutation")
    assert accepted.status_code == 200
    assert accepted.json()["request_id"] == "review-command-1"
    assert len(commands.response_calls) == 1

    memory_id = "d" * 64
    memory_request_id = f"memory-delete:{memory_id}"
    deleted = client.request(
        "DELETE",
        f"/api/v1/memory/{memory_id}",
        json={
            "request_id": memory_request_id,
            "confirm_memory_id": memory_id,
        },
        headers={"Idempotency-Key": memory_request_id},
    )
    assert deleted.status_code == 200
    assert deleted.json()["memory_id"] == memory_id
    assert len(commands.delete_calls) == 1

    document = client.get("/api/v1/openapi.json").json()
    assert set(document["paths"]["/api/v1/threads/{thread_id}/responses"]) == {"post"}
    assert set(document["paths"]["/api/v1/memory/{memory_id}"]) == {"delete"}
    serialized = str(document)
    assert "search_text" not in serialized
    assert "read_code" not in serialized


def test_built_spa_is_same_origin_and_hashed_assets_are_immutable() -> None:
    client = _client(_FakeDashboardService())

    index = client.get("/")
    assert index.status_code == 200
    assert "RepoCompass · Investigation Observatory" in index.text
    assert index.headers["cache-control"] == "no-store"
    asset_match = re.search(r'src="(/assets/[^"]+\.js)"', index.text)
    assert asset_match is not None

    asset = client.get(asset_match.group(1))
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset.headers["x-content-type-options"] == "nosniff"
    api_typo = client.get("/api/v1/typo")
    assert api_typo.status_code == 404
    assert api_typo.headers["content-type"].startswith("application/problem+json")
