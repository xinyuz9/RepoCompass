"""Local FastAPI dashboard contract, error, host, and SSE tests."""

from __future__ import annotations

import re

import pytest
from repocompass.agent.conversation import (
    PublicConversationPage,
    PublicConversationTurn,
)
from fastapi.testclient import TestClient

from repocompass.agent.thread_view import ThreadExecutionState
from repocompass.agent.trace import TraceEventKind
from repocompass.dashboard.api import create_dashboard_app
from repocompass.dashboard.contracts import (
    DashboardAvailability,
    DashboardContinueCommand,
    DashboardHumanResponseCommand,
    DashboardLifecycleOperationStatus,
    DashboardMemoryCatalog,
    DashboardMemoryDeleteCommand,
    DashboardMemoryDeleteResult,
    DashboardRunDetail,
    DashboardSourceLine,
    DashboardSourceSnippet,
    DashboardResumeCommand,
    DashboardStartCommand,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    PublicTraceEvent,
)

from repocompass.dashboard.service import (
    DashboardNotFoundError,
    DashboardSnapshotConflictError,
)
from repocompass.memory.contracts import (
    MemoryAuditAction,
    MemoryAuditReason,
    MemoryScope,
    create_memory_audit_event,
    create_memory_tombstone,
)
from repocompass.service.contracts import (
    InvestigationOperationKind,
    InvestigationOperationResult,
    InvestigationOperationState,
)
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    RawStoreError,
)

SOURCE_ENTITY_ID = "d" * 64


def test_public_conversation_http_is_read_only_bounded_and_repository_scoped():
    client = TestClient(create_dashboard_app(_FakeDashboardService()))
    result = client.get("/api/v1/threads/thread-1/conversation?before_turn=2&limit=1")
    assert result.status_code == 200
    assert result.json()["turns"][0]["termination_status"] == "exhausted"
    assert "messages" not in result.text
    assert client.get("/api/v1/threads/other/conversation").status_code == 404
    assert (
        client.get("/api/v1/threads/thread-1/conversation?limit=21").status_code == 422
    )
    assert (
        client.get("/api/v1/threads/thread-1/conversation?before_turn=0").status_code
        == 422
    )
    assert (
        client.post("/api/v1/threads/thread-1/conversation", json={}).status_code == 405
    )


def _attempt(
    run_id: str = "run-1",
    *,
    status: AttemptStatus = AttemptStatus.COMPLETED,
) -> RawRunAttempt:
    return RawRunAttempt(
        run_id=run_id,
        thread_id="thread-1",
        turn_number=1,
        execution_mode=AttemptExecutionMode.START,
        repository_id="a" * 64,
        commit_sha="b" * 40,
        objective_digest="c" * 64,
        status=status,
        started_at_ms=1_000,
        finished_at_ms=None if status is AttemptStatus.RUNNING else 2_000,
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
        self.source_changed = False
        self.lifecycle_mutations_enabled = lifecycle_mutations_enabled
        self.memory_mutations_enabled = memory_mutations_enabled

    def system_info(self) -> DashboardSystemInfo:
        if self.fail_system:
            raise RuntimeError("private path: /home/operator/repository")
        return DashboardSystemInfo(
            product_version="0.1.0",
            repository_id="a" * 64,
            commit_sha="b" * 40,
            workspace_name="repository",
            memory_available=True,
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

    def get_conversation(self, thread_id: str, *, before_turn=None, limit=10):
        if thread_id != "thread-1":
            raise DashboardNotFoundError("missing thread")
        return PublicConversationPage(
            thread_id=thread_id,
            latest_turn_number=2,
            turns=(
                PublicConversationTurn(
                    turn_number=1,
                    objective="Public goal",
                    summary="Public final summary",
                    termination_status="exhausted",
                ),
            ),
        )

    def get_source(self, thread_id: str, entity_id: str) -> DashboardSourceSnippet:
        if self.source_changed:
            raise DashboardSnapshotConflictError("private source drift detail")
        if thread_id != "thread-1" or entity_id != SOURCE_ENTITY_ID:
            raise DashboardNotFoundError("missing source")
        return DashboardSourceSnippet(
            thread_id=thread_id,
            entity_id=entity_id,
            repository_id="a" * 64,
            commit_sha="b" * 40,
            path="src/auth.py",
            symbol="auth.login",
            source_digest="e" * 64,
            evidence_line_start=2,
            evidence_line_end=2,
            snippet_line_start=1,
            snippet_line_end=3,
            total_lines=3,
            truncated_before=False,
            truncated_after=False,
            lines=(
                DashboardSourceLine(number=1, text="def login():"),
                DashboardSourceLine(number=2, text="    return authenticate()"),
                DashboardSourceLine(number=3, text=""),
            ),
        )

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
        self.start_calls: list[DashboardStartCommand] = []
        self.resume_calls: list[tuple[str, DashboardResumeCommand]] = []
        self.continue_calls: list[tuple[str, DashboardContinueCommand]] = []
        self.response_calls: list[tuple[str, DashboardHumanResponseCommand]] = []
        self.delete_calls: list[tuple[str, DashboardMemoryDeleteCommand]] = []

    def _result(self, request_id: str, thread_id: str) -> InvestigationOperationResult:
        return InvestigationOperationResult(
            request_id=request_id,
            thread_id=thread_id,
            execution_state=ThreadExecutionState.RESUMABLE,
            run_id="lifecycle-run",
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

    def get_operation(self, request_id: str) -> DashboardLifecycleOperationStatus:
        if request_id != "operation-1":
            raise DashboardNotFoundError("missing lifecycle operation")
        return DashboardLifecycleOperationStatus(
            request_id=request_id,
            kind=InvestigationOperationKind.START,
            thread_id="thread-1",
            state=InvestigationOperationState.PENDING,
            reserved_at_ms=500,
            latest_attempt=_attempt("operation-run", status=AttemptStatus.RUNNING),
        )

    def start_investigation(
        self, command: DashboardStartCommand
    ) -> InvestigationOperationResult:
        self.start_calls.append(command)
        return self._result(command.request_id, "thread-started")

    def resume_thread(
        self, thread_id: str, command: DashboardResumeCommand
    ) -> InvestigationOperationResult:
        self.resume_calls.append((thread_id, command))
        return self._result(command.request_id, thread_id)

    def continue_thread(
        self, thread_id: str, command: DashboardContinueCommand
    ) -> InvestigationOperationResult:
        self.continue_calls.append((thread_id, command))
        return self._result(command.request_id, thread_id)

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


def test_source_api_returns_only_an_evidence_bound_numbered_window() -> None:
    client = _client(_FakeDashboardService())

    response = client.get(f"/api/v1/threads/thread-1/source/{SOURCE_ENTITY_ID}")
    invalid = client.get("/api/v1/threads/thread-1/source/not-a-digest")

    assert response.status_code == 200
    assert response.json()["path"] == "src/auth.py"
    assert response.json()["lines"][1] == {
        "number": 2,
        "text": "    return authenticate()",
    }
    assert "repository_root" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert invalid.status_code == 422


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
    service.source_changed = True
    stale_source = client.get(f"/api/v1/threads/thread-1/source/{SOURCE_ENTITY_ID}")
    service.source_changed = False
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
        (stale_source, 409),
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
    operation = client.get("/api/v1/operations/operation-1")
    missing_operation = client.get("/api/v1/operations/operation-missing")
    assert operation.status_code == 200
    assert operation.json()["state"] == "pending"
    assert operation.json()["latest_attempt"]["status"] == "running"
    assert missing_operation.status_code == 404

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

    snapshot = {
        "expected_repository_id": "a" * 64,
        "expected_commit_sha": "b" * 40,
    }
    started = client.post(
        "/api/v1/investigations",
        json={
            "schema_version": 1,
            "request_id": "start-command-1",
            **snapshot,
            "goal": "Trace authentication.",
            "focus_path": ".",
            "human_review_policy": "before_finalize",
            "memory_retrieval_enabled": True,
            "max_attempt_cost_usd": None,
        },
        headers={"Idempotency-Key": "start-command-1"},
    )
    unbound_start = client.post(
        "/api/v1/investigations",
        json={"request_id": "unbound-start", "goal": "Trace."},
        headers={"Idempotency-Key": "unbound-start"},
    )
    assert unbound_start.status_code == 422
    assert len(commands.start_calls) == 1
    resumed = client.post(
        "/api/v1/threads/thread-1/resume",
        json={"schema_version": 1, "request_id": "resume-command-1", **snapshot},
        headers={"Idempotency-Key": "resume-command-1"},
    )
    continued = client.post(
        "/api/v1/threads/thread-1/turns",
        json={
            "schema_version": 1,
            "request_id": "continue-command-1",
            "prompt": "Trace rejection.",
            "focus_path": "src",
            **snapshot,
        },
        headers={"Idempotency-Key": "continue-command-1"},
    )
    assert started.status_code == 200
    assert resumed.status_code == 200
    assert continued.status_code == 200
    assert len(commands.start_calls) == 1
    assert len(commands.resume_calls) == 1
    assert len(commands.continue_calls) == 1

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
    assert set(document["paths"]["/api/v1/investigations"]) == {"post"}
    assert set(document["paths"]["/api/v1/operations/{request_id}"]) == {"get"}
    assert set(document["paths"]["/api/v1/threads/{thread_id}/resume"]) == {"post"}
    assert set(document["paths"]["/api/v1/threads/{thread_id}/turns"]) == {"post"}
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
