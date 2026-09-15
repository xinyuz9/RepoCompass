"""Local-only HTTP projection for the RepoCompass investigation dashboard."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Protocol
from urllib.parse import urlsplit

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
)
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from repocompass.dashboard.contracts import (
    DashboardEventStreamCursor,
    DashboardContinueCommand,
    DashboardHumanResponseCommand,
    DashboardLifecycleOperationStatus,
    DashboardMemoryAudit,
    DashboardMemoryCatalog,
    DashboardMemoryDeleteCommand,
    DashboardMemoryDeleteResult,
    DashboardProblemDetail,
    DashboardRunDetail,
    DashboardSourceSnippet,
    DashboardStartCommand,
    DashboardResumeCommand,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    DashboardThreadDetail,
)
from repocompass.agent.conversation import PublicConversationPage
from repocompass.dashboard.cancellation import DashboardCancellationService
from repocompass.storage.cancellation import (
    CancellationError,
    CancellationRecord,
    CancellationRequest,
)
from repocompass.dashboard.commands import (
    DashboardCommandConflictError,
    DashboardCommandUnavailableError,
)
from repocompass.dashboard.service import (
    DashboardNotFoundError,
    DashboardServiceError,
    DashboardSnapshotConflictError,
)
from repocompass.memory.contracts import MemoryScope
from repocompass.memory.store import MemoryStoreError
from repocompass.storage.raw_store import RawStoreError
from repocompass.service.contracts import (
    InvestigationOperationResult,
    REQUEST_ID_PATTERN,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, SHA256_PATTERN

DASHBOARD_API_VERSION = "v1"
DASHBOARD_API_PREFIX = f"/api/{DASHBOARD_API_VERSION}"
DEFAULT_EVENT_PAGE_SIZE = 250
MAX_EVENT_PAGE_SIZE = 1_000
_ALLOWED_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testserver"})
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'; object-src 'none'; script-src 'self'; "
        "connect-src 'self'; font-src 'self'; style-src-elem 'self'; "
        "style-src-attr 'unsafe-inline'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class DashboardReadService(Protocol):
    """Read methods consumed by the HTTP adapter."""

    def system_info(self) -> DashboardSystemInfo: ...

    def list_threads(self, *, max_results: int = 50) -> DashboardThreadCatalog: ...

    def get_thread(self, thread_id: str) -> DashboardThreadDetail: ...

    def get_conversation(
        self, thread_id: str, *, before_turn: int | None = None, limit: int = 10
    ) -> PublicConversationPage: ...

    def get_source(
        self,
        thread_id: str,
        entity_id: str,
    ) -> DashboardSourceSnippet: ...

    def get_run(self, run_id: str) -> DashboardRunDetail: ...

    def list_memory(
        self,
        *,
        max_results: int = 50,
        memory_scope: MemoryScope | None = None,
    ) -> DashboardMemoryCatalog: ...

    def get_memory_audit(self, memory_id: str) -> DashboardMemoryAudit: ...


class DashboardMutationService(Protocol):
    def get_operation(self, request_id: str) -> DashboardLifecycleOperationStatus: ...

    def start_investigation(
        self, command: DashboardStartCommand
    ) -> InvestigationOperationResult: ...

    def resume_thread(
        self, thread_id: str, command: DashboardResumeCommand
    ) -> InvestigationOperationResult: ...

    def continue_thread(
        self, thread_id: str, command: DashboardContinueCommand
    ) -> InvestigationOperationResult: ...

    @property
    def lifecycle_mutations_enabled(self) -> bool: ...

    @property
    def memory_mutations_enabled(self) -> bool: ...

    def respond_to_review(
        self,
        thread_id: str,
        command: DashboardHumanResponseCommand,
    ) -> InvestigationOperationResult: ...

    def delete_memory(
        self,
        memory_id: str,
        command: DashboardMemoryDeleteCommand,
    ) -> DashboardMemoryDeleteResult: ...


def create_dashboard_app(
    service: DashboardReadService,
    *,
    commands: DashboardMutationService | None = None,
    cancellation: DashboardCancellationService | None = None,
    static_directory: Path | None = None,
) -> FastAPI:
    """Create a local API with optional, explicitly injected command routes."""

    required_methods = (
        "system_info",
        "list_threads",
        "get_thread",
        "get_source",
        "get_run",
        "list_memory",
        "get_memory_audit",
    )
    if not all(callable(getattr(service, method, None)) for method in required_methods):
        raise TypeError("service does not implement the dashboard read boundary")
    capabilities = service.system_info()
    if capabilities.cancellation_enabled != (cancellation is not None):
        raise ValueError("cancellation capability does not reconcile")
    if cancellation is not None and (
        commands is None or not commands.lifecycle_mutations_enabled
    ):
        raise ValueError("cancellation requires lifecycle commands")
    if commands is None:
        if (
            capabilities.lifecycle_mutations_enabled
            or capabilities.memory_mutations_enabled
        ):
            raise ValueError("dashboard capabilities require a command service")
    else:
        command_methods = (
            "get_operation",
            "start_investigation",
            "resume_thread",
            "continue_thread",
            "respond_to_review",
            "delete_memory",
        )
        if not all(callable(getattr(commands, name, None)) for name in command_methods):
            raise TypeError(
                "commands does not implement the dashboard mutation boundary"
            )
        if (
            commands.lifecycle_mutations_enabled
            != capabilities.lifecycle_mutations_enabled
            or commands.memory_mutations_enabled
            != capabilities.memory_mutations_enabled
        ):
            raise ValueError("dashboard command capabilities do not reconcile")

    app = FastAPI(
        title="RepoCompass Local Investigation API",
        summary="Repository-bound investigation replay and observability.",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url=f"{DASHBOARD_API_PREFIX}/openapi.json",
    )
    resolved_static = (
        Path(__file__).with_name("static")
        if static_directory is None
        else static_directory.expanduser().resolve()
    )
    static_index = resolved_static / "index.html"
    static_assets = resolved_static / "assets"
    static_available = static_index.is_file() and static_assets.is_dir()
    if static_available:
        app.mount(
            "/assets",
            StaticFiles(directory=static_assets, check_dir=True),
            name="dashboard-assets",
        )

    @app.middleware("http")
    async def local_host_and_security_headers(request: Request, call_next):
        host = _host_without_port(request.headers.get("host", ""))
        if host not in _ALLOWED_LOCAL_HOSTS:
            response = _problem_response(
                status=400,
                code="invalid-host",
                title="Invalid host",
                detail="The RepoCompass dashboard accepts local Host headers only.",
                instance=request.url.path,
            )
        elif _is_api_mutation(request) and not _mutation_origin_allowed(request, host):
            response = _problem_response(
                status=403,
                code="cross-site-mutation",
                title="Mutation rejected",
                detail="Dashboard mutations require a same-origin local request.",
                instance=request.url.path,
            )
        else:
            response = await call_next(request)
        if request.url.path.startswith("/assets/") and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers.setdefault("Cache-Control", "no-store")
        for name, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.exception_handler(DashboardNotFoundError)
    async def dashboard_not_found(request: Request, _: DashboardNotFoundError):
        return _problem_response(
            status=404,
            code="not-found",
            title="Resource not found",
            detail="The requested repository-bound dashboard resource was not found.",
            instance=request.url.path,
        )

    @app.exception_handler(DashboardCommandConflictError)
    async def dashboard_command_conflict(
        request: Request,
        _: DashboardCommandConflictError,
    ):
        return _problem_response(
            status=409,
            code="mutation-conflict",
            title="Mutation conflict",
            detail="The command conflicts with current state or prior identity.",
            instance=request.url.path,
        )

    @app.exception_handler(DashboardSnapshotConflictError)
    async def dashboard_snapshot_conflict(
        request: Request,
        _: DashboardSnapshotConflictError,
    ):
        return _problem_response(
            status=409,
            code="source-snapshot-conflict",
            title="Evidence source changed",
            detail=(
                "The source file no longer matches the persisted evidence snapshot."
            ),
            instance=request.url.path,
        )

    @app.exception_handler(CancellationError)
    @app.exception_handler(DashboardCommandUnavailableError)
    async def dashboard_command_unavailable(
        request: Request,
        _: DashboardCommandUnavailableError,
    ):
        return _problem_response(
            status=503,
            code="mutation-unavailable",
            title="Mutation unavailable",
            detail="The local mutation could not be completed safely.",
            instance=request.url.path,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_failed(
        request: Request,
        _: RequestValidationError,
    ):
        return _problem_response(
            status=422,
            code="invalid-request",
            title="Invalid request",
            detail="The request does not satisfy the bounded dashboard contract.",
            instance=request.url.path,
        )

    @app.exception_handler(ValidationError)
    @app.exception_handler(ResponseValidationError)
    async def response_validation_failed(
        request: Request,
        _: ValidationError | ResponseValidationError,
    ):
        return _problem_response(
            status=500,
            code="invalid-persisted-data",
            title="Dashboard data unavailable",
            detail="A persisted dashboard projection failed integrity validation.",
            instance=request.url.path,
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, _: ValueError):
        return _problem_response(
            status=422,
            code="invalid-request",
            title="Invalid request",
            detail="The request does not satisfy the bounded dashboard contract.",
            instance=request.url.path,
        )

    @app.exception_handler(DashboardServiceError)
    async def dashboard_service_failed(request: Request, _: DashboardServiceError):
        return _problem_response(
            status=503,
            code="dashboard-unavailable",
            title="Dashboard data unavailable",
            detail="The local dashboard store could not complete this request.",
            instance=request.url.path,
        )

    async def dashboard_storage_failed(request: Request, _: Exception):
        return _problem_response(
            status=503,
            code="dashboard-unavailable",
            title="Dashboard data unavailable",
            detail="The local dashboard store could not complete this request.",
            instance=request.url.path,
        )

    for storage_error in (MemoryStoreError, RawStoreError, sqlite3.Error, OSError):
        app.add_exception_handler(storage_error, dashboard_storage_failed)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        status = error.status_code
        return _problem_response(
            status=status,
            code="not-found" if status == 404 else "http-error",
            title="Resource not found" if status == 404 else "Request rejected",
            detail=(
                "The requested dashboard route was not found."
                if status == 404
                else "The dashboard rejected this HTTP request."
            ),
            instance=request.url.path,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, _: Exception):
        return _problem_response(
            status=500,
            code="internal-error",
            title="Dashboard request failed",
            detail="The local dashboard encountered an unexpected error.",
            instance=request.url.path,
        )

    @app.get(f"{DASHBOARD_API_PREFIX}/system", response_model=DashboardSystemInfo)
    def get_system_info() -> DashboardSystemInfo:
        return service.system_info()

    @app.get(
        f"{DASHBOARD_API_PREFIX}/threads",
        response_model=DashboardThreadCatalog,
    )
    def list_threads(
        max_results: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> DashboardThreadCatalog:
        return service.list_threads(max_results=max_results)

    @app.get(
        f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}",
        response_model=DashboardThreadDetail,
    )
    def get_thread(thread_id: str) -> DashboardThreadDetail:
        return service.get_thread(thread_id)

    @app.get(
        f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}/conversation",
        response_model=PublicConversationPage,
    )
    def get_conversation(
        thread_id: str,
        before_turn: Annotated[
            int | None, Query(ge=1, le=MAX_SAFE_JSON_INTEGER)
        ] = None,
        limit: Annotated[int, Query(ge=1, le=20)] = 10,
    ) -> PublicConversationPage:
        reader = getattr(service, "get_conversation", None)
        if not callable(reader):
            raise DashboardServiceError("public conversation reader is unavailable")
        return reader(thread_id, before_turn=before_turn, limit=limit)

    @app.get(
        f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}/source/{{entity_id}}",
        response_model=DashboardSourceSnippet,
    )
    def get_source(
        thread_id: str,
        entity_id: Annotated[str, PathParameter(pattern=SHA256_PATTERN)],
    ) -> DashboardSourceSnippet:
        return service.get_source(thread_id, entity_id)

    @app.get(
        f"{DASHBOARD_API_PREFIX}/runs/{{run_id}}",
        response_model=DashboardRunDetail,
    )
    def get_run(run_id: str) -> DashboardRunDetail:
        return service.get_run(run_id)

    @app.get(f"{DASHBOARD_API_PREFIX}/runs/{{run_id}}/events")
    def stream_run_events(
        run_id: str,
        after_sequence: Annotated[
            int,
            Query(ge=-1, le=MAX_SAFE_JSON_INTEGER),
        ] = -1,
        max_events: Annotated[
            int,
            Query(ge=1, le=MAX_EVENT_PAGE_SIZE),
        ] = DEFAULT_EVENT_PAGE_SIZE,
    ) -> StreamingResponse:
        detail = service.get_run(run_id)
        pending = tuple(
            event for event in detail.trace_events if event.sequence > after_sequence
        )
        emitted = pending[:max_events]
        next_sequence = None if not emitted else emitted[-1].sequence
        cursor = DashboardEventStreamCursor(
            availability=detail.trace_availability,
            after_sequence=after_sequence,
            emitted_count=len(emitted),
            next_after_sequence=next_sequence,
            truncated=len(pending) > len(emitted),
        )

        async def event_stream() -> AsyncIterator[str]:
            for event in emitted:
                yield _encode_sse(
                    event="trace",
                    event_id=str(event.sequence),
                    data=event.model_dump_json(),
                )
            yield _encode_sse(
                event="snapshot_end",
                data=cursor.model_dump_json(),
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
                "X-RepoCompass-Trace-Availability": detail.trace_availability.value,
            },
        )

    @app.get(
        f"{DASHBOARD_API_PREFIX}/memory",
        response_model=DashboardMemoryCatalog,
    )
    def list_memory(
        max_results: Annotated[int, Query(ge=1, le=50)] = 50,
        memory_scope: MemoryScope | None = None,
    ) -> DashboardMemoryCatalog:
        return service.list_memory(
            max_results=max_results,
            memory_scope=memory_scope,
        )

    @app.get(
        f"{DASHBOARD_API_PREFIX}/memory/{{memory_id}}/audit",
        response_model=DashboardMemoryAudit,
    )
    def get_memory_audit(memory_id: str) -> DashboardMemoryAudit:
        return service.get_memory_audit(memory_id)

    if commands is not None and commands.lifecycle_mutations_enabled:

        def _require_idempotency(command_request_id: str, header: str) -> None:
            if header != command_request_id:
                raise DashboardCommandConflictError(
                    "idempotency header does not match command"
                )

        @app.get(
            f"{DASHBOARD_API_PREFIX}/operations/{{request_id}}",
            response_model=DashboardLifecycleOperationStatus,
        )
        def get_operation(request_id: str) -> DashboardLifecycleOperationStatus:
            return commands.get_operation(request_id)

        @app.post(
            f"{DASHBOARD_API_PREFIX}/investigations",
            response_model=InvestigationOperationResult,
        )
        def start_investigation(
            command: DashboardStartCommand,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> InvestigationOperationResult:
            _require_idempotency(command.request_id, idempotency_key)
            return commands.start_investigation(command)

        @app.post(
            f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}/resume",
            response_model=InvestigationOperationResult,
        )
        def resume_thread(
            thread_id: str,
            command: DashboardResumeCommand,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> InvestigationOperationResult:
            _require_idempotency(command.request_id, idempotency_key)
            return commands.resume_thread(thread_id, command)

        @app.post(
            f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}/turns",
            response_model=InvestigationOperationResult,
        )
        def continue_thread(
            thread_id: str,
            command: DashboardContinueCommand,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> InvestigationOperationResult:
            _require_idempotency(command.request_id, idempotency_key)
            return commands.continue_thread(thread_id, command)

        @app.post(
            f"{DASHBOARD_API_PREFIX}/threads/{{thread_id}}/responses",
            response_model=InvestigationOperationResult,
        )
        def respond_to_review(
            thread_id: str,
            command: DashboardHumanResponseCommand,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> InvestigationOperationResult:
            _require_idempotency(command.request_id, idempotency_key)
            return commands.respond_to_review(thread_id, command)

    if cancellation is not None:

        @app.get(
            f"{DASHBOARD_API_PREFIX}/runs/{{run_id}}/cancellation",
            response_model=CancellationRecord,
        )
        def get_cancellation(
            run_id: Annotated[str, PathParameter(pattern=REQUEST_ID_PATTERN)],
        ) -> CancellationRecord:
            return cancellation.get(run_id)

        @app.post(
            f"{DASHBOARD_API_PREFIX}/runs/{{run_id}}/cancellation",
            response_model=CancellationRecord,
        )
        def request_cancellation(
            run_id: Annotated[str, PathParameter(pattern=REQUEST_ID_PATTERN)],
            command: CancellationRequest,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> CancellationRecord:
            if idempotency_key != command.request_id:
                raise DashboardCommandConflictError(
                    "idempotency header does not match cancellation"
                )
            return cancellation.request(run_id, command)

    if commands is not None and commands.memory_mutations_enabled:

        @app.delete(
            f"{DASHBOARD_API_PREFIX}/memory/{{memory_id}}",
            response_model=DashboardMemoryDeleteResult,
        )
        def delete_memory(
            memory_id: str,
            command: DashboardMemoryDeleteCommand,
            idempotency_key: Annotated[
                str,
                Header(
                    alias="Idempotency-Key",
                    pattern=REQUEST_ID_PATTERN,
                    min_length=1,
                    max_length=128,
                ),
            ],
        ) -> DashboardMemoryDeleteResult:
            if idempotency_key != command.request_id:
                raise DashboardCommandConflictError(
                    "idempotency header does not match command"
                )
            return commands.delete_memory(memory_id, command)

    @app.get("/", include_in_schema=False)
    def get_dashboard_index():
        if not static_available:
            raise HTTPException(status_code=404)
        return FileResponse(static_index)

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def get_dashboard_route(frontend_path: str):
        if frontend_path == "api" or frontend_path.startswith("api/"):
            raise HTTPException(status_code=404)
        if not static_available:
            raise HTTPException(status_code=404)
        return FileResponse(static_index)

    return app


def _problem_response(
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    instance: str,
) -> JSONResponse:
    problem = DashboardProblemDetail(
        type=f"urn:repocompass:problem:{code}",
        title=title,
        status=status,
        detail=detail,
        instance=instance,
    )
    return JSONResponse(
        status_code=status,
        content=problem.model_dump(mode="json"),
        media_type="application/problem+json",
    )


def _encode_sse(*, event: str, data: str, event_id: str | None = None) -> str:
    fields: list[str] = []
    if event_id is not None:
        fields.append(f"id: {event_id}")
    fields.append(f"event: {event}")
    for line in data.splitlines() or [""]:
        fields.append(f"data: {line}")
    return "\n".join(fields) + "\n\n"


def _host_without_port(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized.startswith("["):
        closing = normalized.find("]")
        return "" if closing < 0 else normalized[1:closing]
    if normalized.count(":") == 1:
        return normalized.rsplit(":", maxsplit=1)[0]
    return normalized


def _is_api_mutation(request: Request) -> bool:
    return request.method in {"POST", "PUT", "PATCH", "DELETE"} and (
        request.url.path.startswith(f"{DASHBOARD_API_PREFIX}/")
    )


def _mutation_origin_allowed(request: Request, host: str) -> bool:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site not in {None, "none", "same-origin"}:
        return False
    origin = request.headers.get("origin")
    if origin is None:
        return True
    parsed = urlsplit(origin)
    if (
        parsed.scheme.casefold() != request.url.scheme.casefold()
        or parsed.hostname is None
        or parsed.hostname.casefold() != host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    default_port = 443 if parsed.scheme.casefold() == "https" else 80
    return (origin_port or default_port) == (request.url.port or default_port)
