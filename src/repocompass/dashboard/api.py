"""Local-only HTTP projection for the RepoCompass investigation dashboard."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from repocompass.dashboard.contracts import (
    DashboardEventStreamCursor,
    DashboardMemoryAudit,
    DashboardMemoryCatalog,
    DashboardProblemDetail,
    DashboardRunDetail,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    DashboardThreadDetail,
)
from repocompass.dashboard.service import (
    DashboardNotFoundError,
    DashboardServiceError,
)
from repocompass.memory.contracts import MemoryScope
from repocompass.memory.store import MemoryStoreError
from repocompass.storage.raw_store import RawStoreError
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER

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

    def get_run(self, run_id: str) -> DashboardRunDetail: ...

    def list_memory(
        self,
        *,
        max_results: int = 50,
        memory_scope: MemoryScope | None = None,
    ) -> DashboardMemoryCatalog: ...

    def get_memory_audit(self, memory_id: str) -> DashboardMemoryAudit: ...


def create_dashboard_app(
    service: DashboardReadService,
    *,
    static_directory: Path | None = None,
) -> FastAPI:
    """Create a same-origin read API over one injected repository-bound service."""

    required_methods = (
        "system_info",
        "list_threads",
        "get_thread",
        "get_run",
        "list_memory",
        "get_memory_audit",
    )
    if not all(callable(getattr(service, method, None)) for method in required_methods):
        raise TypeError("service does not implement the dashboard read boundary")

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
