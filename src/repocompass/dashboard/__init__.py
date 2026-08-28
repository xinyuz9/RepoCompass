"""Local, repository-bound dashboard contracts and services."""

from repocompass.dashboard.contracts import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardAvailability,
    DashboardMemoryAudit,
    DashboardMemoryCatalog,
    DashboardRunDetail,
    DashboardRunSummary,
    DashboardSystemInfo,
    DashboardThreadCatalog,
    DashboardThreadDetail,
    DashboardThreadSummary,
    EvaluatorReplayFrame,
    PlanReplayFrame,
    PublicTraceEvent,
)
from repocompass.dashboard.service import (
    DashboardNotFoundError,
    DashboardQueryService,
    DashboardServiceError,
)
from repocompass.dashboard.trace_store import (
    JsonlTraceStore,
    TraceStoreCorruptionError,
    TraceStoreError,
)

__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "DashboardAvailability",
    "DashboardMemoryAudit",
    "DashboardMemoryCatalog",
    "DashboardNotFoundError",
    "DashboardQueryService",
    "DashboardRunDetail",
    "DashboardRunSummary",
    "DashboardServiceError",
    "DashboardSystemInfo",
    "DashboardThreadCatalog",
    "DashboardThreadDetail",
    "DashboardThreadSummary",
    "EvaluatorReplayFrame",
    "JsonlTraceStore",
    "PlanReplayFrame",
    "PublicTraceEvent",
    "TraceStoreCorruptionError",
    "TraceStoreError",
]
