"""Public deterministic evaluation and ablation contracts."""

from repocompass.evaluation.case_file import (
    MAX_CONTEXT_ABLATION_CASE_FILE_BYTES,
    load_context_ablation_case_file,
)
from repocompass.evaluation.context_ablation import (
    AblationVariant,
    ContextAblationCase,
    ContextAblationReport,
    ContextAblationReportSink,
    ContextAblationVariantMetrics,
    LatencyAvailability,
    create_context_ablation_case,
    render_context_ablation_summary,
    run_context_ablation,
)
from repocompass.evaluation.live_smoke import (
    LIVE_SMOKE_INPUT_TOKEN_CEILING,
    LiveProviderSmokeReport,
    LiveProviderSmokeReportSink,
    LiveProviderSmokeError,
    run_live_provider_smoke,
)
from repocompass.evaluation.memory_ablation import (
    MEMORY_ABLATION_SCHEMA_VERSION,
    MemoryAblationCase,
    MemoryAblationReport,
    MemoryAblationReportSink,
    MemoryAblationVariant,
    MemoryAblationVariantMetrics,
    create_memory_ablation_case,
    render_memory_ablation_summary,
    run_memory_ablation,
)
from repocompass.evaluation.memory_case_file import (
    MAX_MEMORY_ABLATION_CASE_FILE_BYTES,
    load_memory_ablation_case_file,
)

__all__ = [
    "AblationVariant",
    "ContextAblationCase",
    "ContextAblationReport",
    "ContextAblationReportSink",
    "ContextAblationVariantMetrics",
    "LatencyAvailability",
    "LIVE_SMOKE_INPUT_TOKEN_CEILING",
    "LiveProviderSmokeReport",
    "LiveProviderSmokeReportSink",
    "LiveProviderSmokeError",
    "MAX_CONTEXT_ABLATION_CASE_FILE_BYTES",
    "MAX_MEMORY_ABLATION_CASE_FILE_BYTES",
    "MEMORY_ABLATION_SCHEMA_VERSION",
    "MemoryAblationCase",
    "MemoryAblationReport",
    "MemoryAblationReportSink",
    "MemoryAblationVariant",
    "MemoryAblationVariantMetrics",
    "create_context_ablation_case",
    "create_memory_ablation_case",
    "load_context_ablation_case_file",
    "load_memory_ablation_case_file",
    "render_context_ablation_summary",
    "render_memory_ablation_summary",
    "run_context_ablation",
    "run_live_provider_smoke",
    "run_memory_ablation",
]
