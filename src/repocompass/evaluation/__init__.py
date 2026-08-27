"""Public deterministic evaluation and ablation contracts."""

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

__all__ = [
    "AblationVariant",
    "ContextAblationCase",
    "ContextAblationReport",
    "ContextAblationReportSink",
    "ContextAblationVariantMetrics",
    "LatencyAvailability",
    "create_context_ablation_case",
    "render_context_ablation_summary",
    "run_context_ablation",
]
