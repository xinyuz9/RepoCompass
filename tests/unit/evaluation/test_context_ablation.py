"""Deterministic Active Context Builder ablation contracts."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.agent.context import (
    ContextSection,
    ContextSourceKind,
    ProvenanceKind,
    ProvenanceLink,
    RawArtifactKind,
    RepositoryLocation,
    VerificationStatus,
    create_context_item,
    create_context_source,
    create_raw_artifact_metadata,
)
from repocompass.agent.context_builder import estimate_context_item_tokens
from repocompass.agent.cost import (
    CostAvailability,
    CostUnavailableReason,
    ModelPrice,
    create_pricing_table,
)
from repocompass.evaluation.context_ablation import (
    AblationVariant,
    ContextAblationReport,
    ContextAblationReportSink,
    LatencyAvailability,
    create_context_ablation_case,
    render_context_ablation_summary,
    run_context_ablation,
)

REPOSITORY_ID = "a" * 64
COMMIT_SHA = "b" * 40


def _objective():
    source = create_context_source(
        kind=ContextSourceKind.OBJECTIVE,
        provenance=(ProvenanceLink(kind=ProvenanceKind.OBJECTIVE, id="objective"),),
    )
    return create_context_item(
        source=source,
        section=ContextSection.OBJECTIVE,
        verification_status=VerificationStatus.HUMAN_APPROVED,
        content="Locate the authentication entry point.",
        estimated_tokens=1,
        priority=100,
        pinned=True,
    )


def _evidence(label: str, *, required: bool, content_repetitions: int = 1):
    content = (f"src/{label}.py defines {label}.\n" * content_repetitions).strip()
    artifact = create_raw_artifact_metadata(
        kind=RawArtifactKind.SOURCE_EXCERPT,
        media_type="text/plain",
        content=content,
    )
    source = create_context_source(
        kind=ContextSourceKind.SOURCE_EXCERPT,
        raw_artifact_id=artifact.artifact_id,
        location=RepositoryLocation(
            repository_id=REPOSITORY_ID,
            commit_sha=COMMIT_SHA,
            path=f"src/{label}.py",
            line_start=1,
            line_end=max(1, content_repetitions),
            symbol=label,
        ),
        provenance=(
            ProvenanceLink(kind=ProvenanceKind.PLAN_STEP, id="locate-auth"),
            ProvenanceLink(kind=ProvenanceKind.TOOL_CALL, id=f"read-{label}"),
        ),
    )
    return create_context_item(
        source=source,
        section=ContextSection.EVIDENCE,
        verification_status=(
            VerificationStatus.VERIFIED if required else VerificationStatus.UNVERIFIED
        ),
        content=content,
        estimated_tokens=1,
        priority=100 if required else 1,
        pinned=False,
    )


def _fixture_case():
    objective = _objective()
    required = _evidence("authenticate", required=True)
    noise = _evidence(
        "unrelated_generated_client",
        required=False,
        content_repetitions=80,
    )
    budget = estimate_context_item_tokens(objective) + estimate_context_item_tokens(
        required
    )
    case = create_context_ablation_case(
        name="context-pressure-v1",
        budget_tokens=budget,
        candidates=(noise, required, objective),
        required_evidence_item_ids=(required.item_id,),
    )
    return case, objective, required, noise


def _pricing(*, model: str = "gpt-ablation"):
    return create_pricing_table(
        version="ablation-pricing-v1",
        effective_from="2026-08-27",
        entries=(
            ModelPrice(
                provider="openai",
                model=model,
                input_usd_per_million_tokens="10",
                output_usd_per_million_tokens="30",
            ),
        ),
    )


def test_case_is_canonical_content_addressed_and_order_independent() -> None:
    case, objective, required, noise = _fixture_case()
    reordered = create_context_ablation_case(
        name=case.name,
        budget_tokens=case.budget_tokens,
        candidates=(objective, noise, required),
        required_evidence_item_ids=(required.item_id,),
    )

    assert case == reordered
    assert len(case.case_id) == 64
    assert [item.item_id for item in case.candidates] == sorted(
        item.item_id for item in case.candidates
    )
    payload = case.model_dump(mode="json")
    with pytest.raises(ValidationError, match="case_id"):
        type(case).model_validate_json(json.dumps({**payload, "case_id": "f" * 64}))


def test_case_rejects_required_evidence_without_traceable_source() -> None:
    objective = _objective()

    with pytest.raises(ValueError, match="required evidence"):
        create_context_ablation_case(
            name="invalid-required",
            budget_tokens=estimate_context_item_tokens(objective),
            candidates=(objective,),
            required_evidence_item_ids=(objective.item_id,),
        )


def test_ablation_compares_budgeted_context_with_all_context_baseline() -> None:
    case, objective, required, noise = _fixture_case()
    pricing = _pricing()

    report = run_context_ablation(
        case,
        pricing_table=pricing,
        provider="openai",
        model="gpt-ablation",
        observed_latency_ms={
            AblationVariant.ACTIVE_CONTEXT: 3,
            AblationVariant.ALL_CONTEXT_BASELINE: 5,
        },
    )
    repeated = run_context_ablation(
        case,
        pricing_table=pricing,
        provider="openai",
        model="gpt-ablation",
        observed_latency_ms={
            AblationVariant.ACTIVE_CONTEXT: 3,
            AblationVariant.ALL_CONTEXT_BASELINE: 5,
        },
    )
    metrics = {item.variant: item for item in report.variants}
    active = metrics[AblationVariant.ACTIVE_CONTEXT]
    baseline = metrics[AblationVariant.ALL_CONTEXT_BASELINE]

    assert report == repeated
    assert report.case_id == case.case_id
    assert report.pricing_table_id == pricing.table_id
    assert len(report.report_id) == 64
    assert active.selected_item_ids == (objective.item_id, required.item_id)
    assert noise.item_id not in active.selected_item_ids
    assert baseline.selected_item_ids == tuple(item.item_id for item in case.candidates)
    assert active.selected_tokens <= case.budget_tokens
    assert active.overflow_tokens == 0
    assert baseline.selected_tokens > case.budget_tokens
    assert baseline.overflow_tokens == baseline.selected_tokens - case.budget_tokens
    assert active.required_evidence_retention_ppm == 1_000_000
    assert baseline.required_evidence_retention_ppm == 1_000_000
    assert active.evidence_provenance_coverage_ppm == 1_000_000
    assert baseline.evidence_provenance_coverage_ppm == 1_000_000
    assert report.token_reduction == baseline.selected_tokens - active.selected_tokens
    assert report.token_reduction > 0
    assert active.cost_availability is CostAvailability.ESTIMATED
    assert baseline.cost_availability is CostAvailability.ESTIMATED
    assert active.estimated_input_cost_usd is not None
    assert baseline.estimated_input_cost_usd is not None
    assert report.estimated_input_cost_savings_usd == str(
        Decimal(baseline.estimated_input_cost_usd)
        - Decimal(active.estimated_input_cost_usd)
    )
    assert Decimal(report.estimated_input_cost_savings_usd) > 0
    assert active.latency_availability is LatencyAvailability.OBSERVED
    assert active.latency_ms == 3
    assert baseline.latency_ms == 5
    assert ContextAblationReport.model_validate_json(report.model_dump_json()) == report


def test_ablation_preserves_unknown_price_and_unmeasured_latency() -> None:
    case, _, _, _ = _fixture_case()

    report = run_context_ablation(
        case,
        pricing_table=_pricing(model="another-model"),
        provider="openai",
        model="gpt-ablation",
    )

    for metrics in report.variants:
        assert metrics.cost_availability is CostAvailability.UNAVAILABLE
        assert metrics.cost_unavailable_reason is CostUnavailableReason.PRICE_NOT_FOUND
        assert metrics.estimated_input_cost_usd is None
        assert metrics.latency_availability is LatencyAvailability.NOT_MEASURED
        assert metrics.latency_ms is None
    assert report.estimated_input_cost_savings_usd is None


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    (
        (("variants", 0, "estimated_input_cost_usd"), "not-a-decimal"),
        (("estimated_input_cost_savings_usd",), "not-a-decimal"),
    ),
)
def test_report_rejects_invalid_decimal_as_validation_error(
    field_path: tuple[str | int, ...],
    invalid_value: str,
) -> None:
    case, _, _, _ = _fixture_case()
    report = run_context_ablation(
        case,
        pricing_table=_pricing(),
        provider="openai",
        model="gpt-ablation",
    )
    payload = report.model_dump(mode="json")
    target = payload
    for component in field_path[:-1]:
        target = target[component]
    target[field_path[-1]] = invalid_value

    with pytest.raises(ValidationError, match="canonical decimal"):
        ContextAblationReport.model_validate_json(json.dumps(payload))


def test_summary_and_sink_publish_bounded_machine_readable_artifact(
    tmp_path: Path,
) -> None:
    case, _, _, noise = _fixture_case()
    report = run_context_ablation(
        case,
        pricing_table=_pricing(),
        provider="openai",
        model="gpt-ablation",
    )
    destination = tmp_path / "nested" / "context-ablation.json"

    written = ContextAblationReportSink().write(report, destination)
    payload = json.loads(written.read_text(encoding="utf-8"))
    summary = render_context_ablation_summary(report)

    assert written == destination.resolve()
    assert payload == report.model_dump(mode="json")
    assert noise.content not in written.read_text(encoding="utf-8")
    assert "all_context_baseline" in summary
    assert "estimate, not provider invoice" in summary
    assert str(report.token_reduction) in summary
    with pytest.raises(FileExistsError):
        ContextAblationReportSink().write(report, destination)
