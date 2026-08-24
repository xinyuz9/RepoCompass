"""Versioned pricing and deterministic attempt cost-report contracts."""

import json

import pytest
from pydantic import ValidationError

from repocompass.agent.cost import (
    AttemptUsageCostReport,
    BreakdownDimension,
    CostAvailability,
    CostBudgetReason,
    CostBudgetStatus,
    CostCoverage,
    CostUnavailableReason,
    MetricCoverage,
    ModelPrice,
    PricingTable,
    build_attempt_usage_cost_report,
    create_pricing_table,
    estimate_invocation_cost,
    validate_cost_budget_usd,
)
from repocompass.agent.usage import (
    ModelInvocationRecord,
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    UsageRecord,
    create_model_invocation_record,
)


def _price(
    *,
    provider: str = "openai",
    model: str = "gpt-test",
    cached: str = "same_as_input",
    reasoning: str = "same_as_output",
) -> ModelPrice:
    return ModelPrice(
        provider=provider,
        model=model,
        input_usd_per_million_tokens="2.5",
        output_usd_per_million_tokens="10",
        cached_input_usd_per_million_tokens=cached,
        reasoning_output_usd_per_million_tokens=reasoning,
    )


def _table(*entries: ModelPrice) -> PricingTable:
    return create_pricing_table(
        version="test-2026-08-24",
        effective_from="2026-08-24",
        entries=entries or (_price(),),
    )


def _usage(
    *,
    input_tokens: int | None = 1_000,
    output_tokens: int | None = 100,
    total_tokens: int | None = 1_100,
    cached_input_tokens: int | None = 400,
    reasoning_output_tokens: int | None = 20,
) -> UsageRecord:
    return UsageRecord(
        availability=UsageAvailability.REPORTED,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )


def _invocation(
    *,
    sequence: int = 0,
    node_name: str = "plan_investigation",
    role: ModelRole = ModelRole.PLANNER,
    provider: str = "openai",
    model: str = "gpt-test",
    status: ModelInvocationStatus = ModelInvocationStatus.SUCCEEDED,
    usage: UsageRecord | None = None,
    latency_ms: int = 10,
) -> ModelInvocationRecord:
    failed = status is ModelInvocationStatus.FAILED
    return create_model_invocation_record(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        sequence=sequence,
        node_name=node_name,
        role=role,
        provider=provider,
        model=model,
        status=status,
        latency_ms=latency_ms,
        usage=usage or _usage(),
        input_artifact_id=chr(ord("a") + sequence) * 64,
        output_artifact_id=None if failed else chr(ord("d") + sequence) * 64,
        context_pack_id=f"{sequence + 1:064x}",
        error_code="model_call_failed" if failed else None,
    )


def test_pricing_table_is_canonical_content_addressed_snapshot() -> None:
    openai = _price()
    deepseek = _price(provider="deepseek", model="deepseek-test")

    first = _table(openai, deepseek)
    reordered = _table(deepseek, openai)

    assert first == reordered
    assert first.table_id == reordered.table_id
    assert [entry.provider for entry in first.entries] == ["deepseek", "openai"]
    assert first.lookup("openai", "gpt-test") == openai
    assert first.lookup("openai", "GPT-TEST") is None
    assert first.lookup("unknown", "gpt-test") is None
    assert PricingTable.model_validate_json(first.model_dump_json()) == first

    payload = first.model_dump(mode="json")
    with pytest.raises(ValidationError, match="table_id"):
        PricingTable.model_validate_json(json.dumps({**payload, "table_id": "f" * 64}))


@pytest.mark.parametrize("rate", ["01", "1.0", "1e-3", "-1", "NaN", "inf"])
def test_pricing_rates_require_canonical_non_negative_decimal_strings(
    rate: str,
) -> None:
    with pytest.raises(ValidationError, match="rate"):
        ModelPrice(
            provider="openai",
            model="gpt-test",
            input_usd_per_million_tokens=rate,
            output_usd_per_million_tokens="10",
        )


def test_pricing_table_rejects_duplicate_exact_model_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _table(_price(), _price())


def test_estimate_applies_explicit_cached_and_reasoning_tiers_exactly() -> None:
    invocation = _invocation()
    pricing = _table(_price(cached="1.25", reasoning="15"))

    estimate = estimate_invocation_cost(invocation, pricing)

    assert estimate.availability is CostAvailability.ESTIMATED
    assert estimate.input_cost_usd == "0.002"
    assert estimate.output_cost_usd == "0.0011"
    assert estimate.estimated_cost_usd == "0.0031"
    assert estimate.unavailable_reason is None
    assert estimate.disclaimer == "estimate_not_invoice"


def test_same_as_base_tiers_do_not_require_optional_usage_dimensions() -> None:
    invocation = _invocation(
        usage=_usage(
            input_tokens=1_000,
            output_tokens=100,
            total_tokens=1_100,
            cached_input_tokens=None,
            reasoning_output_tokens=None,
        )
    )

    estimate = estimate_invocation_cost(invocation, _table())

    assert estimate.availability is CostAvailability.ESTIMATED
    assert estimate.input_cost_usd == "0.0025"
    assert estimate.output_cost_usd == "0.001"
    assert estimate.estimated_cost_usd == "0.0035"


@pytest.mark.parametrize(
    ("invocation", "pricing", "reason"),
    [
        (
            _invocation(usage=UsageRecord(availability=UsageAvailability.UNAVAILABLE)),
            _table(),
            CostUnavailableReason.USAGE_UNAVAILABLE,
        ),
        (
            _invocation(
                usage=_usage(
                    input_tokens=None,
                    output_tokens=100,
                    total_tokens=100,
                    cached_input_tokens=None,
                    reasoning_output_tokens=20,
                )
            ),
            _table(),
            CostUnavailableReason.INPUT_TOKENS_UNAVAILABLE,
        ),
        (
            _invocation(model="unpriced"),
            _table(),
            CostUnavailableReason.PRICE_NOT_FOUND,
        ),
        (
            _invocation(
                usage=_usage(
                    cached_input_tokens=None,
                    reasoning_output_tokens=20,
                )
            ),
            _table(_price(cached="1.25")),
            CostUnavailableReason.CACHED_INPUT_TOKENS_UNAVAILABLE,
        ),
        (
            _invocation(
                usage=_usage(
                    cached_input_tokens=400,
                    reasoning_output_tokens=None,
                )
            ),
            _table(_price(reasoning="15")),
            CostUnavailableReason.REASONING_OUTPUT_TOKENS_UNAVAILABLE,
        ),
    ],
)
def test_unknown_usage_or_price_never_becomes_zero_cost(
    invocation: ModelInvocationRecord,
    pricing: PricingTable,
    reason: CostUnavailableReason,
) -> None:
    estimate = estimate_invocation_cost(invocation, pricing)

    assert estimate.availability is CostAvailability.UNAVAILABLE
    assert estimate.estimated_cost_usd is None
    assert estimate.input_cost_usd is None
    assert estimate.output_cost_usd is None
    assert estimate.unavailable_reason is reason


def test_attempt_report_exposes_partial_coverage_by_role_node_and_model() -> None:
    pricing = _table(
        ModelPrice(
            provider="openai",
            model="gpt-test",
            input_usd_per_million_tokens="2",
            output_usd_per_million_tokens="8",
        )
    )
    invocations = (
        _invocation(
            usage=_usage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cached_input_tokens=10,
                reasoning_output_tokens=5,
            ),
            latency_ms=10,
        ),
        _invocation(
            sequence=1,
            node_name="agent",
            role=ModelRole.EXPLORER,
            usage=_usage(
                input_tokens=50,
                output_tokens=10,
                total_tokens=60,
                cached_input_tokens=None,
                reasoning_output_tokens=None,
            ),
            latency_ms=20,
        ),
        _invocation(
            sequence=2,
            node_name="evaluate_candidate",
            role=ModelRole.EVALUATOR,
            status=ModelInvocationStatus.FAILED,
            usage=UsageRecord(availability=UsageAvailability.UNAVAILABLE),
            latency_ms=30,
        ),
    )

    report = build_attempt_usage_cost_report(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        invocations=invocations,
        pricing_table=pricing,
    )

    total = report.breakdowns[0]
    assert total.dimension is BreakdownDimension.ATTEMPT
    assert total.key == "run-1"
    assert total.invocation_count == 3
    assert total.succeeded_count == 2
    assert total.failed_count == 1
    assert total.latency_ms == 60
    assert total.input_tokens.coverage is MetricCoverage.PARTIAL
    assert total.input_tokens.value == 150
    assert total.input_tokens.contributing_invocations == 2
    assert total.cached_input_tokens.coverage is MetricCoverage.PARTIAL
    assert total.cached_input_tokens.value == 10
    assert total.cost.coverage is CostCoverage.PARTIAL
    assert total.cost.estimated_cost_usd == "0.00054"
    assert total.cost.estimated_invocations == 2
    assert total.cost.unavailable_invocations == 1
    assert total.cost.unavailable_reasons == (CostUnavailableReason.USAGE_UNAVAILABLE,)
    assert [breakdown.dimension for breakdown in report.breakdowns] == [
        BreakdownDimension.ATTEMPT,
        BreakdownDimension.ROLE,
        BreakdownDimension.ROLE,
        BreakdownDimension.ROLE,
        BreakdownDimension.NODE,
        BreakdownDimension.NODE,
        BreakdownDimension.NODE,
        BreakdownDimension.MODEL,
    ]
    assert [item.invocation_id for item in report.invocation_costs] == [
        item.invocation_id for item in invocations
    ]
    assert report.pricing_table_id == pricing.table_id
    assert json.loads(report.model_dump_json())["report_id"] == report.report_id


def test_attempt_report_rejects_mixed_attempt_identity() -> None:
    invocation = _invocation()
    payload = invocation.model_dump(mode="python")
    mismatched = ModelInvocationRecord.model_validate(
        {**payload, "thread_id": "another-thread"}
    )

    with pytest.raises(ValueError, match="attempt identity"):
        build_attempt_usage_cost_report(
            run_id="run-1",
            thread_id="thread-1",
            turn_number=1,
            invocations=(invocation, mismatched),
            pricing_table=_table(),
        )


def test_empty_attempt_report_is_explicitly_unavailable_not_zero() -> None:
    report = build_attempt_usage_cost_report(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        invocations=(),
        pricing_table=_table(),
    )

    total = report.breakdowns[0]
    assert report.invocation_costs == ()
    assert total.invocation_count == 0
    assert total.input_tokens.coverage is MetricCoverage.UNAVAILABLE
    assert total.input_tokens.value is None
    assert total.cost.coverage is CostCoverage.UNAVAILABLE
    assert total.cost.estimated_cost_usd is None
    assert total.cost.unavailable_reasons == (CostUnavailableReason.NO_INVOCATIONS,)


def test_public_builders_fail_deterministically_on_wrong_member_types() -> None:
    with pytest.raises(TypeError, match="ModelPrice"):
        create_pricing_table(
            version="test-v1",
            effective_from="2026-08-24",
            entries=({"provider": "openai"},),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="ModelInvocationRecord"):
        build_attempt_usage_cost_report(
            run_id="run-1",
            thread_id="thread-1",
            turn_number=1,
            invocations=({"sequence": 0},),  # type: ignore[arg-type]
            pricing_table=_table(),
        )


def test_empty_attempt_budget_allows_the_first_call_without_fabricated_spend() -> None:
    report = build_attempt_usage_cost_report(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        invocations=(),
        pricing_table=_table(),
        max_cost_usd="0.01",
    )

    assert report.budget is not None
    assert report.budget.status is CostBudgetStatus.NOT_STARTED
    assert report.budget.reason is CostBudgetReason.NO_INVOCATIONS
    assert report.budget.confirmed_cost_usd is None
    assert report.budget.remaining_cost_usd == "0.01"
    assert report.budget.overshoot_usd is None
    assert report.budget.next_invocation_allowed is True


@pytest.mark.parametrize(
    (
        "limit",
        "status",
        "reason",
        "allowed",
        "remaining",
        "overshoot",
    ),
    [
        (
            "0.01",
            CostBudgetStatus.WITHIN_BUDGET,
            CostBudgetReason.WITHIN_BUDGET,
            True,
            "0.0065",
            None,
        ),
        (
            "0.0035",
            CostBudgetStatus.EXHAUSTED,
            CostBudgetReason.LIMIT_REACHED,
            False,
            "0",
            None,
        ),
        (
            "0.003",
            CostBudgetStatus.OVERSHOT,
            CostBudgetReason.POST_RESPONSE_OVERSHOOT,
            False,
            "0",
            "0.0005",
        ),
    ],
)
def test_attempt_budget_uses_confirmed_cost_and_reports_post_response_overshoot(
    limit: str,
    status: CostBudgetStatus,
    reason: CostBudgetReason,
    allowed: bool,
    remaining: str,
    overshoot: str | None,
) -> None:
    report = build_attempt_usage_cost_report(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        invocations=(_invocation(),),
        pricing_table=_table(),
        max_cost_usd=limit,
    )

    assert report.budget is not None
    assert report.budget.status is status
    assert report.budget.reason is reason
    assert report.budget.confirmed_cost_usd == "0.0035"
    assert report.budget.remaining_cost_usd == remaining
    assert report.budget.overshoot_usd == overshoot
    assert report.budget.next_invocation_allowed is allowed
    assert (
        AttemptUsageCostReport.model_validate_json(report.model_dump_json()) == report
    )


def test_partial_cost_coverage_conservatively_denies_another_call() -> None:
    report = build_attempt_usage_cost_report(
        run_id="run-1",
        thread_id="thread-1",
        turn_number=1,
        invocations=(
            _invocation(usage=UsageRecord(availability=UsageAvailability.UNAVAILABLE)),
        ),
        pricing_table=_table(),
        max_cost_usd="1",
    )

    assert report.budget is not None
    assert report.budget.status is CostBudgetStatus.UNAVAILABLE
    assert report.budget.reason is CostBudgetReason.INCOMPLETE_COST_COVERAGE
    assert report.budget.confirmed_cost_usd is None
    assert report.budget.remaining_cost_usd is None
    assert report.budget.next_invocation_allowed is False


@pytest.mark.parametrize("value", ["0", "-1", "01", "1.0", "NaN", 1])
def test_cost_budget_requires_a_canonical_positive_decimal_string(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="cost budget"):
        validate_cost_budget_usd(value)
