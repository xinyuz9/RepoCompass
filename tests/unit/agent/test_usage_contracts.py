"""Model invocation and provider-usage contract tests."""

import json
import pytest
from pydantic import ValidationError

from repocompass.agent.checkpoint import RepoCompassCheckpointSerializer
from repocompass.agent.usage import (
    ModelInvocationRecord,
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    UsageRecord,
    create_model_invocation_record,
)


def _reported_usage() -> UsageRecord:
    return UsageRecord(
        availability=UsageAvailability.REPORTED,
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cached_input_tokens=20,
        reasoning_output_tokens=5,
    )


def _invocation(**overrides: object) -> ModelInvocationRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "turn_number": 1,
        "sequence": 0,
        "node_name": "plan_investigation",
        "role": ModelRole.PLANNER,
        "provider": "openai",
        "model": "gpt-5-mini",
        "status": ModelInvocationStatus.SUCCEEDED,
        "latency_ms": 42,
        "usage": _reported_usage(),
        "plan_step_id": None,
        "context_pack_id": None,
        "input_artifact_id": "a" * 64,
        "output_artifact_id": "b" * 64,
        "error_code": None,
    }
    values.update(overrides)
    return create_model_invocation_record(**values)


def test_usage_vocabularies_are_closed() -> None:
    assert set(ModelRole) == {
        ModelRole.PLANNER,
        ModelRole.EXPLORER,
        ModelRole.CANDIDATE_BUILDER,
        ModelRole.EVALUATOR,
        ModelRole.REPLANNER,
    }
    assert set(ModelInvocationStatus) == {
        ModelInvocationStatus.SUCCEEDED,
        ModelInvocationStatus.FAILED,
    }
    assert set(UsageAvailability) == {
        UsageAvailability.REPORTED,
        UsageAvailability.UNAVAILABLE,
    }


def test_unavailable_usage_keeps_every_token_dimension_unknown() -> None:
    usage = UsageRecord(availability=UsageAvailability.UNAVAILABLE)

    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.cached_input_tokens is None
    assert usage.reasoning_output_tokens is None

    with pytest.raises(ValidationError, match="unavailable"):
        UsageRecord(
            availability=UsageAvailability.UNAVAILABLE,
            input_tokens=0,
        )


def test_reported_usage_requires_at_least_one_reported_total() -> None:
    with pytest.raises(ValidationError, match="reported"):
        UsageRecord(availability=UsageAvailability.REPORTED)

    usage = UsageRecord(
        availability=UsageAvailability.REPORTED,
        total_tokens=0,
    )
    assert usage.total_tokens == 0


def test_reported_usage_reconciles_totals_and_token_subsets() -> None:
    assert _reported_usage().total_tokens == 150

    with pytest.raises(ValidationError, match="total_tokens"):
        UsageRecord(
            availability=UsageAvailability.REPORTED,
            input_tokens=120,
            output_tokens=30,
            total_tokens=149,
        )
    with pytest.raises(ValidationError, match="cached_input_tokens"):
        UsageRecord(
            availability=UsageAvailability.REPORTED,
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=11,
        )
    with pytest.raises(ValidationError, match="reasoning_output_tokens"):
        UsageRecord(
            availability=UsageAvailability.REPORTED,
            input_tokens=10,
            output_tokens=2,
            reasoning_output_tokens=3,
        )


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
    ],
)
def test_usage_tokens_are_strict_non_negative_integers(field: str) -> None:
    values = _reported_usage().model_dump(mode="python")

    with pytest.raises(ValidationError):
        UsageRecord.model_validate({**values, field: -1})
    with pytest.raises(ValidationError):
        UsageRecord.model_validate({**values, field: True})
    with pytest.raises(ValidationError):
        UsageRecord.model_validate({**values, field: "1"})


def test_model_invocation_has_deterministic_identity_and_bounded_metadata() -> None:
    first = _invocation()
    same = _invocation(
        latency_ms=99, usage=UsageRecord(availability=UsageAvailability.UNAVAILABLE)
    )

    assert first.invocation_id == same.invocation_id
    assert set(ModelInvocationRecord.model_fields) == {
        "schema_version",
        "invocation_id",
        "run_id",
        "thread_id",
        "turn_number",
        "sequence",
        "node_name",
        "role",
        "provider",
        "model",
        "status",
        "latency_ms",
        "usage",
        "plan_step_id",
        "context_pack_id",
        "input_artifact_id",
        "output_artifact_id",
        "error_code",
    }

    payload = first.model_dump(mode="json")
    with pytest.raises(ValidationError, match="invocation_id"):
        ModelInvocationRecord.model_validate_json(
            json.dumps({**payload, "invocation_id": "f" * 64})
        )


def test_model_invocation_status_requires_canonical_error_metadata() -> None:
    with pytest.raises(ValidationError, match="error_code"):
        _invocation(
            status=ModelInvocationStatus.SUCCEEDED,
            error_code="provider_error",
        )
    with pytest.raises(ValidationError, match="error_code"):
        _invocation(
            status=ModelInvocationStatus.FAILED,
            error_code=None,
        )

    failed = _invocation(
        status=ModelInvocationStatus.FAILED,
        error_code="provider_timeout",
        output_artifact_id=None,
        usage=UsageRecord(availability=UsageAvailability.UNAVAILABLE),
    )
    assert failed.error_code == "provider_timeout"


def test_model_invocation_rejects_invalid_identifiers_and_secret_fields() -> None:
    with pytest.raises((ValidationError, ValueError), match="provider"):
        _invocation(provider="Open AI")
    with pytest.raises((ValidationError, ValueError), match="node_name"):
        _invocation(node_name="../../agent")

    payload = _invocation().model_dump(mode="json")
    with pytest.raises(ValidationError):
        ModelInvocationRecord.model_validate_json(
            json.dumps({**payload, "api_key": "secret"})
        )
    with pytest.raises(ValidationError):
        ModelInvocationRecord.model_validate_json(
            json.dumps({**payload, "raw_prompt": "hidden"})
        )


def test_model_invocation_round_trips_through_json_and_checkpoint() -> None:
    invocation = _invocation()

    assert (
        ModelInvocationRecord.model_validate_json(invocation.model_dump_json())
        == invocation
    )

    serializer = RepoCompassCheckpointSerializer()
    encoded = serializer.dumps_typed({"model_invocation": invocation})
    restored = serializer.loads_typed(encoded)

    assert restored == {"model_invocation": invocation}
    assert isinstance(restored["model_invocation"], ModelInvocationRecord)
    assert isinstance(restored["model_invocation"].usage, UsageRecord)


def test_usage_contract_forbids_unknown_fields() -> None:
    payload = _reported_usage().model_dump(mode="json")

    with pytest.raises(ValidationError):
        UsageRecord.model_validate_json(
            json.dumps({**payload, "estimated_cost_usd": 0})
        )
