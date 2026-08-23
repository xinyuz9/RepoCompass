"""Provider-neutral extraction of paid token usage from model responses."""

import pytest
from langchain_core.messages import AIMessage

from repocompass.agent.usage import (
    UsageAvailability,
    UsageRecord,
    extract_usage_record,
)


def test_extracts_normalized_langchain_usage_metadata() -> None:
    response = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 40},
            "output_token_details": {"reasoning": 12},
        },
    )

    assert extract_usage_record(response) == UsageRecord(
        availability=UsageAvailability.REPORTED,
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cached_input_tokens=40,
        reasoning_output_tokens=12,
    )


def test_extracts_usage_from_structured_output_raw_envelope() -> None:
    raw = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
        },
    )

    usage = extract_usage_record(
        {"raw": raw, "parsed": object(), "parsing_error": None}
    )

    assert usage.availability is UsageAvailability.REPORTED
    assert usage.input_tokens == 9
    assert usage.output_tokens == 3
    assert usage.total_tokens == 12


def test_extracts_openai_compatible_response_metadata_fallback() -> None:
    response = AIMessage(
        content="done",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "total_tokens": 250,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        },
    )

    usage = extract_usage_record(response)

    assert usage == UsageRecord(
        availability=UsageAvailability.REPORTED,
        input_tokens=200,
        output_tokens=50,
        total_tokens=250,
        cached_input_tokens=80,
        reasoning_output_tokens=20,
    )


def test_extracts_deepseek_prompt_cache_hit_fallback() -> None:
    response = AIMessage(
        content="done",
        response_metadata={
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 75,
                "total_tokens": 375,
                "prompt_cache_hit_tokens": 125,
            }
        },
    )

    usage = extract_usage_record(response)

    assert usage.availability is UsageAvailability.REPORTED
    assert usage.input_tokens == 300
    assert usage.output_tokens == 75
    assert usage.cached_input_tokens == 125


def test_fallback_checks_all_supported_detail_container_aliases() -> None:
    response = AIMessage(
        content="done",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {},
                "prompt_tokens_details": {"cached_tokens": 25},
                "output_tokens_details": {},
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        },
    )

    usage = extract_usage_record(response)

    assert usage.cached_input_tokens == 25
    assert usage.reasoning_output_tokens == 4


def test_normalized_usage_takes_precedence_over_provider_fallback() -> None:
    response = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        },
        response_metadata={
            "token_usage": {
                "prompt_tokens": 999,
                "completion_tokens": 1,
                "total_tokens": 1000,
            }
        },
    )

    assert extract_usage_record(response).total_tokens == 12


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
        {"prompt_tokens": "10", "completion_tokens": 2, "total_tokens": 12},
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99},
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 11,
        },
    ],
)
def test_malformed_provider_usage_remains_unavailable(
    payload: dict[str, object],
) -> None:
    response = AIMessage(
        content="done",
        response_metadata={"token_usage": payload},
    )

    assert extract_usage_record(response) == UsageRecord(
        availability=UsageAvailability.UNAVAILABLE
    )


@pytest.mark.parametrize("response", [object(), {}, {"raw": object()}])
def test_missing_usage_remains_unknown_instead_of_zero(response: object) -> None:
    usage = extract_usage_record(response)

    assert usage.availability is UsageAvailability.UNAVAILABLE
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
