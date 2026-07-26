"""Unit tests for the structured final-result contract."""

import pytest
from pydantic import ValidationError

from repocompass.agent.result import FinalResult


VALID_RESULT = {
    "relevant_files": ["src/example.py"],
    "evidence": ["src/example.py:1 contains the relevant definition."],
    "summary": "The relevant implementation was identified.",
    "unresolved_questions": [],
}


@pytest.mark.parametrize("missing_field", VALID_RESULT)
def test_final_result_requires_every_field(missing_field: str) -> None:
    """Omitting any report section should fail validation."""

    incomplete_result = {
        key: value for key, value in VALID_RESULT.items() if key != missing_field
    }

    with pytest.raises(ValidationError):
        FinalResult.model_validate(incomplete_result)


def test_final_result_rejects_extra_fields() -> None:
    """State-only metadata must not leak into the report schema."""

    with pytest.raises(ValidationError):
        FinalResult.model_validate(
            {
                **VALID_RESULT,
                "stop_reason": "tool_call_budget_exhausted",
            }
        )
