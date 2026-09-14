"""Bounded public turn history, never a projection of LLM message history."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from langgraph.types import StateSnapshot
from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.lifecycle import HumanReviewHistory, TurnRecord
from repocompass.agent.review_history import validated_review_history
from repocompass.agent.routing import TerminationStatus
from repocompass.agent.thread_view import build_thread_investigation_view
from repocompass.tools.contracts import IDENTIFIER_PATTERN

MAX_PUBLIC_TURN_TEXT = 16_384


class PublicConversationTurn(CheckpointSafeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    turn_number: int = Field(ge=1)
    objective: str = Field(min_length=1, max_length=MAX_PUBLIC_TURN_TEXT)
    summary: str | None = Field(default=None, max_length=MAX_PUBLIC_TURN_TEXT)
    termination_status: TerminationStatus | None = None
    text_truncated: bool = False
    review_history: HumanReviewHistory | None = None

    @model_validator(mode="after")
    def reconcile_terminal_answer(self):
        if (self.summary is None) != (self.termination_status is None):
            raise ValueError("terminal summary and status must agree")
        return self


class PublicConversationPage(CheckpointSafeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    latest_turn_number: int = Field(ge=1)
    turns: tuple[PublicConversationTurn, ...] = Field(max_length=20)
    next_before_turn: int | None = Field(default=None, ge=2)
    provenance: Literal["checkpoint_public_turns"] = "checkpoint_public_turns"

    @model_validator(mode="after")
    def reconcile_cursor(self):
        numbers = [turn.turn_number for turn in self.turns]
        if numbers != sorted(set(numbers)) or any(
            number > self.latest_turn_number for number in numbers
        ):
            raise ValueError("public conversation order is invalid")
        expected = numbers[0] if numbers and numbers[0] > 1 else None
        if self.next_before_turn != expected:
            raise ValueError("public conversation cursor is invalid")
        return self


def build_public_conversation(
    *,
    thread_id: str,
    snapshot: StateSnapshot,
    before_turn: int | None = None,
    limit: int = 10,
) -> PublicConversationPage:
    """An exclusive numeric cursor stays stable as new turns are appended."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("conversation page size must be between 1 and 20")
    if before_turn is not None and (type(before_turn) is not int or before_turn < 1):
        raise ValueError("before_turn must be positive")
    view = build_thread_investigation_view(thread_id=thread_id, snapshot=snapshot)
    values = snapshot.values
    if not isinstance(values, Mapping):
        raise ValueError("checkpoint has no public conversation")
    history = values.get("turn_history", [])
    if not isinstance(history, list) or len(history) != view.turn_number - 1:
        raise ValueError("public turn history is incomplete")
    if any(
        not isinstance(record, TurnRecord) or record.turn_number != index
        for index, record in enumerate(history, 1)
    ):
        raise ValueError("public turn history is not contiguous")
    end = min(view.turn_number + 1, before_turn or view.turn_number + 1)
    start = max(1, end - limit)
    turns = []
    for number in range(start, end):
        if number == view.turn_number:
            objective = view.objective
            summary = None if view.final_result is None else view.final_result.summary
            status = view.termination_status
            review_history = validated_review_history(values.get("review_history"))
        else:
            record = history[number - 1]
            objective = record.objective
            summary = record.final_result.summary
            status = record.termination_status
            review_history = validated_review_history(record.review_history)
        turns.append(
            PublicConversationTurn(
                turn_number=number,
                objective=objective[:MAX_PUBLIC_TURN_TEXT],
                summary=None if summary is None else summary[:MAX_PUBLIC_TURN_TEXT],
                termination_status=status,
                review_history=review_history,
                text_truncated=(
                    len(objective) > MAX_PUBLIC_TURN_TEXT
                    or (summary is not None and len(summary) > MAX_PUBLIC_TURN_TEXT)
                ),
            )
        )
    return PublicConversationPage(
        thread_id=thread_id,
        latest_turn_number=view.turn_number,
        turns=tuple(turns),
        next_before_turn=start if start > 1 else None,
    )
