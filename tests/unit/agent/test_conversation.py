"""Public conversation projection excludes internal messages and heavyweight data."""

import pytest
from langgraph.types import StateSnapshot

from repocompass.agent.conversation import build_public_conversation
from repocompass.agent.lifecycle import HumanReviewHistory, TurnRecord
from repocompass.agent.result import FinalResult


def snapshot(turn=4, *, completed=True):
    result = FinalResult(
        relevant_files=["private/heavy.py"],
        evidence=["not-public-in-transcript"],
        summary="Final answer",
        unresolved_questions=[],
    )
    return StateSnapshot(
        values={
            "objective": f"Goal {turn}",
            "turn_number": turn,
            "plan_revision_count": 0,
            "max_plan_revisions": 1,
            "messages": [{"reasoning_content": "SECRET_INTERNAL_REASONING"}],
            "repo_path": "/private/workspace",
            "turn_history": [
                TurnRecord(
                    turn_number=index,
                    objective=f"Goal {index}",
                    final_result=result,
                    termination_status="exhausted" if index == 2 else "accepted",
                )
                for index in range(1, turn)
            ],
            "final_result": result if completed else None,
            "termination_status": "accepted" if completed else None,
        },
        next=() if completed else ("build_candidate",),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


def test_public_turns_are_chronological_cursor_stable_and_internal_data_free():
    page = build_public_conversation(thread_id="thread-1", snapshot=snapshot(), limit=2)
    assert [turn.turn_number for turn in page.turns] == [3, 4]
    assert page.next_before_turn == 3
    prior = build_public_conversation(
        thread_id="thread-1", snapshot=snapshot(5), before_turn=3, limit=2
    )
    assert [turn.turn_number for turn in prior.turns] == [1, 2]
    assert prior.turns[-1].termination_status == "exhausted"
    assert prior.next_before_turn is None
    for secret in [
        "SECRET_INTERNAL_REASONING",
        "repo_path",
        "/private/",
        "not-public-in-transcript",
    ]:
        assert secret not in page.model_dump_json()


def test_pending_turn_has_no_fabricated_answer():
    page = build_public_conversation(
        thread_id="thread-1", snapshot=snapshot(completed=False), limit=1
    )
    assert page.turns[0].summary is None
    assert page.turns[0].termination_status is None
    assert page.turns[0].review_history is None


def test_new_empty_history_is_distinct_from_unavailable_legacy_history():
    state = snapshot()
    state.values["review_history"] = HumanReviewHistory()
    page = build_public_conversation(thread_id="thread-1", snapshot=state, limit=2)
    assert page.turns[0].review_history is None
    assert page.turns[1].review_history == HumanReviewHistory()
    state.values["review_history"] = {"records": []}
    with pytest.raises(ValueError, match="review history"):
        build_public_conversation(thread_id="thread-1", snapshot=state)


def test_long_public_text_is_bounded_and_explicitly_marked():
    state = snapshot()
    state.values["objective"] = "中" * 20_000
    page = build_public_conversation(thread_id="thread-1", snapshot=state, limit=1)
    assert len(page.turns[0].objective) == 16_384
    assert page.turns[0].text_truncated


@pytest.mark.parametrize(
    "limit,cursor", [(0, None), (21, None), (True, None), (1, 0), (1, True)]
)
def test_cursor_and_page_limits_fail_closed(limit, cursor):
    with pytest.raises(ValueError):
        build_public_conversation(
            thread_id="thread-1", snapshot=snapshot(), limit=limit, before_turn=cursor
        )


@pytest.mark.parametrize("malformed", [[], [None], "messages"])
def test_missing_or_corrupted_history_is_not_invented(malformed):
    state = snapshot()
    state.values["turn_history"] = malformed
    with pytest.raises(ValueError):
        build_public_conversation(thread_id="thread-1", snapshot=state)


def test_first_cursor_is_empty_without_repeating_current_turn():
    page = build_public_conversation(
        thread_id="thread-1", snapshot=snapshot(), before_turn=1
    )
    assert page.turns == ()
    assert page.next_before_turn is None
