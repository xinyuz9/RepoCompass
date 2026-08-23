"""Unit contracts for deterministic Human decision transport and routing."""

import json

import pytest

from repocompass.agent.lifecycle import (
    HumanDecision,
    HumanDecisionAction,
    allowed_human_decision_actions,
    decode_human_decision,
    encode_human_decision,
)
from repocompass.agent.routing import route_human_decision


@pytest.mark.parametrize(
    ("decision", "expected_feedback"),
    [
        (HumanDecision(action=HumanDecisionAction.APPROVE), None),
        (
            HumanDecision(
                action=HumanDecisionAction.REVISE,
                feedback="Inspect middleware registration.",
            ),
            "Inspect middleware registration.",
        ),
        (HumanDecision(action=HumanDecisionAction.CANCEL), None),
    ],
)
def test_human_decision_transport_is_versioned_json_and_round_trips(
    decision: HumanDecision,
    expected_feedback: str | None,
) -> None:
    payload = encode_human_decision(decision)

    assert payload["kind"] == "human_decision"
    assert payload["schema_version"] == 1
    assert payload["decision"]["action"] == decision.action.value
    assert payload["decision"]["feedback"] == expected_feedback
    assert json.loads(json.dumps(payload)) == payload
    assert decode_human_decision(payload) == decision


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"kind": "human_decision", "schema_version": 2, "decision": {}},
        {"kind": "other", "schema_version": 1, "decision": {}},
        {
            "kind": "human_decision",
            "schema_version": 1,
            "decision": {"action": "unknown", "feedback": None},
        },
        {
            "kind": "human_decision",
            "schema_version": 1,
            "decision": {"action": "approve", "feedback": None},
            "extra": True,
        },
    ],
)
def test_human_decision_decoder_rejects_malformed_payload(payload: object) -> None:
    with pytest.raises(ValueError, match="human-decision payload"):
        decode_human_decision(payload)


def test_allowed_human_actions_reflect_remaining_revision_budget() -> None:
    assert allowed_human_decision_actions(0, 2) == tuple(HumanDecisionAction)
    assert allowed_human_decision_actions(2, 2) == (
        HumanDecisionAction.APPROVE,
        HumanDecisionAction.CANCEL,
    )


@pytest.mark.parametrize(
    ("used", "maximum"),
    [(-1, 2), (0, 0), (3, 2), (True, 2), (0, False)],
)
def test_allowed_human_actions_reject_invalid_revision_counters(
    used: int,
    maximum: int,
) -> None:
    with pytest.raises(ValueError, match="revision"):
        allowed_human_decision_actions(used, maximum)


@pytest.mark.parametrize(
    "action",
    [
        HumanDecisionAction.APPROVE,
        HumanDecisionAction.REVISE,
        HumanDecisionAction.CANCEL,
    ],
)
def test_human_decision_router_uses_only_prevalidated_action(
    action: HumanDecisionAction,
) -> None:
    assert route_human_decision({"human_decision_action": action.value}) == action.value


def test_human_decision_router_rejects_missing_action() -> None:
    with pytest.raises(ValueError, match="Human decision"):
        route_human_decision({"human_decision_action": None})
