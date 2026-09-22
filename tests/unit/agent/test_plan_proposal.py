"""Model proposals cannot manufacture execution state or rewrite completed work."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from repocompass.agent.nodes import (
    _parse_initial_plan,
    _parse_plan_revision,
    PlanRevisionIntegrityError,
)
from repocompass.agent.planning import (
    InvestigationPlan,
    PlanProposal,
    PlanStep,
    ProposedPlanStep,
)


def proposal(*ids):
    return PlanProposal(
        steps=[
            ProposedPlanStep(
                id=id, objective="Read source", expected_evidence=["Exact source lines"]
            )
            for id in ids
        ]
    )


def current():
    return InvestigationPlan(
        steps=[
            PlanStep(
                id="done",
                objective="Preserve this exact observation",
                expected_evidence=["Original lines"],
                status="completed",
            ),
            PlanStep(id="work", objective="Remaining work", status="active"),
        ]
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_proposal_status_is_assigned_only_by_controller(wrapped):
    draft = proposal("one", "two")
    value = {"parsed": draft, "parsing_error": None, "raw": None} if wrapped else draft
    plan = _parse_initial_plan(value)
    assert [step.status for step in plan.steps] == ["active", "pending"]
    assert [step.id for step in plan.steps] == ["one", "two"]
    revised = _parse_plan_revision(value, current())
    assert revised.steps[0] == current().steps[0]
    assert [step.status for step in revised.steps] == ["completed", "active", "pending"]


@pytest.mark.parametrize("status", ["active", "pending", "completed"])
def test_provider_contract_has_no_model_owned_status(status):
    with pytest.raises(ValidationError, match="Extra inputs"):
        PlanProposal.model_validate(
            {
                "steps": [
                    {
                        "id": "one",
                        "objective": "Inspect",
                        "expected_evidence": ["Lines"],
                        "status": status,
                    }
                ]
            }
        )
    assert (
        "status"
        not in PlanProposal.model_json_schema()["$defs"]["ProposedPlanStep"][
            "properties"
        ]
    )


@pytest.mark.parametrize("ids", [("done",), ("one", "one"), ("a", "b", "c", "d", "e")])
def test_revision_rejects_id_collision_duplicate_or_excess_total(ids):
    original = current()
    with pytest.raises(PlanRevisionIntegrityError):
        _parse_plan_revision(proposal(*ids), original)
    assert original == current()


def test_custom_model_legacy_initial_plan_cannot_pretend_completion():
    with pytest.raises(ValueError, match="initial plan"):
        _parse_initial_plan(current())
    with pytest.raises(ValueError, match="initial plan"):
        _parse_initial_plan(
            InvestigationPlan(
                steps=[
                    PlanStep(id="later", objective="Later", status="pending"),
                    PlanStep(id="first", objective="First", status="active"),
                ]
            )
        )


def _raw_proposal(*, status=None, name="PlanProposal", count=1):
    args = proposal("next").model_dump(mode="json")
    if status is not None:
        args["steps"][0]["status"] = status
    return {
        "parsed": None,
        "parsing_error": ValueError("strict transport validation"),
        "raw": AIMessage(
            content="",
            tool_calls=[
                {"id": f"call-{index}", "name": name, "args": args, "type": "tool_call"}
                for index in range(count)
            ],
        ),
    }


def test_valid_raw_status_free_proposal_is_strictly_revalidated():
    value = _raw_proposal()
    assert _parse_initial_plan(value) == proposal("next").materialize()
    assert _parse_plan_revision(value, current()) == proposal("next").materialize(
        [current().steps[0]]
    )


@pytest.mark.parametrize("status", ["active", "pending", "completed"])
def test_raw_revision_never_discards_model_owned_status(status):
    with pytest.raises(PlanRevisionIntegrityError):
        _parse_plan_revision(_raw_proposal(status=status), current())


@pytest.mark.parametrize("options", [{"name": "OtherTool"}, {"count": 2}, {"count": 0}])
def test_ambiguous_or_wrong_tool_revision_is_rejected(options):
    with pytest.raises(PlanRevisionIntegrityError):
        _parse_plan_revision(_raw_proposal(**options), current())
