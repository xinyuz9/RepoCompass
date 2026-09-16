"""Model proposals cannot manufacture execution state or rewrite completed work."""

import pytest
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
