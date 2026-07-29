"""Investigation-plan domain contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlanStepStatus = Literal["pending", "active", "completed"]
MAX_PLAN_STEPS = 5


class PlanStep(BaseModel):
    """One observable investigation step, without hidden model reasoning."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)
    status: PlanStepStatus


class InvestigationPlan(BaseModel):
    """Structured execution plan visible to Explorer and reflection nodes."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)

    @property
    def active_step(self) -> PlanStep:
        """Return the plan's single active step."""

        return next(step for step in self.steps if step.status == "active")

    def next_pending_step(self) -> PlanStep | None:
        active_index = next(
            index for index, step in enumerate(self.steps) if step.status == "active"
        )

        return next(
            (
                step
                for step in self.steps[active_index + 1 :]
                if step.status == "pending"
            ),
            None,
        )

    @model_validator(mode="after")
    def validate_plan_structure(self) -> "InvestigationPlan":
        """Require unique step IDs and exactly one active step."""

        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Investigation plan step IDs must be unique")

        active_steps = [step for step in self.steps if step.status == "active"]
        if len(active_steps) != 1:
            raise ValueError("Investigation plan must contain exactly one active step")

        return self
