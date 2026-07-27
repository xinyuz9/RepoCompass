"""Structured domain contracts shared by RepoCompass graph nodes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlanStepStatus = Literal["pending", "active", "completed", "blocked"]


class FinalResult(BaseModel):
    """Structured report produced when the repository investigation finishes."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    relevant_files: list[str] = Field(
        description="Repository-relative paths relevant to the answer."
    )
    evidence: list[str] = Field(
        description="Concise code evidence and observations supporting the summary."
    )
    summary: str = Field(
        description="Concise answer derived from the collected evidence."
    )
    unresolved_questions: list[str] = Field(
        description="Questions that could not be resolved from the available evidence."
    )


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

    task_summary: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    active_step_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_active_step(self) -> "InvestigationPlan":
        """Require unique step IDs and exactly one matching active step."""

        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Investigation plan step IDs must be unique")

        matching_steps = [step for step in self.steps if step.id == self.active_step_id]
        if not matching_steps:
            raise ValueError("Active step ID must identify a plan step")
        if matching_steps[0].status != "active":
            raise ValueError("Active step ID must identify an active step")

        active_steps = [step for step in self.steps if step.status == "active"]
        if len(active_steps) != 1:
            raise ValueError("Investigation plan must contain exactly one active step")

        return self
