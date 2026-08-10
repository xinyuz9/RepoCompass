"""Thread, turn, and human-review contracts for RepoCompass lifecycle APIs."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.result import FinalResult
from repocompass.agent.routing import TerminationStatus


class HumanReviewPolicy(StrEnum):
    """Configured point at which an investigation requires human review."""

    NEVER = "never"
    BEFORE_FINALIZE = "before_finalize"


class HumanDecisionAction(StrEnum):
    """Closed action vocabulary accepted by the first human-review gate."""

    APPROVE = "approve"
    REVISE = "revise"
    CANCEL = "cancel"


class ExecutionStatus(StrEnum):
    """Public state of one Runner execution attempt."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class _LifecycleModel(BaseModel):
    """Strict immutable base for public lifecycle contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class ContinueRequest(_LifecycleModel):
    """New Human turn submitted to an existing completed thread."""

    thread_id: str = Field(
        min_length=1,
        description="Existing checkpoint thread that owns the conversation.",
    )
    prompt: str = Field(
        min_length=1,
        description="New investigation prompt for the next turn.",
    )


class HumanDecision(_LifecycleModel):
    """Structured user response to one pending human-review interrupt."""

    action: HumanDecisionAction = Field(
        description="Deterministic action selected by the user."
    )
    feedback: str | None = Field(
        default=None,
        description="Required investigation guidance when action is revise.",
    )

    @model_validator(mode="after")
    def require_revision_feedback(self) -> "HumanDecision":
        """Require actionable guidance before routing to revision."""

        if self.action is HumanDecisionAction.REVISE and not self.feedback:
            raise ValueError("feedback is required when action is revise")

        return self


class HumanResponseRequest(_LifecycleModel):
    """Human decision bound to the checkpoint thread awaiting that decision."""

    thread_id: str = Field(
        min_length=1,
        description="Checkpoint thread currently paused at human review.",
    )
    decision: HumanDecision


class HumanReviewRequest(_LifecycleModel):
    """Public review payload emitted before an accepted candidate is finalized."""

    turn_number: int = Field(
        ge=1,
        description="One-based turn number within the checkpoint thread.",
    )
    candidate_result: CandidateResult = Field(
        description="Evidence-backed candidate awaiting a human decision."
    )
    evaluation_report: EvaluationReport = Field(
        description="Deterministic and semantic evaluation of the candidate."
    )
    allowed_actions: tuple[HumanDecisionAction, ...] = Field(
        default_factory=lambda: tuple(HumanDecisionAction),
        description="Closed action set accepted by this review gate.",
    )


class TurnRecord(_LifecycleModel):
    """Compact terminal record retained when a thread starts another turn."""

    turn_number: int = Field(
        ge=1,
        description="One-based position of the completed turn in its thread.",
    )
    objective: str = Field(
        min_length=1,
        description="Human objective handled by the completed turn.",
    )
    final_result: FinalResult
    termination_status: TerminationStatus
