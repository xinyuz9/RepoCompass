"""Thread, turn, and human-review contracts for RepoCompass lifecycle APIs."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

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


HUMAN_REVIEW_INTERRUPT_KIND = "human_review"
HUMAN_REVIEW_INTERRUPT_SCHEMA_VERSION = 1


class HumanReviewInterruptPayload(TypedDict):
    kind: Literal["human_review"]
    schema_version: Literal[1]
    request: dict[str, JsonValue]


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

    @model_validator(mode="after")
    def require_complete_action_vocabulary(self) -> "HumanReviewRequest":
        """Prevent persisted payloads from changing the deterministic action set."""

        if self.allowed_actions != tuple(HumanDecisionAction):
            raise ValueError("allowed_actions must contain the complete action vocabulary")

        return self


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


def encode_human_review_interrupt(
    request: HumanReviewRequest,
) -> HumanReviewInterruptPayload:
    """Encode one review request as a stable JSON-compatible interrupt envelope."""

    encoded_request = cast(
        dict[str, JsonValue],
        request.model_dump(
            mode="json",
            exclude_computed_fields=True,
        ),
    )
    return {
        "kind": HUMAN_REVIEW_INTERRUPT_KIND,
        "schema_version": HUMAN_REVIEW_INTERRUPT_SCHEMA_VERSION,
        "request": encoded_request,
    }


def decode_human_review_interrupt(value: object) -> HumanReviewRequest:
    """Validate and decode one untrusted checkpoint interrupt payload."""

    error_message = "Invalid human-review interrupt payload"
    if not isinstance(value, Mapping):
        raise ValueError(error_message)
    if set(value) != {"kind", "schema_version", "request"}:
        raise ValueError(error_message)
    if value.get("kind") != HUMAN_REVIEW_INTERRUPT_KIND:
        raise ValueError(error_message)

    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != HUMAN_REVIEW_INTERRUPT_SCHEMA_VERSION
    ):
        raise ValueError(error_message)

    encoded_request = value.get("request")
    if not isinstance(encoded_request, Mapping):
        raise ValueError(error_message)

    try:
        return HumanReviewRequest.model_validate(
            dict(encoded_request),
            strict=False,
        )
    except ValidationError as error:
        raise ValueError(f"{error_message}: {error}") from error
