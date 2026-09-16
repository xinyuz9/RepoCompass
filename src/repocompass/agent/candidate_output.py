"""Invocation-only candidate schemas; checkpoints retain CandidateResult."""

from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.candidate import EvidenceItem


class SourceObservation(EvidenceItem):
    """Unbound file/line evidence, without a model-invented AST selector."""

    symbol: None = Field(
        default=None, description="Must be null: no AST symbol record was selected."
    )


class EvidenceOnlyCandidateResult(BaseModel):
    """Source observations only: no structural records were selected for this call."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )

    evidence: list[SourceObservation] = Field(
        description="Collected file/line observations; discovery listings are not source evidence."
    )
    summary: str = Field(
        min_length=1,
        description="Current answer supported by the collected source evidence.",
    )
    unresolved_questions: list[str] = Field(
        description="Unknown, unsupported or not-yet-investigated questions."
    )
