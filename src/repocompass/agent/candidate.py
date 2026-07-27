"""Candidate-result domain contracts."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceItem(BaseModel):
    """One candidate claim with a structured repository source location."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    path: str = Field(
        min_length=1,
        description="Repository-relative source path supporting the claim.",
    )
    line_start: int | None = Field(
        default=None,
        ge=1,
        description="One-based first supporting line, when available.",
    )
    line_end: int | None = Field(
        default=None,
        ge=1,
        description="One-based final supporting line, when available.",
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        description="Relevant class, function, method, or symbol, when available.",
    )
    claim: str = Field(
        min_length=1,
        description="Concise evidence-backed observation supported by this location.",
    )

    @model_validator(mode="after")
    def validate_line_range(self) -> "EvidenceItem":
        """Require a complete, ascending line range when an end line is supplied."""

        if self.line_end is not None and self.line_start is None:
            raise ValueError("Evidence line_end requires line_start")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("Evidence line_end must not precede line_start")

        return self


class CandidateResult(BaseModel):
    """Internal investigation result submitted to evaluation before publication."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    relevant_files: list[str] = Field(
        description="Repository-relative paths relevant to the candidate answer."
    )
    evidence: list[EvidenceItem] = Field(
        description="Structured source locations and claims supporting the candidate."
    )
    summary: str = Field(
        min_length=1,
        description="Current best answer derived from the collected evidence.",
    )
    unresolved_questions: list[str] = Field(
        description="Questions not resolved by the collected evidence."
    )
