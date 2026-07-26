from pydantic import BaseModel, ConfigDict, Field


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
