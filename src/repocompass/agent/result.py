"""Public investigation-result contract."""

from pydantic import ConfigDict, Field

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.agent.structural_artifacts import InvestigationStructure


class FinalResult(CheckpointSafeModel):
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
    relevant_symbols: list[str] = Field(
        default_factory=list,
        description="Deterministically resolved symbols relevant to the answer.",
    )
    structure: InvestigationStructure | None = Field(
        default=None,
        description="Content-addressed code graph and traceable reading path.",
    )


def revalidate_final_result(value: object) -> FinalResult:
    """Restore canonical nested types after any checkpoint backend round trip."""

    if not isinstance(value, FinalResult):
        raise TypeError("value must be a FinalResult")
    return FinalResult.model_validate_json(
        value.model_dump_json(
            exclude_computed_fields=True,
            warnings=False,
        )
    )
