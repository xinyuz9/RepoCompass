"""Candidate-result domain contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.structure.contracts import StructuralVerificationStatus
from repocompass.structure.evidence_graph import EvidenceRelationKind

MAX_CANDIDATE_RELATIONS = 64


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


class CandidateRelation(CheckpointSafeModel):
    """One model-proposed relation that deterministic code must resolve."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    source: str = Field(
        min_length=1,
        max_length=2_048,
        description="Exact graph entity ID or canonical source symbol name.",
    )
    relation: EvidenceRelationKind = Field(
        description="Closed structural relation proposed by the candidate."
    )
    target: str = Field(
        min_length=1,
        max_length=2_048,
        description="Exact graph entity ID or canonical target symbol name.",
    )
    verification_status: Literal[StructuralVerificationStatus.INFERRED] = Field(
        default=StructuralVerificationStatus.INFERRED,
        description=(
            "Candidate relations are always inferred until deterministic AST "
            "validation finds an existing verified assertion."
        ),
    )
    supporting_evidence_indices: tuple[int, ...] = Field(
        min_length=1,
        max_length=64,
        description="Canonical candidate evidence indices supporting the proposal.",
    )
    explanation: str = Field(
        min_length=1,
        max_length=2_048,
        description="Why the candidate proposes this non-authoritative relation.",
    )

    @field_validator("source", "target", "explanation")
    @classmethod
    def reject_unsafe_relation_text(cls, value: str) -> str:
        if "\x00" in value or any(character in value for character in ("\r", "\n")):
            raise ValueError("relation text must be single-line canonical text")
        return value

    @field_validator("supporting_evidence_indices")
    @classmethod
    def require_canonical_evidence_indices(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError("relation evidence indices must be non-negative integers")
        if value != tuple(sorted(set(value))):
            raise ValueError("relation evidence indices must be unique and canonical")
        return value


class CandidateResult(CheckpointSafeModel):
    """Internal investigation result submitted to evaluation before publication."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    evidence: list[EvidenceItem] = Field(
        description="Structured source locations and claims supporting the candidate."
    )
    relations: list[CandidateRelation] = Field(
        default_factory=list,
        max_length=MAX_CANDIDATE_RELATIONS,
        description="Evidence-linked structural relations proposed for validation.",
    )
    summary: str = Field(
        min_length=1,
        description="Current best answer derived from the collected evidence.",
    )
    unresolved_questions: list[str] = Field(
        description="Questions not resolved by the collected evidence."
    )

    @model_validator(mode="after")
    def require_unique_candidate_relations(self) -> "CandidateResult":
        identities = tuple(
            (relation.source, relation.relation, relation.target)
            for relation in self.relations
        )
        if len(identities) != len(set(identities)):
            raise ValueError("candidate relations must have unique identities")
        return self
