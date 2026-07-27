"""Candidate validation and semantic-evaluation domain contracts."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.candidate import CandidateResult


class ValidationIssue(BaseModel):
    """One deterministic defect found in a candidate or evidence reference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = Field(default=None, min_length=1)
    evidence_index: int | None = Field(default=None, ge=0)


class EvidenceExcerpt(BaseModel):
    """Validated repository text supplied to the isolated semantic evaluator."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    evidence_index: int = Field(ge=0)
    path: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str | None = Field(default=None, min_length=1)
    claim: str = Field(min_length=1)
    content: str


class SemanticAssessment(BaseModel):
    """Independent model assessment of candidate relevance and support."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    answers_objective: bool
    evidence_supports_summary: bool
    has_critical_omissions: bool
    missing_evidence: list[str]
    recommended_focus: list[str]
    summary: str = Field(
        min_length=1,
        description="Concise assessment, not a hidden reasoning transcript.",
    )


class EvaluationReport(BaseModel):
    """Merged deterministic and semantic evaluation of one candidate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    deterministic_issues: list[ValidationIssue]
    semantic_assessment: SemanticAssessment
    missing_evidence: list[str]
    recommended_focus: list[str]
    passed: bool


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    """Deterministic findings and safe excerpts for semantic evaluation."""

    issues: tuple[ValidationIssue, ...]
    evidence_pack: tuple[EvidenceExcerpt, ...]


def _validate_repository_file(
    repo_path: Path,
    candidate_path: str,
    *,
    evidence_index: int | None = None,
) -> tuple[Path | None, ValidationIssue | None]:
    relative_path = Path(candidate_path)
    if relative_path.is_absolute():
        return None, ValidationIssue(
            code="path_not_relative",
            message="Candidate path must be repository-relative.",
            path=candidate_path,
            evidence_index=evidence_index,
        )

    resolved_repo = repo_path.resolve()
    resolved_path = (resolved_repo / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_repo)
    except ValueError:
        return None, ValidationIssue(
            code="path_outside_repository",
            message="Candidate path resolves outside the repository.",
            path=candidate_path,
            evidence_index=evidence_index,
        )

    if not resolved_path.exists():
        return None, ValidationIssue(
            code="path_not_found",
            message="Candidate path does not exist in the repository.",
            path=candidate_path,
            evidence_index=evidence_index,
        )
    if not resolved_path.is_file():
        return None, ValidationIssue(
            code="path_not_file",
            message="Candidate path does not identify a repository file.",
            path=candidate_path,
            evidence_index=evidence_index,
        )

    return relative_path, None


def validate_candidate(
    repo_path: Path,
    candidate: CandidateResult,
) -> CandidateValidation:
    """Validate candidate locations and build bounded source excerpts."""
    ...
