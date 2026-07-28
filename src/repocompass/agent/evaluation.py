"""Candidate validation and semantic-evaluation domain contracts."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from repocompass.agent.candidate import CandidateResult
from repocompass.tools.repository import read_code


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

    issues = []
    evidence_pack = []

    if not candidate.evidence:
        issues.append(
            ValidationIssue(
                code="evidence_empty",
                message="Candidate contains no evidence.",
            )
        )

    for evidence_index, evidence in enumerate(candidate.evidence):
        relative_path, path_issue = _validate_repository_file(
            repo_path,
            evidence.path,
            evidence_index=evidence_index,
        )

        if path_issue is not None:
            issues.append(path_issue)
            continue

        if evidence.line_start is None:
            # line_range_missing
            issues.append(
                ValidationIssue(
                    code="line_range_missing",
                    message="Start line or End line missing",
                    path=evidence.path,
                    evidence_index=evidence_index,
                )
            )
            continue

        assert relative_path is not None
        line_end = (
            evidence.line_start if evidence.line_end is None else evidence.line_end
        )

        try:
            read_result = read_code(
                repo_path, relative_path, evidence.line_start, line_end
            )

        except ValueError as error:
            issues.append(
                ValidationIssue(
                    code="line_range_invalid",
                    message=str(error),
                    path=evidence.path,
                    evidence_index=evidence_index,
                )
            )
            continue

        evidence_pack.append(
            EvidenceExcerpt(
                evidence_index=evidence_index,
                path=read_result["path"],
                line_start=read_result["start_line"],
                line_end=read_result["end_line"],
                symbol=evidence.symbol,
                claim=evidence.claim,
                content=read_result["content"],
            )
        )

    return CandidateValidation(
        issues=tuple(issues),
        evidence_pack=tuple(evidence_pack),
    )
