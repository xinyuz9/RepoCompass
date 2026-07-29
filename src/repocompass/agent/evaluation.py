"""Candidate validation and semantic-evaluation domain contracts."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

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
    """Model-produced assessment of step progress and overall answer quality."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    active_step_completed: bool = Field(
        description=(
            "Whether the candidate contains enough evidence to complete the active "
            "plan step."
        )
    )
    answers_objective: bool = Field(
        description=(
            "Whether the candidate directly answers the original investigation "
            "objective."
        )
    )
    evidence_supports_summary: bool = Field(
        description=(
            "Whether the validated source excerpts support the candidate summary."
        )
    )
    has_critical_omissions: bool = Field(
        description=(
            "Whether a missing fact or relationship prevents accepting the overall "
            "answer."
        )
    )
    missing_evidence: list[str] = Field(
        description=(
            "Concrete evidence still required to complete the active step or "
            "overall answer."
        )
    )
    recommended_focus: list[str] = Field(
        description=(
            "Focused repository-investigation targets for the next Explorer or "
            "replanner invocation."
        )
    )
    summary: str = Field(
        min_length=1,
        description="Concise assessment, not a hidden reasoning transcript.",
    )


class EvaluationReport(BaseModel):
    """Code-merged deterministic validation and model-produced assessment."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    deterministic_issues: list[ValidationIssue] = Field(
        description=(
            "Authoritative path, line-range, and evidence defects found by code."
        )
    )
    semantic_assessment: SemanticAssessment = Field(
        description="Independent model assessment of the validated candidate."
    )

    @computed_field(
        description=(
            "Final code-computed acceptance gate. True only when deterministic "
            "validation succeeds and the semantic assessment answers the objective, "
            "is evidence-supported, and has no critical omissions."
        ),
        return_type=bool,
    )
    @property
    def passed(self) -> bool:
        """Return the deterministic aggregate acceptance decision."""

        assessment = self.semantic_assessment
        return (
            not self.deterministic_issues
            and assessment.answers_objective
            and assessment.evidence_supports_summary
            and not assessment.has_critical_omissions
        )


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
