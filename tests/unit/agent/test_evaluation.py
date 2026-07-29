"""Unit tests for deterministic candidate validation."""

from pathlib import Path

from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import validate_candidate


def test_validate_candidate_builds_bounded_evidence_pack(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "def authenticate(user):\n    return bool(user)\n",
        encoding="utf-8",
    )
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py",
                line_start=1,
                line_end=2,
                symbol="authenticate",
                claim="authenticate returns whether a user value is present.",
            )
        ],
        summary="Authentication is handled in auth.py.",
        unresolved_questions=[],
    )

    validation = validate_candidate(tmp_path, candidate)

    assert validation.issues == ()
    assert len(validation.evidence_pack) == 1
    excerpt = validation.evidence_pack[0]
    assert excerpt.path == "auth.py"
    assert excerpt.line_start == 1
    assert excerpt.line_end == 2
    assert "1 | def authenticate" in excerpt.content
    assert "2 |     return bool(user)" in excerpt.content


def test_validate_candidate_rejects_path_outside_repository(
    tmp_path: Path,
) -> None:
    outside_file = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside_file.write_text("secret = True\n", encoding="utf-8")
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path=f"../{outside_file.name}",
                line_start=1,
                claim="The outside file is relevant.",
            )
        ],
        summary="An outside file was selected.",
        unresolved_questions=[],
    )

    validation = validate_candidate(tmp_path, candidate)

    assert validation.evidence_pack == ()
    assert {issue.code for issue in validation.issues} == {"path_outside_repository"}


def test_validate_candidate_reports_missing_and_invalid_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "auth.py").write_text("value = True\n", encoding="utf-8")
    candidate = CandidateResult(
        evidence=[
            EvidenceItem(
                path="auth.py",
                claim="The file contains authentication.",
            ),
            EvidenceItem(
                path="auth.py",
                line_start=5,
                claim="A nonexistent line contains authentication.",
            ),
        ],
        summary="Authentication may be present.",
        unresolved_questions=[],
    )

    validation = validate_candidate(tmp_path, candidate)

    assert validation.evidence_pack == ()
    assert [issue.code for issue in validation.issues] == [
        "line_range_missing",
        "line_range_invalid",
    ]
