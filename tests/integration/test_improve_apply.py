"""Explicit proposal-ID-confirmed apply boundary tests."""

from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

import pytest

from repocompass.improve.apply import (
    FocusedApplyError,
    FocusedApplyErrorReason,
    apply_verified_proposal,
)
from repocompass.improve.contracts import (
    ImproveTargetSelection,
    PythonVerificationProfile,
    VerificationSpec,
    create_change_proposal,
)
from repocompass.improve.engine import FocusedImproveEngine, SequenceProposalProducer
from repocompass.improve.sandbox import BubblewrapVerifier
from repocompass.improve.scope import create_focused_improve_scope


def test_apply_requires_accepted_report_and_exact_confirmation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    scope = _scope(repository)
    proposal = _proposal(repository, scope, value=2)
    run_report = FocusedImproveEngine(BubblewrapVerifier()).run(
        repository,
        scope,
        SequenceProposalProducer((proposal,)),
    )
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")

    with pytest.raises(FocusedApplyError) as confirmation_error:
        apply_verified_proposal(
            repository,
            scope,
            proposal,
            run_report,
            confirm_proposal_id="0" * 64,
        )
    assert (
        confirmation_error.value.reason is FocusedApplyErrorReason.CONFIRMATION_MISMATCH
    )
    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == original

    other = _proposal(repository, scope, value=3)
    with pytest.raises(FocusedApplyError) as report_error:
        apply_verified_proposal(
            repository,
            scope,
            other,
            run_report,
            confirm_proposal_id=other.proposal_id,
        )
    assert report_error.value.reason is FocusedApplyErrorReason.REPORT_MISMATCH
    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == original

    lock_path = Path(
        _git(
            repository,
            "rev-parse",
            "--git-path",
            "repocompass-improve.lock",
        ).stdout.strip()
    )
    if not lock_path.is_absolute():
        lock_path = repository / lock_path
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FocusedApplyError) as lock_error:
            apply_verified_proposal(
                repository,
                scope,
                proposal,
                run_report,
                confirm_proposal_id=proposal.proposal_id,
            )
        assert lock_error.value.reason is FocusedApplyErrorReason.REPOSITORY_LOCKED
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    result = apply_verified_proposal(
        repository,
        scope,
        proposal,
        run_report,
        confirm_proposal_id=proposal.proposal_id,
    )

    assert result.proposal_id == proposal.proposal_id
    assert result.applied_paths == ("pkg/example.py",)
    assert "return 2" in (repository / "pkg/example.py").read_text(encoding="utf-8")
    assert _git(repository, "diff", "--name-only").stdout.strip() == "pkg/example.py"
    assert _git(repository, "diff", "--cached", "--quiet", check=False).returncode == 0
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() == scope.commit_sha

    with pytest.raises(FocusedApplyError) as replay_error:
        apply_verified_proposal(
            repository,
            scope,
            proposal,
            run_report,
            confirm_proposal_id=proposal.proposal_id,
        )
    assert replay_error.value.reason is FocusedApplyErrorReason.SNAPSHOT_CHANGED


def test_apply_preserves_a_source_without_final_newline(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path,
        source="def allowed() -> int:\n    return 1",
    )
    scope = _scope(repository)
    proposal = _proposal(repository, scope, value=2)
    report = FocusedImproveEngine(BubblewrapVerifier()).run(
        repository,
        scope,
        SequenceProposalProducer((proposal,)),
    )

    apply_verified_proposal(
        repository,
        scope,
        proposal,
        report,
        confirm_proposal_id=proposal.proposal_id,
    )

    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == (
        "def allowed() -> int:\n    return 2"
    )


def test_apply_quotes_a_canonical_scoped_path_with_spaces(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _git(repository, "mv", "pkg/example.py", "pkg/space name.py")
    _git(repository, "commit", "-qm", "use a spaced path")
    scope = create_focused_improve_scope(
        repository,
        objective="Change one explicitly selected file.",
        targets=(ImproveTargetSelection(path="pkg/space name.py"),),
        verifications=(
            VerificationSpec(
                profile=PythonVerificationProfile.PYTEST,
                arguments=("tests/test_change.py",),
                timeout_seconds=15,
            ),
        ),
        max_changed_lines=20,
    )
    original = (repository / "pkg/space name.py").read_text(encoding="utf-8")
    proposal = create_change_proposal(
        scope=scope,
        replacements={
            "pkg/space name.py": original.replace("return 1", "return 2"),
        },
        rationale="Change the explicitly selected file.",
        revision_number=1,
    )
    report = FocusedImproveEngine(BubblewrapVerifier()).run(
        repository,
        scope,
        SequenceProposalProducer((proposal,)),
    )

    apply_verified_proposal(
        repository,
        scope,
        proposal,
        report,
        confirm_proposal_id=proposal.proposal_id,
    )

    assert "return 2" in (repository / "pkg/space name.py").read_text(encoding="utf-8")


def _scope(repository: Path):
    return create_focused_improve_scope(
        repository,
        objective="Return two from the selected function.",
        targets=(
            ImproveTargetSelection(
                path="pkg/example.py",
                symbol="pkg.example.allowed",
            ),
        ),
        verifications=(
            VerificationSpec(
                profile=PythonVerificationProfile.PYTEST,
                arguments=("tests/test_change.py",),
                timeout_seconds=15,
            ),
        ),
        max_changed_lines=20,
    )


def _proposal(repository: Path, scope, *, value: int):
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")
    return create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("return 1", f"return {value}"),
        },
        rationale=f"Return {value} from the selected function.",
        revision_number=1,
    )


def _repository(
    tmp_path: Path,
    *,
    source: str = "def allowed() -> int:\n    return 1\n",
) -> Path:
    repository = tmp_path / "repository"
    (repository / "pkg").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "pkg/example.py").write_text(source, encoding="utf-8")
    (repository / "tests/test_change.py").write_text(
        "def test_change():\n    assert True\n",
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )
