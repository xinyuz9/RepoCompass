"""Real detached-worktree and bubblewrap verification tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repocompass.improve.contracts import (
    ImproveTargetSelection,
    PythonVerificationProfile,
    VerificationSpec,
    create_change_proposal,
)
from repocompass.improve.sandbox import (
    BubblewrapVerifier,
    SandboxUnavailableError,
    VerificationDecisionReason,
    VerificationStatus,
)
from repocompass.improve.scope import create_focused_improve_scope


def test_bubblewrap_verifies_in_detached_worktree_without_writing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(
        tmp_path,
        test_source=(
            "import os\n"
            "import socket\n"
            "\n"
            "def test_sandbox_boundary():\n"
            "    assert 'REPOCOMPASS_API_KEY' not in os.environ\n"
            "    with __import__('pytest').raises(OSError):\n"
            "        socket.create_connection(('1.1.1.1', 53), timeout=0.1)\n"
        ),
    )
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "must-not-enter-sandbox")
    scope = _scope(
        repository,
        VerificationSpec(
            profile=PythonVerificationProfile.PYTEST,
            arguments=("tests/test_change.py",),
            timeout_seconds=15,
        ),
    )
    proposal = _proposal(repository, scope)
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")

    report = BubblewrapVerifier().verify(repository, scope, proposal)

    assert report.accepted is True
    assert report.decision_reasons == (VerificationDecisionReason.ACCEPTED,)
    assert report.sandbox.backend == "bubblewrap"
    assert report.sandbox.network_access is False
    assert report.sandbox.runtime_read_only is True
    assert report.sandbox.environment_scrubbed is True
    assert report.checks[0].status is VerificationStatus.PASSED
    assert report.side_effect_paths == ()
    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == original
    assert _worktree_count(repository) == 1


def test_all_fixed_python_profiles_run_without_workspace_cache_side_effects(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, test_source="def test_ok():\n    pass\n")
    scope = create_focused_improve_scope(
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
                profile=PythonVerificationProfile.COMPILEALL,
                arguments=("pkg/example.py",),
                timeout_seconds=15,
            ),
            VerificationSpec(
                profile=PythonVerificationProfile.PYTEST,
                arguments=("tests/test_change.py",),
                timeout_seconds=15,
            ),
            VerificationSpec(
                profile=PythonVerificationProfile.RUFF_CHECK,
                arguments=("pkg/example.py",),
                timeout_seconds=15,
            ),
            VerificationSpec(
                profile=PythonVerificationProfile.RUFF_FORMAT_CHECK,
                arguments=("pkg/example.py",),
                timeout_seconds=15,
            ),
        ),
        max_changed_lines=20,
    )

    report = BubblewrapVerifier().verify(
        repository, scope, _proposal(repository, scope)
    )

    assert report.accepted is True, [
        (check.profile.value, check.status.value, check.exit_code, check.output)
        for check in report.checks
    ]
    assert tuple(check.profile for check in report.checks) == tuple(
        PythonVerificationProfile
    )
    assert all(check.status is VerificationStatus.PASSED for check in report.checks)
    assert report.side_effect_paths == ()
    assert _worktree_count(repository) == 1


@pytest.mark.parametrize(
    ("test_source", "timeout", "expected_status", "expected_reason"),
    (
        (
            "def test_failure():\n    assert False\n",
            15,
            VerificationStatus.FAILED,
            VerificationDecisionReason.CHECK_FAILED,
        ),
        (
            "import time\n\ndef test_timeout():\n    time.sleep(10)\n",
            1,
            VerificationStatus.TIMED_OUT,
            VerificationDecisionReason.CHECK_TIMEOUT,
        ),
    ),
)
def test_failed_and_timed_out_checks_are_bounded_and_rejected(
    tmp_path: Path,
    test_source: str,
    timeout: int,
    expected_status: VerificationStatus,
    expected_reason: VerificationDecisionReason,
) -> None:
    repository = _repository(tmp_path, test_source=test_source)
    scope = _scope(
        repository,
        VerificationSpec(
            profile=PythonVerificationProfile.PYTEST,
            arguments=("tests/test_change.py",),
            timeout_seconds=timeout,
        ),
    )

    report = BubblewrapVerifier().verify(
        repository, scope, _proposal(repository, scope)
    )

    assert report.accepted is False
    assert report.checks[0].status is expected_status
    assert expected_reason in report.decision_reasons
    assert report.checks[0].duration_ms < 5_000
    assert len(report.checks[0].output.encode("utf-8")) <= 262_144
    assert "/home/" not in report.checks[0].output
    assert _worktree_count(repository) == 1


def test_workspace_side_effect_outside_proposal_is_detected(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path,
        test_source=(
            "from pathlib import Path\n"
            "\n"
            "def test_side_effect():\n"
            "    Path('outside.txt').write_text('unexpected', encoding='utf-8')\n"
        ),
    )
    scope = _scope(
        repository,
        VerificationSpec(
            profile=PythonVerificationProfile.PYTEST,
            arguments=("tests/test_change.py",),
            timeout_seconds=15,
        ),
    )

    report = BubblewrapVerifier().verify(
        repository, scope, _proposal(repository, scope)
    )

    assert report.checks[0].status is VerificationStatus.PASSED
    assert report.accepted is False
    assert VerificationDecisionReason.SIDE_EFFECT_DETECTED in report.decision_reasons
    assert report.side_effect_paths == ("outside.txt",)
    assert not (repository / "outside.txt").exists()
    assert _worktree_count(repository) == 1


def test_missing_bubblewrap_fails_closed_before_worktree_creation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, test_source="def test_ok():\n    pass\n")
    scope = _scope(repository)

    with pytest.raises(SandboxUnavailableError, match="unavailable"):
        BubblewrapVerifier(bubblewrap_path=Path("/missing/bwrap")).verify(
            repository,
            scope,
            _proposal(repository, scope),
        )

    assert _worktree_count(repository) == 1


def _scope(
    repository: Path,
    verification: VerificationSpec | None = None,
):
    return create_focused_improve_scope(
        repository,
        objective="Return two from the selected function.",
        targets=(
            ImproveTargetSelection(
                path="pkg/example.py",
                symbol="pkg.example.allowed",
            ),
        ),
        verifications=None if verification is None else (verification,),
        max_changed_lines=20,
    )


def _proposal(repository: Path, scope):
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")
    return create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("return 1", "return 2"),
        },
        rationale="Change only the selected function.",
        revision_number=1,
    )


def _repository(tmp_path: Path, *, test_source: str) -> Path:
    repository = tmp_path / "repository"
    (repository / "pkg").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "pkg/example.py").write_text(
        "def allowed() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (repository / "tests/test_change.py").write_text(test_source, encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _worktree_count(repository: Path) -> int:
    output = _git(repository, "worktree", "list", "--porcelain").stdout
    return sum(line.startswith("worktree ") for line in output.splitlines())
