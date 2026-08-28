"""Installed-shape focused-improve CLI dry-run and apply smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_creates_scope_proposal_then_dry_runs_and_applies(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scope_file = artifacts / "scope.json"
    proposal_file = artifacts / "proposal.json"
    dry_report = artifacts / "dry-report.json"
    apply_report = artifacts / "apply-report.json"
    replacement_file = artifacts / "replacement.py"
    replacement_file.write_text(
        "def allowed() -> int:\n    return 2\n",
        encoding="utf-8",
    )

    scope_result = _cli(
        "scope",
        "--repo-path",
        str(repository),
        "--objective",
        "Return two from the selected function.",
        "--symbol",
        "pkg/example.py",
        "pkg.example.allowed",
        "--verify",
        "pytest",
        "tests/test_change.py",
        "--max-changed-lines",
        "20",
        "--output",
        str(scope_file),
    )
    assert scope_result.returncode == 0, scope_result.stderr
    assert scope_file.is_file()

    proposal_result = _cli(
        "proposal",
        "--scope",
        str(scope_file),
        "--replacement",
        "pkg/example.py",
        str(replacement_file),
        "--rationale",
        "Change only the selected function.",
        "--revision",
        "1",
        "--output",
        str(proposal_file),
    )
    assert proposal_result.returncode == 0, proposal_result.stderr
    proposal_id = json.loads(proposal_file.read_text(encoding="utf-8"))["proposal_id"]
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")

    dry_run = _cli(
        "run",
        "--repo-path",
        str(repository),
        "--scope",
        str(scope_file),
        "--proposal",
        str(proposal_file),
        "--report",
        str(dry_report),
    )
    assert dry_run.returncode == 0, dry_run.stderr
    dry_payload = json.loads(dry_report.read_text(encoding="utf-8"))
    assert dry_payload["run_report"]["accepted"] is True
    assert dry_payload["apply_result"] is None
    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == original
    original_report = dry_report.read_bytes()
    duplicate_report = _cli(
        "run",
        "--repo-path",
        str(repository),
        "--scope",
        str(scope_file),
        "--proposal",
        str(proposal_file),
        "--report",
        str(dry_report),
    )
    assert duplicate_report.returncode == 2
    assert dry_report.read_bytes() == original_report
    assert (repository / "pkg/example.py").read_text(encoding="utf-8") == original

    applied = _cli(
        "run",
        "--repo-path",
        str(repository),
        "--scope",
        str(scope_file),
        "--proposal",
        str(proposal_file),
        "--report",
        str(apply_report),
        "--apply",
        "--confirm-proposal-id",
        proposal_id,
    )
    assert applied.returncode == 0, applied.stderr
    apply_payload = json.loads(apply_report.read_text(encoding="utf-8"))
    assert apply_payload["run_report"]["accepted"] is True
    assert apply_payload["apply_result"]["proposal_id"] == proposal_id
    assert "return 2" in (repository / "pkg/example.py").read_text(encoding="utf-8")
    assert _git(repository, "diff", "--name-only").stdout.strip() == "pkg/example.py"


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-m", "repocompass.improve.cli", *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "pkg").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "pkg/example.py").write_text(
        "def allowed() -> int:\n    return 1\n",
        encoding="utf-8",
    )
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


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
