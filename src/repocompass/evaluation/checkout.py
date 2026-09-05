"""Opt-in GitHub-only benchmark source preparation; never execute target code."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from repocompass.evaluation.localization_adapter import create_localization_run_request
from repocompass.evaluation.localization_contracts import (
    LocalizationTask,
    LocalizationVariant,
    create_localization_budget_profile,
)
from repocompass.improve.git_runtime import safe_git_command, safe_git_environment

_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def github_source_url(repository: str) -> str:
    if not _REPOSITORY.fullmatch(repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("benchmark source must be a canonical GitHub owner/repository")
    return f"https://github.com/{repository}.git"


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        safe_git_command(
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=always",
            "-c",
            "init.templateDir=",
            *args,
        ),
        cwd=path,
        env=safe_git_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        # A remote may print credentials or untrusted terminal control strings.
        raise RuntimeError(f"benchmark Git operation failed (exit {result.returncode})")
    return result.stdout.strip()


def validate_snapshot(task: LocalizationTask, path: Path) -> None:
    budget = create_localization_budget_profile(
        max_tool_calls_per_step=1,
        max_explorer_calls_per_step=1,
        max_no_progress_rounds=1,
        max_plan_revisions=1,
        context_budget_tokens=4096,
        max_attempt_cost_usd=None,
        max_repository_memory_results=1,
        max_episodic_memory_results=1,
    )
    create_localization_run_request(
        task=task,
        repository_path=path,
        budget_profile=budget,
        variant=LocalizationVariant.TOOL_WRAPPER,
    )


def prepare_snapshot(
    task: LocalizationTask, cache_root: Path, *, allow_network: bool = False
) -> Path:
    """Reuse only exact clean snapshots; new sources require explicit network consent."""
    remote = github_source_url(task.repository)
    if not _COMMIT.fullmatch(task.base_commit):
        raise ValueError("base commit must be a full SHA1")
    cache_root = cache_root.expanduser()
    if cache_root.is_symlink():
        raise ValueError("cache root must not be a symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    root = cache_root.resolve(strict=True)
    repository_key = task.repository.replace("/", "--")
    mirror = root / "mirrors" / repository_key
    checkout = root / "snapshots" / repository_key / task.base_commit
    for candidate in (mirror, checkout):
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError("benchmark destination escapes cache root")
    if checkout.exists():
        validate_snapshot(task, checkout)
        return checkout
    if not allow_network:
        raise ValueError("snapshot not prepared; rerun prepare with --allow-network")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError("at least 2 GiB free disk is required before fetching sources")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if not mirror.exists():
        mirror.mkdir()
        _git(mirror, "init", "--bare", "--quiet")
    elif _git(mirror, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("source cache is not a bare repository")
    _git(mirror, "fetch", "--quiet", "--depth=1", "--no-tags", remote, task.base_commit)
    if _git(mirror, "rev-parse", "FETCH_HEAD") != task.base_commit:
        raise ValueError("fetched source identity differs")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _git(
        mirror,
        "worktree",
        "add",
        "--quiet",
        "--detach",
        str(checkout),
        task.base_commit,
    )
    validate_snapshot(task, checkout)
    return checkout
