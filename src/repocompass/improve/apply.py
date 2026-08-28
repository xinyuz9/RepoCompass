"""Explicit, report-bound application of one accepted focused proposal."""

from __future__ import annotations

import difflib
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.improve.contracts import FocusedChangeProposal, FocusedImproveScope
from repocompass.improve.engine import FocusedImproveRunReport
from repocompass.improve.proposal import (
    ProposalValidationError,
    validate_focused_change_proposal,
)
from repocompass.tools.contracts import CommitSha, Sha256Digest
from repocompass.tools.gateway import ToolExecutionContext

_GIT_TIMEOUT_SECONDS = 10


class FocusedApplyErrorReason(StrEnum):
    """Stable admission or application denial reason."""

    REPORT_NOT_ACCEPTED = "report_not_accepted"
    REPORT_MISMATCH = "report_mismatch"
    CONFIRMATION_MISMATCH = "confirmation_mismatch"
    REPOSITORY_LOCKED = "repository_locked"
    SNAPSHOT_CHANGED = "snapshot_changed"
    PATCH_CHECK_FAILED = "patch_check_failed"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    POST_APPLY_MISMATCH = "post_apply_mismatch"


class FocusedApplyError(RuntimeError):
    """Sanitized apply failure without patch, source, or absolute path text."""

    def __init__(self, reason: FocusedApplyErrorReason) -> None:
        if not isinstance(reason, FocusedApplyErrorReason):
            raise TypeError("reason must be a FocusedApplyErrorReason")
        self.reason = reason
        super().__init__(f"focused proposal apply rejected: {reason.value}")


class _ApplyContractModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class FocusedApplyResult(_ApplyContractModel):
    """Content-addressed evidence of one source-only, index-free apply."""

    schema_version: Literal[1] = 1
    apply_result_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha
    scope_id: Sha256Digest
    proposal_id: Sha256Digest
    verification_report_id: Sha256Digest
    applied_paths: tuple[str, ...] = Field(min_length=1, max_length=8)
    replacement_sha256: tuple[Sha256Digest, ...] = Field(min_length=1, max_length=8)
    git_index_modified: Literal[False] = False
    commit_created: Literal[False] = False

    @field_validator("applied_paths", "replacement_sha256", mode="before")
    @classmethod
    def canonicalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_apply_identity(self) -> "FocusedApplyResult":
        if len(self.applied_paths) != len(self.replacement_sha256):
            raise ValueError("applied path and digest counts do not reconcile")
        if self.applied_paths != tuple(sorted(set(self.applied_paths))):
            raise ValueError("applied paths must be unique and sorted")
        expected = derive_apply_result_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            scope_id=self.scope_id,
            proposal_id=self.proposal_id,
            verification_report_id=self.verification_report_id,
            applied_paths=self.applied_paths,
            replacement_sha256=self.replacement_sha256,
        )
        if self.apply_result_id != expected:
            raise ValueError("apply_result_id does not match result contents")
        return self


def apply_verified_proposal(
    repository_path: Path,
    scope: FocusedImproveScope,
    proposal: FocusedChangeProposal,
    run_report: FocusedImproveRunReport,
    *,
    confirm_proposal_id: str,
) -> FocusedApplyResult:
    """Revalidate exact authority then all-or-none apply without staging/commit."""

    if not isinstance(repository_path, Path):
        raise TypeError("repository_path must be a pathlib.Path")
    if not isinstance(scope, FocusedImproveScope):
        raise TypeError("scope must be a FocusedImproveScope")
    if not isinstance(proposal, FocusedChangeProposal):
        raise TypeError("proposal must be a FocusedChangeProposal")
    if not isinstance(run_report, FocusedImproveRunReport):
        raise TypeError("run_report must be a FocusedImproveRunReport")
    if confirm_proposal_id != proposal.proposal_id:
        raise FocusedApplyError(FocusedApplyErrorReason.CONFIRMATION_MISMATCH)
    if not run_report.accepted:
        raise FocusedApplyError(FocusedApplyErrorReason.REPORT_NOT_ACCEPTED)
    if (
        run_report.scope_id != scope.scope_id
        or run_report.accepted_proposal_id != proposal.proposal_id
        or not run_report.iterations
        or run_report.iterations[-1].verification is None
        or not run_report.iterations[-1].verification.accepted
        or run_report.iterations[-1].verification.proposal_id != proposal.proposal_id
    ):
        raise FocusedApplyError(FocusedApplyErrorReason.REPORT_MISMATCH)
    repository_root = repository_path.expanduser().resolve(strict=True)
    top_level = _git_top_level(repository_root)
    with _repository_apply_lock(top_level):
        try:
            validate_focused_change_proposal(repository_root, scope, proposal)
        except ProposalValidationError as error:
            raise FocusedApplyError(FocusedApplyErrorReason.SNAPSHOT_CHANGED) from error
        patch = _proposal_patch(repository_root, top_level, proposal)
        check = _git_apply(top_level, patch, check_only=True)
        if check.returncode != 0:
            raise FocusedApplyError(FocusedApplyErrorReason.PATCH_CHECK_FAILED)
        applied = _git_apply(top_level, patch, check_only=False)
        if applied.returncode != 0:
            raise FocusedApplyError(FocusedApplyErrorReason.PATCH_APPLY_FAILED)
        _require_post_apply_state(repository_root, top_level, scope, proposal)
    verification = run_report.iterations[-1].verification
    if verification is None:
        raise FocusedApplyError(FocusedApplyErrorReason.REPORT_MISMATCH)
    paths = tuple(edit.path for edit in proposal.edits)
    digests = tuple(edit.replacement_sha256 for edit in proposal.edits)
    result_id = derive_apply_result_id(
        repository_id=scope.repository_id,
        commit_sha=scope.commit_sha,
        scope_id=scope.scope_id,
        proposal_id=proposal.proposal_id,
        verification_report_id=verification.report_id,
        applied_paths=paths,
        replacement_sha256=digests,
    )
    return FocusedApplyResult(
        apply_result_id=result_id,
        repository_id=scope.repository_id,
        commit_sha=scope.commit_sha,
        scope_id=scope.scope_id,
        proposal_id=proposal.proposal_id,
        verification_report_id=verification.report_id,
        applied_paths=paths,
        replacement_sha256=digests,
    )


def derive_apply_result_id(
    *,
    repository_id: str,
    commit_sha: str,
    scope_id: str,
    proposal_id: str,
    verification_report_id: str,
    applied_paths: Sequence[str],
    replacement_sha256: Sequence[str],
) -> str:
    payload = {
        "repository_id": repository_id,
        "commit_sha": commit_sha,
        "scope_id": scope_id,
        "proposal_id": proposal_id,
        "verification_report_id": verification_report_id,
        "applied_paths": list(applied_paths),
        "replacement_sha256": list(replacement_sha256),
        "git_index_modified": False,
        "commit_created": False,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"repocompass:focused-apply-result:v1\0" + encoded
    ).hexdigest()


def _proposal_patch(
    repository_root: Path,
    top_level: Path,
    proposal: FocusedChangeProposal,
) -> bytes:
    prefix = repository_root.relative_to(top_level)
    chunks: list[str] = []
    for edit in proposal.edits:
        repository_relative = Path(*PurePosixPath(edit.path).parts)
        git_relative = (prefix / repository_relative).as_posix()
        source = (repository_root / repository_relative).read_text(encoding="utf-8")
        old_path = _quote_patch_path(f"a/{git_relative}")
        new_path = _quote_patch_path(f"b/{git_relative}")
        chunks.append(f"diff --git {old_path} {new_path}\n")
        diff_lines = difflib.unified_diff(
            source.splitlines(keepends=True),
            edit.replacement_text.splitlines(keepends=True),
            fromfile=old_path,
            tofile=new_path,
            lineterm="\n",
        )
        for line in diff_lines:
            if line.endswith("\n"):
                chunks.append(line)
            else:
                chunks.extend((line, "\n\\ No newline at end of file\n"))
    patch = "".join(chunks).encode("utf-8")
    if not patch:
        raise FocusedApplyError(FocusedApplyErrorReason.PATCH_CHECK_FAILED)
    return patch


def _quote_patch_path(path: str) -> str:
    if all(character not in path for character in (" ", '"', "\\", "\t")):
        return path
    return json.dumps(path, ensure_ascii=False)


def _require_post_apply_state(
    repository_root: Path,
    top_level: Path,
    scope: FocusedImproveScope,
    proposal: FocusedChangeProposal,
) -> None:
    context = ToolExecutionContext(repository_root)
    if (
        context.repository_id != scope.repository_id
        or context.commit_sha != scope.commit_sha
    ):
        raise FocusedApplyError(FocusedApplyErrorReason.POST_APPLY_MISMATCH)
    prefix = repository_root.relative_to(top_level)
    expected = {
        (prefix / Path(*PurePosixPath(edit.path).parts)).as_posix()
        for edit in proposal.edits
    }
    changed = set(_git_z_paths(top_level, "diff", "--name-only", "-z"))
    staged = set(_git_z_paths(top_level, "diff", "--cached", "--name-only", "-z"))
    if changed != expected or staged:
        raise FocusedApplyError(FocusedApplyErrorReason.POST_APPLY_MISMATCH)
    for edit in proposal.edits:
        target = repository_root / Path(*PurePosixPath(edit.path).parts)
        try:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as error:
            raise FocusedApplyError(
                FocusedApplyErrorReason.POST_APPLY_MISMATCH
            ) from error
        if digest != edit.replacement_sha256:
            raise FocusedApplyError(FocusedApplyErrorReason.POST_APPLY_MISMATCH)


@contextmanager
def _repository_apply_lock(top_level: Path) -> Iterator[None]:
    lock_result = _git(
        top_level,
        "rev-parse",
        "--git-path",
        "repocompass-improve.lock",
    )
    try:
        lock_text = lock_result.stdout.decode("utf-8").strip()
        lock_path = Path(lock_text)
        if not lock_path.is_absolute():
            lock_path = top_level / lock_path
        lock_path.parent.resolve(strict=True)
    except (UnicodeDecodeError, OSError) as error:
        raise FocusedApplyError(FocusedApplyErrorReason.REPOSITORY_LOCKED) from error
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise OSError("lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as error:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass
        raise FocusedApplyError(FocusedApplyErrorReason.REPOSITORY_LOCKED) from error
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _git_apply(
    top_level: Path,
    patch: bytes,
    *,
    check_only: bool,
) -> subprocess.CompletedProcess[bytes]:
    arguments = ["git", "apply", "--whitespace=nowarn"]
    if check_only:
        arguments.append("--check")
    try:
        return subprocess.run(
            arguments,
            cwd=top_level,
            input=patch,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        reason = (
            FocusedApplyErrorReason.PATCH_CHECK_FAILED
            if check_only
            else FocusedApplyErrorReason.PATCH_APPLY_FAILED
        )
        raise FocusedApplyError(reason) from error


def _git_top_level(repository_root: Path) -> Path:
    result = _git(repository_root, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(result.stdout.decode("utf-8").strip()).resolve(strict=True)
        repository_root.relative_to(top_level)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise FocusedApplyError(FocusedApplyErrorReason.SNAPSHOT_CHANGED) from error
    return top_level


def _git_z_paths(cwd: Path, *arguments: str) -> tuple[str, ...]:
    result = _git(cwd, *arguments)
    try:
        return tuple(
            value for value in result.stdout.decode("utf-8").split("\x00") if value
        )
    except UnicodeDecodeError as error:
        raise FocusedApplyError(FocusedApplyErrorReason.POST_APPLY_MISMATCH) from error


def _git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise FocusedApplyError(FocusedApplyErrorReason.SNAPSHOT_CHANGED) from error
    if completed.returncode != 0:
        raise FocusedApplyError(FocusedApplyErrorReason.SNAPSHOT_CHANGED)
    return completed
