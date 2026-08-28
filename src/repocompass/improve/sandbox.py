"""Fail-closed bubblewrap verification in an exact detached Git worktree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.improve.contracts import (
    FocusedChangeProposal,
    FocusedImproveScope,
    PythonVerificationProfile,
    VerificationSpec,
    validate_improve_path,
)
from repocompass.improve.git_runtime import safe_git_command, safe_git_environment
from repocompass.improve.proposal import validate_focused_change_proposal
from repocompass.tools.contracts import Sha256Digest

MAX_SANDBOX_OUTPUT_BYTES = 262_144
SANDBOX_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024
SANDBOX_MAX_PROCESSES = 128
SANDBOX_MAX_OPEN_FILES = 256
_GIT_TIMEOUT_SECONDS = 10
_WORKTREE_PREFIX = "repocompass-improve-"
_RESOURCE_LIMIT_WRAPPER = (
    "import os,resource,sys\n"
    "values=tuple(map(int,sys.argv[1:6]))\n"
    "kinds=(resource.RLIMIT_CPU,resource.RLIMIT_AS,resource.RLIMIT_NPROC,"
    "resource.RLIMIT_NOFILE,resource.RLIMIT_FSIZE)\n"
    "for kind,value in zip(kinds,values,strict=True):"
    " resource.setrlimit(kind,(value,value))\n"
    "resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
    "os.execv(sys.argv[6],sys.argv[6:])\n"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|secret)\b"
    r"\s*[:=]\s*[^\s]+"
)


class _SandboxContractModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class VerificationStatus(StrEnum):
    """Terminal status for one fixed-profile sandbox command."""

    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SANDBOX_ERROR = "sandbox_error"


class VerificationDecisionReason(StrEnum):
    """Closed deterministic acceptance/rejection vocabulary."""

    ACCEPTED = "accepted"
    NO_REQUIRED_CHECKS = "no_required_checks"
    CHECK_FAILED = "check_failed"
    CHECK_TIMEOUT = "check_timeout"
    SANDBOX_ERROR = "sandbox_error"
    SIDE_EFFECT_DETECTED = "side_effect_detected"


class SandboxProvenance(_SandboxContractModel):
    """Machine-readable boundary actually used for verification."""

    schema_version: Literal[1] = 1
    backend: Literal["bubblewrap"] = "bubblewrap"
    network_access: Literal[False] = False
    runtime_read_only: Literal[True] = True
    workspace_mount: Literal["/workspace"] = "/workspace"
    workspace_writable: Literal[True] = True
    environment_scrubbed: Literal[True] = True
    cpu_limit_seconds: int = Field(ge=1, le=302)
    address_space_limit_bytes: int = Field(ge=1)
    process_limit: int = Field(ge=1)
    output_limit_bytes: int = Field(ge=1)


class VerificationCommandResult(_SandboxContractModel):
    """Bounded sanitized output and status for one verification profile."""

    schema_version: Literal[1] = 1
    profile: PythonVerificationProfile
    arguments: tuple[str, ...]
    required: bool
    status: VerificationStatus
    exit_code: int | None = Field(default=None, ge=-255, le=255)
    duration_ms: int = Field(ge=0, le=86_400_000)
    output: str = Field(max_length=MAX_SANDBOX_OUTPUT_BYTES)
    output_sha256: Sha256Digest
    output_truncated: bool

    @field_validator("arguments", mode="before")
    @classmethod
    def canonicalize_arguments(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_output_identity(self) -> "VerificationCommandResult":
        expected = hashlib.sha256(self.output.encode("utf-8")).hexdigest()
        if self.output_sha256 != expected:
            raise ValueError("output_sha256 does not match sanitized output")
        if self.status is VerificationStatus.TIMED_OUT and self.exit_code is not None:
            raise ValueError("timed-out check cannot report an exit code")
        return self


class ProposalVerificationReport(_SandboxContractModel):
    """Content-addressed deterministic evaluator input for one proposal."""

    schema_version: Literal[1] = 1
    report_id: Sha256Digest
    scope_id: Sha256Digest
    proposal_id: Sha256Digest
    accepted: bool
    sandbox: SandboxProvenance
    checks: tuple[VerificationCommandResult, ...] = Field(max_length=8)
    side_effect_paths: tuple[str, ...] = Field(max_length=64)
    decision_reasons: tuple[VerificationDecisionReason, ...] = Field(min_length=1)

    @field_validator("checks", "side_effect_paths", "decision_reasons", mode="before")
    @classmethod
    def canonicalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_decision_and_identity(self) -> "ProposalVerificationReport":
        canonical_reasons = tuple(
            reason
            for reason in VerificationDecisionReason
            if reason in self.decision_reasons
        )
        if self.decision_reasons != canonical_reasons or len(
            set(self.decision_reasons)
        ) != len(self.decision_reasons):
            raise ValueError("decision reasons must be unique and canonical")
        expected_acceptance = self.decision_reasons == (
            VerificationDecisionReason.ACCEPTED,
        )
        if self.accepted is not expected_acceptance:
            raise ValueError("accepted does not match deterministic decision reasons")
        expected_id = derive_verification_report_id(
            scope_id=self.scope_id,
            proposal_id=self.proposal_id,
            accepted=self.accepted,
            sandbox=self.sandbox,
            checks=self.checks,
            side_effect_paths=self.side_effect_paths,
            decision_reasons=self.decision_reasons,
        )
        if self.report_id != expected_id:
            raise ValueError("report_id does not match verification report contents")
        return self


class SandboxUnavailableError(RuntimeError):
    """The required bubblewrap or Python runtime cannot be used safely."""


class SandboxExecutionError(RuntimeError):
    """Detached workspace setup or cleanup failed closed."""


class BubblewrapVerifier:
    """Verify proposals with no network and no writable host runtime mounts."""

    def __init__(
        self,
        *,
        bubblewrap_path: Path | None = None,
        python_executable: Path | None = None,
        output_limit_bytes: int = MAX_SANDBOX_OUTPUT_BYTES,
    ) -> None:
        resolved_bwrap = (
            Path(found)
            if bubblewrap_path is None and (found := shutil.which("bwrap")) is not None
            else bubblewrap_path
        )
        self._bubblewrap_path = resolved_bwrap
        self._python_executable = python_executable or Path(sys.executable)
        if (
            not isinstance(output_limit_bytes, int)
            or isinstance(output_limit_bytes, bool)
            or not 1 <= output_limit_bytes <= MAX_SANDBOX_OUTPUT_BYTES
        ):
            raise ValueError("output_limit_bytes is outside the supported range")
        self._output_limit_bytes = output_limit_bytes

    def verify(
        self,
        repository_path: Path,
        scope: FocusedImproveScope,
        proposal: FocusedChangeProposal,
    ) -> ProposalVerificationReport:
        """Validate, sandbox, scan side effects, and deterministically evaluate."""

        self._require_available_runtime()
        validate_focused_change_proposal(repository_path, scope, proposal)
        source_root = repository_path.expanduser().resolve(strict=True)
        top_level = _git_top_level(source_root)
        relative_root = source_root.relative_to(top_level)
        with _detached_worktree(top_level, scope.commit_sha) as worktree:
            workspace = worktree / relative_root
            _apply_replacements(workspace, proposal)
            expected_paths = _workspace_paths(worktree)
            checks = tuple(
                self._run_check(
                    spec,
                    worktree=worktree,
                    relative_root=relative_root,
                    source_root=source_root,
                )
                for spec in scope.verifications
            )
            side_effect_paths = _side_effect_paths(
                worktree,
                workspace,
                proposal,
                expected_paths,
            )
        reasons = _decision_reasons(scope.verifications, checks, side_effect_paths)
        sandbox = SandboxProvenance(
            cpu_limit_seconds=max(
                (min(spec.timeout_seconds + 2, 302) for spec in scope.verifications),
                default=1,
            ),
            address_space_limit_bytes=SANDBOX_ADDRESS_SPACE_BYTES,
            process_limit=SANDBOX_MAX_PROCESSES,
            output_limit_bytes=self._output_limit_bytes,
        )
        return create_verification_report(
            scope_id=scope.scope_id,
            proposal_id=proposal.proposal_id,
            sandbox=sandbox,
            checks=checks,
            side_effect_paths=side_effect_paths,
            decision_reasons=reasons,
        )

    def _require_available_runtime(self) -> None:
        for path in (self._bubblewrap_path, self._python_executable):
            if (
                path is None
                or not path.is_absolute()
                or not path.is_file()
                or not os.access(path, os.X_OK)
            ):
                raise SandboxUnavailableError(
                    "required focused-improve sandbox runtime is unavailable"
                )
        for runtime in _python_runtime_mounts():
            if not runtime.is_absolute() or not runtime.is_dir():
                raise SandboxUnavailableError(
                    "required focused-improve Python runtime is unavailable"
                )

    def _run_check(
        self,
        spec: VerificationSpec,
        *,
        worktree: Path,
        relative_root: Path,
        source_root: Path,
    ) -> VerificationCommandResult:
        command = _profile_command(spec, self._python_executable)
        limited_command = _resource_limited_command(
            command,
            python=self._python_executable,
            timeout_seconds=spec.timeout_seconds,
            output_limit_bytes=self._output_limit_bytes,
        )
        bwrap_command = self._bwrap_command(
            limited_command,
            worktree=worktree,
            relative_root=relative_root,
        )
        started_ns = time.monotonic_ns()
        exit_code: int | None = None
        status = VerificationStatus.SANDBOX_ERROR
        with tempfile.TemporaryFile() as output_file:
            try:
                process = subprocess.Popen(
                    bwrap_command,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    cwd=source_root,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                    shell=False,
                    start_new_session=True,
                )
            except OSError as error:
                raise SandboxUnavailableError(
                    "required focused-improve sandbox runtime is unavailable"
                ) from error
            try:
                exit_code = process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                status = VerificationStatus.TIMED_OUT
                exit_code = None
            else:
                status = (
                    VerificationStatus.PASSED
                    if exit_code == 0
                    else VerificationStatus.FAILED
                )
            output_size = os.fstat(output_file.fileno()).st_size
            output_file.seek(0)
            raw_output = output_file.read(self._output_limit_bytes)
        duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        output = _sanitize_output(
            raw_output,
            sensitive_paths=(
                source_root,
                worktree,
                Path(sys.prefix),
                Path(sys.base_prefix),
            ),
            maximum_bytes=self._output_limit_bytes,
        )
        if status is VerificationStatus.FAILED and output.lstrip().startswith("bwrap:"):
            status = VerificationStatus.SANDBOX_ERROR
        return VerificationCommandResult(
            profile=spec.profile,
            arguments=spec.arguments,
            required=spec.required,
            status=status,
            exit_code=exit_code,
            duration_ms=duration_ms,
            output=output,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            output_truncated=(output_size > self._output_limit_bytes),
        )

    def _bwrap_command(
        self,
        command: Sequence[str],
        *,
        worktree: Path,
        relative_root: Path,
    ) -> tuple[str, ...]:
        if self._bubblewrap_path is None:
            raise SandboxUnavailableError(
                "required focused-improve sandbox runtime is unavailable"
            )
        arguments: list[str] = [
            str(self._bubblewrap_path),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/bin",
            "/bin",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        runtime_mounts = _python_runtime_mounts()
        for directory in _mount_parent_directories(runtime_mounts):
            arguments.extend(("--dir", directory.as_posix()))
        for runtime in runtime_mounts:
            arguments.extend(("--ro-bind", runtime.as_posix(), runtime.as_posix()))
        sandbox_cwd = Path("/workspace") / relative_root
        arguments.extend(
            (
                "--bind",
                worktree.as_posix(),
                "/workspace",
                "--ro-bind",
                "/dev/null",
                "/workspace/.git",
                "--chdir",
                sandbox_cwd.as_posix(),
                "--clearenv",
                "--setenv",
                "PATH",
                f"{self._python_executable.parent.as_posix()}:/usr/bin:/bin",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--setenv",
                "PYTHONPYCACHEPREFIX",
                "/tmp/pycache",
                "--setenv",
                "PYTEST_ADDOPTS",
                "-p no:cacheprovider",
                "--setenv",
                "RUFF_CACHE_DIR",
                "/tmp/ruff-cache",
                "--",
                *command,
            )
        )
        return tuple(arguments)


def create_verification_report(
    *,
    scope_id: str,
    proposal_id: str,
    sandbox: SandboxProvenance,
    checks: Sequence[VerificationCommandResult],
    side_effect_paths: Sequence[str],
    decision_reasons: Sequence[VerificationDecisionReason],
) -> ProposalVerificationReport:
    """Create one strict report from deterministic verifier facts."""

    normalized_checks = tuple(checks)
    normalized_side_effects = tuple(sorted(set(side_effect_paths)))
    normalized_reasons = tuple(
        reason for reason in VerificationDecisionReason if reason in decision_reasons
    )
    accepted = normalized_reasons == (VerificationDecisionReason.ACCEPTED,)
    report_id = derive_verification_report_id(
        scope_id=scope_id,
        proposal_id=proposal_id,
        accepted=accepted,
        sandbox=sandbox,
        checks=normalized_checks,
        side_effect_paths=normalized_side_effects,
        decision_reasons=normalized_reasons,
    )
    return ProposalVerificationReport(
        report_id=report_id,
        scope_id=scope_id,
        proposal_id=proposal_id,
        accepted=accepted,
        sandbox=sandbox,
        checks=normalized_checks,
        side_effect_paths=normalized_side_effects,
        decision_reasons=normalized_reasons,
    )


def derive_verification_report_id(
    *,
    scope_id: str,
    proposal_id: str,
    accepted: bool,
    sandbox: SandboxProvenance,
    checks: Sequence[VerificationCommandResult],
    side_effect_paths: Sequence[str],
    decision_reasons: Sequence[VerificationDecisionReason],
) -> str:
    payload = {
        "scope_id": scope_id,
        "proposal_id": proposal_id,
        "accepted": accepted,
        "sandbox": sandbox.model_dump(mode="json"),
        "checks": [check.model_dump(mode="json") for check in checks],
        "side_effect_paths": list(side_effect_paths),
        "decision_reasons": [reason.value for reason in decision_reasons],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"repocompass:focused-verification-report:v1\0" + encoded
    ).hexdigest()


def _profile_command(spec: VerificationSpec, python: Path) -> tuple[str, ...]:
    arguments = tuple(_verification_path(argument) for argument in spec.arguments)
    if spec.profile is PythonVerificationProfile.COMPILEALL:
        return (str(python), "-m", "compileall", "-q", *arguments)
    if spec.profile is PythonVerificationProfile.PYTEST:
        return (str(python), "-m", "pytest", "-q", *arguments)
    if spec.profile is PythonVerificationProfile.RUFF_CHECK:
        return (str(python), "-m", "ruff", "check", "--no-cache", *arguments)
    if spec.profile is PythonVerificationProfile.RUFF_FORMAT_CHECK:
        return (
            str(python),
            "-m",
            "ruff",
            "format",
            "--check",
            "--no-cache",
            *arguments,
        )
    raise ValueError("unsupported verification profile")


def _verification_path(argument: str) -> str:
    try:
        return validate_improve_path(argument)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "verification arguments must be canonical relative paths"
        ) from error


def _resource_limited_command(
    command: Sequence[str],
    *,
    python: Path,
    timeout_seconds: int,
    output_limit_bytes: int,
) -> tuple[str, ...]:
    """Apply process limits after bubblewrap has entered its new namespaces."""

    return (
        str(python),
        "-c",
        _RESOURCE_LIMIT_WRAPPER,
        str(min(timeout_seconds + 2, 302)),
        str(SANDBOX_ADDRESS_SPACE_BYTES),
        str(SANDBOX_MAX_PROCESSES),
        str(SANDBOX_MAX_OPEN_FILES),
        str(output_limit_bytes),
        *command,
    )


def _python_runtime_mounts() -> tuple[Path, ...]:
    # uv virtualenv executables may point through a version alias symlink next to
    # ``sys.base_prefix``. Mount that runtime directory, not only the resolved
    # version, so the venv's absolute interpreter symlink remains valid inside
    # the otherwise empty sandbox root.
    mounts = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve().parent}
    return tuple(sorted(mounts, key=lambda path: (len(path.parts), path.as_posix())))


def _mount_parent_directories(mounts: Sequence[Path]) -> tuple[Path, ...]:
    system_roots = tuple(
        Path(path) for path in ("/usr", "/bin", "/lib", "/lib64", "/etc")
    )
    parents: set[Path] = set()
    for mount in mounts:
        for parent in mount.parents:
            if parent == Path("/") or any(
                parent == root or root in parent.parents for root in system_roots
            ):
                continue
            parents.add(parent)
    return tuple(sorted(parents, key=lambda path: (len(path.parts), path.as_posix())))


@contextmanager
def _detached_worktree(top_level: Path, commit_sha: str) -> Iterator[Path]:
    _require_no_checkout_filters(top_level)
    worktree = Path(tempfile.mkdtemp(prefix=_WORKTREE_PREFIX)).resolve(strict=True)
    added = False
    try:
        result = _git(
            top_level,
            "worktree",
            "add",
            "--detach",
            worktree.as_posix(),
            commit_sha,
            check=False,
        )
        if result.returncode != 0:
            raise SandboxExecutionError("detached verification worktree setup failed")
        added = True
        yield worktree
    finally:
        if added:
            removed = _git(
                top_level,
                "worktree",
                "remove",
                "--force",
                worktree.as_posix(),
                check=False,
            )
            _git(top_level, "worktree", "prune", check=False)
            if removed.returncode != 0:
                _safe_remove_temporary_worktree(worktree)
                _git(top_level, "worktree", "prune", check=False)
        else:
            _safe_remove_temporary_worktree(worktree)


def _require_no_checkout_filters(top_level: Path) -> None:
    result = _git(
        top_level,
        "config",
        "--local",
        "--name-only",
        "--get-regexp",
        r"^filter\..*\.(clean|smudge|process)$",
        check=False,
    )
    if result.returncode not in (0, 1) or result.stdout:
        raise SandboxExecutionError(
            "repository checkout filters are unsupported by focused verification"
        )


def _safe_remove_temporary_worktree(worktree: Path) -> None:
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    resolved = worktree.resolve(strict=False)
    if resolved.parent != temporary_root or not resolved.name.startswith(
        _WORKTREE_PREFIX
    ):
        raise SandboxExecutionError("temporary worktree cleanup target is invalid")
    if resolved.exists():
        shutil.rmtree(resolved)


def _apply_replacements(workspace: Path, proposal: FocusedChangeProposal) -> None:
    for edit in proposal.edits:
        relative = Path(*PurePosixPath(edit.path).parts)
        target = workspace / relative
        try:
            target_stat = target.lstat()
            resolved = target.resolve(strict=True)
            resolved.relative_to(workspace)
        except (OSError, ValueError) as error:
            raise SandboxExecutionError(
                "proposal target changed before verification"
            ) from error
        if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
            raise SandboxExecutionError("proposal target changed before verification")
        if hashlib.sha256(target.read_bytes()).hexdigest() != edit.base_sha256:
            raise SandboxExecutionError("proposal base changed before verification")
        temporary = target.with_name(
            f".{target.name}.repocompass-{proposal.proposal_id[:12]}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, stat.S_IMODE(target_stat.st_mode))
        try:
            with os.fdopen(descriptor, "wb") as replacement_file:
                replacement_file.write(edit.replacement_text.encode("utf-8"))
                replacement_file.flush()
                os.fsync(replacement_file.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def _workspace_paths(worktree: Path) -> frozenset[str]:
    paths: set[str] = set()
    for root, directories, files in os.walk(worktree, followlinks=False):
        root_path = Path(root)
        for name in (*directories, *files):
            candidate = root_path / name
            relative = candidate.relative_to(worktree).as_posix()
            if relative != ".git":
                paths.add(relative)
    return frozenset(paths)


def _side_effect_paths(
    worktree: Path,
    workspace: Path,
    proposal: FocusedChangeProposal,
    expected_paths: frozenset[str],
) -> tuple[str, ...]:
    side_effects = set(_workspace_paths(worktree).symmetric_difference(expected_paths))
    repository_prefix = workspace.relative_to(worktree)
    expected_edits = {
        (repository_prefix / Path(*PurePosixPath(edit.path).parts)).as_posix(): edit
        for edit in proposal.edits
    }
    changed = set(
        _git_z_paths(
            worktree,
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
        )
    )
    staged = set(
        _git_z_paths(
            worktree,
            "diff",
            "--cached",
            "--name-only",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
        )
    )
    side_effects.update(changed.difference(expected_edits))
    side_effects.update(staged)
    for path, edit in expected_edits.items():
        candidate = worktree / Path(*PurePosixPath(path).parts)
        try:
            actual_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            side_effects.add(path)
        else:
            if actual_digest != edit.replacement_sha256:
                side_effects.add(path)
    return tuple(sorted(side_effects)[:64])


def _decision_reasons(
    specs: Sequence[VerificationSpec],
    checks: Sequence[VerificationCommandResult],
    side_effect_paths: Sequence[str],
) -> tuple[VerificationDecisionReason, ...]:
    reasons: set[VerificationDecisionReason] = set()
    required_indices = tuple(index for index, spec in enumerate(specs) if spec.required)
    if not required_indices:
        reasons.add(VerificationDecisionReason.NO_REQUIRED_CHECKS)
    for index in required_indices:
        status = checks[index].status
        if status is VerificationStatus.FAILED:
            reasons.add(VerificationDecisionReason.CHECK_FAILED)
        elif status is VerificationStatus.TIMED_OUT:
            reasons.add(VerificationDecisionReason.CHECK_TIMEOUT)
        elif status is VerificationStatus.SANDBOX_ERROR:
            reasons.add(VerificationDecisionReason.SANDBOX_ERROR)
    if side_effect_paths:
        reasons.add(VerificationDecisionReason.SIDE_EFFECT_DETECTED)
    if not reasons:
        return (VerificationDecisionReason.ACCEPTED,)
    return tuple(reason for reason in VerificationDecisionReason if reason in reasons)


def _sanitize_output(
    value: bytes,
    *,
    sensitive_paths: Sequence[Path],
    maximum_bytes: int,
) -> str:
    text = value.decode("utf-8", errors="replace")
    replacements = sorted(
        {path.as_posix() for path in sensitive_paths if path.is_absolute()},
        key=len,
        reverse=True,
    )
    for path in replacements:
        text = text.replace(path, "<host-path>")
    text = text.replace("/workspace", "<workspace>")
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def _git_top_level(repository_root: Path) -> Path:
    result = _git(
        repository_root,
        "rev-parse",
        "--show-toplevel",
        check=False,
    )
    if result.returncode != 0:
        raise SandboxExecutionError("focused verification requires Git")
    try:
        top_level = Path(result.stdout.decode("utf-8").strip()).resolve(strict=True)
        repository_root.relative_to(top_level)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise SandboxExecutionError("focused verification requires Git") from error
    return top_level


def _git_z_paths(cwd: Path, *arguments: str) -> tuple[str, ...]:
    result = _git(cwd, *arguments)
    try:
        values = result.stdout.decode("utf-8").split("\x00")
    except UnicodeDecodeError as error:
        raise SandboxExecutionError("Git returned a non-UTF-8 path") from error
    paths: list[str] = []
    for value in values:
        if not value:
            continue
        try:
            paths.append(validate_improve_path(value))
        except (TypeError, ValueError) as error:
            raise SandboxExecutionError("Git returned an unsafe path") from error
    return tuple(paths)


def _git(
    cwd: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            safe_git_command(*arguments),
            cwd=cwd,
            capture_output=True,
            env=safe_git_environment(),
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise SandboxExecutionError("Git operation failed") from error
    if check and completed.returncode != 0:
        raise SandboxExecutionError("Git operation failed")
    return completed
