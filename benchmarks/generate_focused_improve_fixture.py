"""Regenerate the portable deterministic focused-improve fixture report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repocompass.improve.contracts import (
    ImproveTargetSelection,
    PythonVerificationProfile,
    VerificationSpec,
    create_change_proposal,
    validate_improve_path,
)
from repocompass.improve.engine import FocusedImproveEngine, SequenceProposalProducer
from repocompass.improve.git_runtime import safe_git_command, safe_git_environment
from repocompass.improve.proposal import validate_focused_change_proposal
from repocompass.improve.sandbox import BubblewrapVerifier
from repocompass.improve.scope import create_focused_improve_scope

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = PROJECT_ROOT / "benchmarks/cases/focused-improve-v1/case.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks/results/focused-improve-v1.example.json"
_CASE_BYTE_LIMIT = 64 * 1024
_REPORT_ID_PREFIX = b"repocompass:focused-improve-portable-fixture:v1\0"


@dataclass(frozen=True, slots=True)
class FocusedImproveFixtureCase:
    case_id: str
    objective: str
    repository_files: tuple[str, ...]
    target_path: str
    target_symbol: str
    verification_profile: PythonVerificationProfile
    verification_arguments: tuple[str, ...]
    verification_timeout_seconds: int
    replacement_path: str
    rationale: str
    revision_number: int
    max_changed_lines: int
    max_revision_count: int


def build_report(
    case_path: Path = DEFAULT_CASE,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Run one real sandbox dry-run and return a path-independent projection."""

    case = load_case(case_path)
    workspace_root.mkdir(parents=True, exist_ok=False)
    repository = workspace_root / "repository"
    _copy_repository_fixture(
        case_path.parent / "repository",
        repository,
        case.repository_files,
    )
    commit_sha = _initialize_repository(repository)
    scope = create_focused_improve_scope(
        repository,
        objective=case.objective,
        targets=(
            ImproveTargetSelection(
                path=case.target_path,
                symbol=case.target_symbol,
            ),
        ),
        verifications=(
            VerificationSpec(
                profile=case.verification_profile,
                arguments=case.verification_arguments,
                timeout_seconds=case.verification_timeout_seconds,
            ),
        ),
        max_changed_lines=case.max_changed_lines,
        max_revision_count=case.max_revision_count,
    )
    target = repository / Path(case.target_path)
    before = target.read_bytes()
    replacement = (case_path.parent / Path(case.replacement_path)).read_text(
        encoding="utf-8"
    )
    proposal = create_change_proposal(
        scope=scope,
        replacements={case.target_path: replacement},
        rationale=case.rationale,
        revision_number=case.revision_number,
    )
    validated = validate_focused_change_proposal(repository, scope, proposal)
    run_report = FocusedImproveEngine(BubblewrapVerifier()).run(
        repository,
        scope,
        SequenceProposalProducer((proposal,)),
    )
    after = target.read_bytes()
    if not run_report.iterations or run_report.iterations[-1].verification is None:
        raise RuntimeError("fixture did not produce verification evidence")
    verification = run_report.iterations[-1].verification
    scoped_target = scope.targets[0]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "fixture_commit_sha": commit_sha,
        "scope": {
            "target_count": len(scope.targets),
            "target": {
                "path": scoped_target.path,
                "symbol": scoped_target.symbol,
                "line_start": scoped_target.line_start,
                "line_end": scoped_target.line_end,
                "base_sha256": scoped_target.base_sha256,
            },
            "verification_profiles": [
                item.profile.value for item in scope.verifications
            ],
            "max_changed_lines": scope.max_changed_lines,
            "max_revision_count": scope.max_revision_count,
        },
        "proposal": {
            "revision_number": proposal.revision_number,
            "changed_paths": list(validated.changed_paths),
            "changed_line_count": validated.changed_line_count,
            "replacement_byte_count": validated.replacement_byte_count,
            "replacement_sha256": [edit.replacement_sha256 for edit in proposal.edits],
        },
        "execution": {
            "accepted": run_report.accepted,
            "stop_reason": run_report.stop_reason.value,
            "iteration_count": len(run_report.iterations),
            "decision_reasons": [
                reason.value for reason in verification.decision_reasons
            ],
            "checks": [
                {
                    "profile": check.profile.value,
                    "arguments": list(check.arguments),
                    "required": check.required,
                    "status": check.status.value,
                    "exit_code": check.exit_code,
                    "output_truncated": check.output_truncated,
                }
                for check in verification.checks
            ],
            "side_effect_paths": list(verification.side_effect_paths),
            "dry_run_source_unchanged": before == after,
            "apply_performed": False,
        },
        "sandbox": {
            "backend": verification.sandbox.backend,
            "network_access": verification.sandbox.network_access,
            "runtime_read_only": verification.sandbox.runtime_read_only,
            "workspace_mount": verification.sandbox.workspace_mount,
            "workspace_writable": verification.sandbox.workspace_writable,
            "environment_scrubbed": verification.sandbox.environment_scrubbed,
            "cpu_limit_seconds": verification.sandbox.cpu_limit_seconds,
            "address_space_limit_bytes": (
                verification.sandbox.address_space_limit_bytes
            ),
            "process_limit": verification.sandbox.process_limit,
            "output_limit_bytes": verification.sandbox.output_limit_bytes,
        },
        "cost_semantics": {
            "model_calls": 0,
            "provider_tokens": 0,
            "provider_cost_status": "not_applicable",
            "sandbox_command_count": len(verification.checks),
            "host_compute_cost_included": False,
        },
        "portability": {
            "omitted_raw_fields": [
                "repository_id",
                "scope_id",
                "proposal_id",
                "verification_report_id",
                "run_report_id",
                "duration_ms",
                "output",
                "output_sha256",
            ],
            "reason": (
                "Raw identities bind the absolute trusted repository and command output; "
                "duration is host-dependent. The portable projection keeps deterministic "
                "admission, sandbox, decision, side-effect, and cost facts."
            ),
        },
    }
    payload["portable_report_id"] = derive_portable_report_id(payload)
    return payload


def derive_portable_report_id(payload: dict[str, Any]) -> str:
    if "portable_report_id" in payload:
        raise ValueError("portable report identity must be derived before insertion")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_REPORT_ID_PREFIX + encoded).hexdigest()


def load_case(path: Path) -> FocusedImproveFixtureCase:
    raw = path.read_bytes()
    if len(raw) > _CASE_BYTE_LIMIT:
        raise ValueError("focused-improve fixture case exceeds the byte limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    case = _exact_mapping(
        value,
        {
            "schema_version",
            "case_id",
            "objective",
            "repository_files",
            "target",
            "verification",
            "proposal",
            "max_changed_lines",
            "max_revision_count",
        },
        "case",
    )
    if case["schema_version"] != 1:
        raise ValueError("unsupported focused-improve fixture schema")
    target = _exact_mapping(case["target"], {"path", "symbol"}, "target")
    verification = _exact_mapping(
        case["verification"],
        {"profile", "arguments", "timeout_seconds"},
        "verification",
    )
    proposal = _exact_mapping(
        case["proposal"],
        {"replacement_path", "rationale", "revision_number"},
        "proposal",
    )
    arguments = verification["arguments"]
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ):
        raise ValueError("verification arguments must be a string list")
    repository_files_value = case["repository_files"]
    if not isinstance(repository_files_value, list) or not all(
        isinstance(item, str) for item in repository_files_value
    ):
        raise ValueError("repository_files must be a string list")
    repository_files = tuple(
        validate_improve_path(item) for item in repository_files_value
    )
    if repository_files != tuple(sorted(set(repository_files))):
        raise ValueError("repository_files must be unique and sorted")
    return FocusedImproveFixtureCase(
        case_id=_text(case["case_id"], "case_id"),
        objective=_text(case["objective"], "objective"),
        repository_files=repository_files,
        target_path=validate_improve_path(_text(target["path"], "target path")),
        target_symbol=_text(target["symbol"], "target symbol"),
        verification_profile=PythonVerificationProfile(
            _text(verification["profile"], "verification profile")
        ),
        verification_arguments=tuple(
            validate_improve_path(argument) for argument in arguments
        ),
        verification_timeout_seconds=_integer(
            verification["timeout_seconds"], "verification timeout", 1, 300
        ),
        replacement_path=validate_improve_path(
            _text(proposal["replacement_path"], "replacement path")
        ),
        rationale=_text(proposal["rationale"], "rationale"),
        revision_number=_integer(proposal["revision_number"], "revision number", 1, 3),
        max_changed_lines=_integer(
            case["max_changed_lines"], "max changed lines", 1, 2_000
        ),
        max_revision_count=_integer(
            case["max_revision_count"], "max revision count", 1, 3
        ),
    )


def _initialize_repository(repository: Path) -> str:
    environment = safe_git_environment()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "RepoCompass Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_NAME": "RepoCompass Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
        }
    )
    _git(
        repository,
        "init",
        "-q",
        "--initial-branch=main",
        "--object-format=sha1",
        environment=environment,
    )
    _git(repository, "add", "--all", environment=environment)
    _git(repository, "commit", "-q", "-m", "fixture", environment=environment)
    return _git(
        repository,
        "rev-parse",
        "HEAD",
        environment=environment,
    ).stdout.strip()


def _copy_repository_fixture(
    source_root: Path,
    destination_root: Path,
    relative_paths: tuple[str, ...],
) -> None:
    destination_root.mkdir()
    for relative_path in relative_paths:
        source = source_root / Path(relative_path)
        source_stat = source.lstat()
        if not source.is_file() or source.is_symlink():
            raise ValueError("fixture repository entries must be regular files")
        destination = destination_root / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.stat().st_size != source_stat.st_size:
            raise RuntimeError("fixture repository copy did not preserve file bytes")


def _git(
    repository: Path,
    *arguments: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        safe_git_command(*arguments),
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _exact_mapping(
    value: object,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} fields do not match the fixture schema")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ValueError(f"{name} must be bounded text")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("fixture case contains a duplicate JSON key")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="repocompass-fixture-") as temporary:
        report = build_report(
            arguments.case,
            workspace_root=Path(temporary) / "run",
        )
    mode = "w" if arguments.force else "x"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
