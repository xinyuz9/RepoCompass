"""Standalone CLI for scope creation, proposal assembly, verification, and apply."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from repocompass.improve.apply import (
    FocusedApplyError,
    FocusedApplyErrorReason,
    FocusedApplyResult,
    apply_verified_proposal,
)
from repocompass.improve.contracts import (
    MAX_IMPROVE_FILE_BYTES,
    FocusedChangeProposal,
    FocusedImproveScope,
    ImproveTargetSelection,
    PythonVerificationProfile,
    VerificationSpec,
    create_change_proposal,
)
from repocompass.improve.engine import (
    FocusedImproveEngine,
    FocusedImproveRunReport,
    SequenceProposalProducer,
)
from repocompass.improve.proposal import ProposalValidationError
from repocompass.improve.sandbox import (
    BubblewrapVerifier,
    SandboxExecutionError,
    SandboxUnavailableError,
)
from repocompass.improve.scope import ImproveScopeError, create_focused_improve_scope

MAX_JSON_ARTIFACT_BYTES = 3 * 1024 * 1024


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repocompass-improve",
        description=(
            "Create exact focused scopes, verify replacement proposals in a "
            "no-network bubblewrap sandbox, and apply only with proposal-ID confirmation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scope_parser = subparsers.add_parser("scope", help="create a clean-snapshot scope")
    scope_parser.add_argument("--repo-path", type=Path, required=True)
    scope_parser.add_argument("--objective", required=True)
    scope_parser.add_argument("--file", action="append", default=[], metavar="PATH")
    scope_parser.add_argument(
        "--symbol",
        action="append",
        nargs=2,
        default=[],
        metavar=("PATH", "QUALIFIED_NAME"),
    )
    scope_parser.add_argument(
        "--verify",
        action="append",
        nargs=2,
        default=[],
        metavar=("PROFILE", "PATH"),
    )
    scope_parser.add_argument("--max-changed-lines", type=_positive_int, default=200)
    scope_parser.add_argument("--max-revisions", type=_revision_count, default=3)
    scope_parser.add_argument("--output", type=Path)

    proposal_parser = subparsers.add_parser(
        "proposal", help="assemble a content-addressed proposal from replacement files"
    )
    proposal_parser.add_argument("--scope", type=Path, required=True)
    proposal_parser.add_argument(
        "--replacement",
        action="append",
        nargs=2,
        required=True,
        metavar=("SCOPED_PATH", "REPLACEMENT_FILE"),
    )
    proposal_parser.add_argument("--rationale", required=True)
    proposal_parser.add_argument("--revision", type=_revision_count, required=True)
    proposal_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser(
        "run", help="sandbox one or more revision proposals; default is dry-run"
    )
    run_parser.add_argument("--repo-path", type=Path, required=True)
    run_parser.add_argument("--scope", type=Path, required=True)
    run_parser.add_argument(
        "--proposal",
        action="append",
        type=Path,
        required=True,
    )
    run_parser.add_argument("--report", type=Path)
    run_parser.add_argument("--apply", action="store_true")
    run_parser.add_argument("--confirm-proposal-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "scope":
            return _scope_command(arguments, parser)
        if arguments.command == "proposal":
            return _proposal_command(arguments)
        if arguments.command == "run":
            return _run_command(arguments, parser)
    except (ImproveScopeError, ProposalValidationError, FocusedApplyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (SandboxUnavailableError, SandboxExecutionError):
        print("error: focused-improve sandbox is unavailable", file=sys.stderr)
        return 2
    except (ValidationError, ValueError, TypeError, OSError, json.JSONDecodeError):
        print("error: invalid focused-improve input or output", file=sys.stderr)
        return 2
    parser.error("unsupported focused-improve command")


def _scope_command(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    targets = [ImproveTargetSelection(path=path) for path in arguments.file]
    targets.extend(
        ImproveTargetSelection(path=path, symbol=symbol)
        for path, symbol in arguments.symbol
    )
    if not targets:
        parser.error("scope requires at least one --file or --symbol target")
    verifications = (
        tuple(
            VerificationSpec(
                profile=PythonVerificationProfile(profile),
                arguments=(path,),
            )
            for profile, path in arguments.verify
        )
        if arguments.verify
        else None
    )
    scope = create_focused_improve_scope(
        arguments.repo_path,
        objective=arguments.objective,
        targets=tuple(targets),
        verifications=verifications,
        max_changed_lines=arguments.max_changed_lines,
        max_revision_count=arguments.max_revisions,
    )
    return _emit_json(scope.model_dump(mode="json"), arguments.output)


def _proposal_command(arguments: argparse.Namespace) -> int:
    scope = _load_model(arguments.scope, FocusedImproveScope)
    replacements: dict[str, str] = {}
    for scoped_path, replacement_file in arguments.replacement:
        if scoped_path in replacements:
            raise ValueError("replacement paths must be unique")
        replacements[scoped_path] = _read_replacement(Path(replacement_file))
    proposal = create_change_proposal(
        scope=scope,
        replacements=replacements,
        rationale=arguments.rationale,
        revision_number=arguments.revision,
    )
    return _emit_json(proposal.model_dump(mode="json"), arguments.output)


def _run_command(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if arguments.apply and arguments.confirm_proposal_id is None:
        parser.error("--apply requires --confirm-proposal-id")
    if not arguments.apply and arguments.confirm_proposal_id is not None:
        parser.error("--confirm-proposal-id requires --apply")
    repository_root = arguments.repo_path.expanduser().resolve(strict=True)
    if arguments.report is not None:
        _reject_output_inside_repository(arguments.report, repository_root)
    reservation = (
        _ReservedOutput(arguments.report) if arguments.report is not None else None
    )
    if reservation is not None:
        reservation.reserve()
    try:
        scope = _load_model(arguments.scope, FocusedImproveScope)
        proposals = tuple(
            _load_model(path, FocusedChangeProposal) for path in arguments.proposal
        )
        report = FocusedImproveEngine(BubblewrapVerifier()).run(
            repository_root,
            scope,
            SequenceProposalProducer(proposals),
        )
        apply_result: FocusedApplyResult | None = None
        if arguments.apply:
            accepted = next(
                (
                    proposal
                    for proposal in proposals
                    if proposal.proposal_id == report.accepted_proposal_id
                ),
                None,
            )
            if accepted is None:
                raise FocusedApplyError(FocusedApplyErrorReason.REPORT_NOT_ACCEPTED)
            apply_result = apply_verified_proposal(
                repository_root,
                scope,
                accepted,
                report,
                confirm_proposal_id=arguments.confirm_proposal_id,
            )
        payload = _run_payload(report, apply_result)
        encoded = _json_bytes(payload)
        if reservation is not None:
            reservation.write(encoded)
        print(encoded.decode("utf-8"))
        return 0 if report.accepted else 2
    except Exception:
        if reservation is not None:
            reservation.cancel()
        raise


def _run_payload(
    report: FocusedImproveRunReport,
    apply_result: FocusedApplyResult | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_report": report.model_dump(mode="json"),
        "apply_result": (
            None if apply_result is None else apply_result.model_dump(mode="json")
        ),
    }


def _load_model(path: Path, model_type):
    value = _load_json(path)
    return model_type.model_validate(value)


def _load_json(path: Path) -> object:
    raw = _read_regular_file(
        path,
        maximum_bytes=MAX_JSON_ARTIFACT_BYTES,
        description="JSON artifact",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("JSON artifact must be UTF-8") from error
    return json.loads(text, object_pairs_hook=_unique_object)


def _read_replacement(path: Path) -> str:
    raw = _read_regular_file(
        path,
        maximum_bytes=MAX_IMPROVE_FILE_BYTES,
        description="replacement",
    )
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("replacement must be UTF-8") from error
    if "\x00" in value:
        raise ValueError("replacement is not bounded UTF-8 text")
    return value


def _read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> bytes:
    candidate = path.expanduser()
    before = candidate.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} must be a regular file")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{description} exceeds the byte limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_snapshot(opened) != _file_snapshot(
            before
        ):
            raise ValueError(f"{description} changed before it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{description} exceeds the byte limit")
        after = os.fstat(descriptor)
        if _file_snapshot(after) != _file_snapshot(opened) or total != after.st_size:
            raise ValueError(f"{description} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_snapshot(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _emit_json(value: object, output: Path | None) -> int:
    encoded = _json_bytes(value)
    if output is not None:
        reservation = _ReservedOutput(output)
        reservation.reserve()
        try:
            reservation.write(encoded)
        except Exception:
            reservation.cancel()
            raise
    print(encoded.decode("utf-8"))
    return 0


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON artifact contains a duplicate key")
        value[key] = item
    return value


def _reject_output_inside_repository(output: Path, repository_root: Path) -> None:
    resolved = output.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(repository_root)
    except ValueError:
        return
    raise ValueError("run report must be outside the source repository")


class _ReservedOutput:
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._descriptor: int | None = None
        self._identity: tuple[int, int] | None = None

    def reserve(self) -> None:
        parent = self._path.parent.resolve(strict=True)
        target = parent / self._path.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        item_stat = os.fstat(descriptor)
        self._path = target
        self._descriptor = descriptor
        self._identity = (item_stat.st_dev, item_stat.st_ino)

    def write(self, value: bytes) -> None:
        if self._descriptor is None:
            raise RuntimeError("output path is not reserved")
        view = memoryview(value)
        while view:
            written = os.write(self._descriptor, view)
            view = view[written:]
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        self._descriptor = None

    def cancel(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            item_stat = self._path.lstat()
        except FileNotFoundError:
            return
        if self._identity == (item_stat.st_dev, item_stat.st_ino):
            self._path.unlink()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _revision_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 3:
        raise argparse.ArgumentTypeError("must not exceed three")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
