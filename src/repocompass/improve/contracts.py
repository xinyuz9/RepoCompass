"""Strict, content-addressed contracts for optional focused improvements."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import CommitSha, Sha256Digest

IMPROVE_CONTRACT_SCHEMA_VERSION = 1
MAX_IMPROVE_TARGETS = 8
MAX_IMPROVE_FILE_BYTES = 262_144
MAX_IMPROVE_OBJECTIVE_CHARACTERS = 4_000
MAX_IMPROVE_RATIONALE_CHARACTERS = 4_000
MAX_IMPROVE_CHANGED_LINES = 1_000
MAX_IMPROVE_REVISIONS = 3
MAX_VERIFICATION_ARGUMENTS = 32
MAX_VERIFICATION_ARGUMENT_CHARACTERS = 512

CanonicalPath = Annotated[str, Field(min_length=1, max_length=4_096)]


class _ImproveContractModel(CheckpointSafeModel):
    """Immutable strict base for M9 public and persisted values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class PythonVerificationProfile(StrEnum):
    """Closed Python-only command profiles executed by the M9 sandbox."""

    COMPILEALL = "compileall"
    PYTEST = "pytest"
    RUFF_CHECK = "ruff_check"
    RUFF_FORMAT_CHECK = "ruff_format_check"


class ImproveTargetSelection(_ImproveContractModel):
    """Operator-selected file or exact Python qualified-name target."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    path: CanonicalPath
    symbol: str | None = Field(default=None, min_length=1, max_length=1_024)

    @field_validator("path")
    @classmethod
    def require_canonical_path(cls, value: str) -> str:
        return validate_improve_path(value)

    @field_validator("symbol")
    @classmethod
    def require_safe_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, name="symbol", maximum=1_024)


class ScopedImproveTarget(_ImproveContractModel):
    """One selected target bound to exact source bytes and optional AST span."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    path: CanonicalPath
    base_sha256: Sha256Digest
    byte_count: int = Field(ge=1, le=MAX_IMPROVE_FILE_BYTES)
    line_count: int = Field(ge=1, le=1_000_000)
    symbol: str | None = Field(default=None, min_length=1, max_length=1_024)
    line_start: int | None = Field(default=None, ge=1, le=1_000_000)
    line_end: int | None = Field(default=None, ge=1, le=1_000_000)

    @field_validator("path")
    @classmethod
    def require_canonical_path(cls, value: str) -> str:
        return validate_improve_path(value)

    @field_validator("symbol")
    @classmethod
    def require_safe_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, name="symbol", maximum=1_024)

    @model_validator(mode="after")
    def require_coherent_symbol_span(self) -> "ScopedImproveTarget":
        if self.symbol is None:
            if self.line_start is not None or self.line_end is not None:
                raise ValueError("file target cannot declare a symbol line span")
            return self
        if self.line_start is None or self.line_end is None:
            raise ValueError("symbol target requires a complete line span")
        if self.line_end < self.line_start or self.line_end > self.line_count:
            raise ValueError("symbol line span is outside the source file")
        return self


class VerificationSpec(_ImproveContractModel):
    """One bounded command selected from a closed Python verification profile."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    profile: PythonVerificationProfile
    arguments: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_VERIFICATION_ARGUMENTS,
    )
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    required: bool = True

    @field_validator("arguments", mode="before")
    @classmethod
    def canonicalize_arguments(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("arguments")
    @classmethod
    def require_bounded_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            _bounded_text(
                argument,
                name="verification argument",
                maximum=MAX_VERIFICATION_ARGUMENT_CHARACTERS,
                allow_surrounding_whitespace=True,
            )
        return value


class FocusedImproveScope(_ImproveContractModel):
    """Immutable operator authority for one clean repository snapshot."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    scope_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha
    objective: str = Field(min_length=1, max_length=MAX_IMPROVE_OBJECTIVE_CHARACTERS)
    targets: tuple[ScopedImproveTarget, ...] = Field(
        min_length=1,
        max_length=MAX_IMPROVE_TARGETS,
    )
    verifications: tuple[VerificationSpec, ...] = Field(default=(), max_length=8)
    max_changed_lines: int = Field(default=200, ge=1, le=MAX_IMPROVE_CHANGED_LINES)
    max_revision_count: int = Field(default=MAX_IMPROVE_REVISIONS, ge=1, le=3)

    @field_validator("objective")
    @classmethod
    def require_bounded_objective(cls, value: str) -> str:
        return _bounded_text(
            value,
            name="objective",
            maximum=MAX_IMPROVE_OBJECTIVE_CHARACTERS,
        )

    @field_validator("targets", "verifications", mode="before")
    @classmethod
    def canonicalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_content_identity(self) -> "FocusedImproveScope":
        keys = tuple((target.path, target.symbol or "") for target in self.targets)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("scope targets must be unique and canonically ordered")
        targets_by_path: dict[str, list[ScopedImproveTarget]] = {}
        for target in self.targets:
            targets_by_path.setdefault(target.path, []).append(target)
        for targets in targets_by_path.values():
            if len(targets) > 1 and any(target.symbol is None for target in targets):
                raise ValueError("file scope cannot overlap symbol scopes")
            spans = sorted(
                (target.line_start, target.line_end)
                for target in targets
                if target.line_start is not None and target.line_end is not None
            )
            for previous, current in zip(spans, spans[1:], strict=False):
                if current[0] <= previous[1]:
                    raise ValueError("symbol scopes cannot overlap")
        verification_keys = tuple(
            (item.profile.value, item.arguments) for item in self.verifications
        )
        if len(set(verification_keys)) != len(verification_keys):
            raise ValueError("verification specs must be unique")
        expected = derive_scope_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            objective=self.objective,
            targets=self.targets,
            verifications=self.verifications,
            max_changed_lines=self.max_changed_lines,
            max_revision_count=self.max_revision_count,
        )
        if self.scope_id != expected:
            raise ValueError("scope_id does not match scope contents")
        return self


class ReplacementEdit(_ImproveContractModel):
    """A full UTF-8 replacement bound to the selected file's base digest."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    path: CanonicalPath
    base_sha256: Sha256Digest
    replacement_sha256: Sha256Digest
    replacement_text: str = Field(max_length=MAX_IMPROVE_FILE_BYTES)

    @field_validator("path")
    @classmethod
    def require_canonical_path(cls, value: str) -> str:
        return validate_improve_path(value)

    @field_validator("replacement_text")
    @classmethod
    def require_bounded_utf8_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("replacement text cannot contain NUL")
        if len(value.encode("utf-8")) > MAX_IMPROVE_FILE_BYTES:
            raise ValueError("replacement text exceeds the byte limit")
        return value

    @model_validator(mode="after")
    def require_replacement_digest(self) -> "ReplacementEdit":
        expected = hashlib.sha256(self.replacement_text.encode("utf-8")).hexdigest()
        if self.replacement_sha256 != expected:
            raise ValueError("replacement_sha256 does not match replacement_text")
        return self


class FocusedChangeProposal(_ImproveContractModel):
    """One content-addressed candidate produced outside write/process authority."""

    schema_version: Literal[1] = IMPROVE_CONTRACT_SCHEMA_VERSION
    proposal_id: Sha256Digest
    scope_id: Sha256Digest
    revision_number: int = Field(ge=1, le=MAX_IMPROVE_REVISIONS)
    rationale: str = Field(min_length=1, max_length=MAX_IMPROVE_RATIONALE_CHARACTERS)
    edits: tuple[ReplacementEdit, ...] = Field(
        min_length=1,
        max_length=MAX_IMPROVE_TARGETS,
    )

    @field_validator("rationale")
    @classmethod
    def require_bounded_rationale(cls, value: str) -> str:
        return _bounded_text(
            value,
            name="rationale",
            maximum=MAX_IMPROVE_RATIONALE_CHARACTERS,
        )

    @field_validator("edits", mode="before")
    @classmethod
    def canonicalize_edits(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_canonical_content_identity(self) -> "FocusedChangeProposal":
        paths = tuple(edit.path for edit in self.edits)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("proposal edits must be unique and canonically ordered")
        expected = derive_proposal_id(
            scope_id=self.scope_id,
            revision_number=self.revision_number,
            rationale=self.rationale,
            edits=self.edits,
        )
        if self.proposal_id != expected:
            raise ValueError("proposal_id does not match proposal contents")
        return self


def create_focused_scope_contract(
    *,
    repository_id: str,
    commit_sha: str,
    objective: str,
    targets: Sequence[ScopedImproveTarget],
    verifications: Sequence[VerificationSpec] = (),
    max_changed_lines: int = 200,
    max_revision_count: int = MAX_IMPROVE_REVISIONS,
) -> FocusedImproveScope:
    """Construct one validated scope and derive its stable content identity."""

    normalized_targets = tuple(
        sorted(targets, key=lambda target: (target.path, target.symbol or ""))
    )
    normalized_verifications = tuple(verifications)
    normalized_objective = _bounded_text(
        objective,
        name="objective",
        maximum=MAX_IMPROVE_OBJECTIVE_CHARACTERS,
    )
    scope_id = derive_scope_id(
        repository_id=repository_id,
        commit_sha=commit_sha,
        objective=normalized_objective,
        targets=normalized_targets,
        verifications=normalized_verifications,
        max_changed_lines=max_changed_lines,
        max_revision_count=max_revision_count,
    )
    return FocusedImproveScope(
        scope_id=scope_id,
        repository_id=repository_id,
        commit_sha=commit_sha,
        objective=normalized_objective,
        targets=normalized_targets,
        verifications=normalized_verifications,
        max_changed_lines=max_changed_lines,
        max_revision_count=max_revision_count,
    )


def create_change_proposal(
    *,
    scope: FocusedImproveScope,
    replacements: Mapping[str, str],
    rationale: str,
    revision_number: int,
) -> FocusedChangeProposal:
    """Bind replacement text to exact file digests from an immutable scope."""

    if not isinstance(scope, FocusedImproveScope):
        raise TypeError("scope must be a FocusedImproveScope")
    if not isinstance(replacements, Mapping) or not replacements:
        raise ValueError("replacements must be a non-empty mapping")
    base_by_path = {target.path: target.base_sha256 for target in scope.targets}
    edits: list[ReplacementEdit] = []
    for raw_path, replacement_text in replacements.items():
        path = validate_improve_path(raw_path)
        if path not in base_by_path:
            raise ValueError("replacement path is outside the focused scope")
        if not isinstance(replacement_text, str):
            raise TypeError("replacement text must be a string")
        edits.append(
            ReplacementEdit(
                path=path,
                base_sha256=base_by_path[path],
                replacement_sha256=hashlib.sha256(
                    replacement_text.encode("utf-8")
                ).hexdigest(),
                replacement_text=replacement_text,
            )
        )
    normalized_edits = tuple(sorted(edits, key=lambda edit: edit.path))
    normalized_rationale = _bounded_text(
        rationale,
        name="rationale",
        maximum=MAX_IMPROVE_RATIONALE_CHARACTERS,
    )
    proposal_id = derive_proposal_id(
        scope_id=scope.scope_id,
        revision_number=revision_number,
        rationale=normalized_rationale,
        edits=normalized_edits,
    )
    return FocusedChangeProposal(
        proposal_id=proposal_id,
        scope_id=scope.scope_id,
        revision_number=revision_number,
        rationale=normalized_rationale,
        edits=normalized_edits,
    )


def derive_scope_id(
    *,
    repository_id: str,
    commit_sha: str,
    objective: str,
    targets: Sequence[ScopedImproveTarget | Mapping[str, object]],
    verifications: Sequence[VerificationSpec | Mapping[str, object]],
    max_changed_lines: int,
    max_revision_count: int,
) -> str:
    """Derive a stable digest over every immutable authority-bearing field."""

    target_payload = [_model_payload(ScopedImproveTarget, item) for item in targets]
    verification_payload = [
        _model_payload(VerificationSpec, item) for item in verifications
    ]
    return _content_id(
        b"repocompass:focused-improve-scope:v1\0",
        {
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "objective": objective,
            "targets": target_payload,
            "verifications": verification_payload,
            "max_changed_lines": max_changed_lines,
            "max_revision_count": max_revision_count,
        },
    )


def derive_proposal_id(
    *,
    scope_id: str,
    revision_number: int,
    rationale: str,
    edits: Sequence[ReplacementEdit | Mapping[str, object]],
) -> str:
    """Derive a stable digest over a proposal's complete replacement payload."""

    edit_payload = [_model_payload(ReplacementEdit, item) for item in edits]
    return _content_id(
        b"repocompass:focused-change-proposal:v1\0",
        {
            "scope_id": scope_id,
            "revision_number": revision_number,
            "rationale": rationale,
            "edits": edit_payload,
        },
    )


def validate_improve_path(value: str) -> str:
    """Return one canonical repository-relative POSIX path or fail closed."""

    if not isinstance(value, str):
        raise TypeError("path must be a string")
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("path must be canonical repository-relative POSIX text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("path must be canonical repository-relative POSIX text")
    return value


def _model_payload(model_type: type[_ImproveContractModel], value: object) -> object:
    model = value if isinstance(value, model_type) else model_type.model_validate(value)
    return model.model_dump(mode="json", exclude_computed_fields=True, warnings=False)


def _content_id(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _bounded_text(
    value: str,
    *,
    name: str,
    maximum: int,
    allow_surrounding_whitespace: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (
        not value
        or "\x00" in value
        or any(character in value for character in ("\r", "\n"))
        or (not allow_surrounding_whitespace and value != value.strip())
        or len(value) > maximum
    ):
        raise ValueError(f"{name} must be non-empty bounded single-line text")
    return value
