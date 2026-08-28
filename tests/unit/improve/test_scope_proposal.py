"""Focused-improve scope and replacement proposal boundary tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.improve.contracts import (
    FocusedChangeProposal,
    ImproveTargetSelection,
    PythonVerificationProfile,
    VerificationSpec,
    create_change_proposal,
)
from repocompass.improve.proposal import (
    ProposalValidationError,
    validate_focused_change_proposal,
)
from repocompass.improve.scope import (
    ImproveScopeError,
    create_focused_improve_scope,
)


def test_scope_and_proposal_ids_are_stable_and_strict(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    selection = ImproveTargetSelection(
        path="pkg/example.py",
        symbol="pkg.example.allowed",
    )
    verification = VerificationSpec(
        profile=PythonVerificationProfile.COMPILEALL,
        arguments=("pkg/example.py",),
        timeout_seconds=30,
    )

    first = create_focused_improve_scope(
        repository,
        objective="Increment the allowed result by two.",
        targets=(selection,),
        verifications=(verification,),
        max_changed_lines=20,
    )
    second = create_focused_improve_scope(
        repository,
        objective="Increment the allowed result by two.",
        targets=(selection,),
        verifications=(verification,),
        max_changed_lines=20,
    )
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")
    replacement = original.replace("return value + 1", "return value + 2")
    proposal = create_change_proposal(
        scope=first,
        replacements={"pkg/example.py": replacement},
        rationale="Adjust only the selected function.",
        revision_number=1,
    )
    replay = create_change_proposal(
        scope=first,
        replacements={"pkg/example.py": replacement},
        rationale="Adjust only the selected function.",
        revision_number=1,
    )

    assert first == second
    assert type(first).model_validate(first.model_dump(mode="json")) == first
    assert first.scope_id == second.scope_id
    assert first.targets[0].line_start == 1
    assert first.targets[0].line_end == 2
    assert proposal == replay
    assert proposal.proposal_id == replay.proposal_id
    assert proposal.edits[0].replacement_sha256 != proposal.edits[0].base_sha256

    invalid_id = proposal.model_dump(mode="json")
    invalid_id["proposal_id"] = "0" * 64
    with pytest.raises(ValidationError, match="proposal_id"):
        FocusedChangeProposal.model_validate(invalid_id)

    unknown_field = proposal.model_dump(mode="json")
    unknown_field["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        FocusedChangeProposal.model_validate(unknown_field)


@pytest.mark.parametrize(
    "path",
    (
        "../outside.py",
        "/absolute.py",
        "pkg\\example.py",
        "pkg//example.py",
        " pkg/example.py",
        "pkg/example.py ",
    ),
)
def test_target_paths_must_be_canonical_repository_relative(path: str) -> None:
    with pytest.raises(ValidationError):
        ImproveTargetSelection(path=path)


def test_verification_arguments_cannot_smuggle_command_flags() -> None:
    with pytest.raises(ValidationError, match="command flags"):
        VerificationSpec(
            profile=PythonVerificationProfile.PYTEST,
            arguments=("--pyargs",),
        )


def test_scope_rejects_dirty_snapshot_symlink_and_unknown_symbol(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / "untracked.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(ImproveScopeError) as dirty:
        create_focused_improve_scope(
            repository,
            objective="Change one function.",
            targets=(ImproveTargetSelection(path="pkg/example.py"),),
        )
    assert dirty.value.reason.value == "dirty_snapshot"

    (repository / "untracked.txt").unlink()
    (repository / "pkg/link.py").symlink_to("example.py")
    _git(repository, "add", "pkg/link.py")
    _git(repository, "commit", "-qm", "add symlink")
    with pytest.raises(ImproveScopeError) as symlink:
        create_focused_improve_scope(
            repository,
            objective="Change one function.",
            targets=(ImproveTargetSelection(path="pkg/link.py"),),
        )
    assert symlink.value.reason.value == "symlink"

    with pytest.raises(ImproveScopeError) as unknown_symbol:
        create_focused_improve_scope(
            repository,
            objective="Change one function.",
            targets=(
                ImproveTargetSelection(
                    path="pkg/example.py",
                    symbol="pkg.example.missing",
                ),
            ),
        )
    assert unknown_symbol.value.reason.value == "symbol_not_found"


def test_symbol_scoped_proposal_accepts_only_the_selected_ast_subtree(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    scope = create_focused_improve_scope(
        repository,
        objective="Adjust allowed without touching untouched.",
        targets=(
            ImproveTargetSelection(
                path="pkg/example.py",
                symbol="pkg.example.allowed",
            ),
        ),
        max_changed_lines=20,
    )
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")
    accepted = create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("value + 1", "value + 2"),
        },
        rationale="A focused behavior change.",
        revision_number=1,
    )

    validated = validate_focused_change_proposal(repository, scope, accepted)

    assert validated.proposal_id == accepted.proposal_id
    assert validated.changed_line_count == 2
    assert validated.changed_paths == ("pkg/example.py",)

    outside = create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("value - 1", "value - 2"),
        },
        rationale="An out-of-scope behavior change.",
        revision_number=1,
    )
    with pytest.raises(ProposalValidationError) as outside_error:
        validate_focused_change_proposal(repository, scope, outside)
    assert outside_error.value.reason.value == "out_of_scope_change"

    escaped_ast = create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace(
                "    return value + 1",
                "    return value + 2\ninjected = 1",
            ),
        },
        rationale="Try to add a module-level side effect inside one diff hunk.",
        revision_number=1,
    )
    with pytest.raises(ProposalValidationError) as escaped_error:
        validate_focused_change_proposal(repository, scope, escaped_ast)
    assert escaped_error.value.reason.value == "out_of_scope_ast_change"


def test_proposal_rejects_noop_scope_mismatch_and_changed_snapshot(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    scope = create_focused_improve_scope(
        repository,
        objective="Change the file.",
        targets=(ImproveTargetSelection(path="pkg/example.py"),),
        max_changed_lines=20,
    )
    original = (repository / "pkg/example.py").read_text(encoding="utf-8")
    noop = create_change_proposal(
        scope=scope,
        replacements={"pkg/example.py": original},
        rationale="No change.",
        revision_number=1,
    )
    with pytest.raises(ProposalValidationError) as noop_error:
        validate_focused_change_proposal(repository, scope, noop)
    assert noop_error.value.reason.value == "no_changes"

    other = _repository(tmp_path, name="other")
    other_scope = create_focused_improve_scope(
        other,
        objective="Change the file.",
        targets=(ImproveTargetSelection(path="pkg/example.py"),),
    )
    mismatch_payload = noop.model_dump(mode="json")
    mismatch_payload["scope_id"] = other_scope.scope_id
    mismatch_payload["proposal_id"] = _proposal_id_for_payload(mismatch_payload)
    mismatched = FocusedChangeProposal.model_validate(mismatch_payload)
    with pytest.raises(ProposalValidationError) as mismatch_error:
        validate_focused_change_proposal(repository, scope, mismatched)
    assert mismatch_error.value.reason.value == "scope_mismatch"

    (repository / "pkg/example.py").write_text(
        original.replace("value + 1", "value + 9"),
        encoding="utf-8",
    )
    changed = create_change_proposal(
        scope=scope,
        replacements={
            "pkg/example.py": original.replace("value + 1", "value + 2"),
        },
        rationale="A stale proposal.",
        revision_number=1,
    )
    with pytest.raises(ProposalValidationError) as changed_error:
        validate_focused_change_proposal(repository, scope, changed)
    assert changed_error.value.reason.value == "snapshot_changed"


def _repository(tmp_path: Path, *, name: str = "repo") -> Path:
    repository = tmp_path / name
    (repository / "pkg").mkdir(parents=True)
    (repository / "pkg/example.py").write_text(
        "def allowed(value: int) -> int:\n"
        "    return value + 1\n"
        "\n"
        "\n"
        "def untouched(value: int) -> int:\n"
        "    return value - 1\n",
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "RepoCompass Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _proposal_id_for_payload(payload: dict[str, object]) -> str:
    """Use the public constructor after replacing only the scope identity."""

    from repocompass.improve.contracts import derive_proposal_id

    return derive_proposal_id(
        scope_id=str(payload["scope_id"]),
        revision_number=int(payload["revision_number"]),
        rationale=str(payload["rationale"]),
        edits=tuple(payload["edits"]),
    )
