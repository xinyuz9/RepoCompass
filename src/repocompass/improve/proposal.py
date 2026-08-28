"""Deterministic validation for untrusted focused-change proposals."""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from repocompass.improve.contracts import FocusedChangeProposal, FocusedImproveScope
from repocompass.improve.scope import ImproveScopeError, load_scope_sources


class ProposalValidationErrorReason(StrEnum):
    """Stable proposal denial reasons suitable for bounded revision feedback."""

    SCOPE_MISMATCH = "scope_mismatch"
    SNAPSHOT_CHANGED = "snapshot_changed"
    OUT_OF_SCOPE_PATH = "out_of_scope_path"
    BASE_DIGEST_MISMATCH = "base_digest_mismatch"
    NO_CHANGES = "no_changes"
    CHANGED_LINE_LIMIT = "changed_line_limit"
    OUT_OF_SCOPE_CHANGE = "out_of_scope_change"
    INVALID_REPLACEMENT_SYNTAX = "invalid_replacement_syntax"
    OUT_OF_SCOPE_AST_CHANGE = "out_of_scope_ast_change"


class ProposalValidationError(ValueError):
    """Sanitized validation failure that contains no source or absolute path."""

    def __init__(self, reason: ProposalValidationErrorReason) -> None:
        if not isinstance(reason, ProposalValidationErrorReason):
            raise TypeError("reason must be a ProposalValidationErrorReason")
        self.reason = reason
        super().__init__(f"focused proposal rejected: {reason.value}")


@dataclass(frozen=True, slots=True)
class ValidatedFocusedChange:
    """Minimal authority passed from validation to the future sandbox stage."""

    proposal_id: str
    scope_id: str
    changed_paths: tuple[str, ...]
    changed_line_count: int
    replacement_byte_count: int


def validate_focused_change_proposal(
    repository_path: object,
    scope: FocusedImproveScope,
    proposal: FocusedChangeProposal,
) -> ValidatedFocusedChange:
    """Validate exact snapshot, scope, line ranges, and Python AST isolation."""

    if not isinstance(scope, FocusedImproveScope):
        raise TypeError("scope must be a FocusedImproveScope")
    if not isinstance(proposal, FocusedChangeProposal):
        raise TypeError("proposal must be a FocusedChangeProposal")
    if proposal.scope_id != scope.scope_id:
        raise ProposalValidationError(ProposalValidationErrorReason.SCOPE_MISMATCH)
    try:
        sources = load_scope_sources(repository_path, scope)  # type: ignore[arg-type]
    except ImproveScopeError as error:
        raise ProposalValidationError(
            ProposalValidationErrorReason.SNAPSHOT_CHANGED
        ) from error
    targets_by_path: dict[str, list[object]] = {}
    for target in scope.targets:
        targets_by_path.setdefault(target.path, []).append(target)
    changed_line_count = 0
    replacement_byte_count = 0
    changed_paths: list[str] = []
    for edit in proposal.edits:
        targets = targets_by_path.get(edit.path)
        if targets is None:
            raise ProposalValidationError(
                ProposalValidationErrorReason.OUT_OF_SCOPE_PATH
            )
        base_digests = {target.base_sha256 for target in targets}
        if len(base_digests) != 1 or edit.base_sha256 not in base_digests:
            raise ProposalValidationError(
                ProposalValidationErrorReason.BASE_DIGEST_MISMATCH
            )
        original = sources[edit.path]
        if original == edit.replacement_text:
            continue
        file_change_count = _validate_line_scope(
            original,
            edit.replacement_text,
            tuple(targets),
        )
        allowed_symbols = frozenset(
            target.symbol for target in targets if target.symbol is not None
        )
        if allowed_symbols:
            _require_only_allowed_ast_changes(
                path=edit.path,
                original=original,
                replacement=edit.replacement_text,
                allowed_symbols=allowed_symbols,
            )
        changed_line_count += file_change_count
        replacement_byte_count += len(edit.replacement_text.encode("utf-8"))
        changed_paths.append(edit.path)
    if not changed_paths:
        raise ProposalValidationError(ProposalValidationErrorReason.NO_CHANGES)
    if changed_line_count > scope.max_changed_lines:
        raise ProposalValidationError(ProposalValidationErrorReason.CHANGED_LINE_LIMIT)
    return ValidatedFocusedChange(
        proposal_id=proposal.proposal_id,
        scope_id=scope.scope_id,
        changed_paths=tuple(changed_paths),
        changed_line_count=changed_line_count,
        replacement_byte_count=replacement_byte_count,
    )


def _validate_line_scope(
    original: str,
    replacement: str,
    targets: tuple[object, ...],
) -> int:
    original_lines = original.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        a=original_lines,
        b=replacement_lines,
        autojunk=False,
    )
    file_level = any(getattr(target, "symbol") is None for target in targets)
    ranges = tuple(
        (getattr(target, "line_start"), getattr(target, "line_end"))
        for target in targets
        if getattr(target, "line_start") is not None
        and getattr(target, "line_end") is not None
    )
    changed = 0
    for (
        tag,
        original_start,
        original_end,
        replacement_start,
        replacement_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += (original_end - original_start) + (
            replacement_end - replacement_start
        )
        if file_level:
            continue
        affected = range(original_start + 1, original_end + 1)
        if any(not _line_in_ranges(line, ranges) for line in affected):
            raise ProposalValidationError(
                ProposalValidationErrorReason.OUT_OF_SCOPE_CHANGE
            )
        if original_start == original_end:
            insertion_line = original_start + 1
            at_eof = original_start == len(original_lines)
            if not _line_in_ranges(insertion_line, ranges) and not (
                at_eof and _line_in_ranges(original_start, ranges)
            ):
                raise ProposalValidationError(
                    ProposalValidationErrorReason.OUT_OF_SCOPE_CHANGE
                )
    return changed


def _line_in_ranges(line: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= line <= end for start, end in ranges)


def _require_only_allowed_ast_changes(
    *,
    path: str,
    original: str,
    replacement: str,
    allowed_symbols: frozenset[str],
) -> None:
    try:
        original_tree = ast.parse(
            original, filename=path, mode="exec", type_comments=True
        )
        replacement_tree = ast.parse(
            replacement,
            filename=path,
            mode="exec",
            type_comments=True,
        )
    except SyntaxError as error:
        raise ProposalValidationError(
            ProposalValidationErrorReason.INVALID_REPLACEMENT_SYNTAX
        ) from error
    module_name = _module_name(path)
    original_masker = _AllowedSymbolMasker(module_name, allowed_symbols)
    replacement_masker = _AllowedSymbolMasker(module_name, allowed_symbols)
    original_masked = original_masker.visit(original_tree)
    replacement_masked = replacement_masker.visit(replacement_tree)
    if (
        original_masker.found != allowed_symbols
        or replacement_masker.found != allowed_symbols
        or ast.dump(original_masked, include_attributes=False)
        != ast.dump(replacement_masked, include_attributes=False)
    ):
        raise ProposalValidationError(
            ProposalValidationErrorReason.OUT_OF_SCOPE_AST_CHANGE
        )


class _AllowedSymbolMasker(ast.NodeTransformer):
    def __init__(self, module_name: str, allowed: frozenset[str]) -> None:
        self._scope = [module_name]
        self._allowed = allowed
        self.found: frozenset[str] = frozenset()
        self._found_mutable: set[str] = set()

    def visit(self, node: ast.AST) -> ast.AST:  # type: ignore[override]
        result = super().visit(node)
        self.found = frozenset(self._found_mutable)
        return result  # type: ignore[return-value]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._visit_definition(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_definition(node)

    def _visit_definition(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.AST:
        qualified_name = ".".join((*self._scope, node.name))
        if qualified_name in self._allowed:
            self._found_mutable.add(qualified_name)
            return ast.Expr(value=ast.Constant(value="__repocompass_allowed_symbol__"))
        self._scope.append(node.name)
        try:
            return self.generic_visit(node)
        finally:
            self._scope.pop()


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)
