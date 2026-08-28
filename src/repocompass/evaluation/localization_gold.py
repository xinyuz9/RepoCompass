"""Offline gold-patch file and Python symbol label derivation.

Patch text is accepted only at this offline boundary. The returned artifact is
payload-free: it retains the patch digest, normalized locations, and derivation
coverage, never the patch body itself.
"""

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation.localization_contracts import (
    LOCALIZATION_SCHEMA_VERSION,
    LocalizationSymbol,
)
from repocompass.tools.contracts import MAX_SAFE_JSON_INTEGER, Sha256Digest

MAX_GOLD_PATCH_BYTES = 4_194_304
MAX_GOLD_BASE_SOURCE_BYTES = 1_048_576
MAX_GOLD_PATCH_LINES = 100_000
_HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?$")
_DEFINITION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?:async[ ]+def|def|class)[ ]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
_SECTION_DEFINITION_PATTERN = re.compile(
    r"(?:^|[ :])(?:async[ ]+def|def|class)[ ]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)


class _GoldModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class GoldPatchLocalization(_GoldModel):
    """Content-addressed locations derived from one gold patch digest."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    derivation_id: Sha256Digest
    source_artifact_digest: Sha256Digest
    derivation_method: Literal["unified_diff_ast_v1"] = "unified_diff_ast_v1"
    files: tuple[str, ...] = Field(min_length=1, max_length=128)
    symbols: tuple[LocalizationSymbol, ...] = Field(default=(), max_length=512)
    python_file_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    ast_source_file_count: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)

    @field_validator("files")
    @classmethod
    def require_canonical_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_repository_path(path)
        if value != tuple(sorted(set(value))):
            raise ValueError("gold patch files must be unique and sorted")
        return value

    @field_validator("symbols")
    @classmethod
    def require_canonical_symbols(
        cls, value: tuple[LocalizationSymbol, ...]
    ) -> tuple[LocalizationSymbol, ...]:
        identities = tuple((item.path, item.qualified_name) for item in value)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("gold patch symbols must be unique and sorted")
        return value

    @model_validator(mode="after")
    def require_content_addressed_derivation(self) -> "GoldPatchLocalization":
        file_set = set(self.files)
        if any(symbol.path not in file_set for symbol in self.symbols):
            raise ValueError("gold patch symbol must reference a changed file")
        expected_python_count = sum(path.endswith(".py") for path in self.files)
        if self.python_file_count != expected_python_count:
            raise ValueError("python file count does not match changed files")
        if self.ast_source_file_count > self.python_file_count:
            raise ValueError("AST source count cannot exceed Python file count")
        expected_id = _derivation_id(
            source_artifact_digest=self.source_artifact_digest,
            files=self.files,
            symbols=self.symbols,
            python_file_count=self.python_file_count,
            ast_source_file_count=self.ast_source_file_count,
        )
        if self.derivation_id != expected_id:
            raise ValueError("derivation_id does not match gold patch locations")
        return self


@dataclass(frozen=True, slots=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FileDiff:
    old_path: str | None
    new_path: str | None
    hunks: tuple[_Hunk, ...]

    @property
    def path(self) -> str:
        candidate = self.new_path if self.new_path is not None else self.old_path
        if candidate is None:
            raise ValueError("diff file has neither an old nor a new repository path")
        return candidate


@dataclass(frozen=True, slots=True)
class _NodeSpan:
    qualified_name: str
    line_start: int
    line_end: int
    column_start: int


def derive_gold_patch_localization(
    patch: str,
    *,
    base_sources: Mapping[str, str],
) -> GoldPatchLocalization:
    """Derive normalized files and AST-compatible symbols from a unified diff."""

    if not isinstance(patch, str):
        raise TypeError("gold patch must be text")
    encoded = patch.encode("utf-8")
    if len(encoded) > MAX_GOLD_PATCH_BYTES:
        raise ValueError(f"gold patch exceeds {MAX_GOLD_PATCH_BYTES} bytes")
    if "\x00" in patch:
        raise ValueError("gold patch must not contain NUL")
    if not isinstance(base_sources, Mapping):
        raise TypeError("base_sources must be a path-to-text mapping")
    normalized_sources: dict[str, str] = {}
    for path, source in base_sources.items():
        _validate_repository_path(path)
        if not isinstance(source, str):
            raise TypeError("base source values must be text")
        if len(source.encode("utf-8")) > MAX_GOLD_BASE_SOURCE_BYTES:
            raise ValueError(
                f"base source exceeds {MAX_GOLD_BASE_SOURCE_BYTES} bytes: {path}"
            )
        if "\x00" in source:
            raise ValueError(f"base source contains NUL: {path}")
        normalized_sources[path] = source

    diffs = _parse_unified_diff(patch)
    files = tuple(sorted({item.path for item in diffs}))
    symbols: set[tuple[str, str]] = set()
    parsed_source_count = 0
    for file_diff in diffs:
        path = file_diff.path
        if not path.endswith(".py"):
            continue
        module_name = _module_name(path)
        spans: tuple[_NodeSpan, ...] = ()
        source = normalized_sources.get(path)
        if source is None and file_diff.old_path is not None:
            source = normalized_sources.get(file_diff.old_path)
        if source is not None:
            spans = _parse_node_spans(source, module_name=module_name)
            if spans:
                parsed_source_count += 1
            else:
                try:
                    ast.parse(source)
                except (SyntaxError, ValueError, RecursionError):
                    pass
                else:
                    parsed_source_count += 1
        for hunk in file_diff.hunks:
            hunk_symbols = _symbols_for_hunk(
                hunk,
                module_name=module_name,
                spans=spans,
            )
            symbols.update((path, name) for name in hunk_symbols)

    canonical_symbols = tuple(
        LocalizationSymbol(path=path, qualified_name=name)
        for path, name in sorted(symbols)
    )
    patch_digest = hashlib.sha256(encoded).hexdigest()
    python_file_count = sum(path.endswith(".py") for path in files)
    derivation_id = _derivation_id(
        source_artifact_digest=patch_digest,
        files=files,
        symbols=canonical_symbols,
        python_file_count=python_file_count,
        ast_source_file_count=parsed_source_count,
    )
    return GoldPatchLocalization(
        derivation_id=derivation_id,
        source_artifact_digest=patch_digest,
        files=files,
        symbols=canonical_symbols,
        python_file_count=python_file_count,
        ast_source_file_count=parsed_source_count,
    )


def _parse_unified_diff(patch: str) -> tuple[_FileDiff, ...]:
    lines = patch.splitlines()
    if len(lines) > MAX_GOLD_PATCH_LINES:
        raise ValueError(f"gold patch exceeds {MAX_GOLD_PATCH_LINES} lines")
    files: list[_FileDiff] = []
    old_path: str | None = None
    new_path: str | None = None
    header_old: str | None = None
    header_new: str | None = None
    hunks: list[_Hunk] = []
    current_hunk: tuple[int, int, int, int, str, list[str]] | None = None
    seen_header = False

    def finish_hunk() -> None:
        nonlocal current_hunk
        if current_hunk is None:
            return
        old_start, old_count, new_start, new_count, section, body = current_hunk
        hunks.append(
            _Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                section=section,
                lines=tuple(body),
            )
        )
        current_hunk = None

    def finish_file() -> None:
        nonlocal old_path, new_path, header_old, header_new, hunks
        finish_hunk()
        if not seen_header or (old_path is None and new_path is None):
            return
        effective_old = old_path if old_path is not None else header_old
        effective_new = new_path if new_path is not None else header_new
        files.append(
            _FileDiff(
                old_path=effective_old,
                new_path=effective_new,
                hunks=tuple(hunks),
            )
        )
        old_path = None
        new_path = None
        header_old = None
        header_new = None
        hunks = []

    for line in lines:
        if line.startswith("diff --git "):
            finish_file()
            match = re.fullmatch(r"diff --git a/([^ ]+) b/([^ ]+)", line)
            if match is None:
                raise ValueError("invalid diff path header")
            header_old = _normalize_patch_path(f"a/{match.group(1)}")
            header_new = _normalize_patch_path(f"b/{match.group(2)}")
            seen_header = True
            continue
        if not seen_header:
            if line.strip():
                continue
            continue
        if line.startswith("--- "):
            finish_hunk()
            old_path = _normalize_patch_path(line[4:].split("\t", 1)[0])
            continue
        if line.startswith("+++ "):
            finish_hunk()
            new_path = _normalize_patch_path(line[4:].split("\t", 1)[0])
            continue
        hunk_match = _HUNK_PATTERN.fullmatch(line)
        if hunk_match is not None:
            finish_hunk()
            old_count = 1 if hunk_match.group(2) is None else int(hunk_match.group(2))
            new_count = 1 if hunk_match.group(4) is None else int(hunk_match.group(4))
            current_hunk = (
                int(hunk_match.group(1)),
                old_count,
                int(hunk_match.group(3)),
                new_count,
                hunk_match.group(5) or "",
                [],
            )
            continue
        if current_hunk is not None and (
            line.startswith((" ", "+", "-", "\\")) or line == ""
        ):
            current_hunk[5].append(line)
    finish_file()
    if not files:
        raise ValueError("gold patch contains no valid diff files")
    return tuple(files)


def _normalize_patch_path(value: str) -> str | None:
    if value == "/dev/null":
        return None
    if value.startswith(("a/", "b/")):
        value = value[2:]
    _validate_repository_path(value)
    return value


def _validate_repository_path(value: str) -> None:
    message = "diff path must be a normalized repository-relative POSIX path"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4_096
        or "\\" in value
        or "\x00" in value
        or any(character in value for character in ("\r", "\n"))
    ):
        raise ValueError(message)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ValueError(message)


def _parse_node_spans(source: str, *, module_name: str) -> tuple[_NodeSpan, ...]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ()
    spans: list[_NodeSpan] = []

    def visit(body: list[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((module_name, *parents, node.name))
                end_line = getattr(node, "end_lineno", None)
                if isinstance(end_line, int):
                    spans.append(
                        _NodeSpan(
                            qualified_name=qualified,
                            line_start=node.lineno,
                            line_end=end_line,
                            column_start=node.col_offset,
                        )
                    )
                visit(node.body, (*parents, node.name))

    visit(tree.body, ())
    return tuple(spans)


def _symbols_for_hunk(
    hunk: _Hunk,
    *,
    module_name: str,
    spans: tuple[_NodeSpan, ...],
) -> tuple[str, ...]:
    old_line = hunk.old_start
    changed_old_lines: set[int] = set()
    insertion_points: list[tuple[int, int, str]] = []
    for line in hunk.lines:
        prefix = line[:1]
        content = line[1:] if prefix in {" ", "+", "-"} else line
        if prefix == " ":
            old_line += 1
        elif prefix == "-":
            changed_old_lines.add(old_line)
            old_line += 1
        elif prefix == "+":
            definition = _DEFINITION_PATTERN.match(content)
            if definition is not None:
                indentation = len(definition.group("indent").expandtabs(8))
                insertion_points.append(
                    (old_line, indentation, definition.group("name"))
                )

    symbols: set[str] = set()
    for line_number in sorted(changed_old_lines):
        span = _smallest_containing_span(spans, line_number=line_number)
        if span is not None:
            symbols.add(span.qualified_name)
    for line_number, indentation, name in insertion_points:
        parent = _smallest_containing_span(
            spans,
            line_number=line_number,
            maximum_column=indentation - 1,
            include_line_after=True,
        )
        prefix = module_name if parent is None else parent.qualified_name
        symbols.add(f"{prefix}.{name}")
    if not symbols and hunk.section:
        section_match = _SECTION_DEFINITION_PATTERN.search(hunk.section)
        if section_match is not None:
            leaf_name = section_match.group("name")
            candidates = tuple(
                span
                for span in spans
                if span.qualified_name.rsplit(".", 1)[-1] == leaf_name
                and span.line_start <= hunk.old_start <= span.line_end + 1
            )
            if candidates:
                symbols.add(
                    min(
                        candidates,
                        key=lambda item: (
                            item.line_end - item.line_start,
                            item.qualified_name,
                        ),
                    ).qualified_name
                )
            else:
                symbols.add(f"{module_name}.{leaf_name}")
    return tuple(sorted(symbols))


def _smallest_containing_span(
    spans: tuple[_NodeSpan, ...],
    *,
    line_number: int,
    maximum_column: int | None = None,
    include_line_after: bool = False,
) -> _NodeSpan | None:
    candidates = []
    for span in spans:
        final_line = span.line_end + int(include_line_after)
        if not (span.line_start <= line_number <= final_line):
            continue
        if maximum_column is not None and span.column_start > maximum_column:
            continue
        candidates.append(span)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.line_end - item.line_start,
            -item.column_start,
            item.qualified_name,
        ),
    )


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _derivation_id(
    *,
    source_artifact_digest: str,
    files: tuple[str, ...],
    symbols: tuple[LocalizationSymbol, ...],
    python_file_count: int,
    ast_source_file_count: int,
) -> str:
    payload = {
        "kind": "gold_patch_localization",
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "source_artifact_digest": source_artifact_digest,
        "derivation_method": "unified_diff_ast_v1",
        "files": list(files),
        "symbols": [
            symbol.model_dump(mode="json", exclude_computed_fields=True)
            for symbol in symbols
        ],
        "python_file_count": python_file_count,
        "ast_source_file_count": ast_source_file_count,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
