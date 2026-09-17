"""Bounded, non-executing Python AST indexing."""

import ast
import hashlib
import io
import os
import re
import stat
import tokenize
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from repocompass.structure.contracts import (
    MAX_INDEXED_SOURCE_BYTES,
    MAX_STRUCTURE_RECORDS,
    PythonFileIndex,
    PythonImportKind,
    PythonImportRecord,
    PythonReferenceKind,
    PythonReferenceRecord,
    PythonSymbolKind,
    PythonSymbolRecord,
    SourceLocation,
    create_import_record,
    create_python_file_index,
    create_reference_record,
    create_source_location,
    create_symbol_record,
)
from repocompass.tools.contracts import COMMIT_SHA_PATTERN, SHA256_PATTERN

MAX_PYTHON_SOURCE_BYTES = MAX_INDEXED_SOURCE_BYTES
MAX_PYTHON_AST_DEPTH = 128
MAX_PYTHON_AST_NODES = 100_000


class PythonIndexErrorReason(StrEnum):
    """Stable, non-sensitive reasons for structural indexing failure."""

    INVALID_REPOSITORY = "invalid_repository"
    INVALID_IDENTITY = "invalid_identity"
    INVALID_PATH = "invalid_path"
    UNSUPPORTED_FILE = "unsupported_file"
    FILE_NOT_FOUND = "file_not_found"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    OVERSIZED_FILE = "oversized_file"
    ENCODING_ERROR = "encoding_error"
    SYNTAX_ERROR = "syntax_error"
    AST_LIMIT = "ast_limit"
    FILE_CHANGED = "file_changed"


class PythonIndexError(ValueError):
    """Fail closed without returning source text or an absolute repository path."""

    def __init__(self, reason: PythonIndexErrorReason) -> None:
        if not isinstance(reason, PythonIndexErrorReason):
            raise TypeError("reason must be PythonIndexErrorReason")
        self.reason = reason
        super().__init__(f"python index failed: {reason.value}")


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    repository_id: str
    commit_sha: str | None
    path: str
    source_digest: str
    source_line_count: int
    module_name: str


@dataclass(frozen=True, slots=True)
class _Scope:
    symbol_id: str
    qualified_name: str
    kind: PythonSymbolKind


def build_python_file_index(
    repository_root: Path,
    relative_path: Path,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> PythonFileIndex:
    """Parse one trusted-snapshot Python file without importing or executing it."""

    _validate_snapshot_identity(repository_id, commit_sha)
    path = _validate_relative_python_path(relative_path)
    source_bytes = _read_bounded_regular_file(repository_root, relative_path)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source, source_encoding = _decode_python_source(source_bytes)
    try:
        tree = ast.parse(
            source,
            filename=path,
            mode="exec",
            type_comments=True,
        )
    except SyntaxError as error:
        raise PythonIndexError(PythonIndexErrorReason.SYNTAX_ERROR) from error
    except (MemoryError, RecursionError) as error:
        raise PythonIndexError(PythonIndexErrorReason.AST_LIMIT) from error
    _enforce_ast_limits(tree)

    source_line_count = source.count("\n") + 1
    identity = _SourceIdentity(
        repository_id=repository_id,
        commit_sha=commit_sha,
        path=path,
        source_digest=source_digest,
        source_line_count=source_line_count,
        module_name=_module_name(path),
    )
    try:
        collector = _PythonStructureCollector(identity, source)
        collector.collect(tree)
    except (ValidationError, RecursionError) as error:
        # Valid Python can exceed our bounded record schema (e.g. a 600-character
        # identifier). Return a body-free indexing failure, not a source-bearing
        # Pydantic exception that aborts candidate evaluation.
        raise PythonIndexError(PythonIndexErrorReason.AST_LIMIT) from error
    if (
        len(collector.symbols) + len(collector.imports) + len(collector.references)
        > MAX_STRUCTURE_RECORDS
    ):
        raise PythonIndexError(PythonIndexErrorReason.AST_LIMIT)
    return create_python_file_index(
        repository_id=repository_id,
        commit_sha=commit_sha,
        path=path,
        module_name=identity.module_name,
        source_digest=source_digest,
        source_encoding=source_encoding,
        source_byte_count=len(source_bytes),
        source_line_count=source_line_count,
        symbols=tuple(collector.symbols),
        imports=tuple(collector.imports),
        references=tuple(collector.references),
    )


class _PythonStructureCollector(ast.NodeVisitor):
    def __init__(self, identity: _SourceIdentity, source: str) -> None:
        self._identity = identity
        self._source = source
        self._scopes: list[_Scope] = []
        self.symbols: list[PythonSymbolRecord] = []
        self.imports: list[PythonImportRecord] = []
        self.references: list[PythonReferenceRecord] = []

    @property
    def _scope(self) -> _Scope:
        if not self._scopes:
            raise RuntimeError("Python structure collector has no active scope")
        return self._scopes[-1]

    def collect(self, tree: ast.Module) -> None:
        module_location = self._module_location()
        module_name = self._identity.module_name.rsplit(".", 1)[-1]
        module = create_symbol_record(
            name=module_name,
            qualified_name=self._identity.module_name,
            kind=PythonSymbolKind.MODULE,
            parent_symbol_id=None,
            definition=module_location,
            body=module_location,
            is_test=_is_test_symbol(self._identity.path, module_name),
        )
        self.symbols.append(module)
        self._scopes.append(
            _Scope(
                symbol_id=module.symbol_id,
                qualified_name=module.qualified_name,
                kind=module.kind,
            )
        )
        try:
            for statement in tree.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        record = self._create_definition(node, kind=PythonSymbolKind.CLASS)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)
        self._visit_scoped_body(record, node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = (
            PythonSymbolKind.METHOD
            if self._scope.kind is PythonSymbolKind.CLASS
            else PythonSymbolKind.FUNCTION
        )
        record = self._create_definition(node, kind=kind)
        self._visit_function_outer_expressions(node)
        self._visit_scoped_body(record, node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = (
            PythonSymbolKind.ASYNC_METHOD
            if self._scope.kind is PythonSymbolKind.CLASS
            else PythonSymbolKind.ASYNC_FUNCTION
        )
        record = self._create_definition(node, kind=kind)
        self._visit_function_outer_expressions(node)
        self._visit_scoped_body(record, node.body)

    def visit_Import(self, node: ast.Import) -> None:
        statement_location = self._node_location(node)
        for alias in node.names:
            location = (
                self._node_location(alias)
                if hasattr(alias, "lineno")
                else statement_location
            )
            self.imports.append(
                create_import_record(
                    kind=PythonImportKind.IMPORT,
                    scope_symbol_id=self._scope.symbol_id,
                    module=alias.name,
                    imported_name=alias.name,
                    alias=alias.asname,
                    bound_name=alias.asname or alias.name.split(".", 1)[0],
                    level=0,
                    location=location,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        statement_location = self._node_location(node)
        for alias in node.names:
            location = (
                self._node_location(alias)
                if hasattr(alias, "lineno")
                else statement_location
            )
            self.imports.append(
                create_import_record(
                    kind=PythonImportKind.FROM_IMPORT,
                    scope_symbol_id=self._scope.symbol_id,
                    module=node.module,
                    imported_name=alias.name,
                    alias=alias.asname,
                    bound_name=alias.asname or alias.name,
                    level=node.level,
                    location=location,
                )
            )

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.references.append(
                create_reference_record(
                    kind=PythonReferenceKind.NAME_LOAD,
                    scope_symbol_id=self._scope.symbol_id,
                    name=node.id,
                    location=self._node_location(node),
                )
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            name = _expression_name(node)
            if name is not None:
                self.references.append(
                    create_reference_record(
                        kind=PythonReferenceKind.ATTRIBUTE_LOAD,
                        scope_symbol_id=self._scope.symbol_id,
                        name=name,
                        location=self._node_location(node),
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _expression_name(node.func)
        if name is not None:
            self.references.append(
                create_reference_record(
                    kind=PythonReferenceKind.CALL,
                    scope_symbol_id=self._scope.symbol_id,
                    name=name,
                    location=self._node_location(node.func),
                )
            )
        self.generic_visit(node)

    def _create_definition(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        kind: PythonSymbolKind,
    ) -> PythonSymbolRecord:
        qualified_name = f"{self._scope.qualified_name}.{node.name}"
        record = create_symbol_record(
            name=node.name,
            qualified_name=qualified_name,
            kind=kind,
            parent_symbol_id=self._scope.symbol_id,
            definition=self._node_location(node),
            body=self._body_location(node.body),
            decorators=tuple(
                name
                for decorator in node.decorator_list
                if (name := _decorator_name(decorator)) is not None
            ),
            is_test=_is_test_symbol(self._identity.path, node.name),
        )
        self.symbols.append(record)
        return record

    def _visit_scoped_body(
        self,
        record: PythonSymbolRecord,
        body: list[ast.stmt],
    ) -> None:
        self._scopes.append(
            _Scope(
                symbol_id=record.symbol_id,
                qualified_name=record.qualified_name,
                kind=record.kind,
            )
        )
        try:
            for statement in body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def _visit_function_outer_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if node.args.vararg is not None:
            arguments = (*arguments, node.args.vararg)
        if node.args.kwarg is not None:
            arguments = (*arguments, node.args.kwarg)
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _module_location(self) -> SourceLocation:
        last_line = self._source.rsplit("\n", 1)[-1]
        return self._location(
            line_start=1,
            column_start=0,
            line_end=self._identity.source_line_count,
            column_end=len(last_line.encode("utf-8")),
        )

    def _body_location(self, body: list[ast.stmt]) -> SourceLocation | None:
        if not body:
            return None
        first = body[0]
        last = body[-1]
        return self._location(
            line_start=first.lineno,
            column_start=first.col_offset,
            line_end=last.end_lineno or last.lineno,
            column_end=last.end_col_offset or last.col_offset,
        )

    def _node_location(self, node: ast.AST) -> SourceLocation:
        line_start = getattr(node, "lineno", None)
        column_start = getattr(node, "col_offset", None)
        line_end = getattr(node, "end_lineno", None)
        column_end = getattr(node, "end_col_offset", None)
        if not isinstance(line_start, int) or not isinstance(column_start, int):
            raise PythonIndexError(PythonIndexErrorReason.AST_LIMIT)
        return self._location(
            line_start=line_start,
            column_start=column_start,
            line_end=line_start if line_end is None else line_end,
            column_end=column_start if column_end is None else column_end,
        )

    def _location(
        self,
        *,
        line_start: int,
        column_start: int,
        line_end: int,
        column_end: int,
    ) -> SourceLocation:
        return create_source_location(
            repository_id=self._identity.repository_id,
            commit_sha=self._identity.commit_sha,
            path=self._identity.path,
            source_digest=self._identity.source_digest,
            line_start=line_start,
            column_start=column_start,
            line_end=line_end,
            column_end=column_end,
        )


def _validate_snapshot_identity(repository_id: str, commit_sha: str | None) -> None:
    if (
        not isinstance(repository_id, str)
        or re.fullmatch(
            SHA256_PATTERN,
            repository_id,
        )
        is None
    ):
        raise PythonIndexError(PythonIndexErrorReason.INVALID_IDENTITY)
    if commit_sha is not None and (
        not isinstance(commit_sha, str)
        or re.fullmatch(COMMIT_SHA_PATTERN, commit_sha) is None
    ):
        raise PythonIndexError(PythonIndexErrorReason.INVALID_IDENTITY)


def _validate_relative_python_path(relative_path: Path) -> str:
    if not isinstance(relative_path, Path) or relative_path.is_absolute():
        raise PythonIndexError(PythonIndexErrorReason.INVALID_PATH)
    value = relative_path.as_posix()
    path = PurePosixPath(value)
    if (
        value in {"", "."}
        or "\\" in value
        or "\x00" in value
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise PythonIndexError(PythonIndexErrorReason.INVALID_PATH)
    if path.suffix != ".py":
        raise PythonIndexError(PythonIndexErrorReason.UNSUPPORTED_FILE)
    return value


def _read_bounded_regular_file(
    repository_root: Path,
    relative_path: Path,
) -> bytes:
    if not isinstance(repository_root, Path):
        raise PythonIndexError(PythonIndexErrorReason.INVALID_REPOSITORY)
    try:
        resolved_root = repository_root.resolve(strict=True)
    except OSError as error:
        raise PythonIndexError(PythonIndexErrorReason.INVALID_REPOSITORY) from error
    if not resolved_root.is_dir():
        raise PythonIndexError(PythonIndexErrorReason.INVALID_REPOSITORY)
    candidate = resolved_root / relative_path
    try:
        initial_stat = candidate.lstat()
    except FileNotFoundError as error:
        raise PythonIndexError(PythonIndexErrorReason.FILE_NOT_FOUND) from error
    except OSError as error:
        raise PythonIndexError(PythonIndexErrorReason.SPECIAL_FILE) from error
    if stat.S_ISLNK(initial_stat.st_mode):
        raise PythonIndexError(PythonIndexErrorReason.SYMLINK)
    if not stat.S_ISREG(initial_stat.st_mode):
        raise PythonIndexError(PythonIndexErrorReason.SPECIAL_FILE)
    try:
        resolved_target = candidate.resolve(strict=True)
        resolved_target.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise PythonIndexError(PythonIndexErrorReason.INVALID_PATH) from error
    if initial_stat.st_size > MAX_PYTHON_SOURCE_BYTES:
        raise PythonIndexError(PythonIndexErrorReason.OVERSIZED_FILE)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved_target, flags)
    except OSError as error:
        raise PythonIndexError(PythonIndexErrorReason.FILE_CHANGED) from error
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise PythonIndexError(PythonIndexErrorReason.SPECIAL_FILE)
        if (
            opened_stat.st_dev != initial_stat.st_dev
            or opened_stat.st_ino != initial_stat.st_ino
        ):
            raise PythonIndexError(PythonIndexErrorReason.FILE_CHANGED)
        if opened_stat.st_size > MAX_PYTHON_SOURCE_BYTES:
            raise PythonIndexError(PythonIndexErrorReason.OVERSIZED_FILE)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(MAX_PYTHON_SOURCE_BYTES + 1)
        finished_stat = os.fstat(descriptor)
        if (
            finished_stat.st_size != opened_stat.st_size
            or finished_stat.st_mtime_ns != opened_stat.st_mtime_ns
            or finished_stat.st_dev != opened_stat.st_dev
            or finished_stat.st_ino != opened_stat.st_ino
        ):
            raise PythonIndexError(PythonIndexErrorReason.FILE_CHANGED)
    finally:
        os.close(descriptor)
    if len(content) > MAX_PYTHON_SOURCE_BYTES:
        raise PythonIndexError(PythonIndexErrorReason.OVERSIZED_FILE)
    return content


def _decode_python_source(source: bytes) -> tuple[str, str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        decoded = source.decode(encoding)
    except (LookupError, SyntaxError, UnicodeDecodeError) as error:
        raise PythonIndexError(PythonIndexErrorReason.ENCODING_ERROR) from error
    return decoded, encoding.lower()


def _enforce_ast_limits(tree: ast.AST) -> None:
    count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > MAX_PYTHON_AST_DEPTH or count > MAX_PYTHON_AST_NODES:
            raise PythonIndexError(PythonIndexErrorReason.AST_LIMIT)
        children = tuple(ast.iter_child_nodes(node))
        stack.extend((child, depth + 1) for child in reversed(children))


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _expression_name(node.func)
    return _expression_name(node)


def _expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _is_test_symbol(path: str, name: str) -> bool:
    parts = PurePosixPath(path).parts
    filename = parts[-1]
    return (
        filename.startswith("test_")
        or any(part in {"test", "tests"} for part in parts[:-1])
        or name.startswith("test_")
    )
