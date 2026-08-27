"""Strict content-addressed contracts for Python structural evidence."""

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import (
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

STRUCTURE_SCHEMA_VERSION = 1
MAX_STRUCTURE_RECORDS = 20_000
MAX_STRUCTURAL_NAME_CHARACTERS = 2_048
MAX_INDEXED_SOURCE_BYTES = 1_048_576


class StructuralVerificationStatus(StrEnum):
    """Closed status vocabulary for graph-ready structural facts."""

    VERIFIED = "verified"
    INFERRED = "inferred"
    REJECTED = "rejected"


class PythonSymbolKind(StrEnum):
    """Python definitions distinguished by syntax and immediate scope."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"


class PythonImportKind(StrEnum):
    """Import statement forms represented one alias at a time."""

    IMPORT = "import"
    FROM_IMPORT = "from_import"


class PythonReferenceKind(StrEnum):
    """Deterministically observable expression-level references."""

    NAME_LOAD = "name_load"
    ATTRIBUTE_LOAD = "attribute_load"
    CALL = "call"


class _StructureContractModel(CheckpointSafeModel):
    """Strict immutable base for checkpoint- and JSON-safe structure records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        validate_default=True,
    )


class SourceLocation(_StructureContractModel):
    """A byte-column source span inside one immutable repository snapshot."""

    schema_version: Literal[1] = STRUCTURE_SCHEMA_VERSION
    location_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    path: str = Field(min_length=1, max_length=4_096)
    source_digest: Sha256Digest
    line_start: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    column_start: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    line_end: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    column_end: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)

    @field_validator("path")
    @classmethod
    def require_normalized_python_path(cls, value: str) -> str:
        _validate_python_path(value)
        return value

    @model_validator(mode="after")
    def require_valid_content_addressed_span(self) -> "SourceLocation":
        start = (self.line_start, self.column_start)
        end = (self.line_end, self.column_end)
        if end < start:
            raise ValueError("location end must not precede its start")
        expected = _location_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            path=self.path,
            source_digest=self.source_digest,
            line_start=self.line_start,
            column_start=self.column_start,
            line_end=self.line_end,
            column_end=self.column_end,
        )
        if self.location_id != expected:
            raise ValueError("location_id does not match source location")
        return self


class PythonSymbolRecord(_StructureContractModel):
    """One AST-verified module, class, function, or method definition."""

    schema_version: Literal[1] = STRUCTURE_SCHEMA_VERSION
    symbol_id: Sha256Digest
    name: str = Field(min_length=1, max_length=512)
    qualified_name: str = Field(
        min_length=1,
        max_length=MAX_STRUCTURAL_NAME_CHARACTERS,
    )
    kind: PythonSymbolKind
    verification_status: Literal[StructuralVerificationStatus.VERIFIED] = (
        StructuralVerificationStatus.VERIFIED
    )
    parent_symbol_id: Sha256Digest | None = None
    definition: SourceLocation
    body: SourceLocation | None = None
    decorators: tuple[str, ...] = Field(default=(), max_length=64)
    is_test: bool = False

    @field_validator("name", "qualified_name")
    @classmethod
    def reject_unsafe_names(cls, value: str) -> str:
        _validate_structural_text(value, name="symbol name")
        return value

    @field_validator("decorators")
    @classmethod
    def require_bounded_decorators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for decorator in value:
            _validate_structural_text(decorator, name="decorator")
            if len(decorator) > 512:
                raise ValueError("decorator exceeds the character limit")
        return value

    @model_validator(mode="after")
    def require_verified_content_addressed_symbol(self) -> "PythonSymbolRecord":
        if self.kind is not PythonSymbolKind.MODULE and not self.name.isidentifier():
            raise ValueError("non-module symbol names must be Python identifiers")
        if self.parent_symbol_id == self.symbol_id:
            raise ValueError("symbol cannot contain itself")
        if self.body is not None:
            _require_same_source(self.definition, self.body, name="symbol body")
            definition_start = (
                self.definition.line_start,
                self.definition.column_start,
            )
            definition_end = (self.definition.line_end, self.definition.column_end)
            body_start = (self.body.line_start, self.body.column_start)
            body_end = (self.body.line_end, self.body.column_end)
            if body_start < definition_start or body_end > definition_end:
                raise ValueError("symbol body must be inside its definition")
        expected = _symbol_id(
            name=self.name,
            qualified_name=self.qualified_name,
            kind=self.kind,
            parent_symbol_id=self.parent_symbol_id,
            definition=self.definition,
            body=self.body,
            decorators=self.decorators,
            is_test=self.is_test,
        )
        if self.symbol_id != expected:
            raise ValueError("symbol_id does not match symbol contents")
        return self


class PythonImportRecord(_StructureContractModel):
    """One imported name with its lexical scope and exact statement span."""

    schema_version: Literal[1] = STRUCTURE_SCHEMA_VERSION
    import_id: Sha256Digest
    kind: PythonImportKind
    verification_status: Literal[StructuralVerificationStatus.VERIFIED] = (
        StructuralVerificationStatus.VERIFIED
    )
    scope_symbol_id: Sha256Digest
    module: str | None = Field(default=None, max_length=MAX_STRUCTURAL_NAME_CHARACTERS)
    imported_name: str = Field(min_length=1, max_length=MAX_STRUCTURAL_NAME_CHARACTERS)
    alias: str | None = Field(default=None, min_length=1, max_length=512)
    bound_name: str = Field(min_length=1, max_length=512)
    level: int = Field(default=0, ge=0, le=1_024)
    location: SourceLocation

    @field_validator("module", "imported_name", "alias", "bound_name")
    @classmethod
    def reject_unsafe_import_names(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_structural_text(value, name="import name")
        return value

    @model_validator(mode="after")
    def require_consistent_content_addressed_import(self) -> "PythonImportRecord":
        if self.kind is PythonImportKind.IMPORT:
            if self.level != 0 or self.module != self.imported_name:
                raise ValueError(
                    "import records require level zero and matching module"
                )
        elif self.module is None and self.level == 0:
            raise ValueError("absolute from-import requires a module")
        expected = _import_id(
            kind=self.kind,
            scope_symbol_id=self.scope_symbol_id,
            module=self.module,
            imported_name=self.imported_name,
            alias=self.alias,
            bound_name=self.bound_name,
            level=self.level,
            location=self.location,
        )
        if self.import_id != expected:
            raise ValueError("import_id does not match import contents")
        return self


class PythonReferenceRecord(_StructureContractModel):
    """One syntactically verified load, attribute, or call reference."""

    schema_version: Literal[1] = STRUCTURE_SCHEMA_VERSION
    reference_id: Sha256Digest
    kind: PythonReferenceKind
    verification_status: Literal[StructuralVerificationStatus.VERIFIED] = (
        StructuralVerificationStatus.VERIFIED
    )
    scope_symbol_id: Sha256Digest
    name: str = Field(min_length=1, max_length=MAX_STRUCTURAL_NAME_CHARACTERS)
    location: SourceLocation

    @field_validator("name")
    @classmethod
    def reject_unsafe_reference_name(cls, value: str) -> str:
        _validate_structural_text(value, name="reference name")
        return value

    @model_validator(mode="after")
    def require_content_addressed_reference(self) -> "PythonReferenceRecord":
        expected = _reference_id(
            kind=self.kind,
            scope_symbol_id=self.scope_symbol_id,
            name=self.name,
            location=self.location,
        )
        if self.reference_id != expected:
            raise ValueError("reference_id does not match reference contents")
        return self


class PythonFileIndex(_StructureContractModel):
    """Canonical AST facts for one bounded Python source file."""

    schema_version: Literal[1] = STRUCTURE_SCHEMA_VERSION
    index_id: Sha256Digest
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    path: str = Field(min_length=1, max_length=4_096)
    module_name: str = Field(min_length=1, max_length=MAX_STRUCTURAL_NAME_CHARACTERS)
    source_digest: Sha256Digest
    source_encoding: str = Field(min_length=1, max_length=128)
    source_byte_count: int = Field(ge=0, le=MAX_INDEXED_SOURCE_BYTES)
    source_line_count: int = Field(
        ge=1,
        le=MAX_INDEXED_SOURCE_BYTES + 1,
    )
    symbols: tuple[PythonSymbolRecord, ...] = Field(
        min_length=1,
        max_length=MAX_STRUCTURE_RECORDS,
    )
    imports: tuple[PythonImportRecord, ...] = Field(
        default=(),
        max_length=MAX_STRUCTURE_RECORDS,
    )
    references: tuple[PythonReferenceRecord, ...] = Field(
        default=(),
        max_length=MAX_STRUCTURE_RECORDS,
    )

    @field_validator("path")
    @classmethod
    def require_normalized_index_path(cls, value: str) -> str:
        _validate_python_path(value)
        return value

    @field_validator("module_name", "source_encoding")
    @classmethod
    def reject_unsafe_index_text(cls, value: str) -> str:
        _validate_structural_text(value, name="index text")
        return value

    @model_validator(mode="after")
    def reconcile_index_facts_and_identity(self) -> "PythonFileIndex":
        if (
            len(self.symbols) + len(self.imports) + len(self.references)
            > MAX_STRUCTURE_RECORDS
        ):
            raise ValueError("combined structure records exceed the index limit")
        expected_module_name = _module_name_from_path(self.path)
        if self.module_name != expected_module_name:
            raise ValueError("module_name does not match the indexed path")
        if self.symbols != canonical_symbols(self.symbols):
            raise ValueError("symbols must be in canonical order")
        if self.imports != canonical_imports(self.imports):
            raise ValueError("imports must be in canonical order")
        if self.references != canonical_references(self.references):
            raise ValueError("references must be in canonical order")
        _require_unique_ids(
            tuple(record.symbol_id for record in self.symbols),
            name="symbol",
        )
        _require_unique_ids(
            tuple(record.import_id for record in self.imports),
            name="import",
        )
        _require_unique_ids(
            tuple(record.reference_id for record in self.references),
            name="reference",
        )
        symbols_by_id = {record.symbol_id: record for record in self.symbols}
        module_symbols = [
            record for record in self.symbols if record.kind is PythonSymbolKind.MODULE
        ]
        if len(module_symbols) != 1:
            raise ValueError("index requires exactly one module symbol")
        module = module_symbols[0]
        if (
            module.parent_symbol_id is not None
            or module.qualified_name != self.module_name
            or module.name != self.module_name.rsplit(".", 1)[-1]
            or (module.definition.line_start, module.definition.column_start) != (1, 0)
            or module.definition.line_end != self.source_line_count
        ):
            raise ValueError("module symbol does not match index identity")
        for symbol in self.symbols:
            self._require_index_location(symbol.definition, name="symbol definition")
            if symbol.body is not None:
                self._require_index_location(symbol.body, name="symbol body")
            if symbol.kind is PythonSymbolKind.MODULE:
                continue
            parent = symbols_by_id.get(symbol.parent_symbol_id or "")
            if parent is None:
                raise ValueError("symbol parent does not exist in this index")
            if symbol.qualified_name != f"{parent.qualified_name}.{symbol.name}":
                raise ValueError("symbol qualified_name does not match its parent")
            if not _span_contains(parent.definition, symbol.definition):
                raise ValueError("symbol definition is outside its parent")
            is_method = symbol.kind in {
                PythonSymbolKind.METHOD,
                PythonSymbolKind.ASYNC_METHOD,
            }
            if is_method != (parent.kind is PythonSymbolKind.CLASS):
                raise ValueError("method kind does not match the immediate parent")
        for record in (*self.imports, *self.references):
            scope = symbols_by_id.get(record.scope_symbol_id)
            if scope is None:
                raise ValueError("record scope does not exist in this index")
            self._require_index_location(record.location, name="record")
            if not _span_contains(scope.definition, record.location):
                raise ValueError("record location is outside its lexical scope")
        expected = _file_index_id(
            repository_id=self.repository_id,
            commit_sha=self.commit_sha,
            path=self.path,
            module_name=self.module_name,
            source_digest=self.source_digest,
            source_encoding=self.source_encoding,
            source_byte_count=self.source_byte_count,
            source_line_count=self.source_line_count,
            symbols=self.symbols,
            imports=self.imports,
            references=self.references,
        )
        if self.index_id != expected:
            raise ValueError("index_id does not match Python file index contents")
        return self

    def _require_index_location(self, location: SourceLocation, *, name: str) -> None:
        if (
            location.repository_id != self.repository_id
            or location.commit_sha != self.commit_sha
            or location.path != self.path
            or location.source_digest != self.source_digest
        ):
            raise ValueError(f"{name} has mismatched source provenance")
        if location.line_end > self.source_line_count:
            raise ValueError(f"{name} line exceeds source_line_count")


def create_source_location(
    *,
    repository_id: str,
    commit_sha: str | None,
    path: str,
    source_digest: str,
    line_start: int,
    column_start: int,
    line_end: int,
    column_end: int,
) -> SourceLocation:
    """Create one content-addressed source span."""

    return SourceLocation(
        location_id=_location_id(
            repository_id=repository_id,
            commit_sha=commit_sha,
            path=path,
            source_digest=source_digest,
            line_start=line_start,
            column_start=column_start,
            line_end=line_end,
            column_end=column_end,
        ),
        repository_id=repository_id,
        commit_sha=commit_sha,
        path=path,
        source_digest=source_digest,
        line_start=line_start,
        column_start=column_start,
        line_end=line_end,
        column_end=column_end,
    )


def create_symbol_record(
    *,
    name: str,
    qualified_name: str,
    kind: PythonSymbolKind,
    parent_symbol_id: str | None,
    definition: SourceLocation,
    body: SourceLocation | None,
    decorators: tuple[str, ...] = (),
    is_test: bool = False,
) -> PythonSymbolRecord:
    """Create one AST-verified content-addressed symbol record."""

    return PythonSymbolRecord(
        symbol_id=_symbol_id(
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            parent_symbol_id=parent_symbol_id,
            definition=definition,
            body=body,
            decorators=decorators,
            is_test=is_test,
        ),
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        verification_status=StructuralVerificationStatus.VERIFIED,
        parent_symbol_id=parent_symbol_id,
        definition=definition,
        body=body,
        decorators=decorators,
        is_test=is_test,
    )


def create_import_record(
    *,
    kind: PythonImportKind,
    scope_symbol_id: str,
    module: str | None,
    imported_name: str,
    alias: str | None,
    bound_name: str,
    level: int,
    location: SourceLocation,
) -> PythonImportRecord:
    """Create one AST-verified import alias record."""

    return PythonImportRecord(
        import_id=_import_id(
            kind=kind,
            scope_symbol_id=scope_symbol_id,
            module=module,
            imported_name=imported_name,
            alias=alias,
            bound_name=bound_name,
            level=level,
            location=location,
        ),
        kind=kind,
        verification_status=StructuralVerificationStatus.VERIFIED,
        scope_symbol_id=scope_symbol_id,
        module=module,
        imported_name=imported_name,
        alias=alias,
        bound_name=bound_name,
        level=level,
        location=location,
    )


def create_reference_record(
    *,
    kind: PythonReferenceKind,
    scope_symbol_id: str,
    name: str,
    location: SourceLocation,
) -> PythonReferenceRecord:
    """Create one AST-verified expression reference."""

    return PythonReferenceRecord(
        reference_id=_reference_id(
            kind=kind,
            scope_symbol_id=scope_symbol_id,
            name=name,
            location=location,
        ),
        kind=kind,
        verification_status=StructuralVerificationStatus.VERIFIED,
        scope_symbol_id=scope_symbol_id,
        name=name,
        location=location,
    )


def create_python_file_index(
    *,
    repository_id: str,
    commit_sha: str | None,
    path: str,
    module_name: str,
    source_digest: str,
    source_encoding: str,
    source_byte_count: int,
    source_line_count: int,
    symbols: tuple[PythonSymbolRecord, ...],
    imports: tuple[PythonImportRecord, ...],
    references: tuple[PythonReferenceRecord, ...],
) -> PythonFileIndex:
    """Create a canonical content-addressed file index."""

    ordered_symbols = canonical_symbols(symbols)
    ordered_imports = canonical_imports(imports)
    ordered_references = canonical_references(references)
    return PythonFileIndex(
        index_id=_file_index_id(
            repository_id=repository_id,
            commit_sha=commit_sha,
            path=path,
            module_name=module_name,
            source_digest=source_digest,
            source_encoding=source_encoding,
            source_byte_count=source_byte_count,
            source_line_count=source_line_count,
            symbols=ordered_symbols,
            imports=ordered_imports,
            references=ordered_references,
        ),
        repository_id=repository_id,
        commit_sha=commit_sha,
        path=path,
        module_name=module_name,
        source_digest=source_digest,
        source_encoding=source_encoding,
        source_byte_count=source_byte_count,
        source_line_count=source_line_count,
        symbols=ordered_symbols,
        imports=ordered_imports,
        references=ordered_references,
    )


def canonical_symbols(
    records: tuple[PythonSymbolRecord, ...],
) -> tuple[PythonSymbolRecord, ...]:
    """Return deterministic parent-before-child source order."""

    return tuple(
        sorted(
            records,
            key=lambda record: (
                0 if record.kind is PythonSymbolKind.MODULE else 1,
                record.definition.line_start,
                record.definition.column_start,
                record.definition.line_end,
                record.definition.column_end,
                record.kind.value,
                record.qualified_name,
                record.symbol_id,
            ),
        )
    )


def canonical_imports(
    records: tuple[PythonImportRecord, ...],
) -> tuple[PythonImportRecord, ...]:
    """Return deterministic lexical import order."""

    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.location.line_start,
                record.location.column_start,
                record.kind.value,
                record.module or "",
                record.imported_name,
                record.alias or "",
                record.import_id,
            ),
        )
    )


def canonical_references(
    records: tuple[PythonReferenceRecord, ...],
) -> tuple[PythonReferenceRecord, ...]:
    """Return deterministic lexical reference order."""

    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.location.line_start,
                record.location.column_start,
                record.location.line_end,
                record.location.column_end,
                record.kind.value,
                record.name,
                record.reference_id,
            ),
        )
    )


def _location_id(**payload: object) -> str:
    return _canonical_hash(b"repocompass:source-location:v1", payload)


def _symbol_id(
    *,
    name: str,
    qualified_name: str,
    kind: PythonSymbolKind,
    parent_symbol_id: str | None,
    definition: SourceLocation,
    body: SourceLocation | None,
    decorators: tuple[str, ...],
    is_test: bool,
) -> str:
    return _canonical_hash(
        b"repocompass:python-symbol:v1",
        {
            "name": name,
            "qualified_name": qualified_name,
            "kind": kind.value,
            "parent_symbol_id": parent_symbol_id,
            "definition": definition.model_dump(mode="json"),
            "body": None if body is None else body.model_dump(mode="json"),
            "decorators": decorators,
            "is_test": is_test,
        },
    )


def _import_id(
    *,
    kind: PythonImportKind,
    scope_symbol_id: str,
    module: str | None,
    imported_name: str,
    alias: str | None,
    bound_name: str,
    level: int,
    location: SourceLocation,
) -> str:
    return _canonical_hash(
        b"repocompass:python-import:v1",
        {
            "kind": kind.value,
            "scope_symbol_id": scope_symbol_id,
            "module": module,
            "imported_name": imported_name,
            "alias": alias,
            "bound_name": bound_name,
            "level": level,
            "location": location.model_dump(mode="json"),
        },
    )


def _reference_id(
    *,
    kind: PythonReferenceKind,
    scope_symbol_id: str,
    name: str,
    location: SourceLocation,
) -> str:
    return _canonical_hash(
        b"repocompass:python-reference:v1",
        {
            "kind": kind.value,
            "scope_symbol_id": scope_symbol_id,
            "name": name,
            "location": location.model_dump(mode="json"),
        },
    )


def _file_index_id(
    *,
    repository_id: str,
    commit_sha: str | None,
    path: str,
    module_name: str,
    source_digest: str,
    source_encoding: str,
    source_byte_count: int,
    source_line_count: int,
    symbols: tuple[PythonSymbolRecord, ...],
    imports: tuple[PythonImportRecord, ...],
    references: tuple[PythonReferenceRecord, ...],
) -> str:
    return _canonical_hash(
        b"repocompass:python-file-index:v1",
        {
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "path": path,
            "module_name": module_name,
            "source_digest": source_digest,
            "source_encoding": source_encoding,
            "source_byte_count": source_byte_count,
            "source_line_count": source_line_count,
            "symbols": [record.model_dump(mode="json") for record in symbols],
            "imports": [record.model_dump(mode="json") for record in imports],
            "references": [record.model_dump(mode="json") for record in references],
        },
    )


def _canonical_hash(namespace: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()


def _require_same_source(
    first: SourceLocation,
    second: SourceLocation,
    *,
    name: str,
) -> None:
    first_identity = (
        first.repository_id,
        first.commit_sha,
        first.path,
        first.source_digest,
    )
    second_identity = (
        second.repository_id,
        second.commit_sha,
        second.path,
        second.source_digest,
    )
    if first_identity != second_identity:
        raise ValueError(f"{name} must use the same source identity")


def _span_contains(container: SourceLocation, contained: SourceLocation) -> bool:
    _require_same_source(container, contained, name="contained location")
    container_start = (container.line_start, container.column_start)
    container_end = (container.line_end, container.column_end)
    contained_start = (contained.line_start, contained.column_start)
    contained_end = (contained.line_end, contained.column_end)
    return container_start <= contained_start and contained_end <= container_end


def _module_name_from_path(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _require_unique_ids(values: tuple[str, ...], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} IDs must be unique")


def _validate_python_path(value: str) -> None:
    if "\\" in value or "\x00" in value:
        raise ValueError("path must be a normalized POSIX Python path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or ".." in path.parts
        or path.as_posix() != value
        or path.suffix != ".py"
    ):
        raise ValueError("path must be a normalized relative .py path")


def _validate_structural_text(value: str, *, name: str) -> None:
    if "\x00" in value or any(character in value for character in ("\r", "\n")):
        raise ValueError(f"{name} contains unsafe control characters")
