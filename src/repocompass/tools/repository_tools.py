"""Typed arguments, descriptors, and normalized repository tool executors."""

import json
import keyword
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repocompass.structure.queries import (
    MAX_PYTHON_SCAN_FILES,
    MAX_STRUCTURAL_QUERY_CHARACTERS,
    MAX_STRUCTURAL_RESULTS,
    find_python_references,
    find_python_symbol_definitions,
    inspect_python_imports,
    list_python_symbols,
)
from repocompass.tools.contracts import (
    ToolCapability,
    ToolDescriptor,
    ToolRiskLevel,
    ToolSideEffect,
)
from repocompass.tools.repository import list_tree, read_code, search_text

MAX_LIST_TREE_DEPTH = 8
MAX_LIST_TREE_ENTRIES = 200
MAX_SEARCH_TEXT_RESULTS = 100
MAX_SEARCH_QUERY_CHARACTERS = 4_096
MAX_SEARCH_GLOBS = 16
MAX_GLOB_CHARACTERS = 256
MAX_RELATIVE_PATH_CHARACTERS = 4_096
MAX_READ_CODE_LINES = 400


class _RepositoryToolArguments(BaseModel):
    """Strict immutable base for model-controlled repository arguments."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class ListTreeArguments(_RepositoryToolArguments):
    """Bounded arguments accepted by ``list_tree``."""

    max_depth: int = Field(default=3, ge=0, le=MAX_LIST_TREE_DEPTH)
    max_entries: int = Field(default=100, ge=1, le=MAX_LIST_TREE_ENTRIES)


GlobPattern = Annotated[
    str,
    Field(min_length=1, max_length=MAX_GLOB_CHARACTERS),
]
GlobPatterns = Annotated[
    tuple[GlobPattern, ...],
    Field(max_length=MAX_SEARCH_GLOBS),
]


class SearchTextArguments(_RepositoryToolArguments):
    """Bounded literal-search arguments accepted by ``search_text``."""

    query: str = Field(min_length=1, max_length=MAX_SEARCH_QUERY_CHARACTERS)
    globs: GlobPatterns | None = None
    max_results: int = Field(default=100, ge=1, le=MAX_SEARCH_TEXT_RESULTS)

    @field_validator("query")
    @classmethod
    def reject_nul_query(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("query must not contain NUL")
        return value

    @field_validator("globs")
    @classmethod
    def reject_unstable_globs(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if any("\x00" in glob for glob in value):
            raise ValueError("globs must not contain NUL")
        return value


class ReadCodeArguments(_RepositoryToolArguments):
    """Bounded repository-relative line range accepted by ``read_code``."""

    relative_path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH_CHARACTERS)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def reject_nul_path(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("relative_path must not contain NUL")
        return value

    @model_validator(mode="after")
    def require_bounded_line_range(self) -> "ReadCodeArguments":
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        if self.end_line - self.start_line + 1 > MAX_READ_CODE_LINES:
            raise ValueError(f"read_code is limited to {MAX_READ_CODE_LINES} lines")
        return self


class _PythonFileArguments(_RepositoryToolArguments):
    relative_path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH_CHARACTERS)

    @field_validator("relative_path")
    @classmethod
    def require_python_path(cls, value: str) -> str:
        _validate_python_tool_path(value)
        return value


class ListSymbolsArguments(_PythonFileArguments):
    """Bounded arguments accepted by ``list_symbols``."""

    max_results: int = Field(default=100, ge=1, le=MAX_STRUCTURAL_RESULTS)


class InspectImportsArguments(_PythonFileArguments):
    """Bounded arguments accepted by ``inspect_imports``."""

    max_results: int = Field(default=100, ge=1, le=MAX_STRUCTURAL_RESULTS)


class _PythonRepositoryQueryArguments(_RepositoryToolArguments):
    name: str = Field(min_length=1, max_length=MAX_STRUCTURAL_QUERY_CHARACTERS)
    relative_path: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELATIVE_PATH_CHARACTERS,
    )
    max_files: int = Field(default=200, ge=1, le=MAX_PYTHON_SCAN_FILES)
    max_results: int = Field(default=100, ge=1, le=MAX_STRUCTURAL_RESULTS)

    @field_validator("name")
    @classmethod
    def require_dotted_python_name(cls, value: str) -> str:
        parts = value.split(".")
        if any(not part.isidentifier() or keyword.iskeyword(part) for part in parts):
            raise ValueError("name must be a dotted Python identifier")
        return value

    @field_validator("relative_path")
    @classmethod
    def require_optional_python_path(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_python_tool_path(value)
        return value


class FindSymbolDefinitionArguments(_PythonRepositoryQueryArguments):
    """Bounded arguments accepted by ``find_symbol_definition``."""


class FindReferencesArguments(_PythonRepositoryQueryArguments):
    """Bounded arguments accepted by ``find_references``."""


REPOSITORY_TOOL_DESCRIPTORS = MappingProxyType(
    {
        descriptor.name: descriptor
        for descriptor in (
            ToolDescriptor(
                name="list_tree",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=5_000,
                max_output_bytes=65_536,
                cacheable=True,
            ),
            ToolDescriptor(
                name="search_text",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=6_000,
                max_output_bytes=65_536,
                cacheable=True,
            ),
            ToolDescriptor(
                name="read_code",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=5_000,
                max_output_bytes=131_072,
                cacheable=True,
            ),
            ToolDescriptor(
                name="list_symbols",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=6_000,
                max_output_bytes=262_144,
                cacheable=True,
            ),
            ToolDescriptor(
                name="inspect_imports",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=6_000,
                max_output_bytes=262_144,
                cacheable=True,
            ),
            ToolDescriptor(
                name="find_symbol_definition",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=10_000,
                max_output_bytes=524_288,
                cacheable=True,
            ),
            ToolDescriptor(
                name="find_references",
                capabilities=(ToolCapability.FILESYSTEM_READ,),
                side_effect=ToolSideEffect.READ_ONLY,
                risk_level=ToolRiskLevel.LOW,
                timeout_ms=10_000,
                max_output_bytes=524_288,
                cacheable=True,
            ),
        )
    }
)


def execute_list_tree(repository_root: Path, arguments: ListTreeArguments) -> str:
    """Execute and normalize one bounded tree listing."""

    entries = list_tree(
        repository_root,
        max_depth=arguments.max_depth,
        max_entries=arguments.max_entries,
    )
    return _normalized_json({"entries": entries})


def execute_search_text(
    repository_root: Path,
    arguments: SearchTextArguments,
) -> str:
    """Execute and normalize one bounded literal search."""

    matches = search_text(
        repository_root,
        query=arguments.query,
        globs=list(arguments.globs) if arguments.globs is not None else None,
        max_results=arguments.max_results,
    )
    return _normalized_json({"matches": matches})


def execute_read_code(repository_root: Path, arguments: ReadCodeArguments) -> str:
    """Execute and normalize one bounded code read."""

    result = read_code(
        repository_root,
        relative_path=Path(arguments.relative_path),
        start_line=arguments.start_line,
        end_line=arguments.end_line,
    )
    return _normalized_json(result)


def execute_list_symbols(
    repository_root: Path,
    arguments: ListSymbolsArguments,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> str:
    """Execute one bounded content-addressed symbol listing."""

    return _normalized_json(
        list_python_symbols(
            repository_root,
            arguments.relative_path,
            max_results=arguments.max_results,
            repository_id=repository_id,
            commit_sha=commit_sha,
        )
    )


def execute_inspect_imports(
    repository_root: Path,
    arguments: InspectImportsArguments,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> str:
    """Execute one bounded content-addressed import inspection."""

    return _normalized_json(
        inspect_python_imports(
            repository_root,
            arguments.relative_path,
            max_results=arguments.max_results,
            repository_id=repository_id,
            commit_sha=commit_sha,
        )
    )


def execute_find_symbol_definition(
    repository_root: Path,
    arguments: FindSymbolDefinitionArguments,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> str:
    """Execute one bounded repository-wide definition query."""

    return _normalized_json(
        find_python_symbol_definitions(
            repository_root,
            arguments.name,
            relative_path=arguments.relative_path,
            max_files=arguments.max_files,
            max_results=arguments.max_results,
            repository_id=repository_id,
            commit_sha=commit_sha,
        )
    )


def execute_find_references(
    repository_root: Path,
    arguments: FindReferencesArguments,
    *,
    repository_id: str,
    commit_sha: str | None,
) -> str:
    """Execute one bounded repository-wide syntactic reference query."""

    return _normalized_json(
        find_python_references(
            repository_root,
            arguments.name,
            relative_path=arguments.relative_path,
            max_files=arguments.max_files,
            max_results=arguments.max_results,
            repository_id=repository_id,
            commit_sha=commit_sha,
        )
    )


def _normalized_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_python_tool_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        "\x00" in value
        or "\\" in value
        or not value.endswith(".py")
        or path.as_posix() != value
    ):
        raise ValueError("relative_path must identify a canonical Python file")
