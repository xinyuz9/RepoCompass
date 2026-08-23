"""Typed arguments, descriptors, and normalized repository tool executors."""

import json
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def _normalized_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
