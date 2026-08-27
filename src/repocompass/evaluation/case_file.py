"""Strict file boundary for portable context-ablation cases."""

import json
from pathlib import Path

from pydantic import ValidationError

from repocompass.evaluation.context_ablation import ContextAblationCase

MAX_CONTEXT_ABLATION_CASE_FILE_BYTES = 1_048_576


def load_context_ablation_case_file(path: Path) -> ContextAblationCase:
    """Load one bounded, duplicate-key-free, content-addressed case file."""

    if not isinstance(path, Path):
        raise TypeError("context ablation case file path must be a Path")
    file_size = path.stat().st_size
    if file_size > MAX_CONTEXT_ABLATION_CASE_FILE_BYTES:
        raise ValueError(
            "Context ablation case file exceeds "
            f"{MAX_CONTEXT_ABLATION_CASE_FILE_BYTES} bytes: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_CONTEXT_ABLATION_CASE_FILE_BYTES:
        raise ValueError(
            "Context ablation case file exceeds "
            f"{MAX_CONTEXT_ABLATION_CASE_FILE_BYTES} bytes: {path}"
        )
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_unique_json_object)
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return ContextAblationCase.model_validate_json(canonical_json)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as error:
        raise ValueError(
            f"Invalid context ablation case file {path}: {error}"
        ) from error
    except ValueError as error:
        if str(error).startswith("duplicate JSON key"):
            raise
        raise ValueError(
            f"Invalid context ablation case file {path}: {error}"
        ) from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
