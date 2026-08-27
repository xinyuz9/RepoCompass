"""Bounded untrusted-file boundary for localization dataset manifests."""

import json
from pathlib import Path

from pydantic import ValidationError

from repocompass.evaluation.localization_contracts import (
    LocalizationDatasetManifest,
)

MAX_LOCALIZATION_DATASET_FILE_BYTES = 8_388_608


def load_localization_dataset_file(path: Path) -> LocalizationDatasetManifest:
    """Load one bounded, duplicate-key-free, content-addressed manifest."""

    if not isinstance(path, Path):
        raise TypeError("localization dataset file path must be a Path")
    file_size = path.stat().st_size
    if file_size > MAX_LOCALIZATION_DATASET_FILE_BYTES:
        raise ValueError(
            "Localization dataset file exceeds "
            f"{MAX_LOCALIZATION_DATASET_FILE_BYTES} bytes: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_LOCALIZATION_DATASET_FILE_BYTES:
        raise ValueError(
            "Localization dataset file exceeds "
            f"{MAX_LOCALIZATION_DATASET_FILE_BYTES} bytes: {path}"
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
        return LocalizationDatasetManifest.model_validate_json(canonical_json)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as error:
        raise ValueError(
            f"Invalid localization dataset file {path}: {error}"
        ) from error
    except ValueError as error:
        if str(error).startswith("duplicate JSON key"):
            raise
        raise ValueError(
            f"Invalid localization dataset file {path}: {error}"
        ) from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
