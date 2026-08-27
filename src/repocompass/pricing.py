"""Strict file boundary for user-supplied pricing table snapshots."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from repocompass.agent.cost import ModelPrice, PricingTable, create_pricing_table

MAX_PRICING_FILE_BYTES = 1_048_576
_MAX_PRICE_ENTRIES = 10_000


class _PricingFileDocument(BaseModel):
    """Authoring format whose content-addressed table ID is derived on load."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1]
    version: str
    effective_from: str
    entries: list[ModelPrice] = Field(
        min_length=1,
        max_length=_MAX_PRICE_ENTRIES,
    )


def load_pricing_table_file(path: Path) -> PricingTable:
    """Load a bounded strict JSON document into one immutable price snapshot."""

    if not isinstance(path, Path):
        raise TypeError("pricing file path must be a Path")
    file_size = path.stat().st_size
    if file_size > MAX_PRICING_FILE_BYTES:
        raise ValueError(f"Pricing file exceeds {MAX_PRICING_FILE_BYTES} bytes: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_PRICING_FILE_BYTES:
        raise ValueError(f"Pricing file exceeds {MAX_PRICING_FILE_BYTES} bytes: {path}")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_unique_json_object)
        document = _PricingFileDocument.model_validate(payload)
        return create_pricing_table(
            version=document.version,
            effective_from=document.effective_from,
            entries=document.entries,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as error:
        raise ValueError(f"Invalid pricing file {path}: {error}") from error
    except ValueError as error:
        if str(error).startswith("duplicate JSON key"):
            raise
        raise ValueError(f"Invalid pricing file {path}: {error}") from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
