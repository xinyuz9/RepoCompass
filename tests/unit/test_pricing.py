"""Tests for strict user-supplied pricing table files."""

import json
from pathlib import Path

import pytest

from repocompass.pricing import MAX_PRICING_FILE_BYTES, load_pricing_table_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _entry(provider: str, model: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "input_usd_per_million_tokens": "2",
        "output_usd_per_million_tokens": "8",
        "cached_input_usd_per_million_tokens": "same_as_input",
        "reasoning_output_usd_per_million_tokens": "same_as_output",
    }


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": "local-2026-08-27",
        "effective_from": "2026-08-27",
        "entries": [
            _entry("openai", "gpt-test"),
            _entry("deepseek", "deepseek-test"),
        ],
    }


def test_load_pricing_table_file_builds_canonical_content_addressed_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    pricing = load_pricing_table_file(path)

    assert pricing.version == "local-2026-08-27"
    assert pricing.effective_from == "2026-08-27"
    assert [(item.provider, item.model) for item in pricing.entries] == [
        ("deepseek", "deepseek-test"),
        ("openai", "gpt-test"),
    ]
    assert len(pricing.table_id) == 64
    assert pricing == load_pricing_table_file(path)


def test_bundled_pricing_example_is_loadable_but_clearly_non_provider_data() -> None:
    pricing = load_pricing_table_file(PROJECT_ROOT / "examples/pricing.example.json")

    assert pricing.version == "example-v1"
    assert all("example" in entry.model for entry in pricing.entries)
    assert {entry.provider for entry in pricing.entries} == {"deepseek", "openai"}


def test_bundled_deepseek_v4_snapshot_has_reviewed_exact_rates() -> None:
    pricing = load_pricing_table_file(
        PROJECT_ROOT / "examples/pricing.deepseek-2026-08-27.json"
    )

    pro = pricing.lookup("deepseek", "deepseek-v4-pro")
    flash = pricing.lookup("deepseek", "deepseek-v4-flash")
    assert pricing.version == "deepseek-v4-observed-2026-08-27"
    assert pro is not None
    assert pro.input_usd_per_million_tokens == "0.435"
    assert pro.cached_input_usd_per_million_tokens == "0.003625"
    assert pro.output_usd_per_million_tokens == "0.87"
    assert flash is not None
    assert flash.input_usd_per_million_tokens == "0.14"
    assert flash.cached_input_usd_per_million_tokens == "0.0028"
    assert flash.output_usd_per_million_tokens == "0.28"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update({"unexpected": True}),
        lambda document: document.update({"schema_version": 2}),
        lambda document: document.update({"entries": []}),
    ],
)
def test_load_pricing_table_file_rejects_invalid_document_schema(
    tmp_path: Path,
    mutation,
) -> None:
    document = _document()
    mutation(document)
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid pricing file"):
        load_pricing_table_file(path)


def test_load_pricing_table_file_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"version":"v1",'
        '"effective_from":"2026-08-27","entries":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_pricing_table_file(path)


def test_load_pricing_table_file_rejects_oversized_input_before_json_parse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pricing.json"
    path.write_bytes(b"x" * (MAX_PRICING_FILE_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds"):
        load_pricing_table_file(path)
