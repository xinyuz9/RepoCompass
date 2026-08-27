"""Strict contract tests for M4 Python structural evidence."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.structure import (
    PythonFileIndex,
    StructuralVerificationStatus,
    build_python_file_index,
)


REPOSITORY_ID = hashlib.sha256(b"repository").hexdigest()
COMMIT_SHA = "a" * 40


def _index(tmp_path: Path) -> PythonFileIndex:
    relative_path = Path("pkg/module.py")
    target = tmp_path / relative_path
    target.parent.mkdir()
    target.write_text(
        "class Service:\n    def run(self):\n        return True\n", encoding="utf-8"
    )
    return build_python_file_index(
        tmp_path,
        relative_path,
        repository_id=REPOSITORY_ID,
        commit_sha=COMMIT_SHA,
    )


def test_python_index_is_strict_frozen_and_round_trips_without_identity_loss(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)

    restored = PythonFileIndex.model_validate_json(index.model_dump_json())

    assert restored == index
    assert restored.index_id == index.index_id
    with pytest.raises(ValidationError, match="frozen"):
        restored.path = "other.py"  # type: ignore[misc]


def test_python_index_rejects_unknown_fields_coercion_and_tampered_ids(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    payload = index.model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra"):
        PythonFileIndex.model_validate_json(
            json.dumps({**payload, "api_key": "secret"})
        )

    coerced = {**payload, "source_byte_count": str(payload["source_byte_count"])}
    with pytest.raises(ValidationError):
        PythonFileIndex.model_validate_json(json.dumps(coerced))

    tampered = {**payload, "source_digest": "f" * 64}
    with pytest.raises(ValidationError, match="index_id|source"):
        PythonFileIndex.model_validate_json(json.dumps(tampered))


def test_ast_records_cannot_be_relabelled_as_inferred_or_reordered(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    payload = index.model_dump(mode="json")
    symbols = list(payload["symbols"])
    symbols[0] = {
        **symbols[0],
        "verification_status": StructuralVerificationStatus.INFERRED.value,
    }

    with pytest.raises(ValidationError, match="verified"):
        PythonFileIndex.model_validate_json(json.dumps({**payload, "symbols": symbols}))

    with pytest.raises(ValidationError, match="canonical"):
        PythonFileIndex.model_validate_json(
            json.dumps({**payload, "symbols": list(reversed(payload["symbols"]))})
        )


def test_nested_location_tampering_is_rejected(tmp_path: Path) -> None:
    index = _index(tmp_path)
    payload = index.model_dump(mode="json")
    symbols = list(payload["symbols"])
    symbol = dict(symbols[-1])
    location = dict(symbol["definition"])
    location["line_end"] = 999
    symbol["definition"] = location
    symbols[-1] = symbol

    with pytest.raises(ValidationError, match="location|line|source"):
        PythonFileIndex.model_validate_json(json.dumps({**payload, "symbols": symbols}))
