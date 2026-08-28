"""Focused-improve CLI artifact admission boundary tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from repocompass.improve.cli import (
    MAX_JSON_ARTIFACT_BYTES,
    _load_json,
    _read_replacement,
)
from repocompass.improve.contracts import MAX_IMPROVE_FILE_BYTES


def test_artifact_readers_reject_symlinks_special_files_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        _load_json(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        _load_json(duplicate)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "artifact.fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            _load_json(fifo)


def test_artifact_readers_enforce_byte_encoding_and_nul_limits(
    tmp_path: Path,
) -> None:
    oversized_json = tmp_path / "oversized.json"
    oversized_json.write_bytes(b" " * (MAX_JSON_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        _load_json(oversized_json)

    oversized_replacement = tmp_path / "oversized.py"
    oversized_replacement.write_bytes(b"x" * (MAX_IMPROVE_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        _read_replacement(oversized_replacement)

    invalid_utf8 = tmp_path / "invalid.py"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        _read_replacement(invalid_utf8)

    nul = tmp_path / "nul.py"
    nul.write_bytes(b"value = 1\x00\n")
    with pytest.raises(ValueError, match="bounded UTF-8"):
        _read_replacement(nul)
