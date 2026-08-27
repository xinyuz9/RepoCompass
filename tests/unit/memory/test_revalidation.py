"""Adversarial tests for bounded, no-follow memory source reads."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import repocompass.memory.revalidation as revalidation
from repocompass.memory.revalidation import (
    MAX_MEMORY_SOURCE_BYTES,
    MemorySourceReadError,
    MemorySourceReadReason,
    read_memory_source_digest,
)


def _assert_read_reason(
    repository: Path,
    relative_path: str,
    expected: MemorySourceReadReason,
) -> None:
    with pytest.raises(MemorySourceReadError) as raised:
        read_memory_source_digest(repository, relative_path)
    assert raised.value.reason is expected


def test_digest_hashes_raw_bytes_without_text_decoding(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    content = b"\xff\xfe\x00repository-memory\x80"
    source.write_bytes(content)

    assert (
        read_memory_source_digest(tmp_path, "source.bin")
        == hashlib.sha256(content).hexdigest()
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        "",
        ".",
        "../outside.py",
        "src/../outside.py",
        "/absolute.py",
        "src\\windows.py",
        "src//duplicate.py",
        "nul\x00byte.py",
    ),
)
def test_invalid_relative_paths_are_rejected_before_open(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _assert_read_reason(
        tmp_path,
        relative_path,
        MemorySourceReadReason.INVALID_PATH,
    )


def test_file_and_parent_symlinks_are_never_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "file-link.py").symlink_to(outside)
    (repository / "directory-link").symlink_to(tmp_path, target_is_directory=True)

    _assert_read_reason(
        repository,
        "file-link.py",
        MemorySourceReadReason.SYMLINK,
    )
    _assert_read_reason(
        repository,
        "directory-link/outside.py",
        MemorySourceReadReason.SYMLINK,
    )


def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    _assert_read_reason(
        tmp_path,
        "source.fifo",
        MemorySourceReadReason.SPECIAL_FILE,
    )


def test_directory_source_is_rejected_as_special_file(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()

    _assert_read_reason(
        tmp_path,
        "source",
        MemorySourceReadReason.SPECIAL_FILE,
    )


def test_oversized_source_is_rejected_before_reading_content(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    with source.open("wb") as stream:
        stream.seek(MAX_MEMORY_SOURCE_BYTES)
        stream.write(b"x")

    _assert_read_reason(
        tmp_path,
        "large.bin",
        MemorySourceReadReason.OVERSIZED,
    )


def test_source_metadata_change_during_read_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    real_fstat = revalidation.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal calls
        result = real_fstat(descriptor)
        calls += 1
        if calls != 2:
            return result
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns + 1,
        )

    monkeypatch.setattr(revalidation.os, "fstat", changing_fstat)

    _assert_read_reason(
        tmp_path,
        "source.py",
        MemorySourceReadReason.CHANGED_DURING_READ,
    )


def test_missing_source_and_invalid_repository_are_sanitized(tmp_path: Path) -> None:
    _assert_read_reason(
        tmp_path,
        "missing.py",
        MemorySourceReadReason.MISSING,
    )
    _assert_read_reason(
        tmp_path / "missing-repository",
        "source.py",
        MemorySourceReadReason.INVALID_REPOSITORY,
    )
