"""Source preview reuses the same no-follow byte boundary as repository tools."""

import hashlib
import os
from pathlib import Path

import pytest

from repocompass import source_reader
from repocompass.dashboard.service import (
    DashboardSnapshotConflictError,
    _read_source_snippet,
)
from repocompass.structure.contracts import (
    MAX_INDEXED_SOURCE_BYTES,
    create_source_location,
)
from repocompass.tools.gateway import ToolExecutionContext


def _preview(root, content=b"answer = 42\n"):
    context = ToolExecutionContext(root)
    location = create_source_location(
        repository_id=context.repository_id,
        commit_sha=context.commit_sha,
        path="pkg/source.py",
        source_digest=hashlib.sha256(content).hexdigest(),
        line_start=1,
        column_start=0,
        line_end=1,
        column_end=11,
    )
    return _read_source_snippet(
        thread_id="thread-1",
        entity_id="a" * 64,
        symbol="source",
        context=context,
        location=location,
    )


@pytest.fixture
def root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/source.py").write_bytes(b"answer = 42\n")
    return repo


def test_source_preview_does_not_use_path_based_reads(root, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source preview bypassed the bounded reader")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert _preview(root).lines[0].text == "answer = 42"


def test_source_preview_rejects_parent_swapped_at_open(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    # Identical content makes hash checks alone insufficient for path policy.
    (outside / "source.py").write_bytes(b"answer = 42\n")
    original = source_reader.os.open
    swapped = False

    def swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "pkg" and not swapped:
            swapped = True
            (root / "pkg").rename(root / "original-pkg")
            (root / "pkg").symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(source_reader.os, "open", swap)
    with pytest.raises(DashboardSnapshotConflictError, match="unsafe"):
        _preview(root)
    assert swapped


@pytest.mark.parametrize("kind", ["oversized", "fifo", "symlink"])
def test_source_preview_rejects_unsafe_files(root, tmp_path, kind):
    path = root / "pkg/source.py"
    path.unlink()
    if kind == "oversized":
        path.write_bytes(b"x" * (MAX_INDEXED_SOURCE_BYTES + 1))
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        outside = tmp_path / "outside.py"
        outside.write_bytes(b"answer = 42\n")
        path.symlink_to(outside)
    with pytest.raises(DashboardSnapshotConflictError, match="unsafe"):
        _preview(root)
