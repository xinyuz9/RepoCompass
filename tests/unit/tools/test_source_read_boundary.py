"""Execution-time source boundaries, independent of gateway preflight."""

import io
import os
from pathlib import Path

import pytest

from repocompass.tools.repository import read_code
from repocompass.structure import build_python_file_index
from repocompass.source_reader import (
    SourceReadError,
    SourceReadReason,
    read_source_bytes,
)
import repocompass.tools.gateway as gateway_module
from repocompass.tools.contracts import ToolDecisionOutcome, ToolDecisionReason
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    create_repository_tool_registry,
)


@pytest.mark.parametrize("reader_kind", ["code", "ast"])
def test_parent_swap_at_open_never_returns_outside_source(
    tmp_path, monkeypatch, reader_kind
):
    root = tmp_path / "repo"
    parent = root / "pkg"
    parent.mkdir(parents=True)
    (parent / "code.py").write_text("inside = True\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "code.py").write_text("private_outside_marker = True\n")
    switched = False
    original_io_open, original_os_open = io.open, os.open

    def switch():
        nonlocal switched
        if not switched:
            parent.rename(root / "original-pkg")
            parent.symlink_to(outside, target_is_directory=True)
            switched = True

    def race_io_open(path, *args, **kwargs):
        if path == parent / "code.py":
            switch()
        return original_io_open(path, *args, **kwargs)

    def race_os_open(path, *args, **kwargs):
        if path == parent / "code.py" or (
            path == "pkg" and kwargs.get("dir_fd") is not None
        ):
            switch()
        return original_os_open(path, *args, **kwargs)

    monkeypatch.setattr(io, "open", race_io_open)
    monkeypatch.setattr(os, "open", race_os_open)
    with pytest.raises(ValueError):
        _read(reader_kind, root, Path("pkg/code.py"))
    assert switched


def test_oversized_source_is_rejected_before_full_decode(tmp_path):
    target = tmp_path / "large.py"
    with target.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="oversized"):
        read_code(tmp_path, Path("large.py"), 1, 1)


@pytest.mark.parametrize("parent_link", [False, True])
@pytest.mark.parametrize("reader_kind", ["code", "ast"])
def test_direct_code_reader_rejects_internal_symlink_components(
    tmp_path, parent_link, reader_kind
):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "code.py").write_text("inside = True\n")
    if parent_link:
        (tmp_path / "link").symlink_to(package, target_is_directory=True)
        path = Path("link/code.py")
    else:
        (tmp_path / "link.py").symlink_to(package / "code.py")
        path = Path("link.py")
    with pytest.raises(ValueError):
        _read(reader_kind, tmp_path, path)


def _read(kind, root, path):
    if kind == "code":
        return read_code(root, path, 1, 1)
    return build_python_file_index(
        root, path, repository_id="a" * 64, commit_sha="b" * 40
    )


def test_invalid_line_range_rejected_before_opening_source(tmp_path, monkeypatch):
    (tmp_path / "code.py").write_text("value = 1\n")

    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid line request must not open source")

    monkeypatch.setattr(io, "open", unexpected_open)
    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(ValueError):
        read_code(tmp_path, Path("code.py"), 1, 401)


@pytest.mark.parametrize("kind", ["code", "ast"])
def test_fifo_is_rejected_without_reading_or_blocking(tmp_path, monkeypatch, kind):
    os.mkfifo(tmp_path / "pipe.py")

    def unexpected_read(*args, **kwargs):
        pytest.fail("special file must be rejected before reading")

    monkeypatch.setattr(os, "read", unexpected_read)
    with pytest.raises(ValueError, match="special_file"):
        _read(kind, tmp_path, Path("pipe.py"))


@pytest.mark.parametrize("success", [False, True])
def test_source_reader_closes_all_opened_descriptors(tmp_path, monkeypatch, success):
    parent = tmp_path / "pkg"
    parent.mkdir()
    (parent / "source.py").write_bytes(b"abcd")
    opened = []
    real_open = os.open

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", track_open)
    if success:
        assert read_source_bytes(tmp_path, "pkg/source.py", max_bytes=4) == b"abcd"
    else:
        with pytest.raises(SourceReadError) as error:
            read_source_bytes(tmp_path, "pkg/source.py", max_bytes=3)
        assert error.value.reason is SourceReadReason.OVERSIZED
    assert len(opened) == 3
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_file_growth_during_read_cannot_exceed_byte_budget(tmp_path, monkeypatch):
    target = tmp_path / "source.py"
    target.write_bytes(b"abcd")
    real_read = os.read
    seen = []

    def growing_read(fd, size):
        result = real_read(fd, size)
        seen.append(size)
        if len(seen) == 1:
            with target.open("ab") as stream:
                stream.write(b"e" * 100)
        return result

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(SourceReadError) as error:
        read_source_bytes(tmp_path, "source.py", max_bytes=4)
    assert error.value.reason is SourceReadReason.OVERSIZED
    assert seen == [5, 1]


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"])
def test_source_reader_never_falls_back_without_required_os_flags(
    tmp_path, monkeypatch, flag
):
    monkeypatch.delattr(os, flag)
    with pytest.raises(SourceReadError) as error:
        read_source_bytes(tmp_path, "source.py")
    assert error.value.reason is SourceReadReason.INVALID_REPOSITORY


@pytest.mark.parametrize("tool", ["read_code", "list_symbols"])
def test_gateway_worker_denies_parent_replaced_after_successful_preflight(
    tmp_path, monkeypatch, tool
):
    root = tmp_path / "repo"
    parent = root / "pkg"
    parent.mkdir(parents=True)
    (parent / "code.py").write_text("safe = True\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "code.py").write_text("private_outside_marker = True\n")
    preflight = gateway_module._repository_path_policy
    passed = []

    def replace_after_preflight(*args):
        decision = preflight(*args)
        assert decision is None
        passed.append(True)
        parent.rename(root / "original-pkg")
        parent.symlink_to(outside, target_is_directory=True)
        return decision

    monkeypatch.setattr(
        gateway_module, "_repository_path_policy", replace_after_preflight
    )
    context = ToolExecutionContext(root)
    arguments = {"relative_path": "pkg/code.py"}
    if tool == "read_code":
        arguments.update(start_line=1, end_line=1)
    audit = ToolAuditLog("source-boundary")
    gateway = ToolGateway(create_repository_tool_registry(include_structural=True))
    result = gateway.execute(
        context.bind_call(call_id="read", tool_name=tool, arguments=arguments),
        context=context,
        available_budget=1,
        audit_log=audit,
    )
    assert passed == [True]
    assert result.decision.outcome is ToolDecisionOutcome.DENIED
    assert result.decision.reason is ToolDecisionReason.EXECUTION_ERROR
    assert result.budget_consumed == 1
    assert len(audit.records) == 1 and audit.records[0].result_digest is None
    assert "private_outside_marker" not in result.content
    assert str(outside) not in result.content
