import json
import subprocess
import sys
from pathlib import Path

import pytest

from repocompass.tools import repository
from repocompass.tools.repository import search_text


# Happy path and output parsing


def test_search_text_returns_relative_match(tmp_path: Path) -> None:
    source_lines = [
        "first line",
        "second line",
        "third line",
        "fourth line",
        "fifth line",
    ]

    file_path = tmp_path / "lines.txt"

    file_path.write_text(
        "\n".join(source_lines),
        encoding="utf-8",
    )

    result = search_text(tmp_path, "second")

    assert result == [
        {
            "path": file_path.relative_to(tmp_path).as_posix(),
            "line": 2,
            "text": "second line",
        }
    ]


def test_search_text_preserves_colons_in_matched_text(tmp_path: Path) -> None:
    source_file = tmp_path / "config.txt"
    source_file.write_text("setting: host: localhost\n", encoding="utf-8")

    result = search_text(tmp_path, "host")

    assert result == [
        {
            "path": "config.txt",
            "line": 1,
            "text": "setting: host: localhost",
        }
    ]


def test_search_text_returns_stable_path_order(tmp_path: Path) -> None:
    (tmp_path / "zebra.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("needle\n", encoding="utf-8")

    result = search_text(tmp_path, "needle")

    assert [match["path"] for match in result] == ["alpha.txt", "zebra.txt"]


@pytest.mark.parametrize(
    "filename", ["a:b.py", "line\nbreak.py", "tab\tname.py", "源码.py"]
)
def test_search_preserves_special_filenames(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).write_text("needle: exact\n", encoding="utf-8")
    assert search_text(tmp_path, "needle") == [
        {"path": filename, "line": 1, "text": "needle: exact"}
    ]


def test_search_preserves_non_newline_control_characters_in_content(tmp_path):
    (tmp_path / "source.txt").write_text("needle\vstill same line\r\n")
    assert search_text(tmp_path, "needle") == [
        {"path": "source.txt", "line": 1, "text": "needle\vstill same line"}
    ]


def test_search_rejects_oversized_record_before_returning_content(tmp_path):
    (tmp_path / "source.txt").write_text("needle" + "x" * 262_144 + "\n")
    with pytest.raises(RuntimeError, match="record.*byte limit"):
        search_text(tmp_path, "needle", max_results=1)


# Query behavior and validation


def test_search_ignores_user_config_that_enables_symlink_traversal(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "safe.txt").write_text("safe needle\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("private outside needle\n")
    (root / "linked.txt").symlink_to(outside)
    config = tmp_path / "ripgreprc"
    config.write_text("--follow\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    assert search_text(root, "needle") == [
        {"path": "safe.txt", "line": 1, "text": "safe needle"}
    ]


def test_search_never_executes_configured_preprocessor(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "safe.txt").write_text("needle\n")
    marker = tmp_path / "preprocessor-ran"
    preprocessor = tmp_path / "preprocessor"
    preprocessor.write_text("#!/bin/sh\n" + f"touch '{marker}'\n" + 'cat "$1"\n')
    preprocessor.chmod(0o700)
    config = tmp_path / "ripgreprc"
    config.write_text(f"--pre={preprocessor}\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    assert search_text(root, "needle") == [
        {"path": "safe.txt", "line": 1, "text": "needle"}
    ]
    assert not marker.exists()


def test_search_text_returns_empty_when_no_matches(tmp_path: Path) -> None:
    source_lines = [
        "first line",
        "second line",
        "third line",
        "fourth line",
        "fifth line",
    ]

    file_path = tmp_path / "lines.txt"

    file_path.write_text(
        "\n".join(source_lines),
        encoding="utf-8",
    )

    result = search_text(tmp_path, "cool")

    assert result == []


def test_search_text_treats_query_as_fixed_string(tmp_path: Path) -> None:
    source_file = tmp_path / "patterns.txt"
    source_file.write_text("a.b\naxb\n", encoding="utf-8")

    result = search_text(tmp_path, "a.b")

    assert result == [{"path": "patterns.txt", "line": 1, "text": "a.b"}]


def test_search_text_handles_query_starting_with_hyphen(tmp_path: Path) -> None:
    source_file = tmp_path / "options.txt"
    source_file.write_text("--literal-option\n", encoding="utf-8")

    result = search_text(tmp_path, "--literal-option")

    assert result == [{"path": "options.txt", "line": 1, "text": "--literal-option"}]


def test_search_text_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        search_text(tmp_path, "")


# Glob filtering


def test_search_text_supports_single_glob(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = search_text(tmp_path, "needle", globs=["*.py"])

    assert result == [{"path": "module.py", "line": 1, "text": "needle"}]


def test_search_text_supports_multiple_globs(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("needle\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = search_text(tmp_path, "needle", globs=["*.py", "*.md"])

    assert [match["path"] for match in result] == ["module.py", "readme.md"]


# Result limits


@pytest.mark.parametrize("max_results", [0, -1])
def test_search_text_rejects_non_positive_max_results(
    tmp_path: Path,
    max_results: int,
) -> None:
    with pytest.raises(ValueError, match="Max_results must be positive"):
        search_text(tmp_path, "needle", max_results=max_results)


def test_search_text_stops_at_max_results(tmp_path: Path) -> None:
    source_file = tmp_path / "matches.txt"
    source_file.write_text("needle one\nneedle two\nneedle three\n", encoding="utf-8")

    result = search_text(tmp_path, "needle", max_results=2)

    assert result == [
        {"path": "matches.txt", "line": 1, "text": "needle one"},
        {"path": "matches.txt", "line": 2, "text": "needle two"},
    ]


# Repository and subprocess failures


def _substitute_search_process(monkeypatch, program: str):
    """Run a controlled child with real pipes, signals, and descriptor cleanup."""
    popen = subprocess.Popen
    processes = []

    def spawn(command, **kwargs):
        assert "--json" in command
        assert "--no-config" in command
        assert "--no-follow" in command
        process = popen([sys.executable, "-c", program], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(repository.subprocess, "Popen", spawn)
    return processes


def _assert_processes_closed(processes):
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed
    assert processes[0].stderr.closed


def _match_json(**overrides):
    data = {
        "path": {"text": "./source.py"},
        "lines": {"text": "needle\n"},
        "line_number": 1,
    }
    data.update(overrides)
    return json.dumps({"type": "match", "data": data}).encode() + b"\n"


def test_search_text_rejects_missing_repository(tmp_path: Path) -> None:
    missing_repository = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Repository path does not exist"):
        search_text(missing_repository, "needle")


def test_search_text_rejects_file_as_repository(tmp_path: Path) -> None:
    repository_file = tmp_path / "not-a-directory.txt"
    repository_file.write_text("needle\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        search_text(repository_file, "needle")


def test_search_text_raises_on_rg_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = _substitute_search_process(
        monkeypatch,
        "import sys; sys.stderr.write('synthetic rg failure'); sys.exit(2)",
    )

    with pytest.raises(RuntimeError) as exc_info:
        search_text(tmp_path, "cool")

    assert "synthetic rg failure" in str(exc_info.value)
    _assert_processes_closed(processes)


def test_search_text_reports_missing_rg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("rg")

    monkeypatch.setattr(
        "repocompass.tools.repository.subprocess.Popen",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="was not found on PATH"):
        search_text(tmp_path, "needle")


def test_search_text_reports_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository, "SEARCH_TIMEOUT_SECONDS", 0.1)
    processes = _substitute_search_process(monkeypatch, "import time; time.sleep(30)")

    with pytest.raises(RuntimeError, match="timed out after 0.1 seconds"):
        search_text(tmp_path, "needle")
    _assert_processes_closed(processes)


def test_search_kills_child_when_result_limit_is_reached(tmp_path, monkeypatch):
    processes = _substitute_search_process(
        monkeypatch,
        f"import os,time; os.write(1, {_match_json()!r}); time.sleep(30)",
    )
    assert search_text(tmp_path, "needle", max_results=1) == [
        {"path": "source.py", "line": 1, "text": "needle"}
    ]
    _assert_processes_closed(processes)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_search_bounds_pipe_bytes_even_without_matches(tmp_path, monkeypatch, stream):
    limit = 1024
    monkeypatch.setattr(repository, f"MAX_SEARCH_{stream.upper()}_BYTES", limit)
    payload = b'{"type":"begin"}\n' * 128 if stream == "stdout" else b"e" * 2048
    fd = 1 if stream == "stdout" else 2
    processes = _substitute_search_process(
        monkeypatch, f"import os,time; os.write({fd}, {payload!r}); time.sleep(30)"
    )
    with pytest.raises(RuntimeError, match=f"{stream} exceeded byte limit"):
        search_text(tmp_path, "needle")
    _assert_processes_closed(processes)


@pytest.mark.parametrize("newline", [b"", b"\n"])
def test_search_bounds_record_before_parsing(tmp_path, monkeypatch, newline):
    monkeypatch.setattr(repository, "MAX_SEARCH_RECORD_BYTES", 128)
    payload = b"x" * 129 + newline
    processes = _substitute_search_process(
        monkeypatch, f"import os,time; os.write(1, {payload!r}); time.sleep(30)"
    )
    with pytest.raises(RuntimeError, match="record exceeded byte limit"):
        search_text(tmp_path, "needle")
    _assert_processes_closed(processes)


@pytest.mark.parametrize(
    "payload",
    [
        b"not json\n",
        b"[]\n",
        b'{"type":"unknown"}\n',
        b"\xff\n",
        _match_json(path={"bytes": "//4="}),
        _match_json(lines={"bytes": "//4="}),
        _match_json(path={"text": "../outside.py"}),
        _match_json(path={"text": "/outside.py"}),
        _match_json(line_number=True),
        _match_json(line_number=0),
    ],
)
def test_search_rejects_malformed_or_unrepresentable_matches(
    tmp_path, monkeypatch, payload
):
    processes = _substitute_search_process(
        monkeypatch, f"import os; os.write(1, {payload!r})"
    )
    with pytest.raises(RuntimeError, match="invalid or non-UTF-8"):
        search_text(tmp_path, "needle")
    _assert_processes_closed(processes)


def test_search_rejects_incomplete_record_at_eof(tmp_path, monkeypatch):
    payload = _match_json()[:-1]
    processes = _substitute_search_process(
        monkeypatch, f"import os; os.write(1, {payload!r})"
    )
    with pytest.raises(RuntimeError, match="incomplete JSON record"):
        search_text(tmp_path, "needle")
    _assert_processes_closed(processes)


def test_search_reassembles_records_across_small_reads(tmp_path, monkeypatch):
    original = repository.os.read
    monkeypatch.setattr(
        repository.os, "read", lambda fd, size: original(fd, min(size, 3))
    )
    (tmp_path / "源码.py").write_text("needle 汉字\r\n", encoding="utf-8")
    assert search_text(tmp_path, "needle") == [
        {"path": "源码.py", "line": 1, "text": "needle 汉字"}
    ]
