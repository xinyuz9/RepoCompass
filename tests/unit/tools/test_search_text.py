import subprocess
from pathlib import Path

import pytest

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
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=["rg"], returncode=2, stdout="", stderr="synthetic rg failure"
        )

    monkeypatch.setattr(
        "repocompass.tools.repository.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError) as exc_info:
        search_text(tmp_path, "cool")

    assert "synthetic rg failure" in str(exc_info.value)


def test_search_text_reports_missing_rg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("rg")

    monkeypatch.setattr(
        "repocompass.tools.repository.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="was not found on PATH"):
        search_text(tmp_path, "needle")


def test_search_text_reports_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["rg"], timeout=5)

    monkeypatch.setattr(
        "repocompass.tools.repository.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="timed out after 5 seconds"):
        search_text(tmp_path, "needle")
