from pathlib import Path

import pytest

from repocompass.tools.repository import read_code


# Happy path and line-range behavior


def test_read_code_returns_numbered_inclusive_range(tmp_path: Path) -> None:
    source_lines = [
        "first line",
        "second line",
        "third line",
        "fourth line",
        "fifth line",
    ]

    file_path = tmp_path / "lines.txt"
    relative_path = file_path.relative_to(tmp_path)

    file_path.write_text(
        "\n".join(source_lines),
        encoding="utf-8",
    )

    start_line = 2
    end_line = 4

    result = read_code(
        tmp_path, relative_path, start_line=start_line, end_line=end_line
    )

    expected_content = "2 | second line\n3 | third line\n4 | fourth line"

    assert result == {
        "path": relative_path.as_posix(),
        "start_line": start_line,
        "end_line": end_line,
        "content": expected_content,
    }


def test_read_code_truncates_end_line_at_eof(tmp_path: Path) -> None:
    source_lines = [
        "first line",
        "second line",
        "third line",
    ]

    file_path = tmp_path / "lines.txt"
    relative_path = file_path.relative_to(tmp_path)

    file_path.write_text(
        "\n".join(source_lines),
        encoding="utf-8",
    )

    start_line = 2
    end_line = 10

    result = read_code(
        tmp_path, relative_path, start_line=start_line, end_line=end_line
    )

    expected_content = "2 | second line\n3 | third line"

    assert result == {
        "path": relative_path.as_posix(),
        "start_line": start_line,
        "end_line": 3,
        "content": expected_content,
    }


def test_read_code_preserves_blank_lines(tmp_path: Path) -> None:
    relative_path = Path("blank_lines.py")
    (tmp_path / relative_path).write_text(
        "first line\n\nthird line",
        encoding="utf-8",
    )

    result = read_code(tmp_path, relative_path, start_line=1, end_line=3)

    assert result == {
        "path": "blank_lines.py",
        "start_line": 1,
        "end_line": 3,
        "content": "1 | first line\n2 | \n3 | third line",
    }


# Line-number validation


@pytest.mark.parametrize("start_line", [0, -1])
def test_read_code_rejects_start_line_below_one(
    tmp_path: Path,
    start_line: int,
) -> None:
    relative_path = Path("lines.txt")
    (tmp_path / relative_path).write_text("first line", encoding="utf-8")

    with pytest.raises(ValueError):
        read_code(tmp_path, relative_path, start_line=start_line, end_line=1)


def test_read_code_rejects_end_line_before_start_line(tmp_path: Path) -> None:
    relative_path = Path("lines.txt")
    (tmp_path / relative_path).write_text(
        "first line\nsecond line\nthird line",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_code(tmp_path, relative_path, start_line=3, end_line=2)


def test_read_code_rejects_ranges_over_400_lines(tmp_path: Path) -> None:
    relative_path = Path("large_file.txt")
    source = "\n".join(f"line {line_number}" for line_number in range(1, 402))
    (tmp_path / relative_path).write_text(source, encoding="utf-8")

    with pytest.raises(ValueError):
        read_code(tmp_path, relative_path, start_line=1, end_line=401)


def test_read_code_rejects_start_line_beyond_eof(tmp_path: Path) -> None:
    source_lines = [
        "first line",
        "second line",
        "third line",
    ]

    file_path = tmp_path / "lines.txt"
    relative_path = file_path.relative_to(tmp_path)

    file_path.write_text(
        "\n".join(source_lines),
        encoding="utf-8",
    )

    start_line = 4
    end_line = 5

    with pytest.raises(ValueError) as exc_info:
        read_code(tmp_path, relative_path, start_line=start_line, end_line=end_line)

    assert str(start_line) in str(exc_info.value)


# Repository and file validation


def test_read_code_rejects_missing_repository(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        read_code(missing_repo, Path("code.py"), start_line=1, end_line=1)

    assert str(missing_repo) in str(exc_info.value)


def test_read_code_rejects_file_as_repository(tmp_path: Path) -> None:
    repository_file = tmp_path / "repository.txt"
    repository_file.touch()

    with pytest.raises(NotADirectoryError) as exc_info:
        read_code(repository_file, Path("code.py"), start_line=1, end_line=1)

    assert str(repository_file) in str(exc_info.value)


def test_read_code_rejects_missing_file(tmp_path: Path) -> None:
    relative_path = Path("missing.py")

    with pytest.raises(FileNotFoundError) as exc_info:
        read_code(tmp_path, relative_path, start_line=1, end_line=1)

    assert relative_path.as_posix() in str(exc_info.value)


def test_read_code_rejects_directory_target(tmp_path: Path) -> None:
    relative_path = Path("src")
    (tmp_path / relative_path).mkdir()

    with pytest.raises(IsADirectoryError) as exc_info:
        read_code(tmp_path, relative_path, start_line=1, end_line=1)

    assert relative_path.as_posix() in str(exc_info.value)


def test_read_code_rejects_non_utf8_file(tmp_path: Path) -> None:
    relative_path = Path("binary.dat")
    (tmp_path / relative_path).write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        read_code(tmp_path, relative_path, start_line=1, end_line=1)


# Repository boundary and symlink safety


def test_read_code_rejects_absolute_relative_path(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    absolute_path = repo_path / "inside.py"
    absolute_path.write_text("inside", encoding="utf-8")

    with pytest.raises(ValueError):
        read_code(repo_path, absolute_path, start_line=1, end_line=1)


def test_read_code_rejects_parent_directory_escape(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (tmp_path / "outside.py").write_text("outside", encoding="utf-8")
    escaping_path = Path("..") / "outside.py"

    with pytest.raises(ValueError):
        read_code(repo_path, escaping_path, start_line=1, end_line=1)


def test_read_code_rejects_symlink_outside_repository(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    external_file = tmp_path / "external.py"
    external_file.write_text("outside", encoding="utf-8")
    relative_path = Path("external-link.py")
    (repo_path / relative_path).symlink_to(external_file)

    with pytest.raises(ValueError):
        read_code(repo_path, relative_path, start_line=1, end_line=1)


def test_read_code_allows_symlink_within_repository(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    target_file = repo_path / "target.py"
    target_file.write_text("first line\nsecond line", encoding="utf-8")
    relative_path = Path("target-link.py")
    (repo_path / relative_path).symlink_to(target_file)

    result = read_code(repo_path, relative_path, start_line=1, end_line=2)

    assert result == {
        "path": "target-link.py",
        "start_line": 1,
        "end_line": 2,
        "content": "1 | first line\n2 | second line",
    }
