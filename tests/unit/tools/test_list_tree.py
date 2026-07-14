from pathlib import Path

import pytest

from repocompass.tools.repository import list_tree


def test_list_tree_lists_direct_children(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").touch()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "nested.py").touch()

    result = list_tree(tmp_path, max_depth=1)

    assert result == ["alpha.txt", "src/"]


# Empty repository and depth behavior


def test_list_tree_returns_empty_for_empty_repository(tmp_path: Path) -> None:
    assert list_tree(tmp_path) == []


def test_list_tree_returns_empty_when_max_depth_is_zero(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").touch()
    (tmp_path / "src").mkdir()

    assert list_tree(tmp_path, max_depth=0) == []


def test_list_tree_respects_recursive_depth(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    repocompass_dir = src_dir / "repocompass"
    repocompass_dir.mkdir()
    
    tools_dir = repocompass_dir / "tools"
    tools_dir.mkdir()
    
    (tools_dir / "alpha.txt").touch()
    
    result = list_tree(tmp_path, max_depth=3)
    
    assert result == ["src/", "src/repocompass/", "src/repocompass/tools/"]


def test_list_tree_includes_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    assert list_tree(tmp_path) == ["empty/"]


# Input validation


def test_list_tree_rejects_missing_repository(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        list_tree(missing_dir)

    assert str(missing_dir) in str(exc_info.value)


def test_list_tree_rejects_file_as_repository(tmp_path: Path) -> None:
    file_path = tmp_path / "repository.txt"
    file_path.touch()

    with pytest.raises(NotADirectoryError) as exc_info:
        list_tree(file_path)

    assert str(file_path) in str(exc_info.value)


def test_list_tree_rejects_negative_max_depth(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc_info:
        list_tree(tmp_path, max_depth=-1)

    assert "-1" in str(exc_info.value)


@pytest.mark.parametrize("max_entries", [0, -1])
def test_list_tree_rejects_non_positive_max_entries(
    tmp_path: Path,
    max_entries: int,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        list_tree(tmp_path, max_entries=max_entries)

    assert str(max_entries) in str(exc_info.value)


# Ignored directories


@pytest.mark.parametrize(
    "ignored_name",
    [
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    ],
)
def test_list_tree_ignores_default_directories(
    tmp_path: Path,
    ignored_name: str,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    
    ignored_dir = tmp_path / ignored_name
    ignored_dir.mkdir()
    (ignored_dir / "hidden.py").touch()
    
    result = list_tree(tmp_path)
    
    assert result == ["src/"]


def test_list_tree_ignores_nested_directories(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "visible.py").touch()

    build_dir = src_dir / "build"
    build_dir.mkdir()
    (build_dir / "hidden.py").touch()

    assert list_tree(tmp_path) == ["src/", "src/visible.py"]


def test_list_tree_keeps_files_named_like_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "build").touch()

    assert list_tree(tmp_path) == ["build"]


# Stable ordering and entry limits


def test_list_tree_returns_stable_depth_first_order(tmp_path: Path) -> None:
    (tmp_path / "z.txt").touch()
    (tmp_path / "b.txt").touch()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "z.py").touch()
    (src_dir / "a.py").touch()

    assert list_tree(tmp_path) == [
        "b.txt",
        "src/",
        "src/a.py",
        "src/z.py",
        "z.txt",
    ]


def test_list_tree_stops_at_max_entries(tmp_path: Path) -> None:
    max_entries = 10
    
    entries = []
    for i in range(max_entries):
        file = f"entries_{i}"
        entries.append(file)
        (tmp_path / file).touch()
    
    (tmp_path / "src").mkdir()
    (tmp_path / "tmp").mkdir()
    
    result = list_tree(tmp_path, max_depth=3, max_entries=max_entries)
    
    assert result == entries
        


# Symbolic links


def test_list_tree_lists_symlink_without_following_it(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "hidden.py").touch()

    (repo_path / "external-link").symlink_to(
        external_dir,
        target_is_directory=True,
    )

    assert list_tree(repo_path) == ["external-link"]
