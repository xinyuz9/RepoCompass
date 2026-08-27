"""Memory repository identity resolution tests."""

import subprocess
from pathlib import Path

from repocompass.memory.contracts import MemoryRepositoryIdentityKind
from repocompass.memory.identity import resolve_memory_repository_identity
from repocompass.tools.gateway import ToolExecutionContext


def _initialize_repository(path: Path, remote: str) -> None:
    path.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", remote],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_equivalent_https_and_ssh_remotes_share_key_without_clone_binding(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _initialize_repository(
        first,
        "https://user:secret@GitHub.COM/OpenAI/RepoCompass.git",
    )
    _initialize_repository(second, "git@github.com:OpenAI/RepoCompass.git")

    first_identity = resolve_memory_repository_identity(ToolExecutionContext(first))
    second_identity = resolve_memory_repository_identity(ToolExecutionContext(second))

    assert first_identity.kind is MemoryRepositoryIdentityKind.GIT_REMOTE
    assert first_identity.repository_key == second_identity.repository_key
    assert first_identity.source_repository_id != second_identity.source_repository_id
    serialized = first_identity.model_dump_json()
    assert "secret" not in serialized
    assert "user" not in serialized
    assert "github" not in serialized


def test_repository_subdirectory_is_a_distinct_memory_scope(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository, "https://github.com/example/mono.git")
    package = repository / "packages" / "api"
    package.mkdir(parents=True)

    root_identity = resolve_memory_repository_identity(ToolExecutionContext(repository))
    package_identity = resolve_memory_repository_identity(ToolExecutionContext(package))

    assert root_identity.scope == "."
    assert package_identity.scope == "packages/api"
    assert root_identity.repository_key != package_identity.repository_key


def test_non_git_directories_use_path_bound_local_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    first_identity = resolve_memory_repository_identity(ToolExecutionContext(first))
    repeated = resolve_memory_repository_identity(ToolExecutionContext(first))
    second_identity = resolve_memory_repository_identity(ToolExecutionContext(second))

    assert first_identity.kind is MemoryRepositoryIdentityKind.LOCAL_PATH
    assert first_identity == repeated
    assert first_identity.repository_key != second_identity.repository_key
    assert first_identity.scope == "."
