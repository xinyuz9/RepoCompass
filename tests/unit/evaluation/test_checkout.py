"""Benchmark source admission without network or executing repository code."""

from pathlib import Path
from repocompass.evaluation.localization_case_file import load_localization_dataset_file
import pytest
from repocompass.evaluation.checkout import (
    github_source_url,
    prepare_snapshot,
    validate_snapshot,
)


def test_snapshot_probe_uses_a_valid_production_budget(tmp_path, monkeypatch):
    manifest = load_localization_dataset_file(
        Path("benchmarks/cases/swe-bench-lite-localization-10-v1.json")
    )
    calls = []
    monkeypatch.setattr(
        "repocompass.evaluation.checkout.create_localization_run_request",
        lambda **kwargs: calls.append(kwargs),
    )
    validate_snapshot(manifest.cases[0].task, tmp_path)
    assert calls[0]["budget_profile"].max_plan_revisions >= 1


@pytest.mark.parametrize(
    "value",
    [
        "../etc",
        "https://evil.test/x",
        "a/b/c",
        "a/../b",
        "-upload/x",
        "a/b\n",
        "a/b;touch",
        "a/..",
    ],
)
def test_source_remote_cannot_inject_hosts_paths_or_commands(value):
    with pytest.raises(ValueError):
        github_source_url(value)


def test_only_canonical_public_github_sources():
    assert (
        github_source_url("astropy/astropy") == "https://github.com/astropy/astropy.git"
    )


def test_missing_source_requires_network_opt_in(tmp_path: Path):
    manifest = load_localization_dataset_file(
        Path("benchmarks/cases/swe-bench-lite-localization-10-v1.json")
    )
    with pytest.raises(ValueError, match="allow-network"):
        prepare_snapshot(manifest.cases[0].task, tmp_path / "cache")


def test_source_cache_symlink_is_rejected(tmp_path: Path):
    manifest = load_localization_dataset_file(
        Path("benchmarks/cases/swe-bench-lite-localization-10-v1.json")
    )
    root = tmp_path / "cache"
    root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_snapshot(manifest.cases[0].task, root)
