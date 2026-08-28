"""Frozen SWE-bench-derived localization manifest and generator tests."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from benchmarks.generate_swebench_localization_manifest import (
    FROZEN_SOURCE_PROFILES,
    MANIFEST_VERSION,
    UPSTREAM_DATASET,
    UPSTREAM_RESPONSE_SHA256,
    UPSTREAM_REVISION,
    build_manifest,
    normalize_upstream_rows,
)
from repocompass.evaluation import (
    LocalizationDatasetKind,
    LocalizationLabelSource,
    load_localization_dataset_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "benchmarks/cases/swe-bench-lite-localization-10-v1.json"
MANIFEST_40_PATH = (
    PROJECT_ROOT / "benchmarks/cases/swe-bench-lite-localization-40-v1.json"
)
EXPECTED_INSTANCE_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-14182",
    "astropy__astropy-14365",
    "astropy__astropy-14995",
    "astropy__astropy-6938",
    "astropy__astropy-7746",
    "django__django-10914",
    "django__django-10924",
    "django__django-11001",
    "django__django-11019",
)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return set(value).union(
            key for item in value.values() for key in _walk_keys(item)
        )
    if isinstance(value, list):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def test_bundled_manifest_is_frozen_traceable_and_patch_body_free() -> None:
    manifest = load_localization_dataset_file(MANIFEST_PATH)
    raw_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest.manifest_id == (
        "4a7495dff7a966adb42accba05e123542560dacab3644977140c97d114d85355"
    )
    assert manifest.version == MANIFEST_VERSION
    assert manifest.kind is LocalizationDatasetKind.SWE_BENCH_LITE_DERIVED
    assert tuple(case.task.instance_id for case in manifest.cases) == (
        EXPECTED_INSTANCE_IDS
    )
    assert sum(len(case.labels.gold_files) for case in manifest.cases) == 10
    assert sum(len(case.labels.gold_symbols) for case in manifest.cases) == 18
    assert all(
        case.labels.source is LocalizationLabelSource.SWE_BENCH_GOLD_PATCH
        for case in manifest.cases
    )
    assert all(len(case.labels.source_artifact_digest) == 64 for case in manifest.cases)
    assert _walk_keys(raw_payload).isdisjoint(
        {"patch", "test_patch", "hints_text", "FAIL_TO_PASS", "PASS_TO_PASS"}
    )
    assert "diff --git" not in MANIFEST_PATH.read_text(encoding="utf-8")
    assert UPSTREAM_DATASET == "SWE-bench/SWE-bench_Lite"
    assert UPSTREAM_REVISION == "b0dde1093fe417d83b7184254edf8199c1f0dff5"
    assert UPSTREAM_RESPONSE_SHA256 == (
        "b8bbed5cd0d76b3ef16bb8801a169be209855a85bffb433be9c970ba7878fce9"
    )


def test_frozen_source_profiles_preserve_10_case_identity_and_add_40() -> None:
    ten = FROZEN_SOURCE_PROFILES[10]
    forty = FROZEN_SOURCE_PROFILES[40]

    assert ten.response_sha256 == UPSTREAM_RESPONSE_SHA256
    assert ten.manifest_name == "swe-bench-lite-localization-10"
    assert ten.output_path == MANIFEST_PATH
    assert forty.case_count == 40
    assert forty.response_sha256 == (
        "24f401102bd0ab4e946f447106c873a9ff645317bdb8161782f59b8820da377e"
    )
    assert forty.manifest_name == "swe-bench-lite-localization-40"
    assert forty.output_path == MANIFEST_40_PATH


def test_bundled_40_case_manifest_is_traceable_and_patch_body_free() -> None:
    manifest = load_localization_dataset_file(MANIFEST_40_PATH)
    raw_payload = json.loads(MANIFEST_40_PATH.read_text(encoding="utf-8"))

    assert manifest.manifest_id == (
        "2d783f33fd5d6e7cffd81f471f8fe56e627fb9ca71255001ad100ecc4b8457a6"
    )
    assert manifest.name == "swe-bench-lite-localization-40"
    assert manifest.version == MANIFEST_VERSION
    assert len(manifest.cases) == 40
    assert sum(len(case.labels.gold_files) for case in manifest.cases) == 40
    assert sum(len(case.labels.gold_symbols) for case in manifest.cases) == 52
    assert tuple(case.task.instance_id for case in manifest.cases[:10]) == (
        EXPECTED_INSTANCE_IDS
    )
    assert all(
        case.labels.source is LocalizationLabelSource.SWE_BENCH_GOLD_PATCH
        for case in manifest.cases
    )
    assert _walk_keys(raw_payload).isdisjoint(
        {"patch", "test_patch", "hints_text", "FAIL_TO_PASS", "PASS_TO_PASS"}
    )


def test_generator_passes_only_public_task_fields_to_agent_boundary() -> None:
    manifest = load_localization_dataset_file(MANIFEST_PATH)
    agent_payload = json.dumps(
        [task.model_dump(mode="json") for task in manifest.agent_tasks()],
        sort_keys=True,
    )
    assert "gold_files" not in agent_payload
    assert "gold_symbols" not in agent_payload
    assert all(
        case.labels.source_artifact_digest not in agent_payload
        for case in manifest.cases
    )


def test_generator_derives_symbols_from_normalized_row_and_base_source() -> None:
    patch = """\
diff --git a/src/pkg.py b/src/pkg.py
--- a/src/pkg.py
+++ b/src/pkg.py
@@ -1,2 +1,2 @@ def target():
 def target():
-    return 1
+    return 2
"""
    payload = {
        "rows": [
            {
                "row": {
                    "instance_id": "example__repo-1",
                    "repo": "example/repo",
                    "base_commit": "a" * 40,
                    "problem_statement": "Locate target.\n",
                    "patch": patch,
                    "test_patch": "must not cross the normalizer",
                }
            }
        ]
    }
    rows = normalize_upstream_rows(payload, expected_count=1)
    calls: list[tuple[str, str, str]] = []

    def source_loader(repository: str, commit: str, path: str) -> str:
        calls.append((repository, commit, path))
        return "def target():\n    return 1\n"

    manifest = build_manifest(
        rows,
        source_loader=source_loader,
        name="synthetic-generator",
        version="synthetic-generator-v1",
    )
    case = manifest.cases[0]

    assert calls == [("example/repo", "a" * 40, "src/pkg.py")]
    assert case.task.objective == "Locate target."
    assert case.labels.gold_files == ("src/pkg.py",)
    assert tuple(item.qualified_name for item in case.labels.gold_symbols) == (
        "src.pkg.target",
    )
    assert "test_patch" not in case.model_dump_json()


@pytest.mark.parametrize(
    "payload, message",
    (
        ({"rows": []}, "unexpected case count"),
        ({"rows": [{"row": {}}]}, "invalid instance_id"),
        ({"rows": [{"row": "not-an-object"}]}, "contain an object"),
    ),
)
def test_upstream_normalizer_fails_closed(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_upstream_rows(payload, expected_count=1)
