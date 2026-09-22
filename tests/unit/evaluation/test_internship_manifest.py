"""Freeze the small cross-repository pilot before looking at model outputs."""

from collections import Counter
from pathlib import Path

from repocompass.evaluation import load_localization_dataset_file

CASES = Path(__file__).resolve().parents[3] / "benchmarks" / "cases"
EXPECTED_IDS = (
    "astropy__astropy-14995",
    "astropy__astropy-6938",
    "django__django-10914",
    "django__django-10924",
)


def test_internship_manifest_has_two_repositories_and_four_frozen_cases() -> None:
    manifest = load_localization_dataset_file(CASES / "internship-cross-repo-4-v1.json")
    assert manifest.manifest_id == (
        "c991fbb555620b3c4d118953cbedfcd85d32015dc839752e00634dc105b6d354"
    )
    assert tuple(case.task.instance_id for case in manifest.cases) == EXPECTED_IDS
    assert Counter(case.task.repository for case in manifest.cases) == {
        "astropy/astropy": 2,
        "django/django": 2,
    }
    assert manifest.evaluation_scope == "localization_only"


def test_internship_cases_are_unchanged_upstream_records_not_rewritten_labels() -> None:
    upstream = load_localization_dataset_file(
        CASES / "swe-bench-lite-localization-10-v1.json"
    )
    subset = load_localization_dataset_file(CASES / "internship-cross-repo-4-v1.json")
    assert upstream.manifest_id == (
        "4a7495dff7a966adb42accba05e123542560dacab3644977140c97d114d85355"
    )
    records = {case.task.instance_id: case for case in upstream.cases}
    assert all(case == records[case.task.instance_id] for case in subset.cases)
    assert not set(EXPECTED_IDS) & {
        "astropy__astropy-12907",
        "astropy__astropy-14182",
        "astropy__astropy-14365",
    }


def test_cross_repo_selection_cannot_be_replaced_by_the_first_six_records() -> None:
    upstream = load_localization_dataset_file(
        CASES / "swe-bench-lite-localization-10-v1.json"
    )
    assert {case.task.repository for case in upstream.cases[:6]} == {"astropy/astropy"}
    subset = load_localization_dataset_file(CASES / "internship-cross-repo-4-v1.json")
    tasks = subset.agent_tasks()
    assert all("labels" not in task.model_dump() for task in tasks)
