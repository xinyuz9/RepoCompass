"""Offline semantic/citation oracle and selective context boundary tests."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repocompass.evaluation import drift_answer as benchmark


def answer(case="unchanged"):
    missing = case in {"deleted_source", "syntax_error"}
    path = "alternate.py" if case == "changed_import" else "policy.py"
    value = (
        None
        if missing
        else 14
        if case == "changed_constant"
        else 26
        if case == "changed_import"
        else 10
    )
    return {
        "answers": [
            {
                "id": "entry",
                "value": value,
                "evidence": ["api.py:1"]
                if missing
                else ["api.py:1", "api.py:4", path + ":1", path + ":4"],
            },
            {
                "id": "alternate",
                "value": 23,
                "evidence": ["alternate.py:1", "alternate.py:4"],
            },
            {"id": "utility", "value": 2, "evidence": ["utility_00.py:2"]},
        ]
    }


@pytest.mark.parametrize("case", benchmark.CASES)
def test_paired_fixture_and_independent_oracle(tmp_path: Path, case: str):
    calls = []

    def query(label, payload):
        calls.append((label, payload))
        # Explicit scripted unit-test oracle, never live performance evidence.
        return {"answer": answer("unchanged" if label.endswith("baseline") else case)}

    result = benchmark.run_case(tmp_path / "fixture", case, 0, query)
    assert len(calls) == 3
    assert result["baseline"]["quality"]["all_correct"]
    assert all(r["quality"]["all_correct"] for r in result["results"].values())
    assert result["arm_order"] == ("full", "selective")
    selective = calls[2][1]
    assert "gold" not in selective and "oracle" not in selective
    if case in {"deleted_source", "outside_source"}:
        assert len(selective["current_sources"]) == 12
        assert selective["prior_claims_with_unchanged_sources"] == []
    elif case == "unchanged":
        assert selective["current_sources"] == {}
        assert len(selective["prior_claims_with_unchanged_sources"]) == 3
    else:
        assert len(selective["current_sources"]) in {1, 2}
        assert {a["id"] for a in selective["prior_claims_with_unchanged_sources"]} == {
            "alternate",
            "utility",
        }


def test_wrong_baseline_is_retained_and_measured_not_silently_corrected(tmp_path: Path):
    wrong = answer()
    wrong["answers"][0]["value"] = 99
    result = benchmark.run_case(
        tmp_path / "fixture", "unchanged", 1, lambda *_: {"answer": wrong}
    )
    assert result["arm_order"] == ("selective", "full")
    assert not result["baseline"]["quality"]["all_correct"]
    assert not result["results"]["selective"]["quality"]["all_correct"]
    assert result["results"]["selective"]["cached_claims_sent"] == 3


def test_invalid_baseline_falls_back_without_filtering_failure(tmp_path: Path):
    result = benchmark.run_case(
        tmp_path / "fixture",
        "unchanged",
        0,
        lambda *_: {"answer": None, "error": "ValidationError"},
    )
    assert result["baseline_parse_fallback"]
    assert result["results"]["selective"]["source_files_sent"] == 12
    assert not result["results"]["full"]["quality"]["all_correct"]


def test_grader_rejects_wrong_values_missing_and_fabricated_citations(tmp_path: Path):
    root = tmp_path / "fixture"
    baseline = benchmark.make_fixture(root)
    sources = benchmark.source_pack(root, [a.path for a in baseline.anchors])
    for edit in ("wrong", "missing", "fabricated"):
        value = answer()
        if edit == "wrong":
            value["answers"][0]["value"] = 99
        elif edit == "missing":
            value["answers"][0]["evidence"] = ["api.py:4"]
        else:
            value["answers"][0]["evidence"].append("secret.py:999")
        assert not benchmark.grade(
            benchmark.AnswerSet.model_validate(value), "unchanged", sources
        )["all_correct"]


def test_answer_schema_rejects_bool_duplicate_and_extra_fields():
    for edit in ("bool", "duplicate", "extra"):
        value = answer()
        if edit == "bool":
            value["answers"][0]["value"] = True
        elif edit == "duplicate":
            value["answers"][0]["id"] = "utility"
        else:
            value["reasoning"] = "Not admitted"
        with pytest.raises(ValidationError):
            benchmark.AnswerSet.model_validate(value)


def test_live_opt_in_and_existing_output_are_checked_before_credentials(
    tmp_path: Path, monkeypatch
):
    args = [
        "drift-answer",
        "--output",
        str(tmp_path),
        "--budget-db",
        str(tmp_path / "budget"),
        "--pricing",
        str(tmp_path / "pricing"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", args + ["--live"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
