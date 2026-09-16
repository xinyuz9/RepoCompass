import pytest

from repocompass.evaluation.drift_recheck import CASES, run_case


@pytest.mark.parametrize("case", CASES)
def test_independent_recheck_obligations_and_full_index_agree(case):
    result = run_case(case, repeats=1, file_count=4, functions=2)
    assert result["structural_projection_equal"]
    assert result["hand_labelled_obligations_pass"]
    assert result["semantic_equivalence"] == "not_measured"
    assert result["provider_calls"] == 0
    full, partial = (result["samples"][name][0] for name in ("full", "selective"))
    assert full["indexed_files"] == 4
    assert full["source_read_attempts"] == 4
    assert partial["source_read_attempts"] == 4 + partial["indexed_files"]
    assert partial["elapsed_ns"] > 0
    if case == "unchanged":
        assert partial["indexed_files"] == 0
        assert full["source_bytes_returned"] == partial["source_bytes_returned"]
    if case == "renamed_symbol":
        assert result["recheck_paths"] == ["api.py", "policy.py"]
        assert partial["source_bytes_returned"] > full["source_bytes_returned"]


def test_recheck_experiment_is_bounded():
    with pytest.raises(ValueError):
        run_case("not-a-case")
    with pytest.raises(ValueError):
        run_case("unchanged", repeats=21)
