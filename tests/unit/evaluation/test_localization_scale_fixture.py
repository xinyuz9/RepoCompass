"""Forty-case synthetic scale fixture and statistical closure tests."""

from collections import Counter
from pathlib import Path

from benchmarks.generate_localization_scale_fixture import (
    CASE_COUNT,
    build_scale_artifacts,
)
from repocompass.evaluation import (
    LocalizationAnalysisProvenance,
    LocalizationAnalysisReport,
    LocalizationEvaluationReport,
    LocalizationRegressionReport,
    LocalizationVariant,
    load_localization_dataset_file,
    load_localization_observation_set_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_PATH = PROJECT_ROOT / "benchmarks/cases/localization-scale-40-v1.json"
OBSERVATION_PATH = (
    PROJECT_ROOT / "benchmarks/observations/localization-scale-40-v1.json"
)
REPORT_PATH = PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.example.json"
REPORT_SUMMARY_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.example.md"
)
ANALYSIS_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.analysis.json"
)
ANALYSIS_SUMMARY_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.analysis.md"
)
REGRESSION_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.regression.json"
)
REGRESSION_SUMMARY_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-scale-40-v1.regression.md"
)


def test_scale_fixture_closes_40_by_5_statistical_pipeline() -> None:
    (
        manifest,
        observations,
        report,
        report_summary,
        analysis,
        analysis_summary,
        regression,
        regression_summary,
    ) = build_scale_artifacts()

    assert manifest == load_localization_dataset_file(CASE_PATH)
    assert observations == load_localization_observation_set_file(OBSERVATION_PATH)
    assert report == LocalizationEvaluationReport.model_validate_json(
        REPORT_PATH.read_text(encoding="utf-8")
    )
    assert analysis == LocalizationAnalysisReport.model_validate_json(
        ANALYSIS_PATH.read_text(encoding="utf-8")
    )
    assert regression == LocalizationRegressionReport.model_validate_json(
        REGRESSION_PATH.read_text(encoding="utf-8")
    )
    assert REPORT_SUMMARY_PATH.read_text(encoding="utf-8") == report_summary + "\n"
    assert ANALYSIS_SUMMARY_PATH.read_text(encoding="utf-8") == (
        analysis_summary + "\n"
    )
    assert REGRESSION_SUMMARY_PATH.read_text(encoding="utf-8") == (
        regression_summary + "\n"
    )
    assert manifest.manifest_id == (
        "f88e8822c07af347a50e7cf95a975b479c7aeba6d76c295a2eec21deff9c67aa"
    )
    assert observations.set_id == (
        "5b5a9529dc8667cac20133b05e0228e03dc079a03593aa10e4a55fc9d4cc8cf6"
    )
    assert report.report_id == (
        "32c6ba817e5e9d2d0f3ef9d10070e3a37ff9d95f38a45a3abf2ba125e162aa99"
    )
    assert analysis.analysis_id == (
        "a477e4ac7dc9ed436baaf326136b072bdf35c2f5529069d6e0f84962620eee9c"
    )
    assert regression.regression_id == (
        "724aa41f23a0e3176d5f6f31e73ca34446a361bc1e32379ff2285c730600f9e9"
    )

    assert len(manifest.cases) == CASE_COUNT
    assert len(observations.observations) == CASE_COUNT * len(LocalizationVariant)
    assert Counter(item.variant for item in observations.observations) == {
        variant: CASE_COUNT for variant in LocalizationVariant
    }
    assert all(item.unique_case_count == CASE_COUNT for item in report.aggregates)
    assert analysis.provenance is LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE
    assert analysis.bootstrap_config.resamples == 2_000
    assert len(analysis.metric_intervals) == 60
    assert all(
        item.estimated_observations == CASE_COUNT for item in analysis.cost_aggregates
    )
    assert all(
        item.total_estimated_cost_usd is not None for item in analysis.cost_aggregates
    )
    assert regression.passed is True
    assert len(regression.checks) == 30
    assert all(item.paired_case_count == CASE_COUNT for item in regression.checks)
    assert "not model quality" in analysis_summary
    assert "not a SWE-bench resolved rate" in analysis_summary
