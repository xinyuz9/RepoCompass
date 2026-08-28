"""Golden statistical-analysis and regression artifact tests."""

from pathlib import Path

from benchmarks.generate_localization_analysis_fixture import (
    build_analysis_artifacts,
)
from repocompass.evaluation import (
    LocalizationAnalysisProvenance,
    LocalizationAnalysisReport,
    LocalizationRegressionReport,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.analysis.json"
)
ANALYSIS_SUMMARY_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.analysis.md"
)
REGRESSION_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.regression.json"
)
REGRESSION_SUMMARY_PATH = (
    PROJECT_ROOT / "benchmarks/results/localization-fixture-v1.regression.md"
)


def test_bundled_analysis_and_regression_artifacts_are_reproducible() -> None:
    analysis, analysis_summary, regression, regression_summary = (
        build_analysis_artifacts()
    )
    committed_analysis = LocalizationAnalysisReport.model_validate_json(
        ANALYSIS_PATH.read_text(encoding="utf-8")
    )
    committed_regression = LocalizationRegressionReport.model_validate_json(
        REGRESSION_PATH.read_text(encoding="utf-8")
    )

    assert analysis == committed_analysis
    assert regression == committed_regression
    assert analysis.analysis_id == (
        "79566159b7f411223df25a8af365436fb17bb8134308c02c0f22e42b411ee054"
    )
    assert regression.regression_id == (
        "d07faeaaac96eb0ed135404e1209b84a1a78f9709909760e2583dbded2c0c2d1"
    )
    assert ANALYSIS_SUMMARY_PATH.read_text(encoding="utf-8") == (
        analysis_summary + "\n"
    )
    assert REGRESSION_SUMMARY_PATH.read_text(encoding="utf-8") == (
        regression_summary + "\n"
    )
    assert analysis.provenance is LocalizationAnalysisProvenance.SYNTHETIC_FIXTURE
    assert "not model quality" in analysis_summary
    assert "not a SWE-bench resolved rate" in analysis_summary
    assert regression.passed is True
    assert "Decision: PASS" in regression_summary
