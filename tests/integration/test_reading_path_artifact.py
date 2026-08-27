"""Installed-contract smoke for the deterministic M4.3 public artifact."""

import subprocess
import sys
from pathlib import Path

from repocompass.structure import (
    ReadingStepKind,
    StructuralVerificationStatus,
    TraceableReadingPath,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
ARTIFACT = FIXTURES / "reading_path_artifact.json"
GENERATOR = FIXTURES / "generate_reading_path_artifact.py"


def test_committed_reading_path_artifact_is_current_and_strict() -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = TraceableReadingPath.model_validate_json(
        ARTIFACT.read_text(encoding="utf-8")
    )
    assert (
        artifact.claim_lineages[0].verification_status
        is StructuralVerificationStatus.VERIFIED
    )
    assert artifact.claim_lineages[0].provenance_complete is True
    assert artifact.steps[0].kind is ReadingStepKind.FILE
    assert all(
        location.path == step.path
        for step in artifact.steps
        for location in step.source_locations
    )
