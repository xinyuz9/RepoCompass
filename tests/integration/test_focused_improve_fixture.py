"""Portable focused-improve fixture and committed report regression."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.generate_focused_improve_fixture import (
    DEFAULT_CASE,
    DEFAULT_OUTPUT,
    build_report,
    derive_portable_report_id,
)


def test_focused_improve_fixture_is_portable_and_matches_committed_report(
    tmp_path: Path,
) -> None:
    first = build_report(DEFAULT_CASE, workspace_root=tmp_path / "first")
    second = build_report(DEFAULT_CASE, workspace_root=tmp_path / "second")
    committed = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    assert first == second == committed
    identity_payload = dict(first)
    report_id = identity_payload.pop("portable_report_id")
    assert report_id == derive_portable_report_id(identity_payload)
    assert first["execution"]["accepted"] is True
    assert first["execution"]["stop_reason"] == "accepted"
    assert first["execution"]["dry_run_source_unchanged"] is True
    assert first["execution"]["apply_performed"] is False
    assert first["sandbox"]["backend"] == "bubblewrap"
    assert first["sandbox"]["network_access"] is False
    assert first["cost_semantics"]["model_calls"] == 0
    assert first["cost_semantics"]["provider_cost_status"] == "not_applicable"
