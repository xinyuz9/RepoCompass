"""Missing system tools must fail before the demo creates persistent state."""

import pytest

from repocompass.dashboard import demo


@pytest.mark.parametrize("missing", ["git", "rg"])
def test_demo_reports_missing_dependency_before_creating_state(
    tmp_path, monkeypatch, missing
):
    monkeypatch.setattr(
        demo.shutil,
        "which",
        lambda executable: None if executable == missing else f"/usr/bin/{executable}",
    )
    destination = tmp_path / "demo"

    with pytest.raises(demo.DashboardDemoError, match=f"requires '{missing}' on PATH"):
        demo.create_dashboard_demo(destination)

    assert not destination.exists()
