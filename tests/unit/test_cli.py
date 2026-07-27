"""Unit tests for the RepoCompass command-line assembly boundary."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repocompass import cli
from repocompass.agent.contracts import FinalResult
from repocompass.agent.runner import RunOutcome
from repocompass.agent.trace import RunTrace, TraceEventKind


def test_main_runs_investigation_and_writes_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = object()
    final_result = FinalResult(
        relevant_files=["auth.py"],
        evidence=["auth.py:1 defines authenticate."],
        summary="Authentication is implemented in auth.py.",
        unresolved_questions=[],
    )
    trace = RunTrace("cli-run")
    trace.record(TraceEventKind.RUN_STARTED)
    trace.record(TraceEventKind.RUN_FINISHED, payload={"stop_reason": None})
    outcome = RunOutcome(
        final_result=final_result,
        stop_reason=None,
        trace=trace,
    )
    runner = MagicMock()
    runner.run.return_value = outcome
    runner_factory = MagicMock(return_value=runner)
    create_models = MagicMock(return_value=model)
    monkeypatch.setattr(cli, "AgentRunner", runner_factory)
    monkeypatch.setattr(cli, "create_workflow_models", create_models)
    trace_directory = tmp_path / "traces"

    exit_code = cli.main(
        [
            str(tmp_path),
            "--objective",
            "Inspect authentication.",
            "--max-tool-calls",
            "3",
            "--max-iterations",
            "2",
            "--trace-dir",
            str(trace_directory),
        ]
    )

    captured = capsys.readouterr()
    request = runner.run.call_args.args[0]
    trace_path = trace_directory / "cli-run.jsonl"
    assert exit_code == 0
    assert create_models.call_count == 1
    runner_factory.assert_called_once_with(model)
    assert request.repo_path == tmp_path
    assert request.objective == "Inspect authentication."
    assert request.max_tool_calls == 3
    assert request.max_iterations == 2
    assert json.loads(captured.out) == final_result.model_dump(mode="json")
    assert "Run ID: cli-run" in captured.err
    assert "Stop reason: completed" in captured.err
    assert f"Trace: {trace_path.resolve()}" in captured.err
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 2


def test_main_reports_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    create_models = MagicMock(side_effect=ValueError("missing model settings"))
    monkeypatch.setattr(cli, "create_workflow_models", create_models)

    exit_code = cli.main(
        [
            str(tmp_path),
            "--objective",
            "Inspect the repository.",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "repocompass: error: missing model settings\n"


def test_main_rejects_non_positive_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path),
                "--objective",
                "Inspect the repository.",
                "--max-tool-calls",
                "0",
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "must be greater than zero" in captured.err
