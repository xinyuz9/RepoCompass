"""Unit tests for the RepoCompass command-line assembly boundary."""

import json
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

from repocompass import cli
from repocompass.agent.candidate import CandidateResult, EvidenceItem
from repocompass.agent.evaluation import EvaluationReport, SemanticAssessment
from repocompass.agent.lifecycle import (
    ExecutionStatus,
    HumanDecisionAction,
    HumanReviewPolicy,
    HumanReviewRequest,
)
from repocompass.agent.result import FinalResult
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
    trace.record(
        TraceEventKind.RUN_FINISHED,
        payload={"termination_status": "accepted", "step_stop_reason": None},
    )
    outcome = RunOutcome(
        thread_id="thread-123",
        final_result=final_result,
        termination_status="accepted",
        step_stop_reason=None,
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
            "--human-review-policy",
            "before_finalize",
            "--max-tool-calls-per-step",
            "3",
            "--max-explorer-calls-per-step",
            "2",
            "--max-no-progress-rounds",
            "4",
            "--context-budget-tokens",
            "4096",
            "--trace-dir",
            str(trace_directory),
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ]
    )

    captured = capsys.readouterr()
    request = runner.run.call_args.args[0]
    trace_path = trace_directory / "cli-run.jsonl"
    assert exit_code == 0
    assert create_models.call_count == 1
    runner_factory.assert_called_once_with(model, checkpointer=ANY)
    assert request.repo_path == tmp_path
    assert request.objective == "Inspect authentication."
    assert request.max_tool_calls_per_step == 3
    assert request.max_explorer_calls_per_step == 2
    assert request.max_no_progress_rounds == 4
    assert request.max_plan_revisions == 2
    assert request.context_budget_tokens == 4096
    assert request.human_review_policy is HumanReviewPolicy.BEFORE_FINALIZE
    assert json.loads(captured.out) == final_result.model_dump(mode="json")
    assert "Thread ID: thread-123" in captured.err
    assert "Run ID: cli-run" in captured.err
    assert "Status: accepted" in captured.err
    assert "Last step stop reason: none" in captured.err
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
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "repocompass: error: missing model settings\n"


def test_main_routes_resume_without_rebuilding_start_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary="Recovered result.",
        unresolved_questions=[],
    )
    trace = RunTrace("resume-attempt")
    trace.record(TraceEventKind.RUN_STARTED)
    trace.record(
        TraceEventKind.RUN_FINISHED,
        payload={"termination_status": "accepted", "step_stop_reason": None},
    )
    outcome = RunOutcome(
        thread_id="persisted-thread",
        final_result=final_result,
        termination_status="accepted",
        step_stop_reason=None,
        trace=trace,
    )
    runner = MagicMock()
    runner.resume.return_value = outcome
    monkeypatch.setattr(cli, "AgentRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(cli, "create_workflow_models", MagicMock(return_value=object()))

    exit_code = cli.main(
        [
            "--resume",
            "persisted-thread",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )

    captured = capsys.readouterr()
    resume_request = runner.resume.call_args.args[0]
    assert exit_code == 0
    assert runner.run.call_count == 0
    assert resume_request.thread_id == "persisted-thread"
    assert "Thread ID: persisted-thread" in captured.err


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
                "--max-tool-calls-per-step",
                "0",
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "must be greater than zero" in captured.err


def test_main_rejects_non_positive_context_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                str(tmp_path),
                "--objective",
                "Inspect the repository.",
                "--context-budget-tokens",
                "0",
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "must be greater than zero" in captured.err


def test_main_rejects_context_budget_outside_start_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--resume",
                "persisted-thread",
                "--context-budget-tokens",
                "4096",
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "budget options can only be supplied" in captured.err


def _completed_cli_outcome(
    *,
    thread_id: str,
    run_id: str,
    summary: str,
) -> tuple[RunOutcome, FinalResult]:
    final_result = FinalResult(
        relevant_files=[],
        evidence=[],
        summary=summary,
        unresolved_questions=[],
    )
    trace = RunTrace(run_id)
    trace.record(TraceEventKind.RUN_STARTED)
    trace.record(
        TraceEventKind.RUN_FINISHED,
        payload={"termination_status": "accepted", "step_stop_reason": None},
    )
    return (
        RunOutcome(
            thread_id=thread_id,
            final_result=final_result,
            termination_status="accepted",
            step_stop_reason=None,
            trace=trace,
        ),
        final_result,
    )


def _review_request() -> HumanReviewRequest:
    return HumanReviewRequest(
        turn_number=1,
        candidate_result=CandidateResult(
            evidence=[
                EvidenceItem(
                    path="auth.py",
                    line_start=1,
                    line_end=1,
                    symbol="authenticate",
                    claim="authenticate is defined here.",
                )
            ],
            summary="Authentication is implemented in auth.py.",
            unresolved_questions=[],
        ),
        evaluation_report=EvaluationReport(
            deterministic_issues=[],
            semantic_assessment=SemanticAssessment(
                active_step_completed=True,
                answers_objective=True,
                evidence_supports_summary=True,
                has_critical_omissions=False,
                missing_evidence=[],
                recommended_focus=[],
                summary="The candidate is ready for review.",
            ),
        ),
    )


def test_main_routes_thread_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome, final_result = _completed_cli_outcome(
        thread_id="continued-thread",
        run_id="continue-attempt",
        summary="Continued result.",
    )
    runner = MagicMock()
    runner.continue_thread.return_value = outcome
    monkeypatch.setattr(cli, "AgentRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(cli, "create_workflow_models", MagicMock(return_value=object()))

    exit_code = cli.main(
        [
            "--continue",
            "continued-thread",
            "--prompt",
            "Inspect the follow-up.",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )

    captured = capsys.readouterr()
    request = runner.continue_thread.call_args.args[0]
    assert exit_code == 0
    assert request.thread_id == "continued-thread"
    assert request.prompt == "Inspect the follow-up."
    assert runner.run.call_count == 0
    assert runner.resume.call_count == 0
    assert runner.respond.call_count == 0
    assert json.loads(captured.out) == final_result.model_dump(mode="json")


def test_main_routes_human_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome, final_result = _completed_cli_outcome(
        thread_id="review-thread",
        run_id="respond-attempt",
        summary="Revised result.",
    )
    runner = MagicMock()
    runner.respond.return_value = outcome
    monkeypatch.setattr(cli, "AgentRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(cli, "create_workflow_models", MagicMock(return_value=object()))

    exit_code = cli.main(
        [
            "--respond",
            "review-thread",
            "--interrupt-id",
            "interrupt-123",
            "--decision",
            "revise",
            "--feedback",
            "Inspect middleware registration.",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )

    captured = capsys.readouterr()
    request = runner.respond.call_args.args[0]
    assert exit_code == 0
    assert request.thread_id == "review-thread"
    assert request.interrupt_id == "interrupt-123"
    assert request.decision.action is HumanDecisionAction.REVISE
    assert request.decision.feedback == "Inspect middleware registration."
    assert runner.run.call_count == 0
    assert runner.resume.call_count == 0
    assert runner.continue_thread.call_count == 0
    assert json.loads(captured.out) == final_result.model_dump(mode="json")


def test_main_renders_interrupted_outcome_without_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    review_request = _review_request()
    trace = RunTrace("interrupt-attempt")
    trace.record(TraceEventKind.RUN_STARTED)
    outcome = RunOutcome(
        thread_id="review-thread",
        execution_status=ExecutionStatus.INTERRUPTED,
        human_review_request=review_request,
        interrupt_id="interrupt-123",
        final_result=None,
        termination_status=None,
        step_stop_reason=None,
        trace=trace,
    )
    runner = MagicMock()
    runner.run.return_value = outcome
    monkeypatch.setattr(cli, "AgentRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(cli, "create_workflow_models", MagicMock(return_value=object()))

    exit_code = cli.main(
        [
            str(tmp_path),
            "--objective",
            "Inspect authentication.",
            "--human-review-policy",
            "before_finalize",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )

    captured = capsys.readouterr()
    request = runner.run.call_args.args[0]
    assert exit_code == 0
    assert request.human_review_policy is HumanReviewPolicy.BEFORE_FINALIZE
    assert json.loads(captured.out) == {
        "execution_status": "interrupted",
        "thread_id": "review-thread",
        "interrupt_id": "interrupt-123",
        "human_review_request": review_request.model_dump(mode="json"),
    }
    assert "Status: interrupted" in captured.err
    assert "Interrupt ID: interrupt-123" in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (["--continue", "thread-1"], "--prompt is required with --continue"),
        (
            ["--respond", "thread-1", "--decision", "approve"],
            "--interrupt-id is required with --respond",
        ),
        (
            ["--respond", "thread-1", "--interrupt-id", "interrupt-1"],
            "--decision is required with --respond",
        ),
        (
            ["--resume", "thread-1", "--continue", "thread-2"],
            "not allowed with argument",
        ),
    ],
)
def test_main_rejects_incomplete_or_conflicting_lifecycle_modes(
    arguments: list[str],
    expected_error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert expected_error in captured.err


def test_main_rejects_revise_without_feedback_before_runner_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = MagicMock()
    monkeypatch.setattr(cli, "AgentRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(cli, "create_workflow_models", MagicMock(return_value=object()))

    exit_code = cli.main(
        [
            "--respond",
            "review-thread",
            "--interrupt-id",
            "interrupt-123",
            "--decision",
            "revise",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "feedback is required" in captured.err
    assert runner.respond.call_count == 0
