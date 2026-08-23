"""Cross-process smoke coverage for the public CLI lifecycle."""

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.cli_lifecycle_driver",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_interrupts_then_approves_from_a_new_process(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("value = True\n", encoding="utf-8")
    checkpoint_database = tmp_path / "state" / "checkpoints.sqlite"
    trace_directory = tmp_path / "traces"

    interrupted = _run_cli(
        str(repository),
        "--objective",
        "Inspect the repository.",
        "--human-review-policy",
        "before_finalize",
        "--checkpoint-db",
        str(checkpoint_database),
        "--trace-dir",
        str(trace_directory),
    )

    assert interrupted.returncode == 0, interrupted.stderr
    interrupted_payload = json.loads(interrupted.stdout)
    assert interrupted_payload["execution_status"] == "interrupted"
    assert interrupted_payload["thread_id"]
    assert interrupted_payload["interrupt_id"]
    assert "Status: interrupted" in interrupted.stderr
    assert checkpoint_database.is_file()

    approved = _run_cli(
        "--respond",
        interrupted_payload["thread_id"],
        "--interrupt-id",
        interrupted_payload["interrupt_id"],
        "--decision",
        "approve",
        "--checkpoint-db",
        str(checkpoint_database),
        "--trace-dir",
        str(trace_directory),
    )

    assert approved.returncode == 0, approved.stderr
    assert json.loads(approved.stdout) == {
        "relevant_files": ["example.py"],
        "evidence": ["example.py:1 contains the relevant implementation."],
        "summary": "The relevant implementation is in example.py.",
        "unresolved_questions": [],
    }
    assert f"Thread ID: {interrupted_payload['thread_id']}" in approved.stderr
    assert "Status: accepted" in approved.stderr

    traces = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in trace_directory.glob("*.jsonl")
    ]
    assert len(traces) == 2
    interrupted_trace = next(
        trace for trace in traces if trace[-1]["kind"] == "run_interrupted"
    )
    approved_trace = next(
        trace for trace in traces if trace[-1]["kind"] == "run_finished"
    )
    assert interrupted_trace[-1]["payload"] == {
        "thread_id": interrupted_payload["thread_id"],
        "execution_mode": "start",
        "interrupt_id": interrupted_payload["interrupt_id"],
        "turn_number": 1,
    }
    assert approved_trace[0]["payload"]["human_decision"] == {
        "action": "approve",
        "feedback_provided": False,
    }
    assert approved_trace[-1]["payload"] == {
        "thread_id": interrupted_payload["thread_id"],
        "execution_mode": "respond",
        "termination_status": "accepted",
        "step_stop_reason": None,
    }


def test_cli_reports_missing_thread_without_a_traceback(tmp_path: Path) -> None:
    result = _run_cli(
        "--resume",
        "missing-thread",
        "--checkpoint-db",
        str(tmp_path / "checkpoints.sqlite"),
        "--trace-dir",
        str(tmp_path / "traces"),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        result.stderr
        == "repocompass: error: Checkpoint thread does not exist: missing-thread\n"
    )
    assert "Traceback" not in result.stderr


def test_cli_reports_invalid_database_path_without_a_traceback(
    tmp_path: Path,
) -> None:
    database_directory = tmp_path / "database-directory"
    database_directory.mkdir()

    result = _run_cli(
        "--resume",
        "missing-thread",
        "--checkpoint-db",
        str(database_directory),
        "--trace-dir",
        str(tmp_path / "traces"),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("repocompass: error: ")
    assert "database" in result.stderr.lower()
    assert "Traceback" not in result.stderr
