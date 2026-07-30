"""Command-line entry point for one RepoCompass investigation run."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import (
    AgentRunner,
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_PLAN_REVISIONS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
    ResumeRequest,
    RunRequest,
)
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.models import create_workflow_models

DEFAULT_TRACE_DIRECTORY = Path(".repocompass/traces")
DEFAULT_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")


def _positive_int(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def create_parser() -> argparse.ArgumentParser:
    """Create the RepoCompass command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="repocompass",
        description=(
            "Investigate a repository with bounded read-only tools and save "
            "a local JSONL domain trace."
        ),
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        type=Path,
        help="trusted repository path for a new investigation",
    )
    parser.add_argument(
        "--objective",
        help="repository investigation question for a new investigation",
    )
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="resume an existing checkpoint thread instead of starting a new run",
    )
    parser.add_argument(
        "--max-tool-calls-per-step",
        type=_positive_int,
        default=DEFAULT_MAX_TOOL_CALLS_PER_STEP,
        help=(
            "tool-call budget for each active plan step "
            f"(default: {DEFAULT_MAX_TOOL_CALLS_PER_STEP})"
        ),
    )
    parser.add_argument(
        "--max-explorer-calls-per-step",
        type=_positive_int,
        default=DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
        help=(
            "Explorer invocation budget for each active plan step "
            f"(default: {DEFAULT_MAX_EXPLORER_CALLS_PER_STEP})"
        ),
    )
    parser.add_argument(
        "--max-plan-revisions",
        type=_positive_int,
        default=DEFAULT_MAX_PLAN_REVISIONS,
        help=(
            f"model-based plan-revision budget (default: {DEFAULT_MAX_PLAN_REVISIONS})"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIRECTORY,
        help=f"directory for per-run JSONL traces (default: {DEFAULT_TRACE_DIRECTORY})",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=DEFAULT_CHECKPOINT_DATABASE,
        help=(
            "local SQLite checkpoint database "
            f"(default: {DEFAULT_CHECKPOINT_DATABASE})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one investigation from command-line arguments."""

    parser = create_parser()
    arguments = parser.parse_args(argv)

    if arguments.resume is None:
        if arguments.repo_path is None:
            parser.error("repo_path is required unless --resume is used")
        if arguments.objective is None:
            parser.error("--objective is required unless --resume is used")
    else:
        if arguments.repo_path is not None:
            parser.error("repo_path cannot be supplied with --resume")
        if arguments.objective is not None:
            parser.error("--objective cannot be supplied with --resume")

    try:
        with open_sqlite_checkpointer(arguments.checkpoint_db) as checkpointer:
            runner = AgentRunner(
                create_workflow_models(),
                checkpointer=checkpointer,
            )
            if arguments.resume is None:
                request = RunRequest(
                    repo_path=arguments.repo_path,
                    objective=arguments.objective,
                    max_tool_calls_per_step=arguments.max_tool_calls_per_step,
                    max_explorer_calls_per_step=(
                        arguments.max_explorer_calls_per_step
                    ),
                    max_plan_revisions=arguments.max_plan_revisions,
                )
                outcome = runner.run(request)
            else:
                outcome = runner.resume(
                    ResumeRequest(thread_id=arguments.resume),
                )

        trace_path = JsonlTraceSink().write(
            outcome.trace,
            arguments.trace_dir / f"{outcome.trace.run_id}.jsonl",
        )
    except (NotImplementedError, OSError, ValueError) as error:
        print(f"repocompass: error: {error}", file=sys.stderr)
        return 1

    print(outcome.final_result.model_dump_json(indent=2))
    print(f"Thread ID: {outcome.thread_id}", file=sys.stderr)
    print(f"Run ID: {outcome.trace.run_id}", file=sys.stderr)
    print(f"Status: {outcome.termination_status}", file=sys.stderr)
    print(
        f"Last step stop reason: {outcome.step_stop_reason or 'none'}",
        file=sys.stderr,
    )
    print(f"Trace: {trace_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
