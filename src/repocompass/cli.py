"""Command-line entry point for one RepoCompass investigation run."""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.lifecycle import (
    ContinueRequest,
    ExecutionStatus,
    HumanDecision,
    HumanDecisionAction,
    HumanResponseRequest,
    HumanReviewPolicy,
)
from repocompass.agent.runner import (
    AgentRunner,
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    DEFAULT_MAX_EXPLORER_CALLS_PER_STEP,
    DEFAULT_MAX_NO_PROGRESS_ROUNDS,
    DEFAULT_MAX_PLAN_REVISIONS,
    DEFAULT_MAX_TOOL_CALLS_PER_STEP,
    ResumeRequest,
    RunOutcome,
    RunRequest,
)
from repocompass.agent.trace_sink import JsonlTraceSink
from repocompass.models import create_workflow_models
from repocompass.storage.raw_store import RawStoreError, open_raw_run_store

DEFAULT_TRACE_DIRECTORY = Path(".repocompass/traces")
DEFAULT_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")
DEFAULT_RAW_STORE_DATABASE = Path(".repocompass/raw_runs.sqlite")
LifecycleMode = Literal["start", "resume", "continue", "respond"]


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
            "local trace and raw provenance records."
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
    lifecycle_modes = parser.add_mutually_exclusive_group()
    lifecycle_modes.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="resume an unfinished checkpoint thread after process interruption",
    )
    lifecycle_modes.add_argument(
        "--continue",
        dest="continue_thread",
        metavar="THREAD_ID",
        help="start a new Human turn on one completed checkpoint thread",
    )
    lifecycle_modes.add_argument(
        "--respond",
        metavar="THREAD_ID",
        help="respond to one pending Human-review interrupt",
    )
    parser.add_argument(
        "--prompt",
        help="new Human prompt supplied only with --continue",
    )
    parser.add_argument(
        "--interrupt-id",
        help="exact pending interrupt identity supplied only with --respond",
    )
    parser.add_argument(
        "--decision",
        type=HumanDecisionAction,
        choices=tuple(HumanDecisionAction),
        help="structured Human action supplied only with --respond",
    )
    parser.add_argument(
        "--feedback",
        help="Human revision feedback supplied only with --respond",
    )
    parser.add_argument(
        "--human-review-policy",
        type=HumanReviewPolicy,
        choices=tuple(HumanReviewPolicy),
        help="optional final-candidate review policy for a new investigation",
    )
    parser.add_argument(
        "--max-tool-calls-per-step",
        type=_positive_int,
        help=(
            "tool-call budget for each active plan step "
            f"(default: {DEFAULT_MAX_TOOL_CALLS_PER_STEP})"
        ),
    )
    parser.add_argument(
        "--max-explorer-calls-per-step",
        type=_positive_int,
        help=(
            "Explorer invocation budget for each active plan step "
            f"(default: {DEFAULT_MAX_EXPLORER_CALLS_PER_STEP})"
        ),
    )
    parser.add_argument(
        "--max-no-progress-rounds",
        type=_positive_int,
        help=(
            "consecutive stalled tool rounds per active plan step "
            f"(default: {DEFAULT_MAX_NO_PROGRESS_ROUNDS})"
        ),
    )
    parser.add_argument(
        "--max-plan-revisions",
        type=_positive_int,
        help=(
            f"model-based plan-revision budget (default: {DEFAULT_MAX_PLAN_REVISIONS})"
        ),
    )
    parser.add_argument(
        "--context-budget-tokens",
        type=_positive_int,
        help=(
            "hard token budget for each model invocation's active context "
            f"(default: {DEFAULT_CONTEXT_BUDGET_TOKENS})"
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
            f"local SQLite checkpoint database (default: {DEFAULT_CHECKPOINT_DATABASE})"
        ),
    )
    parser.add_argument(
        "--raw-store-db",
        type=Path,
        default=DEFAULT_RAW_STORE_DATABASE,
        help=(
            "local SQLite raw provenance database "
            f"(default: {DEFAULT_RAW_STORE_DATABASE})"
        ),
    )
    return parser


def _validate_lifecycle_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> LifecycleMode:
    """Select one lifecycle mode and reject ambiguous payload fields."""

    if arguments.resume is not None:
        mode: LifecycleMode = "resume"
    elif arguments.continue_thread is not None:
        mode = "continue"
    elif arguments.respond is not None:
        mode = "respond"
    else:
        mode = "start"

    if mode == "start":
        if arguments.repo_path is None:
            parser.error("repo_path is required for a new investigation")
        if arguments.objective is None:
            parser.error("--objective is required for a new investigation")
        for option, value in (
            ("--prompt", arguments.prompt),
            ("--interrupt-id", arguments.interrupt_id),
            ("--decision", arguments.decision),
            ("--feedback", arguments.feedback),
        ):
            if value is not None:
                parser.error(f"{option} can only be supplied with its lifecycle mode")
        return mode

    mode_option = {
        "resume": "--resume",
        "continue": "--continue",
        "respond": "--respond",
    }[mode]
    if arguments.repo_path is not None:
        parser.error(f"repo_path cannot be supplied with {mode_option}")
    if arguments.objective is not None:
        parser.error(f"--objective cannot be supplied with {mode_option}")
    if arguments.human_review_policy is not None:
        parser.error(f"--human-review-policy cannot be supplied with {mode_option}")
    if any(
        value is not None
        for value in (
            arguments.max_tool_calls_per_step,
            arguments.max_explorer_calls_per_step,
            arguments.max_no_progress_rounds,
            arguments.max_plan_revisions,
            arguments.context_budget_tokens,
        )
    ):
        parser.error("budget options can only be supplied for a new investigation")

    if mode == "resume":
        for option, value in (
            ("--prompt", arguments.prompt),
            ("--interrupt-id", arguments.interrupt_id),
            ("--decision", arguments.decision),
            ("--feedback", arguments.feedback),
        ):
            if value is not None:
                parser.error(f"{option} cannot be supplied with --resume")
        return mode

    if mode == "continue":
        if arguments.prompt is None:
            parser.error("--prompt is required with --continue")
        for option, value in (
            ("--interrupt-id", arguments.interrupt_id),
            ("--decision", arguments.decision),
            ("--feedback", arguments.feedback),
        ):
            if value is not None:
                parser.error(f"{option} cannot be supplied with --continue")
        return mode

    if arguments.prompt is not None:
        parser.error("--prompt cannot be supplied with --respond")
    if arguments.interrupt_id is None:
        parser.error("--interrupt-id is required with --respond")
    if arguments.decision is None:
        parser.error("--decision is required with --respond")
    return mode


def _render_outcome(outcome: RunOutcome) -> str:
    """Render completed and interrupted outcomes without exposing graph state."""

    if outcome.execution_status is ExecutionStatus.COMPLETED:
        if outcome.final_result is None:
            raise ValueError("Completed outcome is missing its final result")
        return outcome.final_result.model_dump_json(indent=2)

    review_request = outcome.human_review_request
    interrupt_id = outcome.interrupt_id
    if review_request is None or interrupt_id is None:
        raise ValueError("Interrupted outcome is missing review metadata")
    return json.dumps(
        {
            "execution_status": outcome.execution_status.value,
            "thread_id": outcome.thread_id,
            "interrupt_id": interrupt_id,
            "human_review_request": review_request.model_dump(mode="json"),
        },
        ensure_ascii=False,
        indent=2,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one investigation from command-line arguments."""

    parser = create_parser()
    arguments = parser.parse_args(argv)
    lifecycle_mode = _validate_lifecycle_arguments(parser, arguments)

    try:
        with (
            open_sqlite_checkpointer(arguments.checkpoint_db) as checkpointer,
            open_raw_run_store(arguments.raw_store_db) as raw_run_store,
        ):
            runner = AgentRunner(
                create_workflow_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_run_store,
            )
            if lifecycle_mode == "start":
                outcome = runner.run(
                    RunRequest(
                        repo_path=arguments.repo_path,
                        objective=arguments.objective,
                        max_tool_calls_per_step=(
                            arguments.max_tool_calls_per_step
                            or DEFAULT_MAX_TOOL_CALLS_PER_STEP
                        ),
                        max_explorer_calls_per_step=(
                            arguments.max_explorer_calls_per_step
                            or DEFAULT_MAX_EXPLORER_CALLS_PER_STEP
                        ),
                        max_no_progress_rounds=(
                            arguments.max_no_progress_rounds
                            or DEFAULT_MAX_NO_PROGRESS_ROUNDS
                        ),
                        max_plan_revisions=(
                            arguments.max_plan_revisions or DEFAULT_MAX_PLAN_REVISIONS
                        ),
                        context_budget_tokens=(
                            arguments.context_budget_tokens
                            or DEFAULT_CONTEXT_BUDGET_TOKENS
                        ),
                        human_review_policy=(
                            arguments.human_review_policy or HumanReviewPolicy.NEVER
                        ),
                    )
                )
            elif lifecycle_mode == "resume":
                outcome = runner.resume(
                    ResumeRequest(thread_id=arguments.resume),
                )
            elif lifecycle_mode == "continue":
                outcome = runner.continue_thread(
                    ContinueRequest(
                        thread_id=arguments.continue_thread,
                        prompt=arguments.prompt,
                    )
                )
            else:
                decision = HumanDecision(
                    action=arguments.decision,
                    feedback=arguments.feedback,
                )
                outcome = runner.respond(
                    HumanResponseRequest(
                        thread_id=arguments.respond,
                        interrupt_id=arguments.interrupt_id,
                        decision=decision,
                    )
                )

        trace_path = JsonlTraceSink().write(
            outcome.trace,
            arguments.trace_dir / f"{outcome.trace.run_id}.jsonl",
        )
        rendered_outcome = _render_outcome(outcome)
    except (
        NotImplementedError,
        OSError,
        ValueError,
        sqlite3.Error,
        RawStoreError,
    ) as error:
        print(f"repocompass: error: {error}", file=sys.stderr)
        return 1

    print(rendered_outcome)
    print(f"Thread ID: {outcome.thread_id}", file=sys.stderr)
    print(f"Run ID: {outcome.trace.run_id}", file=sys.stderr)
    status = outcome.termination_status or outcome.execution_status.value
    print(f"Status: {status}", file=sys.stderr)
    if outcome.interrupt_id is not None:
        print(f"Interrupt ID: {outcome.interrupt_id}", file=sys.stderr)
    print(
        f"Last step stop reason: {outcome.step_stop_reason or 'none'}",
        file=sys.stderr,
    )
    print(f"Trace: {trace_path}", file=sys.stderr)
    print(f"Raw store: {arguments.raw_store_db.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
