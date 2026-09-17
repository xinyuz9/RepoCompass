"""Command-line entry point for one RepoCompass investigation run."""

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.cost import (
    AttemptUsageCostReport,
    PricingTable,
    validate_cost_budget_usd,
)
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
from repocompass.memory.contracts import (
    MemoryScope,
    create_memory_retention_policy,
)
from repocompass.memory.store import MemoryStoreError, open_memory_store
from repocompass.pricing import load_pricing_table_file
from repocompass.storage.raw_store import RawStoreError, open_raw_run_store
from repocompass.tools.gateway import ToolExecutionContext
from repocompass.tools.browser import (
    BrowserPolicy,
    add_browser_arguments,
    validate_browser_runtime,
)

DEFAULT_TRACE_DIRECTORY = Path(".repocompass/traces")
DEFAULT_CHECKPOINT_DATABASE = Path(".repocompass/checkpoints.sqlite")
DEFAULT_RAW_STORE_DATABASE = Path(".repocompass/raw_runs.sqlite")
DEFAULT_MEMORY_DATABASE = Path(".repocompass/memory.sqlite")
LifecycleMode = Literal[
    "start",
    "resume",
    "continue",
    "respond",
    "usage_report",
    "memory_list",
    "memory_list_deleted",
    "memory_delete",
    "memory_retain",
]


def _positive_int(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def _non_negative_int(value: str) -> int:
    """Parse one non-negative command-line integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _positive_cost_budget(value: str) -> str:
    """Parse one canonical positive decimal USD command-line budget."""

    try:
        return validate_cost_budget_usd(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _memory_result_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 10:
        raise argparse.ArgumentTypeError("must not exceed 10")
    return parsed


def _memory_management_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 50:
        raise argparse.ArgumentTypeError("must not exceed 50")
    return parsed


def _retention_deletion_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 1_000:
        raise argparse.ArgumentTypeError("must not exceed 1000")
    return parsed


def _retention_age_days(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 36_500:
        raise argparse.ArgumentTypeError("must not exceed 36500")
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
    add_browser_arguments(parser)
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
    lifecycle_modes.add_argument(
        "--usage-report",
        metavar="RUN_ID",
        help="query one persisted run's machine-readable usage/cost report",
    )
    lifecycle_modes.add_argument(
        "--memory-list",
        action="store_true",
        help="list payload-free live memory metadata for repo_path",
    )
    lifecycle_modes.add_argument(
        "--memory-list-deleted",
        action="store_true",
        help="list payload-free memory tombstones for repo_path",
    )
    lifecycle_modes.add_argument(
        "--memory-delete",
        metavar="MEMORY_ID",
        help="explicitly delete one repository-scoped memory by digest",
    )
    lifecycle_modes.add_argument(
        "--memory-retain",
        metavar="MAX_EPISODES",
        type=_non_negative_int,
        help="apply bounded episodic retention for repo_path",
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
        "--max-attempt-cost-usd",
        type=_positive_cost_budget,
        help=(
            "canonical positive USD budget for a new attempt, or comparison "
            "limit for --usage-report"
        ),
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        help=(
            "strict versioned JSON pricing document used for cost estimates "
            "and required for cost budgets or --usage-report"
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
    parser.add_argument(
        "--memory",
        action="store_true",
        help=(
            "enable commit-aware recall and terminal episode consolidation; "
            "repeat this flag for resume, continue, and respond"
        ),
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=DEFAULT_MEMORY_DATABASE,
        help=(
            "local SQLite memory database used only with --memory "
            f"(default: {DEFAULT_MEMORY_DATABASE})"
        ),
    )
    parser.add_argument(
        "--max-repository-memory-results",
        type=_memory_result_limit,
        help="repository memory candidates per new turn (default: 5, max: 10)",
    )
    parser.add_argument(
        "--max-episodic-memory-results",
        type=_memory_result_limit,
        help="episode history candidates per new turn (default: 5, max: 10)",
    )
    parser.add_argument(
        "--memory-management-limit",
        type=_memory_management_limit,
        help="metadata or tombstone rows to list (default: 20, max: 50)",
    )
    parser.add_argument(
        "--memory-scope",
        type=MemoryScope,
        choices=tuple(MemoryScope),
        help="optional repository or episodic filter for --memory-list",
    )
    parser.add_argument(
        "--memory-max-age-days",
        type=_retention_age_days,
        help="optional episode age bound supplied only with --memory-retain",
    )
    parser.add_argument(
        "--memory-max-deletions",
        type=_retention_deletion_limit,
        help="deletion bound for one retention run (default: 100, max: 1000)",
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
    elif arguments.usage_report is not None:
        mode = "usage_report"
    elif arguments.memory_list:
        mode = "memory_list"
    elif arguments.memory_list_deleted:
        mode = "memory_list_deleted"
    elif arguments.memory_delete is not None:
        mode = "memory_delete"
    elif arguments.memory_retain is not None:
        mode = "memory_retain"
    else:
        mode = "start"

    memory_management_modes = {
        "memory_list",
        "memory_list_deleted",
        "memory_delete",
        "memory_retain",
    }
    if arguments.browser_page and mode in {*memory_management_modes, "usage_report"}:
        parser.error("--browser-page is only valid for investigation execution")
    if mode in memory_management_modes:
        mode_option = {
            "memory_list": "--memory-list",
            "memory_list_deleted": "--memory-list-deleted",
            "memory_delete": "--memory-delete",
            "memory_retain": "--memory-retain",
        }[mode]
        if arguments.repo_path is None:
            parser.error(f"repo_path is required with {mode_option}")
        if arguments.objective is not None:
            parser.error(f"--objective cannot be supplied with {mode_option}")
        if arguments.memory:
            parser.error(f"--memory cannot be supplied with {mode_option}")
        for option, value in (
            ("--prompt", arguments.prompt),
            ("--interrupt-id", arguments.interrupt_id),
            ("--decision", arguments.decision),
            ("--feedback", arguments.feedback),
            ("--human-review-policy", arguments.human_review_policy),
            ("--pricing-file", arguments.pricing_file),
            ("--max-attempt-cost-usd", arguments.max_attempt_cost_usd),
            ("--max-tool-calls-per-step", arguments.max_tool_calls_per_step),
            (
                "--max-explorer-calls-per-step",
                arguments.max_explorer_calls_per_step,
            ),
            ("--max-no-progress-rounds", arguments.max_no_progress_rounds),
            ("--max-plan-revisions", arguments.max_plan_revisions),
            ("--context-budget-tokens", arguments.context_budget_tokens),
            (
                "--max-repository-memory-results",
                arguments.max_repository_memory_results,
            ),
            (
                "--max-episodic-memory-results",
                arguments.max_episodic_memory_results,
            ),
        ):
            if value is not None:
                parser.error(f"{option} cannot be supplied with {mode_option}")
        if mode not in {"memory_list", "memory_list_deleted"} and (
            arguments.memory_management_limit is not None
        ):
            parser.error(
                f"--memory-management-limit cannot be supplied with {mode_option}"
            )
        if mode != "memory_list" and arguments.memory_scope is not None:
            parser.error(f"--memory-scope cannot be supplied with {mode_option}")
        if mode != "memory_retain" and (
            arguments.memory_max_age_days is not None
            or arguments.memory_max_deletions is not None
        ):
            parser.error("retention bounds can only be supplied with --memory-retain")
        return mode

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
        if (
            arguments.max_attempt_cost_usd is not None
            and arguments.pricing_file is None
        ):
            parser.error("--pricing-file is required with --max-attempt-cost-usd")
        if any(
            value is not None
            for value in (
                arguments.memory_management_limit,
                arguments.memory_scope,
                arguments.memory_max_age_days,
                arguments.memory_max_deletions,
            )
        ):
            parser.error("memory management options require a --memory-* mode")
        return mode

    mode_option = {
        "resume": "--resume",
        "continue": "--continue",
        "respond": "--respond",
        "usage_report": "--usage-report",
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
            arguments.max_attempt_cost_usd if mode != "usage_report" else None,
            arguments.max_tool_calls_per_step,
            arguments.max_explorer_calls_per_step,
            arguments.max_no_progress_rounds,
            arguments.max_plan_revisions,
            arguments.context_budget_tokens,
            arguments.max_repository_memory_results,
            arguments.max_episodic_memory_results,
            arguments.memory_management_limit,
            arguments.memory_scope,
            arguments.memory_max_age_days,
            arguments.memory_max_deletions,
        )
    ):
        parser.error("budget options can only be supplied for a new investigation")

    if mode == "usage_report":
        for option, value in (
            ("--prompt", arguments.prompt),
            ("--interrupt-id", arguments.interrupt_id),
            ("--decision", arguments.decision),
            ("--feedback", arguments.feedback),
        ):
            if value is not None:
                parser.error(f"{option} cannot be supplied with --usage-report")
        if arguments.pricing_file is None:
            parser.error("--pricing-file is required with --usage-report")
        return mode

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


def _render_usage_cost_summary(report: AttemptUsageCostReport) -> str:
    """Render one concise summary strictly from the report aggregate."""

    total = report.breakdowns[0]
    token_value = (
        "unavailable"
        if total.total_tokens.value is None
        else str(total.total_tokens.value)
    )
    cost_value = total.cost.estimated_cost_usd or "unavailable"
    budget = ""
    if report.budget is not None:
        budget = (
            f"; budget {report.budget.status.value} "
            f"({report.budget.reason.value}, limit USD {report.budget.limit_usd})"
        )
    return (
        f"Usage/cost: {total.invocation_count} invocations; total tokens "
        f"{token_value} ({total.total_tokens.coverage.value}); estimated cost USD "
        f"{cost_value} ({total.cost.coverage.value}); pricing "
        f"{report.pricing_version}@{report.pricing_effective_from} "
        f"[{report.pricing_table_id[:12]}]{budget}; estimate, not provider invoice"
    )


def _load_optional_pricing_table(path: Path | None) -> PricingTable | None:
    return None if path is None else load_pricing_table_file(path)


def _utc_now_ms() -> int:
    return time.time_ns() // 1_000_000


def _run_memory_management(
    arguments: argparse.Namespace,
    mode: LifecycleMode,
) -> int:
    """Run payload-free memory administration without constructing any model."""

    context = ToolExecutionContext(arguments.repo_path)
    with open_memory_store(arguments.memory_db) as store:
        if mode == "memory_list":
            records = store.list_memory_metadata(
                context=context,
                max_results=arguments.memory_management_limit or 20,
                memory_scope=arguments.memory_scope,
            )
            rendered = json.dumps(
                [record.model_dump(mode="json") for record in records],
                ensure_ascii=False,
                indent=2,
            )
            summary = f"Memory entries: {len(records)}"
        elif mode == "memory_list_deleted":
            tombstones = store.list_memory_tombstones(
                context=context,
                max_results=arguments.memory_management_limit or 20,
            )
            rendered = json.dumps(
                [item.model_dump(mode="json") for item in tombstones],
                ensure_ascii=False,
                indent=2,
            )
            summary = f"Memory tombstones: {len(tombstones)}"
        elif mode == "memory_delete":
            tombstone = store.delete_memory(
                arguments.memory_delete,
                context=context,
                deleted_at_ms=_utc_now_ms(),
            )
            rendered = json.dumps(
                {
                    "status": "deleted" if tombstone is not None else "not_found",
                    "tombstone": (
                        None if tombstone is None else tombstone.model_dump(mode="json")
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            summary = (
                "Memory deletion: tombstoned"
                if tombstone is not None
                else "Memory deletion: not found in repository scope"
            )
        else:
            max_age_ms = (
                None
                if arguments.memory_max_age_days is None
                else arguments.memory_max_age_days * 86_400_000
            )
            policy = create_memory_retention_policy(
                max_episodic_entries=arguments.memory_retain,
                max_age_ms=max_age_ms,
                max_deletions_per_run=arguments.memory_max_deletions or 100,
            )
            report = store.apply_retention_policy(
                policy,
                context=context,
                evaluated_at_ms=_utc_now_ms(),
            )
            rendered = report.model_dump_json(indent=2)
            summary = (
                f"Memory retention: {len(report.deleted_memory_ids)} deleted; "
                f"{report.candidate_count} eligible; truncated={report.truncated}"
            )
    print(rendered)
    print(summary, file=sys.stderr)
    print(f"Memory store: {arguments.memory_db.resolve()}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one investigation from command-line arguments."""

    parser = create_parser()
    arguments = parser.parse_args(argv)
    lifecycle_mode = _validate_lifecycle_arguments(parser, arguments)

    try:
        browser_policy = (
            BrowserPolicy(tuple(arguments.browser_page))
            if arguments.browser_page
            else None
        )
        if browser_policy is not None:
            validate_browser_runtime()
        if lifecycle_mode in {
            "memory_list",
            "memory_list_deleted",
            "memory_delete",
            "memory_retain",
        }:
            return _run_memory_management(arguments, lifecycle_mode)
        pricing_table = _load_optional_pricing_table(arguments.pricing_file)
        if lifecycle_mode == "usage_report":
            assert pricing_table is not None
            with open_raw_run_store(arguments.raw_store_db) as raw_run_store:
                report = raw_run_store.get_attempt_usage_cost_report(
                    arguments.usage_report,
                    pricing_table,
                    max_cost_usd=arguments.max_attempt_cost_usd,
                )
            if report is None:
                raise ValueError(f"Raw run does not exist: {arguments.usage_report}")
            print(report.model_dump_json(indent=2))
            print(_render_usage_cost_summary(report), file=sys.stderr)
            print(
                f"Raw store: {arguments.raw_store_db.resolve()}",
                file=sys.stderr,
            )
            return 0

        memory_context = (
            open_memory_store(arguments.memory_db)
            if arguments.memory
            else nullcontext(None)
        )
        with (
            open_sqlite_checkpointer(arguments.checkpoint_db) as checkpointer,
            open_raw_run_store(arguments.raw_store_db) as raw_run_store,
            memory_context as memory_store,
        ):
            optional_runner_dependencies: dict[str, object] = {}
            if browser_policy is not None:
                optional_runner_dependencies["browser_policy"] = browser_policy
            if pricing_table is not None:
                optional_runner_dependencies["pricing_table"] = pricing_table
            if memory_store is not None:
                optional_runner_dependencies["memory_store"] = memory_store
            runner = AgentRunner(
                create_workflow_models(),
                checkpointer=checkpointer,
                raw_run_store=raw_run_store,
                **optional_runner_dependencies,
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
                        max_attempt_cost_usd=arguments.max_attempt_cost_usd,
                        human_review_policy=(
                            arguments.human_review_policy or HumanReviewPolicy.NEVER
                        ),
                        memory_retrieval_enabled=arguments.memory,
                        max_repository_memory_results=(
                            arguments.max_repository_memory_results or 5
                        ),
                        max_episodic_memory_results=(
                            arguments.max_episodic_memory_results or 5
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
        MemoryStoreError,
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
    if outcome.usage_cost_report is not None:
        print(_render_usage_cost_summary(outcome.usage_cost_report), file=sys.stderr)
    print(f"Raw store: {arguments.raw_store_db.resolve()}", file=sys.stderr)
    if arguments.memory:
        print(f"Memory store: {arguments.memory_db.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
