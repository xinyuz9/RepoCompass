"""Production RunOutcome adapter and identity-gated five-variant runner."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from time import monotonic_ns
from typing import Protocol, runtime_checkable
from uuid import uuid4

from repocompass.agent.candidate import CandidateResult
from repocompass.agent.cost import (
    CostCoverage,
    CostUnavailableReason,
    MetricCoverage,
)
from repocompass.agent.evaluation import EvaluationReport
from repocompass.agent.lifecycle import ExecutionStatus, HumanReviewPolicy
from repocompass.agent.runner import RunOutcome, RunRequest
from repocompass.agent.trace import RunTrace, TraceEventKind
from repocompass.evaluation.localization_contracts import (
    MAX_PREDICTED_FILES,
    MAX_PREDICTED_SYMBOLS,
    MAX_REPOSITORY_PATH_CHARACTERS,
    LocalizationBudgetProfile,
    LocalizationCase,
    LocalizationDatasetManifest,
    LocalizationMetricUnavailableReason,
    LocalizationRunObservation,
    LocalizationRunStatus,
    LocalizationSymbol,
    LocalizationTask,
    LocalizationVariant,
    create_estimated_cost_observation,
    create_localization_prediction,
    create_localization_run_observation,
    create_unavailable_cost_observation,
    observed_localization_metric,
    unavailable_localization_metric,
)
from repocompass.evaluation.localization_harness import (
    LocalizationObservationSet,
    create_localization_observation_set,
)
from repocompass.structure.contracts import StructuralVerificationStatus
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    ToolAuditRecord,
)
from repocompass.tools.gateway import ToolExecutionContext

ClockNs = Callable[[], int]
RunIdFactory = Callable[[], str]

_EVIDENCE_LOCATION_ISSUE_CODES = frozenset(
    {
        "path_not_relative",
        "path_outside_repository",
        "path_not_found",
        "path_not_file",
        "line_range_missing",
        "line_range_invalid",
    }
)


@runtime_checkable
class LocalizationVariantExecutor(Protocol):
    """One explicitly labelled architecture implementation under evaluation."""

    variant: LocalizationVariant

    def run(self, request: RunRequest) -> RunOutcome:
        """Execute one label-free production RunRequest."""


def create_localization_run_request(
    *,
    task: LocalizationTask,
    repository_path: Path,
    budget_profile: LocalizationBudgetProfile,
    variant: LocalizationVariant,
) -> RunRequest:
    """Translate one public task and shared budget into a production request."""

    if not isinstance(task, LocalizationTask):
        raise TypeError("task must be a LocalizationTask")
    if not isinstance(repository_path, Path):
        raise TypeError("repository_path must be a Path")
    if not isinstance(budget_profile, LocalizationBudgetProfile):
        raise TypeError("budget_profile must be a LocalizationBudgetProfile")
    if not isinstance(variant, LocalizationVariant):
        raise TypeError("variant must be a LocalizationVariant")
    resolved_path = _validate_checkout(task, repository_path)
    return RunRequest(
        repo_path=resolved_path,
        objective=task.objective,
        max_tool_calls_per_step=budget_profile.max_tool_calls_per_step,
        max_explorer_calls_per_step=budget_profile.max_explorer_calls_per_step,
        max_no_progress_rounds=budget_profile.max_no_progress_rounds,
        max_plan_revisions=budget_profile.max_plan_revisions,
        context_budget_tokens=budget_profile.context_budget_tokens,
        max_attempt_cost_usd=budget_profile.max_attempt_cost_usd,
        human_review_policy=HumanReviewPolicy.NEVER,
        memory_retrieval_enabled=(
            variant is LocalizationVariant.EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY
        ),
        max_repository_memory_results=(budget_profile.max_repository_memory_results),
        max_episodic_memory_results=budget_profile.max_episodic_memory_results,
    )


def adapt_localization_run_outcome(
    *,
    dataset_manifest_id: str,
    case: LocalizationCase,
    variant: LocalizationVariant,
    budget_profile: LocalizationBudgetProfile,
    outcome: RunOutcome,
    repeat_index: int,
    latency_ms: int,
    pricing_table_id: str,
) -> LocalizationRunObservation:
    """Convert one production outcome without inventing unavailable metrics."""

    _validate_adapter_inputs(
        dataset_manifest_id=dataset_manifest_id,
        case=case,
        variant=variant,
        budget_profile=budget_profile,
        outcome=outcome,
        repeat_index=repeat_index,
        latency_ms=latency_ms,
        pricing_table_id=pricing_table_id,
    )
    run_id = outcome.trace.run_id
    _validate_outcome_identity(outcome, pricing_table_id=pricing_table_id)
    status = _outcome_status(outcome)
    prediction = _prediction_from_outcome(outcome)
    context_lines, context_tokens = _context_exposure(outcome.trace)
    evaluator_iterations = sum(
        event.kind is TraceEventKind.NODE_COMPLETED
        and event.node == "evaluate_candidate"
        for event in outcome.trace.events
    )
    report = outcome.usage_cost_report
    return create_localization_run_observation(
        dataset_manifest_id=dataset_manifest_id,
        case_id=case.case_id,
        task_id=case.task.task_id,
        variant=variant,
        budget_profile_id=budget_profile.profile_id,
        run_id=run_id,
        repeat_index=repeat_index,
        status=status,
        prediction=prediction,
        context_lines=observed_localization_metric(context_lines),
        context_tokens=observed_localization_metric(context_tokens),
        tool_calls=observed_localization_metric(len(outcome.tool_audit_records)),
        latency_ms=observed_localization_metric(latency_ms),
        evaluator_iterations=observed_localization_metric(evaluator_iterations),
        input_tokens=_usage_metric(report, "input_tokens"),
        output_tokens=_usage_metric(report, "output_tokens"),
        cost=_cost_observation(report, pricing_table_id=pricing_table_id),
    )


class LocalizationBenchmarkRunner:
    """Run every labelled architecture against every task under one identity."""

    def __init__(
        self,
        *,
        manifest: LocalizationDatasetManifest,
        budget_profile: LocalizationBudgetProfile,
        pricing_table_id: str,
        executors: Mapping[LocalizationVariant, LocalizationVariantExecutor],
        clock_ns: ClockNs = monotonic_ns,
        failed_run_id_factory: RunIdFactory = lambda: f"failed-{uuid4()}",
    ) -> None:
        if not isinstance(manifest, LocalizationDatasetManifest):
            raise TypeError("manifest must be a LocalizationDatasetManifest")
        if not isinstance(budget_profile, LocalizationBudgetProfile):
            raise TypeError("budget_profile must be a LocalizationBudgetProfile")
        _require_sha256(pricing_table_id, label="pricing_table_id")
        if not isinstance(executors, Mapping):
            raise TypeError("executors must be a mapping")
        expected_variants = set(LocalizationVariant)
        if set(executors) != expected_variants:
            raise ValueError("executors must provide exactly five canonical variants")
        canonical_executors = {
            variant: executors[variant] for variant in LocalizationVariant
        }
        if len({id(executor) for executor in canonical_executors.values()}) != len(
            canonical_executors
        ):
            raise ValueError("each architecture variant requires a distinct executor")
        for variant, executor in canonical_executors.items():
            if not isinstance(executor, LocalizationVariantExecutor):
                raise TypeError("executor must implement LocalizationVariantExecutor")
            if executor.variant is not variant:
                raise ValueError(
                    "executor label does not match its variant mapping key"
                )
        if not callable(clock_ns) or not callable(failed_run_id_factory):
            raise TypeError("clock and failed run ID factory must be callable")
        self._manifest = manifest
        self._budget_profile = budget_profile
        self._pricing_table_id = pricing_table_id
        self._executors = canonical_executors
        self._clock_ns = clock_ns
        self._failed_run_id_factory = failed_run_id_factory

    def run(
        self,
        *,
        repository_paths: Mapping[str, Path],
        repeat_index: int = 0,
    ) -> LocalizationObservationSet:
        """Preflight all checkouts, then execute one complete comparison matrix."""

        if not isinstance(repository_paths, Mapping):
            raise TypeError("repository_paths must be a mapping")
        if (
            not isinstance(repeat_index, int)
            or isinstance(repeat_index, bool)
            or repeat_index < 0
        ):
            raise ValueError("repeat_index must be a non-negative integer")
        expected_task_ids = {case.task.task_id for case in self._manifest.cases}
        if set(repository_paths) != expected_task_ids:
            raise ValueError("repository paths must exactly match manifest task IDs")
        checkouts = {
            case.task.task_id: _validate_checkout(
                case.task,
                _require_path(repository_paths[case.task.task_id]),
            )
            for case in self._manifest.cases
        }

        observations: list[LocalizationRunObservation] = []
        seen_run_ids: set[str] = set()
        for variant in LocalizationVariant:
            executor = self._executors[variant]
            for case in self._manifest.cases:
                request = create_localization_run_request(
                    task=case.task,
                    repository_path=checkouts[case.task.task_id],
                    budget_profile=self._budget_profile,
                    variant=variant,
                )
                started_ns = _clock_value(self._clock_ns())
                try:
                    outcome = executor.run(request)
                except TimeoutError:
                    observation = self._failed_observation(
                        case=case,
                        variant=variant,
                        repeat_index=repeat_index,
                        status=LocalizationRunStatus.TIMED_OUT,
                        latency_ms=_elapsed_ms(started_ns, self._clock_ns()),
                    )
                except Exception:
                    observation = self._failed_observation(
                        case=case,
                        variant=variant,
                        repeat_index=repeat_index,
                        status=LocalizationRunStatus.FAILED,
                        latency_ms=_elapsed_ms(started_ns, self._clock_ns()),
                    )
                else:
                    if not isinstance(outcome, RunOutcome):
                        raise TypeError("variant executor did not return RunOutcome")
                    observation = adapt_localization_run_outcome(
                        dataset_manifest_id=self._manifest.manifest_id,
                        case=case,
                        variant=variant,
                        budget_profile=self._budget_profile,
                        outcome=outcome,
                        repeat_index=repeat_index,
                        latency_ms=_elapsed_ms(started_ns, self._clock_ns()),
                        pricing_table_id=self._pricing_table_id,
                    )
                _validate_checkout(case.task, checkouts[case.task.task_id])
                if observation.run_id in seen_run_ids:
                    raise ValueError("benchmark run IDs must be unique")
                seen_run_ids.add(observation.run_id)
                observations.append(observation)

        return create_localization_observation_set(
            dataset_manifest_id=self._manifest.manifest_id,
            budget_profile=self._budget_profile,
            observations=tuple(observations),
        )

    def _failed_observation(
        self,
        *,
        case: LocalizationCase,
        variant: LocalizationVariant,
        repeat_index: int,
        status: LocalizationRunStatus,
        latency_ms: int,
    ) -> LocalizationRunObservation:
        run_id = self._failed_run_id_factory()
        _require_identifier(run_id, label="failed run ID")
        unavailable = unavailable_localization_metric(
            LocalizationMetricUnavailableReason.RUN_FAILED
        )
        return create_localization_run_observation(
            dataset_manifest_id=self._manifest.manifest_id,
            case_id=case.case_id,
            task_id=case.task.task_id,
            variant=variant,
            budget_profile_id=self._budget_profile.profile_id,
            run_id=run_id,
            repeat_index=repeat_index,
            status=status,
            prediction=create_localization_prediction(
                ranked_files=(),
                ranked_symbols=(),
                quality_unavailable_reason=(
                    LocalizationMetricUnavailableReason.RUN_FAILED
                ),
            ),
            context_lines=unavailable,
            context_tokens=unavailable,
            tool_calls=unavailable,
            latency_ms=observed_localization_metric(latency_ms),
            evaluator_iterations=unavailable,
            input_tokens=unavailable,
            output_tokens=unavailable,
            cost=create_unavailable_cost_observation(
                pricing_table_id=self._pricing_table_id,
                reason=CostUnavailableReason.USAGE_UNAVAILABLE,
            ),
        )


def _validate_checkout(task: LocalizationTask, repository_path: Path) -> Path:
    context = ToolExecutionContext(repository_path)
    if context.commit_sha != task.base_commit:
        raise ValueError(
            "repository checkout does not match localization task base commit"
        )
    probe = context.bind_call(
        call_id="localization-benchmark-checkout",
        tool_name="list_tree",
        arguments={"max_depth": 0, "max_entries": 1},
    )
    if not context.allows_result_reuse(probe):
        raise ValueError(
            "localization benchmark requires a clean immutable Git snapshot"
        )
    return context.repository_root


def _validate_adapter_inputs(
    *,
    dataset_manifest_id: str,
    case: LocalizationCase,
    variant: LocalizationVariant,
    budget_profile: LocalizationBudgetProfile,
    outcome: RunOutcome,
    repeat_index: int,
    latency_ms: int,
    pricing_table_id: str,
) -> None:
    _require_sha256(dataset_manifest_id, label="dataset_manifest_id")
    _require_sha256(pricing_table_id, label="pricing_table_id")
    if not isinstance(case, LocalizationCase):
        raise TypeError("case must be a LocalizationCase")
    if not isinstance(variant, LocalizationVariant):
        raise TypeError("variant must be a LocalizationVariant")
    if not isinstance(budget_profile, LocalizationBudgetProfile):
        raise TypeError("budget_profile must be a LocalizationBudgetProfile")
    if not isinstance(outcome, RunOutcome):
        raise TypeError("outcome must be a RunOutcome")
    for name, value in (("repeat_index", repeat_index), ("latency_ms", latency_ms)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _validate_outcome_identity(outcome: RunOutcome, *, pricing_table_id: str) -> None:
    run_id = outcome.trace.run_id
    _require_identifier(run_id, label="trace run ID")
    if any(event.run_id != run_id for event in outcome.trace.events):
        raise ValueError("trace event run identity is inconsistent")
    if any(
        not isinstance(record, ToolAuditRecord) or record.run_id != run_id
        for record in outcome.tool_audit_records
    ):
        raise ValueError("tool audit identity does not match trace run")
    report = outcome.usage_cost_report
    if report is not None and (
        report.run_id != run_id
        or report.thread_id != outcome.thread_id
        or report.pricing_table_id != pricing_table_id
    ):
        raise ValueError("usage/cost identity does not match benchmark run")


def _outcome_status(outcome: RunOutcome) -> LocalizationRunStatus:
    if outcome.execution_status is ExecutionStatus.INTERRUPTED:
        return LocalizationRunStatus.INTERRUPTED
    mapping = {
        "accepted": LocalizationRunStatus.ACCEPTED,
        "exhausted": LocalizationRunStatus.EXHAUSTED,
        "cancelled": LocalizationRunStatus.CANCELLED,
    }
    assert outcome.termination_status is not None
    return mapping[outcome.termination_status]


def _prediction_from_outcome(outcome: RunOutcome):
    result = outcome.final_result
    if result is None:
        return create_localization_prediction(
            ranked_files=(),
            ranked_symbols=(),
            quality_unavailable_reason=(
                LocalizationMetricUnavailableReason.NOT_APPLICABLE
            ),
        )
    ranked_files = _canonical_ranked_files(result.relevant_files)
    ranked_symbols = _ranked_symbols(result)
    candidate = _latest_model(
        outcome.trace, "build_candidate", "candidate_result", CandidateResult
    )
    evaluation = _latest_model(
        outcome.trace,
        "evaluate_candidate",
        "evaluation_report",
        EvaluationReport,
    )
    if candidate is None or evaluation is None:
        return create_localization_prediction(
            ranked_files=ranked_files,
            ranked_symbols=ranked_symbols,
            quality_unavailable_reason=(
                LocalizationMetricUnavailableReason.NOT_COLLECTED
            ),
        )
    evidence_count = len(candidate.evidence)
    invalid_evidence = {
        issue.evidence_index
        for issue in evaluation.deterministic_issues
        if issue.code in _EVIDENCE_LOCATION_ISSUE_CODES
        and issue.evidence_index is not None
        and issue.evidence_index < evidence_count
    }
    relation_count = len(candidate.relations)
    verified_relations = 0
    structure = result.structure
    if structure is not None:
        edges = {edge.assertion_id: edge for edge in structure.graph.edges}
        verified_relations = sum(
            edges[assertion_id].verification_status
            is StructuralVerificationStatus.VERIFIED
            for assertion_id in structure.candidate_relation_assertion_ids
        )
    return create_localization_prediction(
        ranked_files=ranked_files,
        ranked_symbols=ranked_symbols,
        evidence_count=evidence_count,
        valid_evidence_count=evidence_count - len(invalid_evidence),
        relation_count=relation_count,
        unsupported_relation_count=max(0, relation_count - verified_relations),
    )


def _canonical_ranked_files(values: list[str]) -> tuple[str, ...]:
    ranked: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        path = PurePosixPath(normalized)
        if (
            not normalized
            or len(normalized) > MAX_REPOSITORY_PATH_CHARACTERS
            or "\x00" in normalized
            or any(character in normalized for character in ("\r", "\n"))
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and path.parts[0].endswith(":"))
        ):
            continue
        canonical = path.as_posix()
        if canonical not in seen:
            seen.add(canonical)
            ranked.append(canonical)
        if len(ranked) == MAX_PREDICTED_FILES:
            break
    return tuple(ranked)


def _ranked_symbols(result) -> tuple[LocalizationSymbol, ...]:
    if result.structure is None:
        return ()
    bindings_by_name = {
        binding.canonical_name: binding for binding in result.structure.relevant_symbols
    }
    names = [name for name in result.relevant_symbols if name in bindings_by_name]
    if not names:
        names = [
            binding.canonical_name for binding in result.structure.relevant_symbols
        ]
    ranked: list[LocalizationSymbol] = []
    seen: set[tuple[str, str]] = set()
    for name in names:
        binding = bindings_by_name[name]
        identity = (binding.path, binding.canonical_name)
        if identity not in seen:
            seen.add(identity)
            ranked.append(
                LocalizationSymbol(
                    path=binding.path,
                    qualified_name=binding.canonical_name,
                )
            )
        if len(ranked) == MAX_PREDICTED_SYMBOLS:
            break
    return tuple(ranked)


def _latest_model(
    trace: RunTrace,
    node: str,
    key: str,
    expected_type: type,
):
    for event in reversed(trace.events):
        if event.kind is not TraceEventKind.NODE_COMPLETED or event.node != node:
            continue
        payload = event.payload.get(key)
        if payload is None:
            continue
        try:
            return expected_type.model_validate_json(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"trace {node} payload is invalid") from error
    return None


def _context_exposure(trace: RunTrace) -> tuple[int, int]:
    total_lines = 0
    total_tokens = 0
    for event in trace.events:
        if event.kind is not TraceEventKind.NODE_COMPLETED:
            continue
        pack = event.payload.get("active_context_pack")
        if pack is None:
            continue
        if not isinstance(pack, dict):
            raise ValueError("trace active context pack is invalid")
        estimated_tokens = pack.get("estimated_tokens")
        items = pack.get("items")
        if (
            not isinstance(estimated_tokens, int)
            or isinstance(estimated_tokens, bool)
            or estimated_tokens < 0
            or not isinstance(items, list)
        ):
            raise ValueError("trace active context pack metrics are invalid")
        total_tokens += estimated_tokens
        total_lines += _merged_context_lines(items)
    return total_lines, total_tokens


def _merged_context_lines(items: list[object]) -> int:
    intervals: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("trace context item is invalid")
        location = item.get("location")
        if location is None:
            continue
        if not isinstance(location, dict):
            raise ValueError("trace context location is invalid")
        path = location.get("path")
        line_start = location.get("line_start")
        line_end = location.get("line_end")
        if (
            not isinstance(path, str)
            or not isinstance(line_start, int)
            or isinstance(line_start, bool)
            or not isinstance(line_end, int)
            or isinstance(line_end, bool)
            or line_start < 1
            or line_end < line_start
        ):
            raise ValueError("trace context line range is invalid")
        intervals.setdefault(path, []).append((line_start, line_end))
    line_count = 0
    for path_intervals in intervals.values():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(path_intervals):
            if merged and start <= merged[-1][1] + 1:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        line_count += sum(end - start + 1 for start, end in merged)
    return line_count


def _usage_metric(report, attribute: str):
    if report is None:
        return unavailable_localization_metric(
            LocalizationMetricUnavailableReason.PROVIDER_NOT_REPORTED
        )
    metric = getattr(report.breakdowns[0], attribute)
    if metric.coverage is MetricCoverage.COMPLETE and metric.value is not None:
        return observed_localization_metric(metric.value)
    return unavailable_localization_metric(
        LocalizationMetricUnavailableReason.PROVIDER_NOT_REPORTED
    )


def _cost_observation(report, *, pricing_table_id: str):
    if report is None:
        return create_unavailable_cost_observation(
            pricing_table_id=pricing_table_id,
            reason=CostUnavailableReason.USAGE_UNAVAILABLE,
        )
    summary = report.breakdowns[0].cost
    if (
        summary.coverage is CostCoverage.COMPLETE
        and summary.estimated_cost_usd is not None
    ):
        return create_estimated_cost_observation(
            pricing_table_id=pricing_table_id,
            estimated_cost_usd=summary.estimated_cost_usd,
            usage_cost_report_id=report.report_id,
        )
    reason = (
        summary.unavailable_reasons[0]
        if summary.unavailable_reasons
        else CostUnavailableReason.USAGE_UNAVAILABLE
    )
    return create_unavailable_cost_observation(
        pricing_table_id=pricing_table_id,
        reason=reason,
    )


def _elapsed_ms(started_ns: int, finished_ns: object) -> int:
    finished = _clock_value(finished_ns)
    if finished < started_ns:
        raise ValueError("benchmark monotonic clock moved backwards")
    elapsed_ns = finished - started_ns
    return (elapsed_ns + 999_999) // 1_000_000


def _clock_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("benchmark clock must return a non-negative integer")
    return value


def _require_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("repository path values must be Path objects")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(SHA256_PATTERN, value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"{label} is invalid")
    return value
