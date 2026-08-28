"""Strict contracts for SWE-bench-derived localization evaluation.

The agent-visible task and evaluator-only labels are deliberately separate.
Gold patch bodies are never represented by these contracts; only an immutable
source digest and derived file/symbol labels cross the dataset boundary.
"""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from repocompass.agent.cost import (
    COST_DISCLAIMER,
    CostAvailability,
    CostUnavailableReason,
)
from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    CommitSha,
    Sha256Digest,
)

LOCALIZATION_SCHEMA_VERSION = 1
LOCALIZATION_SCOPE_DISCLAIMER = "swe_bench_derived_localization_not_resolved_rate"
MAX_LOCALIZATION_CASES = 1_000
MAX_GOLD_FILES = 128
MAX_GOLD_SYMBOLS = 512
MAX_PREDICTED_FILES = 256
MAX_PREDICTED_SYMBOLS = 1_024
MAX_REPOSITORY_PATH_CHARACTERS = 4_096
MAX_OBJECTIVE_CHARACTERS = 100_000
_REPOSITORY_PATTERN = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
_INSTANCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"
_CANONICAL_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")


class LocalizationDatasetKind(StrEnum):
    """Closed provenance categories supported by the localization harness."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    SWE_BENCH_LITE_DERIVED = "swe_bench_lite_derived"
    SWE_BENCH_VERIFIED_DERIVED = "swe_bench_verified_derived"


class LocalizationLabelSource(StrEnum):
    """Offline authority from which file and symbol labels were derived."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    SWE_BENCH_GOLD_PATCH = "swe_bench_gold_patch"


class LocalizationVariant(StrEnum):
    """The five canonical architecture variants compared by M6."""

    TOOL_WRAPPER = "tool_wrapper"
    EVALUATOR_LOOP = "evaluator_loop"
    EVALUATOR_LOOP_SYMBOL_TOOLS = "evaluator_loop_symbol_tools"
    EVALUATOR_LOOP_CONTEXT_BUILDER = "evaluator_loop_context_builder"
    EVALUATOR_LOOP_CONTEXT_BUILDER_MEMORY = "evaluator_loop_context_builder_memory"


class LocalizationRunStatus(StrEnum):
    """Closed terminal or harness-level status for one observed run."""

    ACCEPTED = "accepted"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class LocalizationMetricAvailability(StrEnum):
    """Whether a non-negative integer metric was actually observed."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


class LocalizationMetricUnavailableReason(StrEnum):
    """Closed reasons that a run metric cannot truthfully be reported."""

    NOT_COLLECTED = "not_collected"
    NOT_APPLICABLE = "not_applicable"
    RUN_FAILED = "run_failed"
    PROVIDER_NOT_REPORTED = "provider_not_reported"


class _LocalizationModel(CheckpointSafeModel):
    """Immutable strict base shared by localization evaluation artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LocalizationSymbol(_LocalizationModel):
    """One repository-relative symbol label or ranked prediction."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    path: str = Field(min_length=1, max_length=MAX_REPOSITORY_PATH_CHARACTERS)
    qualified_name: str = Field(min_length=1, max_length=2_048)

    @field_validator("path")
    @classmethod
    def require_normalized_path(cls, value: str) -> str:
        _validate_repository_path(value)
        return value

    @field_validator("qualified_name")
    @classmethod
    def require_single_line_name(cls, value: str) -> str:
        _validate_single_line(value, label="qualified_name")
        return value


class LocalizationTask(_LocalizationModel):
    """Agent-visible issue input without evaluator labels or patch metadata."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    task_id: Sha256Digest
    dataset_kind: LocalizationDatasetKind
    instance_id: str = Field(pattern=_INSTANCE_PATTERN)
    repository: str = Field(pattern=_REPOSITORY_PATTERN)
    base_commit: CommitSha
    objective: str = Field(min_length=1, max_length=MAX_OBJECTIVE_CHARACTERS)
    evaluation_scope: Literal["localization_only"] = "localization_only"

    @field_validator("objective")
    @classmethod
    def reject_unsafe_objective_controls(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("objective must not contain NUL")
        return value

    @model_validator(mode="after")
    def require_content_addressed_task(self) -> "LocalizationTask":
        expected = _task_id(
            dataset_kind=self.dataset_kind,
            instance_id=self.instance_id,
            repository=self.repository,
            base_commit=self.base_commit,
            objective=self.objective,
        )
        if self.task_id != expected:
            raise ValueError("task_id does not match localization task contents")
        return self


class LocalizationLabels(_LocalizationModel):
    """Evaluator-only locations derived offline from a trusted source artifact."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    label_id: Sha256Digest
    task_id: Sha256Digest
    source: LocalizationLabelSource
    source_artifact_digest: Sha256Digest
    gold_files: tuple[str, ...] = Field(min_length=1, max_length=MAX_GOLD_FILES)
    gold_symbols: tuple[LocalizationSymbol, ...] = Field(
        default=(),
        max_length=MAX_GOLD_SYMBOLS,
    )
    evaluation_scope_disclaimer: Literal[
        "swe_bench_derived_localization_not_resolved_rate"
    ] = LOCALIZATION_SCOPE_DISCLAIMER

    @field_validator("gold_files")
    @classmethod
    def require_canonical_gold_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_repository_path(path)
        if value != tuple(sorted(set(value))):
            raise ValueError("gold files must be unique and in canonical sorted order")
        return value

    @field_validator("gold_symbols")
    @classmethod
    def require_canonical_gold_symbols(
        cls, value: tuple[LocalizationSymbol, ...]
    ) -> tuple[LocalizationSymbol, ...]:
        identities = tuple((item.path, item.qualified_name) for item in value)
        if identities != tuple(sorted(set(identities))):
            raise ValueError(
                "gold symbols must be unique and in canonical sorted order"
            )
        return value

    @model_validator(mode="after")
    def require_closed_content_addressed_labels(self) -> "LocalizationLabels":
        gold_file_set = set(self.gold_files)
        if any(item.path not in gold_file_set for item in self.gold_symbols):
            raise ValueError("every gold symbol path must reference a gold file")
        expected = _label_id(
            task_id=self.task_id,
            source=self.source,
            source_artifact_digest=self.source_artifact_digest,
            gold_files=self.gold_files,
            gold_symbols=self.gold_symbols,
        )
        if self.label_id != expected:
            raise ValueError("label_id does not match localization labels")
        return self


class LocalizationCase(_LocalizationModel):
    """Offline binding between one public task and its hidden gold labels."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    case_id: Sha256Digest
    task: LocalizationTask
    labels: LocalizationLabels

    @model_validator(mode="after")
    def require_closed_case_binding(self) -> "LocalizationCase":
        if self.labels.task_id != self.task.task_id:
            raise ValueError("case labels must be bound to the exact task_id")
        expected_source = (
            LocalizationLabelSource.SYNTHETIC_FIXTURE
            if self.task.dataset_kind is LocalizationDatasetKind.SYNTHETIC_FIXTURE
            else LocalizationLabelSource.SWE_BENCH_GOLD_PATCH
        )
        if self.labels.source is not expected_source:
            raise ValueError("label source does not match dataset provenance")
        expected = _case_id(self.task.task_id, self.labels.label_id)
        if self.case_id != expected:
            raise ValueError("case_id does not match task and label identities")
        return self


class LocalizationDatasetManifest(_LocalizationModel):
    """Bounded immutable dataset snapshot consumed by the offline harness."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    manifest_id: Sha256Digest
    name: str = Field(pattern=IDENTIFIER_PATTERN)
    version: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: LocalizationDatasetKind
    cases: tuple[LocalizationCase, ...] = Field(
        min_length=1,
        max_length=MAX_LOCALIZATION_CASES,
    )
    evaluation_scope: Literal["localization_only"] = "localization_only"
    disclaimer: Literal["swe_bench_derived_localization_not_resolved_rate"] = (
        LOCALIZATION_SCOPE_DISCLAIMER
    )

    @model_validator(mode="after")
    def require_canonical_content_addressed_manifest(
        self,
    ) -> "LocalizationDatasetManifest":
        if any(case.task.dataset_kind is not self.kind for case in self.cases):
            raise ValueError("every case kind must match the dataset kind")
        ordering = tuple((case.task.instance_id, case.case_id) for case in self.cases)
        if ordering != tuple(sorted(ordering)):
            raise ValueError("dataset cases must use canonical instance order")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("dataset case IDs must be unique")
        if len({case.task.task_id for case in self.cases}) != len(self.cases):
            raise ValueError("dataset task IDs must be unique")
        if len({case.task.instance_id for case in self.cases}) != len(self.cases):
            raise ValueError("dataset instance IDs must be unique")
        expected = _manifest_id(
            name=self.name,
            version=self.version,
            kind=self.kind,
            cases=self.cases,
        )
        if self.manifest_id != expected:
            raise ValueError("manifest_id does not match dataset contents")
        return self

    def agent_tasks(self) -> tuple[LocalizationTask, ...]:
        """Return only agent-visible tasks; labels remain outside this boundary."""

        return tuple(case.task for case in self.cases)


class LocalizationBudgetProfile(_LocalizationModel):
    """Content-addressed production RunRequest budget used by one comparison."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    profile_id: Sha256Digest
    max_tool_calls_per_step: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    max_explorer_calls_per_step: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    max_no_progress_rounds: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    max_plan_revisions: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    context_budget_tokens: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    max_attempt_cost_usd: str | None = Field(default=None, max_length=128)
    max_repository_memory_results: int = Field(ge=1, le=10)
    max_episodic_memory_results: int = Field(ge=1, le=10)
    human_review_enabled: Literal[False] = False

    @field_validator("max_attempt_cost_usd", mode="before")
    @classmethod
    def require_positive_canonical_cost_cap(cls, value: object) -> object:
        if value is not None:
            _validate_canonical_decimal(value, label="cost cap", positive=True)
        return value

    @model_validator(mode="after")
    def require_content_addressed_profile(self) -> "LocalizationBudgetProfile":
        expected = _budget_profile_id(
            max_tool_calls_per_step=self.max_tool_calls_per_step,
            max_explorer_calls_per_step=self.max_explorer_calls_per_step,
            max_no_progress_rounds=self.max_no_progress_rounds,
            max_plan_revisions=self.max_plan_revisions,
            context_budget_tokens=self.context_budget_tokens,
            max_attempt_cost_usd=self.max_attempt_cost_usd,
            max_repository_memory_results=self.max_repository_memory_results,
            max_episodic_memory_results=self.max_episodic_memory_results,
        )
        if self.profile_id != expected:
            raise ValueError("profile_id does not match localization budget")
        return self


class LocalizationMetricObservation(_LocalizationModel):
    """One observed non-negative count, or an explicit unknown value."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    availability: LocalizationMetricAvailability
    value: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    unavailable_reason: LocalizationMetricUnavailableReason | None = None

    @model_validator(mode="after")
    def reconcile_availability(self) -> "LocalizationMetricObservation":
        if self.availability is LocalizationMetricAvailability.OBSERVED:
            if self.value is None or self.unavailable_reason is not None:
                raise ValueError("observed metric requires a value and no reason")
        elif self.value is not None or self.unavailable_reason is None:
            raise ValueError("unavailable metric requires a reason and no value")
        return self


class LocalizationCostObservation(_LocalizationModel):
    """Run-level estimated cost with explicit pricing and coverage semantics."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    availability: CostAvailability
    pricing_table_id: Sha256Digest | None = None
    usage_cost_report_id: Sha256Digest | None = None
    estimated_cost_usd: str | None = Field(default=None, max_length=128)
    unavailable_reason: CostUnavailableReason | None = None
    disclaimer: Literal["estimate_not_invoice"] = COST_DISCLAIMER

    @field_validator("estimated_cost_usd", mode="before")
    @classmethod
    def require_canonical_cost(cls, value: object) -> object:
        if value is not None:
            _validate_canonical_decimal(value, label="estimated cost", positive=False)
        return value

    @model_validator(mode="after")
    def reconcile_cost_availability(self) -> "LocalizationCostObservation":
        if self.availability is CostAvailability.ESTIMATED:
            if (
                self.pricing_table_id is None
                or self.estimated_cost_usd is None
                or self.unavailable_reason is not None
            ):
                raise ValueError(
                    "estimated cost requires pricing, amount, and no unavailable reason"
                )
        elif (
            self.estimated_cost_usd is not None
            or self.usage_cost_report_id is not None
            or self.unavailable_reason is None
        ):
            raise ValueError(
                "unavailable cost requires a reason and no amount or report identity"
            )
        return self


class LocalizationPrediction(_LocalizationModel):
    """Ranked locations and payload-free quality counts from one run."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    prediction_id: Sha256Digest
    ranked_files: tuple[str, ...] = Field(max_length=MAX_PREDICTED_FILES)
    ranked_symbols: tuple[LocalizationSymbol, ...] = Field(
        max_length=MAX_PREDICTED_SYMBOLS
    )
    quality_availability: LocalizationMetricAvailability
    evidence_count: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    valid_evidence_count: int | None = Field(
        default=None, ge=0, le=MAX_SAFE_JSON_INTEGER
    )
    relation_count: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    unsupported_relation_count: int | None = Field(
        default=None, ge=0, le=MAX_SAFE_JSON_INTEGER
    )
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None = None

    @field_validator("ranked_files")
    @classmethod
    def require_ranked_unique_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_repository_path(path)
        if len(value) != len(set(value)):
            raise ValueError("ranked files must be unique")
        return value

    @field_validator("ranked_symbols")
    @classmethod
    def require_ranked_unique_symbols(
        cls, value: tuple[LocalizationSymbol, ...]
    ) -> tuple[LocalizationSymbol, ...]:
        identities = tuple((item.path, item.qualified_name) for item in value)
        if len(identities) != len(set(identities)):
            raise ValueError("ranked symbols must be unique")
        return value

    @model_validator(mode="after")
    def require_content_addressed_prediction(self) -> "LocalizationPrediction":
        counts = (
            self.evidence_count,
            self.valid_evidence_count,
            self.relation_count,
            self.unsupported_relation_count,
        )
        if self.quality_availability is LocalizationMetricAvailability.OBSERVED:
            if any(value is None for value in counts):
                raise ValueError("observed prediction quality requires every count")
            if self.quality_unavailable_reason is not None:
                raise ValueError("observed prediction quality must not have a reason")
            assert self.evidence_count is not None
            assert self.valid_evidence_count is not None
            assert self.relation_count is not None
            assert self.unsupported_relation_count is not None
            if self.valid_evidence_count > self.evidence_count:
                raise ValueError("valid evidence count cannot exceed evidence count")
            if self.unsupported_relation_count > self.relation_count:
                raise ValueError(
                    "unsupported relation count cannot exceed relation count"
                )
        elif any(value is not None for value in counts) or (
            self.quality_unavailable_reason is None
        ):
            raise ValueError(
                "unavailable prediction quality requires a reason and no counts"
            )
        expected = _prediction_id(
            ranked_files=self.ranked_files,
            ranked_symbols=self.ranked_symbols,
            quality_availability=self.quality_availability,
            evidence_count=self.evidence_count,
            valid_evidence_count=self.valid_evidence_count,
            relation_count=self.relation_count,
            unsupported_relation_count=self.unsupported_relation_count,
            quality_unavailable_reason=self.quality_unavailable_reason,
        )
        if self.prediction_id != expected:
            raise ValueError("prediction_id does not match localization prediction")
        return self


class LocalizationRunObservation(_LocalizationModel):
    """Content-addressed raw observation for one case, variant, and repeat."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    observation_id: Sha256Digest
    dataset_manifest_id: Sha256Digest
    case_id: Sha256Digest
    task_id: Sha256Digest
    variant: LocalizationVariant
    budget_profile_id: Sha256Digest
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    repeat_index: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    status: LocalizationRunStatus
    prediction: LocalizationPrediction
    context_lines: LocalizationMetricObservation
    context_tokens: LocalizationMetricObservation
    tool_calls: LocalizationMetricObservation
    latency_ms: LocalizationMetricObservation
    evaluator_iterations: LocalizationMetricObservation
    input_tokens: LocalizationMetricObservation
    output_tokens: LocalizationMetricObservation
    cost: LocalizationCostObservation
    evaluation_scope: Literal["localization_only"] = "localization_only"
    disclaimer: Literal["swe_bench_derived_localization_not_resolved_rate"] = (
        LOCALIZATION_SCOPE_DISCLAIMER
    )

    @model_validator(mode="after")
    def require_content_addressed_observation(self) -> "LocalizationRunObservation":
        expected = _observation_id(
            dataset_manifest_id=self.dataset_manifest_id,
            case_id=self.case_id,
            task_id=self.task_id,
            variant=self.variant,
            budget_profile_id=self.budget_profile_id,
            run_id=self.run_id,
            repeat_index=self.repeat_index,
            status=self.status,
            prediction=self.prediction,
            context_lines=self.context_lines,
            context_tokens=self.context_tokens,
            tool_calls=self.tool_calls,
            latency_ms=self.latency_ms,
            evaluator_iterations=self.evaluator_iterations,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=self.cost,
        )
        if self.observation_id != expected:
            raise ValueError("observation_id does not match run observation")
        return self


def create_localization_task(
    *,
    dataset_kind: LocalizationDatasetKind,
    instance_id: str,
    repository: str,
    base_commit: str,
    objective: str,
) -> LocalizationTask:
    """Create one canonical task that can safely be passed to an Agent."""

    string_values = (instance_id, repository, base_commit, objective)
    if any(not isinstance(value, str) for value in string_values):
        raise TypeError("localization task text fields must be strings")
    canonical_instance_id = instance_id.strip()
    canonical_repository = repository.strip()
    canonical_base_commit = base_commit.strip()
    canonical_objective = objective.strip()
    task_id = _task_id(
        dataset_kind=dataset_kind,
        instance_id=canonical_instance_id,
        repository=canonical_repository,
        base_commit=canonical_base_commit,
        objective=canonical_objective,
    )
    return LocalizationTask(
        task_id=task_id,
        dataset_kind=dataset_kind,
        instance_id=canonical_instance_id,
        repository=canonical_repository,
        base_commit=canonical_base_commit,
        objective=canonical_objective,
    )


def create_localization_labels(
    *,
    task_id: str,
    source: LocalizationLabelSource,
    source_artifact_digest: str,
    gold_files: tuple[str, ...],
    gold_symbols: tuple[LocalizationSymbol, ...],
) -> LocalizationLabels:
    """Create canonical hidden labels without retaining a gold patch body."""

    canonical_files = tuple(sorted(set(gold_files)))
    canonical_symbols = tuple(
        sorted(gold_symbols, key=lambda item: (item.path, item.qualified_name))
    )
    label_id = _label_id(
        task_id=task_id,
        source=source,
        source_artifact_digest=source_artifact_digest,
        gold_files=canonical_files,
        gold_symbols=canonical_symbols,
    )
    return LocalizationLabels(
        label_id=label_id,
        task_id=task_id,
        source=source,
        source_artifact_digest=source_artifact_digest,
        gold_files=canonical_files,
        gold_symbols=canonical_symbols,
    )


def create_localization_case(
    *, task: LocalizationTask, labels: LocalizationLabels
) -> LocalizationCase:
    """Bind one public task to its offline evaluator labels."""

    if not isinstance(task, LocalizationTask):
        raise TypeError("task must be a LocalizationTask")
    if not isinstance(labels, LocalizationLabels):
        raise TypeError("labels must be LocalizationLabels")
    return LocalizationCase(
        case_id=_case_id(task.task_id, labels.label_id),
        task=task,
        labels=labels,
    )


def create_localization_dataset_manifest(
    *,
    name: str,
    version: str,
    kind: LocalizationDatasetKind,
    cases: tuple[LocalizationCase, ...],
) -> LocalizationDatasetManifest:
    """Create a canonical, content-addressed dataset snapshot."""

    if not isinstance(cases, tuple) or not all(
        isinstance(case, LocalizationCase) for case in cases
    ):
        raise TypeError("cases must be a tuple of LocalizationCase values")
    canonical_cases = tuple(
        sorted(cases, key=lambda case: (case.task.instance_id, case.case_id))
    )
    return LocalizationDatasetManifest(
        manifest_id=_manifest_id(
            name=name,
            version=version,
            kind=kind,
            cases=canonical_cases,
        ),
        name=name,
        version=version,
        kind=kind,
        cases=canonical_cases,
    )


def create_localization_budget_profile(
    *,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
    max_no_progress_rounds: int,
    max_plan_revisions: int,
    context_budget_tokens: int,
    max_attempt_cost_usd: str | None,
    max_repository_memory_results: int,
    max_episodic_memory_results: int,
) -> LocalizationBudgetProfile:
    """Create a stable comparison identity for production run limits."""

    profile_id = _budget_profile_id(
        max_tool_calls_per_step=max_tool_calls_per_step,
        max_explorer_calls_per_step=max_explorer_calls_per_step,
        max_no_progress_rounds=max_no_progress_rounds,
        max_plan_revisions=max_plan_revisions,
        context_budget_tokens=context_budget_tokens,
        max_attempt_cost_usd=max_attempt_cost_usd,
        max_repository_memory_results=max_repository_memory_results,
        max_episodic_memory_results=max_episodic_memory_results,
    )
    return LocalizationBudgetProfile(
        profile_id=profile_id,
        max_tool_calls_per_step=max_tool_calls_per_step,
        max_explorer_calls_per_step=max_explorer_calls_per_step,
        max_no_progress_rounds=max_no_progress_rounds,
        max_plan_revisions=max_plan_revisions,
        context_budget_tokens=context_budget_tokens,
        max_attempt_cost_usd=max_attempt_cost_usd,
        max_repository_memory_results=max_repository_memory_results,
        max_episodic_memory_results=max_episodic_memory_results,
    )


def observed_localization_metric(value: int) -> LocalizationMetricObservation:
    """Wrap one actually observed non-negative integer metric."""

    return LocalizationMetricObservation(
        availability=LocalizationMetricAvailability.OBSERVED,
        value=value,
    )


def unavailable_localization_metric(
    reason: LocalizationMetricUnavailableReason,
) -> LocalizationMetricObservation:
    """Represent an unknown metric without silently substituting zero."""

    return LocalizationMetricObservation(
        availability=LocalizationMetricAvailability.UNAVAILABLE,
        unavailable_reason=reason,
    )


def create_estimated_cost_observation(
    *,
    pricing_table_id: str,
    estimated_cost_usd: str,
    usage_cost_report_id: str | None = None,
) -> LocalizationCostObservation:
    """Create a priced observation; the amount remains explicitly an estimate."""

    return LocalizationCostObservation(
        availability=CostAvailability.ESTIMATED,
        pricing_table_id=pricing_table_id,
        usage_cost_report_id=usage_cost_report_id,
        estimated_cost_usd=estimated_cost_usd,
    )


def create_unavailable_cost_observation(
    *,
    reason: CostUnavailableReason,
    pricing_table_id: str | None = None,
) -> LocalizationCostObservation:
    """Create a cost observation whose missing amount remains explicit."""

    return LocalizationCostObservation(
        availability=CostAvailability.UNAVAILABLE,
        pricing_table_id=pricing_table_id,
        unavailable_reason=reason,
    )


def create_localization_prediction(
    *,
    ranked_files: tuple[str, ...],
    ranked_symbols: tuple[LocalizationSymbol, ...],
    evidence_count: int | None = None,
    valid_evidence_count: int | None = None,
    relation_count: int | None = None,
    unsupported_relation_count: int | None = None,
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None = None,
) -> LocalizationPrediction:
    """Create ranked output with either complete or explicitly unknown quality counts."""

    availability = (
        LocalizationMetricAvailability.UNAVAILABLE
        if quality_unavailable_reason is not None
        else LocalizationMetricAvailability.OBSERVED
    )
    prediction_id = _prediction_id(
        ranked_files=ranked_files,
        ranked_symbols=ranked_symbols,
        quality_availability=availability,
        evidence_count=evidence_count,
        valid_evidence_count=valid_evidence_count,
        relation_count=relation_count,
        unsupported_relation_count=unsupported_relation_count,
        quality_unavailable_reason=quality_unavailable_reason,
    )
    return LocalizationPrediction(
        prediction_id=prediction_id,
        ranked_files=ranked_files,
        ranked_symbols=ranked_symbols,
        quality_availability=availability,
        evidence_count=evidence_count,
        valid_evidence_count=valid_evidence_count,
        relation_count=relation_count,
        unsupported_relation_count=unsupported_relation_count,
        quality_unavailable_reason=quality_unavailable_reason,
    )


def create_localization_run_observation(
    *,
    dataset_manifest_id: str,
    case_id: str,
    task_id: str,
    variant: LocalizationVariant,
    budget_profile_id: str,
    run_id: str,
    repeat_index: int,
    status: LocalizationRunStatus,
    prediction: LocalizationPrediction,
    context_lines: LocalizationMetricObservation,
    context_tokens: LocalizationMetricObservation,
    tool_calls: LocalizationMetricObservation,
    latency_ms: LocalizationMetricObservation,
    evaluator_iterations: LocalizationMetricObservation,
    input_tokens: LocalizationMetricObservation,
    output_tokens: LocalizationMetricObservation,
    cost: LocalizationCostObservation,
) -> LocalizationRunObservation:
    """Create one immutable raw run observation before metric evaluation."""

    values = (
        prediction,
        context_lines,
        context_tokens,
        tool_calls,
        latency_ms,
        evaluator_iterations,
        input_tokens,
        output_tokens,
        cost,
    )
    expected_types = (
        LocalizationPrediction,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationMetricObservation,
        LocalizationCostObservation,
    )
    if any(
        not isinstance(value, expected)
        for value, expected in zip(values, expected_types)
    ):
        raise TypeError("run observation inputs must use localization contract types")
    observation_id = _observation_id(
        dataset_manifest_id=dataset_manifest_id,
        case_id=case_id,
        task_id=task_id,
        variant=variant,
        budget_profile_id=budget_profile_id,
        run_id=run_id,
        repeat_index=repeat_index,
        status=status,
        prediction=prediction,
        context_lines=context_lines,
        context_tokens=context_tokens,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        evaluator_iterations=evaluator_iterations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )
    return LocalizationRunObservation(
        observation_id=observation_id,
        dataset_manifest_id=dataset_manifest_id,
        case_id=case_id,
        task_id=task_id,
        variant=variant,
        budget_profile_id=budget_profile_id,
        run_id=run_id,
        repeat_index=repeat_index,
        status=status,
        prediction=prediction,
        context_lines=context_lines,
        context_tokens=context_tokens,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        evaluator_iterations=evaluator_iterations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )


def _task_id(
    *,
    dataset_kind: LocalizationDatasetKind,
    instance_id: str,
    repository: str,
    base_commit: str,
    objective: str,
) -> str:
    return _content_digest(
        "localization_task",
        {
            "dataset_kind": _enum_value(dataset_kind),
            "instance_id": instance_id,
            "repository": repository,
            "base_commit": base_commit,
            "objective": objective,
            "evaluation_scope": "localization_only",
        },
    )


def _label_id(
    *,
    task_id: str,
    source: LocalizationLabelSource,
    source_artifact_digest: str,
    gold_files: tuple[str, ...],
    gold_symbols: tuple[LocalizationSymbol, ...],
) -> str:
    return _content_digest(
        "localization_labels",
        {
            "task_id": task_id,
            "source": _enum_value(source),
            "source_artifact_digest": source_artifact_digest,
            "gold_files": list(gold_files),
            "gold_symbols": [_model_json(item) for item in gold_symbols],
            "evaluation_scope_disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
        },
    )


def _case_id(task_id: str, label_id: str) -> str:
    return _content_digest(
        "localization_case",
        {"task_id": task_id, "label_id": label_id},
    )


def _manifest_id(
    *,
    name: str,
    version: str,
    kind: LocalizationDatasetKind,
    cases: tuple[LocalizationCase, ...],
) -> str:
    return _content_digest(
        "localization_dataset_manifest",
        {
            "name": name,
            "version": version,
            "kind": _enum_value(kind),
            "case_ids": [case.case_id for case in cases],
            "evaluation_scope": "localization_only",
            "disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
        },
    )


def _budget_profile_id(
    *,
    max_tool_calls_per_step: int,
    max_explorer_calls_per_step: int,
    max_no_progress_rounds: int,
    max_plan_revisions: int,
    context_budget_tokens: int,
    max_attempt_cost_usd: str | None,
    max_repository_memory_results: int,
    max_episodic_memory_results: int,
) -> str:
    return _content_digest(
        "localization_budget_profile",
        {
            "max_tool_calls_per_step": max_tool_calls_per_step,
            "max_explorer_calls_per_step": max_explorer_calls_per_step,
            "max_no_progress_rounds": max_no_progress_rounds,
            "max_plan_revisions": max_plan_revisions,
            "context_budget_tokens": context_budget_tokens,
            "max_attempt_cost_usd": max_attempt_cost_usd,
            "max_repository_memory_results": max_repository_memory_results,
            "max_episodic_memory_results": max_episodic_memory_results,
            "human_review_enabled": False,
        },
    )


def _prediction_id(
    *,
    ranked_files: tuple[str, ...],
    ranked_symbols: tuple[LocalizationSymbol, ...],
    quality_availability: LocalizationMetricAvailability,
    evidence_count: int | None,
    valid_evidence_count: int | None,
    relation_count: int | None,
    unsupported_relation_count: int | None,
    quality_unavailable_reason: LocalizationMetricUnavailableReason | None,
) -> str:
    return _content_digest(
        "localization_prediction",
        {
            "ranked_files": list(ranked_files),
            "ranked_symbols": [_model_json(item) for item in ranked_symbols],
            "quality_availability": _enum_value(quality_availability),
            "evidence_count": evidence_count,
            "valid_evidence_count": valid_evidence_count,
            "relation_count": relation_count,
            "unsupported_relation_count": unsupported_relation_count,
            "quality_unavailable_reason": _optional_enum_value(
                quality_unavailable_reason
            ),
        },
    )


def _observation_id(
    *,
    dataset_manifest_id: str,
    case_id: str,
    task_id: str,
    variant: LocalizationVariant,
    budget_profile_id: str,
    run_id: str,
    repeat_index: int,
    status: LocalizationRunStatus,
    prediction: LocalizationPrediction,
    context_lines: LocalizationMetricObservation,
    context_tokens: LocalizationMetricObservation,
    tool_calls: LocalizationMetricObservation,
    latency_ms: LocalizationMetricObservation,
    evaluator_iterations: LocalizationMetricObservation,
    input_tokens: LocalizationMetricObservation,
    output_tokens: LocalizationMetricObservation,
    cost: LocalizationCostObservation,
) -> str:
    return _content_digest(
        "localization_run_observation",
        {
            "dataset_manifest_id": dataset_manifest_id,
            "case_id": case_id,
            "task_id": task_id,
            "variant": _enum_value(variant),
            "budget_profile_id": budget_profile_id,
            "run_id": run_id,
            "repeat_index": repeat_index,
            "status": _enum_value(status),
            "prediction_id": prediction.prediction_id,
            "context_lines": _model_json(context_lines),
            "context_tokens": _model_json(context_tokens),
            "tool_calls": _model_json(tool_calls),
            "latency_ms": _model_json(latency_ms),
            "evaluator_iterations": _model_json(evaluator_iterations),
            "input_tokens": _model_json(input_tokens),
            "output_tokens": _model_json(output_tokens),
            "cost": _model_json(cost),
            "evaluation_scope": "localization_only",
            "disclaimer": LOCALIZATION_SCOPE_DISCLAIMER,
        },
    )


def _content_digest(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {
            "kind": kind,
            "schema_version": LOCALIZATION_SCHEMA_VERSION,
            "payload": payload,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_json(value: _LocalizationModel) -> dict[str, object]:
    return value.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )


def _enum_value(value: StrEnum) -> str:
    if not isinstance(value, StrEnum):
        raise TypeError("localization enum values must use the closed vocabulary")
    return value.value


def _optional_enum_value(value: StrEnum | None) -> str | None:
    return None if value is None else _enum_value(value)


def _validate_repository_path(value: str) -> None:
    message = "path must be a normalized repository-relative POSIX path"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_REPOSITORY_PATH_CHARACTERS
        or "\\" in value
        or "\x00" in value
        or any(character in value for character in ("\r", "\n"))
    ):
        raise ValueError(message)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ValueError(message)


def _validate_single_line(value: str, *, label: str) -> None:
    if "\x00" in value or any(character in value for character in ("\r", "\n")):
        raise ValueError(f"{label} must be single-line text")


def _validate_canonical_decimal(value: object, *, label: str, positive: bool) -> None:
    if (
        not isinstance(value, str)
        or _CANONICAL_DECIMAL_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{label} must be a {qualifier}finite decimal")
