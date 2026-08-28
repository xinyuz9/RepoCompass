"""Portable observation bundles, report persistence, and human summary."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.evaluation.localization_contracts import (
    LOCALIZATION_SCHEMA_VERSION,
    LocalizationBudgetProfile,
    LocalizationMetricAvailability,
    LocalizationRunObservation,
    LocalizationVariant,
)
from repocompass.evaluation.localization_metrics import (
    LocalizationEvaluationReport,
    LocalizationScalarScore,
)
from repocompass.tools.contracts import Sha256Digest

MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES = 16_777_216
MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES = 33_554_432
MAX_LOCALIZATION_OBSERVATIONS = 5_000


class _HarnessModel(CheckpointSafeModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class LocalizationObservationSet(_HarnessModel):
    """Canonical comparable observations plus their complete budget profile."""

    schema_version: Literal[1] = LOCALIZATION_SCHEMA_VERSION
    set_id: Sha256Digest
    dataset_manifest_id: Sha256Digest
    budget_profile: LocalizationBudgetProfile
    pricing_table_id: Sha256Digest | None = None
    observations: tuple[LocalizationRunObservation, ...] = Field(
        min_length=1,
        max_length=MAX_LOCALIZATION_OBSERVATIONS,
    )

    @model_validator(mode="after")
    def require_canonical_content_addressed_set(self) -> "LocalizationObservationSet":
        if any(
            observation.dataset_manifest_id != self.dataset_manifest_id
            for observation in self.observations
        ):
            raise ValueError("every observation must reference the dataset manifest")
        if any(
            observation.budget_profile_id != self.budget_profile.profile_id
            for observation in self.observations
        ):
            raise ValueError("every observation must reference the budget profile")
        if any(
            observation.cost.pricing_table_id != self.pricing_table_id
            for observation in self.observations
        ):
            raise ValueError("every observation must reference the pricing table")
        if self.observations != tuple(
            sorted(self.observations, key=_observation_sort_key)
        ):
            raise ValueError(
                "observations must use canonical variant/case/repeat order"
            )
        identities = tuple(item.observation_id for item in self.observations)
        if len(identities) != len(set(identities)):
            raise ValueError("observation IDs must be unique")
        logical = tuple(
            (item.case_id, item.variant, item.repeat_index)
            for item in self.observations
        )
        if len(logical) != len(set(logical)):
            raise ValueError("each logical observation may occur only once")
        expected = _observation_set_id(
            dataset_manifest_id=self.dataset_manifest_id,
            budget_profile_id=self.budget_profile.profile_id,
            pricing_table_id=self.pricing_table_id,
            observation_ids=identities,
        )
        if self.set_id != expected:
            raise ValueError("set_id does not match localization observations")
        return self


def create_localization_observation_set(
    *,
    dataset_manifest_id: str,
    budget_profile: LocalizationBudgetProfile,
    observations: tuple[LocalizationRunObservation, ...],
) -> LocalizationObservationSet:
    """Create one sorted, identity-gated observation bundle."""

    if not isinstance(budget_profile, LocalizationBudgetProfile):
        raise TypeError("budget_profile must be a LocalizationBudgetProfile")
    if (
        not isinstance(observations, tuple)
        or not observations
        or not all(
            isinstance(item, LocalizationRunObservation) for item in observations
        )
    ):
        raise TypeError("observations must be a nonempty tuple")
    canonical = tuple(sorted(observations, key=_observation_sort_key))
    pricing_ids = {item.cost.pricing_table_id for item in canonical}
    if len(pricing_ids) != 1:
        raise ValueError("observations must share one pricing table")
    pricing_table_id = next(iter(pricing_ids))
    set_id = _observation_set_id(
        dataset_manifest_id=dataset_manifest_id,
        budget_profile_id=budget_profile.profile_id,
        pricing_table_id=pricing_table_id,
        observation_ids=tuple(item.observation_id for item in canonical),
    )
    return LocalizationObservationSet(
        set_id=set_id,
        dataset_manifest_id=dataset_manifest_id,
        budget_profile=budget_profile,
        pricing_table_id=pricing_table_id,
        observations=canonical,
    )


def load_localization_observation_set_file(path: Path) -> LocalizationObservationSet:
    """Load a bounded, duplicate-key-free, content-addressed observation set."""

    if not isinstance(path, Path):
        raise TypeError("localization observation file path must be a Path")
    file_size = path.stat().st_size
    if file_size > MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES:
        raise ValueError(
            "Localization observation file exceeds "
            f"{MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES} bytes: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES:
        raise ValueError(
            "Localization observation file exceeds "
            f"{MAX_LOCALIZATION_OBSERVATION_SET_FILE_BYTES} bytes: {path}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return LocalizationObservationSet.model_validate_json(canonical)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as error:
        raise ValueError(
            f"Invalid localization observation file {path}: {error}"
        ) from error
    except ValueError as error:
        if str(error).startswith("duplicate JSON key"):
            raise
        raise ValueError(
            f"Invalid localization observation file {path}: {error}"
        ) from error


def load_localization_evaluation_report_file(
    path: Path,
) -> LocalizationEvaluationReport:
    """Load a bounded duplicate-key-free content-addressed evaluation report."""

    if not isinstance(path, Path):
        raise TypeError("localization evaluation report path must be a Path")
    if path.stat().st_size > MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES:
        raise ValueError(
            "Localization evaluation report exceeds "
            f"{MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES} bytes: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES:
        raise ValueError(
            "Localization evaluation report exceeds "
            f"{MAX_LOCALIZATION_EVALUATION_REPORT_FILE_BYTES} bytes: {path}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return LocalizationEvaluationReport.model_validate_json(canonical)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as error:
        raise ValueError(
            f"Invalid localization evaluation report {path}: {error}"
        ) from error
    except ValueError as error:
        if str(error).startswith("duplicate JSON key"):
            raise
        raise ValueError(
            f"Invalid localization evaluation report {path}: {error}"
        ) from error


class LocalizationEvaluationReportSink:
    """Persist one immutable report without overwriting prior evidence."""

    def write(self, report: LocalizationEvaluationReport, destination: Path) -> Path:
        if not isinstance(report, LocalizationEvaluationReport):
            raise TypeError("report must be a LocalizationEvaluationReport")
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        resolved = destination.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2, exclude_computed_fields=True))
            stream.write("\n")
        return resolved


def render_localization_summary(report: LocalizationEvaluationReport) -> str:
    """Render a compact deterministic Markdown comparison table."""

    if not isinstance(report, LocalizationEvaluationReport):
        raise TypeError("report must be a LocalizationEvaluationReport")
    lines = [
        "# RepoCompass localization evaluation",
        "",
        (
            f"Report `{report.report_id}` covers {report.case_count} dataset cases "
            f"and {report.observation_count} observations."
        ),
        "",
        (
            "Scope: localization-only, SWE-bench-derived evidence. This is not a "
            "SWE-bench resolved rate because no patch was generated or executed by "
            "the official harness."
        ),
        "",
        "| Variant | Runs | File Hit@1 | File Hit@3 | File Hit@5 | Recall@5 macro | Recall@5 micro | MRR | Symbol Hit@5 | Evidence validity | Unsupported relations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in report.aggregates:
        lines.append(
            "| "
            + " | ".join(
                (
                    aggregate.variant.value,
                    str(aggregate.observation_count),
                    _score(aggregate.file_hit_at_1.macro),
                    _score(aggregate.file_hit_at_3.macro),
                    _score(aggregate.file_hit_at_5.macro),
                    _score(aggregate.file_recall_at_5.macro),
                    _score(aggregate.file_recall_at_5.micro),
                    _score(aggregate.mean_reciprocal_rank.macro),
                    _score(aggregate.symbol_hit_at_5.macro),
                    _score(aggregate.evidence_validity.micro),
                    _score(aggregate.unsupported_relation_rate.micro),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _score(score: LocalizationScalarScore) -> str:
    if score.availability is LocalizationMetricAvailability.UNAVAILABLE:
        assert score.unavailable_reason is not None
        return f"n/a ({score.unavailable_reason.value})"
    assert score.value_ppm is not None
    whole = score.value_ppm // 10_000
    fraction = score.value_ppm % 10_000
    return f"{whole}.{fraction:04d}%"


def _observation_sort_key(
    observation: LocalizationRunObservation,
) -> tuple[int, str, int, str]:
    return (
        tuple(LocalizationVariant).index(observation.variant),
        observation.case_id,
        observation.repeat_index,
        observation.observation_id,
    )


def _observation_set_id(
    *,
    dataset_manifest_id: str,
    budget_profile_id: str,
    pricing_table_id: str | None,
    observation_ids: tuple[str, ...],
) -> str:
    encoded = json.dumps(
        {
            "kind": "localization_observation_set",
            "schema_version": LOCALIZATION_SCHEMA_VERSION,
            "dataset_manifest_id": dataset_manifest_id,
            "budget_profile_id": budget_profile_id,
            "pricing_table_id": pricing_table_id,
            "observation_ids": list(observation_ids),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
