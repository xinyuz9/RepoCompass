"""Provider-neutral model invocation and token-usage contracts."""

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from repocompass.agent.domain_model import CheckpointSafeModel
from repocompass.tools.contracts import (
    IDENTIFIER_PATTERN,
    MAX_SAFE_JSON_INTEGER,
    Sha256Digest,
)

USAGE_SCHEMA_VERSION = 1
NODE_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
PROVIDER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
ERROR_CODE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class ModelRole(StrEnum):
    """Closed workflow role vocabulary for model-level accounting."""

    PLANNER = "planner"
    EXPLORER = "explorer"
    CANDIDATE_BUILDER = "candidate_builder"
    EVALUATOR = "evaluator"
    REPLANNER = "replanner"


class ModelInvocationStatus(StrEnum):
    """Terminal status of one provider model invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class UsageAvailability(StrEnum):
    """Whether token counts were actually reported by the provider."""

    REPORTED = "reported"
    UNAVAILABLE = "unavailable"


class _UsageContractModel(CheckpointSafeModel):
    """Strict immutable base for model observability records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class UsageRecord(_UsageContractModel):
    """Provider-reported token dimensions; absent values remain unknown."""

    schema_version: Literal[1] = USAGE_SCHEMA_VERSION
    availability: UsageAvailability
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )
    cached_input_tokens: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )
    reasoning_output_tokens: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SAFE_JSON_INTEGER,
    )

    @model_validator(mode="after")
    def reconcile_availability_and_totals(self) -> "UsageRecord":
        dimensions = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.reasoning_output_tokens,
        )
        if self.availability is UsageAvailability.UNAVAILABLE:
            if any(value is not None for value in dimensions):
                raise ValueError("unavailable usage must keep token values unknown")
            return self
        if all(value is None for value in dimensions):
            raise ValueError("reported usage requires at least one token value")
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        if (
            self.cached_input_tokens is not None
            and self.input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("cached_input_tokens must not exceed input_tokens")
        if (
            self.reasoning_output_tokens is not None
            and self.output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_output_tokens must not exceed output_tokens")
        return self


class ModelInvocationRecord(_UsageContractModel):
    """Bounded metadata linking one model call to context and raw artifacts."""

    schema_version: Literal[1] = USAGE_SCHEMA_VERSION
    invocation_id: Sha256Digest
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    thread_id: str = Field(pattern=IDENTIFIER_PATTERN)
    turn_number: int = Field(ge=1, le=MAX_SAFE_JSON_INTEGER)
    sequence: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    node_name: str = Field(pattern=NODE_NAME_PATTERN)
    role: ModelRole
    provider: str = Field(pattern=PROVIDER_PATTERN)
    model: str = Field(min_length=1, max_length=256)
    status: ModelInvocationStatus
    latency_ms: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    usage: UsageRecord
    plan_step_id: str | None = Field(default=None, min_length=1, max_length=128)
    context_pack_id: Sha256Digest | None = None
    input_artifact_id: Sha256Digest
    output_artifact_id: Sha256Digest | None = None
    error_code: str | None = Field(default=None, pattern=ERROR_CODE_PATTERN)

    @model_validator(mode="after")
    def require_canonical_identity_and_status(self) -> "ModelInvocationRecord":
        expected_id = _model_invocation_id(
            run_id=self.run_id,
            node_name=self.node_name,
            sequence=self.sequence,
        )
        if self.invocation_id != expected_id:
            raise ValueError("invocation_id does not match run, node, and sequence")
        if self.status is ModelInvocationStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError("successful invocation must not have error_code")
            if self.output_artifact_id is None:
                raise ValueError("successful invocation requires output_artifact_id")
        elif self.error_code is None:
            raise ValueError("failed invocation requires error_code")
        return self


def create_model_invocation_record(
    *,
    run_id: str,
    thread_id: str,
    turn_number: int,
    sequence: int,
    node_name: str,
    role: ModelRole,
    provider: str,
    model: str,
    status: ModelInvocationStatus,
    latency_ms: int,
    usage: UsageRecord,
    input_artifact_id: str,
    output_artifact_id: str | None,
    plan_step_id: str | None = None,
    context_pack_id: str | None = None,
    error_code: str | None = None,
) -> ModelInvocationRecord:
    """Create one deterministic invocation identity from trusted run metadata."""

    return ModelInvocationRecord(
        invocation_id=_model_invocation_id(
            run_id=run_id,
            node_name=node_name,
            sequence=sequence,
        ),
        run_id=run_id,
        thread_id=thread_id,
        turn_number=turn_number,
        sequence=sequence,
        node_name=node_name,
        role=role,
        provider=provider,
        model=model,
        status=status,
        latency_ms=latency_ms,
        usage=usage,
        plan_step_id=plan_step_id,
        context_pack_id=context_pack_id,
        input_artifact_id=input_artifact_id,
        output_artifact_id=output_artifact_id,
        error_code=error_code,
    )


def _model_invocation_id(*, run_id: str, node_name: str, sequence: int) -> str:
    payload = json.dumps(
        {
            "run_id": run_id,
            "node_name": node_name,
            "sequence": sequence,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"repocompass:model-invocation:v1\0" + payload).hexdigest()
