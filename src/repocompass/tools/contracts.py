"""Strict contracts for tool authorization, audit, and progress detection.

The models in this module are data-only boundaries. Execution, caching, policy
evaluation, and audit persistence belong to the Tool Gateway introduced in
M3.1. Keeping these contracts independent of LangChain makes them reusable by
the graph, CLI, tests, and future MCP interface.
"""

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from repocompass.agent.domain_model import CheckpointSafeModel

TOOL_CONTRACT_SCHEMA_VERSION = 1
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_SHA_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
MAX_TOOL_ARGUMENT_BYTES = 65_536
MAX_JSON_NESTING = 32
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1

Sha256Digest = Annotated[str, Field(pattern=SHA256_PATTERN)]
CommitSha = Annotated[str, Field(pattern=COMMIT_SHA_PATTERN)]


class ToolCapability(StrEnum):
    """Closed vocabulary of effects a model-visible tool may request."""

    FILESYSTEM_READ = "filesystem_read"
    NETWORK = "network"
    PROCESS = "process"
    FILESYSTEM_WRITE = "filesystem_write"
    DESTRUCTIVE = "destructive"


class ToolSideEffect(StrEnum):
    """Maximum repository-side effect permitted for one tool."""

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class ToolRiskLevel(StrEnum):
    """Ordered review tier derived from a tool's declared capabilities."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class ToolDecisionOutcome(StrEnum):
    """Stable externally observable result of a gateway policy decision."""

    ALLOWED = "allowed"
    DENIED = "denied"
    CACHED = "cached"
    DEDUPLICATED = "deduplicated"
    NO_PROGRESS = "no_progress"


class ToolDecisionReason(StrEnum):
    """Closed reason vocabulary used by policy, routing, and audit."""

    ALLOWED = "allowed"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    PATH_ESCAPE = "path_escape"
    CAPABILITY_DENIED = "capability_denied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DUPLICATE_CALL = "duplicate_call"
    CACHE_HIT = "cache_hit"
    TIMEOUT = "timeout"
    OVERSIZED_OUTPUT = "oversized_output"
    EXECUTION_ERROR = "execution_error"
    NO_PROGRESS = "no_progress"


class NoProgressAction(StrEnum):
    """Deterministic action available to the future no-progress detector."""

    CONTINUE = "continue"
    STOP = "stop"
    REPLAN = "replan"


class NoProgressReason(StrEnum):
    """Closed explanation for a no-progress action."""

    PROGRESS_OBSERVED = "progress_observed"
    BELOW_THRESHOLD = "below_threshold"
    NO_PROGRESS_LIMIT = "no_progress_limit"
    EVALUATOR_REPLAN = "evaluator_replan"


class _ToolContractModel(CheckpointSafeModel):
    """Immutable strict base shared by checkpoint-safe M3 contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ToolDescriptor(_ToolContractModel):
    """Versioned static policy metadata for one registered tool."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    capabilities: tuple[ToolCapability, ...] = Field(min_length=1)
    side_effect: ToolSideEffect
    risk_level: ToolRiskLevel
    timeout_ms: int = Field(ge=1, le=60_000)
    max_output_bytes: int = Field(ge=1, le=10_000_000)
    cacheable: bool

    @model_validator(mode="after")
    def require_coherent_capabilities(self) -> "ToolDescriptor":
        """Reject ambiguous ordering, unsafe caching, and understated risk."""

        capability_set = set(self.capabilities)
        canonical_capabilities = tuple(
            capability for capability in ToolCapability if capability in capability_set
        )
        if (
            len(capability_set) != len(self.capabilities)
            or self.capabilities != canonical_capabilities
        ):
            raise ValueError("capabilities must be unique and in canonical order")

        if self.cacheable and self.side_effect is not ToolSideEffect.READ_ONLY:
            raise ValueError("only read-only tools may be cacheable")

        has_write = ToolCapability.FILESYSTEM_WRITE in capability_set
        has_destructive = ToolCapability.DESTRUCTIVE in capability_set
        if self.side_effect is ToolSideEffect.READ_ONLY and (
            has_write or has_destructive
        ):
            raise ValueError("read-only tools cannot declare write capabilities")
        if self.side_effect is ToolSideEffect.WRITE and (
            not has_write or has_destructive
        ):
            raise ValueError("write tools require only the filesystem-write effect")
        if self.side_effect is ToolSideEffect.DESTRUCTIVE and not has_destructive:
            raise ValueError("destructive tools require the destructive capability")
        if has_destructive and self.side_effect is not ToolSideEffect.DESTRUCTIVE:
            raise ValueError("destructive capability requires destructive side effect")

        minimum_risk = ToolRiskLevel.LOW
        if capability_set.intersection(
            {ToolCapability.NETWORK, ToolCapability.PROCESS}
        ):
            minimum_risk = ToolRiskLevel.ELEVATED
        if has_write:
            minimum_risk = ToolRiskLevel.HIGH
        if has_destructive:
            minimum_risk = ToolRiskLevel.CRITICAL

        risk_order = tuple(ToolRiskLevel)
        if risk_order.index(self.risk_level) < risk_order.index(minimum_risk):
            raise ValueError("risk_level understates the declared capabilities")

        return self


class ToolCallRequest(_ToolContractModel):
    """One untrusted model tool call bound to trusted repository identity."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    call_id: str = Field(pattern=IDENTIFIER_PATTERN)
    tool_name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, JsonValue]
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def require_stable_json_arguments(cls, value: object) -> object:
        """Reject coercible containers and values without stable JSON encoding."""

        _validate_stable_json(value, path="arguments", require_object=True)
        if len(_canonical_json_bytes(value)) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError(
                f"arguments exceed the {MAX_TOOL_ARGUMENT_BYTES}-byte limit"
            )
        return value


class ToolCallFingerprint(_ToolContractModel):
    """Canonical digest identifying an execution-equivalent tool call."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    algorithm: Literal["sha256"] = "sha256"
    digest: Sha256Digest


class ToolPolicyDecision(_ToolContractModel):
    """Closed outcome/reason pair emitted by the Tool Gateway."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    outcome: ToolDecisionOutcome
    reason: ToolDecisionReason

    @model_validator(mode="after")
    def require_canonical_pair(self) -> "ToolPolicyDecision":
        """Prevent consumers from interpreting contradictory policy metadata."""

        valid_reasons = {
            ToolDecisionOutcome.ALLOWED: {ToolDecisionReason.ALLOWED},
            ToolDecisionOutcome.DENIED: {
                ToolDecisionReason.UNKNOWN_TOOL,
                ToolDecisionReason.INVALID_ARGUMENTS,
                ToolDecisionReason.PATH_ESCAPE,
                ToolDecisionReason.CAPABILITY_DENIED,
                ToolDecisionReason.BUDGET_EXHAUSTED,
                ToolDecisionReason.TIMEOUT,
                ToolDecisionReason.OVERSIZED_OUTPUT,
                ToolDecisionReason.EXECUTION_ERROR,
            },
            ToolDecisionOutcome.CACHED: {ToolDecisionReason.CACHE_HIT},
            ToolDecisionOutcome.DEDUPLICATED: {ToolDecisionReason.DUPLICATE_CALL},
            ToolDecisionOutcome.NO_PROGRESS: {ToolDecisionReason.NO_PROGRESS},
        }
        if self.reason not in valid_reasons[self.outcome]:
            raise ValueError("outcome and reason are not a canonical policy pair")

        return self


class ToolAuditRecord(_ToolContractModel):
    """Immutable bounded metadata for one append-only tool audit entry.

    Raw arguments, raw output, credentials, authorization headers, and hidden
    model reasoning deliberately have no fields in this contract.
    """

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    sequence: int = Field(ge=0)
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    call_id: str = Field(pattern=IDENTIFIER_PATTERN)
    tool_name: str = Field(pattern=TOOL_NAME_PATTERN)
    repository_id: Sha256Digest
    commit_sha: CommitSha | None = None
    fingerprint: ToolCallFingerprint
    decision: ToolPolicyDecision
    latency_ms: int | None = Field(default=None, ge=0)
    output_bytes: int | None = Field(default=None, ge=0)
    result_digest: Sha256Digest | None = None


class ProgressSnapshot(_ToolContractModel):
    """Canonical signals compared by the future no-progress detector."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    active_step_id: str = Field(min_length=1, max_length=128)
    evidence_fingerprints: tuple[Sha256Digest, ...] = ()
    successful_tool_fingerprints: tuple[Sha256Digest, ...] = ()
    evaluator_request_digest: Sha256Digest | None = None
    plan_revision_count: int = Field(default=0, ge=0)

    @field_validator(
        "evidence_fingerprints",
        "successful_tool_fingerprints",
    )
    @classmethod
    def require_canonical_fingerprints(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require stable set encoding so equal progress hashes identically."""

        if value != tuple(sorted(set(value))):
            raise ValueError("fingerprints must be unique and in canonical order")
        return value


class NoProgressDecision(_ToolContractModel):
    """Deterministic output contract for one no-progress policy evaluation."""

    schema_version: Literal[1] = TOOL_CONTRACT_SCHEMA_VERSION
    action: NoProgressAction
    reason: NoProgressReason
    consecutive_stalled_rounds: int = Field(ge=0)
    threshold: int = Field(ge=1)
    snapshot_digest: Sha256Digest

    @model_validator(mode="after")
    def require_consistent_action(self) -> "NoProgressDecision":
        """Make stop/continue/replan behavior derivable from explicit facts."""

        valid_reasons = {
            NoProgressAction.CONTINUE: {
                NoProgressReason.PROGRESS_OBSERVED,
                NoProgressReason.BELOW_THRESHOLD,
            },
            NoProgressAction.STOP: {NoProgressReason.NO_PROGRESS_LIMIT},
            NoProgressAction.REPLAN: {NoProgressReason.EVALUATOR_REPLAN},
        }
        if self.reason not in valid_reasons[self.action]:
            raise ValueError("action and reason are not a canonical progress pair")

        if (
            self.reason is NoProgressReason.PROGRESS_OBSERVED
            and self.consecutive_stalled_rounds != 0
        ):
            raise ValueError("progress_observed requires zero stalled rounds")
        if self.reason is NoProgressReason.BELOW_THRESHOLD and not (
            0 < self.consecutive_stalled_rounds < self.threshold
        ):
            raise ValueError("below_threshold requires a positive sub-limit count")
        if (
            self.reason is NoProgressReason.NO_PROGRESS_LIMIT
            and self.consecutive_stalled_rounds < self.threshold
        ):
            raise ValueError("no_progress_limit requires the threshold to be met")

        return self


def fingerprint_tool_call(request: ToolCallRequest) -> ToolCallFingerprint:
    """Hash only fields that determine whether execution is equivalent."""

    if not isinstance(request, ToolCallRequest):
        raise TypeError("request must be a ToolCallRequest")

    payload: dict[str, JsonValue] = {
        "schema_version": TOOL_CONTRACT_SCHEMA_VERSION,
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "repository_id": request.repository_id,
        "commit_sha": request.commit_sha,
    }
    return ToolCallFingerprint(digest=_fingerprint_json(payload))


def fingerprint_progress_snapshot(snapshot: ProgressSnapshot) -> str:
    """Return the canonical digest used to compare two progress snapshots."""

    if not isinstance(snapshot, ProgressSnapshot):
        raise TypeError("snapshot must be a ProgressSnapshot")

    payload = snapshot.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    return _fingerprint_json(payload)


def _fingerprint_json(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    """Serialize one prevalidated value with a cross-process stable encoding."""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return serialized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("canonical JSON contains invalid Unicode") from error
    except (TypeError, ValueError) as error:
        raise ValueError("value cannot be encoded as canonical JSON") from error


def _validate_stable_json(
    value: object,
    *,
    path: str,
    require_object: bool = False,
    depth: int = 0,
) -> None:
    """Validate exact JSON runtime types without coercion or custom encoders."""

    if depth > MAX_JSON_NESTING:
        raise ValueError(f"{path} exceeds the JSON nesting limit")
    if require_object and type(value) is not dict:
        raise ValueError(f"{path} must be a JSON object")

    if value is None or type(value) in {str, bool}:
        return
    if type(value) is int:
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise ValueError(f"{path} exceeds the cross-runtime safe integer range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_stable_json(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string object key")
            _validate_stable_json(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return

    raise ValueError(f"{path} contains a value without stable JSON encoding")
