"""Fail-closed recording for lifecycle attempts and model invocations."""

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter_ns, time_ns
from typing import TypeVar

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from repocompass.agent.context import (
    RawArtifactKind,
    create_raw_artifact_metadata,
)
from repocompass.agent.context_gathering import GatheredContext
from repocompass.agent.usage import (
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
    UsageRecord,
    create_model_invocation_record,
    extract_usage_record,
)
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    RawRunAttempt,
    RawRunStore,
)
from repocompass.tools.contracts import IDENTIFIER_PATTERN
from repocompass.tools.gateway import ToolExecutionContext

_PROVIDER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MODEL_METADATA_PROVIDER = "repocompass_provider"
_MODEL_METADATA_NAME = "repocompass_model"
_MISSING = object()
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Non-secret provider and model labels attached to one configured client."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not _PROVIDER_PATTERN.fullmatch(
            self.provider
        ):
            raise ValueError("provider must be a canonical non-secret identifier")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 256
        ):
            raise ValueError("model must be a non-empty bounded string")
        object.__setattr__(self, "model", self.model.strip())


def describe_model(model: object) -> ModelIdentity:
    """Resolve explicit adapter metadata, then conservative public fallbacks."""

    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, Mapping):
        configured_provider = metadata.get(_MODEL_METADATA_PROVIDER)
        configured_model = metadata.get(_MODEL_METADATA_NAME)
        if isinstance(configured_provider, str) and isinstance(configured_model, str):
            return ModelIdentity(
                provider=configured_provider,
                model=configured_model,
            )

    module_name = type(model).__module__.lower()
    provider = "openai" if "langchain_openai" in module_name else "unknown"
    model_name = _first_string_attribute(model, "model_name", "model")
    if model_name is None:
        model_name = type(model).__name__ or "unknown-model"
    return ModelIdentity(provider=provider, model=model_name)


class RawRunRecorder:
    """Persist raw context and invocation provenance without entering graph state."""

    def __init__(
        self,
        store: RawRunStore,
        *,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(store, RawRunStore):
            raise TypeError("store must be a RawRunStore")
        self._store = store
        self._wall_clock_ms = wall_clock_ms or _wall_clock_ms
        self._monotonic_ns = monotonic_ns or perf_counter_ns

    def start_attempt(
        self,
        *,
        run_id: str,
        thread_id: str,
        execution_mode: AttemptExecutionMode,
        state: Mapping[str, object],
    ) -> RawRunAttempt:
        """Record one running attempt and its exact objective artifact."""

        run_id = _identifier(run_id, name="run_id")
        thread_id = _identifier(thread_id, name="thread_id")
        if not isinstance(execution_mode, AttemptExecutionMode):
            raise TypeError("execution_mode must be AttemptExecutionMode")
        turn_number = _positive_integer(state.get("turn_number"), name="turn_number")
        repo_path = state.get("repo_path")
        if not isinstance(repo_path, Path):
            raise TypeError("attempt repo_path must be a pathlib.Path")
        objective = state.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("attempt objective must be a non-empty string")
        objective_bytes = objective.strip().encode("utf-8")
        objective_metadata = create_raw_artifact_metadata(
            kind=RawArtifactKind.HUMAN_INPUT,
            media_type="text/plain",
            content=objective_bytes,
        )
        repository = ToolExecutionContext(repo_path)
        started_at_ms = _nonnegative_clock_value(
            self._wall_clock_ms(),
            name="wall clock",
        )
        attempt = RawRunAttempt(
            run_id=run_id,
            thread_id=thread_id,
            turn_number=turn_number,
            execution_mode=execution_mode,
            repository_id=repository.repository_id,
            commit_sha=repository.commit_sha,
            objective_digest=objective_metadata.content_digest,
            status=AttemptStatus.RUNNING,
            started_at_ms=started_at_ms,
            finished_at_ms=None,
        )
        self._store.record_attempt(attempt)
        self._store.put_artifact(run_id, objective_metadata, objective_bytes)
        return attempt

    def finish_attempt(
        self,
        run_id: str,
        *,
        status: AttemptStatus,
    ) -> RawRunAttempt:
        """Finish one attempt using a bounded wall-clock timestamp."""

        run_id = _identifier(run_id, name="run_id")
        finished_at_ms = _nonnegative_clock_value(
            self._wall_clock_ms(),
            name="wall clock",
        )
        return self._store.finish_attempt(
            run_id,
            status=status,
            finished_at_ms=finished_at_ms,
        )

    def invoke_model(
        self,
        *,
        state: Mapping[str, object],
        config: RunnableConfig,
        node_name: str,
        role: ModelRole,
        identity: ModelIdentity,
        gathered_context: GatheredContext,
        messages: Sequence[AnyMessage],
        plan_step_id: str | None,
        invoke: Callable[[list[AnyMessage]], T],
        parse: Callable[[T], R],
    ) -> R:
        """Persist selected context and exact I/O around one model call."""

        run_id, thread_id, turn_number = _runtime_identity(config)
        state_turn_number = state.get("turn_number")
        if state_turn_number is not None and (
            _positive_integer(state_turn_number, name="turn_number") != turn_number
        ):
            raise ValueError(
                "node turn_number does not match the trusted runtime identity"
            )
        sequence = _nonnegative_integer(
            state.get("context_build_count", 0),
            name="context_build_count",
        )
        if not isinstance(role, ModelRole):
            raise TypeError("role must be ModelRole")
        if not isinstance(identity, ModelIdentity):
            raise TypeError("identity must be ModelIdentity")
        if not isinstance(gathered_context, GatheredContext):
            raise TypeError("gathered_context must be GatheredContext")
        invocation_messages = list(messages)
        if not invocation_messages or not all(
            isinstance(message, BaseMessage) for message in invocation_messages
        ):
            raise TypeError("messages must contain at least one BaseMessage")

        for artifact in gathered_context.raw_artifacts:
            self._store.put_artifact(run_id, artifact.metadata, artifact.content)
        input_content = _canonical_json_bytes(
            {"messages": invocation_messages},
        )
        input_metadata = create_raw_artifact_metadata(
            kind=RawArtifactKind.MODEL_INPUT,
            media_type="application/json",
            content=input_content,
        )
        self._store.put_artifact(run_id, input_metadata, input_content)
        self._store.record_context_pack(
            run_id,
            sequence=sequence,
            pack=gathered_context.pack,
        )

        started_ns = _nonnegative_clock_value(
            self._monotonic_ns(),
            name="monotonic clock",
        )
        raw_response: object = _MISSING
        provider_completed = False
        try:
            raw_response = invoke(invocation_messages)
            provider_completed = True
            parsed = parse(raw_response)
        except Exception:
            latency_ms = self._latency_ms(started_ns)
            output_artifact_id = None
            if provider_completed:
                output_artifact_id = self._put_model_output(run_id, raw_response)
            self._record_invocation(
                run_id=run_id,
                thread_id=thread_id,
                turn_number=turn_number,
                sequence=sequence,
                node_name=node_name,
                role=role,
                identity=identity,
                status=ModelInvocationStatus.FAILED,
                latency_ms=latency_ms,
                plan_step_id=plan_step_id,
                context_pack_id=gathered_context.pack.pack_id,
                input_artifact_id=input_metadata.artifact_id,
                output_artifact_id=output_artifact_id,
                error_code=(
                    "invalid_model_response"
                    if provider_completed
                    else "model_call_failed"
                ),
                usage=(
                    extract_usage_record(raw_response)
                    if provider_completed
                    else UsageRecord(availability=UsageAvailability.UNAVAILABLE)
                ),
            )
            raise

        latency_ms = self._latency_ms(started_ns)
        output_artifact_id = self._put_model_output(run_id, raw_response)
        self._record_invocation(
            run_id=run_id,
            thread_id=thread_id,
            turn_number=turn_number,
            sequence=sequence,
            node_name=node_name,
            role=role,
            identity=identity,
            status=ModelInvocationStatus.SUCCEEDED,
            latency_ms=latency_ms,
            plan_step_id=plan_step_id,
            context_pack_id=gathered_context.pack.pack_id,
            input_artifact_id=input_metadata.artifact_id,
            output_artifact_id=output_artifact_id,
            error_code=None,
            usage=extract_usage_record(raw_response),
        )
        return parsed

    def _latency_ms(self, started_ns: int) -> int:
        finished_ns = _nonnegative_clock_value(
            self._monotonic_ns(),
            name="monotonic clock",
        )
        return max(0, (finished_ns - started_ns) // 1_000_000)

    def _put_model_output(self, run_id: str, response: object) -> str:
        output_content = _canonical_json_bytes({"response": response})
        output_metadata = create_raw_artifact_metadata(
            kind=RawArtifactKind.MODEL_OUTPUT,
            media_type="application/json",
            content=output_content,
        )
        self._store.put_artifact(run_id, output_metadata, output_content)
        return output_metadata.artifact_id

    def _record_invocation(
        self,
        *,
        run_id: str,
        thread_id: str,
        turn_number: int,
        sequence: int,
        node_name: str,
        role: ModelRole,
        identity: ModelIdentity,
        status: ModelInvocationStatus,
        latency_ms: int,
        plan_step_id: str | None,
        context_pack_id: str,
        input_artifact_id: str,
        output_artifact_id: str | None,
        error_code: str | None,
        usage: UsageRecord,
    ) -> None:
        record = create_model_invocation_record(
            run_id=run_id,
            thread_id=thread_id,
            turn_number=turn_number,
            sequence=sequence,
            node_name=node_name,
            role=role,
            provider=identity.provider,
            model=identity.model,
            status=status,
            latency_ms=latency_ms,
            usage=usage,
            plan_step_id=plan_step_id,
            context_pack_id=context_pack_id,
            input_artifact_id=input_artifact_id,
            output_artifact_id=output_artifact_id,
            error_code=error_code,
        )
        self._store.record_model_invocation(record)


def _runtime_identity(config: RunnableConfig) -> tuple[str, str, int]:
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        raise ValueError("Runnable config configurable payload is invalid")
    run_id = configurable.get("repocompass_run_id")
    if not isinstance(run_id, str):
        raise ValueError("Runnable config is missing repocompass_run_id")
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str):
        raise ValueError("Runnable config is missing thread_id")
    turn_number = configurable.get("repocompass_turn_number")
    return (
        _identifier(run_id, name="repocompass_run_id"),
        _identifier(thread_id, name="thread_id"),
        _positive_integer(turn_number, name="repocompass_turn_number"),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, BaseException):
        return {"error_type": type(value).__name__}
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("raw model JSON mappings require string keys")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("raw model JSON must not contain NaN or Infinity")
        return value
    raise TypeError(f"raw model value is not JSON-compatible: {type(value).__name__}")


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_clock_value(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must return a non-negative integer")
    return value


def _first_string_attribute(value: object, *names: str) -> str | None:
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _wall_clock_ms() -> int:
    return time_ns() // 1_000_000
