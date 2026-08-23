"""Raw Run Store recording at the model-invocation boundary."""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from repocompass.agent.context import RawArtifactKind
from repocompass.agent.context_gathering import (
    gather_planner_context,
    render_model_context,
)
from repocompass.agent.invocation import (
    ModelIdentity,
    RawRunRecorder,
    describe_model,
)
from repocompass.agent.usage import (
    ModelInvocationStatus,
    ModelRole,
    UsageAvailability,
)
from repocompass.storage.raw_store import (
    AttemptExecutionMode,
    AttemptStatus,
    open_raw_run_store,
)


def _next(values: list[int]) -> Callable[[], int]:
    iterator: Iterator[int] = iter(values)
    return lambda: next(iterator)


def _state(tmp_path: Path, *, count: int = 0) -> dict[str, object]:
    return {
        "repo_path": tmp_path,
        "objective": "Inspect authentication.",
        "turn_number": 1,
        "context_build_count": count,
    }


def _config(run_id: str = "run-1", thread_id: str = "thread-1") -> dict:
    return {
        "configurable": {
            "repocompass_run_id": run_id,
            "repocompass_turn_number": 1,
            "thread_id": thread_id,
        }
    }


def _gathered():
    return gather_planner_context(
        objective="Inspect authentication.",
        max_tool_calls_per_step=6,
        max_explorer_calls_per_step=3,
        budget_tokens=4096,
    )


def _messages() -> list[AnyMessage]:
    gathered = _gathered()
    return [
        SystemMessage(content="Plan safely."),
        HumanMessage(content=render_model_context(ModelRole.PLANNER, gathered.pack)),
    ]


def test_successful_invocation_persists_context_and_exact_io(
    tmp_path: Path,
) -> None:
    database = tmp_path / "raw.sqlite"
    gathered = _gathered()
    messages = _messages()
    response = AIMessage(
        content="Model output retained verbatim.",
        response_metadata={"model_name": "gpt-test"},
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "input_token_details": {"cache_read": 20},
        },
    )

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(
            store,
            wall_clock_ms=_next([1_000, 1_200]),
            monotonic_ns=_next([10_000_000, 35_000_000]),
        )
        recorder.start_attempt(
            run_id="run-1",
            thread_id="thread-1",
            execution_mode=AttemptExecutionMode.START,
            state=_state(tmp_path),
        )

        parsed = recorder.invoke_model(
            state=_state(tmp_path),
            config=_config(),
            node_name="plan_investigation",
            role=ModelRole.PLANNER,
            identity=ModelIdentity(provider="openai", model="gpt-test"),
            gathered_context=gathered,
            messages=messages,
            plan_step_id=None,
            invoke=lambda _: response,
            parse=lambda raw: raw,
        )
        recorder.finish_attempt("run-1", status=AttemptStatus.COMPLETED)

        assert parsed is response
        attempt = store.get_attempt("run-1")
        assert attempt is not None
        assert attempt.status is AttemptStatus.COMPLETED
        assert attempt.started_at_ms == 1_000
        assert attempt.finished_at_ms == 1_200
        packs = store.list_context_packs("run-1")
        assert packs == (gathered.pack,)
        invocations = store.list_model_invocations("run-1")
        assert len(invocations) == 1
        invocation = invocations[0]
        assert invocation.sequence == 0
        assert invocation.node_name == "plan_investigation"
        assert invocation.role is ModelRole.PLANNER
        assert invocation.provider == "openai"
        assert invocation.model == "gpt-test"
        assert invocation.status is ModelInvocationStatus.SUCCEEDED
        assert invocation.latency_ms == 25
        assert invocation.usage.availability is UsageAvailability.REPORTED
        assert invocation.usage.input_tokens == 100
        assert invocation.usage.output_tokens == 25
        assert invocation.usage.total_tokens == 125
        assert invocation.usage.cached_input_tokens == 20
        assert invocation.context_pack_id == gathered.pack.pack_id

        artifacts = {
            metadata.artifact_id: metadata for metadata in store.list_artifacts("run-1")
        }
        assert {
            RawArtifactKind.HUMAN_INPUT,
            RawArtifactKind.MODEL_INPUT,
            RawArtifactKind.MODEL_OUTPUT,
        }.issubset({metadata.kind for metadata in artifacts.values()})
        stored_input = store.get_artifact(invocation.input_artifact_id)
        stored_output = store.get_artifact(invocation.output_artifact_id or "")
        assert stored_input is not None
        assert stored_output is not None
        assert b"role=planner" in stored_input.content
        assert b"Model output retained verbatim." in stored_output.content


def test_provider_failure_is_recorded_without_exception_text(
    tmp_path: Path,
) -> None:
    database = tmp_path / "raw.sqlite"
    called = False

    def fail(_: list[AnyMessage]) -> object:
        nonlocal called
        called = True
        raise RuntimeError("api-key=must-not-enter-metadata")

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(
            store,
            wall_clock_ms=_next([2_000, 2_100]),
            monotonic_ns=_next([1_000_000, 3_000_000]),
        )
        state = _state(tmp_path)
        recorder.start_attempt(
            run_id="run-fail",
            thread_id="thread-fail",
            execution_mode=AttemptExecutionMode.START,
            state=state,
        )

        with pytest.raises(RuntimeError, match="must-not-enter"):
            recorder.invoke_model(
                state=state,
                config=_config("run-fail", "thread-fail"),
                node_name="agent",
                role=ModelRole.EXPLORER,
                identity=ModelIdentity(provider="deepseek", model="deepseek-test"),
                gathered_context=_gathered(),
                messages=_messages(),
                plan_step_id="inspect-auth",
                invoke=fail,
                parse=lambda raw: raw,
            )
        recorder.finish_attempt("run-fail", status=AttemptStatus.FAILED)

        assert called is True
        invocation = store.list_model_invocations("run-fail")[0]
        assert invocation.status is ModelInvocationStatus.FAILED
        assert invocation.error_code == "model_call_failed"
        assert invocation.output_artifact_id is None
        assert "must-not-enter" not in invocation.model_dump_json()
        assert store.get_attempt("run-fail").status is AttemptStatus.FAILED  # type: ignore[union-attr]


def test_parse_failure_retains_raw_response_and_uses_stable_error_code(
    tmp_path: Path,
) -> None:
    database = tmp_path / "raw.sqlite"
    raw_response = {
        "raw": AIMessage(
            content="provider response before parse failure",
            usage_metadata={
                "input_tokens": 40,
                "output_tokens": 5,
                "total_tokens": 45,
            },
        ),
        "parsed": None,
        "parsing_error": ValueError("secret parser detail"),
    }

    def reject_parse(_: object) -> object:
        raise TypeError("structured response is invalid")

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(
            store,
            wall_clock_ms=_next([3_000]),
            monotonic_ns=_next([5_000_000, 9_000_000]),
        )
        state = _state(tmp_path)
        recorder.start_attempt(
            run_id="run-parse",
            thread_id="thread-parse",
            execution_mode=AttemptExecutionMode.START,
            state=state,
        )

        with pytest.raises(TypeError, match="structured response"):
            recorder.invoke_model(
                state=state,
                config=_config("run-parse", "thread-parse"),
                node_name="evaluate_candidate",
                role=ModelRole.EVALUATOR,
                identity=ModelIdentity(provider="openai", model="gpt-test"),
                gathered_context=_gathered(),
                messages=_messages(),
                plan_step_id="inspect-auth",
                invoke=lambda _: raw_response,
                parse=reject_parse,
            )

        invocation = store.list_model_invocations("run-parse")[0]
        assert invocation.status is ModelInvocationStatus.FAILED
        assert invocation.error_code == "invalid_model_response"
        assert invocation.usage.availability is UsageAvailability.REPORTED
        assert invocation.usage.total_tokens == 45
        assert invocation.output_artifact_id is not None
        output = store.get_artifact(invocation.output_artifact_id)
        assert output is not None
        assert b"provider response before parse failure" in output.content
        assert b"secret parser detail" not in output.content


def test_invalid_runtime_identity_fails_before_model_call(tmp_path: Path) -> None:
    database = tmp_path / "raw.sqlite"
    invoke = MagicMock()

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(store)
        with pytest.raises(ValueError, match="repocompass_run_id"):
            recorder.invoke_model(
                state=_state(tmp_path),
                config={"configurable": {"thread_id": "thread-1"}},
                node_name="agent",
                role=ModelRole.EXPLORER,
                identity=ModelIdentity(provider="openai", model="gpt-test"),
                gathered_context=_gathered(),
                messages=_messages(),
                plan_step_id=None,
                invoke=invoke,
                parse=lambda raw: raw,
            )

    invoke.assert_not_called()


def test_model_identity_uses_explicit_non_secret_adapter_metadata() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.metadata = {
        "repocompass_provider": "deepseek",
        "repocompass_model": "deepseek-v4-flash",
        "api_key": "must-not-be-read",
    }

    assert describe_model(model) == ModelIdentity(
        provider="deepseek",
        model="deepseek-v4-flash",
    )
