"""Raw Run Store recording at the model-invocation boundary."""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from repocompass.agent.context import RawArtifactKind
from repocompass.agent.cost import (
    CostBudgetReason,
    CostBudgetStatus,
    ModelPrice,
    create_pricing_table,
)
from repocompass.agent.context_gathering import (
    gather_planner_context,
    render_model_context,
)
from repocompass.agent.invocation import (
    CostBudgetExceededError,
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
    RawStoreConflictError,
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


def test_identical_context_reuse_records_and_charges_each_invocation(tmp_path: Path):
    database = tmp_path / "raw.sqlite"
    gathered = _gathered()
    calls = []

    def invoke(_):
        calls.append(True)
        return AIMessage(
            content="Same context, new observed response.",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
            },
        )

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(store)
        recorder.start_attempt(
            run_id="run-1",
            thread_id="thread-1",
            execution_mode=AttemptExecutionMode.START,
            state=_state(tmp_path),
        )
        for sequence in (0, 1):
            recorder.invoke_model(
                state=_state(tmp_path, count=sequence),
                config=_config(),
                node_name="build_candidate",
                role=ModelRole.CANDIDATE_BUILDER,
                identity=ModelIdentity(provider="openai", model="gpt-test"),
                gathered_context=gathered,
                messages=_messages(),
                plan_step_id=None,
                invoke=invoke,
                parse=lambda raw: raw,
            )
        assert len(calls) == 2
        assert store.list_context_packs("run-1") == (gathered.pack,)
        records = store.list_model_invocations("run-1")
        assert [record.sequence for record in records] == [0, 1]
        assert len({record.invocation_id for record in records}) == 2
        assert {record.context_pack_id for record in records} == {gathered.pack.pack_id}
        assert sum(record.usage.input_tokens for record in records) == 200
        assert sum(record.usage.output_tokens for record in records) == 50
        prices = create_pricing_table(
            version="repeat-context-test",
            effective_from="2026-09-08",
            entries=(
                ModelPrice(
                    provider="openai",
                    model="gpt-test",
                    input_usd_per_million_tokens="1",
                    output_usd_per_million_tokens="2",
                ),
            ),
        )
        report = store.get_attempt_usage_cost_report("run-1", prices)
        assert report is not None
        assert report.breakdowns[0].cost.estimated_cost_usd == "0.0003"
        assert report.breakdowns[0].invocation_count == 2

        other = gather_planner_context(
            objective="Different content.",
            max_tool_calls_per_step=6,
            max_explorer_calls_per_step=3,
            budget_tokens=4096,
        )
        for artifact in other.raw_artifacts:
            store.put_artifact("run-1", artifact.metadata, artifact.content)
        # Sequence 1 reused a snapshot first stored at sequence 0; it is still
        # owned by the recorded invocation and cannot be replaced by another pack.
        with pytest.raises(RawStoreConflictError, match="context sequence"):
            store.record_context_pack("run-1", sequence=1, pack=other.pack)
    with open_raw_run_store(database) as store:
        assert len(store.list_model_invocations("run-1")) == 2
        assert store.list_context_packs("run-1") == (gathered.pack,)


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


def test_recorder_blocks_a_second_call_after_post_response_budget_overshoot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "raw.sqlite"
    pricing = create_pricing_table(
        version="test-v1",
        effective_from="2026-08-24",
        entries=(
            ModelPrice(
                provider="openai",
                model="gpt-test",
                input_usd_per_million_tokens="2.5",
                output_usd_per_million_tokens="10",
            ),
        ),
    )
    response = AIMessage(
        content="paid response",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    )
    denied_provider_call = MagicMock()
    state = {
        **_state(tmp_path),
        "max_attempt_cost_usd": "0.0001",
    }

    with open_raw_run_store(database) as store:
        recorder = RawRunRecorder(store, pricing_table=pricing)
        recorder.start_attempt(
            run_id="run-budget",
            thread_id="thread-budget",
            execution_mode=AttemptExecutionMode.START,
            state=state,
        )
        recorder.invoke_model(
            state=state,
            config=_config("run-budget", "thread-budget"),
            node_name="plan_investigation",
            role=ModelRole.PLANNER,
            identity=ModelIdentity(provider="openai", model="gpt-test"),
            gathered_context=_gathered(),
            messages=_messages(),
            plan_step_id=None,
            invoke=lambda _: response,
            parse=lambda raw: raw,
        )

        second_state = {**state, "context_build_count": 1}
        with pytest.raises(CostBudgetExceededError) as captured:
            recorder.invoke_model(
                state=second_state,
                config=_config("run-budget", "thread-budget"),
                node_name="agent",
                role=ModelRole.EXPLORER,
                identity=ModelIdentity(provider="openai", model="gpt-test"),
                gathered_context=_gathered(),
                messages=_messages(),
                plan_step_id="inspect-auth",
                invoke=denied_provider_call,
                parse=lambda raw: raw,
            )

        decision = captured.value.budget
        assert decision.status is CostBudgetStatus.OVERSHOT
        assert decision.reason is CostBudgetReason.POST_RESPONSE_OVERSHOOT
        assert decision.next_invocation_allowed is False
        assert len(store.list_model_invocations("run-budget")) == 1
        assert len(store.list_context_packs("run-budget")) == 1
    denied_provider_call.assert_not_called()
