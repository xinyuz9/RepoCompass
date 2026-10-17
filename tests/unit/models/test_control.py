"""No network: deadlines, cancellation cleanup and configuration."""

import asyncio

import pytest

from repocompass.models.config import load_model_settings
from repocompass.models.control import (
    AttemptDeadlineExceeded,
    ModelCallPolicy,
    ModelCallTimeout,
    await_model_request,
    check_attempt_deadline,
    model_execution_scope,
)
from repocompass.storage.cancellation import RunCancellationRequested


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "60"])
@pytest.mark.parametrize(
    "field", ["request_timeout_seconds", "attempt_timeout_seconds"]
)
def test_policy_rejects_unbounded_timeouts(field, value):
    with pytest.raises(ValueError, match="positive finite"):
        ModelCallPolicy(**{field: value})


def test_load_timeout_configuration(monkeypatch):
    monkeypatch.setattr("repocompass.models.config.load_dotenv", lambda **_: None)
    monkeypatch.setenv("REPOCOMPASS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("REPOCOMPASS_MODEL", "offline")
    monkeypatch.setenv("REPOCOMPASS_API_KEY", "not-a-real-key")
    monkeypatch.setenv("REPOCOMPASS_MODEL_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REPOCOMPASS_ATTEMPT_TIMEOUT_SECONDS", "45")
    assert load_model_settings().call_policy == ModelCallPolicy(12.5, 45)
    monkeypatch.setenv("REPOCOMPASS_MODEL_TIMEOUT_SECONDS", "secret-like-invalid-value")
    with pytest.raises(
        ValueError, match="^model timeouts must be positive finite seconds$"
    ):
        load_model_settings()


@pytest.mark.parametrize("stop", ["request", "attempt", "cancel"])
def test_control_joins_cancelled_request_and_resets_scope(stop):
    cleaned = []

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    checks = 0

    def check_cancel():
        nonlocal checks
        checks += 1
        if stop == "cancel" and checks > 1:
            raise RunCancellationRequested("run", "cancel")

    expected = {
        "request": ModelCallTimeout,
        "attempt": AttemptDeadlineExceeded,
        "cancel": RunCancellationRequested,
    }[stop]
    with pytest.raises(expected):
        with model_execution_scope(
            check_cancel, timeout_seconds=0.02 if stop == "attempt" else 10
        ):
            asyncio.run(
                await_model_request(
                    pending, timeout_seconds=0.02 if stop == "request" else 10
                )
            )
    assert cleaned == [True]
    check_attempt_deadline()  # No leaked deadline in the caller's context.


def test_pre_cancelled_request_never_starts():
    invoked = []

    async def invoke():
        invoked.append(True)

    def cancelled():
        raise RunCancellationRequested("run", "cancel")

    with model_execution_scope(cancelled, timeout_seconds=10):
        with pytest.raises(RunCancellationRequested):
            asyncio.run(await_model_request(invoke, timeout_seconds=1))
    assert invoked == []


def test_completed_result_wins_late_cancellation():
    async def invoke():
        return "received usage"

    checks = 0

    def cancel_after_start():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RunCancellationRequested("run", "late")

    with model_execution_scope(cancel_after_start, timeout_seconds=10):
        assert (
            asyncio.run(await_model_request(invoke, timeout_seconds=1))
            == "received usage"
        )


def test_outer_async_cancellation_cleans_up_request():
    async def scenario():
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def pending():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(await_model_request(pending, timeout_seconds=10))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    asyncio.run(scenario())
