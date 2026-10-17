"""Real SDK and loopback HTTP; no credentials or external provider requests."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest
from openai import InternalServerError
from pydantic import BaseModel

from repocompass.agent.checkpoint import open_sqlite_checkpointer
from repocompass.agent.runner import AgentRunner, ResumeRequest, RunRequest
from repocompass.models.control import (
    AttemptDeadlineExceeded,
    ModelCallPolicy,
    ModelCallTimeout,
)
from repocompass.models.controlled_openai import ControlledChatOpenAI
from repocompass.models.factory import WorkflowModels
from repocompass.storage.cancellation import (
    CancellationRequest,
    open_cancellation_store,
)
from repocompass.storage.raw_store import open_raw_run_store


@pytest.fixture
def provider():
    state = SimpleNamespace(mode="ok", calls=[], entered=Event(), disconnected=Event())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(payload)
            state.entered.set()
            if state.mode == "hang":
                self.connection.settimeout(5)
                try:
                    if self.connection.recv(1) == b"":
                        state.disconnected.set()
                except TimeoutError:
                    pass
                return
            if state.mode == "error":
                status, body = (
                    500,
                    {"error": {"message": "synthetic failure", "type": "server_error"}},
                )
            else:
                message = {"role": "assistant", "content": "ok"}
                if payload.get("tools"):
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": payload["tools"][0]["function"]["name"],
                                    "arguments": '{"value": 7}',
                                },
                            }
                        ],
                    }
                status, body = (
                    200,
                    {
                        "id": "local",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "offline",
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                )
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def model_for(provider, *, timeout=2):
    return ControlledChatOpenAI(
        model="offline",
        api_key="not-a-real-key",
        base_url=provider.url,
        timeout=httpx.Timeout(2, connect=1),
        max_retries=0,
        model_call_timeout=timeout,
        use_responses_api=False,
    )


def test_sdk_success_usage_structured_output_and_repeated_event_loops(provider):
    class Payload(BaseModel):
        value: int

    model = model_for(provider)
    for _ in range(2):
        result = model.invoke("local only")
        assert result.content == "ok"
        assert result.usage_metadata["total_tokens"] == 5
    structured = model.with_structured_output(
        Payload, method="function_calling", include_raw=True
    )
    assert structured.invoke("local schema")["parsed"] == Payload(value=7)
    assert asyncio.run(model.ainvoke("async local only")).content == "ok"
    assert len(provider.calls) == 4


def test_sdk_does_not_retry_failed_request(provider):
    provider.mode = "error"
    with pytest.raises(InternalServerError):
        model_for(provider).invoke("local failure")
    assert len(provider.calls) == 1


def test_total_request_timeout_closes_connection(provider):
    provider.mode = "hang"
    with pytest.raises(ModelCallTimeout):
        model_for(provider, timeout=0.1).invoke("local timeout")
    assert provider.disconnected.wait(2)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("stop", ["cancel", "deadline", "request"])
def test_inflight_control_is_durable_and_never_replays_implicitly(
    tmp_path, provider, stop
):
    provider.mode = "hang"
    (tmp_path / "example.py").write_text("value = 1\n")
    request_timeout = 0.3 if stop == "request" else 5
    model = model_for(provider, timeout=request_timeout)
    policy = ModelCallPolicy(request_timeout, 0.6 if stop == "deadline" else 10)
    models = WorkflowModels(model, model, model, call_policy=policy)
    database = tmp_path / "checkpoint.sqlite"
    with (
        open_sqlite_checkpointer(database) as checkpoint,
        open_raw_run_store(tmp_path / "raw.sqlite") as raw,
        open_cancellation_store(tmp_path / "cancel.sqlite") as cancellation,
    ):
        runner = AgentRunner(
            models,
            checkpointer=checkpoint,
            raw_run_store=raw,
            cancellation_store=cancellation,
            run_id_factory=lambda: "controlled-run",
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runner.run,
                RunRequest(
                    repo_path=tmp_path,
                    objective="inspect code",
                    max_tool_calls_per_step=3,
                    max_explorer_calls_per_step=2,
                ),
            )
            assert provider.entered.wait(3)
            if stop == "cancel":
                row = cancellation.get("controlled-run")
                cancellation.request(
                    CancellationRequest(
                        run_id=row.run_id,
                        thread_id=row.thread_id,
                        repository_id=row.repository_id,
                        commit_sha=row.commit_sha,
                        request_id="cancel-request",
                    )
                )
                outcome = future.result(timeout=3)
                assert outcome.termination_status == "cancelled"
                assert cancellation.get("controlled-run").state == "cancelled"
                assert outcome.final_result.evidence == []
                thread_id = outcome.thread_id
            else:
                expected_error = (
                    AttemptDeadlineExceeded if stop == "deadline" else ModelCallTimeout
                )
                with pytest.raises(expected_error):
                    future.result(timeout=3)
                assert raw.get_attempt("controlled-run").status.value == "failed"
                assert cancellation.get("controlled-run").state == "finished"
                thread_id = raw.get_attempt("controlled-run").thread_id
        assert provider.disconnected.wait(2)
        records = raw.list_model_invocations("controlled-run")
        assert len(records) == 1
        assert records[0].status.value == "failed"
        assert records[0].usage.availability.value == "unavailable"
        assert records[0].usage.total_tokens is None
        assert (
            records[0].error_code
            == {
                "cancel": "model_call_cancelled",
                "deadline": "attempt_deadline_exceeded",
                "request": "model_call_timeout",
            }[stop]
        )
        assert len(provider.calls) == 1
    # Merely opening/inspecting the persisted checkpoint never starts another call.
    with open_sqlite_checkpointer(database) as checkpoint:
        runner = AgentRunner(models, checkpointer=checkpoint)
        runner.inspect_thread(ResumeRequest(thread_id=thread_id))
        if stop == "cancel":
            assert (
                runner.resume(ResumeRequest(thread_id=thread_id)).termination_status
                == "cancelled"
            )
    assert len(provider.calls) == 1
