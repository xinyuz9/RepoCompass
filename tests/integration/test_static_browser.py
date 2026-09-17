"""Real Chromium + bwrap tests, explicitly opt-in; never calls a provider."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from repocompass.tools.browser import BrowserPage, BrowserPolicy, _sandbox_command
from repocompass.tools.contracts import ToolCapability, ToolDecisionReason
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    create_repository_tool_registry,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("REPOCOMPASS_BROWSER_TESTS") != "1",
    reason="explicit Linux browser/bubblewrap integration opt-in required",
)


@pytest.fixture
def site():
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            observations.append((self.command, self.path, dict(self.headers)))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/secret")
                self.end_headers()
                return
            body = b"""<title>Fixture</title><h1>Visible documentation</h1>
              <script>document.body.textContent='executed'; fetch('/secret');
                new WebSocket('ws://127.0.0.1:9999/');</script>
              <img src="/secret"><iframe src="/secret"></iframe>
              <link rel="stylesheet" href="/secret"><object data="/secret"></object>
              <form action="/secret" method="POST"><input type="file"><button>Submit</button></form>
              <a href="/next">Next page</a><a download href="/secret">Download</a>
              <a href="file:///etc/passwd">Local file</a><a href="https://example.com/">Escape</a>"""
            if self.path == "/large":
                body = b"<p>" + b"a" * 262145
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Set-Cookie", "credential=must-not-be-reused")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def configured(tmp_path, origin):
    policy = BrowserPolicy(
        tuple(
            BrowserPage(name, origin + path)
            for name, path in (
                ("docs", "/"),
                ("next", "/next"),
                ("redirect", "/redirect"),
                ("large", "/large"),
            )
        )
    )
    gateway = ToolGateway(
        create_repository_tool_registry(browser_policy=policy),
        allowed_capabilities=frozenset(
            {ToolCapability.NETWORK, ToolCapability.PROCESS}
        ),
    )
    return gateway, ToolExecutionContext(tmp_path), ToolAuditLog("browser-integration")


def invoke(config, page, call="one", budget=1):
    gateway, context, log = config
    return gateway.execute(
        context.bind_call(
            call_id=call, tool_name="browser_read_page", arguments={"page_id": page}
        ),
        context=context,
        available_budget=budget,
        audit_log=log,
    )


def test_real_browser_no_script_subresources_credentials_actions_or_repeat(
    tmp_path, site, monkeypatch
):
    origin, observations = site
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-forward")
    config = configured(tmp_path, origin)
    first = invoke(config, "docs")
    assert first.succeeded, first.content
    snapshot = json.loads(first.content)
    assert snapshot["title"] == "Fixture"
    assert "Visible documentation" in snapshot["text"]
    assert "executed" not in snapshot["text"]
    assert snapshot["links"] == [{"page_id": "next", "label": "Next page"}]
    assert (
        invoke(config, "docs", "repeat").decision.reason
        == ToolDecisionReason.DUPLICATE_CALL
    )
    assert invoke(config, "next", "next").succeeded
    assert [(verb, path) for verb, path, _ in observations] == [
        ("GET", "/"),
        ("GET", "/next"),
    ]
    for _, _, headers in observations:
        assert not {name.lower() for name in headers} & {
            "cookie",
            "authorization",
            "referer",
        }
        assert "test-secret" not in json.dumps(headers)
    assert len(config[2].records) == 3
    assert first.audit_record.result_digest is not None


def test_redirect_oversize_unknown_and_budget_fail_closed(tmp_path, site):
    origin, observations = site
    config = configured(tmp_path, origin)
    assert (
        invoke(config, "docs", budget=0).decision.reason
        == ToolDecisionReason.BUDGET_EXHAUSTED
    )
    assert (
        invoke(config, "unknown", "unknown").decision.reason
        == ToolDecisionReason.EXECUTION_ERROR
    )
    for name in ("redirect", "large"):
        failed = invoke(config, name, name)
        assert failed.decision.reason == ToolDecisionReason.EXECUTION_ERROR
        assert "secret" not in failed.content
        assert failed.budget_consumed == 1
    assert [path for _, path, _ in observations] == ["/redirect", "/large"]


def test_actual_sandbox_hides_host_repo_secrets_and_network(
    tmp_path, site, monkeypatch
):
    origin, observations = site
    sentinel = tmp_path / "private.txt"
    sentinel.write_text("private host source", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-exist")
    command = _sandbox_command()
    worker = command.index("--") + 1
    script = f"""import json, os, socket
from pathlib import Path
reachable = False
try:
    socket.create_connection(('127.0.0.1', {origin.rsplit(":", 1)[1]}), timeout=.2).close()
    reachable = True
except OSError:
    pass
print(json.dumps({{'private_exists': Path({str(sentinel)!r}).exists(), 'credential': 'OPENAI_API_KEY' in os.environ, 'network': reachable}}))
"""
    result = subprocess.run(
        command[: worker + 1] + ["-c", script],
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "private_exists": False,
        "credential": False,
        "network": False,
    }
    assert observations == []


def test_production_runner_records_browser_tool_and_verifies_repository_separately(
    tmp_path, site
):
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from repocompass.agent.runner import AgentRunner, RunRequest
    from repocompass.agent.result import FinalResult
    from tests.model_doubles import with_scripted_planner
    from tests.unit.agent.test_graph import _create_scripted_model

    origin, observations = site
    (tmp_path / "entry.py").write_text("VALUE = 7\n", encoding="utf-8")
    final = FinalResult(
        relevant_files=["entry.py"],
        evidence=["entry.py:1 assigns VALUE = 7."],
        summary="The repository constant is 7.",
        unresolved_questions=[],
    )
    explorer = _create_scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "browser_read_page",
                        "args": {"page_id": "docs"},
                        "id": "browser-one",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_code",
                        "args": {
                            "relative_path": "entry.py",
                            "start_line": 1,
                            "end_line": 1,
                        },
                        "id": "source-one",
                    }
                ],
            ),
            AIMessage(content="The source is verified."),
        ],
        final,
    )
    runner = AgentRunner(
        with_scripted_planner(explorer),
        checkpointer=InMemorySaver(),
        browser_policy=BrowserPolicy((BrowserPage("docs", origin + "/"),)),
    )
    outcome = runner.run(
        RunRequest(
            repo_path=tmp_path,
            objective="Inspect docs, then verify VALUE in entry.py.",
            max_tool_calls_per_step=3,
            max_explorer_calls_per_step=4,
        )
    )
    assert outcome.termination_status == "accepted"
    assert [item.tool_name for item in outcome.tool_audit_records] == [
        "browser_read_page",
        "read_code",
    ]
    assert all(
        item.decision.reason == ToolDecisionReason.ALLOWED
        for item in outcome.tool_audit_records
    )
    assert outcome.final_result.relevant_files == ["entry.py"]
    assert [path for _, path, _ in observations] == ["/"]
    tools = explorer.bind_tools.call_args.args[0]
    assert "browser_read_page" in {tool.name for tool in tools}


def test_sandbox_children_die_when_tool_parent_is_killed(site):
    import select
    import sys
    import time
    from pathlib import Path

    command = _sandbox_command()
    worker = command.index("--") + 1
    command = command[: worker + 1] + [
        "-c",
        "import time; print('ready', flush=True); time.sleep(60)",
    ]
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json,subprocess,sys; subprocess.run(json.loads(sys.argv[1]), check=True)",
            json.dumps(command),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    descendants = set()
    try:
        assert select.select([parent.stdout], [], [], 5)[0], "sandbox did not start"
        assert parent.stdout.readline() == b"ready\n"

        def children(pid):
            path = Path(f"/proc/{pid}/task/{pid}/children")
            if path.exists():
                for value in path.read_text().split():
                    child = int(value)
                    descendants.add(child)
                    children(child)

        children(parent.pid)
        assert len(descendants) >= 2
        parent.kill()
        parent.wait(timeout=2)

        def live_children():
            live = []
            for pid in descendants:
                try:
                    state = (
                        Path(f"/proc/{pid}/stat")
                        .read_text()
                        .split(")", 1)[1]
                        .split()[0]
                    )
                except FileNotFoundError:
                    continue
                if state != "Z":
                    live.append(pid)
            return live

        deadline = time.monotonic() + 3
        while live_children() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert live_children() == []
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        parent.stdout.close()
