"""Offline admission and failure tests; no browser or provider required."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from repocompass import cli
from repocompass.dashboard import runtime
from repocompass.tools import browser, gateway
from repocompass.tools.browser import (
    BrowserExecutor,
    BrowserPage,
    BrowserPolicy,
    BrowserReadArguments,
)
from repocompass.tools.contracts import ToolCapability, ToolDecisionReason
from repocompass.tools.gateway import (
    ToolAuditLog,
    ToolExecutionContext,
    ToolGateway,
    create_repository_tool_registry,
)


def policy():
    return BrowserPolicy((BrowserPage("docs", "http://127.0.0.1:8888/docs.html"),))


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://localhost:8888/",
        "http://127.1:8888/",
        "http://2130706433:8888/",
        "http://[::1]:8888/",
        "file:///etc/passwd",
        "http://127.0.0.1:8888@evil.com/",
        "http://user:pass@127.0.0.1:8888/",
        "http://127.0.0.1:8888/../secret",
        "http://127.0.0.1:8888/%2e%2e/secret",
        "http://127.0.0.1:8888//other",
        "http://127.0.0.1:8888/./other",
        "http://127.0.0.1:8888/page?token=secret",
        "http://127.0.0.1:8888/page#fragment",
        "http://127.0.0.1:65536/",
        "http://127.0.0.1:80/",
        "http://127.0.0.1:8888/\n",
    ],
)
def test_only_canonical_explicit_loopback_pages(url):
    with pytest.raises(ValueError):
        BrowserPage("docs", url)


def test_policy_bounds_one_origin_and_no_aliases():
    page = policy().pages[0]
    for pages in (
        (),
        [page],
        (page, page),
        (page,) * 9,
        (page, BrowserPage("other", page.url)),
        (page, BrowserPage("other", "http://127.0.0.1:8889/")),
    ):
        with pytest.raises(ValueError):
            BrowserPolicy(pages)
    other = BrowserPage("other", "http://127.0.0.1:8888/other")
    assert BrowserPolicy((page, other)).digest == BrowserPolicy((other, page)).digest
    assert BrowserPolicy((other,)).digest != policy().digest


def test_no_model_url_javascript_or_action_arguments():
    for arguments in (
        {"page_id": "docs", "url": "http://evil.com"},
        {"page_id": "docs", "javascript": "fetch('secret')"},
        {"page_id": "docs", "action": "submit"},
        {"page_id": "../file"},
    ):
        with pytest.raises(ValidationError):
            BrowserReadArguments.model_validate(arguments)


def test_browser_default_absent_and_requires_explicit_network_process_capabilities(
    tmp_path,
):
    assert "browser_read_page" not in create_repository_tool_registry().names
    registry = create_repository_tool_registry(browser_policy=policy())
    context = ToolExecutionContext(tmp_path)
    result = ToolGateway(registry).execute(
        context.bind_call(
            call_id="one", tool_name="browser_read_page", arguments={"page_id": "docs"}
        ),
        context=context,
        available_budget=1,
        audit_log=ToolAuditLog("test"),
    )
    assert result.decision.reason == ToolDecisionReason.CAPABILITY_DENIED
    assert result.budget_consumed == 0


@pytest.mark.parametrize("failure", [False, True])
def test_browser_deduplicates_across_steps_even_without_git(
    tmp_path, monkeypatch, failure
):
    execute = Mock(
        return_value=gateway._IsolatedExecutionResult(
            reason=ToolDecisionReason.EXECUTION_ERROR if failure else None,
            output=None if failure else '{"snapshot":"untrusted"}',
        )
    )
    monkeypatch.setattr(gateway, "_execute_isolated", execute)
    configured = ToolGateway(
        create_repository_tool_registry(browser_policy=policy()),
        allowed_capabilities=frozenset(
            {ToolCapability.NETWORK, ToolCapability.PROCESS}
        ),
    )
    context = ToolExecutionContext(tmp_path)
    log = ToolAuditLog("test")
    request = context.bind_call(
        call_id="one", tool_name="browser_read_page", arguments={"page_id": "docs"}
    )
    first = configured.execute(
        request, context=context, available_budget=1, audit_log=log
    )
    assert first.budget_consumed == 1
    next_step = ToolAuditLog("test", records=log.records, deduplication_floor=1)
    second = configured.execute(
        request.model_copy(update={"call_id": "two"}),
        context=context,
        available_budget=1,
        audit_log=next_step,
    )
    assert second.decision.reason == ToolDecisionReason.DUPLICATE_CALL
    assert second.budget_consumed == 0
    assert execute.call_count == 1
    assert len(next_step.records) == 2
    # No external snapshot cached under an immutable repository SHA.
    assert configured.result_cache.get(first.audit_record.fingerprint) is None


def test_unknown_page_and_missing_runtime_do_not_fetch(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(browser, "_fetch_html", fetch)
    executor = BrowserExecutor(policy())
    with pytest.raises(ValueError):
        executor(Path("."), BrowserReadArguments(page_id="other"), "repo", None)
    monkeypatch.setattr(browser.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        executor(Path("."), BrowserReadArguments(page_id="docs"), "repo", None)
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "status,headers,body",
    [
        (302, {"Location": "http://evil.test/"}, b""),
        (200, {"Content-Type": "application/json"}, b"{}"),
        (
            200,
            {"Content-Type": "text/html", "Content-Disposition": "attachment"},
            b"file",
        ),
        (200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, b"compressed"),
        (200, {"Content-Type": "text/html", "Content-Length": "999999999999"}, b""),
        (200, {"Content-Type": "text/html", "Content-Length": "invalid"}, b""),
        (200, {"Content-Type": "text/html"}, b"x" * (browser.MAX_HTML_BYTES + 1)),
        (200, {"Content-Type": "text/html"}, b"\xff"),
    ],
)
def test_reject_redirect_download_compression_oversize_encoding(
    monkeypatch, status, headers, body
):
    response = Mock(status=status)
    response.getheader.side_effect = lambda key, default=None: headers.get(key, default)
    response.read.return_value = body
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(browser.http.client, "HTTPConnection", factory)
    with pytest.raises(ValueError):
        browser._fetch_html(policy().pages[0].url)
    factory.assert_called_once_with("127.0.0.1", 8888, timeout=3)
    connection.request.assert_called_once_with(
        "GET",
        "/docs.html",
        headers={"Accept": "text/html", "Accept-Encoding": "identity"},
    )
    connection.close.assert_called_once()


def test_executor_does_not_retry_and_filters_page_links(monkeypatch):
    monkeypatch.setattr(browser, "_sandbox_command", lambda: ["sandbox"])
    fetch = Mock(return_value=b"<h1>test</h1>")
    monkeypatch.setattr(browser, "_fetch_html", fetch)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "title": "test",
                    "text": "untrusted",
                    "truncated": False,
                    "links": [
                        {"url": policy().pages[0].url, "label": "docs"},
                        {"url": "http://other.test/", "label": "escape"},
                    ],
                }
            ).encode(),
            b"",
        )
    )
    monkeypatch.setattr(browser.subprocess, "run", run)
    result = json.loads(
        BrowserExecutor(policy())(
            Path("."), BrowserReadArguments(page_id="docs"), "repo", None
        )
    )
    assert result["links"] == [{"page_id": "docs", "label": "docs"}]
    assert result["policy_digest"] == policy().digest
    assert result["kind"] == "untrusted_static_browser_snapshot"
    assert len(result["source_sha256"]) == 64
    assert run.call_args.kwargs["timeout"] == 10
    assert set(run.call_args.kwargs["env"]) == {"PATH", "LANG"}
    fetch.assert_called_once()
    run.assert_called_once()


def test_cli_page_parsing_and_dashboard_requires_mutation_opt_in():
    args = cli.create_parser().parse_args(
        ["--browser-page", "docs=http://127.0.0.1:8888/docs.html"]
    )
    assert tuple(args.browser_page) == policy().pages
    with pytest.raises(SystemExit):
        cli.create_parser().parse_args(["--browser-page", "docs=https://example.com/"])
    with pytest.raises(SystemExit):
        runtime.main(
            ["--demo", "--browser-page", "docs=http://127.0.0.1:8888/docs.html"]
        )


def test_invalid_policy_fails_before_loading_provider(tmp_path, monkeypatch):
    create = Mock()
    monkeypatch.setattr(cli, "create_workflow_models", create)
    assert (
        cli.main(
            [
                str(tmp_path),
                "--objective",
                "test",
                "--browser-page",
                "docs=http://127.0.0.1:8888/docs.html",
                "--browser-page",
                "docs=http://127.0.0.1:8888/other.html",
            ]
        )
        == 1
    )
    create.assert_not_called()


def test_unavailable_browser_stops_cli_before_models_or_storage(tmp_path, monkeypatch):
    create = Mock()
    monkeypatch.setattr(cli, "create_workflow_models", create)
    monkeypatch.setattr(
        cli, "validate_browser_runtime", Mock(side_effect=ValueError("unavailable"))
    )
    db = tmp_path / "must-not-create.sqlite"
    assert (
        cli.main(
            [
                str(tmp_path),
                "--objective",
                "test",
                "--browser-page",
                "docs=http://127.0.0.1:8888/docs.html",
                "--checkpoint-db",
                str(db),
            ]
        )
        == 1
    )
    create.assert_not_called()
    assert not db.exists()


@pytest.mark.parametrize(
    "stdout,returncode,success",
    [
        (b'{"title":"isolation-probe"}', 0, True),
        (b'{"title":"wrong"}', 0, False),
        (b"not json", 0, False),
        (b"", 1, False),
    ],
)
def test_startup_probe_never_fetches_operator_page(
    monkeypatch, stdout, returncode, success
):
    monkeypatch.setattr(browser, "_sandbox_command", lambda: ["sandbox"])
    fetch = Mock()
    monkeypatch.setattr(browser, "_fetch_html", fetch)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], returncode, stdout, b"private diagnostic"
        )
    )
    monkeypatch.setattr(browser.subprocess, "run", run)
    if success:
        browser.validate_browser_runtime()
    else:
        with pytest.raises(ValueError, match="no unsafe fallback") as error:
            browser.validate_browser_runtime()
        assert "private diagnostic" not in str(error.value)
    fetch.assert_not_called()
    run.assert_called_once()


def test_browser_metadata_does_not_accept_arbitrary_or_unbound_capabilities():
    from repocompass.dashboard.contracts import DashboardSystemInfo

    fields = dict(
        product_version="0.1.0",
        repository_id="a" * 64,
        workspace_name="fixture",
        memory_available=False,
        pricing_configured=False,
        lifecycle_mutations_enabled=True,
        memory_mutations_enabled=False,
    )
    assert DashboardSystemInfo(**fields).static_browser_page_ids == ()
    assert DashboardSystemInfo(
        **fields, static_browser_page_ids=("docs",)
    ).static_browser_page_ids == ("docs",)
    for ids in (
        ("docs", "docs"),
        ("http://evil.test",),
        tuple(f"doc{i}" for i in range(9)),
    ):
        with pytest.raises(ValidationError):
            DashboardSystemInfo(**fields, static_browser_page_ids=ids)
    with pytest.raises(ValidationError):
        DashboardSystemInfo(
            **{**fields, "lifecycle_mutations_enabled": False},
            static_browser_page_ids=("docs",),
        )
    with pytest.raises(ValidationError):
        DashboardSystemInfo(**fields, demo_mode=True, static_browser_page_ids=("docs",))
