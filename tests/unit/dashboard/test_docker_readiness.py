"""Readiness probes must tolerate transport startup, not hide HTTP failures."""

import http.client
import urllib.error

import pytest

from tests.integration import test_docker_image as smoke


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("starting"),
        http.client.RemoteDisconnected("starting"),
        urllib.error.URLError("connection refused"),
        TimeoutError("starting"),
    ],
)
def test_readiness_retries_transient_transport_errors(monkeypatch, error):
    calls = []

    def probe(url):
        calls.append(url)
        if len(calls) == 1:
            raise error
        return "{}"

    monkeypatch.setattr(smoke, "get", probe)
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    smoke.wait_until_ready("http://127.0.0.1:8766")
    assert calls == ["http://127.0.0.1:8766/api/v1/system"] * 2


def test_readiness_has_a_deadline(monkeypatch):
    def unavailable(_):
        raise ConnectionResetError("still starting")

    monkeypatch.setattr(smoke, "get", unavailable)
    with pytest.raises(TimeoutError, match="readiness timed out"):
        smoke.wait_until_ready("http://127.0.0.1:8766", timeout=0)


@pytest.mark.parametrize("status", [400, 500])
def test_readiness_preserves_http_errors(monkeypatch, status):
    error = urllib.error.HTTPError("http://localhost", status, "failed", {}, None)

    def rejected(_):
        raise error

    monkeypatch.setattr(smoke, "get", rejected)
    with pytest.raises(urllib.error.HTTPError) as raised:
        smoke.wait_until_ready("http://127.0.0.1:8766")
    assert raised.value is error
