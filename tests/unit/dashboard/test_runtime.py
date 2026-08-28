"""Credential-free loopback Dashboard runtime tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repocompass.dashboard import runtime


def test_inspection_models_fail_closed_without_provider_credentials() -> None:
    models = runtime.create_inspection_models()

    assert models.explorer.bind_tools(()).with_structured_output(object) is (
        models.explorer
    )
    with pytest.raises(RuntimeError, match="cannot invoke models"):
        models.explorer.invoke([])


def test_runtime_serves_loopback_read_api_without_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = tmp_path / "state"
    observed: dict[str, object] = {}

    def fake_run(app, **kwargs) -> None:
        observed.update(kwargs)
        response = TestClient(app).get("/api/v1/system")
        threads = TestClient(app).get("/api/v1/threads")
        assert response.status_code == 200
        assert response.json()["pricing_configured"] is False
        assert threads.json()["threads"] == []

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(runtime.uvicorn, "run", fake_run)

    result = runtime.main(
        [
            "--repo-path",
            str(repository),
            "--checkpoint-db",
            str(state / "checkpoints.sqlite"),
            "--raw-store-db",
            str(state / "raw.sqlite"),
            "--memory-db",
            str(state / "memory.sqlite"),
            "--trace-dir",
            str(state / "traces"),
        ]
    )

    assert result == 0
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8766
    assert observed["proxy_headers"] is False
    assert observed["forwarded_allow_ips"] == ""
    assert observed["limit_concurrency"] == 32


def test_runtime_parser_rejects_non_loopback_host_and_invalid_port(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as invalid_host:
        runtime.main(
            [
                "--repo-path",
                str(tmp_path),
                "--host",
                "0.0.0.0",
            ]
        )
    with pytest.raises(SystemExit) as invalid_port:
        runtime.main(
            [
                "--repo-path",
                str(tmp_path),
                "--port",
                "70000",
            ]
        )

    assert invalid_host.value.code == 2
    assert invalid_port.value.code == 2


def test_runtime_reports_missing_repository_without_starting_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invoked = False

    def fake_run(*args, **kwargs) -> None:
        nonlocal invoked
        invoked = True

    monkeypatch.setattr(runtime.uvicorn, "run", fake_run)
    result = runtime.main(
        [
            "--repo-path",
            str(tmp_path / "missing"),
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
            "--raw-store-db",
            str(tmp_path / "raw.sqlite"),
            "--memory-db",
            str(tmp_path / "memory.sqlite"),
        ]
    )

    assert result == 1
    assert invoked is False
    assert "repocompass-dashboard: error:" in capsys.readouterr().err
