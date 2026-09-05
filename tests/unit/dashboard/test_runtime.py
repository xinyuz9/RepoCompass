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
        client = TestClient(app)
        response = client.get("/api/v1/system")
        threads = client.get("/api/v1/threads")
        document = client.get("/api/v1/openapi.json").json()
        assert response.status_code == 200
        assert response.json()["pricing_configured"] is False
        assert response.json()["lifecycle_mutations_enabled"] is False
        assert response.json()["memory_mutations_enabled"] is False
        assert threads.json()["threads"] == []
        assert "/api/v1/threads/{thread_id}/responses" not in document["paths"]
        assert "/api/v1/memory/{memory_id}" not in document["paths"]

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


def test_runtime_memory_mutation_is_independent_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = tmp_path / "state"
    operation_database = state / "operations.sqlite"

    def forbid_workflow_models():
        raise AssertionError("memory-only runtime must not load provider models")

    def fake_run(app, **kwargs) -> None:
        del kwargs
        client = TestClient(app)
        system = client.get("/api/v1/system").json()
        document = client.get("/api/v1/openapi.json").json()
        assert system["lifecycle_mutations_enabled"] is False
        assert system["memory_mutations_enabled"] is True
        assert "/api/v1/threads/{thread_id}/responses" not in document["paths"]
        assert set(document["paths"]["/api/v1/memory/{memory_id}"]) == {"delete"}

    monkeypatch.setattr(runtime, "create_workflow_models", forbid_workflow_models)
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
            "--operation-db",
            str(operation_database),
            "--trace-dir",
            str(state / "traces"),
            "--enable-memory-mutations",
        ]
    )

    assert result == 0
    assert not operation_database.exists()


def test_runtime_lifecycle_mutation_loads_models_and_durable_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = tmp_path / "state"
    operation_database = state / "operations.sqlite"
    model_factory_calls = 0

    def fake_workflow_models():
        nonlocal model_factory_calls
        model_factory_calls += 1
        return runtime.create_inspection_models()

    def fake_run(app, **kwargs) -> None:
        del kwargs
        client = TestClient(app)
        system = client.get("/api/v1/system").json()
        document = client.get("/api/v1/openapi.json").json()
        assert system["lifecycle_mutations_enabled"] is True
        assert system["memory_mutations_enabled"] is False
        assert set(document["paths"]["/api/v1/threads/{thread_id}/responses"]) == {
            "post"
        }
        assert "/api/v1/memory/{memory_id}" not in document["paths"]

    monkeypatch.setattr(runtime, "create_workflow_models", fake_workflow_models)
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
            "--operation-db",
            str(operation_database),
            "--trace-dir",
            str(state / "traces"),
            "--enable-lifecycle-mutations",
        ]
    )

    assert result == 0
    assert model_factory_calls == 1
    assert operation_database.is_file()


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


def test_runtime_demo_is_ephemeral_read_only_and_credential_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "demo-repository"
    repository.mkdir()
    state = tmp_path / "demo-state"
    generated_destinations: list[Path] = []

    def fake_demo(destination: Path):
        generated_destinations.append(destination)
        return type(
            "DemoWorkspace",
            (),
            {
                "repository_path": repository,
                "checkpoint_database": state / "checkpoints.sqlite",
                "raw_store_database": state / "raw.sqlite",
                "memory_database": state / "memory.sqlite",
                "trace_directory": state / "traces",
                "pricing_table": None,
            },
        )()

    def forbid_workflow_models():
        raise AssertionError("demo serving must use inspection-only models")

    def fake_run(app, **kwargs) -> None:
        del kwargs
        client = TestClient(app)
        system = client.get("/api/v1/system").json()
        document = client.get("/api/v1/openapi.json").json()
        assert system["demo_mode"] is True
        assert system["lifecycle_mutations_enabled"] is False
        assert system["memory_mutations_enabled"] is False
        assert client.get("/api/v1/threads").json()["threads"] == []
        assert "/api/v1/investigations" not in document["paths"]
        assert "/api/v1/memory/{memory_id}" not in document["paths"]

    monkeypatch.setattr(runtime, "create_dashboard_demo", fake_demo)
    monkeypatch.setattr(runtime, "create_workflow_models", forbid_workflow_models)
    monkeypatch.setattr(runtime.uvicorn, "run", fake_run)

    assert runtime.main(["--demo"]) == 0
    assert len(generated_destinations) == 1
    assert not generated_destinations[0].exists()


@pytest.mark.parametrize(
    "conflict",
    [
        ("--repo-path", "."),
        ("--checkpoint-db", "state.sqlite"),
        ("--pricing-file", "pricing.json"),
        ("--enable-lifecycle-mutations",),
        ("--enable-memory-mutations",),
    ],
)
def test_runtime_demo_rejects_real_workspace_and_mutation_options(
    conflict: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as rejected:
        runtime.main(["--demo", *conflict])

    assert rejected.value.code == 2


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
