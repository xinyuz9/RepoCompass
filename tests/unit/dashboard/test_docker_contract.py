"""Cheap packaging guards; image behavior is covered by opt-in Docker tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_image_uses_locked_builds_and_non_root_runtime():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "npm ci" in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert "COPY --from=frontend" in dockerfile
    runtime = dockerfile.split("AS runtime", 1)[1]
    assert "USER 10001:10001" in runtime
    assert "git ripgrep" in runtime
    assert "safe.directory /workspace/repository" in runtime
    assert "safe.directory '*'" not in runtime
    assert "HEALTHCHECK" in runtime
    assert 'CMD ["--demo", "--container-network"]' in runtime
    assert "COPY . " not in dockerfile
    assert "REPOCOMPASS_API_KEY" not in dockerfile


def test_build_context_is_allowlisted_and_excludes_secrets_and_local_state():
    rules = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rules[0] == "**"
    assert {line for line in rules if line.startswith("!")} == {
        "!Dockerfile",
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!LICENSE",
        "!src/",
        "!src/**",
        "!ui/",
        "!ui/**",
    }
    for rule in (
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/*.key",
        "**/.git",
        "**/.venv",
        "**/node_modules",
        "ui/test-results*",
        "src/repocompass/dashboard/static",
    ):
        assert rule in rules


def test_compose_keeps_demo_inert_and_workspace_explicit():
    base = (ROOT / "compose.yaml").read_text()
    workspace = (ROOT / "compose.workspace.yaml").read_text()
    assert "127.0.0.1:${REPOCOMPASS_PORT:-8767}:8766" in base
    assert "read_only: true" in base
    assert "cap_drop: [ALL]" in base
    assert "no-new-privileges:true" in base
    assert "env_file:" not in base
    assert "enable-lifecycle-mutations" not in base
    assert "--enable-lifecycle-mutations" in workspace
    assert "format: raw" in workspace
    assert workspace.count("create_host_path: false") == 2
    assert workspace.count("read_only: true") == 2
    assert "source: investigation-data" in workspace
    assert "target: /data" in workspace
    assert "docker.sock" not in base + workspace
