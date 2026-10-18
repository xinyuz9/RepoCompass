# Docker deployment

The multi-stage image builds React and installs the locked Python environment. Runtime tools
include Git and ripgrep. The process runs as non-root UID/GID 10001.

## Demo

Use Docker Engine/Desktop and Compose 2.30+. In WSL, enable the distribution in Docker
Desktop's WSL Integration settings. `docker version` must show client and server.

```bash
docker compose up --build -d
docker compose ps
```

Open <http://127.0.0.1:8767/>. Scripted models exercise the real workflow without credentials
or provider calls. Demo state is temporary. Stop with `docker compose down`.
Set `REPOCOMPASS_PORT=8877` to change the host port.

## Real repository

Select a readable clean Git checkout without secrets. A read-only mount prevents writes,
not disclosure. Keep provider configuration and reviewed pricing outside the checkout.

```bash
export REPOCOMPASS_REPOSITORY=/absolute/path/to/clean-checkout
export REPOCOMPASS_ENV_FILE=/absolute/path/to/private/provider.env
export REPOCOMPASS_PRICING_FILE=/absolute/path/to/reviewed-pricing.json
docker compose -f compose.yaml -f compose.workspace.yaml up --build -d
```

See [Configuration](getting-started.md) for provider variables. The env file uses raw
`NAME=value` entries without enclosing quotes. It is injected at runtime, not built into the
image. Startup does not submit a paid task; select the goal, mode, and cap in the UI.

Repository and pricing mounts are read-only. A named volume preserves investigation stores
under `/data/.repocompass/` across recreation. Do not use `down -v` unless deleting that data
is intended. Use a separate project (`-p another-workspace`) for a different repository.
Container and native histories have different identities; do not merge their databases.

## Boundaries

Compose publishes only to host loopback, uses a read-only root filesystem, drops capabilities,
prevents privilege escalation, and limits memory, processes, and writable `/tmp`. No Docker
socket or home/SSH directory is mounted. Do not expose this unauthenticated API to untrusted networks.

Chromium and bubblewrap are not in the base image; Docker packaging does not enable optional
browser/sandbox tools. Docker administrators can read runtime environment variables. Avoid
sharing full inspection/configuration output; use `docker compose ... config --quiet` to validate.
Dependency lockfiles are used, but base-image series tags and apt packages are not digest-pinned.

## Test

```bash
docker compose config --quiet
docker compose build
REPOCOMPASS_RUN_DOCKER_TESTS=1 uv run pytest -q tests/integration/test_docker_image.py
```

Tests use temporary repositories and volumes, no keys. They cover UI assets, Host validation,
non-root execution, read-only mounts, and SQLite persistence across recreation. Readiness waits
through transient transport errors with a fixed deadline.
