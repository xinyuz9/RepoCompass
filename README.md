# RepoCompass

**Understand a codebase with source-backed answers, resumable investigations, and visible costs.**

[![Quality](https://github.com/xinyuz9/RepoCompass/actions/workflows/ci.yml/badge.svg)](https://github.com/xinyuz9/RepoCompass/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RepoCompass is a local-first repository investigation agent built with Python, LangGraph,
FastAPI, and React. Give it a repository and a question; it finds relevant files, follows
code relationships, and returns an answer with inspectable source references.

Use it to trace a request through unfamiliar code, locate a verification gate, or build a
reading path before making a change. Chat is the entry point; Path, Plan, Memory, Trace,
and Cost explain how the answer was produced.

![RepoCompass chat workbench showing a source-backed investigation result](docs/assets/workbench.png)

*The included demo: scripted model responses, real workflow and storage.*

## Try it in Docker

```bash
git clone https://github.com/xinyuz9/RepoCompass.git
cd RepoCompass
docker compose up --build -d
```

Open **[localhost:8767](http://127.0.0.1:8767/)**. This free, read-only demo uses scripted
model responses with real tools, workflow, and storage. It needs no key and makes no provider
calls. It demonstrates execution and the interface, not model quality.

Stop with `docker compose down`. See [Docker deployment](docs/docker.md) for read-only
repository mounts and persistent investigation data.

## Run locally

Linux or WSL, Python 3.12+, [uv](https://docs.astral.sh/uv/), Git, and ripgrep:

```bash
uv sync --locked
uv run repocompass-dashboard --demo
```

Open [localhost:8766](http://127.0.0.1:8766/). The production UI is included; Node.js is
only needed when developing or rebuilding the frontend.

For your own repository, configure an OpenAI or DeepSeek model and reviewed pricing:

```bash
uv run repocompass-dashboard \
  --repo-path /absolute/path/to/repository \
  --pricing-file /absolute/path/to/pricing.json \
  --enable-lifecycle-mutations
```

Declare the goal in Chat. [Configuration and usage](docs/getting-started.md) covers keys,
pricing, task settings, follow-ups, and recovery. Real investigations can incur provider charges.

## Capabilities

| Capability | Behavior |
|---|---|
| Quick lookup | Bounded exploration and source validation for focused questions |
| Detailed investigation | Planning, structural exploration, semantic review, and controlled revision |
| Evidence and reading paths | File/line references and bounded Python symbol/import relationships |
| Memory | Opt-in repository facts and episodes with provenance and freshness checks |
| Recovery and cancellation | SQLite checkpoints, idempotent commands, deadlines, cooperative cancellation |
| Observability | Per-call traces, selected context, usage, and pricing-backed cost estimates |
| Interfaces | Chat UI, CLI, and MCP tools sharing the same execution service |

## Architecture

```text
Chat UI / CLI / MCP
        |
Repository-bound lifecycle service
        |
LangGraph investigation --> candidate --> evidence checks --> result
        |                                  |
Bounded tool gateway               optional semantic review
        |                                  |
Source snapshots                continue / revise / human review
        |
SQLite checkpoints + memory + trace / usage records
```

The model proposes actions; code enforces paths, budgets, duplicate-call checks, and stopping
conditions. Results retain the identity of their source snapshot. See [Architecture](docs/architecture.md).

## Scope and safety

- Investigation is read-only. The local operator selects the repository root.
- The dashboard is single-user and has no remote authentication. Keep it on loopback.
- Valid references and an accepted workflow do not prove an explanation correct; inspect the evidence.
- Structural analysis targets Python. Static-page inspection and focused improvement are separate
  opt-in capabilities, not unrestricted browsing or shell access.
- Provider calls can send selected source code outside the machine. Local stores and traces may
  contain that code. Cost reports are estimates, not invoices.

[Security](SECURITY.md) · [Advanced tools](docs/advanced.md)

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cd ui
npm ci
npm test
npm run lint
npm run build
```

Ordinary tests use local fixtures. Paid tests require explicit opt-in. Browser and container
setup is documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Bundled dependencies and fonts retain their licenses; see [notices](NOTICE.md).
