# Advanced interfaces

## MCP

```bash
uv run repocompass-mcp --repo-path /absolute/path/to/repository \
  --pricing-file /absolute/path/to/pricing.json
```

The default is stdio. Configure your MCP client with an absolute `uv` executable and a fixed
working directory for persistent stores. Seven high-level tools share the investigation service;
clients cannot select arbitrary roots or invoke internal primitives. Exact request retries are
idempotent. `--transport streamable-http` is a trusted-local, loopback alternative, not a
remotely authenticated service. Paid investigations retain the same budget implications.

## Static HTML

On Linux/WSL with bubblewrap:

```bash
uv sync --locked --extra browser
uv run --extra browser playwright install --only-shell chromium
uv run --extra browser repocompass-dashboard \
  --repo-path /absolute/path/to/repository \
  --browser-page help=http://127.0.0.1:8000/help.html
```

The operator provides bounded self-owned pages from one loopback origin. Isolated Chromium
renders supplied static content without general network access. This is not arbitrary URL,
login, or browser-action automation. Interactive investigations also need pricing and lifecycle opt-in.

## Focused improvement

`repocompass-improve` is separate from read-only investigation. It requires a clean Git snapshot,
explicit Python file/symbol scope, full replacement proposal, and fixed verification profile.
A no-network bubblewrap sandbox verifies proposals. Runs default to dry-run; applying requires
the exact accepted proposal ID. It does not commit or push.

```bash
uv run repocompass-improve scope --help
uv run repocompass-improve proposal --help
uv run repocompass-improve run --help
```

## Evaluation utilities

`evaluation/` and `benchmarks/` provide fixture generators, localization metrics, context/memory
comparisons, and opt-in live harnesses. Example outputs are test data, not aggregate performance
claims. Live runs require cost acknowledgement, compatible pricing, a budget, and a new output
directory. Ordinary tests and Docker startup make no paid requests.
