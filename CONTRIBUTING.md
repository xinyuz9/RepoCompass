# Contributing

Keep changes focused, explain behavior, and add regression coverage. Discuss large architectural
changes first. Execution authority belongs in code; provider-specific behavior stays in adapters.

## Local checks

Linux/WSL, Python 3.12+, uv, Git, ripgrep, Node.js 24:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=repocompass --cov-report=term --cov-fail-under=85 -q
uv build
cd ui
npm ci
npm test
npm run test:recording
npm run lint
npm run build
```

Commit the rebuilt SPA in `src/repocompass/dashboard/static/`, including license output,
when changing the frontend. Python installations then work without Node.

## Browser checks

```bash
cd ui
npx playwright install --with-deps --only-shell chromium
npm run test:e2e
npm run test:e2e:cancellation
npm run test:e2e:recovery
npx playwright test --config playwright.modes.config.ts
```

These start scripted servers on dedicated ports. Optional backend browser tests require
bubblewrap, `uv sync --locked --extra browser`, and a Chromium installation:

```bash
REPOCOMPASS_BROWSER_TESTS=1 uv run --extra browser pytest -q tests/integration/test_static_browser.py
```

## Container checks

```bash
docker compose build
REPOCOMPASS_RUN_DOCKER_TESTS=1 uv run pytest -q tests/integration/test_docker_image.py
```

## Review checklist

- Cover failure paths as well as the happy path.
- Preserve path, cost, duplicate-call, evidence, and lifecycle identity constraints.
- Use fixtures in ordinary tests and CI; never require provider keys or paid calls.
- Keep `.env`, databases, raw responses, and private source out of commits.
- Distinguish test coverage from model-quality measurements.
- Update public documentation when commands or contracts change.

Original code is MIT-licensed; preserve dependency notices when redistributing.
