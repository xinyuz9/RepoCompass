# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NODE_IMAGE=node:24-bookworm-slim

FROM ${NODE_IMAGE} AS frontend
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python
COPY src/ ./src/
COPY --from=frontend /build/src/repocompass/dashboard/static/ ./src/repocompass/dashboard/static/
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 repocompass \
    && useradd --uid 10001 --gid repocompass --create-home --shell /usr/sbin/nologin repocompass \
    && mkdir /data \
    && chown repocompass:repocompass /data \
    && git config --system --add safe.directory /workspace/repository
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANGSMITH_TRACING=false \
    LANGGRAPH_CLI_NO_ANALYTICS=1
WORKDIR /data
USER 10001:10001
EXPOSE 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/api/v1/system', timeout=3).close()"]
ENTRYPOINT ["repocompass-dashboard"]
CMD ["--demo", "--container-network"]
