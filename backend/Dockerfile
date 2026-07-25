# syntax=docker/dockerfile:1
#
# Build context MUST be the repository root (see docker-compose.yml / the
# `docker build -f backend/Dockerfile .` invocation in the README) --
# `backend` depends on the `sentinelml` package in `ml/` via a uv workspace
# source, so the builder stage needs the whole workspace (root
# pyproject.toml + uv.lock + ml/ + backend/), not just this directory.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.32

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- builder --------------------------------------------------------------
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install third-party dependencies first, before copying source, so this
# (slow) layer is cached across source-only changes.
COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml backend/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-packages --no-install-workspace

# Now install the workspace packages themselves (ml/ + backend/app/).
COPY ml/ ml/
COPY backend/ backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-packages

# ---- runtime ----------------------------------------------------------
FROM base AS runtime

# lightgbm's compiled backend needs libgomp (the OpenMP runtime) on Linux,
# same as it needs libomp on macOS -- see ml/training/README.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/ml /app/ml
COPY --from=builder --chown=appuser:appuser /app/backend /app/backend

ENV PATH="/app/.venv/bin:${PATH}"
WORKDIR /app/backend
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1

# Model artifacts (ml/models/artifacts/) are not baked into the image --
# they're gitignored, training-run-specific, and often large. Mount them
# (docker-compose.yml does this for local dev) or bake a custom image on
# top of this one for a real deployment; see the README's deployment
# section. Without them the service still starts -- /health is 200,
# /ready is 503 until artifacts are present.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
