FROM ghcr.io/astral-sh/uv:0.11.11 AS uv

FROM python:3.12-slim AS base

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    binutils \
    build-essential \
    ca-certificates \
    curl \
    file \
    git \
    jq \
    less \
    ripgrep \
    strace \
    xxd \
    && rm -rf /var/lib/apt/lists/*

FROM base AS aster-build

# Build from the repository root. Dockerfile.dockerignore selects only the
# framework, workspace manifests, and this example; local data stays outside.
COPY . /opt/aster
WORKDIR /opt/aster
ARG ASTER_VERSION=0.0.0+checkout
RUN uv export --frozen --no-dev --extra tracing --package aster \
        --no-emit-project --no-emit-workspace \
        --output-file /tmp/aster-requirements.txt \
    && UV_DYNAMIC_VERSIONING_BYPASS="$ASTER_VERSION" \
        uv build --no-sources --wheel --package aster --out-dir /tmp/aster-wheels

FROM base

COPY --from=aster-build /tmp/aster-requirements.txt /tmp/aster-requirements.txt
COPY --from=aster-build /tmp/aster-wheels /tmp/aster-wheels
RUN uv pip install --system --require-hashes -r /tmp/aster-requirements.txt \
    && uv pip install --system --no-deps /tmp/aster-wheels/*.whl \
    && rm /tmp/aster-requirements.txt \
    && rm -rf /tmp/uv-cache /tmp/aster-wheels

WORKDIR /app
RUN mkdir -p /logs/artifacts && ln -s /logs/artifacts /app/artifacts
COPY examples/cybergym/aster_cybergym ./aster_cybergym
COPY examples/cybergym/aster_cybergym/llm_config.yaml ./.aster/llm_config.yaml

CMD ["python", "-m", "aster_cybergym.main", "--help"]
