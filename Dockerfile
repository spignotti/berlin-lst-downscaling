FROM python:3.12-slim AS base

# ── System dependencies (GDAL for rasterio) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (fast Python package manager) ──
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Application code ──
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# ── Entrypoint ──
# Override Hydra output dir to /tmp (avoid writing to container filesystem)
ENTRYPOINT ["uv", "run", "python", "scripts/ard_process.py", "hydra.run.dir=/tmp/ard_outputs"]
