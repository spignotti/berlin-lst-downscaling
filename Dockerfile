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
# Run the unified ARD pipeline.  Override Hydra output dir to /tmp.
# Usage:
#   uv run python scripts/run_ard.py --config-name cloud_pilot \
#     manifest_uri=gs://berlin-lst-data/manifests/manifest.parquet \
#     output_root=gs://berlin-lst-data/ard/
ENTRYPOINT ["uv", "run", "python", "scripts/run_ard.py", "hydra.run.dir=/tmp/ard_outputs"]
