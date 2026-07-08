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
COPY data/boundaries/ data/boundaries/   # AOI masks (aoi_10m.tif, aoi_100m.tif)

# ── Entrypoint ──
# Usage examples:
#   docker run <image> --config-name smoke_primary \
#     manifest_uri=gs://bucket/manifest.parquet \
#     output_root=gs://bucket/ard/
#
#   docker run <image> --config-name smoke_primary \
#     manifest_uri=gs://bucket/manifest.parquet \
#     output_root=gs://bucket/ard/ ecostress.stage_base=gs://bucket/_staging/ecostress
ENTRYPOINT ["uv", "run", "python", "scripts/run_ard.py", "hydra.run.dir=/tmp/ard_outputs"]
