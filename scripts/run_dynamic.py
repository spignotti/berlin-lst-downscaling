# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Dynamic scene pipeline entry point — Hydra-driven.

Usage
-----
    # Local smoke test (requires local static smoke products + CDS access)
    uv run python scripts/run_dynamic.py --config-name smoke \
        manifest_uri=data/ard/manifests/v3/.../manifest.parquet

    # Full run on VM
    uv run python scripts/run_dynamic.py --config-name full \
        manifest_uri=gs://berlin-lst-data/manifests/v3/.../manifest.parquet \
        output_root=gs://berlin-lst-data/dynamic/full
"""
from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.dynamic.pipeline import run_dynamic
from berlin_lst_downscaling.data.io import RunLogSession


@hydra.main(config_path="../configs/dynamic", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to dynamic pipeline."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="dynamic", run_id=run_id, level=level):
        return run_dynamic(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
