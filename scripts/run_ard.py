# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Unified ARD pipeline entry point — Hydra-driven, mode=full only.

Replaces ``run_ard_landsat.py``, ``run_ard_sentinel2.py``,
``run_ard_ecostress.py``.

Usage
-----
    # Smoke test (manifest-driven, all sources)
    uv run python scripts/run_ard.py --config-name smoke_primary \
        manifest_uri=data/smoke/primary/manifest.parquet

    # Single-source run (e.g. Landsat full)
    uv run python scripts/run_ard.py --config-name landsat/default \
        mode=full manifest_uri=data/ard/manifest.parquet
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run as ard_run
from berlin_lst_downscaling.data.io import RunLogSession, log_event

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/ard", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to ard_run."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)

    with RunLogSession(output_root, pipeline="ard", run_id=run_id):
        log_event(_logger, logging.INFO, "config",
            mode=cfg.mode,
            sources=list(cfg.sources),
            output_root=output_root,
            manifest_uri=cfg.get("manifest_uri", "N/A"),
            bbox=list(cfg.bbox),
        )
        return ard_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
