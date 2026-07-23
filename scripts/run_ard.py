# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Unified ARD pipeline entry point — Hydra-driven, mode=full only.

Usage
-----
    # Smoke test (manifest-driven, all sources)
    uv run python scripts/run_ard.py --config-name smoke_primary \
        manifest_uri=data/smoke/primary/manifest.parquet

    # Production
    #   manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet
    uv run python scripts/run_ard.py --config-name full_all
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run
from berlin_lst_downscaling.data.io import RunLogSession, log_event

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/ard", config_name="full_all", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to the ARD pipeline."""
    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri:
        raise SystemExit(
            "manifest_uri is required — provide the published bundle, e.g.\n"
            "  manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet"
        )
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="ard", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            mode=cfg.mode,
            sources=list(cfg.sources),
            output_root=output_root,
            manifest_uri=manifest_uri,
            bbox=list(cfg.bbox),
        )
        return run(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
