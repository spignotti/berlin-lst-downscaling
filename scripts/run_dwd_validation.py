# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""DWD-vs-ERA5 validation entrypoint — Hydra-driven.

Reads published dynamic-pipeline outputs (COG provenance + ledger),
acquires DWD hourly 2 m air-temperature observations for stations
inside the Berlin AOI, joins them at every Landsat anchor's
normalised UTC hour, and persists a reproducible comparison table
and QA report. DWD is validation-only — never fed into training,
normalisation, or the downstream model.

Usage
-----
    #   manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet
    uv run python scripts/run_dwd_validation.py --config-name default \
        dynamic_full_root=gs://berlin-lst-data/dynamic/full/<run-id> \
        dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026/<run-id>
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.dynamic.dwd_validation import run_dwd_validation
from berlin_lst_downscaling.data.io import RunLogSession

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/dwd_validation", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to DWD-vs-ERA5 validation."""
    manifest_uri = cfg.get("manifest_uri")
    if not manifest_uri:
        raise SystemExit(
            "manifest_uri is required — provide the published bundle, e.g.\n"
            "  manifest_uri=gs://berlin-lst-data/manifests/v3/...-r2/manifest.parquet"
        )
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="dwd_validation", run_id=run_id, level=level):
        return run_dwd_validation(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
