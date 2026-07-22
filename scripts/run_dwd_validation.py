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
    # Local validation against the published v3 manifest + dynamic roots
    uv run python scripts/run_dwd_validation.py --config-name default \
        manifest_uri=data/ard/manifests/v3/.../manifest.parquet \
        dynamic_full_root=gs://berlin-lst-data/dynamic/full \
        dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026
"""
from __future__ import annotations

import os

# decision: strip env vars wetterdienst pydantic-settings rejects before
# importing it (see berlin_lst_downscaling.data.dynamic.dwd for context).
for _key in (
    "google_application_credentials",
    "wandb_api_key",
    "earthdata_token",
):
    os.environ.pop(_key, None)

import logging  # noqa: E402
from uuid import uuid4  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from berlin_lst_downscaling.data.dynamic.dwd_validation import run_dwd_validation  # noqa: E402
from berlin_lst_downscaling.data.io import RunLogSession  # noqa: E402


@hydra.main(config_path="../configs/dwd_validation", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to DWD-vs-ERA5 validation."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="dwd_validation", run_id=run_id, level=level):
        return run_dwd_validation(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
