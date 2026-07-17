# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Secondary-data pipeline entry point — Hydra-driven.

Usage
-----
    # Local fixture (validates pipeline lifecycle)
    uv run python scripts/run_secondary.py --config-name fixture

    # Cloud fixture (requires ADC / Workload Identity)
    uv run python scripts/run_secondary.py --config-name fixture \
        output_root=gs://berlin-lst-data/secondary/smoke/my_run

    # Full run — imperviousness (both vintages)
    uv run python scripts/run_secondary.py --config-name imperviousness \
        output_root=gs://berlin-lst-data/secondary/full_20260714

    # Full run — vegetation height (2020 only)
    uv run python scripts/run_secondary.py --config-name vegetation_height \
        output_root=gs://berlin-lst-data/secondary/full_20260714

    # Cloud smoke — vegetation height
    uv run nox -s cloud-secondary-vegetation-height
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.data.secondary.pipeline import run as secondary_run


@hydra.main(config_path="../configs/secondary", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to secondary pipeline."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="secondary", run_id=run_id, level=level):
        return secondary_run(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
