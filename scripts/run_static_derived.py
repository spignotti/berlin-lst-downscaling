# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Pipeline B entry point — derived geometry product computation.

Usage
-----
    uv run python scripts/run_static_derived.py --config-name smoke
    uv run python scripts/run_static_derived.py --config-name full \
        source_root=gs://berlin-lst-data/static/sources/full \
        derived_root=gs://berlin-lst-data/static/derived/full
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.data.secondary.derived_pipeline import run_derived


@hydra.main(
    config_path="../configs/static_derived",
    config_name="smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to derived geometry pipeline."""
    run_id = uuid4().hex[:8]
    derived_root = str(cfg.derived_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(derived_root, pipeline="static-derived", run_id=run_id, level=level):
        return run_derived(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
