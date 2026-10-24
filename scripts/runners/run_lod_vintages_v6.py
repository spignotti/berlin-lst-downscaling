# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Historical LoD vintages entry point — backfill into the v6 roots.

Usage
-----
    uv run python scripts/runners/run_lod_vintages_v6.py --config-name v6
    uv run python scripts/runners/run_lod_vintages_v6.py --config-name v6 \
        source_root=gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2 \
        derived_root=gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2
"""

from __future__ import annotations

import logging
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.data.secondary.lod_vintages_v6 import run_lod_vintages_v6


@hydra.main(
    config_path="../../configs/lod_vintages",
    config_name="v6",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to the historical-vintage pipeline."""
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(source_root, pipeline="lod-vintages-v6", run_id=run_id, level=level):
        return run_lod_vintages_v6(cfg, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
