# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Pipeline A entry point — static source product acquisition.

Usage
-----
    uv run python scripts/run_static_sources.py --config-name smoke
    uv run python scripts/run_static_sources.py --config-name full \
        source_root=gs://berlin-lst-data/static/sources/full
"""
from __future__ import annotations

from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession
from berlin_lst_downscaling.data.secondary.source_pipeline import run_sources


@hydra.main(
    config_path="../configs/static_sources",
    config_name="smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to static sources pipeline."""
    run_id = uuid4().hex[:8]
    source_root = str(cfg.source_root)

    with RunLogSession(source_root, pipeline="static-sources", run_id=run_id):
        return run_sources(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
