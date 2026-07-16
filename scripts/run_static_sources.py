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

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.secondary.source_pipeline import run_sources


@hydra.main(
    config_path="../configs/static_sources",
    config_name="smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to static sources pipeline."""
    return run_sources(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
