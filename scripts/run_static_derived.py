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

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.secondary.derived_pipeline import run_derived


@hydra.main(
    config_path="../configs/static_derived",
    config_name="smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to derived geometry pipeline."""
    return run_derived(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
