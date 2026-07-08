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

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run as ard_run


@hydra.main(config_path="../configs/ard", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — dispatch to ard_run."""
    print("=" * 60, flush=True)
    print(f"ARD Pipeline — mode={cfg.mode}", flush=True)
    print(f"  sources      : {list(cfg.sources)}", flush=True)
    print(f"  output_root  : {cfg.output_root}", flush=True)
    print(f"  manifest_uri : {cfg.get('manifest_uri', 'N/A')}", flush=True)
    print(f"  bbox         : {cfg.bbox}", flush=True)
    print("=" * 60, flush=True)

    return ard_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
