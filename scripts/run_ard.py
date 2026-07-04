# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""ARD pipeline entry point — Hydra-driven, mode=smoke|full.

Usage
-----
    uv run python scripts/run_ard.py --config-name smoke
    uv run python scripts/run_ard.py --config-name smoke scene_date=2024-07-15
    uv run python scripts/run_ard.py --config-name full

"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run as ard_run


@hydra.main(config_path="../configs/ard", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — print config summary, then dispatch."""
    print("=" * 60, flush=True)
    print(f"ARD Pipeline — mode={cfg.mode}", flush=True)
    print(f"  sources      : {cfg.sources}", flush=True)
    print(f"  scene_date   : {cfg.scene_date}", flush=True)
    print(f"  bbox         : {cfg.bbox}", flush=True)
    print(f"  output_root  : {cfg.output_root}", flush=True)
    print(f"  ecostress    : enabled={cfg.ecostress.enabled}", flush=True)
    print(f"  dilation     : {cfg.cloud_dilation_px} px", flush=True)
    print(f"  cloud_base   : {cfg.cloud_base_height_m} m", flush=True)
    print("=" * 60, flush=True)

    return ard_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
