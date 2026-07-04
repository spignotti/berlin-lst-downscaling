# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""ARD pipeline entry point — Hydra-driven, mode=smoke|full.

Usage
-----
    uv run python scripts/run_ard.py --config-name default
    uv run python scripts/run_ard.py --config-name smoke
    uv run python scripts/run_ard.py --config-name full
    uv run python scripts/run_ard.py --config-name smoke scene_date=2024-07-15

"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/ard", config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — print resolved config and dispatch."""
    print("=" * 60)
    print(f"ARD Pipeline — mode={cfg.mode}")
    print(f"  sources      : {cfg.sources}")
    print(f"  scene_date   : {cfg.scene_date}")
    print(f"  bbox         : {cfg.bbox}")
    print(f"  target_crs   : {cfg.target_crs}")
    print(f"  output_root  : {cfg.output_root}")
    print(f"  ecostress    : enabled={cfg.ecostress.enabled}")
    print(f"  cloud_dilation_px : {cfg.cloud_dilation_px}")
    print(f"  cloud_base_height_m: {cfg.cloud_base_height_m}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
