# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Sentinel-2 ARD pipeline entry point — Hydra-driven, mode=smoke|full.

Usage
-----
    uv run python scripts/run_ard_sentinel2.py --config-name smoke
    uv run python scripts/run_ard_sentinel2.py --config-name smoke scene_date=2024-06-29 viz=false
    uv run python scripts/run_ard_sentinel2.py --config-name full

"""

from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run as ard_run

_HYDRA_CONFIG_PATH = "../configs/ard/sentinel2"


@hydra.main(config_path=_HYDRA_CONFIG_PATH, config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — print config summary, then dispatch."""
    # Assert single-source contract
    if cfg.mode == "smoke" and len(cfg.sources) != 1:
        raise ValueError(f"smoke mode requires exactly 1 source, got {cfg.sources}")
    if cfg.sources != ["sentinel-2-l2a"]:
        raise ValueError(
            f"run_ard_sentinel2.py requires sources=[sentinel-2-l2a], "
            f"got {cfg.sources}"
        )

    print("=" * 60, flush=True)
    print(f"ARD Pipeline (Sentinel-2) — mode={cfg.mode}", flush=True)
    print(f"  sources      : {cfg.sources}", flush=True)
    print(f"  scene_date   : {cfg.scene_date}", flush=True)
    print(f"  bbox         : {cfg.bbox}", flush=True)
    print(f"  output_root  : {cfg.output_root}", flush=True)
    print(f"  s2cloudless : {cfg.s2cloudless_threshold}", flush=True)
    print(f"  cloud_base   : {cfg.cloud_base_height_m} m", flush=True)
    print("=" * 60, flush=True)

    result = ard_run(cfg)

    # Post-run visualization (smoke mode only)
    if cfg.get("viz", False) and cfg.mode == "smoke":
        from pathlib import Path  # noqa: F401

        viz_script = Path(__file__).parent / "visualize_smoke.py"
        print("\n[+] Running smoke visualization ...", flush=True)
        res = subprocess.run(  # noqa: S603 — script path is hard-coded, not user input
            [sys.executable, viz_script, str(cfg.output_root)],
            capture_output=True,
            text=True,
        )
        if res.stdout:
            print(res.stdout, end="", flush=True)
        if res.stderr:
            print(res.stderr, end="", file=sys.stderr, flush=True)
        if res.returncode != 0:
            print(f"[!] Visualization exited with code {res.returncode}", flush=True)

    return result


if __name__ == "__main__":
    raise SystemExit(main())
