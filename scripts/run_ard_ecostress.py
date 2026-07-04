# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""ECOSTRESS ARD pipeline entry point — Hydra-driven, mode=smoke|full.

Usage
-----
    # Smoke (requires fixture — run scripts/download_ecostress_fixture.py first)
    uv run python scripts/run_ard_ecostress.py --config-name smoke

    # Full (requires manifest — run scripts/build_manifest_ecostress.py first)
    uv run python scripts/run_ard_ecostress.py --config-name full

"""

from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.pipeline import run as ard_run

_HYDRA_CONFIG_PATH = "../configs/ard/ecostress"


@hydra.main(config_path=_HYDRA_CONFIG_PATH, config_name="default", version_base=None)
def main(cfg: DictConfig) -> int:
    """Hydra entry point — print config summary, then dispatch."""
    # Assert single-source contract
    if cfg.mode == "smoke" and len(cfg.sources) != 1:
        raise ValueError(f"smoke mode requires exactly 1 source, got {cfg.sources}")
    if cfg.sources != ["ecostress"]:
        raise ValueError(f"run_ard_ecostress.py requires sources=[ecostress], got {cfg.sources}")

    print("=" * 60, flush=True)
    print(f"ARD Pipeline (ECOSTRESS) — mode={cfg.mode}", flush=True)
    print(f"  sources      : {cfg.sources}", flush=True)
    print(f"  scene_date   : {cfg.scene_date}", flush=True)
    print(f"  bbox         : {cfg.bbox}", flush=True)
    print(f"  output_root  : {cfg.output_root}", flush=True)
    print(f"  raw_dir      : {cfg.ecostress.raw_dir}", flush=True)
    print(f"  enabled      : {cfg.ecostress.enabled}", flush=True)
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
