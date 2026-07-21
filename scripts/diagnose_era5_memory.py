#!/usr/bin/env python3
"""Memory diagnostic for ERA5 scene processing.

Runs one or more ERA5 scenes through the prepare+finalize path and reports
memory usage at each stage. Designed for systemd with memory accounting.

Usage:
    uv run python scripts/diagnose_era5_memory.py \
        --manifest-uri gs://.../manifest.parquet \
        --output-root gs://<unique-diag-root> \
        --scene-ids LC08_..._20200721_...  # or --count 10
"""
from __future__ import annotations

import argparse
import os
import resource
import tempfile
import time
from pathlib import Path

import psutil


def rss_mb() -> float:
    """Current RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def fd_count() -> int:
    """Open file descriptor count."""
    try:
        return psutil.Process(os.getpid()).num_fds()
    except Exception:
        return -1


def log_stage(stage: str, t0: float | None = None) -> None:
    elapsed = f" ({time.perf_counter() - t0:.1f}s)" if t0 else ""
    print(f"[{rss_mb():.1f} MB RSS, {fd_count()} fds] {stage}{elapsed}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--count", type=int, default=1, help="Process N scenes from manifest")
    args = parser.parse_args()

    from berlin_lst_downscaling.common.grid import canon_grid_10m
    from berlin_lst_downscaling.data.dynamic.manifest import load_landsat_anchors
    from berlin_lst_downscaling.data.dynamic.paths import ledger_path, scene_product_dir
    from berlin_lst_downscaling.data.dynamic.era5 import prepare_era5_scene
    from berlin_lst_downscaling.data.secondary.product import finalize_secondary_product
    from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
    from berlin_lst_downscaling.data.secondary.idempotency import reconcile

    log_stage("start")

    # Load manifest
    report = load_landsat_anchors(
        args.manifest_uri,
        scene_ids=args.scene_ids,
    )
    if not report.ok:
        print(f"Manifest load failed: {report.errors}")
        return 1

    scenes = report.scenes[:args.count]
    log_stage(f"manifest loaded: {len(scenes)} scenes to process")

    grid = canon_grid_10m()
    led = SecondaryLedger.open(ledger_path(args.output_root))

    for i, scene in enumerate(scenes):
        log_stage(f"scene {i+1}/{len(scenes)}: {scene.scene_id}")

        era5_item_id = f"era5_land_{scene.scene_id}"
        era5_todo = reconcile([(era5_item_id, "era5_land", scene.scene_id)], led, "diag")

        if not era5_todo:
            log_stage(f"  skipped (already done)")
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_dir = Path(tmp_dir)

            # Prepare
            log_stage(f"  prepare_era5_scene start")
            t0 = time.perf_counter()
            try:
                prepared = prepare_era5_scene(
                    scene.scene_id, scene.acquisition_datetime,
                    args.output_root, "diag", grid=grid, local_dir=local_dir)
                log_stage(f"  prepare_era5_scene done", t0)
            except Exception as e:
                log_stage(f"  prepare FAILED: {e}")
                continue

            # Finalize
            log_stage(f"  finalize start")
            t0 = time.perf_counter()
            try:
                prod_dir = scene_product_dir(args.output_root, "era5_land", scene.scene_id)
                artifacts = finalize_secondary_product(
                    prepared, grid, args.output_root, "diag",
                    product_dir_override=prod_dir)
                log_stage(f"  finalize done", t0)
            except Exception as e:
                log_stage(f"  finalize FAILED: {e}")
                continue

        log_stage(f"  scene {i+1} complete")

    log_stage("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
