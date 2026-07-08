# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pystac-client",
#     "planetary-computer",
#     "odc-stac>=0.4.0",
#     "pyarrow>=24.0.0",
#     "hydra-core>=1.3.3",
#     "omegaconf",
#     "earthaccess>=0.10.0",
#     "rasterio>=1.4.3",
#     "rioxarray>=0.18.0",
#     "numpy",
#     "pytz",
#     "geopandas",
# ]
# ///

"""
Szenen-Selektion & Kopplung — ARD manifest builder.

Hydra entry point with two modes:
  couple  — full pixel-coupled manifest (slow, produces manifest.parquet)
  scan    — metadata-only volume scan (fast, produces scan_report.{json,md})

Usage
-----
    # Smoke test (1 month, couple mode)
    uv run python scripts/build_manifest.py \
        --config-dir configs/selection \
        --config-name smoke_jul2024

    # Full volume scan (2017–2025)
    uv run python scripts/build_manifest.py \
        --config-dir configs/selection \
        --config-name full_2017_2025

    # Override individual params
    uv run python scripts/build_manifest.py \
        --config-dir configs/selection \
        --config-name default \
        years='[2024]' months='[7]' \
        mode=couple
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from omegaconf import DictConfig

from berlin_lst_downscaling.data.selection import (
    build_anchors,
    couple_all,
    match_s2_candidates_with_clear_frac,
    write_manifest,
)
from berlin_lst_downscaling.data.selection.ecostress_subset import build_ecostress_subset
from berlin_lst_downscaling.data.selection.scan import run_scan

# ── couple mode ──────────────────────────────────────────────────────────────


def _run_couple(cfg: DictConfig) -> None:
    """Run full pixel-coupled manifest generation."""
    # Suppress transient rasterio CPLE warnings (SAS token expiry, Azure
    # blob read errors) — the pipeline handles these via retry + graceful
    # degradation (clear_frac=None on failure, coupling continues).
    import logging
    logging.getLogger("rasterio._err").setLevel(logging.ERROR)
    logging.getLogger("odc.loader._rio").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    print(json.dumps({
        "mode": "couple",
        "years": list(cfg.years),
        "months": list(cfg.months),
        "bbox": list(cfg.bbox),
    }), file=sys.stderr)

    # ── 1. Build Landsat anchors ─────────────────────────────────────────────
    print("  [1/5] Searching Landsat anchors ...", file=sys.stderr)
    anchors, anchor_stats = build_anchors(cfg)
    print(f"  [1/5] Found {anchor_stats['n_total']} anchors, "
          f"kept {anchor_stats['n_kept']} after pixel filter "
          f"({anchor_stats['n_dropped']} dropped)", file=sys.stderr)
    if not anchors:
        print("ERROR: No Landsat anchors found for the configured range.", file=sys.stderr)
        raise SystemExit(1)

    # ── 2. Search S2 candidates per anchor + compute clear_frac (parallel) ──
    print("  [2/5] Searching S2 candidates + computing clear_frac ...", file=sys.stderr)
    s2_by_anchor: dict[str, list] = {}
    ckpt_path = "data/ard/couple_checkpoint.pkl"

    # Load checkpoint if exists (resume from partial run)
    if Path(ckpt_path).exists():
        try:
            with open(ckpt_path, "rb") as f:
                s2_by_anchor = pickle.load(f)  # noqa: S301 — internal checkpoint, not untrusted
            print(f"  [2/5] Resumed from checkpoint: {len(s2_by_anchor)} anchors already done",
                  file=sys.stderr)
        except Exception:
            print("  [2/5] Checkpoint load failed — starting fresh", file=sys.stderr)
            s2_by_anchor = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_one_anchor(anchor: dict) -> tuple[str, list]:
        """Resolve Landsat items + compute S2 candidates for one anchor."""
        try:
            l8_items = _resolve_landsat_items(anchor, cfg)
            candidates = match_s2_candidates_with_clear_frac(anchor, l8_items, cfg)
            return anchor["scene_id"], candidates
        except Exception as exc:
            print(f"  [2/5] ERROR anchor {anchor.get('scene_id', '???')}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return anchor["scene_id"], []

    # Filter anchors already processed in a previous run
    todo_anchors = [a for a in anchors if a["scene_id"] not in s2_by_anchor]
    n_total = len(anchors)
    done_count = len(s2_by_anchor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_process_one_anchor, a): a for a in todo_anchors}
        for future in as_completed(futures):
            scene_id, candidates = future.result()
            s2_by_anchor[scene_id] = candidates
            done_count += 1
            # Save checkpoint every 50 anchors
            if done_count % 50 == 0 or done_count == n_total:
                with open(ckpt_path, "wb") as f:
                    pickle.dump(s2_by_anchor, f)
                print(f"  [2/5] Progress: {done_count}/{n_total} anchors processed"
                      f" (last: {scene_id}) — checkpoint saved", file=sys.stderr)

    # Delete checkpoint on successful completion
    Path(ckpt_path).unlink(missing_ok=True)
    print(f"  [2/5] Done — processed {len(anchors)} anchors", file=sys.stderr)

    # ── 3. Score + Tie-Break + Drop ──────────────────────────────────────────
    print("  [3/5] Scoring and coupling ...", file=sys.stderr)
    coupled, dropped = couple_all(anchors, s2_by_anchor, cfg)
    print(f"  [3/5] Coupled: {len(coupled)}, Dropped: {len(dropped)}", file=sys.stderr)

    # ── 4. ECOSTRESS subset ─────────────────────────────────────────────────
    print("  [4/5] Building ECOSTRESS subset ...", file=sys.stderr)
    ecostress_by_anchor = build_ecostress_subset(coupled, cfg)
    n_eco = sum(len(v) for v in ecostress_by_anchor.values())
    print(f"  [4/5] ECOSTRESS matches: {n_eco}", file=sys.stderr)

    # ── 5. Write manifest ────────────────────────────────────────────────────
    print("  [5/5] Writing manifest ...", file=sys.stderr)
    manifest_out = cfg.get("manifest_out", f"{cfg.output_root}/manifest.parquet")
    result = write_manifest(coupled, dropped, ecostress_by_anchor, manifest_out)

    coupling_rate = result.n_coupled / result.n_anchors if result.n_anchors > 0 else 0.0
    print(json.dumps({
        "event": "manifest_done",
        "n_anchors_total": anchor_stats["n_total"],
        "n_anchors_kept_after_pixel_filter": anchor_stats["n_kept"],
        "n_anchors_dropped_pixel_filter": anchor_stats["n_dropped"],
        "n_anchors": result.n_anchors,
        "n_coupled": result.n_coupled,
        "n_dropped": result.n_dropped,
        "n_ecostress": result.n_ecostress,
        "coupling_rate_observed": round(coupling_rate, 4),
        "manifest_path": result.manifest_path,
    }), file=sys.stderr)

    print(f"  [OK] Manifest written: {result.manifest_path}", file=sys.stderr)
    print(f"       Anchors: {result.n_anchors} | Coupled: {result.n_coupled} | "
          f"Dropped: {result.n_dropped} | ECOSTRESS: {result.n_ecostress}")


def _resolve_landsat_items(anchor: dict, cfg) -> list:
    """Resolve STAC items for one Landsat anchor by date (± 1 day tolerance)."""
    from datetime import timedelta

    from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog

    cat = get_catalog()
    day_start = anchor["datetime"] - timedelta(days=1)
    day_end = anchor["datetime"] + timedelta(days=1)

    search = cat.search(
        collections=[cfg.landsat.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{day_start.strftime('%Y-%m-%d')}/{day_end.strftime('%Y-%m-%d')}",
        # scene-level cloud_cover filter removed — resolved anchor has already
        # passed the pixel-wise QA_PIXEL ∩ AOI gate in build_anchors.
    )
    return list(search.items())


# ── scan mode ────────────────────────────────────────────────────────────────


def _run_scan(cfg: DictConfig) -> None:
    """Run metadata-only volume scan."""
    print(json.dumps({
        "mode": "scan",
        "years": list(cfg.years),
        "months": list(cfg.months),
    }), file=sys.stderr)

    print("  [scan] Running metadata-only volume scan ...", file=sys.stderr)
    report = run_scan(cfg)

    print(json.dumps({
        "event": "scan_done",
        "n_landsat_total": report.n_landsat_total,
        "n_landsat_coupled": report.n_landsat_coupled,
        "n_landsat_dropped": report.n_landsat_dropped,
        "n_s2_candidates": report.n_s2_candidates,
        "n_ecostress_matches": report.n_ecostress_matches,
        "est_total_gb": report.est_total_gb,
    }), file=sys.stderr)

    print(f"  [OK] Scan report: {report.metadata_json}", file=sys.stderr)
    print(f"       Anchors: {report.n_landsat_total} | Coupled: {report.n_landsat_coupled} "
          f"| Dropped: {report.n_landsat_dropped} | ECOSTRESS: {report.n_ecostress_matches}")
    print(f"       Volume: {report.est_total_gb:.1f} GB total")


# ── Hydra main ────────────────────────────────────────────────────────────────


def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "couple")
    if mode == "scan":
        _run_scan(cfg)
    else:
        _run_couple(cfg)


if __name__ == "__main__":
    import hydra

    # Run with Hydra
    @hydra.main(
        config_path="../configs/selection",
        config_name="default",
        version_base=None,
    )
    def _hydra_main(cfg: DictConfig) -> None:
        main(cfg)

    _hydra_main()
