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
        --config-name smoke_2024_mai_sep

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

import logging
import pickle
from pathlib import Path

from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.data.selection import (
    build_anchors,
    couple_all,
    match_s2_candidates_with_clear_frac,
    write_bundle,
)
from berlin_lst_downscaling.data.selection.scan import run_scan

_logger = logging.getLogger(__name__)

# ── couple mode ──────────────────────────────────────────────────────────────


def _run_couple(cfg: DictConfig) -> None:
    """Run full pixel-coupled manifest generation."""
    log_event(_logger, logging.INFO, "start", mode="couple",
        years=list(cfg.years), months=list(cfg.months), bbox=list(cfg.bbox))

    # ── 1. Build Landsat anchors ─────────────────────────────────────────────
    log_event(_logger, logging.INFO, "searching_anchors")
    anchors, anchor_stats = build_anchors(cfg)
    log_event(_logger, logging.INFO, "anchors_found",
        n_total=anchor_stats['n_total'],
        n_kept=anchor_stats['n_kept'],
        n_dropped=anchor_stats['n_dropped'])
    if not anchors:
        log_event(_logger, logging.ERROR, "no_anchors")
        raise SystemExit(1)

    # ── 2. Search S2 candidates per anchor + compute clear_frac (parallel) ──
    log_event(_logger, logging.INFO, "searching_s2_candidates")
    s2_by_anchor: dict[str, list] = {}
    ckpt_path = "data/ard/couple_checkpoint.pkl"

    # Load checkpoint if exists (resume from partial run)
    if Path(ckpt_path).exists():
        try:
            with open(ckpt_path, "rb") as f:
                s2_by_anchor = pickle.load(f)  # noqa: S301 — internal checkpoint, not untrusted
            log_event(_logger, logging.INFO, "checkpoint_resumed",
                n_anchors=len(s2_by_anchor))
        except Exception:
            log_event(_logger, logging.WARNING, "checkpoint_load_failed")
            s2_by_anchor = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_one_anchor(anchor: dict) -> tuple[str, list]:
        """Resolve Landsat items + compute S2 candidates for one anchor."""
        try:
            l8_items = _resolve_landsat_items(anchor, cfg)
            candidates = match_s2_candidates_with_clear_frac(anchor, l8_items, cfg)
            return anchor["scene_id"], candidates
        except Exception as exc:
            log_event(_logger, logging.ERROR, "anchor_failed",
                scene_id=anchor.get('scene_id', '???'), error=str(exc),
                exc_info=True)
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
                log_event(_logger, logging.INFO, "s2_progress",
                    done=done_count, total=n_total, last_scene=scene_id)

    # Delete checkpoint on successful completion
    Path(ckpt_path).unlink(missing_ok=True)
    log_event(_logger, logging.INFO, "s2_done", n_anchors=len(anchors))

    # ── 3. Score + Tie-Break + Drop ──────────────────────────────────────────
    log_event(_logger, logging.INFO, "coupling")
    coupled, dropped = couple_all(anchors, s2_by_anchor, cfg)
    log_event(_logger, logging.INFO, "coupling_done",
        n_coupled=len(coupled), n_dropped=len(dropped))

    # ── 4. ECOSTRESS validation granules (fixed allowlist) ──────────────
    log_event(_logger, logging.INFO, "resolving_ecostress")
    eco_granules = _resolve_ecostress_allowlist(cfg)
    log_event(_logger, logging.INFO, "ecostress_resolved",
        n_granules=len(eco_granules))

    # ── 5. Write manifest bundle ────────────────────────────────────────
    log_event(_logger, logging.INFO, "writing_bundle")
    manifest_out = cfg.get("manifest_out", f"{cfg.output_root}/manifest.parquet")
    pairings_out = cfg.get("pairings_out", f"{cfg.output_root}/pairings.parquet")
    report_out = cfg.get("report_out", f"{cfg.output_root}/manifest_report.json")
    cutoff = cfg.get("cutoff_utc") or "2026-07-17T23:59:59Z"

    result = write_bundle(
        coupled, dropped, eco_granules,
        manifest_path=manifest_out,
        pairings_path=pairings_out,
        report_path=report_out,
        cutoff_utc=cutoff,
        cfg=cfg,
    )

    coupling_rate = result.n_coupled / result.n_anchors if result.n_anchors > 0 else 0.0
    log_event(_logger, logging.INFO, "bundle_written",
        n_anchors_total=anchor_stats["n_total"],
        n_anchors_kept_after_pixel_filter=anchor_stats["n_kept"],
        n_anchors_dropped_pixel_filter=anchor_stats["n_dropped"],
        n_anchors=result.n_anchors,
        n_coupled=result.n_coupled,
        n_dropped=result.n_dropped,
        n_ecostress=result.n_ecostress,
        coupling_rate_observed=round(coupling_rate, 4),
        manifest_path=result.manifest_path,
        pairings_path=result.pairings_path,
        report_path=result.report_path)


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


def _resolve_ecostress_allowlist(cfg: DictConfig) -> list[dict]:
    """Resolve the fixed ECOSTRESS validation granules from the allowlist.

    Uses the configured validation_ids to build granule dicts without CMR search.
    """
    from berlin_lst_downscaling.data.acquisition.ecostress import (
        parse_granule_datetime,
        parse_granule_mgrs,
    )
    from berlin_lst_downscaling.data.selection.schema import ECOSTRESS_VALIDATION_IDS

    ids = cfg.get("ecostress", {}).get("validation_ids", ECOSTRESS_VALIDATION_IDS)
    granules: list[dict] = []
    for gid in ids:
        dt = parse_granule_datetime(gid)
        if dt is None:
            log_event(_logger, logging.WARNING, "ecostress_datetime_parse_failed",
                granule_id=gid)
            continue
        mgrs = parse_granule_mgrs(gid)
        granules.append({
            "granule_id": gid,
            "source": "ecostress",
            "year": dt.year,
            "datetime": dt,
            "date": dt.strftime("%Y-%m-%d"),
            "dt_hours": 0.0,
            "mgrs_tile": mgrs,
            "overlap_frac": 1.0,
            "clear_frac": None,
        })
    return granules


# ── scan mode ────────────────────────────────────────────────────────────────


def _run_scan(cfg: DictConfig) -> None:
    """Run metadata-only volume scan."""
    log_event(_logger, logging.INFO, "start", mode="scan",
        years=list(cfg.years), months=list(cfg.months))

    log_event(_logger, logging.INFO, "running_scan")
    report = run_scan(cfg)

    log_event(_logger, logging.INFO, "scan_done",
        n_landsat_total=report.n_landsat_total,
        n_landsat_coupled=report.n_landsat_coupled,
        n_landsat_dropped=report.n_landsat_dropped,
        n_s2_candidates=report.n_s2_candidates,
        n_ecostress_matches=report.n_ecostress_matches,
        est_total_gb=report.est_total_gb)


# ── Hydra main ────────────────────────────────────────────────────────────────


def main(cfg: DictConfig) -> None:
    import logging

    output_root = str(cfg.get("output_root", "data/ard"))
    run_id = __import__("uuid").uuid4().hex[:8]
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)

    with RunLogSession(output_root, pipeline="selection", run_id=run_id, level=level):
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
