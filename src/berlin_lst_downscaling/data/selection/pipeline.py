"""Selection-pipeline orchestration.

Owns the multi-anchor pipeline that turns a Hydra config into the
canonical v3 manifest bundle. The thin Hydra adapter in
``scripts/build_manifest.py`` is the entry point; this module holds
the work so the heavy lifting lives alongside the selection stages.
"""
from __future__ import annotations

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.selection import (
    build_anchors,
    couple_all,
    match_s2_candidates_with_clear_frac,
    write_bundle,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

_logger = logging.getLogger(__name__)

def run_couple(cfg: DictConfig) -> int:
    """Build the canonical v3 manifest bundle from the configured selection policy.

    Pipeline:
      1. Landsat anchors via PC STAC search + pixel-wise QA_PIXEL gate
      2. Sentinel-2 candidates per anchor with clear_frac
      3. Score + tie-break + drop (coupling rate observed)
      4. ECOSTRESS validation granules (fixed allowlist)
      5. Write manifest.parquet + pairings.parquet + manifest_report.json

    Returns 0 on success, 1 on hard failure (no anchors, missing cutoff).
    """
    log_event(
        _logger,
        logging.INFO,
        "start",
        mode="couple",
        years=list(cfg.years),
        months=list(cfg.months),
        bbox=list(cfg.bbox),
    )

    log_event(_logger, logging.INFO, "searching_anchors")
    anchors, anchor_stats = build_anchors(cfg)
    log_event(
        _logger,
        logging.INFO,
        "anchors_found",
        n_total=anchor_stats["n_total"],
        n_kept=anchor_stats["n_kept"],
        n_dropped=anchor_stats["n_dropped"],
    )
    if not anchors:
        log_event(_logger, logging.ERROR, "no_anchors")
        return 1

    log_event(_logger, logging.INFO, "searching_s2_candidates")
    s2_by_anchor: dict[str, list] = {}
    ckpt_dir = cfg.get("checkpoint_dir") or f"{cfg.output_root}/checkpoints"
    ckpt_path = Path(ckpt_dir) / "couple_checkpoint.pkl"
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                s2_by_anchor = pickle.load(f)  # noqa: S301 — internal checkpoint
            log_event(_logger, logging.INFO, "checkpoint_resumed", n_anchors=len(s2_by_anchor))
        except Exception:
            log_event(_logger, logging.WARNING, "checkpoint_load_failed")
            s2_by_anchor = {}

    def _process_one(anchor: dict) -> tuple[str, list]:
        try:
            l8_items = _resolve_landsat_items(anchor, cfg)
            candidates = match_s2_candidates_with_clear_frac(anchor, l8_items, cfg)
            return anchor["scene_id"], candidates
        except Exception as exc:
            log_event(
                _logger,
                logging.ERROR,
                "anchor_failed",
                scene_id=anchor.get("scene_id", "???"),
                error=str(exc),
                exc_info=True,
            )
            return anchor["scene_id"], []

    todo = [a for a in anchors if a["scene_id"] not in s2_by_anchor]
    n_total = len(anchors)
    done = len(s2_by_anchor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_process_one, a): a for a in todo}
        for fut in as_completed(futures):
            sid, candidates = fut.result()
            s2_by_anchor[sid] = candidates
            done += 1
            if done % 50 == 0 or done == n_total:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_path, "wb") as f:
                    pickle.dump(s2_by_anchor, f)
                log_event(
                    _logger,
                    logging.INFO,
                    "s2_progress",
                    done=done,
                    total=n_total,
                    last_scene=sid,
                )
    ckpt_path.unlink(missing_ok=True)
    log_event(_logger, logging.INFO, "s2_done", n_anchors=len(anchors))

    log_event(_logger, logging.INFO, "coupling")
    coupled, dropped = couple_all(anchors, s2_by_anchor, cfg)
    log_event(
        _logger,
        logging.INFO,
        "coupling_done",
        n_coupled=len(coupled),
        n_dropped=len(dropped),
    )

    log_event(_logger, logging.INFO, "resolving_ecostress")
    eco_granules = _resolve_ecostress_allowlist(cfg)
    log_event(_logger, logging.INFO, "ecostress_resolved", n_granules=len(eco_granules))

    log_event(_logger, logging.INFO, "writing_bundle")
    output_root = str(cfg.output_root)
    manifest_out = f"{output_root.rstrip('/')}/manifest.parquet"
    pairings_out = f"{output_root.rstrip('/')}/pairings.parquet"
    report_out = f"{output_root.rstrip('/')}/manifest_report.json"

    years = list(cfg.get("years", []))
    if years and max(years) >= datetime.now(UTC).year and not cfg.get("cutoff_utc"):
        log_event(
            _logger,
            logging.ERROR,
            "cutoff_required",
            current_year=datetime.now(UTC).year,
        )
        return 1

    result = write_bundle(
        coupled,
        dropped,
        eco_granules,
        manifest_path=manifest_out,
        pairings_path=pairings_out,
        report_path=report_out,
        cutoff_utc=cfg.get("cutoff_utc"),
        cfg=cfg,
    )
    coupling_rate = (
        result.n_coupled / result.n_anchors if result.n_anchors > 0 else 0.0
    )
    log_event(
        _logger,
        logging.INFO,
        "bundle_written",
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
        report_path=result.report_path,
    )
    return 0

def _resolve_landsat_items(anchor: dict, cfg) -> list:
    """Resolve STAC items for one Landsat anchor (± 1 day tolerance)."""
    from datetime import timedelta

    from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog

    cat = get_catalog()
    start = anchor["datetime"] - timedelta(days=1)
    end = anchor["datetime"] + timedelta(days=1)
    search = cat.search(
        collections=[cfg.landsat.collection],
        bbox=tuple(cfg.bbox),
        datetime=f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
    )
    return list(search.items())

def _resolve_ecostress_allowlist(cfg: DictConfig) -> list[dict]:
    """Resolve the fixed ECOSTRESS validation granules from the configured allowlist."""
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
            log_event(_logger, logging.WARNING, "ecostress_datetime_parse_failed", granule_id=gid)
            continue
        mgrs = parse_granule_mgrs(gid)
        granules.append(
            {
                "granule_id": gid,
                "source": "ecostress",
                "year": dt.year,
                "datetime": dt,
                "date": dt.strftime("%Y-%m-%d"),
                "dt_hours": 0.0,
                "mgrs_tile": mgrs,
                "overlap_frac": 1.0,
                "clear_frac": None,
            }
        )
    return granules

__all__ = ["run_couple"]