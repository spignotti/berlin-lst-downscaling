"""QA report — ledger summary + file existence checks."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from omegaconf import DictConfig

from berlin_lst_downscaling.data.ard.ledger import Ledger
from berlin_lst_downscaling.data.io import exists


def qa_report(
    ledger: Ledger,
    cfg: DictConfig,
    run_id: str,
) -> dict[str, Any]:
    """Generate a QA report from the ledger state and COG/STAC file checks.

    Returns a dict with per-source counts and overall status.
    The report is emitted as a structured log line (event ``qa_report``)
    by the pipeline; no JSON file is written.
    """
    sources: list[str] = list(cfg.get("sources", []))

    per_source: dict[str, dict[str, Any]] = {}
    for src in sources:
        counts = ledger.status_counts(src)
        rows = ledger.scenes_for_source(src)

        # Check file existence for done scenes (local or GCS URI)
        cog_ok = 0
        stac_ok = 0
        cog_missing = 0
        stac_missing = 0
        for r in rows:
            if r.status != "done":
                continue
            if r.path_cog and exists(r.path_cog):
                cog_ok += 1
            else:
                cog_missing += 1
            if r.path_stac and exists(r.path_stac):
                stac_ok += 1
            else:
                stac_missing += 1

        # Flag COG existence
        flag_missing = 0
        for r in rows:
            if r.status != "done" or not r.path_cog:
                continue
            # Path-based suffix replacement breaks for gs:// URIs (Path strips double slash).
            # Flag path always follows deterministic naming: cog.tif → cog.flag.tif
            flag_uri = r.path_cog.replace(".tif", ".flag.tif")
            if not exists(flag_uri):
                flag_missing += 1

        # AOI aggregate metrics (schema v4 — only for done scenes with metrics)
        done_with_aoi = [r for r in rows if r.status == "done" and r.aoi_clear_frac is not None]
        aoi_stats: dict[str, Any] = {}
        if done_with_aoi:
            fracs: list[float] = [r.aoi_clear_frac for r in done_with_aoi]  # type: ignore[list-item]
            overlaps: list[int] = [  # type: ignore[assignment]
                r.aoi_overlap_px for r in done_with_aoi if r.aoi_overlap_px is not None
            ]
            aoi_stats = {
                "aoi_scenes": len(fracs),
                "aoi_mean_clear_frac": round(sum(fracs) / len(fracs), 4),
                "aoi_min_clear_frac": round(min(fracs), 4),
                "aoi_max_clear_frac": round(max(fracs), 4),
                "aoi_total_overlap_px": sum(overlaps) if overlaps else None,
            }

        per_source[src] = {
            "total": len(rows),
            **counts,
            "cog_exists": cog_ok,
            "cog_missing": cog_missing,
            "flag_missing": flag_missing,
            "stac_exists": stac_ok,
            "stac_missing": stac_missing,
            **aoi_stats,
        }

    failed = sum(
        per_source[s].get("failed", 0) + per_source[s].get("cog_missing", 0)
        for s in (sources or per_source)
        if s in per_source
    )

    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "per_source": per_source,
        "total_failed": failed,
        "success": failed == 0,
    }

    return report


__all__ = [
    "qa_report",
]
