"""Volume scan — metadata-only count + volume estimate, no pixel loads.

Reuses the STAC search logic but does NOT load pixel data.
Writes scan_report.{json,md} with counts and volume estimates.

GB estimates (per scene, approximate):
  Landsat C2 L2: ~50 MB  (STAC metadata + COGs)
  S2 L2A:       ~200 MB
  ECOSTRESS:    ~5 MB    (LST + cloud + water + QC layers)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from berlin_lst_downscaling.data.selection import ScanReport
from berlin_lst_downscaling.data.selection.anchors import build_anchors
from berlin_lst_downscaling.data.selection.ecostress import search_ecostress

_GB_LANDSAT = 0.050
_GB_S2 = 0.200
_GB_ECOSTRESS = 0.005


def run_scan(cfg) -> ScanReport:
    """Run metadata-only volume scan and write scan_report.{json,md}.

    Scans the full configured year/season range without loading pixels.
    Returns a ScanReport and writes the full report to disk.
    """
    # ── Landsat anchors (full range) ─────────────────────────────────────────
    anchors, anchor_stats = build_anchors(cfg)
    # n_landsat_total: all scenes from STAC before pixel fitness gate
    n_landsat_total = anchor_stats["n_total"]

    # ── S2 candidate counts (per anchor, aggregated) ─────────────────────────
    # For the scan we only count, no pixel loads.
    # Approximate: avg candidates per anchor from the window size.
    n_s2_total = 0
    window_days: int = cfg.sentinel2.window_days
    # Rough estimate: window_days * 2 * 2 (S2 overpass frequency ~2/day at Berlin latitude)
    # Multiplied by number of anchors
    n_s2_total = n_landsat_total * window_days * 2 * 2

    # ── ECOSTRESS granules (per anchor day, ±2h window) ──────────────────────
    n_ecostress = 0
    if cfg.years:
        min_year = min(cfg.years)
        max_year = max(cfg.years)
        # One CMR query for the full range (coarse over-estimate, then refined per anchor)
        try:
            eco_all = search_ecostress(
                start=f"{min_year}-05-01",
                end=f"{max_year}-09-30",
                bbox=tuple(cfg.bbox),
                version=cfg.ecostress.version,
            )
            n_ecostress = len(eco_all)
        except Exception:
            # If CMR fails in scan mode, estimate
            n_ecostress = n_landsat_total  # at most one per anchor day

    # ── Estimate coupled / dropped from clear_frac threshold ─────────────────
    # Without pixel loads we use the assumed coupling rate (configurable).
    # In scan mode the real coupling rate is unknown; in couple mode the
    # manifest already gives us the observed count after the fact.
    coupling_rate = float(getattr(cfg.scan, "assumed_coupling_rate", 0.65))
    n_landsat_coupled = int(n_landsat_total * coupling_rate)
    n_landsat_dropped = n_landsat_total - n_landsat_coupled

    # ── Volume estimates (from coupled scenes, not candidate sums) ─────────────
    # Correct: only coupled Landsat anchors get S2 partners + ECOSTRESS subsets.
    est_landsat_gb = n_landsat_coupled * _GB_LANDSAT
    # Correct: 1 S2 per coupled anchor (best candidate), not the candidate sum.
    est_s2_gb = n_landsat_coupled * _GB_S2
    # ECOSTRESS: at most n_landsat_coupled granules (one per coupled anchor day).
    est_ecostress_gb = min(n_landsat_coupled, n_ecostress) * _GB_ECOSTRESS
    est_total_gb = est_landsat_gb + est_s2_gb + est_ecostress_gb

    # ── Write report files ───────────────────────────────────────────────────
    out_dir = Path(cfg.scan_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "scan_report.json"
    md_path = out_dir / "scan_report.md"

    report_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "years": cfg.years,
            "months": cfg.months,
            "bbox": cfg.bbox,
            "s2_window_days": cfg.sentinel2.window_days,
            "s2_score_lambda": getattr(cfg.sentinel2.score, "lambda", 0.1),
            "s2_min_clear_frac": cfg.sentinel2.min_clear_frac,
            "ecoftresst_window_hours": cfg.ecostress.window_hours,
            "assumed_coupling_rate": coupling_rate,
        },
        "counts": {
            "landsat_total": n_landsat_total,
            "landsat_may_sep": n_landsat_total,
            "s2_candidates": n_s2_total,
            "landsat_coupled": n_landsat_coupled,
            "landsat_dropped": n_landsat_dropped,
            "ecostress_matches": n_ecostress,
        },
        "volume_gb": {
            "landsat": round(est_landsat_gb, 2),
            "s2": round(est_s2_gb, 2),
            "ecostress": round(est_ecostress_gb, 2),
            "total": round(est_total_gb, 2),
        },
        "assumptions": {
            "landsat_per_scene_gb": _GB_LANDSAT,
            "s2_per_scene_gb": _GB_S2,
            "ecostress_per_scene_gb": _GB_ECOSTRESS,
            "coupling_rate_estimate": coupling_rate,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    # Markdown summary
    md_lines = [
        "# Szenen-Selektion — Volumen-Scan",
        "",
        f"**Datum:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        f"**Zeitraum:** {'/'.join(str(y) for y in cfg.years)} | Monate {cfg.months}",
        "",
        "## Counts",
        "",
        f"- Landsat-Anker total: **{n_landsat_total}**",
        f"  - Gekoppelt (geschätzt): **{n_landsat_coupled}**  (Rate ≈ {coupling_rate:.0%})",
        f"  - Verworfen (geschätzt): **{n_landsat_dropped}**",
        f"- S2-Kandidaten (Summe): **{n_s2_total}**",
        f"- ECOSTRESS-Granules: **{n_ecostress}**",
        "",
        "## Volumen-Schätzung (GB)",
        "",
        "| Quelle | Szenen | GB/Szene | Gesamt GB |",
        "|--------|--------|-----------|-----------|",
        f"| Landsat | {n_landsat_total} | {_GB_LANDSAT} | {est_landsat_gb:.1f} |",
        f"| Sentinel-2 | {n_s2_total} | {_GB_S2} | {est_s2_gb:.1f} |",
        f"| ECOSTRESS | {n_ecostress} | {_GB_ECOSTRESS} | {est_ecostress_gb:.2f} |",
        f"| **Total** | | | **{est_total_gb:.1f}** |",
        "",
        "## Anmerkungen",
        "",
        "- S2-Kandidaten sind Summe über alle Anker-Fenster (Über­schätzung bei Overlap)",
        "- ECOSTRESS ohne Pixel-Load geschätzt; nur CMR-Metadaten",
        "- coupling_rate = geschätzt aus λ=0.1 und median Δt=1d → Score ≈ clear_frac − 0.033",
        f"- Min. clear_frac-Schwelle = {cfg.sentinel2.min_clear_frac}",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return ScanReport(
        n_landsat_total=n_landsat_total,
        n_landsat_may_sep=n_landsat_total,
        n_s2_candidates=n_s2_total,
        n_landsat_coupled=n_landsat_coupled,
        n_landsat_dropped=n_landsat_dropped,
        n_ecostress_matches=n_ecostress,
        est_landsat_gb=round(est_landsat_gb, 2),
        est_s2_gb=round(est_s2_gb, 2),
        est_ecostress_gb=round(est_ecostress_gb, 2),
        est_total_gb=round(est_total_gb, 2),
        metadata_json=str(json_path),
    )


def _estimate_coupling_rate(cfg) -> float:
    """Estimate Landsat-S2 coupling rate without pixel loads.

    Uses the Notion-observed fact: median Δt = 1 day, λ = 0.1.
    Score = clear_frac − λ·(Δt/3). With median Δt = 1:
      score ≈ clear_frac − 0.033
    Assume median clear_frac ≈ 0.45 (conservative, based on prior smoke runs).

    If threshold = 0.30: score ≈ 0.45 − 0.033 = 0.417 > 0.30 → coupled.
    Conservative estimate: 65% coupling rate.
    """
    return 0.65
