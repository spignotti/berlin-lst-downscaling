#!/usr/bin/env python3
"""Temporal coupling analysis: Landsat ↔ Sentinel-2 for Berlin.

Read-only analysis that pairs each Landsat scene with its temporally-nearest
Sentinel-2 acquisition. Reports Δt distributions and match rates at various
Δt thresholds. No imagery downloaded — metadata only.

Usage:
    uv run python scripts/coupling_analysis.py
"""

import csv
import datetime
from pathlib import Path
from typing import Any

import ee
import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.boundary import buffered_bbox_wgs84
from berlin_lst_downscaling.data.gee_client import initialize
from berlin_lst_downscaling.data.gee_scenes import list_landsat_scenes, list_sentinel2_scenes

# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_acquisitions(
    collection: ee.ImageCollection, cloud_property: str
) -> list[dict[str, Any]]:
    """Batch-extract acquisition metadata from an ImageCollection.

    Uses the proven pattern from ``gee_export.py``: ``select([])`` drops bands,
    ``toList(n).getInfo()`` fetches all properties in one server round trip.

    Args:
        collection: Filtered ``ee.ImageCollection`` (raw, unprocessed).
        cloud_property: Property name for cloud cover percentage.

    Returns:
        List of dicts with ``scene_id``, ``date`` (datetime.date), ``cloud_pct``.
    """
    n_raw = collection.size().getInfo()
    n = int(n_raw) if n_raw else 0
    if n == 0:
        return []

    props_list = collection.select([]).toList(n).getInfo() or []
    records: list[dict[str, Any]] = []
    for i in range(n):
        props = props_list[i].get("properties", {})
        scene_id = str(props.get("system:index", f"scene_{i}"))
        time_ms = props.get("system:time_start")
        if time_ms is None:
            continue
        date = datetime.datetime.fromtimestamp(
            time_ms / 1000, tz=datetime.UTC
        ).date()
        cloud_pct = props.get(cloud_property)
        records.append(
            {
                "scene_id": scene_id,
                "date": date,
                "cloud_pct": float(cloud_pct) if cloud_pct is not None else None,
            }
        )

    return records


def _nearest_date(
    target: datetime.date, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the temporally-nearest candidate to a target date.

    Linear scan — both lists are small (hundreds of Landsat, thousands of S2),
    so no bisect needed.

    Args:
        target: Date to find a neighbor for.
        candidates: List of dicts with a ``date`` key (datetime.date).

    Returns:
        The nearest candidate dict, or ``None`` if candidates is empty.
    """
    if not candidates:
        return None

    best = None
    best_delta = None
    for c in candidates:
        delta = abs((target - c["date"]).days)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = c
            if delta == 0:
                break

    return best


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile for a sorted list."""
    k = (len(sorted_data) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = k - lo
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * frac


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs/ard", config_name="gee_export")
def main(cfg: DictConfig) -> None:
    """Run the coupling analysis."""
    print("=" * 64)
    print("  Coupling Analysis: Landsat ↔ Sentinel-2")
    print("=" * 64)
    print()
    print(f"  AOI:          {list(buffered_bbox_wgs84(cfg.ard.aoi.boundary_file))}")
    print(f"  Period:       {cfg.ard.time.start_year}\u2013{cfg.ard.time.end_year}, "
          f"months {list(cfg.ard.time.months)}")
    print(f"  Landsat cols: {list(cfg.landsat.collections)}")
    print(f"  S2 col:       {cfg.sentinel2.collection}")
    print("  Cloud filter: none (all scenes \u2014 cloud % logged as context)")
    print()

    initialize(cfg)

    # ── 1. Fetch Landsat metadata ──
    print("  [1/5] Fetching Landsat acquisitions...")
    landsat_col = list_landsat_scenes(cfg)
    landsat_records = _extract_acquisitions(landsat_col, "CLOUD_COVER")
    print(f"         \u2192 {len(landsat_records)} scenes")

    # ── 2. Fetch Sentinel-2 metadata ──
    print("  [2/5] Fetching Sentinel-2 acquisitions...")
    s2_col = list_sentinel2_scenes(cfg)
    s2_records = _extract_acquisitions(s2_col, "CLOUDY_PIXEL_PERCENTAGE")
    print(f"         \u2192 {len(s2_records)} scenes")

    if not landsat_records:
        print("\n  ERROR: No Landsat scenes found. Exiting.")
        return
    if not s2_records:
        print("\n  ERROR: No Sentinel-2 scenes found. Exiting.")
        return

    # Sort S2 by date for pairing
    s2_records.sort(key=lambda r: r["date"])

    # ── 3. Pair each Landsat scene with its temporally-nearest S2 ──
    print("  [3/5] Pairing Landsat \u2192 nearest S2...")
    pairs: list[dict[str, Any]] = []
    for lsat in landsat_records:
        nearest = _nearest_date(lsat["date"], s2_records)
        if nearest is None:
            continue
        delta = (lsat["date"] - nearest["date"]).days
        pairs.append(
            {
                "landsat_id": lsat["scene_id"],
                "landsat_date": lsat["date"],
                "landsat_cloud_pct": lsat["cloud_pct"],
                "s2_id": nearest["scene_id"],
                "s2_date": nearest["date"],
                "s2_cloud_pct": nearest["cloud_pct"],
                "delta_days": delta,
                "abs_delta_days": abs(delta),
            }
        )

    n_matched = len(pairs)
    print(f"         \u2192 {n_matched} paired ({len(landsat_records)} total Landsat)")

    # ── 4. Statistics ──
    print("  [4/5] Computing statistics...")
    thresholds = [1, 3, 5, 7]
    deltas = [p["abs_delta_days"] for p in pairs]
    sorted_deltas = sorted(deltas)

    match_counts = {t: sum(1 for d in deltas if d <= t) for t in thresholds}

    stats = {
        "n": len(deltas),
        "min": sorted_deltas[0] if sorted_deltas else None,
        "max": sorted_deltas[-1] if sorted_deltas else None,
        "median": _percentile(sorted_deltas, 50),
        "mean": sum(deltas) / len(deltas) if deltas else None,
        "p90": _percentile(sorted_deltas, 90),
        "p95": _percentile(sorted_deltas, 95),
    }

    # Per-year breakdown
    years_in_data = sorted(set(p["landsat_date"].year for p in pairs))
    per_year: list[dict] = []
    for y in years_in_data:
        y_pairs = [p for p in pairs if p["landsat_date"].year == y]
        y_deltas = [p["abs_delta_days"] for p in y_pairs]
        per_year.append(
            {
                "year": y,
                "total": len(y_pairs),
                "match_counts": {
                    t: sum(1 for d in y_deltas if d <= t) for t in thresholds
                },
                "median_delta": sorted(y_deltas)[len(y_deltas) // 2] if y_deltas else None,
            }
        )

    # ── 5. Write outputs ──
    print("  [5/5] Writing outputs...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = reports_dir / "coupling_pairs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "landsat_id",
                "landsat_date",
                "landsat_cloud_pct",
                "s2_id",
                "s2_date",
                "s2_cloud_pct",
                "delta_days",
                "abs_delta_days",
            ]
        )
        for p in pairs:
            w.writerow(
                [
                    p["landsat_id"],
                    p["landsat_date"].isoformat(),
                    p["landsat_cloud_pct"],
                    p["s2_id"],
                    p["s2_date"].isoformat() if p["s2_date"] else "",
                    p["s2_cloud_pct"],
                    p["delta_days"],
                    p["abs_delta_days"],
                ]
            )
    print(f"         \u2192 {csv_path}")

    # Build markdown report
    md = _build_report(cfg, landsat_records, s2_records, pairs, thresholds, stats, per_year)
    md_path = reports_dir / "coupling_analysis.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"         \u2192 {md_path}")

    # ── 6. Print results to stdout ──
    print()
    print("=" * 64)
    print("  Results")
    print("=" * 64)
    print()
    print(f"  Landsat scenes: {len(landsat_records)}")
    print(f"  S2 scenes:      {len(s2_records)}")
    print(f"  Paired:         {n_matched}")
    print()
    print("  Match rates by \u0394t threshold:")
    for t in thresholds:
        rate = 100.0 * match_counts[t] / n_matched if n_matched > 0 else 0.0
        print(f"    \u00b1{t:2d} days: {match_counts[t]:4d} pairs ({rate:5.1f}%)")
    print()
    print("  \u0394t distribution (absolute):")
    print(f"    Median: {stats['median']:.1f} days")
    print(f"    Mean:   {stats['mean']:.1f} days")
    print(f"    P90:    {stats['p90']:.1f} days")
    print(f"    P95:    {stats['p95']:.1f} days")
    print(f"    Max:    {stats['max']} days")
    print()

    # ── Recommendation ──
    rates = {t: (match_counts[t] / n_matched * 100) if n_matched > 0 else 0.0 for t in thresholds}
    print("  Recommendation:")
    if rates[3] > 90:
        print(f"    Use \u00b13 days \u2014 captures {rates[3]:.0f}% of pairs, "
              f"median \u0394t {stats['median']:.0f}d.")
    elif rates[5] > 95:
        print(f"    Use \u00b15 days \u2014 captures {rates[5]:.0f}% of pairs. "
              f"\u00b13 is too restrictive ({rates[3]:.0f}%).")
    else:
        print(f"    Use \u00b17 days \u2014 captures {rates[7]:.0f}% of pairs. "
              f"Coverage sparser than expected.")
    print("    Pixel-level cloud masking is applied independently per-pixel")
    print("    in the training batch and does not depend on the \u0394t window.")
    print()


def _build_report(
    cfg: DictConfig,
    landsat_records: list[dict],
    s2_records: list[dict],
    pairs: list[dict],
    thresholds: list[int],
    stats: dict,
    per_year: list[dict],
) -> str:
    """Build the markdown report string."""
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_matched = len(pairs)
    match_counts = {t: sum(1 for p in pairs if p["abs_delta_days"] <= t) for t in thresholds}
    rates = {t: (match_counts[t] / n_matched * 100) if n_matched > 0 else 0.0 for t in thresholds}

    lines = [
        "# Coupling Analysis: Landsat \u2194 Sentinel-2",
        "",
        f"**Run date:** {now}",
        f"**AOI:** Berlin + 2 km buffer `{list(buffered_bbox_wgs84(cfg.ard.aoi.boundary_file))}`",
        f"**Period:** May\u2013Sep {cfg.ard.time.start_year}\u2013{cfg.ard.time.end_year}",
        f"**Landsat collections:** `{'`, `'.join(cfg.landsat.collections)}`",
        f"**Sentinel-2 collection:** `{cfg.sentinel2.collection}`",
        "**Cloud filter:** None \u2014 all scenes included. Cloud % reported as context.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Landsat scenes | {len(landsat_records)} |",
        f"| Total Sentinel-2 scenes | {len(s2_records)} |",
        f"| Paired (nearest S2 found) | {n_matched} |",
        "",
        "### Match Rate by \u0394t Threshold",
        "",
        "| \u0394t threshold (days) | Matched pairs | Match rate (%) |",
        "|---------------------|---------------|----------------|",
    ]
    for t in thresholds:
        lines.append(f"| \u00b1{t} | {match_counts[t]} | {rates[t]:.1f} |")

    lines += [
        "",
        "### \u0394t Distribution (abs)",
        "",
        "| Metric | Value (days) |",
        "|--------|-------------|",
        f"| N | {stats['n']} |",
        f"| Min | {stats['min']} |",
        f"| Median | {stats['median']:.1f} |",
        f"| Mean | {stats['mean']:.1f} |",
        f"| P90 | {stats['p90']:.1f} |",
        f"| P95 | {stats['p95']:.1f} |",
        f"| Max | {stats['max']} |",
        "",
        "## Per-Year Breakdown",
        "",
        "| Year | Landsat scenes | \u00b11 | \u00b13 | \u00b15 | \u00b17 | Median \u0394t (days) |",
        "|------|---------------|------|------|------|------|-------------------|",
    ]
    for row in per_year:
        mc = row["match_counts"]
        md_val = f"{row['median_delta']:.0f}" if row["median_delta"] is not None else "-"
        lines.append(
            f"| {row['year']} | {row['total']} | "
            f"{mc[1]} | {mc[3]} | {mc[5]} | {mc[7]} | {md_val} |"
        )

    # Recommendation text
    lines += [
        "",
        "## Recommendation",
        "",
    ]
    if rates[3] > 90:
        lines.append(
            f"- Use **\u00b13 days** \u2014 captures {rates[3]:.0f}% of pairs "
            f"(median \u0394t {stats['median']:.0f}d)."
        )
        lines.append("  Strong recommendation: tight temporal proximity with high coverage.")
    elif rates[5] > 95:
        lines.append(
            f"- Use **\u00b15 days** \u2014 captures {rates[5]:.0f}% of pairs. "
            f"\u00b13 is too restrictive ({rates[3]:.0f}%)."
        )
    else:
        lines.append(
            f"- Use **\u00b17 days** \u2014 captures {rates[7]:.0f}% of pairs. "
            f"Coverage is sparser than expected."
        )

    lines += [
        f"- The median \u0394t of {stats['median']:.0f} days reflects frequent "
        "dual-sensor S2 coverage over Berlin.",
        "- Cloud filtering at pixel level (s2cloudless) is independent of the "
        "\u0394t window and will be applied per-pixel in the training batch.",
        "- Final \u0394t window should be validated with actual pixel-level "
        "coupling once the ARD pipeline is operational.",
        "",
        "---",
        "",
        f"_Analysis generated by `scripts/coupling_analysis.py` on {now}_",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
