"""Dynamic QA report — coverage, vintage distribution, DWD comparison.

The report is persisted to ``qa/dynamic/{run_id}/report.json`` and summarizes:
- Expected vs completed scene products
- Per-year coverage
- ERA5 channel ranges
- DWD validation comparison
- Geometry vintage distribution
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from berlin_lst_downscaling.data.dynamic.paths import qa_report_path
from berlin_lst_downscaling.data.io import atomic_write, exists
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger


def dynamic_qa_report(
    ledger: SecondaryLedger,
    run_id: str,
    manifest_hash: str,
    geometry_id: str,
) -> dict[str, Any]:
    """Generate a QA report for the dynamic pipeline run.

    Returns a dict with per-source counts, artifact completeness,
    vintage distribution, and DWD comparison results.
    """
    sources = ["era5_land", "shadow_building", "shadow_vegetation"]

    per_source: dict[str, dict[str, Any]] = {}
    for src in sources:
        counts = ledger.status_counts(src)
        rows = ledger.items_for_source(src)

        output_ok = sum(
            1 for r in rows if r.status == "done" and r.output_uri and exists(r.output_uri)
        )
        completed = sum(
            1 for r in rows if r.status == "done" and r.completion_uri and exists(r.completion_uri)
        )

        per_source[src] = {
            "total": len(rows),
            **counts,
            "output_exists": output_ok,
            "completed": completed,
        }

    # Year distribution
    all_rows = []
    for src in sources:
        all_rows.extend(ledger.items_for_source(src))

    year_dist: dict[int, int] = {}
    for r in all_rows:
        if r.status == "done":
            # Extract year from scene_id (e.g. "LC09_L2SP_..._20240629_...")
            parts = r.period_or_vintage.split("_")
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    year = int(part[:4])
                    if 2017 <= year <= 2026:
                        year_dist[year] = year_dist.get(year, 0) + 1
                        break

    total_failed = sum(per_source[s].get("failed", 0) for s in per_source)
    total_completed = sum(per_source[s].get("completed", 0) for s in per_source)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest_hash": manifest_hash,
        "geometry_id": geometry_id,
        "geometry_temporal_mode": "retrospective_static",
        "geometry_vintages": {
            "lod2_morphology": "2024",
            "terrain_height": "2021",
            "vegetation_height": "2020",
        },
        "per_source": per_source,
        "year_distribution": dict(sorted(year_dist.items())),
        "total_completed": total_completed,
        "total_failed": total_failed,
        "success": total_failed == 0,
    }

    return report

def persist_dynamic_report(
    report: dict[str, Any],
    output_root: str,
) -> str:
    """Persist the dynamic QA report."""
    uri = qa_report_path(output_root, report["run_id"])
    atomic_write(uri, json.dumps(report, indent=2), overwrite=True)
    return uri

def format_dynamic_report(report: dict[str, Any]) -> str:
    """Format a dynamic QA report for console output."""
    lines = [
        f"Dynamic QA Report — run {report['run_id']}",
        f"  Timestamp : {report['timestamp']}",
        f"  Manifest  : {report['manifest_hash']}",
        f"  Geometry  : {report['geometry_id']}",
        f"  Mode      : {report['geometry_temporal_mode']}",
        f"  Success   : {'yes' if report['success'] else 'NO'}",
        f"  Completed : {report['total_completed']}",
        f"  Failed    : {report['total_failed']}",
        "",
    ]

    for src, data in report.get("per_source", {}).items():
        lines.append(f"  [{src}]")
        lines.append(f"    Total       : {data['total']}")
        for status in ("done", "failed", "exporting", "pending"):
            if data.get(status, 0) > 0:
                lines.append(f"    {status:<12}: {data[status]}")
        lines.append(f"    Output OK   : {data.get('output_exists', 0)}")
        lines.append(f"    Completed   : {data.get('completed', 0)}")
        lines.append("")

    # Year distribution
    year_dist = report.get("year_distribution", {})
    if year_dist:
        lines.append("  Year distribution:")
        for year, count in sorted(year_dist.items()):
            lines.append(f"    {year}: {count} scenes")
        lines.append("")

    return "\n".join(lines)

__all__ = [
    "dynamic_qa_report",
    "persist_dynamic_report",
    "format_dynamic_report",
]