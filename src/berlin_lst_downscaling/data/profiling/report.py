"""Report generation for WB2c-1 profiling.

Aggregates per-asset profile rows into summary statistics and generates
the fixed artifact bundle: profiles.parquet, profiles.csv, summary.json,
and notion-summary.md.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io.storage import atomic_write
from berlin_lst_downscaling.data.profiling.models import ProfileRow
from berlin_lst_downscaling.data.profiling.paths import (
    notion_summary_path,
    profiles_csv_path,
    profiles_parquet_path,
    summary_json_path,
)


def aggregate_profiles(rows: list[ProfileRow]) -> dict[str, Any]:
    """Aggregate profile rows into summary statistics."""
    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "artifact_status": "complete",
        "total_assets": len(rows),
        "hard_failures": sum(1 for r in rows if r.has_hard_failure),
        "by_source": {},
        "by_partition": {"training": 0, "inference": 0, "shared_static": 0},
        "by_year": {},
        "structural_checks": {
            "cog_exists": sum(1 for r in rows if r.cog_exists),
            "cog_valid": sum(1 for r in rows if r.cog_valid),
            "stac_exists": sum(1 for r in rows if r.stac_exists),
            "stac_valid": sum(1 for r in rows if r.stac_valid),
            "provenance_exists": sum(1 for r in rows if r.provenance_exists),
            "completion_exists": sum(1 for r in rows if r.completion_exists),
        },
    }

    # Aggregate by source
    for row in rows:
        if row.source not in summary["by_source"]:
            summary["by_source"][row.source] = {
                "total": 0,
                "valid": 0,
                "failures": 0,
                "missing_cog": 0,
            }
        summary["by_source"][row.source]["total"] += 1
        if row.cog_valid:
            summary["by_source"][row.source]["valid"] += 1
        if row.has_hard_failure:
            summary["by_source"][row.source]["failures"] += 1
        if not row.cog_exists:
            summary["by_source"][row.source]["missing_cog"] += 1

    # Aggregate by partition
    for row in rows:
        summary["by_partition"][row.partition] = summary["by_partition"].get(row.partition, 0) + 1

    # Aggregate by year
    for row in rows:
        if row.year is not None:
            summary["by_year"][row.year] = summary["by_year"].get(row.year, 0) + 1

    # Aggregate band statistics
    band_stats_agg: dict[str, dict[str, Any]] = {}
    for row in rows:
        for stats in row.band_stats:
            if stats.band_name not in band_stats_agg:
                band_stats_agg[stats.band_name] = {
                    "count": 0,
                    "total_valid": 0,
                    "total_missing": 0,
                    "min_value": float("inf"),
                    "max_value": float("-inf"),
                }
            band_stats_agg[stats.band_name]["count"] += 1
            band_stats_agg[stats.band_name]["total_valid"] += stats.valid_count
            band_stats_agg[stats.band_name]["total_missing"] += stats.missing_count
            if stats.valid_count > 0:
                band_stats_agg[stats.band_name]["min_value"] = min(
                    band_stats_agg[stats.band_name]["min_value"], stats.min_value
                )
                band_stats_agg[stats.band_name]["max_value"] = max(
                    band_stats_agg[stats.band_name]["max_value"], stats.max_value
                )

    summary["band_statistics"] = band_stats_agg

    return summary


def profiles_to_dataframe(rows: list[ProfileRow]) -> pd.DataFrame:
    """Convert profile rows to a pandas DataFrame."""
    records = []
    for row in rows:
        record = {
            "item_id": row.item_id,
            "source": row.source,
            "cog_uri": row.cog_uri,
            "partition": row.partition,
            "year": row.year,
            "season": row.season,
            "resolution_m": row.resolution_m,
            "cog_exists": row.cog_exists,
            "cog_valid": row.cog_valid,
            "stac_exists": row.stac_exists,
            "provenance_exists": row.provenance_exists,
            "completion_exists": row.completion_exists,
            "has_hard_failure": row.has_hard_failure,
            "failure_reasons": "|".join(row.failure_reasons) if row.failure_reasons else "",
        }

        # Add band statistics as flattened columns
        for stats in row.band_stats:
            prefix = f"band_{stats.band_name}"
            record[f"{prefix}_valid_count"] = stats.valid_count
            record[f"{prefix}_missing_rate"] = stats.missing_rate
            record[f"{prefix}_min"] = stats.min_value
            record[f"{prefix}_max"] = stats.max_value
            record[f"{prefix}_mean"] = stats.mean_value
            record[f"{prefix}_std"] = stats.std_value
            record[f"{prefix}_p50"] = stats.p50

        records.append(record)

    return pd.DataFrame(records)


def emit_artifacts(
    rows: list[ProfileRow],
    output_root: str,
) -> None:
    """Emit the fixed artifact bundle."""
    df = profiles_to_dataframe(rows)
    summary = aggregate_profiles(rows)

    # Write profiles.parquet
    table = pa.Table.from_pandas(df)
    parquet_buf = io.BytesIO()
    pq.write_table(table, parquet_buf)
    atomic_write(profiles_parquet_path(output_root), parquet_buf.getvalue(), overwrite=True)

    # Write profiles.csv
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    atomic_write(profiles_csv_path(output_root), csv_buf.getvalue(), overwrite=True)

    # Write summary.json (last - signals completion)
    summary_bytes = json.dumps(summary, indent=2, default=str).encode("utf-8")
    atomic_write(summary_json_path(output_root), summary_bytes, overwrite=True)

    # Write notion-summary.md
    md_content = _format_notion_summary(summary)
    atomic_write(notion_summary_path(output_root), md_content.encode("utf-8"), overwrite=True)


def _format_notion_summary(summary: dict[str, Any]) -> str:
    """Format summary as Notion-ready Markdown."""
    lines = [
        "# WB2c-1 Data Profiling Summary",
        "",
        f"**Timestamp:** {summary['timestamp']}",
        f"**Status:** {summary['artifact_status']}",
        f"**Total Assets:** {summary['total_assets']}",
        f"**Hard Failures:** {summary['hard_failures']}",
        "",
        "## By Source",
        "",
    ]

    for source, stats in summary["by_source"].items():
        lines.append(
            f"- **{source}**: {stats['total']} total, {stats['valid']} valid, "
            f"{stats['failures']} failures, {stats['missing_cog']} missing COGs"
        )

    lines.extend(
        [
            "",
            "## By Partition",
            "",
        ]
    )

    for partition, count in summary["by_partition"].items():
        lines.append(f"- **{partition}**: {count}")

    lines.extend(
        [
            "",
            "## Structural Checks",
            "",
        ]
    )

    checks = summary["structural_checks"]
    lines.append(f"- COG exists: {checks['cog_exists']}/{summary['total_assets']}")
    lines.append(f"- COG valid: {checks['cog_valid']}/{summary['total_assets']}")
    lines.append(f"- STAC exists: {checks['stac_exists']}/{summary['total_assets']}")
    lines.append(f"- Provenance exists: {checks['provenance_exists']}/{summary['total_assets']}")
    lines.append(f"- Completion exists: {checks['completion_exists']}/{summary['total_assets']}")

    return "\n".join(lines)


__all__ = [
    "aggregate_profiles",
    "profiles_to_dataframe",
    "emit_artifacts",
]
