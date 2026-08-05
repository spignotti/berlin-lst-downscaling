"""Report generation for WB2c-1 profiling.

Aggregates per-asset profile rows into summary statistics and generates
the fixed artifact bundle: profiles.parquet, profiles.csv, and summary.json.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io.storage import atomic_write
from berlin_lst_downscaling.data.profiling.models import (
    CompletenessResult,
    CoverageResult,
    ProfileRow,
)
from berlin_lst_downscaling.data.profiling.paths import (
    profiles_csv_path,
    profiles_parquet_path,
    summary_json_path,
)


def profiles_to_long_dataframe(rows: list[ProfileRow]) -> pd.DataFrame:
    """Convert profile rows to a long-format DataFrame.

    Each row in the output represents one band of one asset, with all
    metrics flattened into columns. This preserves full detail without
    exploding the schema.
    """
    records = []
    for row in rows:
        base = {
            "scope": "asset",
            "item_id": row.item_id,
            "source": row.source,
            "partition": row.partition,
            "year": row.year,
            "season": row.season,
            "resolution_m": row.resolution_m,
            "cog_uri": row.cog_uri,
            "cog_exists": row.cog_exists,
            "cog_valid": row.cog_valid,
            "stac_exists": row.stac_exists,
            "provenance_exists": row.provenance_exists,
            "completion_exists": row.completion_exists,
            "has_hard_failure": row.has_hard_failure,
            "failure_reasons": "|".join(row.failure_reasons) if row.failure_reasons else "",
            "cog_errors": "|".join(row.cog_errors) if row.cog_errors else "",
        }

        for stats in row.band_stats:
            record = {
                **base,
                "band_index": stats.band_index,
                "band_name": stats.band_name,
                "valid_count": stats.valid_count,
                "missing_count": stats.missing_count,
                "missing_rate": stats.missing_rate,
                "qa_masked_count": stats.qa_masked_count,
                "qa_masked_rate": stats.qa_masked_rate,
                "min_value": stats.min_value,
                "max_value": stats.max_value,
                "mean_value": stats.mean_value,
                "std_value": stats.std_value,
                "p1": stats.p1,
                "p5": stats.p5,
                "p25": stats.p25,
                "p50": stats.p50,
                "p75": stats.p75,
                "p95": stats.p95,
                "p99": stats.p99,
            }
            # Serialize histogram as JSON strings for Parquet compatibility
            if stats.histogram_bins:
                record["histogram_bins"] = json.dumps(list(stats.histogram_bins))
                record["histogram_counts"] = json.dumps(list(stats.histogram_counts))
            else:
                record["histogram_bins"] = "[]"
                record["histogram_counts"] = "[]"
            record["histogram_underflow"] = stats.histogram_underflow
            record["histogram_overflow"] = stats.histogram_overflow

            records.append(record)

    # Add aggregate records per source/partition/year/season
    agg_df = _build_aggregate_records(rows)
    df = pd.DataFrame(records)

    if not agg_df.empty:
        df = pd.concat([df, agg_df], ignore_index=True)

    return df


def _combined_cdf_percentiles(
    bin_edges: tuple[float, ...],
    counts: tuple[int, ...],
) -> tuple[float, ...]:
    """Derive p1/p5/p25/p50/p75/p95/p99 from a combined histogram CDF.

    Returns NaN placeholders if the histogram is empty.
    """
    total = sum(counts)
    empty = (float("nan"),) * 7
    if total == 0 or len(bin_edges) < 2:
        return empty

    cdf = np.cumsum(counts) / total
    edges = np.array(bin_edges, dtype=float)
    percentile_values: list[float] = []
    for p in (1, 5, 25, 50, 75, 95, 99):
        idx = int(np.searchsorted(cdf, p / 100.0))
        if idx >= len(edges) - 1:
            percentile_values.append(float(edges[-1]))
        elif idx == 0:
            percentile_values.append(float(edges[0]))
        else:
            lo, hi = edges[idx - 1], edges[idx]
            c_lo, c_hi = float(cdf[idx - 1]), float(cdf[idx])
            if c_hi > c_lo:
                frac = (p / 100.0 - c_lo) / (c_hi - c_lo)
                percentile_values.append(float(lo + frac * (hi - lo)))
            else:
                percentile_values.append(float(lo))
    return tuple(percentile_values)


def _build_aggregate_records(rows: list[ProfileRow]) -> pd.DataFrame:
    """Build aggregate records by source/partition/year/season."""
    # Group by (source, partition, year, season)
    groups: dict[tuple[str, str, int | None, str | None], list[ProfileRow]] = {}
    for row in rows:
        key = (row.source, row.partition, row.year, row.season)
        groups.setdefault(key, []).append(row)

    agg_records = []
    for (source, partition, year, season), group_rows in groups.items():
        # Aggregate band stats across all assets in this group
        band_groups: dict[str, list] = {}
        for row in group_rows:
            for stats in row.band_stats:
                band_groups.setdefault(stats.band_name, []).append(stats)

        total_assets = len(group_rows)
        hard_failures = sum(1 for r in group_rows if r.has_hard_failure)
        cog_valid = sum(1 for r in group_rows if r.cog_valid)

        for band_name, band_stats_list in band_groups.items():
            total_valid = sum(s.valid_count for s in band_stats_list)
            total_missing = sum(s.missing_count for s in band_stats_list)
            total_qa_masked = sum(s.qa_masked_count for s in band_stats_list)
            total_px = total_valid + total_missing + total_qa_masked
            weighted_mean = float("nan")
            weighted_std = float("nan")
            finite = [
                s
                for s in band_stats_list
                if s.valid_count > 0 and s.mean_value == s.mean_value and s.std_value == s.std_value
            ]
            if finite and total_valid > 0:
                weighted_sum = sum(s.mean_value * s.valid_count for s in finite)
                weighted_sum_sq = sum(
                    (s.std_value**2 + s.mean_value**2) * s.valid_count for s in finite
                )
                weighted_mean = weighted_sum / total_valid
                variance = (weighted_sum_sq / total_valid) - weighted_mean**2
                weighted_std = max(0.0, variance) ** 0.5

            # Combined histogram + underflow/overflow + CDF percentiles
            combined_bins: tuple[float, ...] = ()
            combined_counts: tuple[int, ...] = ()
            combined_underflow = 0
            combined_overflow = 0
            if band_stats_list and band_stats_list[0].histogram_bins:
                combined_bins = band_stats_list[0].histogram_bins
                combined_counts = tuple(
                    sum(s.histogram_counts[j] for s in band_stats_list if s.histogram_counts)
                    for j in range(len(combined_bins) - 1)
                )
                combined_underflow = sum(s.histogram_underflow for s in band_stats_list)
                combined_overflow = sum(s.histogram_overflow for s in band_stats_list)
            p1, p5, p25, p50, p75, p95, p99 = _combined_cdf_percentiles(
                combined_bins, combined_counts
            )

            agg_record = {
                "scope": "aggregate",
                "item_id": "",
                "source": source,
                "partition": partition,
                "year": year,
                "season": season,
                "resolution_m": group_rows[0].resolution_m,
                "cog_uri": "",
                "cog_exists": True,
                "cog_valid": cog_valid == total_assets,
                "stac_exists": True,
                "provenance_exists": True,
                "completion_exists": True,
                "has_hard_failure": hard_failures > 0,
                "failure_reasons": "",
                "cog_errors": "",
                "band_index": band_stats_list[0].band_index,
                "band_name": band_name,
                "total_assets": total_assets,
                "hard_failures": hard_failures,
                # Aggregate metrics
                "valid_count": total_valid,
                "missing_count": total_missing,
                "missing_rate": total_missing / max(1, total_px),
                "qa_masked_count": total_qa_masked,
                "qa_masked_rate": total_qa_masked / max(1, total_px),
                "min_value": min(
                    (s.min_value for s in band_stats_list if s.valid_count > 0),
                    default=float("nan"),
                ),
                "max_value": max(
                    (s.max_value for s in band_stats_list if s.valid_count > 0),
                    default=float("nan"),
                ),
                "mean_value": weighted_mean,
                "std_value": weighted_std,
                "p1": p1,
                "p5": p5,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
                "p99": p99,
                "histogram_bins": json.dumps(list(combined_bins)) if combined_bins else "[]",
                "histogram_counts": (
                    json.dumps(list(combined_counts)) if combined_counts else "[]"
                ),
                "histogram_underflow": combined_underflow,
                "histogram_overflow": combined_overflow,
            }
            agg_records.append(agg_record)

    return pd.DataFrame(agg_records) if agg_records else pd.DataFrame()


def aggregate_summary(
    rows: list[ProfileRow],
    completeness: CompletenessResult | None = None,
    coverage: list[CoverageResult] | None = None,
) -> dict[str, Any]:
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
            "flag_required": sum(1 for r in rows if r.flag_required),
            "flag_valid": sum(1 for r in rows if r.flag_valid),
        },
        "contract_checks": {
            "dtype_mismatches": sum(
                len(r.contract_check.dtype_mismatches) for r in rows
            ),
            "nodata_mismatches": sum(
                len(r.contract_check.nodata_mismatches) for r in rows
            ),
            "channel_order_mismatches": sum(
                len(r.contract_check.channel_order_mismatches) for r in rows
            ),
            "prose_description_absent": sum(
                len(r.contract_check.prose_description_absent) for r in rows
            ),
            "unit_absent": sum(
                len(r.contract_check.unit_absent) for r in rows
            ),
        },
    }

    # Completeness result
    if completeness is not None:
        summary["manifest_ledger_completeness"] = {
            "manifest_key_count": completeness.manifest_key_count,
            "ledger_key_count": completeness.ledger_key_count,
            "missing_in_ledger": completeness.missing_in_ledger[:10],
            "extra_in_ledger": completeness.extra_in_ledger[:10],
            "duplicate_keys": completeness.duplicate_keys[:10],
            "ok": completeness.ok,
        }

    # Dynamic coverage results
    if coverage is not None:
        summary["dynamic_coverage"] = [
            {
                "partition": c.partition,
                "expected": c.expected,
                "found": c.found,
                "missing": c.missing[:10],
                "extra": c.extra[:10],
                "ok": c.ok,
            }
            for c in coverage
        ]

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


def emit_artifacts(
    rows: list[ProfileRow],
    output_root: str,
    completeness: CompletenessResult | None = None,
    coverage: list[CoverageResult] | None = None,
) -> None:
    """Emit the fixed artifact bundle."""
    df = profiles_to_long_dataframe(rows)
    summary = aggregate_summary(rows, completeness=completeness, coverage=coverage)

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


__all__ = [
    "aggregate_summary",
    "profiles_to_long_dataframe",
    "emit_artifacts",
]
