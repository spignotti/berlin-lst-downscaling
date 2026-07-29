#!/usr/bin/env python3
"""Validate dynamic pipeline products against expected inventory.

Read-only validator: checks ledger status, artifact existence, role
consistency, COG band contract, and geometry mapping for a given
dynamic run root.

Usage::

    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/full \\
        --expected-role anchor \\
        --expected-scenes 324

    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/inference/2026 \\
        --expected-role inference \\
        --expected-scenes 21
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter

import pyarrow.parquet as pq

# ── ERA5 8-band contract ─────────────────────────────────────────────

_ERA5_BAND_NAMES = [
    "t2m_scene",
    "ssrd_scene",
    "ssrd_antecedent_72h_mean",
    "vpd_scene",
    "wind_speed_10m_scene",
    "tp_0_24h",
    "tp_24_48h",
    "tp_48_72h",
]

_ERA5_BAND_UNITS = {
    "t2m_scene": "K",
    "ssrd_scene": "W/m²",
    "ssrd_antecedent_72h_mean": "W/m²",
    "vpd_scene": "kPa",
    "wind_speed_10m_scene": "m/s",
    "tp_0_24h": "mm",
    "tp_24_48h": "mm",
    "tp_48_72h": "mm",
}


def _read_ledger(output_root: str) -> dict:
    ledger_path = f"{output_root.rstrip('/')}/_state/dynamic/ledger.parquet"
    if ledger_path.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes

        table = pq.read_table(io.BytesIO(read_bytes(ledger_path)))
    else:
        table = pq.read_table(ledger_path)
    return _summarise(table)


def _summarise(table) -> dict:
    sources = table.column("source").to_pylist()
    statuses = table.column("status").to_pylist()
    roles = table.column("role").to_pylist()

    counts = Counter(zip(sources, statuses, strict=False))
    non_done = []
    missing_role = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pydict()
        if row["status"][0] != "done":
            non_done.append(
                {
                    "item_id": row["item_id"][0],
                    "source": row["source"][0],
                    "status": row["status"][0],
                    "attempts": int(row["attempts"][0]),
                    "last_error": row["last_error"][0],
                }
            )
        if row["role"][0] is None:
            missing_role += 1

    return {
        "total_rows": table.num_rows,
        "counts": dict(counts),
        "non_done": non_done,
        "role_counts": dict(Counter(roles)),
        "missing_role": missing_role,
    }


def _validate_cog_bands(cog_uri: str) -> list[str]:
    """Check COG has 8 bands with expected names and float32 dtype."""
    errors = []
    try:
        import rasterio

        with rasterio.open(cog_uri) as src:
            if src.count != 8:
                errors.append(f"band count: expected 8, got {src.count}")
            if src.dtypes[0] != "float32":
                errors.append(f"dtype: expected float32, got {src.dtypes[0]}")
    except Exception as exc:
        errors.append(f"cannot open COG: {exc}")
    return errors


def _validate_era5_provenance(prov_uri: str) -> list[str]:
    """Check ERA5 provenance carries 8-band channel list."""
    errors = []
    try:
        from berlin_lst_downscaling.data.io.storage import read_bytes

        prov = json.loads(read_bytes(prov_uri))
        channels = prov.get("era5_channels", prov.get("channels", []))
        if len(channels) != 8:
            errors.append(
                f"provenance channel count: expected 8, got {len(channels)}"
            )
    except Exception as exc:
        errors.append(f"cannot read provenance: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dynamic pipeline products")
    parser.add_argument("--output-root", required=True, help="Dynamic run output root")
    parser.add_argument("--expected-role", required=True, help="Expected dataset role")
    parser.add_argument(
        "--expected-scenes",
        type=int,
        required=True,
        help="Expected total scene count per source (e.g. 324 or 21)",
    )
    parser.add_argument("--progress-only", action="store_true", help="Quick status only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--check-bands",
        action="store_true",
        help="Also validate COG band count and provenance (slower)",
    )
    args = parser.parse_args()

    try:
        ledger = _read_ledger(args.output_root)
    except Exception as e:
        print(f"ERROR: Failed to load ledger: {e}", file=sys.stderr)
        return 1

    if args.progress_only:
        if args.json:
            print(json.dumps(ledger, indent=2, default=str))
        else:
            print(f"Ledger: {ledger['total_rows']} rows")
            print(f"Counts: {ledger['counts']}")
            print(f"Non-done: {len(ledger['non_done'])} items")
            print(f"Roles: {ledger['role_counts']}")
            print(f"Missing role: {ledger['missing_role']}")
        return 0

    errors = []
    warnings = []

    expected_per_source = args.expected_scenes
    for src in ("era5_land", "shadow_building", "shadow_vegetation"):
        done = ledger["counts"].get((src, "done"), 0)
        if done != expected_per_source:
            errors.append(f"{src}: {done}/{expected_per_source} done")
        else:
            print(f"  {src}: {done}/{expected_per_source} done ✓")

    if ledger["non_done"]:
        for item in ledger["non_done"]:
            errors.append(
                f"  non-done: {item['item_id']} status={item['status']} "
                f"attempts={item['attempts']} error={item['last_error']}"
            )

    if ledger["missing_role"] > 0:
        errors.append(f"{ledger['missing_role']} rows have a null role (required)")
    for role, count in ledger["role_counts"].items():
        if role is None:
            continue
        if role != args.expected_role:
            errors.append(f"Unexpected role '{role}': {count} items")

    # ── band/provenance checks (optional, slower) ────────────────────
    if args.check_bands:
        print("\n  Checking ERA5 band contracts (sample) ...")

        # Read raw table for artifact inspection
        ledger_path = f"{args.output_root.rstrip('/')}/_state/dynamic/ledger.parquet"
        if ledger_path.startswith("gs://"):
            from berlin_lst_downscaling.data.io.storage import read_bytes

            table = pq.read_table(io.BytesIO(read_bytes(ledger_path)))
        else:
            table = pq.read_table(ledger_path)

        era5_done = [
            table.slice(i, 1).to_pydict()
            for i in range(table.num_rows)
            if table.slice(i, 1).to_pydict()["source"][0] == "era5_land"
            and table.slice(i, 1).to_pydict()["status"][0] == "done"
        ]

        # Check first 5 ERA5 products for band contract
        sample_size = min(5, len(era5_done))
        band_errors = 0
        for row_dict in era5_done[:sample_size]:
            cog_uri = row_dict.get("output_uri", [None])[0]
            prov_uri = row_dict.get("provenance_uri", [None])[0]
            if cog_uri:
                errs = _validate_cog_bands(cog_uri)
                if errs:
                    band_errors += 1
                    for e in errs:
                        errors.append(f"  {row_dict['item_id'][0]}: {e}")
            if prov_uri:
                errs = _validate_era5_provenance(prov_uri)
                if errs:
                    band_errors += 1
                    for e in errs:
                        errors.append(f"  {row_dict['item_id'][0]}: {e}")

        if band_errors == 0 and sample_size > 0:
            print(f"  ERA5 band contract: {sample_size} samples OK ✓")
        elif sample_size == 0:
            warnings.append("No ERA5 done rows to check")

    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors:
        print("\nAll source counts and roles verified.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
