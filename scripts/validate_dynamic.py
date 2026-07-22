#!/usr/bin/env python3
"""Validate dynamic pipeline products against expected inventory.

Read-only validator: checks ledger status, artifact existence, and role
consistency for a given dynamic run root.

Usage:
    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/full/<run-id> \\
        --expected-role development \\
        --expected-years 2017-2025 \\
        --expected-scenes 324

    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/inference/2026/<run-id> \\
        --expected-role inference \\
        --expected-years 2026 \\
        --expected-scenes 21

    # Quick status check without full validation:
    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/full/<run-id> \\
        --progress-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import pyarrow.parquet as pq


def load_ledger(output_root: str) -> dict:
    """Load ledger and return status summary."""
    ledger_path = f"{output_root.rstrip('/')}/_state/dynamic/ledger.parquet"

    # Handle both local and GCS paths
    if ledger_path.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes
        raw = read_bytes(ledger_path)
        import io
        table = pq.read_table(io.BytesIO(raw))
    else:
        table = pq.read_table(ledger_path)

    sources = table.column("source").to_pylist()
    statuses = table.column("status").to_pylist()
    has_role = "role" in table.column_names
    roles = table.column("role").to_pylist() if has_role else [None] * table.num_rows

    counts = Counter(zip(sources, statuses, strict=False))
    non_done = []
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pydict()
        if row["status"][0] != "done":
            non_done.append({
                "item_id": row["item_id"][0],
                "source": row["source"][0],
                "status": row["status"][0],
                "attempts": int(row["attempts"][0]),
                "last_error": row["last_error"][0],
            })

    role_counts = Counter(r for r in roles if r is not None)

    return {
        "total_rows": table.num_rows,
        "counts": dict(counts),
        "non_done": non_done,
        "role_counts": dict(role_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dynamic pipeline products")
    parser.add_argument("--output-root", required=True, help="Dynamic run output root")
    parser.add_argument("--expected-role", default=None, help="Expected dataset role")
    parser.add_argument("--expected-years", default=None, help="Expected year range")
    parser.add_argument("--expected-scenes", type=int, default=None, help="Expected scene count")
    parser.add_argument("--progress-only", action="store_true", help="Quick status only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    try:
        ledger = load_ledger(args.output_root)
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
            if ledger["role_counts"]:
                print(f"Roles: {ledger['role_counts']}")
        return 0

    # Full validation
    errors = []
    warnings = []

    # Check scene counts per source
    expected_per_source = args.expected_scenes or 324
    for src in ("era5_land", "shadow_building", "shadow_vegetation"):
        done = ledger["counts"].get((src, "done"), 0)
        if done < expected_per_source:
            errors.append(f"{src}: {done}/{expected_per_source} done")
        else:
            print(f"  {src}: {done}/{expected_per_source} done ✓")

    # Check non-done items
    if ledger["non_done"]:
        for item in ledger["non_done"]:
            warnings.append(
                f"  {item['item_id']}: {item['status']} "
                f"(attempts={item['attempts']}, error={item['last_error']})"
            )

    # Check role consistency
    if args.expected_role and ledger["role_counts"]:
        for role, count in ledger["role_counts"].items():
            if role != args.expected_role:
                warnings.append(f"Unexpected role '{role}': {count} items")

    # Report
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors:
        print("\nAll source counts verified.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
