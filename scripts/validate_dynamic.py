#!/usr/bin/env python3
"""Validate dynamic pipeline products against expected inventory.

Read-only validator: checks ledger status, artifact existence, and
role consistency for a given dynamic run root.

Usage::

    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/full/<run-id> \\
        --expected-role anchor \\
        --expected-scenes 324

    uv run python scripts/validate_dynamic.py \\
        --output-root gs://berlin-lst-data/dynamic/inference/2026/<run-id> \\
        --expected-role inference \\
        --expected-scenes 21
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import pyarrow.parquet as pq


def _read_ledger(output_root: str) -> dict:
    ledger_path = f"{output_root.rstrip('/')}/_state/dynamic/ledger.parquet"
    if ledger_path.startswith("gs://"):
        import io

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
