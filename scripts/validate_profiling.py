#!/usr/bin/env python3
"""Validate WB2c-1 profiling artifacts.

Read-only validator: checks artifact existence, structure, and
optionally verifies zero hard failures and expected asset count.

Usage::

    # Validate smoke run
    uv run python scripts/validate_profiling.py \\
        --output-root gs://berlin-lst-data/profiling/wb2c-1-smoke

    # Validate full run with strict checks
    uv run python scripts/validate_profiling.py \\
        --output-root gs://berlin-lst-data/profiling/wb2c-1 \\
        --require-clean --expected-assets 2079
"""

from __future__ import annotations

import argparse
import json

from berlin_lst_downscaling.data.io.storage import exists, read_bytes
from berlin_lst_downscaling.data.profiling.paths import (
    profiles_csv_path,
    profiles_parquet_path,
    summary_json_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WB2c-1 profiling artifacts")
    parser.add_argument("--output-root", required=True, help="Profiling output root")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Require zero hard failures",
    )
    parser.add_argument(
        "--expected-assets",
        type=int,
        default=None,
        help="Expected number of assets",
    )
    args = parser.parse_args()

    errors: list[str] = []

    # Check artifact existence
    parquet_uri = profiles_parquet_path(args.output_root)
    csv_uri = profiles_csv_path(args.output_root)
    summary_uri = summary_json_path(args.output_root)

    for label, uri in [
        ("profiles.parquet", parquet_uri),
        ("profiles.csv", csv_uri),
        ("summary.json", summary_uri),
    ]:
        if not exists(uri):
            errors.append(f"Missing {label}: {uri}")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    # Load and validate summary.json
    try:
        summary_bytes = read_bytes(summary_uri)
        summary = json.loads(summary_bytes)
    except Exception as exc:
        print(f"  ✗ Cannot read summary.json: {exc}")
        return 1

    # Check artifact_status
    if summary.get("artifact_status") != "complete":
        status = summary.get("artifact_status")
        errors.append(f"artifact_status: expected 'complete', got {status!r}")

    # Check hard failures
    hard_failures = summary.get("hard_failures", 0)
    if args.require_clean and hard_failures > 0:
        errors.append(f"hard_failures: expected 0, got {hard_failures}")

    # Check asset count
    total_assets = summary.get("total_assets", 0)
    if args.expected_assets is not None and total_assets != args.expected_assets:
        errors.append(f"total_assets: expected {args.expected_assets}, got {total_assets}")

    # Check contract checks
    contract = summary.get("contract_checks", {})
    if args.require_clean:
        for key in ("dtype_mismatches", "nodata_mismatches",
                     "band_description_mismatches", "band_order_mismatches"):
            count = contract.get(key, 0)
            if count > 0:
                errors.append(f"contract_checks.{key}: {count} mismatches")

    # Check completeness
    completeness = summary.get("manifest_ledger_completeness", {})
    if completeness and not completeness.get("ok", True):
        missing = len(completeness.get("missing_in_ledger", []))
        extra = len(completeness.get("extra_in_ledger", []))
        dups = len(completeness.get("duplicate_keys", []))
        errors.append(
            f"completeness: {missing} missing, {extra} extra, {dups} duplicates"
        )

    # Print summary
    print(f"Artifact root: {args.output_root}")
    print(f"  Total assets: {total_assets}")
    print(f"  Hard failures: {hard_failures}")
    print(f"  By source: {len(summary.get('by_source', {}))} sources")
    print(f"  By partition: {summary.get('by_partition', {})}")
    if contract:
        print(f"  Contract: dtype={contract.get('dtype_mismatches', 0)}, "
              f"nodata={contract.get('nodata_mismatches', 0)}, "
              f"desc={contract.get('band_description_mismatches', 0)}")
    if completeness:
        print(f"  Completeness: ok={completeness.get('ok', 'n/a')}, "
              f"manifest={completeness.get('manifest_key_count', 0)}, "
              f"ledger={completeness.get('ledger_key_count', 0)}")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print("\nValidation passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
