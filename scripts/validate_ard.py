# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "numpy",
# ]
# ///
"""Standalone ARD validation — read ledger, validate all done scenes.

Usage
-----
    uv run python scripts/validate_ard.py --ledger data/smoke/primary/ard/ledger.parquet

Output
------
Summary:  X/Y scenes passed
Failed:
  [source] scene_id  - error
"""

from __future__ import annotations

import argparse
import sys

import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ARD COGs from ledger")
    parser.add_argument("--ledger", required=True, help="Path to ledger.parquet")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all scenes, not just failed"
    )
    args = parser.parse_args()

    # ── Load ledger ────────────────────────────────────────────────────
    try:
        tbl = pq.read_table(args.ledger)
    except Exception as exc:
        print(f"Error: cannot read ledger at {args.ledger}: {exc}", file=sys.stderr)
        return 1

    print(f"Ledger: {tbl.num_rows} rows")

    from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.validate import (
        ValidationResult,
        format_validation_report,
        validate_cog,
        validate_flag_cog,
    )

    # ── Resolution lookup ──────────────────────────────────────────────
    _RES_MAP = {"landsat-c2-l2": 100, "sentinel-2-l2a": 10, "ecostress": 70}

    # ── Filter done scenes ─────────────────────────────────────────────
    done_mask = [s == "done" for s in tbl.column("status").to_pylist()]
    n_done = sum(done_mask)
    if n_done == 0:
        print("No 'done' scenes to validate.")
        return 0

    results: list[ValidationResult] = []
    for i in range(tbl.num_rows):
        row = tbl.slice(i, 1).to_pylist()[0]
        if row["status"] != "done":
            continue

        source = row["source"]
        scene_id = row["scene_id"]
        cog_uri = row.get("path_cog")
        flag_uri = row.get("path_flag")
        if not flag_uri and cog_uri:
            flag_uri = cog_uri.replace(".tif", ".flag.tif")

        if not cog_uri:
            results.append(
                ValidationResult(
                    scene_id=scene_id,
                    source=source,
                    ok=False,
                    errors=["No path_cog in ledger"],
                )
            )
            continue

        res = _RES_MAP.get(source, 10)
        expected_grid = canon_grid_for_resolution(res)
        contract = contract_for_source(source)

        # Validate main COG
        vr = validate_cog(cog_uri, contract, expected_grid)
        vr.scene_id = scene_id
        vr.source = source

        # Validate flag COG
        if flag_uri and contract.flag_mode == "separate":
            vf = validate_flag_cog(flag_uri, expected_grid)
            vf.scene_id = scene_id
            vf.source = source
            if not vf.ok:
                vr.ok = False
                vr.errors.extend(vf.errors)

        results.append(vr)

    # ── Report ─────────────────────────────────────────────────────────
    print(format_validation_report(results))

    total_ok = sum(1 for r in results if r.ok)
    if total_ok < len(results):
        print(f"\nFAILED: {len(results) - total_ok}/{len(results)} scenes have issues")
        return 1
    else:
        print(f"\nAll {len(results)} done scenes validated successfully")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
