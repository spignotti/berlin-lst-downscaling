#!/usr/bin/env python3
"""Re-read a local Stadtstruktur release and check its published contract.

Usage
-----
uv run python scripts/validators/validate_stadtstruktur.py \
    --release-dir /tmp/stadtstruktur-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from berlin_lst_downscaling.data.stadtstruktur.sources import RELEASE_ID, TARGET_CRS


def _fail(messages: list[str]) -> int:
    for message in messages:
        print(f"FAIL {message}")
    print(f"{len(messages)} failure(s)")
    return 1


def validate(release_dir: Path) -> int:
    """Return 0 when the release matches the report and the evaluation contract."""
    report_path = release_dir / "qa" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("release_id") != RELEASE_ID:
        failures.append(f"release_id is {report.get('release_id')}")
    if report.get("crs") != TARGET_CRS:
        failures.append("report CRS is not EPSG:25833")
    if not report.get("qa_pass"):
        failures.append("report qa_pass is false")
    complete = release_dir / "complete.json"
    if report.get("qa_pass") and not complete.exists():
        failures.append("complete.json is missing")
    if not report.get("qa_pass") and complete.exists():
        failures.append("complete.json exists for a failed release")

    groupings = json.loads(
        (release_dir / "qa" / "candidate_groupings.json").read_text(encoding="utf-8")
    )
    if groupings.get("status") != "proposal_awaiting_scientific_approval":
        failures.append("candidate groupings are not marked as an unapproved proposal")
    if groupings.get("candidate_b", {}).get("unmapped_observed_typ"):
        failures.append("candidate B does not cover every observed typ")

    for vintage in report["vintages"]:
        year = str(vintage["vintage"])
        blocks = gpd.read_parquet(release_dir / "source" / year / "blocks.parquet")
        shares = pd.read_parquet(release_dir / "grid" / "100m" / year / "shares.parquet")
        cells = pd.read_parquet(release_dir / "grid" / "100m" / year / "cells.parquet")
        block_shares = pd.read_parquet(release_dir / "grid" / "1000m" / year / "shares.parquet")
        epsg = None if blocks.crs is None else blocks.crs.to_epsg()
        if epsg != 25833:
            failures.append(f"{year}: source EPSG is {epsg}")
        if len(blocks) != vintage["feature_count"]:
            failures.append(f"{year}: block row count differs from the report")
        if int(blocks["schluessel"].duplicated().sum()) != 0:
            failures.append(f"{year}: duplicate schluessel on read-back")
        if int((~blocks.geometry.is_valid).sum()) != 0:
            failures.append(f"{year}: invalid geometry on read-back")
        share_area = float(shares["area_m2"].sum())
        cell_area = float(cells["covered_m2"].sum())
        block_area = float(block_shares["area_m2"].sum())
        if abs(share_area - cell_area) > 1.0:
            failures.append(f"{year}: share area and cell area differ")
        if abs(share_area - block_area) > 1.0:
            failures.append(f"{year}: 100 m and 1,000 m areas differ")
        if abs(share_area - float(vintage["grid"]["share_area_m2"])) > 1.0:
            failures.append(f"{year}: share area differs from the report")
        mixed_from_shares = int(shares.groupby("cell_id")["typ"].nunique().gt(1).sum())
        mixed_from_cells = int(cells["mixed"].sum())
        if mixed_from_shares != mixed_from_cells:
            failures.append(f"{year}: mixed-cell count is inconsistent")
        if shares["share_of_cell"].max() > 1.001:
            failures.append(f"{year}: a cell share exceeds 1")

    if failures:
        return _fail(failures)
    print(
        f"GO {RELEASE_ID}: {len(report['vintages'])} vintages, "
        f"CRS {TARGET_CRS}, candidate groupings unapproved"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local Stadtstruktur release")
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(validate(args.release_dir))


if __name__ == "__main__":
    main()
