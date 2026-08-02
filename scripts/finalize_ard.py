# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "numpy",
#     "pystac>=1.11.0",
#     "pystac-client>=0.9.0",
#     "xarray",
#     "rioxarray",
# ]
# ///
"""ARD metadata finalization / repair — idempotent, generation-guarded.

Writes provenance.json, STAC item, and complete.json against existing
validated COG/flag assets.  Never mutates COG/flag bytes.

Usage:
    # Dry-run (default) — reports what would change
    uv run python scripts/finalize_ard.py \
        --ledger gs://.../ledger.parquet

    # Apply — writes metadata sidecars with GCS generation preconditions
    uv run python scripts/finalize_ard.py \
        --ledger gs://.../ledger.parquet --apply
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RepairAction:
    scene_id: str
    source: str
    year: int
    artifact: str  # "stac", "provenance", "complete"
    uri: str
    reason: str  # "missing" | "invalid" | "stale"


@dataclass
class RepairReport:
    actions: list[RepairAction] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = True
    repair_commit: str = ""
    start_time: str = ""
    end_time: str = ""
    total_scenes: int = 0
    repaired_scenes: int = 0


def _read_table(uri: str):
    """Read a Parquet table from local path or GCS."""
    import pyarrow.parquet as pq
    if uri.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes
        return pq.read_table(io.BytesIO(read_bytes(uri)))
    return pq.read_table(uri)


def main() -> int:

    parser = argparse.ArgumentParser(description="ARD metadata finalization / repair")
    parser.add_argument("--ledger", required=True, help="Path to ARD ledger.parquet")
    parser.add_argument(
        "--apply", action="store_true",
        help="Write metadata sidecars (default: dry-run)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dry_run = not args.apply
    report = RepairReport(dry_run=dry_run)
    report.start_time = datetime.now(UTC).isoformat()

    # Load ledger
    try:
        ledger_tbl = _read_table(args.ledger)
    except Exception as exc:
        print(f"Error: cannot read ledger: {exc}", file=sys.stderr)
        return 1

    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.io import exists

    done_rows = []
    for i in range(ledger_tbl.num_rows):
        row_dict = ledger_tbl.slice(i, 1).to_pydict()
        if row_dict["status"][0] == "done":
            done_rows.append(row_dict)

    report.total_scenes = len(done_rows)
    print(f"Ledger: {len(done_rows)} done scenes")
    print(f"Mode: {'APPLY' if not dry_run else 'DRY-RUN'}")

    # Determine git commit for provenance
    repair_commit = _git_commit_hash()
    report.repair_commit = repair_commit
    if repair_commit:
        print(f"Repair commit: {repair_commit[:12]}")

    run_id = f"repair-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"

    for row_dict in done_rows:
        scene_id = row_dict["scene_id"][0]
        source = row_dict["source"][0]
        year = int(row_dict["year"][0])
        path_cog = row_dict["path_cog"][0]
        path_flag = row_dict.get("path_flag", [None])[0]

        if not path_cog:
            report.errors.append(f"{scene_id}: no path_cog")
            continue

        scene_dir = os.path.dirname(path_cog)
        stac_uri = f"{scene_dir}/{scene_id}.stac.json"
        prov_uri = f"{scene_dir}/provenance.json"
        comp_uri = f"{scene_dir}/complete.json"

        contract = contract_for_source(source)

        # Check what needs writing
        needs_stac = not exists(stac_uri) or _stac_invalid(stac_uri)
        needs_prov = not exists(prov_uri) or _prov_invalid(prov_uri, source, scene_id)
        needs_comp = not exists(comp_uri)

        if not needs_stac and not needs_prov and not needs_comp:
            report.skipped += 1
            if args.verbose:
                print(f"  {scene_id}/{source}: OK (all sidecars present)")
            continue

        report.repaired_scenes += 1
        actions = []

        if needs_prov:
            reason = "missing" if not exists(prov_uri) else "invalid"
            actions.append(RepairAction(
                scene_id, source, year, "provenance", prov_uri, reason,
            ))
        if needs_stac:
            reason = "missing" if not exists(stac_uri) else "invalid"
            actions.append(RepairAction(
                scene_id, source, year, "stac", stac_uri, reason,
            ))
        if needs_comp:
            actions.append(RepairAction(
                scene_id, source, year, "complete", comp_uri, "missing",
            ))

        report.actions.extend(actions)

        if args.verbose:
            for a in actions:
                print(f"  {scene_id}/{source}: {a.artifact} ({a.reason})")

        if not dry_run:
            # Apply: write sidecars
            _apply_sidecars(
                scene_id, source, year, path_cog, path_flag,
                contract, run_id, repair_commit,
            )

    report.end_time = datetime.now(UTC).isoformat()

    # Write report
    report_path = os.path.join(os.path.dirname(args.ledger), "repair_report.json")
    if not args.ledger.startswith("gs://"):
        report_path = f"{os.path.dirname(args.ledger)}/repair_report.json"
    else:
        # For GCS, report stays local in a task-specific temp dir
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="ard_repair_")
        report_path = f"{tmp_dir}/repair_report.json"

    report_dict = {
        "dry_run": report.dry_run,
        "repair_commit": report.repair_commit,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "total_scenes": report.total_scenes,
        "repaired_scenes": report.repaired_scenes,
        "skipped": report.skipped,
        "actions": [
            {"scene_id": a.scene_id, "source": a.source, "artifact": a.artifact,
             "uri": a.uri, "reason": a.reason}
            for a in report.actions
        ],
        "errors": report.errors,
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nSummary: {report.repaired_scenes}/{report.total_scenes} scenes need repair")
    print(f"  Actions: {len(report.actions)}")
    print(f"  Skipped: {report.skipped}")
    if report.errors:
        print(f"  Errors: {len(report.errors)}")
    print(f"\nReport saved: {report_path}")

    return 0 if not report.errors else 1


def _git_commit_hash() -> str:
    """Return the current git commit hash, or empty string."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _stac_invalid(stac_uri: str) -> bool:
    """Check if a STAC item is missing or structurally invalid."""
    from berlin_lst_downscaling.data.io import exists, read_bytes
    if not exists(stac_uri):
        return True
    try:
        item = json.loads(read_bytes(stac_uri))
        # Must have bbox, extensions, assets with raster:bands
        if "bbox" not in item:
            return True
        exts = set(item.get("stac_extensions", []))
        if "https://stac-extensions.github.io/raster/v1.1.0/schema.json" not in exts:
            return True
        # Check for invalid nodata=null
        for asset in item.get("assets", {}).values():
            for band in asset.get("raster:bands", []):
                if band.get("nodata") is None and "nodata" in band:
                    return True
        return False
    except Exception:
        return True


def _prov_invalid(prov_uri: str, source: str, scene_id: str) -> bool:
    """Check if provenance.json is missing or has wrong source/scene_id."""
    from berlin_lst_downscaling.data.io import exists, read_bytes
    if not exists(prov_uri):
        return True
    try:
        prov = json.loads(read_bytes(prov_uri))
        return prov.get("source") != source or prov.get("scene_id") != scene_id
    except Exception:
        return True


def _apply_sidecars(
    scene_id: str,
    source: str,
    year: int,
    path_cog: str,
    path_flag: str | None,
    contract: Any,
    run_id: str,
    repair_commit: str,
) -> None:
    """Write provenance, STAC, and completion sidecars for one scene."""
    from berlin_lst_downscaling.data.ard.product import (
        build_ard_provenance,
        build_ard_stac_item,
    )
    from berlin_lst_downscaling.data.io import atomic_write, exists

    scene_dir = os.path.dirname(path_cog)
    stac_uri = f"{scene_dir}/{scene_id}.stac.json"
    prov_uri = f"{scene_dir}/provenance.json"
    comp_uri = f"{scene_dir}/complete.json"

    # Read COG metadata for STAC construction
    import rasterio

    with rasterio.open(path_cog) as src:
        crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width

    # Build minimal xarray Dataset with CRS/transform metadata
    import numpy as np
    import xarray as xr

    dummy = xr.DataArray(
        np.zeros((1, height, width), dtype="float32"),
        dims=("time", "y", "x"),
    )
    dummy = dummy.rio.write_crs(str(crs))
    dummy = dummy.rio.write_transform(transform)
    dummy = dummy.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dataset = xr.Dataset({"band": dummy})

    target_resolution = (
        100 if source == "landsat-c2-l2" else
        70 if source == "ecostress" else 10
    )

    # 1. Provenance
    provenance = build_ard_provenance(
        scene_id, source, year, contract, run_id,
        repair=True, repair_commit=repair_commit,
    )
    prov_gen = 0 if not exists(prov_uri) else None
    atomic_write(prov_uri, json.dumps(provenance, indent=2), overwrite=True,
                 if_generation_match=prov_gen)

    # 2. STAC item
    stac_item = build_ard_stac_item(
        scene_id, source, year, dataset, contract,
        cog_href=f"{scene_id}.tif",
        target_resolution=target_resolution,
        flag_href=f"{scene_id}.flag.tif" if path_flag else None,
        provenance_href="provenance.json",
    )
    stac_gen = 0 if not exists(stac_uri) else None
    stac_bytes = json.dumps(stac_item, indent=2).encode("utf-8")
    atomic_write(stac_uri, stac_bytes, overwrite=True,
                 if_generation_match=stac_gen)

    # 3. Completion marker
    from datetime import UTC, datetime
    completed_at = datetime.now(UTC).isoformat()
    comp_gen = 0 if not exists(comp_uri) else None
    atomic_write(
        comp_uri,
        json.dumps({"published_at": completed_at, "run_id": run_id}, indent=2),
        overwrite=True,
        if_generation_match=comp_gen,
    )


if __name__ == "__main__":
    raise SystemExit(main())
