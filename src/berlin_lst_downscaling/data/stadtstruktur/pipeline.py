"""Publish the Stadtstruktur evaluation release.

Source, purpose, grain
----------------------
Source: Berlin GDI WFS for Umweltatlas 06.07 / 06.08, ISU5 block parts.
Purpose: a versioned evaluation layer for descriptive statistics and
stratified comparison. It is not a V3 feature channel.
Grain: one source row is one ISU5 block or block part. Share rows are the
area of one official ``typ`` inside one canonical 100 m cell or 1,000 m block.

Run
---
``uv run python -m berlin_lst_downscaling.data.stadtstruktur.pipeline``
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import contains_xy

from berlin_lst_downscaling.common.grid import canon_grid_100m
from berlin_lst_downscaling.data.io.storage import atomic_upload
from berlin_lst_downscaling.data.stadtstruktur.aggregate import (
    area_shares,
    audit_cell_tiling,
    block_shares,
    repair_polygons,
)
from berlin_lst_downscaling.data.stadtstruktur.fetch import crosswalk, fetch_vintage
from berlin_lst_downscaling.data.stadtstruktur.sources import (
    CANDIDATE_B,
    CANDIDATE_B_JUSTIFICATION,
    CANDIDATE_B_LABELS,
    CANDIDATE_B_STATUS,
    CELL_SIZE_M,
    LICENSE_ID,
    LICENSE_NAME,
    LICENSE_URL,
    OPEN_DATA_RECORDS,
    PUBLISHER,
    RELEASE_ID,
    SCENE_YEAR_COVERAGE,
    TARGET_CRS,
    VINTAGES,
    VintageSource,
)
from berlin_lst_downscaling.data.stadtstruktur.tables import (
    as_int_required,
    float_sum,
    frame_of,
    series_of,
    true_count,
)

_OVERLAP_FRACTION_LIMIT = 0.001
_SAMPLE_DIFF_LIMIT_M2 = 1.0
_RASTER_RELATIVE_LIMIT = 0.05


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_aoi(path: Path) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(path)
    if aoi.crs is None:
        raise RuntimeError(f"{path} has no CRS")
    aoi = aoi.to_crs(TARGET_CRS)
    aoi["geometry"] = aoi.geometry.make_valid()
    dissolved = gpd.GeoDataFrame(
        frame_of(aoi.dissolve()), geometry="geometry", crs=TARGET_CRS
    ).reset_index(drop=True)
    dissolved = gpd.GeoDataFrame(dissolved, geometry="geometry", crs=TARGET_CRS)
    if not bool(dissolved.geometry.iloc[0].is_valid):
        raise RuntimeError("Berlin AOI geometry is invalid after repair")
    return dissolved


def _mark_inside_aoi(cells: pd.DataFrame, aoi: gpd.GeoDataFrame, grid) -> pd.DataFrame:
    """Flag cells whose centre lies inside the Berlin boundary.

    ``data/boundaries/aoi_100m.tif`` is not on the canonical 100 m lattice
    (its origin and pixel size differ), so the vector boundary is the mask.
    """
    transform = grid.transform
    columns = cells["col"].to_numpy(dtype=float)
    rows = cells["row"].to_numpy(dtype=float)
    x = float(transform.c) + (columns + 0.5) * CELL_SIZE_M
    y = float(transform.f) - (rows + 0.5) * CELL_SIZE_M
    marked = cells.copy()
    marked["inside_aoi"] = np.asarray(contains_xy(aoi.geometry.iloc[0], x, y), dtype=bool)
    return marked


def _raster_area_check(frame: gpd.GeoDataFrame) -> dict:
    """1 m raster area against shapely area for small polygons."""
    small = frame.loc[frame.geometry.area < 20_000].reset_index(drop=True)
    if small.empty:
        return {"sample_size": 0, "max_relative_diff": None}
    count = min(8, len(small))
    picked = np.random.default_rng(1).choice(len(small), size=count, replace=False)
    worst = 0.0
    for index in picked:
        geom = small.geometry.iloc[int(index)]
        minx, miny, maxx, maxy = geom.bounds
        width = max(1, int(np.ceil(maxx - minx)) + 2)
        height = max(1, int(np.ceil(maxy - miny)) + 2)
        transform = from_origin(minx - 1.0, maxy + 1.0, 1.0, 1.0)
        burned = rasterize(
            [(geom, 1)], out_shape=(height, width), transform=transform, dtype="uint8"
        )
        raster_area = float(np.asarray(burned).sum())
        vector_area = float(geom.area)
        if vector_area > 0:
            worst = max(worst, abs(raster_area - vector_area) / vector_area)
    return {"sample_size": int(count), "max_relative_diff": worst}


def _prevalence(frame: gpd.GeoDataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    measured = frame.assign(_area_m2=frame.geometry.area)
    geometry_area = series_of(measured.groupby(["vintage", "typ"], dropna=False)["_area_m2"].sum())
    flalle = series_of(measured.groupby(["vintage", "typ"], dropna=False)["flalle_m2"].sum())
    counts = series_of(measured.groupby(["vintage", "typ"], dropna=False).size())
    grid_area = series_of(shares.groupby("typ")["area_m2"].sum())
    name_sizes = series_of(
        frame.groupby(["typ", "typklar", "ststrnr", "ststrname"], dropna=False).size()
    )
    names = frame_of(name_sizes.reset_index(name="_n")).drop(columns="_n").drop_duplicates("typ")
    table = pd.DataFrame(
        {"n_polygons": counts, "area_m2_geometry": geometry_area, "area_m2_flalle": flalle}
    )
    table = frame_of(table.reset_index()).merge(names, on="typ", how="left")
    lookup = {as_int_required(key): float(value) for key, value in grid_area.items()}
    table["area_m2_on_grid"] = [lookup.get(as_int_required(value), 0.0) for value in table["typ"]]
    total = float_sum(series_of(table["area_m2_geometry"]))
    table["share_of_source_area"] = table["area_m2_geometry"] / total
    return table.sort_values("typ", kind="mergesort")


def _coverage(frame: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> dict:
    """Overlap and AOI gap. Street land is expected to remain outside the blocks."""
    source_area = float(frame.geometry.area.sum())
    union = frame.geometry.union_all()
    union_area = float(union.area)
    aoi_geom = aoi.geometry.iloc[0]
    aoi_area = float(aoi_geom.area)
    inside = union.intersection(aoi_geom)
    gap = aoi_geom.difference(union)
    outside = union.difference(aoi_geom)
    return {
        "source_area_m2": source_area,
        "union_area_m2": union_area,
        "overlap_m2": source_area - union_area,
        "overlap_fraction": (source_area - union_area) / source_area,
        "aoi_area_m2": aoi_area,
        "source_inside_aoi_m2": float(inside.area),
        "aoi_uncovered_m2": float(gap.area),
        "source_outside_aoi_m2": float(outside.area),
        "aoi_uncovered_fraction": float(gap.area) / aoi_area,
    }


def _gates(vintage_report: dict) -> list[str]:
    failures: list[str] = []
    repair = vintage_report["repair"]
    coverage = vintage_report["coverage"]
    grid = vintage_report["grid"]
    if vintage_report["feature_count"] != vintage_report["service_number_matched"]:
        failures.append("feature count differs from the WFS numberMatched value")
    if vintage_report["duplicate_schluessel"]:
        failures.append("duplicate ISU5 schluessel")
    if vintage_report["null_schluessel"] or vintage_report["null_typ"]:
        failures.append("null ISU5 key or typ")
    if repair["invalid_after"] or repair["non_polygonal_or_empty_dropped"]:
        failures.append("geometry repair dropped or left invalid polygons")
    if coverage["overlap_fraction"] > _OVERLAP_FRACTION_LIMIT:
        failures.append("polygon overlap exceeds 0.1 percent of source area")
    residual = abs(
        grid["share_area_m2"] + grid["outside_or_boundary_residual_m2"] - grid["geometry_area_m2"]
    )
    if residual > 1.0:
        failures.append("cell areas do not conserve polygon area")
    if abs(vintage_report["block_area_m2"] - grid["share_area_m2"]) > 1.0:
        failures.append("1,000 m areas do not conserve 100 m areas")
    if vintage_report["cell_tiling_audit"]["max_abs_diff_m2"] > _SAMPLE_DIFF_LIMIT_M2:
        failures.append("cell tiling disagrees with the grid-envelope area")
    raster = vintage_report["raster_area_audit"]["max_relative_diff"]
    if raster is not None and raster > _RASTER_RELATIVE_LIMIT:
        failures.append("1 m raster area disagrees with shapely area")
    if grid["ambiguous_typ_codes"]:
        failures.append("one typ maps to more than one 06.07 code or name")
    return failures


def publish_vintage(
    source: VintageSource,
    release_dir: Path,
    aoi: gpd.GeoDataFrame,
    grid,
) -> dict:
    """Fetch, check, and write one vintage. Returns the vintage section of the report."""
    print(f"stadtstruktur {source.vintage}: downloading", flush=True)
    frame, receipt = fetch_vintage(source)
    frame, repair = repair_polygons(frame)
    if str(frame.crs) != TARGET_CRS:
        raise RuntimeError(f"{source.vintage}: CRS {frame.crs}")

    print(f"stadtstruktur {source.vintage}: intersecting canonical grid", flush=True)
    shares, cells, grid_summary = area_shares(frame, grid)
    shares.insert(0, "vintage", source.vintage)
    cells.insert(0, "vintage", source.vintage)
    cells = _mark_inside_aoi(cells, aoi, grid)
    blocks = block_shares(shares)
    blocks.insert(0, "vintage", source.vintage)

    print(f"stadtstruktur {source.vintage}: coverage union", flush=True)
    coverage = _coverage(frame, aoi)
    prevalence = _prevalence(frame, shares)
    mixed_ids = cells.loc[np.asarray(cells["mixed"], dtype=bool), "cell_id"]
    mixed_area = float_sum(series_of(shares.loc[shares["cell_id"].isin(mixed_ids), "area_m2"]))
    covered_area = float_sum(series_of(shares["area_m2"]))
    flalle = prevalence["area_m2_flalle"].to_numpy(dtype=float)
    geom = prevalence["area_m2_geometry"].to_numpy(dtype=float)
    relative = np.abs(flalle - geom) / np.maximum(geom, 1.0)
    report = {
        "vintage": source.vintage,
        "reference_date": source.reference_date,
        "type_name": source.type_name,
        "wfs_url": source.wfs_url,
        "feature_count": int(len(frame)),
        "service_number_matched": receipt["service_number_matched"],
        "page_sha256": receipt["page_sha256"],
        "duplicate_schluessel": true_count(series_of(frame["schluessel"].duplicated())),
        "null_schluessel": true_count(series_of(frame["schluessel"].isna())),
        "null_typ": true_count(series_of(frame["typ"].isna())),
        "null_ststrnr": true_count(series_of(frame["ststrnr"].isna())),
        "repair": repair,
        "coverage": coverage,
        "grid": grid_summary,
        "block_area_m2": float_sum(series_of(blocks["area_m2"])),
        "n_blocks": int(len(blocks[["block_row", "block_col"]].drop_duplicates())),
        "mixed_cell_area_m2": mixed_area,
        "mixed_cell_area_fraction": mixed_area / covered_area,
        "n_typ_histogram": {
            str(as_int_required(key)): as_int_required(value)
            for key, value in series_of(cells["n_typ"].value_counts().sort_index()).items()
        },
        "cells_inside_aoi": true_count(series_of(cells["inside_aoi"])),
        "cells_outside_aoi": true_count(series_of(~series_of(cells["inside_aoi"]))),
        "flalle_vs_geometry_median_relative_diff": float(np.median(relative)),
        "flalle_vs_geometry_max_relative_diff": float(np.max(relative)),
        "cell_tiling_audit": audit_cell_tiling(frame, grid),
        "raster_area_audit": _raster_area_check(frame),
        "notes": source.notes,
    }
    report["failures"] = _gates(report)

    vintage_dir = release_dir / "source" / str(source.vintage)
    vintage_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "qa").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(vintage_dir / "blocks.parquet", index=False)
    _write_json(vintage_dir / "receipt.json", receipt)
    grid_dir = release_dir / "grid" / "100m" / str(source.vintage)
    grid_dir.mkdir(parents=True, exist_ok=True)
    shares.to_parquet(grid_dir / "shares.parquet", index=False)
    cells.to_parquet(grid_dir / "cells.parquet", index=False)
    block_dir = release_dir / "grid" / "1000m" / str(source.vintage)
    block_dir.mkdir(parents=True, exist_ok=True)
    blocks.to_parquet(block_dir / "shares.parquet", index=False)
    prevalence.to_parquet(
        release_dir / "qa" / f"class_prevalence_{source.vintage}.parquet", index=False
    )
    crosswalk(frame).to_parquet(
        release_dir / "qa" / f"crosswalk_{source.vintage}.parquet", index=False
    )
    print(
        f"stadtstruktur {source.vintage}: {len(frame)} blocks, "
        f"{report['grid']['n_cells']} cells, failures={report['failures'] or 'none'}",
        flush=True,
    )
    return report


def _candidate_payload(prevalence: pd.DataFrame) -> dict:
    observed = sorted({int(value) for value in prevalence["typ"].dropna().unique()})
    unmapped = [code for code in observed if code not in CANDIDATE_B]
    rows = []
    for code in observed:
        names = prevalence.loc[prevalence["typ"] == code, "typklar"].dropna().unique()
        rows.append(
            {
                "typ": code,
                "typklar": None if len(names) == 0 else str(names[0]),
                "proposal_b": CANDIDATE_B.get(code),
            }
        )
    return {
        "status": CANDIDATE_B_STATUS,
        "candidate_a": (
            "Official Umweltatlas 06.07 ststrnr, including a null code on water. "
            "This is the publisher's consolidation of area types, not a new grouping."
        ),
        "candidate_b": {
            "justification": CANDIDATE_B_JUSTIFICATION,
            "labels": CANDIDATE_B_LABELS,
            "typ_to_group": {str(code): group for code, group in sorted(CANDIDATE_B.items())},
            "observed_mapping": rows,
            "unmapped_observed_typ": unmapped,
            "complete_for_observed_codes": not unmapped,
        },
        "approval": (
            "Final analytical classes are not frozen. Both candidates wait for scientific review."
        ),
    }


def build_release(release_dir: Path, aoi_path: Path, years: list[int]) -> dict:
    """Build the local release directory and the compact report."""
    grid = canon_grid_100m()
    aoi = _load_aoi(aoi_path)
    selected = [source for source in VINTAGES if source.vintage in years]
    if len(selected) != len(years):
        raise KeyError(f"unknown vintage in {years}")
    reports = [publish_vintage(source, release_dir, aoi, grid) for source in selected]
    prevalence_paths = sorted((release_dir / "qa").glob("class_prevalence_*.parquet"))
    prevalence = pd.concat([pd.read_parquet(path) for path in prevalence_paths], ignore_index=True)
    candidates = _candidate_payload(prevalence)
    failures = [item for report in reports for item in report["failures"]]
    payload = {
        "release_id": RELEASE_ID,
        "accessed_at": datetime.now(UTC).isoformat(),
        "publisher": PUBLISHER,
        "license_id": LICENSE_ID,
        "license_name": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "open_data_records": {str(key): value for key, value in OPEN_DATA_RECORDS.items()},
        "crs": TARGET_CRS,
        "cell_size_m": CELL_SIZE_M,
        "block_size_m": CELL_SIZE_M * 10,
        "grid_shape_row_col": [int(grid.shape.y), int(grid.shape.x)],
        "scene_year_coverage": {str(key): value for key, value in SCENE_YEAR_COVERAGE.items()},
        "temporal_limit": (
            "No open Stadtstruktur vintage exists for 2017–2019 or for 2025. "
            "This release does not assign those scene years to a neighbouring vintage."
        ),
        "street_land": (
            "ISU5 street polygons are a separate layer and are not part of Stadtstruktur. "
            "Uncovered land inside the Berlin boundary is expected and is not given a class."
        ),
        "mixed_cell_semantics": (
            "shares.parquet keeps every overlapping typ. cells.parquet majority_* columns "
            "are diagnostics. A mixed cell is not stored as one class."
        ),
        "vintages": reports,
        "candidate_groupings": candidates,
        "qa_pass": not failures,
        "failures": failures,
    }
    _write_json(release_dir / "qa" / "report.json", payload)
    _write_json(release_dir / "qa" / "candidate_groupings.json", candidates)
    if payload["qa_pass"]:
        _write_json(
            release_dir / "complete.json",
            {
                "release_id": RELEASE_ID,
                "completed_at": datetime.now(UTC).isoformat(),
                "qa_pass": True,
            },
        )
    return payload


def upload_release(release_dir: Path, output_root: str) -> None:
    """Upload the local release. ``complete.json`` is uploaded last."""
    root = output_root.rstrip("/")
    files = sorted(
        path for path in release_dir.rglob("*") if path.is_file() and path.name != "complete.json"
    )
    for path in files:
        relative = path.relative_to(release_dir).as_posix()
        print(f"upload {relative}", flush=True)
        atomic_upload(path, f"{root}/{relative}")
    complete = release_dir / "complete.json"
    if complete.exists():
        atomic_upload(complete, f"{root}/complete.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the Stadtstruktur evaluation release")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--aoi", type=Path, default=Path("data/boundaries/berlin_landesgrenze.geojson")
    )
    parser.add_argument("--vintages", default="2020,2021,2022,2023,2024")
    args = parser.parse_args()
    years = [int(part) for part in args.vintages.split(",") if part]
    payload = build_release(args.release_dir, args.aoi, years)
    if args.output_root and payload["qa_pass"]:
        upload_release(args.release_dir, args.output_root)
    if not payload["qa_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
