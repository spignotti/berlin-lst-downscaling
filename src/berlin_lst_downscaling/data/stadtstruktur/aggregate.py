"""Area-weighted shares of ISU5 polygons on the canonical grids.

A 100 m cell keeps every overlapping ``typ``. Mixed cells are rows with
``n_typ > 1`` in the cell table. No majority label replaces those rows.
1,000 m blocks are the 10 by 10 groups of those cells aligned to the same
origin. A block on the grid edge contains fewer than 100 cells.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.geobox import GeoBox
from shapely import area, box, intersection, make_valid, union_all
from shapely.geometry.base import BaseGeometry

from berlin_lst_downscaling.data.stadtstruktur.sources import BLOCK_CELLS, CELL_SIZE_M
from berlin_lst_downscaling.data.stadtstruktur.tables import as_int, float_sum, frame_of, series_of

_EDGE_EPS = 1e-6


def repair_polygons(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Repair invalid polygons and drop geometries that are not polygonal."""
    invalid_before = int((~frame.geometry.is_valid).sum())
    empty_before = int(frame.geometry.is_empty.sum())
    repaired = make_valid(frame.geometry.to_numpy())
    polygonal: list[BaseGeometry | None] = []
    dropped = 0
    for geom in repaired:
        if geom is None or geom.is_empty:
            polygonal.append(None)
            dropped += 1
            continue
        kind = geom.geom_type
        if kind in {"Polygon", "MultiPolygon"}:
            polygonal.append(geom)
            continue
        if kind == "GeometryCollection":
            parts = [part for part in geom.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
            if not parts:
                polygonal.append(None)
                dropped += 1
                continue
            polygonal.append(union_all(parts))
            continue
        polygonal.append(None)
        dropped += 1
    keep = [geom is not None for geom in polygonal]
    out = frame.loc[keep].copy()
    out = out.set_geometry([geom for geom in polygonal if geom is not None])
    empty = np.asarray(out.geometry.is_empty, dtype=bool)
    out = out.loc[~empty].reset_index(drop=True)
    invalid_after = int(np.count_nonzero(~np.asarray(out.geometry.is_valid, dtype=bool)))
    stats = {
        "invalid_before": invalid_before,
        "empty_before": empty_before,
        "non_polygonal_or_empty_dropped": dropped,
        "invalid_after": invalid_after,
        "feature_count_after": int(len(out)),
    }
    return out, stats


def _window(
    geom: BaseGeometry, xoff: float, yoff: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    minx, miny, maxx, maxy = geom.bounds
    col0 = math.floor((minx - xoff) / CELL_SIZE_M)
    col1 = math.floor((maxx - xoff - _EDGE_EPS) / CELL_SIZE_M)
    row0 = math.floor((yoff - maxy) / CELL_SIZE_M)
    row1 = math.floor((yoff - miny - _EDGE_EPS) / CELL_SIZE_M)
    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(width - 1, col1)
    row1 = min(height - 1, row1)
    if col1 < col0 or row1 < row0:
        return None
    return col0, col1, row0, row1


def area_shares(frame: gpd.GeoDataFrame, grid: GeoBox) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Intersect source polygons with the canonical 100 m grid.

    Returns the long share table, the per-cell diagnostic table, and a
    coverage summary. ``majority_*`` columns are diagnostics. The share
    table is the evaluation product.
    """
    transform = grid.transform
    if abs(abs(transform.a) - CELL_SIZE_M) > 1e-6 or abs(abs(transform.e) - CELL_SIZE_M) > 1e-6:
        raise ValueError(f"expected a {CELL_SIZE_M} m grid, got {transform.a} by {transform.e}")
    xoff = float(transform.c)
    yoff = float(transform.f)
    width = int(grid.shape.x)
    height = int(grid.shape.y)

    counts: dict[int, dict[tuple[str | None, int | None, str | None], int]] = {}
    columns = frame[["typ", "typklar", "ststrnr", "ststrname"]]
    for typ, label, st_code, st_name in columns.itertuples(index=False, name=None):
        code = as_int(typ)
        if code is None:
            continue
        parsed = (
            None if label is None else str(label),
            as_int(st_code),
            None if st_name is None else str(st_name),
        )
        bucket = counts.setdefault(code, {})
        bucket[parsed] = bucket.get(parsed, 0) + 1
    name_for: dict[int, tuple[str | None, int | None, str | None]] = {}
    ambiguous: list[int] = []
    label_disagreements: list[dict[str, object]] = []
    for code, variants in counts.items():
        st_codes = {item[1] for item in variants}
        labels = {item[0] for item in variants}
        modal = max(variants.items(), key=lambda item: item[1])[0]
        name_for[code] = modal
        if len(st_codes) > 1:
            ambiguous.append(code)
        if len(labels) > 1 or len(st_codes) > 1:
            label_disagreements.append(
                {
                    "typ": code,
                    "variants": [
                        {
                            "typklar": label,
                            "ststrnr": st_code,
                            "ststrname": st_name,
                            "n_polygons": count,
                        }
                        for (label, st_code, st_name), count in sorted(
                            variants.items(), key=lambda item: -item[1]
                        )
                    ],
                }
            )

    totals: dict[tuple[int, int, int], float] = {}
    outside_m2 = 0.0
    geometry_m2 = 0.0
    for geom, typ in zip(frame.geometry, frame["typ"].to_numpy(), strict=True):
        code_value = as_int(typ)
        if code_value is None or geom is None:
            continue
        geometry_m2 += float(geom.area)
        window = _window(geom, xoff, yoff, width, height)
        if window is None:
            outside_m2 += float(geom.area)
            continue
        col0, col1, row0, row1 = window
        cols = np.arange(col0, col1 + 1)
        rows = np.arange(row0, row1 + 1)
        col_grid, row_grid = np.meshgrid(cols, rows)
        x0 = xoff + col_grid * CELL_SIZE_M
        y1 = yoff - row_grid * CELL_SIZE_M
        pieces = intersection(box(x0, y1 - CELL_SIZE_M, x0 + CELL_SIZE_M, y1), geom)
        piece_area = area(pieces)
        code = code_value
        flat_area = np.asarray(piece_area, dtype=np.float64).ravel()
        flat_row = row_grid.ravel()
        flat_col = col_grid.ravel()
        covered = 0.0
        for row, col, part in zip(flat_row, flat_col, flat_area, strict=True):
            if part <= 0.0:
                continue
            key = (int(row), int(col), code)
            totals[key] = totals.get(key, 0.0) + float(part)
            covered += float(part)
        outside_m2 += float(geom.area) - covered

    records = []
    for (row, col, code), part in totals.items():
        typklar, ststrnr, ststrname = name_for[code]
        records.append(
            {
                "row": row,
                "col": col,
                "cell_id": f"{row:04d}_{col:04d}",
                "typ": code,
                "typklar": typklar,
                "ststrnr": ststrnr,
                "ststrname": ststrname,
                "area_m2": part,
                "share_of_cell": part / (CELL_SIZE_M * CELL_SIZE_M),
            }
        )
    shares = pd.DataFrame.from_records(
        records,
        columns=[
            "row",
            "col",
            "cell_id",
            "typ",
            "typklar",
            "ststrnr",
            "ststrname",
            "area_m2",
            "share_of_cell",
        ],
    )
    if shares.empty:
        raise RuntimeError("grid intersection produced no covered cells")

    cell_area = shares.groupby(["row", "col", "cell_id"], sort=False)["area_m2"].transform("sum")
    shares["share_of_covered"] = shares["area_m2"] / cell_area
    shares["block_row"] = shares["row"] // BLOCK_CELLS
    shares["block_col"] = shares["col"] // BLOCK_CELLS
    shares["block_id"] = (
        shares["block_row"].astype(str).str.zfill(3)
        + "_"
        + shares["block_col"].astype(str).str.zfill(3)
    )

    cells = _cell_diagnostics(shares)
    summary = {
        "geometry_area_m2": geometry_m2,
        "share_area_m2": float_sum(series_of(shares["area_m2"])),
        "outside_or_boundary_residual_m2": outside_m2,
        "n_share_rows": int(len(shares)),
        "n_cells": int(len(cells)),
        "n_mixed_cells": int(np.count_nonzero(np.asarray(cells["mixed"], dtype=bool))),
        "ambiguous_typ_codes": ambiguous,
        "label_disagreements": label_disagreements,
        "grid_shape": [height, width],
        "grid_origin_xy": [xoff, yoff],
    }
    return shares, cells, summary


def _cell_diagnostics(shares: pd.DataFrame) -> pd.DataFrame:
    keys = ["row", "col", "cell_id"]
    covered = series_of(shares.groupby(keys, sort=False)["area_m2"].sum()).rename("covered_m2")
    n_typ = series_of(shares.groupby(keys, sort=False)["typ"].nunique()).rename("n_typ")
    n_ststr = series_of(shares.groupby(keys, sort=False)["ststrnr"].nunique(dropna=True)).rename(
        "n_ststr"
    )
    top = frame_of(shares.loc[series_of(shares.groupby(keys, sort=False)["area_m2"].idxmax())])
    cells = top.merge(covered.reset_index(), on=keys, how="left")
    cells = cells.merge(n_typ.reset_index(), on=keys, how="left")
    cells = cells.merge(n_ststr.reset_index(), on=keys, how="left")
    cell_m2 = float(CELL_SIZE_M * CELL_SIZE_M)
    cells["covered_fraction"] = cells["covered_m2"] / cell_m2
    cells["mixed"] = cells["n_typ"] > 1
    cells["majority_typ"] = cells["typ"].astype(int)
    cells["majority_typklar"] = cells["typklar"]
    cells["majority_share_of_covered"] = cells["area_m2"] / cells["covered_m2"]
    cells["majority_share_of_cell"] = cells["area_m2"] / cell_m2
    cells["majority_ststrnr"] = cells["ststrnr"]
    keep = [
        "row",
        "col",
        "cell_id",
        "block_row",
        "block_col",
        "covered_m2",
        "covered_fraction",
        "n_typ",
        "n_ststr",
        "mixed",
        "majority_typ",
        "majority_typklar",
        "majority_share_of_covered",
        "majority_share_of_cell",
        "majority_ststrnr",
    ]
    return frame_of(cells[keep].reset_index(drop=True))


def audit_cell_tiling(frame: gpd.GeoDataFrame, grid: GeoBox, *, n: int = 20) -> dict:
    """Compare cell-summed area with one envelope intersection for a sample.

    The envelope path does not use the per-cell window. Both measure planar
    area inside the canonical rectangle. A fixed seed keeps the sample stable.
    """
    transform = grid.transform
    xoff = float(transform.c)
    yoff = float(transform.f)
    width = int(grid.shape.x)
    height = int(grid.shape.y)
    envelope = box(xoff, yoff - height * CELL_SIZE_M, xoff + width * CELL_SIZE_M, yoff)
    count = min(n, len(frame))
    picked = np.random.default_rng(0).choice(len(frame), size=count, replace=False)
    worst = 0.0
    for index in picked:
        geom = frame.geometry.iloc[int(index)]
        direct = float(geom.intersection(envelope).area)
        window = _window(geom, xoff, yoff, width, height)
        if window is None:
            tiled = 0.0
        else:
            col0, col1, row0, row1 = window
            cols = np.arange(col0, col1 + 1)
            rows = np.arange(row0, row1 + 1)
            col_grid, row_grid = np.meshgrid(cols, rows)
            x0 = xoff + col_grid * CELL_SIZE_M
            y1 = yoff - row_grid * CELL_SIZE_M
            tiled = float(
                area(intersection(box(x0, y1 - CELL_SIZE_M, x0 + CELL_SIZE_M, y1), geom)).sum()
            )
        worst = max(worst, abs(direct - tiled))
    return {"sample_size": int(count), "max_abs_diff_m2": worst}


def block_shares(shares: pd.DataFrame) -> pd.DataFrame:
    """Sum 100 m share areas into origin-aligned 1,000 m blocks."""
    keys = ["block_row", "block_col", "block_id", "typ", "typklar", "ststrnr", "ststrname"]
    summed = series_of(shares.groupby(keys, dropna=False, sort=False)["area_m2"].sum())
    grouped = frame_of(summed.reset_index())
    nominal = float(BLOCK_CELLS * CELL_SIZE_M) ** 2
    grouped["share_of_nominal_block"] = grouped["area_m2"] / nominal
    covered = grouped.groupby(["block_row", "block_col"], sort=False)["area_m2"].transform("sum")
    grouped["share_of_covered"] = grouped["area_m2"] / covered
    grouped["n_typ_in_block"] = grouped.groupby(["block_row", "block_col"], sort=False)[
        "typ"
    ].transform("nunique")
    return grouped
