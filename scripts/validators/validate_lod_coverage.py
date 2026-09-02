# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Independent validator for LoD2 source-coverage evidence (read-only).

Reconstructs the per-vintage 1 km tile coverage from the immutable
published provenance — raw archive manifests (2017/2021/2022) and the LoD
provenance tile receipts (2024) — and cross-checks it against the
published ``lod2_morphology`` source COGs:

- artifact integrity: expected member/tile counts, parseable tile keys,
  non-empty 2017 LoD1 ∩ LoD2 tile intersection,
- no mixed finite/NaN state across the four LoD bands in any source COG
  (the bands are jointly NaN or jointly finite by construction),
- finite source values lie inside the reconstructed coverage (a boundary
  pixel outside nominal coverage is reported as a diagnostic, never
  zeroed by the composer — finite values always win),
- semantic counters for acceptance: building, covered-no-building,
  source-gap, and inside-AOI gap pixels.

The validator never writes anything and never imports the feature
composer or the coverage resolver implementation; it re-derives the
coverage from the same published evidence independently.

Usage
-----
    uv run python scripts/validators/validate_lod_coverage.py
    uv run python scripts/validators/validate_lod_coverage.py \\
        --static-sources-root gs://berlin-lst-data/static/sources/full \\
        --bbox 13.35,52.45,13.55,52.60
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.warp import Resampling
from rasterio.warp import reproject as rwarp_reproject
from rasterio.windows import Window

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.io import read_bytes
from berlin_lst_downscaling.data.qa.stage1_raw import _window_offset, analysis_grid_10m

_LOD1_TILE_RE = re.compile(r"LoD1_(\d{3})_(\d{4})_")
_LOD2_TILE_RE = re.compile(r"LoD2_(?:\d+_)?(\d{3})_(\d{4})_")

_LOD_BANDS = (1, 2, 3, 4)  # building_height_mean/std/coverage/max
_EXPECTED_MEMBERS = {2017: 1006, 2021: 928, 2022: 928}
_EXPECTED_TILES_2024 = 923


def _tile_key(member: str, *, lod1: bool) -> str:
    rx = _LOD1_TILE_RE if lod1 else _LOD2_TILE_RE
    m = rx.search(member)
    if not m:
        raise ValueError(f"cannot parse tile key from member {member!r}")
    return f"{m.group(1)}_{m.group(2)}"


def _manifest_members(uri: str) -> list[str]:
    payload = json.loads(read_bytes(uri))
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"raw archive manifest {uri} has no members list")
    return [str(x) for x in members]


def _coverage_mask(tile_keys: set[str], grid: GeoBox) -> np.ndarray:
    """Recompute the centre-in-tile coverage mask (independent of the composer)."""
    key_codes = np.array(
        [int(e) * 100_000 + int(n) for e, n in (k.split("_") for k in tile_keys)],
        dtype=np.int64,
    )
    centers_x = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    centers_y = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    code = (centers_x // 1000.0).astype(np.int64)[None, :] * 100_000
    code = code + (centers_y // 1000.0).astype(np.int64)[:, None]
    return np.isin(code, key_codes)


def _load_aoi(uri: str, grid: GeoBox) -> np.ndarray | None:
    try:
        with rasterio.open(uri) as src:
            src_arr = src.read(1).astype(np.uint8)
            src_crs, src_transform, src_nodata = src.crs, src.transform, src.nodata
    except Exception as exc:  # AOI is optional for diagnostics
        print(f"WARNING: cannot read AOI {uri}: {exc}")
        return None

    dst = np.zeros((grid.shape.y, grid.shape.x), dtype=np.uint8)
    rwarp_reproject(
        source=src_arr,
        src_crs=src_crs,
        src_transform=src_transform,
        src_nodata=src_nodata,
        destination=dst,
        dst_crs=grid.crs,
        dst_transform=grid.transform,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst == 1


def _provenance_tile_count(root: str, vintage: int) -> int | None:
    """Return the published provenance tile_count for a vintage (or None)."""
    try:
        prov = json.loads(
            read_bytes(f"{root}/ard/static/sources/lod2_morphology/{vintage}/provenance.json")
        )
    except Exception:
        return None
    sm = prov.get("source_metadata") or {}
    qa = prov.get("qa_stats") or {}
    return sm.get("tile_count") or qa.get("tile_count")


def _check_vintage(
    vintage: int,
    tile_keys: set[str],
    cog_uri: str,
    grid: GeoBox,
    aoi: np.ndarray | None,
    receipt_tile_count: int | None,
    findings: list[str],
) -> dict:
    """Check one LoD source COG against its reconstructed coverage."""
    # The coverage roster must match the processing receipt: a manifest
    # naming a tile the run never processed would otherwise let the
    # composer publish zeros over a true data gap.
    if receipt_tile_count is not None and receipt_tile_count != len(tile_keys):
        findings.append(
            f"vintage {vintage}: provenance tile_count {receipt_tile_count} != "
            f"coverage keys {len(tile_keys)}"
        )
    cov = _coverage_mask(tile_keys, grid)
    with rasterio.open(cog_uri) as src:
        off = _window_offset(src, grid, 10.0)
        win = Window.from_slices((off[1], off[1] + grid.shape.y), (off[0], off[0] + grid.shape.x))
        bands = src.read(_LOD_BANDS, window=win).astype(np.float32)  # (4, H, W)

    nan = np.isnan(bands)
    all_nan = np.all(nan, axis=0)
    any_nan = np.any(nan, axis=0)
    mixed = any_nan & ~all_nan
    if np.any(mixed):
        n = int(np.sum(mixed))
        findings.append(f"vintage {vintage}: mixed finite/NaN LoD state on {n} px")
        raise SystemExit(1)

    finite = ~all_nan
    finite_outside = int(np.sum(finite & ~cov))
    if finite_outside:
        # Boundary rasterisation can put a footprint pixel outside the
        # nominal tile list; the composer preserves finite values. Large
        # counts would indicate a coverage defect.
        print(f"  diagnostic: {finite_outside} finite px outside nominal coverage")

    gap = all_nan & ~cov
    covered_no_building = all_nan & cov
    counts = {
        "building_px": int(np.sum(finite)),
        "covered_no_building_px": int(np.sum(covered_no_building)),
        "source_gap_px": int(np.sum(gap)),
        "source_gap_inside_aoi_px": int(np.sum(gap & aoi)) if aoi is not None else None,
    }
    print(
        f"vintage {vintage}: building {counts['building_px']:,} | "
        f"covered-no-building {counts['covered_no_building_px']:,} | "
        f"gap {counts['source_gap_px']:,}"
        + (f" | gap-in-AOI {counts['source_gap_inside_aoi_px']:,}" if aoi is not None else "")
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--static-sources-root",
        default="gs://berlin-lst-data/static/sources/full",
        help="Published static source root",
    )
    parser.add_argument(
        "--aoi",
        default="data/boundaries/aoi_10m.tif",
        help="Berlin AOI mask (10 m, EPSG:25833)",
    )
    parser.add_argument(
        "--bbox",
        default=None,
        help="Optional WGS84 bbox west,south,east,north for a bounded check",
    )
    args = parser.parse_args()

    root = args.static_sources_root.rstrip("/")
    if args.bbox:
        grid = analysis_grid_10m(tuple(float(v) for v in args.bbox.split(",")))
    else:
        grid = canon_grid_10m()
    aoi = _load_aoi(args.aoi, grid)

    findings: list[str] = []
    all_counts: dict[int, dict] = {}

    # ── 2021 / 2022 ──────────────────────────────────────────────────
    for vintage in (2021, 2022):
        uri = f"{root}/ard/static/sources/lod_vintages/raw_manifest_{vintage}.json"
        members = _manifest_members(uri)
        if len(members) != _EXPECTED_MEMBERS[vintage]:
            findings.append(
                f"raw manifest {uri}: {len(members)} members, expected {_EXPECTED_MEMBERS[vintage]}"
            )
        keys = {_tile_key(m, lod1=False) for m in members}
        cog = f"{root}/ard/static/sources/lod2_morphology/{vintage}/lod2_morphology_{vintage}.tif"
        all_counts[vintage] = _check_vintage(
            vintage, keys, cog, grid, aoi, _provenance_tile_count(root, vintage), findings
        )

    # ── 2017: LoD1 ∩ LoD2 ────────────────────────────────────────────
    lod1_uri = f"{root}/ard/static/sources/lod_vintages/raw_manifest_2017.json"
    lod1_members = _manifest_members(lod1_uri)
    if len(lod1_members) != _EXPECTED_MEMBERS[2017]:
        findings.append(
            f"raw manifest {lod1_uri}: {len(lod1_members)} members, "
            f"expected {_EXPECTED_MEMBERS[2017]}"
        )
    lod1_keys = {_tile_key(m, lod1=True) for m in lod1_members}
    lod2_2021 = {
        _tile_key(m, lod1=False)
        for m in _manifest_members(f"{root}/ard/static/sources/lod_vintages/raw_manifest_2021.json")
    }
    keys_2017 = lod1_keys & lod2_2021
    if not keys_2017:
        findings.append("2017: empty LoD1 ∩ LoD2 tile intersection")
    cog = f"{root}/ard/static/sources/lod2_morphology/2017/lod2_morphology_2017.tif"
    all_counts[2017] = _check_vintage(
        2017, keys_2017, cog, grid, aoi, _provenance_tile_count(root, 2017), findings
    )

    # ── 2024: provenance tile receipts ───────────────────────────────
    prov_uri = f"{root}/ard/static/sources/lod2_morphology/2024/provenance.json"
    prov = json.loads(read_bytes(prov_uri))
    tiles = (prov.get("source_metadata") or {}).get("tiles")
    if not isinstance(tiles, list) or not tiles:
        findings.append(f"LoD provenance {prov_uri} has no source_metadata.tiles")
        keys_2024: set[str] = set()
    else:
        keys_2024 = {f"{t['easting'] // 1000}_{t['northing'] // 1000}" for t in tiles}
        if len(keys_2024) != _EXPECTED_TILES_2024:
            findings.append(
                f"LoD provenance {prov_uri}: {len(keys_2024)} tiles, "
                f"expected {_EXPECTED_TILES_2024}"
            )
    cog = f"{root}/ard/static/sources/lod2_morphology/2024/lod2_morphology_2024.tif"
    all_counts[2024] = _check_vintage(
        2024, keys_2024, cog, grid, aoi, _provenance_tile_count(root, 2024), findings
    )

    if findings:
        print("FINDINGS:")
        for f in findings:
            print(f"  ✗ {f}")
        return 1
    print(f"OK: LoD coverage evidence consistent across vintages {sorted(all_counts)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
