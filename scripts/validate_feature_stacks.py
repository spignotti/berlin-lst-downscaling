# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Independent validator for published scene feature stacks.

Read-only probe over a feature-stack root (local or ``gs://``): re-reads
the features ledger and every done scene's artifacts and verifies

- ledger completeness and artifact existence (COG, mask, provenance,
  STAC, complete),
- 28-band COG contract: band count, dtype, CRS, grid, channel order,
- mask COG: uint8, single band, same grid, values in {0, 1},
- mask semantics (blockwise): ``mask == 1`` implies all 28 channels are
  finite and in-range; ``mask == 0`` permits finite values in any
  channel (availability is per channel),
- AOI semantics: outside the exact Berlin AOI the mask is 0 and every
  channel is NaN,
- per-channel value ranges on valid pixels only,
- sidecar content: provenance channel order / coverage / mask semantics,
  STAC data + feature_valid assets with 28 raster bands,
- provenance coverage numbers against independently recomputed counts.

The validator imports only the channel contract (names/ranges) — never
the composer or pipeline implementation. It writes nothing.

Usage
-----
    uv run python scripts/validate_feature_stacks.py \
        --root gs://berlin-lst-data/features/v2
    uv run python scripts/validate_feature_stacks.py --root data/smoke/features
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pyarrow.parquet as pq
import rasterio
import rasterio.warp as rwarp
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES, FEATURE_CHANNELS
from berlin_lst_downscaling.data.io import exists, read_bytes

_N_EXPECTED_BANDS = 28
_TILE = 1024  # blockwise scan tile (multiple of COG 512px blocks)


# ── helpers ──────────────────────────────────────────────────────────


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _grid_from_cog(uri: str) -> GeoBox:
    with rasterio.open(uri) as src:
        return GeoBox.from_rio(src)


def _aoi_on_grid(aoi_uri: str, grid: GeoBox) -> np.ndarray:
    """Reproject the Berlin AOI mask onto *grid* (nearest; independent of
    the composer's helper)."""
    with rasterio.open(aoi_uri) as src:
        source = src.read(1).astype(np.uint8)
        src_crs, src_transform, src_nodata = src.crs, src.transform, src.nodata
    destination = np.zeros((grid.shape.y, grid.shape.x), dtype=np.uint8)
    rwarp.reproject(
        source=source,
        src_crs=src_crs,
        src_transform=src_transform,
        src_nodata=src_nodata,
        destination=destination,
        dst_crs=grid.crs,
        dst_transform=grid.transform,
        dst_nodata=0,
        resampling=rwarp.Resampling.nearest,
    )
    return destination == 1


# ── per-scene checks ─────────────────────────────────────────────────


def _check_metadata(cog_uri: str, mask_uri: str, errors: list[str]) -> GeoBox | None:
    try:
        with rasterio.open(cog_uri) as src:
            if src.count != _N_EXPECTED_BANDS:
                errors.append(f"{cog_uri}: band count {src.count}, expected {_N_EXPECTED_BANDS}")
            if src.dtypes[0] != "float32":
                errors.append(f"{cog_uri}: dtype {src.dtypes[0]!r}, expected float32")
            if str(src.crs).upper() != "EPSG:25833":
                errors.append(f"{cog_uri}: CRS {src.crs!r}")
            names = [d for d in (src.descriptions or ()) if d]
            if tuple(names) != FEATURE_CHANNEL_NAMES:
                errors.append(
                    f"{cog_uri}: channel order {names} != contract {FEATURE_CHANNEL_NAMES}"
                )
            grid = GeoBox.from_rio(src)
            # Verify the COG sits on the canonical 10 m grid.
            canonical = canon_grid_10m()
            if not np.allclose(
                grid.transform[:6], canonical.transform[:6], rtol=0.0, atol=1e-6
            ):
                errors.append(
                    f"{cog_uri}: transform not on canonical 10 m grid — "
                    f"{grid.transform}"
                )
    except Exception as exc:  # read-only probe
        errors.append(f"{cog_uri}: cannot open: {exc}")
        return None

    try:
        with rasterio.open(mask_uri) as src:
            if src.count != 1:
                errors.append(f"{mask_uri}: band count {src.count}, expected 1")
            if src.dtypes[0] != "uint8":
                errors.append(f"{mask_uri}: dtype {src.dtypes[0]!r}, expected uint8")
            if str(src.crs).upper() != "EPSG:25833":
                errors.append(f"{mask_uri}: CRS {src.crs!r}")
            if (src.height, src.width) != (grid.shape.y, grid.shape.x):
                errors.append(
                    f"{mask_uri}: shape ({src.height}, {src.width}) != COG "
                    f"({grid.shape.y}, {grid.shape.x})"
                )
            mask_grid = GeoBox.from_rio(src)
            if not np.allclose(
                grid.transform[:6], mask_grid.transform[:6], rtol=0.0, atol=1e-6
            ):
                errors.append(
                    f"{mask_uri}: transform mismatch with COG — "
                    f"mask {mask_grid.transform} != COG {grid.transform}"
                )
    except Exception as exc:
        errors.append(f"{mask_uri}: cannot open: {exc}")
    return grid


def _check_pixels(
    cog_uri: str,
    mask_uri: str,
    aoi: np.ndarray,
    errors: list[str],
    stats: dict,
) -> None:
    """Blockwise mask, AOI, and range checks."""
    with rasterio.open(cog_uri) as cog, rasterio.open(mask_uri) as msk:
        h, w = cog.height, cog.width
        valid_total = 0
        aoi_inside = int(aoi.sum())
        for r0 in range(0, h, _TILE):
            r1 = min(r0 + _TILE, h)
            for c0 in range(0, w, _TILE):
                c1 = min(c0 + _TILE, w)
                win = Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore[call-arg]
                bands = cog.read(window=win)  # (28, bh, bw)
                mask = msk.read(1, window=win)
                aoi_t = aoi[r0:r1, c0:c1]

                if not set(np.unique(mask)).issubset({0, 1}):
                    errors.append(f"{cog_uri}: mask values {np.unique(mask)} not in {{0,1}}")
                # mask == 1 must imply all channels finite AND in-range.
                # mask == 0 does not constrain individual channels (per-channel
                # availability): only unavailable bands are NaN.
                claim = mask == 1
                complete = np.all(np.isfinite(bands), axis=0)
                for i, spec in enumerate(FEATURE_CHANNELS):
                    if spec.valid_range is None:
                        continue
                    lo, hi = spec.valid_range
                    complete &= (bands[i] >= lo) & (bands[i] <= hi)
                if np.any(claim & ~complete):
                    n = int(np.sum(claim & ~complete))
                    errors.append(
                        f"{cog_uri}: mask==1 with non-finite/out-of-range values on {n} px "
                        f"(tile {r0},{c0})"
                    )
                if np.any((aoi_t == 0) & (mask == 1)):
                    n = int(np.sum((aoi_t == 0) & (mask == 1)))
                    errors.append(f"{cog_uri}: {n} px outside AOI marked valid")
                # Outside the AOI every channel must be NaN.
                if np.any((aoi_t == 0) & np.any(np.isfinite(bands), axis=0)):
                    n = int(np.sum((aoi_t == 0) & np.any(np.isfinite(bands), axis=0)))
                    errors.append(f"{cog_uri}: {n} px outside AOI with finite values")

                # range checks on valid pixels
                valid = mask == 1
                for i, spec in enumerate(FEATURE_CHANNELS):
                    if spec.valid_range is None:
                        continue
                    lo, hi = spec.valid_range
                    vals = bands[i][valid]
                    if vals.size:
                        if vals.min() < lo:
                            errors.append(
                                f"{cog_uri}: band {i + 1} {spec.name} min {float(vals.min()):.4f} "
                                f"< {lo}"
                            )
                        if vals.max() > hi:
                            errors.append(
                                f"{cog_uri}: band {i + 1} {spec.name} max {float(vals.max()):.4f} "
                                f"> {hi}"
                            )
                valid_total += int(np.sum(valid))

    stats["feature_valid_px"] = valid_total
    stats["inside_aoi_px"] = aoi_inside
    stats["outside_aoi_px"] = int(aoi.size) - aoi_inside


def _check_sidecars(scene_id: str, cog_uri: str, mask_uri: str, prov_uri: str,
                    stac_uri: str, comp_uri: str,
                    stats: dict, errors: list[str]) -> None:
    if not exists(comp_uri):
        errors.append(f"{scene_id}: complete.json missing")
        return
    comp = _read_json(comp_uri)
    if not comp.get("published_at"):
        errors.append(f"{scene_id}: complete.json missing published_at")

    prov = _read_json(prov_uri)
    if tuple(prov.get("channel_order", ())) != FEATURE_CHANNEL_NAMES:
        errors.append(f"{scene_id}: provenance channel_order mismatch")
    for key in ("config_hash", "coverage", "mask_semantics", "vegetation_height_policy"):
        if key not in prov:
            errors.append(f"{scene_id}: provenance missing {key!r}")
    cov = prov.get("coverage", {})
    for key in ("feature_valid_px", "inside_aoi_px", "outside_aoi_px"):
        if key not in cov:
            errors.append(f"{scene_id}: coverage missing {key!r}")
    if cov.get("feature_valid_px") != stats.get("feature_valid_px"):
        errors.append(
            f"{scene_id}: provenance feature_valid_px {cov.get('feature_valid_px')} != "
            f"recomputed {stats.get('feature_valid_px')}"
        )

    stac = _read_json(stac_uri)
    dt = stac.get("properties", {}).get("datetime")
    if not dt or "T" not in str(dt):
        errors.append(f"{scene_id}: STAC datetime missing or not RFC 3339: {dt!r}")
    assets = stac.get("assets", {})
    if "data" not in assets or "feature_valid" not in assets:
        errors.append(f"{scene_id}: STAC missing data/feature_valid assets")
        return
    raster_bands = assets["data"].get("raster:bands", [])
    if len(raster_bands) != _N_EXPECTED_BANDS:
        errors.append(f"{scene_id}: STAC data raster:bands {len(raster_bands)}, expected 28")
    fv_bands = assets["feature_valid"].get("raster:bands", [])
    if not fv_bands or fv_bands[0].get("data_type") != "uint8":
        errors.append(f"{scene_id}: STAC feature_valid asset not uint8")
    if assets["data"].get("href") != cog_uri:
        errors.append(f"{scene_id}: STAC data href does not match ledger COG")
    if assets["feature_valid"].get("href") != mask_uri:
        errors.append(f"{scene_id}: STAC feature_valid href does not match ledger mask")


# ── orchestration ────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent validator for feature stacks")
    parser.add_argument("--root", required=True, help="Features root (local or gs://)")
    parser.add_argument("--aoi", default="data/boundaries/aoi_10m.tif",
                        help="Berlin AOI mask (10 m, EPSG:25833)")
    parser.add_argument("--scene-ids", nargs="*", default=[], help="Restrict to these scene IDs")
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=None,
        help="Require exactly this many ledger rows, all done + complete; "
             "fail on any failed/exporting/incomplete/absent row",
    )
    args = parser.parse_args()

    root = args.root.rstrip("/")
    ledger_uri = f"{root}/_state/features/ledger.parquet"
    if not exists(ledger_uri):
        print(f"FAILURE: features ledger missing: {ledger_uri}")
        return 1

    table = _read_table(ledger_uri)
    cols = table.to_pydict()

    if args.expected_scenes is not None:
        n_rows = table.num_rows
        bad: list[str] = []
        for i in range(n_rows):
            scene_id = str(cols["period_or_vintage"][i])
            status = str(cols["status"][i])
            if status != "done":
                bad.append(f"{scene_id}: status {status!r} (expected 'done')")
                continue
            for label, key in (("output_uri", "output_uri"), ("stac_uri", "stac_uri"),
                               ("provenance_uri", "provenance_uri"),
                               ("completion_uri", "completion_uri")):
                if not cols[key][i]:
                    bad.append(f"{scene_id}: missing {label}")
        if n_rows != args.expected_scenes:
            print(
                f"FAILURE: expected exactly {args.expected_scenes} ledger rows, "
                f"found {n_rows}"
            )
            return 1
        if bad:
            print("FAILURE: ledger has non-done / incomplete rows:")
            for b in bad:
                print(f"  ✗ {b}")
            return 1

    rows = []
    for i in range(table.num_rows):
        if str(cols["status"][i]) != "done":
            continue
        rows.append(
            {
                "scene_id": str(cols["period_or_vintage"][i]),
                "config_hash": str(cols["config_hash"][i] or ""),
                "cog": str(cols["output_uri"][i] or ""),
                "stac": str(cols["stac_uri"][i] or ""),
                "prov": str(cols["provenance_uri"][i] or ""),
                "comp": str(cols["completion_uri"][i] or ""),
            }
        )

    if args.scene_ids:
        wanted = set(args.scene_ids)
        rows = [r for r in rows if r["scene_id"] in wanted]
        if len(rows) != len(wanted):
            print(f"FAILURE: expected {len(wanted)} done scenes, ledger has {len(rows)}")
            return 1

    if not rows:
        print("FAILURE: no done scenes in ledger")
        return 1

    print(f"Validating {len(rows)} feature stacks under {root}")
    errors: list[str] = []
    total_valid = 0
    total_inside = 0
    total_outside = 0

    for row in rows:
        scene_id = row["scene_id"]
        mask_uri = row["cog"].replace(".tif", ".feature_valid.tif")
        if not row["cog"] or not row["stac"] or not row["prov"] or not row["comp"]:
            errors.append(f"{scene_id}: incomplete ledger URIs")
            continue
        for uri, label in (
            (row["cog"], "COG"),
            (mask_uri, "mask"),
            (row["stac"], "STAC"),
            (row["prov"], "provenance"),
            (row["comp"], "complete"),
        ):
            if not exists(uri):
                errors.append(f"{scene_id}: {label} missing: {uri}")
        if not row["config_hash"]:
            errors.append(f"{scene_id}: missing config_hash")

        grid = _check_metadata(row["cog"], mask_uri, errors)
        if grid is None:
            continue
        aoi = _aoi_on_grid(args.aoi, grid)
        stats: dict = {}
        _check_pixels(row["cog"], mask_uri, aoi, errors, stats)
        _check_sidecars(
            scene_id, row["cog"], mask_uri, row["prov"], row["stac"],
            row["comp"], stats, errors,
        )
        total_valid += stats.get("feature_valid_px", 0)
        total_inside += stats.get("inside_aoi_px", 0)
        total_outside += stats.get("outside_aoi_px", 0)
        print(f"  {scene_id}: feature_valid_px={stats.get('feature_valid_px')} "
              f"(inside AOI {stats.get('inside_aoi_px')}, outside {stats.get('outside_aoi_px')})")

    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print(
        f"OK: {len(rows)} stacks, {total_valid} feature-valid px "
        f"({total_inside} inside AOI, {total_outside} outside)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
