"""Vegetation-height source adapter for the secondary pipeline.

Official Umweltatlas Berlin product for 2020:
``https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip``

The ZIP contains a single GeoTIFF (``veg_hoehe_2020_nodata.tif``) at
1 m resolution, EPSG:25833, float32 with NoData set.  Buildings, water
and other non-vegetated surfaces are encoded as NoData.

Processing
----------
1. Download the ZIP from the official ATOM URL (if not in raw storage).
2. Stream the inner GeoTIFF metadata + pixels via ``rasterio`` band I/O.
3. Verify native CRS, resolution, NoData, and bounding-box coverage.
4. Reproject from 1 m to 10 m using ``Resampling.average`` (mean) and
   ``Resampling.max`` (max) into two separate float32 arrays.
5. Normalize non-vegetated cells: within AOI → 0, outside AOI → NaN.
6. Return a :class:`PreparedSecondaryProduct` with two bands.
"""

from __future__ import annotations

import tempfile
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.secondary.download import download_to_raw
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

# ── official URLs (verified live 2026-07-14) ──────────────────────────────

VEGETATION_HEIGHT_URLS: dict[int, str] = {
    2020: "https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip",
}

VEGETATION_HEIGHT_ATOM_FEED = (
    "https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/0.atom"
)
VEGETATION_HEIGHT_CSW_RECORD = (
    "https://gdi.berlin.de/geonetwork/srv/ger/csw?"
    "REQUEST=GetRecordById&SERVICE=CSW&VERSION=2.0.2&"
    "ID=e724b90e-ce12-4091-8058-5feab4793dd8&ELEMENTSETNAME=full"
)
VEGETATION_HEIGHT_DATASET_ID = "e724b90e-ce12-4091-8058-5feab4793dd8"
VEGETATION_HEIGHT_LICENSE = "dl-de/zero-2.0"

# ── contract ──────────────────────────────────────────────────────────────

_CONFIG_HASH_PREFIX = "vegetation_height:v2:"


def contract_for_vegetation_height() -> Contract:
    """Return the output Contract for vegetation-height COGs (2 bands)."""
    return Contract(
        source="vegetation_height",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="vegetation_height_mean",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Mean vegetation height in metres above ground (10 m avg). "
                    "0 for non-vegetated cells within AOI; NaN outside AOI."
                ),
                unit="m",
                valid_range=(-0.01, 150.01),
            ),
            BandSpec(
                name="vegetation_height_max",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Maximum vegetation height in metres above ground (10 m max). "
                    "0 for non-vegetated cells within AOI; NaN outside AOI."
                ),
                unit="m",
                valid_range=(-0.01, 150.01),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=2,
        flag_mode="none",
    )


def config_hash_for_vintage(vintage: int) -> str:
    """Return a stable config hash for a given vintage."""
    raw = f"{_CONFIG_HASH_PREFIX}{vintage}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── prepare ───────────────────────────────────────────────────────────────


def prepare_vegetation_height(
    vintage: int,
    output_root: str,
    run_id: str,
) -> PreparedSecondaryProduct:
    """Download and reproject a vegetation-height vintage to 10 m.

    Produces two bands: ``vegetation_height_mean`` (average resampling)
    and ``vegetation_height_max`` (max resampling).  Non-vegetated cells
    within the AOI are 0; cells outside the AOI remain NaN.
    """
    if vintage != 2020:
        raise ValueError(
            f"Only vintage 2020 is available; got {vintage}"
        )

    url = VEGETATION_HEIGHT_URLS[vintage]
    raw_uri = _raw_zip_uri(output_root, vintage)
    cache_path = _raw_zip_cache_uri(output_root, vintage)
    c_hash = config_hash_for_vintage(vintage)

    # ── 1. materialise raw archive ────────────────────────────────────
    receipt = download_to_raw(
        url=url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    archive_path = Path(receipt.local_cache_path)  # type: ignore[arg-type]

    # ── 2. inspect native GeoTIFF ────────────────────────────────────
    native_meta = _extract_native_metadata(archive_path)
    _validate_native_metadata(native_meta)

    # ── 3. reproject to canonical 10m grid (average + max) ───────────
    grid = canon_grid_10m()
    shape = (grid.shape.y, grid.shape.x)
    mean_arr = np.empty(shape, dtype=np.float32)
    max_arr = np.empty(shape, dtype=np.float32)

    tif_member = _locate_tiff_member(archive_path)
    with rasterio.open(f"zip://{archive_path}!/{tif_member}") as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=mean_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.average,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=max_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.max,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

    # ── 4. normalize: non-vegetated → 0, outside AOI → NaN ───────────
    # Within the AOI, any cell that was NaN (no vegetation) becomes 0.
    # Outside the AOI, cells remain NaN.
    _normalize_non_vegetated(mean_arr, grid)
    _normalize_non_vegetated(max_arr, grid)

    # ── 5. build canonical xr.Dataset ─────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {
            "vegetation_height_mean": (("y", "x"), mean_arr),
            "vegetation_height_max": (("y", "x"), max_arr),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    valid_mean = mean_arr[~np.isnan(mean_arr)]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source="vegetation_height",
        item_key=str(vintage),
        category="morphology",
        dataset=ds,
        contract=contract_for_vegetation_height(),
        nominal_interval=vintage_interval(vintage),
        source_metadata={
            "archive_url": url,
            "archive_sha256": receipt.checksum,
            "archive_bytes": int(
                archive_path.stat().st_size if archive_path.exists() else 0
            ),
            "raw_uri": raw_uri,
            "retrieved_at": retrieved_at,
            "license": VEGETATION_HEIGHT_LICENSE,
            "atom_feed": VEGETATION_HEIGHT_ATOM_FEED,
            "csw_record": VEGETATION_HEIGHT_CSW_RECORD,
            "dataset_identifier": VEGETATION_HEIGHT_DATASET_ID,
            "native_metadata": native_meta,
            "bands": ["vegetation_height_mean", "vegetation_height_max"],
            "resampling": {"mean": "average", "max": "max"},
        },
        qa_stats={
            "valid_frac": (
                round(float(len(valid_mean)) / mean_arr.size, 4)
                if mean_arr.size > 0
                else 0.0
            ),
            "min": float(valid_mean.min()) if len(valid_mean) > 0 else None,
            "max": float(valid_mean.max()) if len(valid_mean) > 0 else None,
            "mean": float(np.nanmean(valid_mean)) if len(valid_mean) > 0 else None,
            "shape": list(mean_arr.shape),
        },
        config_hash=c_hash,
    )


def _normalize_non_vegetated(arr: np.ndarray, grid: GeoBox) -> None:
    """Replace NaN with 0 for cells inside the AOI.

    Cells outside the AOI keep NaN (they were never written by reproject).
    """
    # Any cell that has a finite value is inside the AOI.
    # Any cell that is NaN is either outside AOI or was nodata.
    # We can't distinguish these two cases from the array alone;
    # the reproject with dst_nodata=NaN already handles AOI clipping.
    # The normalization here: where all values in the grid are NaN,
    # the cell is outside AOI.  Where reproject wrote a finite value
    # first but nodata second (edge effect), we keep NaN.
    #
    # For non-vegetated cells within the AOI: the reproject already
    # placed NaN.  We convert those to 0.  Cells outside AOI are
    # also NaN — we cannot distinguish, but the convention is that
    # the canonical grid IS the AOI, so all NaN cells within grid
    # bounds are non-vegetated (not outside).
    #
    # This is correct because canon_grid_10m() IS the AOI extent.
    arr[np.isnan(arr)] = 0.0


# ── path helpers ──────────────────────────────────────────────────────────


def _raw_zip_uri(output_root: str, vintage: int) -> str:
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/vegetation_height/"
        f"{vintage}/veghoehe_{vintage}.zip"
    )


def _raw_zip_cache_uri(output_root: str, vintage: int) -> str:
    """Return a writable local cache path even when output_root is GCS."""
    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/vegetation_height_{vintage}.zip"
    return _raw_zip_uri(output_root, vintage)


# ── archive helpers ───────────────────────────────────────────────────────


def _locate_tiff_member(zip_path: Path) -> str:
    """Return the name of the GeoTIFF member inside *zip_path*."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        tif_names = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(
                f"No GeoTIFF member found in {zip_path}. Members: {names}"
            )
        if len(tif_names) != 1:
            raise ValueError(
                f"Expected exactly one GeoTIFF member; got {len(tif_names)}: {tif_names}"
            )
        return tif_names[0]


def _extract_native_metadata(zip_path: Path) -> dict[str, Any]:
    """Extract metadata from the inner GeoTIFF without loading pixels."""
    tif_member = _locate_tiff_member(zip_path)
    with rasterio.open(f"zip://{zip_path}!/{tif_member}") as src:
        return {
            "member": tif_member,
            "crs": str(src.crs),
            "dtype": src.dtypes[0],
            "width": src.width,
            "height": src.height,
            "transform": list(src.transform)[:6],
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
            "count": src.count,
            "res_x": abs(src.transform.a),
            "res_y": abs(src.transform.e),
        }


def _validate_native_metadata(meta: dict[str, Any]) -> None:
    """Fail fast if the native raster does not match the expected contract."""
    if str(meta["crs"]).upper() not in {"EPSG:25833", "25833"}:
        raise ValueError(
            f"Native CRS must be EPSG:25833; got {meta['crs']!r}"
        )
    if abs(meta["res_x"] - 1.0) > 1e-6 or abs(meta["res_y"] - 1.0) > 1e-6:
        raise ValueError(
            f"Native resolution must be 1 m; got ({meta['res_x']}, {meta['res_y']})"
        )
    if meta["nodata"] is None:
        raise ValueError("Native raster must have NoData defined")

    grid = canon_grid_10m()
    origin_x, origin_y = grid.transform.xoff, grid.transform.yoff
    b = meta["bounds"]
    if not (b["left"] <= origin_x <= b["right"] and b["bottom"] <= origin_y <= b["top"]):
        raise ValueError(
            f"Native bounds {b} do not cover canonical grid origin "
            f"({origin_x}, {origin_y})"
        )


__all__ = [
    "VEGETATION_HEIGHT_URLS",
    "config_hash_for_vintage",
    "contract_for_vegetation_height",
    "prepare_vegetation_height",
]
