"""Vegetation-height source adapter for the secondary pipeline.

Official Umweltatlas Berlin product for 2020:
``https://gdi.berlin.de/data/ua_vegetationshoehen_2020/atom/veghoehe_2020.zip``

The ZIP contains a single GeoTIFF (``veg_hoehe_2020_nodata.tif``) at
1 m resolution, EPSG:25833, float32 with NoData set.  Buildings, water
and other non-vegetated surfaces are encoded as NoData.

Processing
----------
1. Download the ZIP from the official ATOM URL (if not in raw storage).
2. Stream the inner GeoTIFF metadata + pixels via ``rasterio`` band I/O
   — never materialise the full 1 m raster in memory.
3. Verify native CRS, resolution, NoData, and bounding-box coverage of
   the canonical grid.
4. Reproject from 1 m to 10 m using ``Resampling.average`` into a
   float32 destination with NaN nodata.
5. Write a validated COG via ``write_cog_atomic`` and record provenance
   in a ``source.json`` sidecar alongside the raw archive.
"""

from __future__ import annotations

import json
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
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic
from berlin_lst_downscaling.data.io.storage import atomic_write
from berlin_lst_downscaling.data.secondary.download import download_to_raw

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

_CONFIG_HASH_PREFIX = "vegetation_height:v1:"


def contract_for_vegetation_height() -> Contract:
    """Return the output Contract for vegetation-height COGs."""
    return Contract(
        source="vegetation_height",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="vegetation_height",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Vegetation height in metres above ground (m). "
                    "NoData over buildings, water, and other non-vegetated surfaces."
                ),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
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
) -> dict[str, Any]:
    """Download, convert, and COG-write a vegetation-height vintage.

    Parameters
    ----------
    vintage :
        Must be ``2020``.
    output_root :
        Root URI for all outputs (local path or ``gs://bucket/...``).
    run_id :
        Unique run identifier (for provenance and staging).

    Returns
    -------
    dict
        QA payload with keys: ``vintage``, ``archive_checksum``, ``shape``,
        ``valid_frac``, ``min``, ``max``, ``output_uri``, ``config_hash``,
        ``native_metadata``.
    """
    if vintage != 2020:
        raise ValueError(
            f"Only vintage 2020 is available; got {vintage}"
        )

    url = VEGETATION_HEIGHT_URLS[vintage]
    raw_uri = _raw_zip_uri(output_root, vintage)
    dst_uri = _cog_uri(output_root, vintage)
    cache_path = _raw_zip_cache_uri(output_root, vintage)
    source_json_uri = _source_json_uri(output_root, vintage)
    c_hash = config_hash_for_vintage(vintage)

    # ── 1. materialise raw archive (streaming, no full-RAM load) ──────────
    receipt = download_to_raw(
        url=url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    archive_path = Path(receipt.local_cache_path)  # type: ignore[arg-type]

    # ── 2. inspect native GeoTIFF (no full-raster read) ──────────────────
    native_meta = _extract_native_metadata(archive_path)
    _validate_native_metadata(native_meta)

    # ── 3. reproject to canonical 10m grid (average resampling) ──────────
    grid = canon_grid_10m()
    dst_arr = np.empty((grid.shape.y, grid.shape.x), dtype=np.float32)

    tif_member = _locate_tiff_member(archive_path)
    with rasterio.open(f"zip://{archive_path}!/{tif_member}") as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.average,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

    # ── 4. write COG ────────────────────────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"vegetation_height": (("y", "x"), dst_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    contract = contract_for_vegetation_height()
    write_cog_atomic(ds, dst_uri, contract, overwrite=True)

    # ── 5. source.json sidecar (provenance) ───────────────────────────────
    sidecar = {
        "source": "vegetation_height",
        "vintage": vintage,
        "archive_url": url,
        "atom_feed": VEGETATION_HEIGHT_ATOM_FEED,
        "csw_record": VEGETATION_HEIGHT_CSW_RECORD,
        "dataset_identifier": VEGETATION_HEIGHT_DATASET_ID,
        "license": VEGETATION_HEIGHT_LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": receipt.checksum,
        "archive_bytes": int(
            archive_path.stat().st_size if archive_path.exists() else 0
        ),
        "native_metadata": native_meta,
    }
    atomic_write(source_json_uri, json.dumps(sidecar, indent=2), overwrite=True)

    # ── 6. QA payload ───────────────────────────────────────────────────
    valid = dst_arr[~np.isnan(dst_arr)]
    return {
        "vintage": vintage,
        "output_uri": dst_uri,
        "config_hash": c_hash,
        "archive_checksum": receipt.checksum,
        "shape": list(dst_arr.shape),
        "valid_frac": (
            round(float(len(valid)) / dst_arr.size, 4) if dst_arr.size > 0 else 0.0
        ),
        "min": float(valid.min()) if len(valid) > 0 else None,
        "max": float(valid.max()) if len(valid) > 0 else None,
        "native_metadata": native_meta,
    }


# ── path helpers ──────────────────────────────────────────────────────────


def _raw_zip_uri(output_root: str, vintage: int) -> str:
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/vegetation_height/"
        f"{vintage}/veghoehe_{vintage}.zip"
    )


def _raw_zip_cache_uri(output_root: str, vintage: int) -> str:
    """Return a writable local cache path even when output_root is GCS.

    For local ``output_root`` we reuse the raw URI directly.  For
    ``gs://`` we stage the cache under ``$TMPDIR`` so we never write
    huge archives into the CWD.
    """
    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/vegetation_height_{vintage}.zip"
    return _raw_zip_uri(output_root, vintage)


def _cog_uri(output_root: str, vintage: int) -> str:
    return (
        f"{output_root.rstrip('/')}/ard/static/morphology/"
        f"vegetation_height/{vintage}/vegetation_height_{vintage}.tif"
    )


def _source_json_uri(output_root: str, vintage: int) -> str:
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/vegetation_height/"
        f"{vintage}/source.json"
    )


# ── archive helpers ───────────────────────────────────────────────────────


def _locate_tiff_member(zip_path: Path) -> str:
    """Return the name of the GeoTIFF member inside *zip_path*.

    Raises ``ValueError`` if no TIFF member is found.
    """
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
    """Fail fast if the native raster does not match the expected contract.

    Hard checks:
    - native CRS is EPSG:25833
    - native resolution is 1 m (both axes)
    - NoData is set (required for downstream reproject)
    - native bounding box covers the canonical grid origin
    """
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
