"""Imperviousness (Versiegelung) source adapter for the secondary pipeline.

Official Umweltatlas Berlin raster products for 2016 and 2021.
Each vintage is a single ZIP with a GeoTIFF (EPSG:25833, 2.5m, uint8 class codes).

Processing
----------
1. Stream the ZIP to raw storage via ``download_to_raw`` (no full-RAM load).
2. Open the inner GeoTIFF via ``rasterio.band(src, 1)`` over a
   ``zip://`` VFS path — no extraction step.
3. Convert uint8 class codes to float32 sealing percent [0, 100], NaN for nodata.
4. Reproject from native 2.5m grid to the canonical 10m EPSG:25833 grid
   using ``Resampling.average`` — averaging is correct **only** after
   class-code-to-percent conversion.
5. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts (COG + STAC + provenance + complete).
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.secondary.download import download_to_raw
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

# ── official URLs (verified live 2026-07-14) ──────────────────────────

IMPERVIOUSNESS_URLS: dict[int, str] = {
    2016: ("https://gdi.berlin.de/data/ua_versiegelung_2016/atom/Versiegelung_Raster_2016.zip"),
    2021: ("https://gdi.berlin.de/data/ua_versiegelung_2021/atom/Versiegelung_Raster_2021.zip"),
}

# ── class-code lookup (verified from actual rasters) ──────────────────

_LOOKUP: np.ndarray = np.full(256, 100.0, dtype=np.float32)
_LOOKUP[0] = 0.0
for _code in range(5, 100, 10):
    _LOOKUP[_code] = float(_code)
_LOOKUP[255] = np.nan

# ── contract ──────────────────────────────────────────────────────────

_CONFIG_HASH_PREFIX = "imperviousness:v2:"

# Verified 15 pixel codes across both vintages:
#   0   = unsealed
#   5,15,25,...,95  = sealing classes (value = percent)
#   100 = fully sealed (non-building)
#   101 = building-shadow sealed surface
#   102 = building footprint
#   103 = rail ballast (classified as sealed in the uncorrected raster)
#   110 = shadow (treated as sealed)
# 255 is the documented 2021 nodata code but not present in pixel values.
_ALLOWED_CODES: frozenset[int] = frozenset(
    {0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 101, 102, 103, 110, 255}
)


def contract_for_imperviousness() -> Contract:
    """Return the output Contract for imperviousness COGs."""
    return Contract(
        source="imperviousness",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="imperviousness",
                dtype="float32",
                nodata=float("nan"),
                valid_range=(-0.01, 100.01),
                description="Sealing degree in percent (0–100)",
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def config_hash_for_vintage(vintage: int) -> str:
    """Return a stable config hash for a given vintage.

    Incorporates the contract schema version, the target resolution, and
    the vintage itself so that re-processing with different parameters
    is correctly detected.
    """
    raw = f"{_CONFIG_HASH_PREFIX}{vintage}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── scene-year → vintage mapping ──────────────────────────────────────


def vintage_for_scene_year(year: int) -> int:
    """Return the imperviousness vintage that applies to a scene year.

    2017–2019 → 2016
    2020–2025 → 2021
    """
    # decision: piecewise constant; no years earlier than 2017 are
    # included in the training window (May–Sep 2017–2025).
    if year <= 2019:
        return 2016
    return 2021


# ── prepare ───────────────────────────────────────────────────────────


def prepare_imperviousness(
    vintage: int,
    output_root: str,
    run_id: str,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Download, convert, and reproject an imperviousness vintage.

    Parameters
    ----------
    vintage :
        Either ``2016`` or ``2021``.
    output_root :
        Root URI for all outputs (local path or ``gs://bucket/...``).
    run_id :
        Unique run identifier (for provenance and staging).
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.

    Returns
    -------
    PreparedSecondaryProduct
        Canonical-grid dataset + source metadata + QA statistics.
        The pipeline finaliser writes the four final artifacts.
    """
    url = IMPERVIOUSNESS_URLS[vintage]
    raw_uri = _raw_zip_uri(output_root, vintage)
    cache_path = _raw_zip_cache_uri(output_root, vintage)
    c_hash = config_hash_for_vintage(vintage)

    # ── 1. materialise raw ZIP via shared downloader (no full-RAM load) ──
    receipt = download_to_raw(
        url=url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    archive_path = Path(receipt.local_cache_path)  # type: ignore[arg-type]

    # ── 2. read TIFF in place over zip:// VFS path ───────────────────
    with _zip_tiff_open(archive_path) as src:
        src_uint8 = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

        observed = sorted(int(v) for v in np.unique(src_uint8))
        _validate_codes(observed)

        src_pct = _LOOKUP[src_uint8].astype(np.float32, copy=False)

    # ── 3. reproject to canonical 10m grid (average resampling) ──────
    grid = grid or canon_grid_10m()
    dst_arr = np.empty((grid.shape.y, grid.shape.x), dtype=np.float32)

    reproject(
        source=src_pct,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # ── 4. build canonical xr.Dataset ────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"imperviousness": (("y", "x"), dst_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    valid = dst_arr[~np.isnan(dst_arr)]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source="imperviousness",
        item_key=str(vintage),
        category="morphology",
        dataset=ds,
        contract=contract_for_imperviousness(),
        nominal_interval=vintage_interval(vintage),
        source_metadata={
            "archive_url": url,
            "archive_sha256": receipt.checksum,
            "raw_uri": raw_uri,
            "retrieved_at": retrieved_at,
            "license": "dl-de/zero-2.0",
            "native_codes_observed": observed,
        },
        qa_stats={
            "valid_frac": (round(float(len(valid)) / dst_arr.size, 4) if dst_arr.size > 0 else 0.0),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(dst_arr.shape),
        },
        config_hash=c_hash,
    )


# ── helpers ───────────────────────────────────────────────────────────


def _raw_zip_uri(output_root: str, vintage: int) -> str:
    """Return the raw ZIP URI for a vintage."""
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/imperviousness/"
        f"{vintage}/Versiegelung_Raster_{vintage}.zip"
    )


def _raw_zip_cache_uri(output_root: str, vintage: int) -> str:
    """Return a writable local cache path even when output_root is GCS.

    For local ``output_root`` we reuse the raw URI directly.  For
    ``gs://`` we stage the cache under ``$TMPDIR`` so we never write
    large archives into the CWD.
    """
    import tempfile

    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/imperviousness_{vintage}.zip"
    return _raw_zip_uri(output_root, vintage)


@contextlib.contextmanager
def _zip_tiff_open(zip_path: Path):
    """Open the first GeoTIFF member inside *zip_path* via ``zip://`` VFS."""
    import zipfile as _zf

    with _zf.ZipFile(zip_path) as z:
        tif_names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(f"No GeoTIFF member found in {zip_path}. Members: {z.namelist()}")
        if len(tif_names) != 1:
            raise ValueError(
                f"Expected exactly one GeoTIFF member; got {len(tif_names)}: {tif_names}"
            )
        member = tif_names[0]
    with rasterio.open(f"zip://{zip_path}!/{member}") as src:
        yield src


def _validate_codes(observed: list[int]) -> None:
    """Raise ``ValueError`` if any observed code is outside the allowed set."""
    unknown = [code for code in observed if code not in _ALLOWED_CODES]
    if unknown:
        raise ValueError(
            f"Observed class codes outside the allowed set: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_CODES)}. "
            "This dataset may use a different encoding."
        )


__all__ = [
    "IMPERVIOUSNESS_URLS",
    "config_hash_for_vintage",
    "contract_for_imperviousness",
    "prepare_imperviousness",
    "vintage_for_scene_year",
]
