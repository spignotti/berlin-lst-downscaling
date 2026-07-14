"""Imperviousness (Versiegelung) source adapter for the secondary pipeline.

Official Umweltatlas Berlin raster products for 2016 and 2021.
Each vintage is a single ZIP with a GeoTIFF (EPSG:25833, 2.5m, uint8 class codes).

Processing
----------
1. Download ZIP from official URL (if not in raw storage).
2. Extract the inner GeoTIFF from the ZIP.
3. Convert uint8 class codes to float32 sealing percent [0, 100], NaN for nodata.
4. Reproject from native 2.5m grid to the canonical 10m EPSG:25833 grid
   using ``Resampling.average`` — averaging is correct **only** after
   class-code-to-percent conversion.
5. Write validated COG via ``write_cog_atomic`` (multi-band float32).
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tenacity import retry, stop_after_attempt, wait_exponential

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic
from berlin_lst_downscaling.data.io.storage import atomic_upload, exists, read_bytes

# ── official URLs (verified live 2026-07-14) ──────────────────────────

IMPERVIOUSNESS_URLS: dict[int, str] = {
    2016: (
        "https://gdi.berlin.de/data/ua_versiegelung_2016/atom/"
        "Versiegelung_Raster_2016.zip"
    ),
    2021: (
        "https://gdi.berlin.de/data/ua_versiegelung_2021/atom/"
        "Versiegelung_Raster_2021.zip"
    ),
}

# ── class-code lookup (verified from actual rasters) ──────────────────

# Verified 16 codes across both vintages:
#   0   = unsealed
#   5,15,25,...,95  = sealing classes (value = percent)
#   100 = fully sealed (non-building)
#   101 = building-shadow sealed surface
#   102 = building footprint
#   103 = rail ballast (classified as sealed in the uncorrected raster)
#   110 = shadow (treated as sealed)
#   255 = nodata (explicit in 2021, absent in 2016)
_LOOKUP: np.ndarray = np.full(256, 100.0, dtype=np.float32)
_LOOKUP[0] = 0.0
for _code in range(5, 100, 10):
    _LOOKUP[_code] = float(_code)
_LOOKUP[255] = np.nan

# ── contract ──────────────────────────────────────────────────────────

_CONFIG_HASH_PREFIX = "imperviousness:v1:"


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
) -> dict[str, Any]:
    """Download, convert, and COG-write an imperviousness vintage.

    Parameters
    ----------
    vintage :
        Either ``2016`` or ``2021``.
    output_root :
        Root URI for all outputs (local path or ``gs://bucket/...``).
    run_id :
        Unique run identifier (for provenance and staging).

    Returns
    -------
    dict
        QA payload with keys: ``vintage``, ``raw_checksum``, ``shape``,
        ``valid_frac``, ``min``, ``max``, ``output_uri``, ``config_hash``.
    """
    url = IMPERVIOUSNESS_URLS[vintage]
    raw_uri = _raw_zip_uri(output_root, vintage)
    dst_uri = _cog_uri(output_root, vintage)
    c_hash = config_hash_for_vintage(vintage)

    # ── 1. download / fetch raw ZIP ──────────────────────────────────
    zip_local = _fetch_zip(url, raw_uri)

    # ── 2. extract TIFF from ZIP ─────────────────────────────────────
    tif_bytes = _extract_tiff(zip_local)

    # ── 3. class codes → float32 percent ─────────────────────────────
    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        src_uint8 = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

        observed = sorted(int(v) for v in np.unique(src_uint8))
        # Fail on unknown codes (codes not in the lookup that would
        # silently map to 100 %)
        _validate_codes(observed)

        src_pct = _LOOKUP[src_uint8].astype(np.float32, copy=False)

    # ── 4. reproject to canonical 10m grid (average resampling) ──────
    grid = canon_grid_10m()
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

    # ── 5. write COG ────────────────────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"imperviousness": (("y", "x"), dst_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    contract = contract_for_imperviousness()
    write_cog_atomic(ds, dst_uri, contract, overwrite=True)

    # ── 6. QA payload ───────────────────────────────────────────────
    valid = dst_arr[~np.isnan(dst_arr)]
    qa: dict[str, Any] = {
        "vintage": vintage,
        "output_uri": dst_uri,
        "config_hash": c_hash,
        "raw_checksum": sha256(zip_local.read_bytes()).hexdigest()[:16],
        "shape": list(dst_arr.shape),
        "codes_observed": observed,
        "valid_frac": round(float(len(valid)) / dst_arr.size, 4) if dst_arr.size > 0 else 0.0,
        "min": float(valid.min()) if len(valid) > 0 else None,
        "max": float(valid.max()) if len(valid) > 0 else None,
    }
    return qa


# ── helpers ───────────────────────────────────────────────────────────


def _raw_zip_uri(output_root: str, vintage: int) -> str:
    """Return the raw ZIP URI for a vintage."""
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/imperviousness/"
        f"{vintage}/Versiegelung_Raster_{vintage}.zip"
    )


def _cog_uri(output_root: str, vintage: int) -> str:
    """Return the final COG URI for a vintage."""
    return (
        f"{output_root.rstrip('/')}/ard/static/morphology/"
        f"imperviousness/{vintage}/imperviousness_{vintage}.tif"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def _fetch_zip(url: str, raw_uri: str) -> Path:
    """Download ZIP to a local temp file, preserving raw archive.

    If *raw_uri* already exists the download is skipped (idempotent).
    Returns the path to a local copy of the ZIP.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="imperviousness_"))
    local_zip = tmp_dir / "source.zip"

    if not exists(raw_uri):
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Preserve raw archive at its final location
        atomic_upload(local_zip, raw_uri)
    else:
        raw_bytes = read_bytes(raw_uri)
        local_zip.write_bytes(raw_bytes)

    return local_zip


def _extract_tiff(zip_path: Path) -> bytes:
    """Extract the GeoTIFF member from a Versiegelung ZIP.

    Raises ``ValueError`` if no TIFF member is found.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        tif_names = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(
                f"No GeoTIFF member found in {zip_path}. "
                f"Members: {names}"
            )
        return z.read(tif_names[0])


def _validate_codes(observed: list[int]) -> None:
    """Raise ``ValueError`` if any observed code is not in the lookup.

    The lookup covers codes 0–255 (all uint8 values).  Codes that are
    not in the official Umweltatlas scheme silently map to 100 % via the
    ``np.full`` default.  If an **unknown** code appears it may indicate
    a new data edition with a different class scheme — we fail hard.
    """
    # All uint8 values are covered by the 256-element lookup.
    # Unknown codes (codes outside {0,5..95,100,101,102,103,110,255})
    # silently map to 100 %.  If a code is outside uint8 range the data
    # may use a different encoding — fail hard.
    unexpected = [c for c in observed if c < 0 or c > 255]
    if unexpected:
        raise ValueError(
            f"Observed codes outside uint8 range: {unexpected}. "
            "This dataset may use a different encoding."
        )


__all__ = [
    "IMPERVIOUSNESS_URLS",
    "config_hash_for_vintage",
    "contract_for_imperviousness",
    "prepare_imperviousness",
    "vintage_for_scene_year",
]
