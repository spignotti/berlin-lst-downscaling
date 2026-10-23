"""TESSERA annual product layout — int8 COG plus scale band.

Canonical product prefix (new, operator-written only)::

    gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate/<year>/
        ├─ tessera_<year>_int8.tif    # 128-band int8 COG on the canonical 10 m grid
        ├─ tessera_<year>_scale.tif   # 1-band float32 scale COG (NaN = gap)
        ├─ tessera_<year>_valid.tif   # 1-band uint8 validity mask COG (1 = valid)
        ├─ provenance.json
        └─ complete.json              # publication marker (written last)

Storage estimate: int8 + scale ≈ 2.3 GB per year on the Berlin bbox.

Rationale: the quantized store (int8 + per-pixel scale) is ~4× smaller
than float32 embeddings and round-trips exactly through the nearest
reprojection; the product keeps that representation instead of a 128-band
float32 COG. ``dequantized = int8 * scale``; scale NaN (or a sub-threshold
quality pixel recorded in provenance) means TESSERA-invalid.
"""

from __future__ import annotations

from typing import Any

TESSERA_PRODUCT_PREFIX: str = "gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate"


def product_dir_for_year(root: str, year: int) -> str:
    """Return the product directory for one TESSERA year."""
    return f"{root.rstrip('/')}/{int(year)}"


def product_uri_for_year(root: str = TESSERA_PRODUCT_PREFIX, year: int = 0) -> str:
    """Return the product directory URI for one year (default canonical root)."""
    return product_dir_for_year(root, year)


def product_cog_uri(root: str, year: int) -> str:
    """Return the 128-band int8 COG URI for one year."""
    return f"{product_dir_for_year(root, year)}/tessera_{int(year)}_int8.tif"


def product_scale_uri(root: str, year: int) -> str:
    """Return the float32 scale COG URI for one year."""
    return f"{product_dir_for_year(root, year)}/tessera_{int(year)}_scale.tif"


def product_valid_uri(root: str, year: int) -> str:
    """Return the uint8 validity-mask COG URI for one year."""
    return f"{product_dir_for_year(root, year)}/tessera_{int(year)}_valid.tif"


def product_provenance_uri(root: str, year: int) -> str:
    """Return the provenance JSON URI for one year."""
    return f"{product_dir_for_year(root, year)}/provenance.json"


def product_completion_uri(root: str, year: int) -> str:
    """Return the completion-marker URI for one year (written last)."""
    return f"{product_dir_for_year(root, year)}/complete.json"


def assert_tessera_product_year(year: int) -> int:
    """Validate a product year against the 2017–2025 contract (2026 refused)."""
    from berlin_lst_downscaling.data.embeddings.tessera import validate_tessera_year

    return validate_tessera_year(int(year))


def write_tessera_year_product(
    *,
    quantized: Any,
    scale: Any,
    valid: Any,
    transform: Any,
    crs: str,
    output_dir: str,
    year: int,
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Write one annual TESSERA product year to a local directory (operator).

    Writes the int8 COG, the scale COG, the validity mask COG, the
    provenance, and the completion marker (last). All rasters are on the
    canonical 10 m grid passed in ``transform``/``crs``. Local-only: the
    operator uploads to GCS; tests call this against ``.scratch/``.

    Returns the artifact paths keyed by role.
    """
    import json
    import os

    import numpy as np
    import rasterio
    from rasterio.crs import CRS as _CRS

    assert_tessera_product_year(int(year))
    q = np.asarray(quantized)
    s = np.asarray(scale)
    v = np.asarray(valid)
    if q.ndim != 3 or q.shape[2] != 128:
        raise ValueError(f"quantized must be (H, W, 128) int8, got {tuple(q.shape)}")
    if q.dtype != np.int8:
        raise ValueError(f"quantized must be int8, got {q.dtype}")
    if s.ndim != 2 or s.shape != q.shape[:2]:
        raise ValueError("scale must be 2-D matching the quantized spatial shape")
    if v.shape != q.shape[:2]:
        raise ValueError("valid mask must match the quantized spatial shape")

    os.makedirs(output_dir, exist_ok=True)
    h, w = q.shape[:2]
    crs_obj = _CRS.from_string(str(crs))
    profile_int8 = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 128,
        "dtype": "int8",
        "crs": crs_obj,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    int8_path = os.path.join(output_dir, f"tessera_{int(year)}_int8.tif")
    with rasterio.open(int8_path, "w", **profile_int8) as dst:
        for band in range(128):
            dst.write(q[:, :, band], band + 1)

    profile_scale = dict(profile_int8)
    profile_scale.update({"count": 1, "dtype": "float32", "nodata": float("nan")})
    scale_path = os.path.join(output_dir, f"tessera_{int(year)}_scale.tif")
    with rasterio.open(scale_path, "w", **profile_scale) as dst:
        dst.write(s.astype(np.float32), 1)

    profile_valid = dict(profile_int8)
    profile_valid.update({"count": 1, "dtype": "uint8", "nodata": None})
    valid_path = os.path.join(output_dir, f"tessera_{int(year)}_valid.tif")
    with rasterio.open(valid_path, "w", **profile_valid) as dst:
        dst.write(v.astype(np.uint8), 1)

    provenance_path = os.path.join(output_dir, "provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)

    completion_path = os.path.join(output_dir, "complete.json")
    with open(completion_path, "w", encoding="utf-8") as fh:
        import datetime as _dt

        json.dump(
            {
                "year": int(year),
                "complete": True,
                "published_at": _dt.datetime.now(_dt.UTC).isoformat(),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return {
        "int8": int8_path,
        "scale": scale_path,
        "valid": valid_path,
        "provenance": provenance_path,
        "complete": completion_path,
    }


__all__ = [
    "TESSERA_PRODUCT_PREFIX",
    "assert_tessera_product_year",
    "product_completion_uri",
    "product_cog_uri",
    "product_dir_for_year",
    "product_provenance_uri",
    "product_scale_uri",
    "product_uri_for_year",
    "product_valid_uri",
    "write_tessera_year_product",
]
