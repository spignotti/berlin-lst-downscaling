"""Shared COG layout validation and semantic raster comparison.

Provides strict COG validation and blockwise raster comparison used by
the writer, profiler, and repair harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_validate


def validate_strict_cog(uri: str, *, ignore_warnings: bool = True) -> list[str]:
    """Validate COG strict layout using rio-cogeo.

    Returns a list of errors; empty means strict-clean.
    Warnings are ignored by default since they don't affect functionality.
    """
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    try:
        valid, errors, warnings = cog_validate(gdal_uri(uri), strict=True, quiet=True)
        result: list[str] = []
        for err in errors:
            result.append(f"COG strict: {err}")
        if not ignore_warnings:
            for warn in warnings:
                result.append(f"COG strict warning: {warn}")
        return result
    except FileNotFoundError:
        return ["rio-cogeo not found in PATH"]
    except Exception as exc:
        return [f"COG strict validation failed: {exc}"]


def assert_raster_equivalent(
    source_uri: str,
    repaired_path: str | Path,
    *,
    layout_changed: bool = True,
) -> list[str]:
    """Compare source and repaired rasters for semantic equivalence.

    Requires identical base pixels and existing overview values, plus
    identical semantic metadata. Layout fields (compression, IFD offsets,
    block shapes) are intentionally excluded when layout_changed=True.

    Returns a list of errors; empty means equivalent.
    """
    errors: list[str] = []

    try:
        with rasterio.open(source_uri) as src, rasterio.open(str(repaired_path)) as rep:
            # Basic shape and band count
            if src.width != rep.width or src.height != rep.height:
                errors.append(
                    f"Shape mismatch: source ({src.width}, {src.height}) "
                    f"vs repaired ({rep.width}, {rep.height})"
                )
                return errors

            if src.count != rep.count:
                errors.append(f"Band count mismatch: {src.count} vs {rep.count}")
                return errors

            # CRS and transform
            src_crs = str(src.crs).upper() if src.crs else "None"
            rep_crs = str(rep.crs).upper() if rep.crs else "None"
            if src_crs != rep_crs:
                errors.append(f"CRS mismatch: {src_crs} vs {rep_crs}")

            if abs(src.transform.a - rep.transform.a) > 0.001:
                errors.append(
                    f"Resolution mismatch: {abs(src.transform.a)} vs {abs(rep.transform.a)}"
                )
            if abs(src.transform.c - rep.transform.c) > 0.01:
                errors.append(
                    f"X origin mismatch: {src.transform.c} vs {rep.transform.c}"
                )
            if abs(src.transform.f - rep.transform.f) > 0.01:
                errors.append(
                    f"Y origin mismatch: {src.transform.f} vs {rep.transform.f}"
                )

            # NoData
            if src.nodata is None and rep.nodata is not None:
                errors.append(f"NoData mismatch: None vs {rep.nodata}")
            elif src.nodata is not None and rep.nodata is None:
                errors.append(f"NoData mismatch: {src.nodata} vs None")
            elif src.nodata is not None and rep.nodata is not None:
                # Handle NaN comparison
                import math
                src_is_nan = isinstance(src.nodata, float) and math.isnan(src.nodata)
                rep_is_nan = isinstance(rep.nodata, float) and math.isnan(rep.nodata)
                if src_is_nan != rep_is_nan:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")
                elif not src_is_nan and src.nodata != rep.nodata:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")

            # Per-band checks
            for i in range(1, src.count + 1):
                # Description
                src_desc = src.descriptions[i - 1] if src.descriptions else ""
                rep_desc = rep.descriptions[i - 1] if rep.descriptions else ""
                if src_desc != rep_desc:
                    errors.append(f"Band {i} description mismatch: {src_desc!r} vs {rep_desc!r}")

                # Scales and offsets
                if src.scales[i - 1] != rep.scales[i - 1]:
                    errors.append(
                        f"Band {i} scale mismatch: {src.scales[i - 1]} vs {rep.scales[i - 1]}"
                    )
                if src.offsets[i - 1] != rep.offsets[i - 1]:
                    errors.append(
                        f"Band {i} offset mismatch: {src.offsets[i - 1]} vs {rep.offsets[i - 1]}"
                    )

            # Base pixel comparison
            for i in range(1, src.count + 1):
                src_arr = src.read(i)
                rep_arr = rep.read(i)

                # Handle NaN
                if np.issubdtype(src_arr.dtype, np.floating):
                    src_valid = ~np.isnan(src_arr)
                else:
                    src_valid = np.ones(src_arr.shape, dtype=bool)
                if np.issubdtype(rep_arr.dtype, np.floating):
                    rep_valid = ~np.isnan(rep_arr)
                else:
                    rep_valid = np.ones(rep_arr.shape, dtype=bool)

                if not np.array_equal(src_valid, rep_valid):
                    errors.append(f"Band {i} NaN mask mismatch")
                    continue

                if not np.array_equal(src_arr[src_valid], rep_arr[rep_valid]):
                    errors.append(f"Band {i} base pixel values mismatch")
                    continue

            # Overview comparison (if source has overviews)
            src_overviews = src.overviews(1)
            rep_overviews = rep.overviews(1)
            if src_overviews and rep_overviews:
                for src_ov, rep_ov in zip(src_overviews, rep_overviews, strict=True):
                    for i in range(1, src.count + 1):
                        # Read overview using out_shape
                        src_h = src.height // src_ov
                        src_w = src.width // src_ov
                        rep_h = rep.height // rep_ov
                        rep_w = rep.width // rep_ov
                        
                        src_ov_arr = src.read(i, out_shape=(src_h, src_w))
                        rep_ov_arr = rep.read(i, out_shape=(rep_h, rep_w))

                        if src_ov_arr.shape != rep_ov_arr.shape:
                            errors.append(
                                f"Band {i} overview {src_ov} shape mismatch: "
                                f"{src_ov_arr.shape} vs {rep_ov_arr.shape}"
                            )
                            continue

                        if np.issubdtype(src_ov_arr.dtype, np.floating):
                            src_valid = ~np.isnan(src_ov_arr)
                        else:
                            src_valid = np.ones(src_ov_arr.shape, dtype=bool)
                        if np.issubdtype(rep_ov_arr.dtype, np.floating):
                            rep_valid = ~np.isnan(rep_ov_arr)
                        else:
                            rep_valid = np.ones(rep_ov_arr.shape, dtype=bool)

                        if not np.array_equal(src_valid, rep_valid):
                            errors.append(f"Band {i} overview {src_ov} NaN mask mismatch")
                            continue

                        if not np.array_equal(src_ov_arr[src_valid], rep_ov_arr[rep_valid]):
                            errors.append(f"Band {i} overview {src_ov} values mismatch")

    except Exception as exc:
        errors.append(f"Comparison failed: {exc}")

    return errors


def validate_gdal_cog_options(
    creation_options: dict[str, Any],
    overview_resampling: str,
    has_overviews: bool,
) -> list[str]:
    """Validate that GDAL COG creation options are supported by installed GDAL.

    Returns a list of errors; empty means all options are valid.
    """
    errors: list[str] = []

    required_keys = [
        "BLOCKSIZE", "COMPRESS", "PREDICTOR", "BIGTIFF",
    ]
    if has_overviews:
        required_keys.append("OVERVIEW_RESAMPLING")

    for key in required_keys:
        if key not in creation_options:
            errors.append(f"Missing required COG option: {key}")

    return errors


__all__ = [
    "validate_strict_cog",
    "assert_raster_equivalent",
    "validate_gdal_cog_options",
]
