"""Shared COG layout validation and semantic raster comparison.

Provides strict COG validation and blockwise raster comparison used by
the writer, profiler, and repair harness.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Any

# ── layout classification ─────────────────────────────────────────────


def compute_layout_signature(
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> str:
    """Compute a deterministic signature for a COG's layout issues."""
    combined = sorted(list(strict_errors) + list(strict_warnings))
    if not combined:
        return "CLEAN"
    return hashlib.sha256("|".join(combined).encode()).hexdigest()[:12]


class LayoutClass(StrEnum):
    STRICT_CLEAN = "strict_clean"
    MISSING_OVERVIEW = "missing_overview"
    HARD_LAYOUT = "hard_layout"
    UNEXPECTED = "unexpected"


def classify_layout(
    strict_valid: bool,
    strict_errors: tuple[str, ...],
    strict_warnings: tuple[str, ...],
) -> LayoutClass:
    """Classify COG layout quality from strict validation results."""
    if strict_valid and not strict_errors and not strict_warnings:
        return LayoutClass.STRICT_CLEAN

    if strict_errors:
        known_hard_patterns = [
            "The offset of the main IFD should be < 300",
            "The offset of the first block of overview of index",
            "The offset of the first block of the main resolution image",
            "The file is greater than 512xH or 512xW, but is not tiled",
        ]
        for error in strict_errors:
            for pattern in known_hard_patterns:
                if pattern in error:
                    return LayoutClass.HARD_LAYOUT
        return LayoutClass.UNEXPECTED

    if strict_warnings:
        if len(strict_warnings) == 1:
            warning = strict_warnings[0]
            if "internal overviews" in warning.lower():
                return LayoutClass.MISSING_OVERVIEW
        return LayoutClass.UNEXPECTED

    return LayoutClass.UNEXPECTED


# ── strict validation result ──────────────────────────────────────────


class StrictCogResult:
    """Structured result from strict COG validation.

    ``valid=True`` only when there are zero errors AND zero warnings.
    """

    def __init__(
        self,
        valid: bool,
        errors: tuple[str, ...],
        warnings: tuple[str, ...],
        source: str = "",
    ):
        self.valid = valid and not errors and not warnings
        self.errors = errors
        self.warnings = warnings
        self.source = source
        self.layout_signature = compute_layout_signature(errors, warnings)
        self.layout_class = classify_layout(valid, errors, warnings)

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (
            f"StrictCogResult(valid={self.valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
            f"class={self.layout_class})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "source": self.source,
            "layout_signature": self.layout_signature,
            "layout_class": self.layout_class,
        }


# ── shared validation ─────────────────────────────────────────────────


def validate_strict_cog(uri: str) -> StrictCogResult:
    """Validate COG strict layout using rio-cogeo.

    Returns a ``StrictCogResult`` with ``valid=True`` ONLY when there are
    zero errors AND zero warnings.
    """
    from rio_cogeo.cogeo import cog_validate

    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    try:
        valid, errors, warnings = cog_validate(gdal_uri(uri), strict=True, quiet=True)
        return StrictCogResult(
            valid=valid,
            errors=tuple(f"COG strict: {e}" for e in errors),
            warnings=tuple(f"COG strict warning: {w}" for w in warnings),
            source=uri,
        )
    except FileNotFoundError:
        return StrictCogResult(
            valid=False,
            errors=("rio-cogeo not found in PATH",),
            warnings=(),
            source=uri,
        )
    except Exception as exc:
        return StrictCogResult(
            valid=False,
            errors=(f"COG strict validation failed: {exc}",),
            warnings=(),
            source=uri,
        )


def assert_raster_equivalent(
    source_uri: str,
    repaired_path: str | Path,
    *,
    layout_changed: bool = True,
) -> list[str]:
    """Compare source and repaired rasters for semantic equivalence.

    Requires identical base pixels and overview values, plus identical
    semantic metadata.  Layout fields are excluded when
    ``layout_changed=True``.

    Returns a list of errors; empty means equivalent.
    """
    import math

    import numpy as np
    import rasterio

    errors: list[str] = []

    try:
        with (
            rasterio.open(source_uri) as src,
            rasterio.open(str(repaired_path)) as rep,
        ):
            if src.width != rep.width or src.height != rep.height:
                errors.append(
                    f"Shape mismatch: source ({src.width}, {src.height}) "
                    f"vs repaired ({rep.width}, {rep.height})"
                )
                return errors

            if src.count != rep.count:
                errors.append(f"Band count mismatch: {src.count} vs {rep.count}")
                return errors

            src_crs = str(src.crs).upper() if src.crs else "None"
            rep_crs = str(rep.crs).upper() if rep.crs else "None"
            if src_crs != rep_crs:
                errors.append(f"CRS mismatch: {src_crs} vs {rep_crs}")

            if abs(src.transform.a - rep.transform.a) > 0.001:
                errors.append(
                    f"Resolution mismatch: {abs(src.transform.a)} vs "
                    f"{abs(rep.transform.a)}"
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
                src_nan = isinstance(src.nodata, float) and math.isnan(src.nodata)
                rep_nan = isinstance(rep.nodata, float) and math.isnan(rep.nodata)
                if src_nan != rep_nan:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")
                elif not src_nan and src.nodata != rep.nodata:
                    errors.append(f"NoData mismatch: {src.nodata} vs {rep.nodata}")

            # Per-band metadata
            for i in range(1, src.count + 1):
                src_desc = src.descriptions[i - 1] if src.descriptions else ""
                rep_desc = rep.descriptions[i - 1] if rep.descriptions else ""
                if src_desc != rep_desc:
                    errors.append(
                        f"Band {i} description mismatch: {src_desc!r} vs {rep_desc!r}"
                    )

                if src.scales[i - 1] != rep.scales[i - 1]:
                    errors.append(
                        f"Band {i} scale mismatch: {src.scales[i - 1]} vs "
                        f"{rep.scales[i - 1]}"
                    )
                if src.offsets[i - 1] != rep.offsets[i - 1]:
                    errors.append(
                        f"Band {i} offset mismatch: {src.offsets[i - 1]} vs "
                        f"{rep.offsets[i - 1]}"
                    )

            # Base pixel comparison
            for i in range(1, src.count + 1):
                src_arr = src.read(i)
                rep_arr = rep.read(i)

                src_valid = (
                    ~np.isnan(src_arr)
                    if np.issubdtype(src_arr.dtype, np.floating)
                    else np.ones(src_arr.shape, dtype=bool)
                )
                rep_valid = (
                    ~np.isnan(rep_arr)
                    if np.issubdtype(rep_arr.dtype, np.floating)
                    else np.ones(rep_arr.shape, dtype=bool)
                )

                if not np.array_equal(src_valid, rep_valid):
                    errors.append(f"Band {i} NaN mask mismatch")
                    continue

                if not np.array_equal(src_arr[src_valid], rep_arr[rep_valid]):
                    errors.append(f"Band {i} base pixel values mismatch")
                    continue

            # Overview comparison
            src_overviews = src.overviews(1)
            rep_overviews = rep.overviews(1)
            if src_overviews and rep_overviews:
                for src_ov, rep_ov in zip(
                    src_overviews, rep_overviews, strict=True,
                ):
                    for i in range(1, src.count + 1):
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

                        src_valid = (
                            ~np.isnan(src_ov_arr)
                            if np.issubdtype(src_ov_arr.dtype, np.floating)
                            else np.ones(src_ov_arr.shape, dtype=bool)
                        )
                        rep_valid = (
                            ~np.isnan(rep_ov_arr)
                            if np.issubdtype(rep_ov_arr.dtype, np.floating)
                            else np.ones(rep_ov_arr.shape, dtype=bool)
                        )

                        if not np.array_equal(src_valid, rep_valid):
                            errors.append(
                                f"Band {i} overview {src_ov} NaN mask mismatch"
                            )
                            continue

                        if not np.array_equal(
                            src_ov_arr[src_valid], rep_ov_arr[rep_valid],
                        ):
                            errors.append(
                                f"Band {i} overview {src_ov} values mismatch"
                            )

    except Exception as exc:
        errors.append(f"Comparison failed: {exc}")

    return errors


# ── GDAL COG option validation ────────────────────────────────────────


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
    "StrictCogResult",
    "LayoutClass",
    "compute_layout_signature",
    "classify_layout",
]
