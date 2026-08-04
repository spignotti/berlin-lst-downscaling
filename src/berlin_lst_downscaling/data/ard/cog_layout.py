"""Shared COG layout validation and semantic raster comparison.

Provides strict COG validation and blockwise raster comparison used by
the writer, profiler, and repair harness.

Re-exports ``validate_strict_cog`` and ``assert_raster_equivalent``
from ``cog_recovery_state`` for backward compatibility with existing
importers (``writer.py``, ``cog_repair.py``, etc.).
"""

from __future__ import annotations

from typing import Any

from berlin_lst_downscaling.data.ard.cog_recovery_state import (
    StrictCogResult,
    assert_raster_equivalent,
    validate_strict_cog,
)


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
]
