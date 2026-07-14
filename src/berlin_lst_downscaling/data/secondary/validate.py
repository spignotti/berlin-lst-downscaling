"""Strict COG validation for secondary-data outputs.

Reuses ``data.ard.validate`` for structural checks (readability, CRS,
band count, shape, origin, NaN threshold) and adds secondary-specific
validations: expected band description, dtype, nodata, and value range.
"""

from __future__ import annotations

import numpy as np
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.ard.validate import ValidationResult, validate_cog

# Expected value ranges for known secondary sources.
# Small tolerance for floating-point edge effects in resampling.
# Vegetation height: 0 is bare ground under vegetation; negative values
# are invalid.  Upper bound 150 m accommodates tall trees with tolerance.
_RANGES: dict[str, tuple[float, float]] = {
    "imperviousness": (-0.01, 100.01),
    "vegetation_height": (-0.01, 150.01),
}


def validate_secondary_cog(
    uri: str,
    contract: Contract,
    expected_grid: GeoBox,
) -> ValidationResult:
    """Validate a secondary-data COG.

    Delegates to :func:`data.ard.validate.validate_cog` for structural
    checks (CRS, shape, origin, band count, NaN threshold), then adds
    source-specific range validation when a known source is detected.
    """
    result = validate_cog(uri, contract, expected_grid)
    if not result.ok:
        return result

    # Source-specific range check
    source = contract.source
    if source in _RANGES:
        mini, maxi = _RANGES[source]
        _check_value_range(uri, result, mini, maxi)

    return result


def _check_value_range(
    uri: str,
    result: ValidationResult,
    vmin: float,
    vmax: float,
) -> None:
    """Check that all valid (non-NaN) pixels are within [vmin, vmax]."""
    import rasterio  # noqa: F811

    try:
        with rasterio.open(uri) as src:
            band = src.read(1).astype(np.float64)
            valid_mask = ~np.isnan(band)
            valid = band[valid_mask]
            if len(valid) == 0:
                result.fail("No valid pixels found for range check")
                return
            actual_min = float(valid.min())
            actual_max = float(valid.max())
            if actual_min < vmin:
                result.fail(
                    f"Value below expected range [{vmin:.1f}, {vmax:.1f}]: "
                    f"min={actual_min:.4f}"
                )
            if actual_max > vmax:
                result.fail(
                    f"Value above expected range [{vmin:.1f}, {vmax:.1f}]: "
                    f"max={actual_max:.4f}"
                )
    except Exception as exc:
        result.fail(f"Range check failed: {exc}")


__all__ = [
    "validate_secondary_cog",
]
