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
    "vegetation_height_mean": (-0.01, 150.01),
    "vegetation_height_max": (-0.01, 150.01),
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

    # Per-band range validation using contract valid_range
    _check_all_band_ranges(uri, contract, result)

    # Source-level range check (backwards compat for sources without per-band range)
    source = contract.source
    if source in _RANGES:
        mini, maxi = _RANGES[source]
        _check_value_range(uri, result, mini, maxi, band=1)

    return result


def _check_all_band_ranges(
    uri: str,
    contract: Contract,
    result: ValidationResult,
) -> None:
    """Check valid_range for each band that has one defined in the contract."""
    import rasterio  # noqa: F811

    try:
        with rasterio.open(uri) as src:
            for i, spec in enumerate(contract.output_bands, 1):
                if spec.valid_range is None:
                    continue
                vmin, vmax = spec.valid_range
                band = src.read(i).astype(np.float64)
                valid_mask = ~np.isnan(band)
                valid = band[valid_mask]
                if len(valid) == 0:
                    result.fail(f"Band {i} ({spec.name}): no valid pixels for range check")
                    continue
                actual_min = float(valid.min())
                actual_max = float(valid.max())
                if actual_min < vmin:
                    result.fail(
                        f"Band {i} ({spec.name}): value below range "
                        f"[{vmin:.1f}, {vmax:.1f}]: min={actual_min:.4f}"
                    )
                if actual_max > vmax:
                    result.fail(
                        f"Band {i} ({spec.name}): value above range "
                        f"[{vmin:.1f}, {vmax:.1f}]: max={actual_max:.4f}"
                    )
    except Exception as exc:
        result.fail(f"Per-band range check failed: {exc}")


def _check_value_range(
    uri: str,
    result: ValidationResult,
    vmin: float,
    vmax: float,
    band: int = 1,
) -> None:
    """Check that all valid (non-NaN) pixels in a band are within [vmin, vmax]."""
    import rasterio  # noqa: F811

    try:
        with rasterio.open(uri) as src:
            arr = src.read(band).astype(np.float64)
            valid_mask = ~np.isnan(arr)
            valid = arr[valid_mask]
            if len(valid) == 0:
                result.fail(f"Band {band}: no valid pixels found for range check")
                return
            actual_min = float(valid.min())
            actual_max = float(valid.max())
            if actual_min < vmin:
                result.fail(
                    f"Band {band}: value below expected range [{vmin:.1f}, {vmax:.1f}]: "
                    f"min={actual_min:.4f}"
                )
            if actual_max > vmax:
                result.fail(
                    f"Band {band}: value above expected range [{vmin:.1f}, {vmax:.1f}]: "
                    f"max={actual_max:.4f}"
                )
    except Exception as exc:
        result.fail(f"Range check failed for band {band}: {exc}")


__all__ = [
    "validate_secondary_cog",
]
