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


def validate_secondary_cog(
    uri: str,
    contract: Contract,
    expected_grid: GeoBox,
) -> ValidationResult:
    """Validate a secondary-data COG.

    Delegates to :func:`data.ard.validate.validate_cog` for structural
    checks (CRS, shape, origin, band count, NaN threshold), then runs
    the per-band range check for every band that defines ``valid_range``.
    """
    result = validate_cog(uri, contract, expected_grid)
    if not result.ok:
        return result

    _check_all_band_ranges(uri, contract, result)
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
                band = src.read(i)
                nodata = spec.nodata

                if np.issubdtype(band.dtype, np.floating):
                    valid_mask = ~np.isnan(band)
                elif nodata is not None:
                    valid_mask = band != nodata
                else:
                    valid_mask = np.ones(band.shape, dtype=bool)

                band_f = band.astype(np.float64)
                valid = band_f[valid_mask]
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

__all__ = [
    "validate_secondary_cog",
]