"""Strict COG validation for secondary-data outputs.

Reuses ``data.ard.validate`` for structural checks (readability, CRS,
band count, shape, origin, NaN threshold) and provides a secondary-specific
entry point for future extension.
"""

from __future__ import annotations

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
    checks (CRS, shape, origin, band count, NaN threshold).

    Secondary-specific checks (band names, dtype, coverage fraction) are
    added alongside specific source modules in later pipeline stages.
    """
    return validate_cog(uri, contract, expected_grid)


__all__ = [
    "validate_secondary_cog",
]
