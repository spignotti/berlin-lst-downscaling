"""Secondary-data contracts — reuses ard.contract with ``flag_mode='none'``.

Secondary sources (static morphology, dynamic meteorology, derived layers
like SVF and shadow masks) do not carry the per-pixel flag bands that
satellite sources do.  Every secondary contract uses ``flag_mode='none'``.

Specific channel contracts (building_height, imperviousness, canopy_height,
terrain, SVF, t2m, ssrd, etc.) are defined alongside their source modules
in later pipeline stages.
"""

from berlin_lst_downscaling.data.ard.contract import (
    BandSpec,
    Contract,
    TilingSpec,
)

__all__ = [
    "BandSpec",
    "Contract",
    "TilingSpec",
]
