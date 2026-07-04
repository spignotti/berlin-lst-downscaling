"""ARD processing — write-out pipeline (contracts, masking, COG/STAC, ledger, orchestration).

Phase A: Landsat C2-L2 + Sentinel-2 L2A via PC STAC (ECOSTRESS Phase B).
"""

from berlin_lst_downscaling.data.ard.contract import (
    BandSpec,
    Contract,
    TilingSpec,
    contract_for_source,
)
from berlin_lst_downscaling.data.ard.paths import cog_path, scene_dir, stac_path, tmp_dir

__all__ = [
    "BandSpec",
    "Contract",
    "TilingSpec",
    "contract_for_source",
    "scene_dir",
    "cog_path",
    "stac_path",
    "tmp_dir",
]
