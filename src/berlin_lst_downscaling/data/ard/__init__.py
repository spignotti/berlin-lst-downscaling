"""ARD processing — write-out pipeline (contracts, masking, COG/STAC, ledger, orchestration).

Phase A: Landsat C2-L2 + Sentinel-2 L2A via PC STAC (ECOSTRESS Phase B).
"""

from berlin_lst_downscaling.data.ard.contract import (
    BandSpec,
    Contract,
    TilingSpec,
    contract_for_source,
)
from berlin_lst_downscaling.data.ard.masking import mask_landsat, mask_s2
from berlin_lst_downscaling.data.ard.paths import cog_path, scene_dir, stac_path, tmp_dir
from berlin_lst_downscaling.data.ard.solar_position import solar_position, solar_position_from_stac
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic, write_stac_atomic

__all__ = [
    "BandSpec",
    "Contract",
    "TilingSpec",
    "contract_for_source",
    "scene_dir",
    "cog_path",
    "stac_path",
    "tmp_dir",
    "mask_landsat",
    "mask_s2",
    "solar_position",
    "solar_position_from_stac",
    "write_cog_atomic",
    "write_stac_atomic",
]
