"""ARD processing — write-out pipeline (contracts, masking, COG/STAC, ledger, orchestration).

Supports Landsat C2-L2, Sentinel-2 L2A, and ECOSTRESS L2T.
"""

from berlin_lst_downscaling.data.ard.contract import (
    BandSpec,
    Contract,
    TilingSpec,
    contract_for_source,
)
from berlin_lst_downscaling.data.ard.idempotency import reconcile
from berlin_lst_downscaling.data.ard.ledger import Ledger, LedgerRow
from berlin_lst_downscaling.data.ard.masking import mask_landsat, mask_s2
from berlin_lst_downscaling.data.ard.paths import (
    cog_path,
    flag_path,
    scene_dir,
    stac_path,
)
from berlin_lst_downscaling.data.ard.pipeline import run as ard_run
from berlin_lst_downscaling.data.ard.reports import qa_report
from berlin_lst_downscaling.data.ard.solar_position import (
    solar_position,
)
from berlin_lst_downscaling.data.ard.writer import (
    write_cog_atomic,
    write_flag_cog_atomic,
    write_stac_atomic,
)

__all__ = [
    "BandSpec",
    "Contract",
    "TilingSpec",
    "contract_for_source",
    "scene_dir",
    "cog_path",
    "flag_path",
    "stac_path",
    "mask_landsat",
    "mask_s2",
    "solar_position",
    "write_cog_atomic",
    "write_flag_cog_atomic",
    "write_stac_atomic",
    "Ledger",
    "LedgerRow",
    "reconcile",
    "ard_run",
    "qa_report",
]