"""Landsat 30 m to 100 m target: area mean for temperature, bit-OR for QA.

``odc-stac`` 0.5.2 defaults ``resampling`` to ``nearest`` when the argument
is omitted (``odc.loader.resolve_load_cfg``). The ARD config names the
method explicitly so the ledger schema hash records it.

The canonical 100 m grid stays put. Its cell centers already sit on the
USGS 30 m center lattice. Cell edges do not: 100 is not a multiple of 30,
so each cell covers three full 30 m pixels plus a 5 m strip of the
neighbor on each side. ``lwir11`` is delivered on that 30 m grid (the STAC
asset is tagged 100 m GSD and has no separate native transform).

QA_PIXEL is a packed bitmask. It is never averaged and never reduced with
``max`` on the integer. Each relevant condition is evaluated at 30 m, then
a 100 m cell takes the condition if any overlapping 30 m pixel has it.
Water stays clear. Cloud dilation stays on the 100 m grid, so
``cloud_dilation_px=2`` is still a 200 m buffer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.data.ard.contract import Contract

# USGS C2 pixel centers fall on multiples of 30 m, so GeoTIFF corners
# (the transform origin) sit 15 m off that lattice.
USGS_30M_M = 30.0
USGS_30M_CORNER_PHASE_M = 15.0
LWIR11_NODATA_DN = 0

ST_RESAMPLING_AVERAGE = "average"
QA_AGGREGATION_BIT_OR = "native_bit_or"
LEGACY_NEAREST = "nearest"


def landsat_policy_token(st_resampling: str, qa_aggregation: str) -> str:
    """Stable token stored in the Landsat ledger schema hash."""
    return f"st={st_resampling}|qa={qa_aggregation}"


def landsat_schema_hash(
    schema_version: int,
    st_resampling: str,
    qa_aggregation: str,
) -> str:
    """Schema hash for one Landsat row.

    The integer schema version stays 7. Reconcile compares that integer,
    so publishing this token does not mark the previous ARD root as
    ``schema_changed``. The feature config hash still changes, because it
    fingerprints the whole ARD ledger file.
    """
    return f"{schema_version}|{landsat_policy_token(st_resampling, qa_aggregation)}"


def policy_from_config(cfg: Any) -> tuple[str, str]:
    """Return ``(st_resampling, qa_aggregation)`` from a Hydra config."""
    st = str(cfg.get("landsat_st_resampling", LEGACY_NEAREST))
    qa = str(cfg.get("landsat_qa_aggregation", LEGACY_NEAREST))
    return st, qa


def usgs_30m_geobox_covering(target: Any) -> Any:
    """Return a 30 m GeoBox on the USGS corner phase that covers ``target``.

    ``target`` is the canonical 100 m GeoBox. The result is expanded
    outward so every 100 m cell is fully covered by 30 m pixels.
    """
    from odc.geo.geobox import GeoBox

    transform = target.transform
    x_min = float(transform.c)
    y_max = float(transform.f)
    x_max = x_min + float(transform.a) * int(target.shape.x)
    y_min = y_max + float(transform.e) * int(target.shape.y)
    x0 = _snap_floor(x_min, USGS_30M_M, USGS_30M_CORNER_PHASE_M)
    y0 = _snap_ceil(y_max, USGS_30M_M, USGS_30M_CORNER_PHASE_M)
    width = int(math.ceil((x_max - x0) / USGS_30M_M))
    height = int(math.ceil((y0 - y_min) / USGS_30M_M))
    affine = rasterio.Affine(USGS_30M_M, 0.0, x0, 0.0, -USGS_30M_M, y0)
    return GeoBox((height, width), affine, target.crs)


def qa_condition_planes(qa: np.ndarray) -> dict[str, np.ndarray]:
    """Relevant QA conditions at native resolution. Water is not one of them."""
    cloud_conf = (qa >> 8) & 0b11
    return {
        "fill": (qa & 1) != 0,
        "cloudy": cloud_conf >= 2,
        "shadow": ((qa >> 4) & 1) != 0,
        "cirrus": ((qa >> 2) & 1) != 0,
        "snow": ((qa >> 5) & 1) != 0,
    }


def aggregate_condition_planes(
    planes: dict[str, np.ndarray],
    src_transform: rasterio.Affine,
    dst_transform: rasterio.Affine,
    dst_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """OR each condition onto the destination grid via resampling max.

    Max of a 0/1 plane is a bitwise OR across every source pixel that
    overlaps the destination cell, including partial edge strips. The
    packed QA integer is not resampled.
    """
    out: dict[str, np.ndarray] = {}
    for name, plane in planes.items():
        destination = np.zeros(dst_shape, dtype=np.uint8)
        reproject(
            plane.astype(np.uint8),
            destination,
            src_transform=src_transform,
            src_crs="EPSG:25833",
            dst_transform=dst_transform,
            dst_crs="EPSG:25833",
            resampling=Resampling.max,
        )
        out[name] = destination.astype(bool)
    return out


def pack_landsat_flags(
    planes: dict[str, np.ndarray],
    contract: Contract,
    *,
    dilate_px: int,
) -> np.ndarray:
    """Pack aggregated conditions and dilate cloud on this grid."""
    flag = np.zeros(planes["fill"].shape, dtype=np.uint8)
    cloudy = planes["cloudy"]
    flag[planes["fill"]] |= contract.FLAG_FILL
    flag[cloudy] |= contract.FLAG_CLOUDY
    flag[planes["shadow"]] |= contract.FLAG_SHADOW
    flag[planes["cirrus"]] |= contract.FLAG_CIRRUS
    flag[planes["snow"]] |= contract.FLAG_SNOW_ICE
    if dilate_px > 0 and cloudy.any():
        from scipy.ndimage import binary_dilation

        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
        buffered = np.asarray(binary_dilation(cloudy, structure=struct), dtype=bool)
        flag[buffered & ~cloudy] |= contract.FLAG_FILL
    return flag


def _snap_floor(value: float, step: float, phase: float) -> float:
    return math.floor((value - phase) / step) * step + phase


def _snap_ceil(value: float, step: float, phase: float) -> float:
    return math.ceil((value - phase) / step) * step + phase
