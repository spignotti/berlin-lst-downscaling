"""Output contract per sensor: band specs, tiling, schema version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

_NAN = float("nan")

# ── band / tiling specs ──────────────────────────────────────────────


@dataclass(frozen=True)
class BandSpec:
    """Description of one band in the output COG."""

    name: str
    dtype: str
    nodata: float | None
    description: str
    unit: str = ""
    valid_range: tuple[float, float] | None = None  # (min, max) inclusive


@dataclass(frozen=True)
class TilingSpec:
    """COG internal tiling and compression."""

    blocksize: int = 512
    overviews: tuple[int, ...] = (2, 4, 8, 16)
    compress: str = "deflate"
    predictor: int = 2


# ── contract ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Contract:
    """Immutable specification for one sensor's ARD output."""

    source: str
    target_crs: str
    output_bands: tuple[BandSpec, ...]
    tiling: TilingSpec
    schema_version: int
    flag_mode: str = "separate"  # "separate" (own COG), "inline" (Phase A), or "none"

    # ── canonical flag-band bit layout (shared across sources) ──────
    FLAG_FILL: ClassVar[int] = 1 << 0
    FLAG_CLOUDY: ClassVar[int] = 1 << 1
    FLAG_SHADOW: ClassVar[int] = 1 << 2
    FLAG_CIRRUS: ClassVar[int] = 1 << 3
    FLAG_SATURATED: ClassVar[int] = 1 << 4
    # bits 5-7 reserved

    # ── helpers ─────────────────────────────────────────────────────

    def schema_version_str(self) -> str:
        """Return schema version as a string for ledger/STAC storage."""
        return str(self.schema_version)


# ── factories ────────────────────────────────────────────────────────


def contract_for_source(source: str) -> Contract:
    """Return the :class:`Contract` for a given sensor source key.

    The source key must match the Hydra ``sources`` list entries.
    """
    _bands = _CONTRACTS[source]
    return Contract(
        source=source,
        target_crs="EPSG:25833",
        output_bands=_bands,  # flag band is separate (own uint8 COG)
        tiling=TilingSpec(),
        schema_version=6,
        flag_mode="separate",
    )


# ── per-source band lists ────────────────────────────────────────────

_LANDSAT_BANDS = (
    BandSpec(
        name="st",
        dtype="float32",
        nodata=_NAN,
        description="Surface Temperature derived from LWIR11; Kelvin",
    ),
)

_S2_BANDS = (
    BandSpec(
        name="B02",
        dtype="float32",
        nodata=_NAN,
        description="Sentinel-2 band 2 (blue); scaled reflectance 0-1",
    ),
    BandSpec(
        name="B03",
        dtype="float32",
        nodata=_NAN,
        description="Sentinel-2 band 3 (green); scaled reflectance 0-1",
    ),
    BandSpec(
        name="B04",
        dtype="float32",
        nodata=_NAN,
        description="Sentinel-2 band 4 (red); scaled reflectance 0-1",
    ),
    BandSpec(
        name="B08",
        dtype="float32",
        nodata=_NAN,
        description="Sentinel-2 band 8 (NIR); scaled reflectance 0-1",
    ),
)

_ECOSTRESS_BANDS = (
    BandSpec(
        name="lst",
        dtype="float32",
        nodata=_NAN,
        description="ECOSTRESS LST; Kelvin (ECO_L2T_LSTE.002 native).",
    ),
)

_CONTRACTS: dict[str, tuple[BandSpec, ...]] = {
    "landsat-c2-l2": _LANDSAT_BANDS,
    "sentinel-2-l2a": _S2_BANDS,
    "ecostress": _ECOSTRESS_BANDS,
}

__all__ = [
    "BandSpec",
    "TilingSpec",
    "Contract",
    "contract_for_source",
]
