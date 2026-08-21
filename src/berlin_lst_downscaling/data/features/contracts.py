"""Feature-stack contract — 28-channel roster, formulas, and mask semantics.

The scene feature stack is the model input interface: exactly 28 float32
channels in a fixed order, published per Landsat anchor scene on the
canonical 10 m grid, plus a co-registered ``feature_valid`` uint8 mask.

Validity rules
--------------
A pixel is ``feature_valid`` only when **all** of the following hold:

- inside the exact Berlin AOI (``aoi_10m.tif``, reprojected onto the
  canonical grid with nearest resampling),
- the Sentinel-2 ARD flag band is ``0`` (clear pixel; ``data/ard/contract.py``),
- all 28 channels are finite and within their declared ``valid_range``.

Where ``feature_valid == 0``, all 28 channels are published as NaN.
The mask is *not* a training-eligibility mask: it excludes the Landsat
target validity and cannot authorize training selection. The Stage-2 QA
gate verifies the stacks and reports feature support against valid
targets; publication of the ``training_eligible@100m`` selection mask is
a WB2c-4 (training-data preparation) decision.

vegetation_height carry-forward
-------------------------------
``vegetation_height/2020`` is the only vegetation source. It is read
directly from the static source COG (bands: vegetation_height_mean,
vegetation_height_max) and carried forward to every scene year via the
2024 geometry profile.

Albedo proxy
------------
``s2_broadband_albedo`` is the six-band weighted shortwave-reflectance
proxy of Bonafoni & Sekertekin (2020), IEEE GRSL 17(9):1618-1622
(coefficients as published in the ALBEDO implementation,
Zenodo 10.5281/zenodo.21111867). It estimates a broadband HDRF-style
reflectance under Lambertian/clear-sky assumptions — documented as a
proxy, not a BRDF-corrected physical albedo.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── channel spec ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureChannel:
    """One channel of the fixed 28-band feature stack."""

    name: str
    family: str  # spectral | index | morphology | era5 | shadow
    description: str
    unit: str = "1"
    valid_range: tuple[float, float] | None = None  # (min, max) inclusive


# Bonafoni & Sekertekin (2020) weights for B02/B03/B04/B08/B11/B12.
# Applied to ARD reflectance already scaled to [0, 1]; coefficients sum to 1.
ALBEDO_WEIGHTS: tuple[float, ...] = (0.2266, 0.1236, 0.1573, 0.3417, 0.1170, 0.0338)

# Formulas + channel schema version. Bump when index math or the channel
# roster changes — the value feeds the config hash and invalidates every
# published stack (model interface contract).
FEATURE_SCHEMA_VERSION: int = 2

# Sentinel-2 ARD band names, in published COG order (data/ard/contract.py).
_S2_BAND_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12")

# ERA5-Land band names, in published COG order (data/dynamic/era5.py).
_ERA5_BAND_NAMES = (
    "t2m_scene",
    "ssrd_scene",
    "ssrd_antecedent_72h_mean",
    "vpd_scene",
    "wind_speed_10m_scene",
    "tp_0_24h",
    "tp_24_48h",
    "tp_48_72h",
)

_SVF_RANGE = (-0.01, 1.01)  # mirrors data/qa/contracts.py SVF_RANGE
_INDEX_RANGE = (-1.0, 1.0)
_SHADOW_RANGE = (0.0, 1.0)  # shadows cast to float 0/1; 255 nodata becomes NaN

_ERA5_RANGES: dict[str, tuple[float, float]] = {
    "t2m_scene": (200.0, 350.0),
    "ssrd_scene": (-1.0, 1500.0),
    "ssrd_antecedent_72h_mean": (-1.0, 1500.0),
    "vpd_scene": (0.0, 10.0),
    "wind_speed_10m_scene": (0.0, 50.0),
    "tp_0_24h": (0.0, 500.0),
    "tp_24_48h": (0.0, 500.0),
    "tp_48_72h": (0.0, 500.0),
}


def _s2_channels() -> list[FeatureChannel]:
    return [
        FeatureChannel(
            name=b,
            family="spectral",
            description=f"Sentinel-2 {b} scaled reflectance [0,1]",
            valid_range=(0.0, 1.0),
        )
        for b in _S2_BAND_NAMES
    ]


def _index_channels() -> list[FeatureChannel]:
    return [
        FeatureChannel(
            name="ndvi",
            family="index",
            description="Normalized Difference Vegetation Index (B08-B04)/(B08+B04)",
            valid_range=_INDEX_RANGE,
        ),
        FeatureChannel(
            name="ndwi_mcfeeters",
            family="index",
            description="McFeeters NDWI (B03-B08)/(B03+B08)",
            valid_range=_INDEX_RANGE,
        ),
        FeatureChannel(
            name="ndbi",
            family="index",
            description="Normalized Difference Built-up Index (B11-B08)/(B11+B08)",
            valid_range=_INDEX_RANGE,
        ),
        FeatureChannel(
            name="s2_broadband_albedo",
            family="index",
            description=(
                "Six-band broadband shortwave reflectance proxy (Bonafoni & "
                "Sekertekin 2020); not BRDF-corrected albedo"
            ),
            valid_range=(0.0, 1.0),
        ),
    ]


def _morphology_channels() -> list[FeatureChannel]:
    return [
        FeatureChannel(
            name="building_height_mean",
            family="morphology",
            description="Mean building height in metres above ground (LoD2)",
            unit="m",
            valid_range=(0.0, 200.0),
        ),
        FeatureChannel(
            name="building_height_std",
            family="morphology",
            description="Standard deviation of building heights within cell (LoD2)",
            unit="m",
            valid_range=(0.0, 100.0),
        ),
        FeatureChannel(
            name="building_coverage_ratio",
            family="morphology",
            description="Fraction of cell area covered by building footprints (LoD2)",
            valid_range=(-0.01, 1.01),
        ),
        FeatureChannel(
            name="building_height_max",
            family="morphology",
            description="Maximum building height in cell (LoD2)",
            unit="m",
            valid_range=(0.0, 200.0),
        ),
        FeatureChannel(
            name="vegetation_height_mean",
            family="morphology",
            description=(
                "Mean vegetation height in metres above ground; "
                "fixed vegetation_height/2020 carried forward to every scene year"
            ),
            unit="m",
            valid_range=(-0.01, 400.01),
        ),
        FeatureChannel(
            name="vegetation_height_max",
            family="morphology",
            description=(
                "Maximum vegetation height in metres above ground; "
                "fixed vegetation_height/2020 carried forward to every scene year"
            ),
            unit="m",
            valid_range=(-0.01, 400.01),
        ),
        FeatureChannel(
            name="imperviousness",
            family="morphology",
            description="Sealing degree in percent (Versiegelung)",
            unit="%",
            valid_range=(-0.01, 100.01),
        ),
        FeatureChannel(
            name="svf",
            family="morphology",
            description="Sky View Factor [0,1] from combined DSM (Zakek 2011)",
            valid_range=_SVF_RANGE,
        ),
    ]


def _era5_channels() -> list[FeatureChannel]:
    # Units mirror the upstream ERA5 contract (data/dynamic/era5.py).
    units = {
        "t2m_scene": "K",
        "ssrd_scene": "W/m²",
        "ssrd_antecedent_72h_mean": "W/m²",
        "vpd_scene": "kPa",
        "wind_speed_10m_scene": "m/s",
        "tp_0_24h": "mm",
        "tp_24_48h": "mm",
        "tp_48_72h": "mm",
    }
    return [
        FeatureChannel(
            name=n,
            family="era5",
            description=f"ERA5-Land {n} at acquisition",
            unit=units[n],
            valid_range=_ERA5_RANGES[n],
        )
        for n in _ERA5_BAND_NAMES
    ]


def _shadow_channels() -> list[FeatureChannel]:
    return [
        FeatureChannel(
            name=n,
            family="shadow",
            description=f"{n} (0 lit, 1 shadowed); 255 nodata cast to NaN",
            valid_range=_SHADOW_RANGE,
        )
        for n in ("shadow_building", "shadow_vegetation")
    ]


# Fixed 28-channel order — the model input interface. Never reorder without
# bumping FEATURE_SCHEMA_VERSION and publishing under a new features URI version.
FEATURE_CHANNELS: tuple[FeatureChannel, ...] = tuple(
    [
        *_s2_channels(),
        *_index_channels(),
        *_morphology_channels(),
        *_era5_channels(),
        *_shadow_channels(),
    ]
)

FEATURE_CHANNEL_NAMES: tuple[str, ...] = tuple(ch.name for ch in FEATURE_CHANNELS)

_N_CHANNELS = len(FEATURE_CHANNELS)
if _N_CHANNELS != 28:  # pragma: no cover — module invariant
    raise AssertionError(f"Feature stack must have 28 channels, got {_N_CHANNELS}")


__all__ = [
    "ALBEDO_WEIGHTS",
    "FEATURE_CHANNELS",
    "FEATURE_CHANNEL_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FeatureChannel",
]
