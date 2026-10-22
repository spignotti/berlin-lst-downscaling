"""Feature-stack contract — channel roster, formulas, and mask semantics.

``FEATURE_CHANNELS`` is the published features/v5 roster (schema version 3):
28 float32 channels in a fixed order. features/v6 is schema version 5 and
is returned by :func:`feature_schema`. Both are published per Landsat
anchor scene on the canonical 10 m grid, plus a co-registered
``feature_valid`` uint8 mask. Channel count is ``len(schema.channels)``.

Validity rules
--------------
A pixel is ``feature_valid`` only when **all** of the following hold:

- inside the exact Berlin AOI (``aoi_10m.tif``, reprojected onto the
  canonical grid with nearest resampling),
- the Sentinel-2 ARD flag band is ``0`` (clear pixel; ``data/ard/contract.py``),
- all 28 channels are finite and within their declared ``valid_range``.

Availability is per channel: where a channel is unavailable (e.g. the
S2 bands and their derived indices under a non-clear flag, or a genuine
source-data gap in a morphology product) that band alone is NaN — the
independently known values of the other channels are preserved in the
stack. Only outside the AOI are all 28 channels published as NaN. The
``feature_valid`` mask is the aggregate "complete 28-channel vector"
availability and is *not* a training-eligibility mask: it excludes the
Landsat target validity and cannot authorize training selection. The
Stage-2 QA gate verifies the stacks and reports feature support against
valid targets; publication of the ``training_eligible@100m`` selection
mask is a WB2c-4 (training-data preparation) decision.

LoD2 no-building semantics
--------------------------
``building_height_mean/std/max`` and ``building_coverage_ratio`` are
``0`` in source-covered cells without a building (a known value), and
``NaN`` in cells outside the LoD2 source-tile coverage (a true data gap).
Coverage is reconstructed at composition time from the immutable raw
archive manifests / LoD provenance (``data/features/lod_coverage.py``);
finite source values always win over the coverage classification.

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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
# 4: S2 reflectance is the baseline-harmonized ARD (schema 9). The stack
# does not apply a second offset. LoD morphology consumed by the stack
# is the BuildingPart / area-exact product (lod2_morphology config v2).
FEATURE_SCHEMA_VERSION: int = 4

# features/v6 channel roster. Version 4 stays the 28-channel roster from
# the data-correctness bump. Version 5 drops building height std/max,
# replaces vegetation_height_max with canopy cover, and slims ERA5.
FEATURE_SCHEMA_VERSION_V6: int = 5

# Provisional canopy threshold (metres). Literature may change the v6
# config; the value is part of the v6 config hash, not a silent constant
# inside the aggregator.
CANOPY_COVER_THRESHOLD_M_DEFAULT: float = 2.5

ERA5_REPRESENTATION_SCENE_VALUE: str = "scene_value"
ERA5_REPRESENTATION_BILINEAR: str = "bilinear"
ERA5_REPRESENTATIONS: tuple[str, ...] = (
    ERA5_REPRESENTATION_SCENE_VALUE,
    ERA5_REPRESENTATION_BILINEAR,
)
# Loader default for schema v6. The published dynamic v5 product stays
# bilinear unless a v6 dynamic config selects scene_value.
ERA5_REPRESENTATION_V6_DEFAULT: str = ERA5_REPRESENTATION_SCENE_VALUE

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
            description=(
                f"Sentinel-2 {b} BOA reflectance harmonized per scene "
                "from product metadata, scaled to [0, 1]"
            ),
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


def _morphology_channels_v6(canopy_cover_threshold_m: float) -> list[FeatureChannel]:
    """v6 morphology: drop height std/max, replace vegetation max with canopy cover."""
    threshold = f"{canopy_cover_threshold_m:g}"
    return [
        FeatureChannel(
            name="building_height_mean",
            family="morphology",
            description="Mean building height in metres above ground (LoD2)",
            unit="m",
            valid_range=(0.0, 200.0),
        ),
        FeatureChannel(
            name="building_coverage_ratio",
            family="morphology",
            description="Fraction of cell area covered by building footprints (LoD2)",
            valid_range=(-0.01, 1.01),
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
            name="canopy_cover_fraction",
            family="morphology",
            description=(
                "Fraction of 1 m vegetation-height pixels strictly above "
                f"{threshold} m in the 10 m cell. Threshold is provisional."
            ),
            valid_range=(-0.01, 1.01),
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


def _era5_channels_v6() -> list[FeatureChannel]:
    """v6 ERA5: t2m, ssrd, wind speed, and one 0–72 h precipitation sum."""
    specs = (
        (
            "t2m_scene",
            "K",
            (200.0, 350.0),
            "ERA5-Land 2m air temperature at acquisition",
        ),
        (
            "ssrd_scene",
            "W/m²",
            (-1.0, 1500.0),
            "ERA5-Land surface solar radiation downwards at acquisition",
        ),
        (
            "wind_speed_10m_scene",
            "m/s",
            (0.0, 50.0),
            "ERA5-Land 10m wind speed at acquisition",
        ),
        (
            "tp_0_72h",
            "mm",
            (0.0, 1500.0),
            "ERA5-Land total precipitation sum over the 72 hours before acquisition",
        ),
    )
    return [
        FeatureChannel(
            name=name,
            family="era5",
            description=description,
            unit=unit,
            valid_range=valid_range,
        )
        for name, unit, valid_range, description in specs
    ]


@dataclass(frozen=True)
class FeatureSchema:
    """One versioned feature roster. Channel count is ``len(channels)``."""

    version: int
    release: str
    channels: tuple[FeatureChannel, ...]
    canopy_cover_threshold_m: float | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def names_in_family(self, family: str) -> tuple[str, ...]:
        """Channel names of one family, in roster order."""
        return tuple(channel.name for channel in self.channels if channel.family == family)


def _v6_schema(canopy_cover_threshold_m: float) -> FeatureSchema:
    channels = (
        *_s2_channels(),
        *_index_channels(),
        *_morphology_channels_v6(canopy_cover_threshold_m),
        *_era5_channels_v6(),
        *_shadow_channels(),
    )
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION_V6,
        release="v6",
        channels=channels,
        canopy_cover_threshold_m=canopy_cover_threshold_m,
    )


# Fixed v5 order — the published model input interface. Never reorder
# without bumping FEATURE_SCHEMA_VERSION and publishing under a new URI.
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

# Published features/v5 width. The literal locks that roster; v6 code uses
# ``schema.n_channels`` instead of this number.
_PUBLISHED_V5_CHANNEL_COUNT = 28
if len(FEATURE_CHANNELS) != _PUBLISHED_V5_CHANNEL_COUNT:  # pragma: no cover
    raise AssertionError(
        f"features/v5 roster must have {_PUBLISHED_V5_CHANNEL_COUNT} channels, "
        f"got {len(FEATURE_CHANNELS)}"
    )

_V5_SCHEMA = FeatureSchema(
    version=FEATURE_SCHEMA_VERSION,
    release="v5",
    channels=FEATURE_CHANNELS,
    canopy_cover_threshold_m=None,
)


def feature_schema(
    version: int = FEATURE_SCHEMA_VERSION,
    *,
    canopy_cover_threshold_m: float | None = None,
) -> FeatureSchema:
    """Return the feature roster for a schema version.

    Version 4 is the 28-channel roster. Version 5 is the features/v6
    channel roster. The v6 canopy
    threshold defaults to :data:`CANOPY_COVER_THRESHOLD_M_DEFAULT`.
    """
    if version == FEATURE_SCHEMA_VERSION:
        if canopy_cover_threshold_m is not None:
            raise ValueError("canopy_cover_threshold_m applies to feature schema v6 only")
        return _V5_SCHEMA
    if version == FEATURE_SCHEMA_VERSION_V6:
        threshold = (
            CANOPY_COVER_THRESHOLD_M_DEFAULT
            if canopy_cover_threshold_m is None
            else float(canopy_cover_threshold_m)
        )
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"canopy_cover_threshold_m must be finite and >= 0, got {threshold}")
        return _v6_schema(threshold)
    raise ValueError(f"unknown feature schema version {version}")


# Published stacks before the data-correctness bump. Version 2 is the retired
# features/v2 product. Version 3 is features/v3 through the published
# features/v5 product. Both recorded the same 28-channel order as
# FEATURE_CHANNELS. feature_schema() does not accept them: a new stack is
# written at version 4 or 5 only.
_PUBLISHED_28_CHANNEL_SCHEMA_VERSIONS = frozenset({2, 3})


def schema_from_provenance(provenance: Mapping[str, Any]) -> FeatureSchema:
    """Resolve the roster recorded on a feature-stack provenance document.

    Schema versions 2 and 3 are the published 28-channel order. Version 4 is
    that same roster after the data-correctness bump. Version 5 is features/v6.
    """
    version = int(provenance.get("schema_version", FEATURE_SCHEMA_VERSION))
    raw_threshold = provenance.get("canopy_cover_threshold_m")
    threshold = None if raw_threshold is None else float(raw_threshold)
    if version == FEATURE_SCHEMA_VERSION or version in _PUBLISHED_28_CHANNEL_SCHEMA_VERSIONS:
        return FeatureSchema(
            version=version,
            release="v5",
            channels=FEATURE_CHANNELS,
            canopy_cover_threshold_m=None,
        )
    return feature_schema(version, canopy_cover_threshold_m=threshold)


def schema_matching_count(count: int) -> FeatureSchema:
    """Return the unique roster whose width equals ``count``.

    Used when a reader sees a stack width and no provenance. v5 and v6
    widths differ, so the match is unambiguous.
    """
    matches = [
        feature_schema(version)
        for version in (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V6)
        if feature_schema(version).n_channels == count
    ]
    if len(matches) != 1:
        known = [
            (feature_schema(version).release, feature_schema(version).n_channels)
            for version in (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V6)
        ]
        raise ValueError(f"band count {count} does not match one feature schema; known {known}")
    return matches[0]


def resolve_era5_representation(value: str | None) -> str:
    """Return a v6 ERA5 representation. ``None`` selects the v6 default."""
    if value is None or value == "":
        return ERA5_REPRESENTATION_V6_DEFAULT
    if value not in ERA5_REPRESENTATIONS:
        raise ValueError(
            f"era5_representation must be one of {ERA5_REPRESENTATIONS}, got {value!r}"
        )
    return value


def era5_representation_for_schema(schema: FeatureSchema, configured: str | None) -> str | None:
    """Representation applied when composing a feature stack.

    Schema v5 keeps the published ERA5 field (``None``). Schema v6
    defaults to ``scene_value`` when ``configured`` is omitted.
    """
    if (
        schema.version == FEATURE_SCHEMA_VERSION
        or schema.version in _PUBLISHED_28_CHANNEL_SCHEMA_VERSIONS
    ):
        if configured not in (None, ERA5_REPRESENTATION_BILINEAR):
            raise ValueError(
                "features/v5 keeps the published bilinear ERA5 field; "
                f"got era5_representation={configured!r}"
            )
        return None
    return resolve_era5_representation(configured)


__all__ = [
    "ALBEDO_WEIGHTS",
    "CANOPY_COVER_THRESHOLD_M_DEFAULT",
    "ERA5_REPRESENTATION_BILINEAR",
    "ERA5_REPRESENTATION_SCENE_VALUE",
    "ERA5_REPRESENTATION_V6_DEFAULT",
    "ERA5_REPRESENTATIONS",
    "FEATURE_CHANNELS",
    "FEATURE_CHANNEL_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION_V6",
    "FeatureChannel",
    "FeatureSchema",
    "era5_representation_for_schema",
    "feature_schema",
    "resolve_era5_representation",
    "schema_from_provenance",
    "schema_matching_count",
]
