"""Stage-1 raw-input QA contract — roster and physical value bounds.

The contract declares every published raw input the Stage-1 gate
validates, which of them participate in the pixel-level joint-support
scan, and the fixed physical bounds applied per channel.

Scope decisions (mirrored from the Stage-1 raw QA brief and the
planning session):

- **In-support layers** (pixel validity drives the 100 m support
  statistic): Landsat target (block-constant at 100 m), S2 spectral
  bands + flag, static derived morphology (DSMs + SVF for the scene's
  geometry vintage), ERA5-Land, and both shadow products.
- **Metadata-only layers**: static source products (imperviousness,
  vegetation height, terrain height, LoD2 morphology) and the
  building/vegetation horizons. They are upstream of the in-support
  layers — the horizon products feed the shadow computation, the
  source products feed the derived products — so their validity is
  already reflected by the in-support products. They get structural
  COG/grid checks only.
- **Excluded from Stage 1**: ECOSTRESS (validation-only role, never a
  training input) and the optional coarse-LST input (no raw product
  exists yet; it is created by a later stage).
"""

from __future__ import annotations

from dataclasses import dataclass

# ── fixed physical bounds (Kelvin / reflectance / derived) ────────────

# Landsat Collection-2 Surface Temperature, Kelvin. Broad physical band
# that preserves genuine thermal extremes (no statistical outlier rule).
LST_RANGE_K: tuple[float, float] = (150.0, 400.0)

# S2 scaled reflectance (ARD clips to [0, 1] at masking time).
S2_REFLECTANCE_RANGE: tuple[float, float] = (0.0, 1.0)

# Static derived morphology (mirrors ``data/secondary/dsm.py`` and
# ``data/secondary/svf.py`` BandSpec ranges).
DSM_RANGE_M: tuple[float, float] = (-10.0, 600.0)
SVF_RANGE: tuple[float, float] = (-0.01, 1.01)

# Shadow products: uint8, 0=lit, 1=shadowed, 255=nodata.
SHADOW_VALID_VALUES: frozenset[int] = frozenset({0, 1})
SHADOW_NODATA: int = 255

# ── layer registry ────────────────────────────────────────────────────

# Static derived morphology products that enter the pixel-level support
# scan. Single source of truth — consumed by the inventory resolver and
# the scan core so the roster cannot drift.
STATIC_DERIVED_MORPHOLOGY_PRODUCTS: tuple[str, ...] = (
    "building_dsm",
    "combined_dsm",
    "svf",
)
STATIC_DERIVED_OPTIONAL_PRODUCTS: tuple[str, ...] = ("vegetation_dsm",)


@dataclass(frozen=True)
class RawLayer:
    """One declared raw input of the Stage-1 gate."""

    key: str  # stable identifier used in reports
    family: str  # target | spectral | static_derived |
    # static_source | era5 | shadow
    resolution_m: int  # native resolution on the canonical grid
    in_support: bool  # participates in the pixel-level joint-support scan
    metadata_only: bool = False  # structural COG/grid check only, no pixel scan
    n_bands: int = 1
    dtype: str = "float32"
    description: str = ""


def _s2_bands() -> list[RawLayer]:
    return [
        RawLayer(
            key=f"s2_{b}",
            family="spectral",
            resolution_m=10,
            in_support=True,
            n_bands=1,
            dtype="float32",
            description=f"Sentinel-2 {b} scaled reflectance [0,1]",
        )
        for b in ("B02", "B03", "B04", "B08")
    ]


def _era5_bands() -> list[RawLayer]:
    names = (
        "t2m_scene",
        "ssrd_scene",
        "ssrd_antecedent_72h_mean",
        "vpd_scene",
        "wind_speed_10m_scene",
        "tp_0_24h",
        "tp_24_48h",
        "tp_48_72h",
    )
    return [
        RawLayer(
            key=f"era5_{n}",
            family="era5",
            resolution_m=10,
            in_support=True,
            n_bands=1,
            dtype="float32",
            description=f"ERA5-Land {n}",
        )
        for n in names
    ]


def _static_source_layers() -> list[RawLayer]:
    # Historical LoD2 morphology vintages enter the per-vintage roster at
    # inventory time; the canonical (source, vintage) set is declared here.
    return [
        RawLayer(
            key="static_terrain_height",
            family="static_source",
            resolution_m=10,
            in_support=False,
            metadata_only=True,
            dtype="float32",
            description="terrain_height (DGM 1 m, 2021)",
        ),
        RawLayer(
            key="static_vegetation_height",
            family="static_source",
            resolution_m=10,
            in_support=False,
            metadata_only=True,
            dtype="float32",
            description="vegetation_height (2020)",
        ),
        RawLayer(
            key="static_lod2_morphology",
            family="static_source",
            resolution_m=10,
            in_support=False,
            metadata_only=True,
            dtype="float32",
            description="lod2_morphology (LoD2 CityGML morphometry)",
        ),
        RawLayer(
            key="static_imperviousness",
            family="static_source",
            resolution_m=10,
            in_support=False,
            metadata_only=True,
            dtype="float32",
            description="imperviousness (Versiegelung, 2016/2021)",
        ),
    ]


@dataclass(frozen=True)
class RawInputContract:
    """Declared raw-input roster for the Stage-1 gate.

    ``static_derived_keys`` holds the morphology products that enter the
    pixel-level support scan; the exact per-vintage subset is resolved
    against the static derived ledger at inventory time.
    """

    static_derived_keys: tuple[str, ...] = STATIC_DERIVED_MORPHOLOGY_PRODUCTS
    static_derived_optional_keys: tuple[str, ...] = STATIC_DERIVED_OPTIONAL_PRODUCTS

    def in_support_layers(self) -> list[RawLayer]:
        layers = [
            RawLayer(
                key="landsat_st",
                family="target",
                resolution_m=100,
                in_support=True,
                dtype="float32",
                description="Landsat C2 Surface Temperature [K]",
            ),
            RawLayer(
                key="landsat_flag",
                family="target",
                resolution_m=100,
                in_support=False,
                metadata_only=True,
                dtype="uint8",
                description="Landsat ARD flag band",
            ),
        ]
        layers += _s2_bands()
        layers.append(
            RawLayer(
                key="s2_flag",
                family="spectral",
                resolution_m=10,
                in_support=True,
                dtype="uint8",
                description="Sentinel-2 ARD flag band",
            )
        )
        for key in (*self.static_derived_keys, *self.static_derived_optional_keys):
            layers.append(
                RawLayer(
                    key=key,
                    family="static_derived",
                    resolution_m=10,
                    in_support=True,
                    dtype="float32",
                    description=f"Static derived morphology {key}",
                )
            )
        layers += _era5_bands()
        layers += [
            RawLayer(
                key="shadow_building",
                family="shadow",
                resolution_m=10,
                in_support=True,
                dtype="uint8",
                description="Building shadow (0 lit, 1 shadowed, 255 nodata)",
            ),
            RawLayer(
                key="shadow_vegetation",
                family="shadow",
                resolution_m=10,
                in_support=True,
                dtype="uint8",
                description="Vegetation shadow (0 lit, 1 shadowed, 255 nodata)",
            ),
        ]
        return layers

    def metadata_only_layers(self) -> list[RawLayer]:
        return [*_static_source_layers()]


def raw_input_contract() -> RawInputContract:
    """Return the sole public Stage-1 raw-input contract instance."""
    return RawInputContract()


__all__ = [
    "DSM_RANGE_M",
    "LST_RANGE_K",
    "RawInputContract",
    "RawLayer",
    "S2_REFLECTANCE_RANGE",
    "SHADOW_NODATA",
    "SHADOW_VALID_VALUES",
    "STATIC_DERIVED_MORPHOLOGY_PRODUCTS",
    "STATIC_DERIVED_OPTIONAL_PRODUCTS",
    "SVF_RANGE",
    "raw_input_contract",
]
