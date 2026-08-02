"""Channel-to-bin registry for WB2c-1 profiling.

Defines fixed, physically meaningful histogram bins for every band
in the pipeline. Bands not mapped here will cause a hard failure
rather than receiving generic bins.
"""

from __future__ import annotations

from berlin_lst_downscaling.data.profiling.models import HistogramSpec

# ── ERA5 8-band contract ──────────────────────────────────────────────

ERA5_HISTOGRAMS: dict[str, HistogramSpec] = {
    "t2m_scene": HistogramSpec(
        band_name="t2m_scene",
        bin_edges=tuple(range(260, 330, 5)),
        description="2m temperature (K)",
    ),
    "ssrd_scene": HistogramSpec(
        band_name="ssrd_scene",
        bin_edges=tuple(range(0, 1200, 100)),
        description="Solar radiation (W/m²)",
    ),
    "ssrd_antecedent_72h_mean": HistogramSpec(
        band_name="ssrd_antecedent_72h_mean",
        bin_edges=tuple(range(0, 1200, 100)),
        description="72h mean solar radiation (W/m²)",
    ),
    "vpd_scene": HistogramSpec(
        band_name="vpd_scene",
        bin_edges=tuple(x / 10 for x in range(0, 11)),
        description="Vapour pressure deficit (kPa)",
    ),
    "wind_speed_10m_scene": HistogramSpec(
        band_name="wind_speed_10m_scene",
        bin_edges=tuple(range(0, 30, 2)),
        description="Wind speed (m/s)",
    ),
    "tp_0_24h": HistogramSpec(
        band_name="tp_0_24h",
        bin_edges=tuple(range(0, 50, 5)),
        description="Precipitation 0-24h (mm)",
    ),
    "tp_24_48h": HistogramSpec(
        band_name="tp_24_48h",
        bin_edges=tuple(range(0, 50, 5)),
        description="Precipitation 24-48h (mm)",
    ),
    "tp_48_72h": HistogramSpec(
        band_name="tp_48_72h",
        bin_edges=tuple(range(0, 50, 5)),
        description="Precipitation 48-72h (mm)",
    ),
}

# ── Shadow bands ──────────────────────────────────────────────────────

_SHADOW_HISTOGRAMS: dict[str, HistogramSpec] = {
    "shadow_building": HistogramSpec(
        band_name="shadow_building",
        bin_edges=(0.0, 1.0, 255.0),
        description="Building shadow mask (0=lit, 1=shadowed, 255=nodata)",
    ),
    "shadow_vegetation": HistogramSpec(
        band_name="shadow_vegetation",
        bin_edges=(0.0, 1.0, 255.0),
        description="Vegetation shadow mask (0=lit, 1=shadowed, 255=nodata)",
    ),
}

# ── ARD bands (Landsat/Sentinel-2/ECOSTRESS) ──────────────────────────

_ARD_HISTOGRAMS: dict[str, HistogramSpec] = {
    "st": HistogramSpec(
        band_name="st",
        bin_edges=tuple(range(260, 340, 5)),
        description="Landsat surface temperature (K)",
    ),
    "lst": HistogramSpec(
        band_name="lst",
        bin_edges=tuple(range(260, 340, 5)),
        description="ECOSTRESS LST (K)",
    ),
    "B02": HistogramSpec(
        band_name="B02",
        bin_edges=tuple(x / 100 for x in range(0, 110, 10)),
        description="Sentinel-2 blue reflectance [0, 1]",
    ),
    "B03": HistogramSpec(
        band_name="B03",
        bin_edges=tuple(x / 100 for x in range(0, 110, 10)),
        description="Sentinel-2 green reflectance [0, 1]",
    ),
    "B04": HistogramSpec(
        band_name="B04",
        bin_edges=tuple(x / 100 for x in range(0, 110, 10)),
        description="Sentinel-2 red reflectance [0, 1]",
    ),
    "B08": HistogramSpec(
        band_name="B08",
        bin_edges=tuple(x / 100 for x in range(0, 110, 10)),
        description="Sentinel-2 NIR reflectance [0, 1]",
    ),
}

# ── Static product bands ──────────────────────────────────────────────

_STATIC_HISTOGRAMS: dict[str, HistogramSpec] = {
    # Terrain/DSM products (meters)
    "building_dsm": HistogramSpec(
        band_name="building_dsm",
        bin_edges=tuple(range(-10, 200, 10)),
        description="Building DSM height (m)",
    ),
    "vegetation_dsm": HistogramSpec(
        band_name="vegetation_dsm",
        bin_edges=tuple(range(-10, 200, 10)),
        description="Vegetation DSM height (m)",
    ),
    "combined_dsm": HistogramSpec(
        band_name="combined_dsm",
        bin_edges=tuple(range(-10, 200, 10)),
        description="Combined DSM height (m)",
    ),
    "terrain_height": HistogramSpec(
        band_name="terrain_height",
        bin_edges=tuple(range(-10, 200, 10)),
        description="Terrain height (m)",
    ),
    "vegetation_height_mean": HistogramSpec(
        band_name="vegetation_height_mean",
        bin_edges=tuple(range(0, 50, 5)),
        description="Mean vegetation height (m)",
    ),
    "vegetation_height_max": HistogramSpec(
        band_name="vegetation_height_max",
        bin_edges=tuple(range(0, 50, 5)),
        description="Max vegetation height (m)",
    ),
    # Morphology products
    "building_height_mean": HistogramSpec(
        band_name="building_height_mean",
        bin_edges=tuple(range(0, 100, 10)),
        description="Mean building height (m)",
    ),
    "building_height_std": HistogramSpec(
        band_name="building_height_std",
        bin_edges=tuple(range(0, 50, 5)),
        description="Building height std dev (m)",
    ),
    "building_coverage_ratio": HistogramSpec(
        band_name="building_coverage_ratio",
        bin_edges=tuple(x / 10 for x in range(0, 11)),
        description="Building coverage ratio [0, 1]",
    ),
    "building_height_max": HistogramSpec(
        band_name="building_height_max",
        bin_edges=tuple(range(0, 100, 10)),
        description="Max building height (m)",
    ),
    # SVF
    "svf": HistogramSpec(
        band_name="svf",
        bin_edges=tuple(x / 10 for x in range(0, 11)),
        description="Sky view factor [0, 1]",
    ),
    # Imperviousness
    "imperviousness": HistogramSpec(
        band_name="imperviousness",
        bin_edges=tuple(range(0, 110, 10)),
        description="Imperviousness percentage [0, 100]",
    ),
}

# Horizon bands (36 azimuths) - reuse for both building and vegetation
_HORIZON_BANDS = [f"az_{int(az):03d}" for az in range(0, 360, 10)]
HORIZON_HISTOGRAMS: dict[str, HistogramSpec] = {
    band: HistogramSpec(
        band_name=band,
        bin_edges=tuple(range(0, 9100, 1000)),
        description=f"Horizon angle at {int(az)}° azimuth (centidegrees)",
    )
    for band, az in zip(_HORIZON_BANDS, range(0, 360, 10), strict=True)
}

# ── Master registry ───────────────────────────────────────────────────

ALL_HISTOGRAMS: dict[str, HistogramSpec] = {
    **ERA5_HISTOGRAMS,
    **_SHADOW_HISTOGRAMS,
    **_ARD_HISTOGRAMS,
    **_STATIC_HISTOGRAMS,
    **HORIZON_HISTOGRAMS,
}


def get_histogram_spec(band_name: str) -> HistogramSpec | None:
    """Return the HistogramSpec for a band, or None if unmapped."""
    return ALL_HISTOGRAMS.get(band_name)


def require_histogram_spec(band_name: str) -> HistogramSpec:
    """Return the HistogramSpec for a band, raising if unmapped."""
    spec = ALL_HISTOGRAMS.get(band_name)
    if spec is None:
        raise KeyError(
            f"No histogram spec for band {band_name!r}. "
            f"Available bands: {sorted(ALL_HISTOGRAMS.keys())}"
        )
    return spec


__all__ = [
    "ERA5_HISTOGRAMS",
    "_SHADOW_HISTOGRAMS",
    "_ARD_HISTOGRAMS",
    "_STATIC_HISTOGRAMS",
    "HORIZON_HISTOGRAMS",
    "ALL_HISTOGRAMS",
    "get_histogram_spec",
    "require_histogram_spec",
]
