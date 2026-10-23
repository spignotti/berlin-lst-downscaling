"""Read-only emissivity diagnostic physics for Landsat Collection 2 ST.

This module holds the pure (no-IO) mathematics behind the question: how
much does the Landsat Collection 2 Level-2 surface temperature target
(``ST_B10``) depend on the scene-NDVI emissivity adjustment?

Background
----------
USGS Collection 2 surface temperature uses the single-channel algorithm
(RIT / NASA JPL): top-of-atmosphere thermal radiance (``TRAD``) is
corrected with upwelled (``URAD``) and downwelled (``DRAD``) path
radiance and atmospheric transmittance (``ATRAN``), then inverted
through Planck's law with the TIRS Band 10 constants (``K1``/``K2``).
The emissivity input (``EMIS``) starts from the ASTER Global Emissivity
Database v3 (ASTER GED, 2000–2008 mean; Hulley et al. 2015) and is
adjusted per scene with the Landsat TOA NDVI (Malakar et al. 2018;
USGS LSDS-1619).

Forward model (radiative transfer equation solved for ``B``, the
surface-leaving radiance)::

    B(T) = ((TRAD - URAD) / ATRAN - (1 - eps) * DRAD) / eps

followed by the Planck inversion::

    LST = K2 / ln(K1 / B(T) + 1)

All scaling constants below are read from the Planetary Computer
``landsat-c2-l2`` STAC ``raster:bands`` metadata (verified 2026-10-23);
the thermal constants default to the per-scene MTL values
(``K1_CONSTANT_BAND_10`` / ``K2_CONSTANT_BAND_10``) with the documented
USGS constants as fallback.
"""

from __future__ import annotations

import numpy as np

# ── Collection 2 Level-2 scaling (STAC raster:bands / LSDS-1619 Table 6-1) ──

ST_SCALE = 0.00341802  # ST_B10 DN -> Kelvin scale
ST_OFFSET = 149.0  # K
THERMAL_RADIANCE_SCALE = 0.001  # TRAD/URAD/DRAD DN -> W/(m^2.sr.um)
ATRAN_SCALE = 0.0001  # ATRAN DN -> unitless transmittance
EMIS_SCALE = 0.0001  # EMIS/EMSD DN -> unitless emissivity
AUX_NODATA_DN = -9999  # int16 nodata code for TRAD/URAD/DRAD/ATRAN/EMIS/EMSD
ST_FILL_DN = 0  # uint16 fill code for ST_B10

# Optical surface-reflectance scaling (STAC raster:bands, all OLI bands).
OLI_SR_SCALE = 2.75e-05
OLI_SR_OFFSET = -0.2

# ── TIRS Band 10 thermal constants (MTL K1/K2_CONSTANT_BAND_10) ──────────
# L9 values verified against scene LC09_L2SP_193024_20240629_02_T1 MTL;
# L8 values are the documented USGS constants (the diagnostic script
# re-reads the per-scene MTL and records any mismatch).

K1_L8_B10 = 774.8853
K2_L8_B10 = 1321.0789
K1_L9_B10 = 799.0284
K2_L9_B10 = 1329.2405

# QA_RADSAT bit for TIRS Band 10 saturation (LSDS-1619 Table 6-4: bits
# 0-9 are OLI bands 1-9 — value 512(flags OLI band 9) — so bits 10 and
# 11 are TIRS bands 10 and 11).
QA_RADSAT_TIRS_B10_BIT = 10

# NDVI reporting bins (diagnostic presentation only, not a product
# threshold): water/negative, bare/urban-low, sparse, moderate, dense.
NDVI_BIN_EDGES: tuple[float, ...] = (-1.0, 0.0, 0.2, 0.4, 0.6, 1.0)
NDVI_BIN_LABELS: tuple[str, ...] = (
    "water/negative",
    "bare/urban-low",
    "sparse",
    "moderate",
    "dense",
)


# ── scaling ──────────────────────────────────────────────────────────────


def st_dn_to_kelvin(dn: np.ndarray) -> np.ndarray:
    """Convert ST_B10 digital numbers to Kelvin (fill DN 0 -> NaN)."""
    out = dn.astype(np.float64) * ST_SCALE + ST_OFFSET
    out[dn == ST_FILL_DN] = np.nan
    return out


def aux_dn_to_units(dn: np.ndarray, scale: float) -> np.ndarray:
    """Convert an int16 auxiliary DN array to physical units (nodata -> NaN)."""
    out = dn.astype(np.float64) * scale
    out[dn == AUX_NODATA_DN] = np.nan
    return out


def olisr_dn_to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert OLI L2 surface-reflectance DN to unitless reflectance."""
    return dn.astype(np.float64) * OLI_SR_SCALE + OLI_SR_OFFSET


def thermal_constants_for_platform(platform: str) -> tuple[float, float]:
    """Return (K1, K2) for TIRS Band 10 given ``'landsat-8'``/``'landsat-9'``."""
    if platform == "landsat-8":
        return (K1_L8_B10, K2_L8_B10)
    if platform == "landsat-9":
        return (K1_L9_B10, K2_L9_B10)
    raise ValueError(f"unknown platform {platform!r}")


# ── Planck / single-channel inversion ────────────────────────────────────


def planck_invert_lst(radiance: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Invert Planck's law: spectral radiance -> brightness temperature (K).

    Non-positive radiances are unphysical as a logarithm argument and map
    to NaN (counted by the caller, never silently clamped).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = k2 / np.log(k1 / radiance + 1.0)
    out[~(radiance > 0)] = np.nan
    return out


def planck_forward_radiance(temperature_k: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Forward Planck's law: temperature (K) -> spectral radiance."""
    with np.errstate(over="ignore", invalid="ignore"):
        return k1 / (np.exp(k2 / temperature_k) - 1.0)


def surface_leaving_radiance(
    trad: np.ndarray,
    urad: np.ndarray,
    drad: np.ndarray,
    atran: np.ndarray,
    emis: np.ndarray,
) -> np.ndarray:
    """Solve the thermal RTE for surface-leaving radiance ``B(T)``.

    All inputs are in physical units (radiances in W/(m^2.sr.um),
    ``atran``/``emis`` unitless). Pixels with non-positive transmittance
    or emissivity outside (0, 1] map to NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ((trad - urad) / atran - (1.0 - emis) * drad) / emis
    valid = (atran > 0) & (emis > 0) & (emis <= 1.0)
    out[~valid] = np.nan
    return out


def lst_from_aux(
    trad: np.ndarray,
    urad: np.ndarray,
    drad: np.ndarray,
    atran: np.ndarray,
    emis: np.ndarray,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Full single-channel retrieval: auxiliary fields + emissivity -> LST (K)."""
    return planck_invert_lst(surface_leaving_radiance(trad, urad, drad, atran, emis), k1, k2)


def forward_trad(
    lst_k: np.ndarray,
    urad: np.ndarray,
    drad: np.ndarray,
    atran: np.ndarray,
    emis: np.ndarray,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Forward model used by the synthetic tests: LST -> TOA radiance."""
    return atran * (emis * planck_forward_radiance(lst_k, k1, k2) + (1.0 - emis) * drad) + urad


# ── NDVI / masks ─────────────────────────────────────────────────────────


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized difference vegetation index; zero denominator -> NaN."""
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (nir - red) / denom
    out[denom == 0] = np.nan
    return out


def tirs_b10_saturated(qa_radsat: np.ndarray) -> np.ndarray:
    """Boolean mask of TIRS Band 10 radiometrically saturated pixels."""
    return (qa_radsat.astype(np.uint16) >> QA_RADSAT_TIRS_B10_BIT) & 1 == 1


def assign_ndvi_bins(values: np.ndarray) -> np.ndarray:
    """Bin NDVI values to ``NDVI_BIN_LABELS`` indices; out-of-range -> -1."""
    return np.digitize(values, NDVI_BIN_EDGES[1:-1], right=False)


# ── static emissivity proxies ────────────────────────────────────────────


def uniform_static_emissivity(emis: np.ndarray, mask: np.ndarray) -> float:
    """Scene-median emissivity over the analysis mask (documented fallback).

    This is NOT a true static ASTER GED product: it removes the spatial
    pattern of the scene-NDVI adjustment while keeping the scene mean,
    so ``LST(a) - LST(b)`` isolates the spatial-adjustment effect. Used
    only because ASTER GED granules require Earthdata credentials that
    are unavailable to this read-only diagnostic (see the report).
    """
    selected = emis[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("uniform_static_emissivity: empty analysis mask")
    return float(np.median(selected))


def aster_ged_b10_proxy(ged_band13: np.ndarray, ged_band14: np.ndarray) -> np.ndarray:
    """Static TIRS-B10 emissivity proxy from two ASTER GED TIR bands.

    TIRS Band 10 (10.6–11.19 um) sits inside the joint passband of ASTER
    TIR bands 13 (10.25–10.95 um) and 14 (10.95–11.65 um); the USGS C2 ST
    code regresses the ASTER GED TIR bands onto the TIRS bandpass
    (Malakar et al. 2018; Hulley et al. 2015). This helper applies the
    documented first-order form — the band mean — to caller-supplied
    100 m ASTER GED rasters. Scale handling (GED tiles store
    ``emis * 1000`` as uint16) is the caller's responsibility and must
    be stated wherever this is used.
    """
    return 0.5 * (ged_band13.astype(np.float64) + ged_band14.astype(np.float64))


# ── summary statistics ───────────────────────────────────────────────────


def binned_stats(values: np.ndarray, bin_index: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    """Mean / p05 / p95 / count of finite ``values`` per bin index."""
    stats: list[dict[str, float]] = []
    for b in range(n_bins):
        sel = values[(bin_index == b) & np.isfinite(values)]
        if sel.size == 0:
            stats.append({"mean": float("nan"), "p05": float("nan"), "p95": float("nan"), "n": 0})
        else:
            stats.append(
                {
                    "mean": float(np.mean(sel)),
                    "p05": float(np.percentile(sel, 5)),
                    "p95": float(np.percentile(sel, 95)),
                    "n": int(sel.size),
                }
            )
    return stats


def ndvi_lst_relation(ndvi_values: np.ndarray, lst_values: np.ndarray) -> dict[str, float]:
    """Least-squares slope (K per unit NDVI) and Pearson r of LST on NDVI."""
    both = np.isfinite(ndvi_values) & np.isfinite(lst_values)
    x = ndvi_values[both]
    y = lst_values[both]
    n = int(x.size)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "n": n}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "r": r, "n": n}


__all__ = [
    "AUX_NODATA_DN",
    "ATRAN_SCALE",
    "EMIS_SCALE",
    "K1_L8_B10",
    "K1_L9_B10",
    "K2_L8_B10",
    "K2_L9_B10",
    "NDVI_BIN_EDGES",
    "NDVI_BIN_LABELS",
    "OLI_SR_OFFSET",
    "OLI_SR_SCALE",
    "QA_RADSAT_TIRS_B10_BIT",
    "ST_FILL_DN",
    "ST_OFFSET",
    "ST_SCALE",
    "THERMAL_RADIANCE_SCALE",
    "assign_ndvi_bins",
    "aster_ged_b10_proxy",
    "aux_dn_to_units",
    "binned_stats",
    "forward_trad",
    "lst_from_aux",
    "ndvi",
    "ndvi_lst_relation",
    "olisr_dn_to_reflectance",
    "planck_forward_radiance",
    "planck_invert_lst",
    "st_dn_to_kelvin",
    "surface_leaving_radiance",
    "thermal_constants_for_platform",
    "tirs_b10_saturated",
    "uniform_static_emissivity",
]
