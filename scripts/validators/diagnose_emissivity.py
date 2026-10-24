#!/usr/bin/env python3
"""Read-only emissivity diagnostic for the Landsat C2 L2 ST target.

Quantifies how much the surface temperature target (``ST_B10``) depends
on the scene-NDVI emissivity adjustment, per pixel::

    B(T) = ((TRAD - URAD) / ATRAN - (1 - eps) * DRAD) / eps

with scale/offset from the STAC ``raster:bands`` metadata and Planck
inversion with the TIRS Band 10 constants (per-scene MTL ``K1``/``K2``,
documented USGS constants as fallback).

Experiments per scene
---------------------
(a) ``eps`` = product ``EMIS``: sanity check, must reproduce ``ST_B10``.
(b) UNIFORM STATIC: the scene-median ``EMIS`` everywhere. This removes ALL
    spatial emissivity structure — not only the NDVI adjustment — so ΔLST
    and slope changes against it are an UPPER BOUND on the spatial-pattern
    effect, not the NDVI-adjustment effect alone. (True static ASTER GED
    needs Earthdata credentials; ``--aster-ged-tif`` is the hook for the
    real comparison: a caller-supplied B10-adjusted raster, reprojected
    onto each 30 m scene grid with bilinear resampling.)
(c) fixed-eps scenarios (0.97, 0.98).

Scene selection rule (pinned IDs below for reproducibility): per season
(JJA summer / DJF winter) x platform (Landsat 8 / 9) group, the lowest
``eo:cloud_cover`` scenes over the Berlin bbox in the Planetary Computer
``landsat-c2-l2`` catalog (2021-2025 window), preferring WRS-2 193/024
(full AOI coverage); the winter pool is thin, so adjacent paths are
accepted. The GCS v4 selection-manifest bundle is not readable from
this box (no GCS credentials); the PC catalog over the same AOI is the
scene universe instead. Every pinned scene falls inside the 2017-2026
selection window. No training, no GCS writes, no cloud jobs: only
Planetary Computer reads plus local writes under ``--out-dir``.

Examples
--------
    uv run python scripts/validators/diagnose_emissivity.py --synthetic
    uv run python scripts/validators/diagnose_emissivity.py --with-s2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from berlin_lst_downscaling.analysis.emissivity import (
    ATRAN_SCALE,
    EMIS_SCALE,
    IMPERV_BIN_LABELS,
    NDVI_BIN_LABELS,
    THERMAL_RADIANCE_SCALE,
    assign_imperv_bins,
    assign_ndvi_bins,
    aster_ged_b10_proxy,
    aux_dn_to_units,
    binned_stats,
    forward_trad,
    ged_b10_on_scene_grid,
    imperv_vintage_for_year,
    lst_from_aux,
    ndvi,
    ndvi_lst_relation,
    olisr_dn_to_reflectance,
    pool_overall_stats,
    qa_water,
    st_dn_to_kelvin,
    thermal_constants_for_platform,
    thermal_saturated,
    uniform_static_emissivity,
)
from berlin_lst_downscaling.data.acquisition.pc_client import pc_search, stac_load
from berlin_lst_downscaling.data.ard.masking import landsat_qa_to_clear_bits

BERLIN_BBOX = (13.08, 52.34, 13.76, 52.68)
LOAD_CRS = "EPSG:32633"  # Landsat UTM 33N native for Berlin
LOAD_RES_M = 30

THERMAL_BANDS = (
    "lwir11",
    "qa_pixel",
    "qa_radsat",
    "qa",  # ST_QA: product uncertainty (K), context for the sanity residual
    "emis",
    "emsd",
    "trad",
    "urad",
    "drad",
    "atran",
    "red",
    "nir08",
)

# Scale of the ST_QA product-uncertainty band (LSDS-1619 Table 6-1).
ST_QA_SCALE = 0.01

# Pinned representative scenes: (scene_id, season, platform).
SCENES: tuple[tuple[str, str, str], ...] = (
    ("LC08_L2SP_193024_20240824_02_T1", "summer", "landsat-8"),
    ("LC08_L2SP_193024_20230603_02_T1", "summer", "landsat-8"),
    ("LC08_L2SP_193024_20230907_02_T1", "summer", "landsat-8"),
    ("LC09_L2SP_193023_20220624_02_T1", "summer", "landsat-9"),
    ("LC09_L2SP_192024_20220719_02_T1", "summer", "landsat-9"),
    ("LC09_L2SP_193023_20220811_02_T1", "summer", "landsat-9"),
    ("LC08_L2SP_192024_20230119_02_T1", "winter", "landsat-8"),
    ("LC09_L2SP_194023_20230226_02_T1", "winter", "landsat-9"),
    ("LC09_L2SP_194023_20240128_02_T1", "winter", "landsat-9"),
)

FIXED_EPS_SCENARIOS = (0.97, 0.98)

# SCL classes treated as clear for the S2 NDVI join (mirrors the ARD
# flag derivation: only fill/saturated/defect/shadow/cloud/cirrus/snow
# set flag bits; dark (2) and unclassified (7) stay clear).
S2_CLEAR_SCL = (2, 4, 5, 6, 7)


# ── STAC helpers ─────────────────────────────────────────────────────────


def _resolve_landsat_item(scene_id: str):  # type: ignore[no-untyped-def]
    items = pc_search(collections=["landsat-c2-l2"], ids=[scene_id], max_items=1)
    if not items:
        raise RuntimeError(f"STAC item {scene_id!r} not found")
    if items[0].id != scene_id:
        raise RuntimeError(f"STAC returned {items[0].id!r}, expected {scene_id!r}")
    return items[0]


def _mtl_thermal_constants(item) -> tuple[float, float, str]:  # type: ignore[no-untyped-def]
    """Read per-scene MTL K1/K2; return (k1, k2, source).

    Tries ``mtl.json``, then ``mtl.txt`` (ODL regex), then the documented
    USGS sensor constants. The source string records which branch served.
    """
    import re

    platform = "landsat-9" if item.id.startswith("LC09") else "landsat-8"
    fallback = thermal_constants_for_platform(platform)
    try:
        href = item.assets["mtl.json"].href
        with urllib.request.urlopen(href, timeout=60) as resp:  # noqa: S310
            mtl = json.load(resp)
        consts = mtl["LANDSAT_METADATA_FILE"]["LEVEL1_THERMAL_CONSTANTS"]
        return (
            float(consts["K1_CONSTANT_BAND_10"]),
            float(consts["K2_CONSTANT_BAND_10"]),
            "mtl.json",
        )
    except Exception:  # noqa: BLE001, S110 — older scenes may lack the JSON MTL
        pass
    try:
        href = item.assets["mtl.txt"].href
        with urllib.request.urlopen(href, timeout=60) as resp:  # noqa: S310
            txt = resp.read().decode("utf-8", "replace")
        k1 = re.findall(r"K1_CONSTANT_BAND_10\s*=\s*([0-9.]+)", txt)
        k2 = re.findall(r"K2_CONSTANT_BAND_10\s*=\s*([0-9.]+)", txt)
        if k1 and k2:
            return (float(k1[0]), float(k2[0]), "mtl.txt")
    except Exception:  # noqa: BLE001, S110 — fall through to documented constants
        pass
    return (fallback[0], fallback[1], "documented-fallback")


def _load_landsat_window(item, bbox):  # type: ignore[no-untyped-def]
    ds = stac_load(
        items=[item],
        bands=list(THERMAL_BANDS),
        bbox=bbox,
        crs=LOAD_CRS,
        resolution=LOAD_RES_M,
        resampling="nearest",  # explicit: never rely on the odc-stac default
        chunks={"x": 2048, "y": 2048},
    )
    out: dict[str, np.ndarray] = {}
    for band in THERMAL_BANDS:
        out[band] = np.asarray(ds[band].values).squeeze()
    transform = ds["lwir11"].rio.transform()
    return out, transform


def _find_s2_companion(item, bbox, window_days: int):  # type: ignore[no-untyped-def]
    """Least-cloudy Sentinel-2 L2A scene over ``bbox`` near the Landsat date."""
    center = item.datetime
    lo_dt = datetime.fromtimestamp(center.timestamp() - window_days * 86400).strftime("%Y-%m-%d")
    hi_dt = datetime.fromtimestamp(center.timestamp() + window_days * 86400).strftime("%Y-%m-%d")
    items = pc_search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{lo_dt}/{hi_dt}",
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=20,
    )
    if not items:
        return None
    items.sort(
        key=lambda it: (
            it.properties.get("eo:cloud_cover", 100.0),
            abs((it.datetime - center).total_seconds()),
        )
    )
    return items[0]


def _load_s2_ndvi(s2_item, bbox, dst_shape, dst_transform):  # type: ignore[no-untyped-def]
    """Harmonized S2 NDVI (repo ARD reflectance code) reprojected to 30 m."""
    from berlin_lst_downscaling.data.ard.s2_reflectance import (  # lazy: heavy import
        reflectance_scale_from_item,
        scale_s2_digital_numbers,
    )

    ds = stac_load(
        items=[s2_item],
        bands=["B04", "B08", "SCL"],
        bbox=bbox,
        crs=LOAD_CRS,
        resolution=10,
        resampling="nearest",  # explicit; bands resampled below with bilinear
        chunks={"x": 2048, "y": 2048},
    )
    scale = reflectance_scale_from_item(s2_item)
    red = scale_s2_digital_numbers(np.asarray(ds["B04"].values).squeeze(), scale)
    nir = scale_s2_digital_numbers(np.asarray(ds["B08"].values).squeeze(), scale)
    scl = np.round(np.asarray(ds["SCL"].values).squeeze()).astype(np.uint8)
    clear = np.isin(scl, S2_CLEAR_SCL)
    ndvi_10m = ndvi(nir.astype(np.float64), red.astype(np.float64))
    ndvi_10m[~clear] = np.nan
    src_transform = ds["B08"].rio.transform()
    dst = np.full(dst_shape, np.nan, dtype=np.float64)
    src = np.where(np.isfinite(ndvi_10m), ndvi_10m, np.nan)
    src = np.ascontiguousarray(src.astype(np.float32))
    reproject(
        src,
        dst,
        src_transform=src_transform,
        src_crs=LOAD_CRS,
        dst_transform=dst_transform,
        dst_crs=LOAD_CRS,
        resampling=Resampling.average,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
    )
    return dst, s2_item.id


# ── static-GED loader ────────────────────────────────────────────────────


def _unitless(arr: np.ndarray, dtype: object, what: str) -> np.ndarray:
    """Return a unitless emissivity array, applying the GED /1000 convention.

    ASTER GED tiles store ``emis * 1000`` as integers; a float raster is
    taken as unitless emissivity already. The choice is recorded in the
    run metadata (``note``), never guessed silently per pixel.
    """
    out = arr.astype(np.float64)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return out / 1000.0
    print(f"note: {what} dtype {np.dtype(dtype)} taken as unitless emissivity")
    return out


def _load_ged_source(path: str) -> dict:
    """Load ``--aster-ged-tif`` into a CRS-tagged dict for per-scene reprojection.

    Accepts a single-band B10-adjusted static emissivity raster in any CRS
    (reprojected onto each 30 m scene grid with bilinear resampling), or a
    5-band 100 m ASTER GED tile (bands 10-14; B10 proxy from bands 13+14).
    Anything else raises a clear error.
    """
    with rasterio.open(path) as src:
        transform = src.transform
        crs = src.crs
        if crs is None:
            raise ValueError(f"static GED raster {path!r} has no CRS; cannot reproject")
        if src.count == 5:
            b13 = _unitless(src.read(4), src.dtypes[3], "GED band 13")
            b14 = _unitless(src.read(5), src.dtypes[4], "GED band 14")
            arr = aster_ged_b10_proxy(b13, b14)
            note = "5-band ASTER GED tile: mean of bands 13+14 as the TIRS B10 proxy"
        elif src.count == 1:
            arr = _unitless(src.read(1), src.dtypes[0], "single-band static raster")
            note = "single-band B10-adjusted static emissivity raster"
        else:
            raise ValueError(
                f"static GED raster {path!r} has {src.count} bands; "
                "expected 1 (B10-adjusted) or 5 (ASTER GED bands 10-14)"
            )
    arr = np.where(np.isfinite(arr), arr, np.nan)
    # Emissivity is strictly positive; non-positive codes are nodata sentinels.
    arr = np.where(arr > 0, arr, np.nan)
    return {"array": arr, "transform": transform, "crs": crs, "note": note}


# ── imperviousness loader ────────────────────────────────────────────────


def _load_imperv_vintage(path: str) -> dict:
    """Read one published imperviousness v3 COG (percent, NaN nodata)."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"imperviousness raster {path!r} has no CRS; cannot reproject")
        arr = src.read(1).astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return {"array": arr, "transform": src.transform, "crs": src.crs}


def _resolve_imperv_source(spec: str) -> dict[int, dict]:
    """Resolve ``--impervious-tif`` to vintage sources without GCS access.

    ``spec`` is either a single COG (used for every scene) or a directory
    holding ``imperviousness_2016.tif`` / ``imperviousness_2021.tif`` (the
    published v3 layout; vintage picked per scene by acquisition year).
    Returns a dict vintage-year -> loader spec; rasters load lazily.
    """
    from berlin_lst_downscaling.analysis.emissivity import IMPERV_VINTAGE_FILES

    if os.path.isdir(spec):
        out: dict[int, dict] = {}
        for vintage, fname in IMPERV_VINTAGE_FILES.items():
            candidate = os.path.join(spec, fname)
            if os.path.exists(candidate):
                out[vintage] = {"path": candidate, "loaded": None}
        if not out:
            raise ValueError(f"no imperviousness vintages found in {spec!r}")
        return out
    if os.path.exists(spec):
        return {-1: {"path": spec, "loaded": None}}
    raise ValueError(f"--impervious-tif {spec!r} is neither a file nor a directory")


def _imperv_for_scene(sources: dict[int, dict], year: int) -> dict:
    """Pick (and lazily load) the imperviousness vintage for a scene year."""
    if -1 in sources:
        key = -1
    else:
        vintage = imperv_vintage_for_year(year)
        key = vintage if vintage in sources else sorted(sources)[-1]
    entry = sources[key]
    if entry["loaded"] is None:
        entry["loaded"] = _load_imperv_vintage(entry["path"])
        print(f"    imperviousness: loaded {entry['path']}")
    out = dict(entry["loaded"])
    out["vintage_key"] = key
    out["path"] = entry["path"]
    return out


def _imperv_on_scene_grid(source: dict, dst_shape: tuple[int, int], dst_transform) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Area-average the 10 m sealing-percent field onto the 30 m scene grid.

    Documented method: the field is a continuous percent (class codes were
    already converted upstream), so averaging is correct — the same
    rationale as the repo's own imperviousness 2.5 m → 10 m step. NaN
    nodata propagates instead of smearing.
    """
    destination = np.full(dst_shape, np.nan, dtype=np.float64)
    reproject(
        np.ascontiguousarray(source["array"].astype(np.float64)),
        destination,
        src_transform=source["transform"],
        src_crs=source["crs"],
        dst_transform=dst_transform,
        dst_crs=LOAD_CRS,
        resampling=Resampling.average,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
    )
    return destination


# ── per-scene diagnostic ─────────────────────────────────────────────────


def analyze_scene(
    scene_id: str,
    season: str,
    platform: str,
    bbox: tuple[float, float, float, float],
    *,
    with_s2: bool,
    s2_window_days: int,
    ged_source: dict | None,
    imperv_sources: dict | None,
) -> dict:
    """Run experiments (a)/(b)/(c) for one scene; return a JSON-able report.

    ``ged_source`` is None (uniform-EMIS fallback) or a dict with
    ``array``/``transform``/``crs``/``note`` describing a static B10
    emissivity raster in any CRS — it is reprojected onto the 30 m scene
    grid inside (bilinear, explicit).
    """
    item = _resolve_landsat_item(scene_id)
    k1, k2, k_source = _mtl_thermal_constants(item)
    doc_k1, doc_k2 = thermal_constants_for_platform(platform)
    raw, transform = _load_landsat_window(item, bbox)

    qa = raw["qa_pixel"].astype(np.uint16)
    clear = landsat_qa_to_clear_bits(qa)
    # TIRS has no QA_RADSAT bit (LSDS-1619 v6 Table 6-4); saturation is the
    # ST uint16 maximum (§6.1.2). QA_RADSAT is kept only as evidence that
    # bits 8-10 stay unset over these scenes.
    saturated = thermal_saturated(raw["lwir11"])
    qa_radsat = raw["qa_radsat"].astype(np.uint16)
    fill = raw["lwir11"] == 0

    trad = aux_dn_to_units(raw["trad"].astype(np.int32), THERMAL_RADIANCE_SCALE)
    urad = aux_dn_to_units(raw["urad"].astype(np.int32), THERMAL_RADIANCE_SCALE)
    drad = aux_dn_to_units(raw["drad"].astype(np.int32), THERMAL_RADIANCE_SCALE)
    atran = aux_dn_to_units(raw["atran"].astype(np.int32), ATRAN_SCALE)
    emis = aux_dn_to_units(raw["emis"].astype(np.int32), EMIS_SCALE)
    emsd = aux_dn_to_units(raw["emsd"].astype(np.int32), EMIS_SCALE)
    st_product = st_dn_to_kelvin(raw["lwir11"])
    st_qa = aux_dn_to_units(raw["qa"].astype(np.int32), ST_QA_SCALE)

    aux_finite = np.isfinite(trad) & np.isfinite(urad) & np.isfinite(drad) & np.isfinite(atran)
    emis_valid = np.isfinite(emis) & (emis > 0) & (emis <= 1.0)
    analysis_mask = clear & ~saturated & ~fill & aux_finite & emis_valid
    st_finite = np.isfinite(st_product)
    sanity_mask = analysis_mask & st_finite

    # (a) sanity: reproduce ST_B10 with product EMIS.
    lst_a = lst_from_aux(trad, urad, drad, atran, emis, k1, k2)
    sanity_diff = np.abs(lst_a[sanity_mask] - st_product[sanity_mask])
    sanity = {
        "n": int(sanity_diff.size),
        "max_abs_diff_K": float(np.max(sanity_diff)) if sanity_diff.size else float("nan"),
        "mean_abs_diff_K": float(np.mean(sanity_diff)) if sanity_diff.size else float("nan"),
    }

    # Implied uniform emissivity offset: eps* that would reproduce ST_B10
    # exactly, minus product EMIS. Characterizes the sanity residual as a
    # calibration-level constant rather than scene-NDVI structure.
    with np.errstate(divide="ignore", invalid="ignore"):
        b_prod = k1 / (np.exp(k2 / st_product) - 1.0)
        eps_star = ((trad - urad) / atran - drad) / (b_prod - drad)
    eps_offset = eps_star[sanity_mask] - emis[sanity_mask]
    eps_offset = eps_offset[np.isfinite(eps_offset)]
    sanity["implied_eps_offset_median"] = (
        float(np.median(eps_offset)) if eps_offset.size else float("nan")
    )
    sanity["implied_eps_offset_std"] = (
        float(np.std(eps_offset)) if eps_offset.size else float("nan")
    )
    sanity["st_qa_mean_K"] = (
        float(np.nanmean(st_qa[sanity_mask])) if np.any(sanity_mask) else float("nan")
    )

    # (b) static emissivity counterfactuals.
    variants: dict[str, np.ndarray] = {}
    static_info: dict[str, object] = {}
    if ged_source is not None:
        ged_b10 = ged_b10_on_scene_grid(
            ged_source["array"],
            ged_source["transform"],
            ged_source["crs"],
            emis.shape,
            transform,
            LOAD_CRS,
        )
        static_info["static_ged"] = ged_source["note"]
        static_info["static_ged_n_nan"] = int(np.sum(~np.isfinite(ged_b10)))
        ged_valid = np.isfinite(ged_b10) & (ged_b10 > 0) & (ged_b10 <= 1.0)
        if not np.any(ged_valid & analysis_mask):
            raise ValueError("static GED raster has no valid pixels on the analysis mask")
        ged_field = np.where(ged_valid, ged_b10, np.nan)
        variants["static_ged"] = lst_from_aux(trad, urad, drad, atran, ged_field, k1, k2)
    else:
        eps_static = uniform_static_emissivity(emis, analysis_mask)
        variants["static_uniform"] = lst_from_aux(
            trad, urad, drad, atran, np.full_like(emis, eps_static), k1, k2
        )
        static_info["static_uniform_eps"] = eps_static
        static_info["static_uniform_note"] = (
            "FALLBACK: scene-median EMIS applied uniformly (ASTER GED "
            "requires Earthdata credentials); isolates the spatial "
            "NDVI-adjustment pattern, not an absolute GED product."
        )
    for eps_fix in FIXED_EPS_SCENARIOS:
        variants[f"fixed_{eps_fix:.2f}"] = lst_from_aux(
            trad, urad, drad, atran, np.full_like(emis, eps_fix), k1, k2
        )

    # Landsat SR NDVI stratification.
    red = olisr_dn_to_reflectance(raw["red"].astype(np.int32))
    nir = olisr_dn_to_reflectance(raw["nir08"].astype(np.int32))
    landsat_ndvi = ndvi(nir, red)
    ndvi_mask = analysis_mask & np.isfinite(landsat_ndvi)
    bins = assign_ndvi_bins(np.where(ndvi_mask, landsat_ndvi, np.nan))

    delta_stats: dict[str, object] = {}
    # Land-cover split available in every scene: Fmask water (QA bit 7)
    # vs land. Impervious/stadtstruktur layers are GCS-only (see report).
    water = qa_water(qa)
    surface_index = np.where(ndvi_mask, np.where(water, 1, 0), -1)
    imperv_index = np.full(ndvi_mask.shape, -1)
    imperv_note: object = None
    if imperv_sources is not None:
        year = item.datetime.year
        imperv = _imperv_for_scene(imperv_sources, year)
        imperv_30m = _imperv_on_scene_grid(imperv, emis.shape, transform)
        imperv_note = {"vintage_key": imperv["vintage_key"], "path": imperv.get("path")}
        imperv_index = assign_imperv_bins(np.where(ndvi_mask, imperv_30m, np.nan))
    for name, lst_b in variants.items():
        delta = np.where(ndvi_mask, lst_a - lst_b, np.nan)
        by_surface = binned_stats(delta, surface_index, 2)
        entry: dict[str, object] = {
            "overall": binned_stats(delta, np.zeros_like(bins), 1)[0],
            "by_ndvi_bin": dict(zip(NDVI_BIN_LABELS, binned_stats(delta, bins, 5), strict=True)),
            "by_surface": {"land": by_surface[0], "water": by_surface[1]},
        }
        if imperv_sources is not None:
            entry["by_imperv"] = dict(
                zip(IMPERV_BIN_LABELS, binned_stats(delta, imperv_index, 5), strict=True)
            )
        delta_stats[name] = entry

    relations: dict[str, object] = {
        "landsat_ndvi_vs_a": ndvi_lst_relation(landsat_ndvi[ndvi_mask], lst_a[ndvi_mask]),
    }
    first_b = next(iter(variants.values()))
    relations["landsat_ndvi_vs_b"] = ndvi_lst_relation(landsat_ndvi[ndvi_mask], first_b[ndvi_mask])
    relations["b_variant_for_relation"] = next(iter(variants.keys()))
    if imperv_sources is not None:
        # NDVI–LST slope stratified by impervious class (cheap: same arrays).
        strat: dict[str, object] = {}
        for i, label in enumerate(IMPERV_BIN_LABELS):
            sel = ndvi_mask & (imperv_index == i)
            strat[label] = {
                "a": ndvi_lst_relation(landsat_ndvi[sel], lst_a[sel]),
                "b": ndvi_lst_relation(landsat_ndvi[sel], first_b[sel]),
            }
        relations["imperv_stratified"] = strat

    # Optional same-season S2 NDVI join.
    s2_info: dict[str, object] = {"requested": with_s2, "scene_id": None}
    if with_s2:
        s2_item = _find_s2_companion(item, bbox, s2_window_days)
        if s2_item is not None:
            s2_ndvi_30m, s2_id = _load_s2_ndvi(s2_item, bbox, emis.shape, transform)
            s2_info["scene_id"] = s2_id
            joint = ndvi_mask & np.isfinite(s2_ndvi_30m)
            s2_info["n_joint"] = int(np.sum(joint))
            relations["s2_ndvi_vs_a"] = ndvi_lst_relation(s2_ndvi_30m[joint], lst_a[joint])
            relations["s2_ndvi_vs_b"] = ndvi_lst_relation(s2_ndvi_30m[joint], first_b[joint])
            # Scatter subsample for the figure (deterministic stride).
            idx = np.flatnonzero(joint)[:: max(1, int(np.sum(joint)) // 1200)]
            s2_info["scatter_idx"] = [int(i) for i in idx]
            s2_info["scatter_s2_ndvi"] = [float(v) for v in s2_ndvi_30m.ravel()[idx]]
            s2_info["scatter_landsat_ndvi"] = [float(v) for v in landsat_ndvi.ravel()[idx]]
            s2_info["scatter_a"] = [float(v) for v in lst_a.ravel()[idx]]
            s2_info["scatter_b"] = [float(v) for v in first_b.ravel()[idx]]

    snow_frac = float(np.mean(((qa >> 5) & 1) == 1))
    report = {
        "scene_id": scene_id,
        "season": season,
        "platform": platform,
        "acquisition_utc": item.datetime.isoformat(),
        "k1": k1,
        "k2": k2,
        "k_source": k_source,
        "k_matches_documented": math.isclose(k1, doc_k1, rel_tol=1e-6)
        and math.isclose(k2, doc_k2, rel_tol=1e-6),
        "grid_shape": list(raw["lwir11"].shape),
        "n_analysis": int(np.sum(analysis_mask)),
        "frac_clear": float(np.mean(clear)),
        "frac_saturated_b10": float(np.mean(saturated)),
        "qa_radsat_max": int(np.max(qa_radsat)),
        "frac_qa_radsat_bit8_set": float(np.mean(((qa_radsat >> 8) & 1) == 1)),
        "frac_qa_radsat_bit9_set": float(np.mean(((qa_radsat >> 9) & 1) == 1)),
        "frac_qa_radsat_bit10_set": float(np.mean(((qa_radsat >> 10) & 1) == 1)),
        "frac_snow_qa": snow_frac,
        "emis_median": float(np.nanmedian(emis[analysis_mask])) if np.any(analysis_mask) else None,
        "emis_std": float(np.nanstd(emis[analysis_mask])) if np.any(analysis_mask) else None,
        "emsd_mean": float(np.nanmean(emsd[analysis_mask])) if np.any(analysis_mask) else None,
        "sanity_a_vs_st": sanity,
        "static_info": static_info,
        "imperv_note": imperv_note,
        "delta_stats": delta_stats,
        "relations": relations,
        "s2": s2_info,
    }
    return report


# ── synthetic offline self-check ─────────────────────────────────────────


def synthetic_self_check() -> dict:
    """Offline end-to-end check of the analysis math (no network)."""
    rng = np.random.default_rng(7)
    shape = (40, 40)
    lst_true = 290.0 + 10.0 * rng.random(shape)
    emis = 0.965 + 0.025 * rng.random(shape)
    atran = 0.8 + 0.1 * rng.random(shape)
    urad = 1.2 + rng.random(shape)
    drad = 1.8 + rng.random(shape)
    k1, k2 = thermal_constants_for_platform("landsat-9")
    trad = forward_trad(lst_true, urad, drad, atran, emis, k1, k2)
    lst_a = lst_from_aux(trad, urad, drad, atran, emis, k1, k2)
    max_diff = float(np.nanmax(np.abs(lst_a - lst_true)))
    eps_static = uniform_static_emissivity(emis, np.ones(shape, dtype=bool))
    lst_b = lst_from_aux(trad, urad, drad, atran, np.full(shape, eps_static), k1, k2)
    delta = lst_a - lst_b
    fake_ndvi = (emis - emis.min()) / (emis.max() - emis.min()) * 0.8
    bins = assign_ndvi_bins(fake_ndvi)
    return {
        "synthetic_roundtrip_max_abs_diff_K": max_diff,
        "roundtrip_ok": max_diff < 1e-6,
        "delta_mean_K": float(np.nanmean(delta)),
        "static_eps": eps_static,
        "by_bin": dict(zip(NDVI_BIN_LABELS, binned_stats(delta, bins, 5), strict=True)),
        "relation_a": ndvi_lst_relation(fake_ndvi.ravel(), lst_a.ravel()),
        "relation_b": ndvi_lst_relation(fake_ndvi.ravel(), lst_b.ravel()),
    }


# ── SVG figures (dependency-free) ────────────────────────────────────────


def _svg_bar_groups(path: Path, title: str, groups: dict[str, dict[str, float]]) -> None:
    """Grouped mean/p05-p95 bars; groups: label -> {mean, p05, p95}."""
    width, height = 640, 320
    margin = 60
    labels = list(groups.keys())
    vals = [groups[k] for k in labels]
    all_v = [v for d in vals for v in (d["mean"], d["p05"], d["p95"]) if math.isfinite(v)]
    lo, hi = (min(all_v), max(all_v)) if all_v else (0.0, 1.0)
    pad = max(0.1, (hi - lo) * 0.2)
    lo, hi = lo - pad, hi + pad
    n = len(labels)
    slot = (width - 2 * margin) / max(n, 1)

    def y(v: float) -> float:
        return height - margin - (v - lo) / (hi - lo) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width // 2}" y="24" text-anchor="middle" font-size="14">{title}</text>',
        f'<line x1="{margin}" y1="{y(0)}" x2="{width - margin}" y2="{y(0)}" stroke="grey"/>',
    ]
    for i, (label, d) in enumerate(zip(labels, vals, strict=True)):
        cx = margin + slot * (i + 0.5)
        if math.isfinite(d["mean"]):
            parts.append(
                f'<rect x="{cx - 18}" y="{min(y(d["mean"]), y(0))}" width="36" '
                f'height="{abs(y(d["mean"]) - y(0)) + 1}" fill="steelblue"/>'
            )
        if math.isfinite(d["p05"]) and math.isfinite(d["p95"]):
            parts.append(
                f'<line x1="{cx}" y1="{y(d["p05"])}" x2="{cx}" y2="{y(d["p95"])}" '
                'stroke="black" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{cx}" y="{height - margin + 16}" text-anchor="middle" '
            f'font-size="10">{label}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{height - margin + 30}" text-anchor="middle" '
            f'font-size="10">{d["mean"]:+.2f} K</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _svg_scatter(path: Path, title: str, series: dict[str, tuple[list, list, str]]) -> None:
    """Scatter of up to a few thousand points per named series with colors."""
    width, height = 640, 400
    margin = 56
    xs = [x for _, (x, _, _) in series.items() for x in x]
    ys = [yy for _, (_, yy, _) in series.items() for yy in yy]
    if not xs:
        path.write_text("<svg/>")
        return
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    def px(x: float) -> float:
        return margin + (x - x0) / (x1 - x0) * (width - 2 * margin)

    def py(yy: float) -> float:
        return height - margin - (yy - y0) / (y1 - y0) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width // 2}" y="24" text-anchor="middle" font-size="14">{title}</text>',
    ]
    for name, (x, yy, color) in series.items():
        dots = " ".join(
            f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="1.6" fill="{color}"/>'
            for a, b in zip(x, yy, strict=True)
        )
        parts.append(f"<g>{dots}</g>")
        parts.append(
            f'<text x="{width - margin}" y="{margin + 16 * list(series).index(name)}" '
            f'text-anchor="end" font-size="11" fill="{color}">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


# ── figures (regenerable offline from saved artifacts) ───────────────────


def build_figures(
    out_dir: Path, ok: list[dict], rows: list[dict], fig_variant: str
) -> dict[str, object]:
    """Write the SVG figures; returns {"scatter_scene": id | None}."""
    # Season groups pool the per-scene OVERALL stats (n-weighted mean of
    # scene means, min/max whiskers) so pixel counts — not bin counts —
    # set the weight. Matches the report's group table.
    groups: dict[str, dict[str, float]] = {}
    for season in ("summer", "winter"):
        overalls = [
            r["delta_stats"][fig_variant]["overall"]
            for r in ok
            if r["season"] == season and fig_variant in r["delta_stats"]
        ]
        groups[f"{season} ({fig_variant})"] = pool_overall_stats(overalls)
    _svg_bar_groups(out_dir / "fig_delta_groups.svg", "Mean ΔLST (a)-(b) by season", groups)

    # ΔLST by NDVI bin, pooled per season (n-weighted mean of scene means).
    short = {
        "water/negative": "NDVI≤0",
        "bare/urban-low": "0–0.2",
        "sparse": "0.2–0.4",
        "moderate": "0.4–0.6",
        "dense": ">0.6",
    }
    for season in ("summer", "winter"):
        binned: dict[str, dict[str, float]] = {}
        for label in NDVI_BIN_LABELS:
            sel = [
                r
                for r in rows
                if r["season"] == season and r["variant"] == fig_variant and r["ndvi_bin"] == label
            ]
            n_tot = sum(int(r["n"]) for r in sel)
            if n_tot:
                mean = (
                    sum(
                        float(r["mean_K"]) * int(r["n"])
                        for r in sel
                        if math.isfinite(float(r["mean_K"]))
                    )
                    / n_tot
                )
            else:
                mean = float("nan")
            p05 = [float(r["p05_K"]) for r in sel if math.isfinite(float(r["p05_K"]))]
            p95 = [float(r["p95_K"]) for r in sel if math.isfinite(float(r["p95_K"]))]
            binned[short[label]] = {
                "mean": float(mean),
                "p05": float(min(p05)) if p05 else float("nan"),
                "p95": float(max(p95)) if p95 else float("nan"),
            }
        _svg_bar_groups(
            out_dir / f"fig_delta_ndvi_{season}.svg",
            f"ΔLST (a)-(b) by NDVI bin, {season} ({fig_variant})",
            binned,
        )

    # ΔLST by impervious class, pooled per season (same n-weighted pooling).
    if any("by_imperv" in r["delta_stats"].get(fig_variant, {}) for r in ok):
        for season in ("summer", "winter"):
            imp: dict[str, dict[str, float]] = {}
            for label in IMPERV_BIN_LABELS:
                cls_stats = [
                    r["delta_stats"][fig_variant]["by_imperv"][label]
                    for r in ok
                    if r["season"] == season and fig_variant in r["delta_stats"]
                ]
                imp[label] = pool_overall_stats(cls_stats)
            _svg_bar_groups(
                out_dir / f"fig_delta_imperv_{season}.svg",
                f"ΔLST (a)-(b) by impervious class, {season} ({fig_variant})",
                imp,
            )

    # NDVI-LST scatter: summer scene with the largest S2 joint sample.
    candidates = [
        r
        for r in ok
        if r["season"] == "summer"
        and isinstance(r["s2"].get("scatter_s2_ndvi"), list)
        and r["s2"].get("n_joint", 0)
    ]
    info: dict[str, object] = {"scatter_scene": None}
    if candidates:
        rep = max(candidates, key=lambda r: int(r["s2"].get("n_joint", 0)))
        info["scatter_scene"] = rep["scene_id"]
        _svg_scatter(
            out_dir / "fig_ndvi_lst_scatter.svg",
            f"NDVI-LST {rep['scene_id']}",
            {
                "LST(a) vs S2 NDVI": (
                    rep["s2"]["scatter_s2_ndvi"],
                    rep["s2"]["scatter_a"],
                    "steelblue",
                ),
                "LST(b) vs S2 NDVI": (
                    rep["s2"]["scatter_s2_ndvi"],
                    rep["s2"]["scatter_b"],
                    "tomato",
                ),
            },
        )
    else:
        print("note: no S2 scatter available; skipping scatter figure")
    return info


def _coerce(value: str) -> object:
    """Restore CSV scalars to int/float where possible (for --figures-only)."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# ── driver ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Landsat emissivity diagnostic")
    parser.add_argument("--synthetic", action="store_true", help="Offline math self-check only.")
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=list(BERLIN_BBOX), help="WGS84 analysis bbox."
    )
    parser.add_argument(
        "--out-dir",
        default=".scratch/emissivity-diagnostic",
        help="Local output directory (gitignored scratch; never GCS).",
    )
    parser.add_argument("--with-s2", action="store_true", help="Join same-season S2 NDVI.")
    parser.add_argument("--s2-window-days", type=int, default=16)
    parser.add_argument(
        "--aster-ged-tif",
        default=None,
        help="Optional local single-band static B10 emissivity raster.",
    )
    parser.add_argument(
        "--impervious-tif",
        default=None,
        help=(
            "Optional local imperviousness v3 source: a single COG or a "
            "directory with imperviousness_2016.tif/imperviousness_2021.tif "
            "(vintage picked per scene; no GCS access in code)."
        ),
    )
    parser.add_argument(
        "--scenes", nargs="*", default=None, help="Subset of pinned scene IDs to run."
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Rebuild SVGs from saved per_scene.json/delta_by_bin.csv (no network).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        per_scene = json.loads((out_dir / "per_scene.json").read_text())
        ok = [r for r in per_scene if "error" not in r]
        with open(out_dir / "delta_by_bin.csv", newline="") as f:
            rows = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]
        fig_variant = next(
            (v for v in ("static_uniform", "static_ged") if any(v in r["delta_stats"] for r in ok)),
            "fixed_0.98",
        )
        info = build_figures(out_dir, ok, rows, fig_variant)
        print(json.dumps(info, indent=2))
        return 0

    if args.synthetic:
        report = synthetic_self_check()
        (out_dir / "synthetic_self_check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not report["roundtrip_ok"]:
            print("SYNTHETIC SELF-CHECK FAILED")
            return 1
        print("synthetic self-check OK")
        return 0

    bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
    ged_source: dict | None = None
    ged_meta: dict[str, object] = {"provided": False}
    if args.aster_ged_tif:
        ged_source = _load_ged_source(args.aster_ged_tif)
        ged_meta = {"provided": True, "path": args.aster_ged_tif, "note": ged_source["note"]}
    imperv_sources: dict | None = None
    if args.impervious_tif:
        imperv_sources = _resolve_imperv_source(args.impervious_tif)

    wanted = set(args.scenes) if args.scenes else None
    scenes = [s for s in SCENES if wanted is None or s[0] in wanted]
    per_scene: list[dict] = []
    for scene_id, season, platform in scenes:
        print(f"--- {scene_id} ({season}, {platform})", flush=True)
        try:
            rep = analyze_scene(
                scene_id,
                season,
                platform,
                bbox,
                with_s2=args.with_s2,
                s2_window_days=args.s2_window_days,
                ged_source=ged_source,
                imperv_sources=imperv_sources,
            )
        except Exception as exc:  # noqa: BLE001 — one scene must not kill the batch
            print(f"    FAILED: {exc}")
            per_scene.append({"scene_id": scene_id, "error": str(exc)})
            continue
        per_scene.append(rep)
        s = rep["sanity_a_vs_st"]
        print(
            f"    sanity(a) vs ST_B10: n={s['n']} max={s['max_abs_diff_K']:.4f} K "
            f"mean={s['mean_abs_diff_K']:.4f} K; K source: {rep['k_source']}"
        )

    (out_dir / "per_scene.json").write_text(json.dumps(per_scene, indent=2))

    # Pooled group table: (season, platform) x variant x NDVI bin.
    ok = [r for r in per_scene if "error" not in r]
    rows: list[dict] = []
    for rep in ok:
        for variant, stats in rep["delta_stats"].items():
            for b, label in enumerate(NDVI_BIN_LABELS):
                s = stats["by_ndvi_bin"][label]
                rows.append(
                    {
                        "scene_id": rep["scene_id"],
                        "season": rep["season"],
                        "platform": rep["platform"],
                        "variant": variant,
                        "ndvi_bin": label,
                        "bin_index": b,
                        "mean_K": s["mean"],
                        "p05_K": s["p05"],
                        "p95_K": s["p95"],
                        "n": s["n"],
                    }
                )
    with open(out_dir / "delta_by_bin.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    # Same layout by impervious class (only when --impervious-tif was given).
    imperv_rows: list[dict] = []
    for rep in ok:
        for variant, stats in rep["delta_stats"].items():
            if "by_imperv" not in stats:
                continue
            for b, label in enumerate(IMPERV_BIN_LABELS):
                s = stats["by_imperv"][label]
                imperv_rows.append(
                    {
                        "scene_id": rep["scene_id"],
                        "season": rep["season"],
                        "platform": rep["platform"],
                        "variant": variant,
                        "imperv_class": label,
                        "bin_index": b,
                        "mean_K": s["mean"],
                        "p05_K": s["p05"],
                        "p95_K": s["p95"],
                        "n": s["n"],
                    }
                )
    if imperv_rows:
        with open(out_dir / "delta_by_imperv.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(imperv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(imperv_rows)

    # Group-pooled means for the figure (variant static_uniform/first).
    fig_variant = next(
        (v for v in ("static_uniform", "static_ged") if any(v in r["delta_stats"] for r in ok)),
        "fixed_0.98",
    )
    build_figures(out_dir, ok, rows, fig_variant)

    summary = {
        "scenes_requested": len(scenes),
        "scenes_ok": len(ok),
        "scenes_failed": len(per_scene) - len(ok),
        "ged": ged_meta,
        "figure_variant": fig_variant,
        "relations": [
            {
                "scene_id": r["scene_id"],
                "season": r["season"],
                "platform": r["platform"],
                **{k: v for k, v in r["relations"].items()},
            }
            for r in ok
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
