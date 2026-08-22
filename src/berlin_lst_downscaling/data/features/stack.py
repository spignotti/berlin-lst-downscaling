"""Feature-stack composer — 28-band stack + feature_valid mask.

Reads the resolved per-scene inputs (S2 ARD + flag, morphology source COGs,
ERA5-Land, shadows) plus the exact Berlin AOI mask and produces one
canonical-grid dataset of 28 float32 channels plus the uint8
``feature_valid`` mask.

Mask semantics (data/features/contracts.py): a pixel is valid only when
it lies inside the AOI, the S2 ARD flag is clear (``flag == 0``), and all
28 channels are finite and within their declared ranges. Availability is
per channel: an unavailable channel is NaN in that band only — known
values of the other channels are preserved. ``feature_valid`` is the
aggregate "complete 28-channel vector" availability, never a
training-selection mask. All channels are NaN only outside the AOI.
The Landsat target and ECOSTRESS never enter this stack.

Reading is windowed against the analysis grid: the full canonical grid by
default, a canonical-aligned bbox subset for local smoke tests. All inputs
are published on the same 10 m lattice, so the window offset is exact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import rasterio
import rasterio.warp as rwarp
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.data.features.contracts import (
    ALBEDO_WEIGHTS,
    FEATURE_CHANNELS,
)
from berlin_lst_downscaling.data.io import log_event

_logger = logging.getLogger(__name__)

_GRID_CRS = "EPSG:25833"
_SHADOW_NODATA = 255

# ── input resolution ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureInputs:
    """Resolved per-scene input COG URIs (all on the canonical 10 m grid)."""

    s2_cog: str
    s2_flag: str
    morphology: dict[str, tuple[str, int]]  # channel -> (COG URI, band number)
    era5_cog: str
    shadows: dict[str, str]  # shadow_building, shadow_vegetation
    lod_coverage: np.ndarray  # bool (grid.shape), True = covered by a LoD source tile


# ── AOI ───────────────────────────────────────────────────────────────


def load_aoi_mask_on_grid(aoi_uri: str, grid: GeoBox) -> np.ndarray:
    """Return a bool mask (True = inside Berlin) on *grid*.

    # decision: ``aoi_10m.tif`` sits on its own 10 m EPSG:25833 window
    # (origin offset ~790.8/1130.7 m from the canonical lattice, verified
    # 2026-08-18). It is reprojected onto the analysis grid with nearest
    # resampling — the same treatment as ``data/selection/_aoi.py``. For a
    # binary AOI mask this moves boundary cells by at most one 10 m cell.
    """
    with rasterio.open(aoi_uri) as src:
        source = src.read(1).astype(np.uint8)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata

    destination = np.zeros((grid.shape.y, grid.shape.x), dtype=np.uint8)
    rwarp.reproject(
        source=source,
        src_crs=src_crs,
        src_transform=src_transform,
        src_nodata=src_nodata,
        destination=destination,
        dst_crs=grid.crs,
        dst_transform=grid.transform,
        dst_nodata=0,
        resampling=rwarp.Resampling.nearest,
    )
    return destination == 1


# ── windowed reads ────────────────────────────────────────────────────


def _window_offset(src: rasterio.DatasetReader, grid: GeoBox) -> tuple[int, int]:
    """Return (col_off, row_off) mapping analysis-grid indices onto *src* pixels."""
    col_off = round((grid.transform.xoff - src.transform.xoff) / 10.0)
    row_off = round((src.transform.yoff - grid.transform.yoff) / 10.0)
    return col_off, row_off


def _read_band(uri: str, band: int, grid: GeoBox, dtype: str = "float32") -> np.ndarray:
    """Read one band of *uri* on the analysis grid (windowed when bounded)."""
    with rasterio.open(uri) as src:
        co, ro = _window_offset(src, grid)
        window = Window(co, ro, grid.shape.x, grid.shape.y)  # type: ignore[call-arg]
        return src.read(band, window=window).astype(dtype)


def _read_shadow(uri: str, grid: GeoBox) -> np.ndarray:
    """Read a uint8 shadow COG and cast to float32 with 255 → NaN."""
    arr = _read_band(uri, 1, grid, dtype="uint8").astype(np.float32)
    arr[arr == _SHADOW_NODATA] = np.nan
    return arr


# ── composition ───────────────────────────────────────────────────────


@dataclass
class ComposedFeatureStack:
    """The 28-band dataset plus the validity mask and coverage metrics."""

    dataset: xr.Dataset  # 28 float32 bands on the analysis grid
    mask: np.ndarray  # uint8 (grid.shape), 1 = valid
    coverage: dict  # total/inside/outside/feature-valid pixel counts


def compose_feature_stack(
    inputs: FeatureInputs,
    aoi: np.ndarray,
    grid: GeoBox,
) -> ComposedFeatureStack:
    """Compose the 28-band feature stack for one scene.

    *aoi* must already be a bool mask on *grid* (see
    :func:`load_aoi_mask_on_grid`).
    """
    # ── S2 spectral bands + flag ───────────────────────────────────────
    s2 = [_read_band(inputs.s2_cog, i, grid) for i in range(1, 7)]
    b02, b03, b04, b08, b11, b12 = s2
    s2_flag = _read_band(inputs.s2_flag, 1, grid, dtype="uint8")
    s2_clear = s2_flag == 0
    s2_invalid = ~s2_clear

    # The six spectral channels are unavailable where the ARD flag is set;
    # static and dynamic channels keep their values there.
    for arr in s2:
        arr[s2_invalid] = np.nan

    # ── spectral indices (NaN-safe ratios) ────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(b08 + b04 != 0, (b08 - b04) / (b08 + b04), np.nan)
        ndwi = np.where(b03 + b08 != 0, (b03 - b08) / (b03 + b08), np.nan)
        ndbi = np.where(b11 + b08 != 0, (b11 - b08) / (b11 + b08), np.nan)

    albedo = np.zeros_like(b02)
    for w, b in zip(ALBEDO_WEIGHTS, s2, strict=True):
        albedo += w * b

    # S2-derived channels share the S2 flag availability.
    for arr in (ndvi, ndwi, ndbi, albedo):
        arr[s2_invalid] = np.nan

    # ── morphology (semantic predictors from source COGs) ─────────────
    # Each entry is (uri, band_number). Multi-band source COGs (lod2, vh)
    # are read at the specified band; single-band sources (imp, svf) at 1.
    morphology = {
        name: _read_band(uri, band, grid) for name, (uri, band) in sorted(inputs.morphology.items())
    }

    # LoD2 semantics: a cell without a building inside the source-covered
    # area is a known zero; a cell outside the source coverage is a true
    # data gap and stays NaN. The four LoD bands are jointly NaN or jointly
    # finite (same rasterization count) — a mixed state is corrupt.
    _LOD_BAND_NAMES = (
        "building_height_mean",
        "building_height_std",
        "building_coverage_ratio",
        "building_height_max",
    )
    lod = [morphology[name] for name in _LOD_BAND_NAMES]
    lod_nan = np.stack([np.isnan(arr) for arr in lod], axis=0)
    lod_all_nan = np.all(lod_nan, axis=0)
    if np.any(lod_nan & ~lod_all_nan):
        raise ValueError(
            f"LoD morphology bands have mixed finite/NaN state on "
            f"{int(np.sum(lod_nan & ~lod_all_nan))} px"
        )
    cov = np.asarray(inputs.lod_coverage, dtype=bool)
    if cov.shape != lod_all_nan.shape:
        raise ValueError(
            f"lod_coverage shape {cov.shape} != grid {lod_all_nan.shape}"
        )
    covered_no_building = lod_all_nan & cov & aoi
    for arr in lod:
        arr[covered_no_building] = 0.0

    # ── ERA5-Land (8 bands) + shadows ─────────────────────────────────
    era5 = [_read_band(inputs.era5_cog, i, grid) for i in range(1, 9)]
    shadows = {name: _read_shadow(uri, grid) for name, uri in sorted(inputs.shadows.items())}

    # ── fixed channel order ───────────────────────────────────────────
    channels: list[np.ndarray] = [
        b02,
        b03,
        b04,
        b08,
        b11,
        b12,
        ndvi,
        ndwi,
        ndbi,
        albedo,
        morphology["building_height_mean"],
        morphology["building_height_std"],
        morphology["building_coverage_ratio"],
        morphology["building_height_max"],
        morphology["vegetation_height_mean"],
        morphology["vegetation_height_max"],
        morphology["imperviousness"],
        morphology["svf"],
        *era5,
        shadows["shadow_building"],
        shadows["shadow_vegetation"],
    ]
    if len(channels) != 28:  # pragma: no cover — contract guard
        raise AssertionError(f"composed {len(channels)} channels, expected 28")

    # ── validity: AOI ∩ S2-clear ∩ finite ∩ in-range ──────────────────
    valid = aoi & s2_clear
    for arr, spec in zip(channels, FEATURE_CHANNELS, strict=True):
        finite = np.isfinite(arr)
        if spec.valid_range is None:
            valid &= finite
            continue
        lo, hi = spec.valid_range
        valid &= finite & (arr >= lo) & (arr <= hi)

    mask = valid.astype(np.uint8)
    stack = np.stack(channels, axis=0)
    # All channels are NaN only outside the AOI. Inside the AOI, channels
    # keep their individually known values; ``feature_valid`` (the mask)
    # marks the aggregate complete-vector availability.
    stack[:, ~aoi] = np.nan

    # ── dataset (canonical coords + transform) ────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    data_vars = {ch.name: (("y", "x"), stack[i]) for i, ch in enumerate(FEATURE_CHANNELS)}
    ds = xr.Dataset(data_vars, coords={"x": xs, "y": ys})
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    total = int(aoi.size)
    inside = int(np.sum(aoi))
    outside = total - inside
    feature_valid = int(np.sum(valid))
    coverage = {
        "total_px": total,
        "inside_aoi_px": inside,
        "outside_aoi_px": outside,
        "feature_valid_px": feature_valid,
        "aoi_frac": round(inside / total, 6) if total else 0.0,
        "feature_valid_frac_of_aoi": round(feature_valid / inside, 6) if inside else 0.0,
        # LoD semantic classification (diagnostic): known no-building cells
        # are zeroed, source gaps stay NaN. Finite LoD values always win.
        "lod_building_px": int(np.sum(~lod_all_nan)),
        "lod_covered_no_building_px": int(np.sum(covered_no_building)),
        "lod_source_gap_px": int(np.sum(lod_all_nan & ~cov)),
        "lod_source_gap_inside_aoi_px": int(np.sum(lod_all_nan & ~cov & aoi)),
    }
    log_event(
        _logger,
        logging.INFO,
        "feature_stack_composed",
        inside_aoi_px=inside,
        outside_aoi_px=outside,
        feature_valid_px=feature_valid,
        lod_covered_no_building_px=coverage["lod_covered_no_building_px"],
        lod_source_gap_inside_aoi_px=coverage["lod_source_gap_inside_aoi_px"],
    )

    return ComposedFeatureStack(dataset=ds, mask=mask, coverage=coverage)


__all__ = [
    "ComposedFeatureStack",
    "FeatureInputs",
    "compose_feature_stack",
    "load_aoi_mask_on_grid",
]
