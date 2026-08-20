"""Stage-1 raw-input QA core — metadata checks and blockwise joint support.

Validates every published raw input of the training universe against the
Stage-1 contract and computes a **diagnostic** joint-support statistic at
100 m (valid Landsat target cell plus all present 10 m subpixels valid).
No validity mask, no selection artifact, and no resampling is produced —
the report is the only output. The final training-eligibility mask is a
post-Stage-2 concern by user decision.

Performance design (bounded GCS reads, blockwise processing):

- Pixel scans use 2560x2560 10 m tiles — an exact multiple of both the
  COG 512 px block size and the 10x10 aggregation factor.
- S2 support uses the authoritative ARD flag band (``flag == 0`` means
  clear, ``data/ard/aoi.py``); reflectance ranges are sanity-checked on a
  coarse overview sample per scene because the ARD masking step already
  clips reflectance to [0, 1] pixel-exactly.
- ERA5 has no flag band — its 8 bands are read per tile and validity is
  ``finite and within the declared per-band valid_range``.
- Static derived morphology is vintage-fixed: a packed uint8 validity
  mask is cached per ``(product, geometry_id)`` in a temporary directory
  and reused by every scene of that profile; the cache is removed in
  ``finally`` and is never an output artifact.
- Landsat target/flag are read once per scene at 100 m (tiny).
"""

from __future__ import annotations

import csv as _csv
import io as _io
import logging
import math
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.common.grid import canon_grid_10m, canon_grid_100m
from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.dynamic.era5 import contract_for_era5_scene
from berlin_lst_downscaling.data.io import atomic_write, log_event
from berlin_lst_downscaling.data.qa.contracts import (
    DSM_RANGE_M,
    LST_RANGE_K,
    S2_REFLECTANCE_RANGE,
    SHADOW_VALID_VALUES,
    SVF_RANGE,
)
from berlin_lst_downscaling.data.qa.inventory import (
    INFERENCE_EXCLUSION_REASON,
    ResolvedScene,
    build_inventory,
)

_logger = logging.getLogger(__name__)

_TILE = 2560  # 10 m tile size (512 x 5, 10 x 256)
_GRID_CRS = "EPSG:25833"
_SAMPLE_FACTOR = 32  # overview sampling for coarse range checks

# Sentinel-2 spectral band count, derived from the ARD contract so the
# Stage-1 gate cannot drift from the published S2 schema.
_S2_N_BANDS = len(contract_for_source("sentinel-2-l2a").output_bands)

# Expected exclusions: reported, never findings. Anything else that
# prevents assessment of a paired anchor is a hard finding.
_EXPECTED_EXCLUSIONS = frozenset({INFERENCE_EXCLUSION_REASON})

_SUPPORT_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
_BUCKET_LABELS = ("0-25", "25-50", "50-75", "75-90", "90-99", "99-100", "100")

_STATIC_RANGES: dict[str, tuple[float, float]] = {
    "building_dsm": DSM_RANGE_M,
    "combined_dsm": DSM_RANGE_M,
    "vegetation_dsm": DSM_RANGE_M,
    "svf": SVF_RANGE,
}


# ── report types ─────────────────────────────────────────────────────


@dataclass
class SceneMetrics:
    """Per-scene Stage-1 results (one row in the scene table)."""

    scene_id: str
    year: int
    s2_scene_id: str
    geometry_id: str
    assessed: bool
    exclusion_reason: str | None
    target_valid_cells: int = 0
    all_100_cells: int = 0
    full_support_cells: int = 0
    support_mean_frac: float = 0.0
    support_histogram: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class Stage1Report:
    """Complete Stage-1 QA report (serialised to summary.json + tables)."""

    run_id: str
    timestamp: str
    grid: dict
    inputs: dict
    fingerprints: dict
    total_pairings: int
    assessed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    aggregate: dict
    layers: dict[str, dict]
    findings: list[str]
    scenes: list[SceneMetrics]

    @property
    def ok(self) -> bool:
        return len(self.findings) == 0


# ── analysis grid ────────────────────────────────────────────────────


def analysis_grid_10m(bbox_wgs84: tuple[float, float, float, float] | None) -> GeoBox:
    """Return the 10 m analysis grid, optionally a canonical-aligned subset.

    ``None`` returns the full canonical Berlin grid. A bbox is snapped to
    the canonical **100 m lattice** (a multiple of both 10 m and 100 m) so
    that ``zoom_out(10)`` yields a 100 m analysis grid whose origin maps
    pixel-exactly onto the published 100 m Landsat COGs as well as the
    10 m layers.
    """
    if bbox_wgs84 is None:
        return canon_grid_10m()
    from rasterio.warp import transform_bounds

    origin = canon_grid_10m().transform
    ox, oy = origin.xoff, origin.yoff
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", _GRID_CRS, *bbox_wgs84)
    step = 100.0
    # Snap outward on every edge so the subset fully contains the bbox
    # and stays on the canonical 100 m lattice (x grows east, y grows
    # south, so top/north uses floor, bottom/south uses ceil).
    x0 = ox + step * math.floor((minx - ox) / step)
    x1 = ox + step * math.ceil((maxx - ox) / step)
    y1 = oy - step * math.floor((oy - maxy) / step)
    y0 = oy - step * math.ceil((oy - miny) / step)
    return GeoBox.from_bbox((x0, y0, x1, y1), crs=_GRID_CRS, resolution=10)


# ── metadata checks (no full-band reads) ─────────────────────────────


def _check_metadata(
    uri: str,
    expected_grid: GeoBox,
    *,
    n_bands: int | None,
    dtype: str,
) -> list[str]:
    """Lightweight structural check: openable, CRS, band count, dtype, grid.

    ``n_bands=None`` skips the band-count assertion (used for metadata-only
    static source products whose band layout varies, e.g. LoD2 morphology
    with height/std/BCR bands — the exact contract is enforced at write
    time by ``validate_secondary_cog``).
    """
    errors: list[str] = []
    try:
        with rasterio.open(uri) as src:
            crs = str(src.crs).upper() if src.crs else "None"
            if crs != _GRID_CRS:
                errors.append(f"{uri}: CRS {crs!r}, expected {_GRID_CRS!r}")
            if n_bands is not None and src.count != n_bands:
                errors.append(f"{uri}: band count {src.count}, expected {n_bands}")
            if src.dtypes[0] != dtype:
                errors.append(f"{uri}: dtype {src.dtypes[0]!r}, expected {dtype!r}")
            ex, ey = expected_grid.shape.x, expected_grid.shape.y
            if src.width != ex or src.height != ey:
                errors.append(f"{uri}: shape ({src.width}, {src.height}), expected ({ex}, {ey})")
            if (
                abs(src.transform.xoff - expected_grid.transform.xoff) > 0.01
                or abs(src.transform.yoff - expected_grid.transform.yoff) > 0.01
            ):
                errors.append(f"{uri}: origin does not match canonical grid")
    except Exception as exc:  # noqa: BLE001 — report, never crash the run
        errors.append(f"{uri}: cannot open: {exc}")
    return errors


def _check_metadata_10m(
    uri: str, *, n_bands: int | None, dtype: str
) -> list[str]:
    return _check_metadata(uri, canon_grid_10m(), n_bands=n_bands, dtype=dtype)


def _check_metadata_100m(
    uri: str, *, n_bands: int | None, dtype: str
) -> list[str]:
    return _check_metadata(uri, canon_grid_100m(), n_bands=n_bands, dtype=dtype)


# ── window mapping (exact, shared origin) ────────────────────────────


def _window_offset(src: rasterio.DatasetReader, analysis: GeoBox, res: float) -> tuple[int, int]:
    """Return (col_off, row_off) mapping analysis-grid indices onto *src* pixels."""
    col_off = round((analysis.transform.xoff - src.transform.xoff) / res)
    row_off = round((src.transform.yoff - analysis.transform.yoff) / res)
    return col_off, row_off


def _tile_windows(analysis: GeoBox, tile: int = _TILE):
    """Yield ((r0, c0, r1, c1), Window) over the analysis grid, in tile steps."""
    h, w = analysis.shape.y, analysis.shape.x
    for r0 in range(0, h, tile):
        r1 = min(r0 + tile, h)
        for c0 in range(0, w, tile):
            c1 = min(c0 + tile, w)
            yield (r0, c0, r1, c1), Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore[call-arg]


# ── coarse sampled range check (S2 reflectance) ──────────────────────


def _sample_band_range(uri: str, band: int) -> tuple[float, float] | None:
    """Return (min, max) over a coarse overview sample, or None on failure."""
    try:
        with rasterio.open(uri) as src:
            out_h = max(1, src.height // _SAMPLE_FACTOR)
            out_w = max(1, src.width // _SAMPLE_FACTOR)
            arr = src.read(band, out_shape=(out_h, out_w)).astype(np.float64)
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return None
            return float(valid.min()), float(valid.max())
    except Exception as exc:  # noqa: BLE001
        log_event(_logger, logging.WARNING, "sample_range_failed", uri=uri, error=str(exc))
        return None


# ── static derived validity cache (per geometry profile) ─────────────


def _static_validity_mask(
    product: str,
    cog_uri: str,
    profile_key: str,
    tmp_dir: Path,
) -> tuple[str, list[str]]:
    """Build (and cache) a packed uint8 validity mask for one static product.

    Validity: finite and within the product's declared physical range.
    Returns the mask file path and any findings.
    """
    cache_path = tmp_dir / f"static_{product}_{profile_key}.tif"
    findings: list[str] = []
    if cache_path.exists():
        return str(cache_path), findings

    vmin, vmax = _STATIC_RANGES[product]
    grid = canon_grid_10m()
    profile = {
        "driver": "GTiff",
        "width": grid.shape.x,
        "height": grid.shape.y,
        "count": 1,
        "dtype": "uint8",
        "crs": _GRID_CRS,
        "transform": grid.transform,
        "compress": "deflate",
    }
    try:
        with rasterio.open(cog_uri) as src:
            with rasterio.open(cache_path, "w", **profile) as dst:
                for _, win in _tile_windows(grid):
                    arr = src.read(1, window=win).astype(np.float32)
                    valid = np.isfinite(arr) & (arr >= vmin) & (arr <= vmax)
                    dst.write(valid.astype(np.uint8), 1, window=win)
    except Exception as exc:  # noqa: BLE001
        cache_path.unlink(missing_ok=True)
        findings.append(f"static {product} ({profile_key}): cannot build mask: {exc}")
        return "", findings
    return str(cache_path), findings


# ── per-scene scan ───────────────────────────────────────────────────


def _shadow_valid(src: rasterio.DatasetReader, win: Window) -> np.ndarray:
    """Return validity mask for a shadow COG window (0/1 valid, 255 nodata)."""
    arr = src.read(1, window=win)
    valid = np.zeros(arr.shape, dtype=bool)
    for value in SHADOW_VALID_VALUES:
        valid |= arr == value
    return valid


def _scan_scene(
    scene: ResolvedScene,
    analysis_10: GeoBox,
    analysis_100: GeoBox,
    static_masks: dict[str, str],
    era5_ranges: list[tuple[float, float]],
) -> tuple[SceneMetrics, dict[str, int], list[str]]:
    """Run the blockwise scan for one assessable scene.

    Returns (metrics, layer_invalid_px, findings).
    """
    findings: list[str] = []
    errors: list[str] = []

    # ── metadata checks (canonical grids, no pixel reads) ──────────────
    errors += _check_metadata_100m(scene.landsat_cog, n_bands=1, dtype="float32")
    errors += _check_metadata_100m(scene.landsat_flag, n_bands=1, dtype="uint8")
    errors += _check_metadata_10m(scene.s2_cog, n_bands=_S2_N_BANDS, dtype="float32")
    errors += _check_metadata_10m(scene.s2_flag, n_bands=1, dtype="uint8")
    era5_cog = scene.dynamic.get("era5_land", "")
    if era5_cog:
        errors += _check_metadata_10m(era5_cog, n_bands=8, dtype="float32")
    for source in ("shadow_building", "shadow_vegetation"):
        uri = scene.dynamic.get(source, "")
        if uri:
            errors += _check_metadata_10m(uri, n_bands=1, dtype="uint8")

    # ── Landsat target (100 m, full read — tiny) ───────────────────────
    st = flag = None
    ls_off = (0, 0)
    try:
        with rasterio.open(scene.landsat_cog) as src:
            st = src.read(1).astype(np.float32)
            ls_off = _window_offset(src, analysis_100, 100.0)
        with rasterio.open(scene.landsat_flag) as src:
            flag = src.read(1).astype(np.uint8)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene.scene_id}: cannot read Landsat target: {exc}")

    H100, W100 = analysis_100.shape.y, analysis_100.shape.x
    target_valid = np.zeros((H100, W100), dtype=bool)
    if st is not None and flag is not None:
        co, ro = ls_off
        y0, y1 = max(ro, 0), min(ro + H100, st.shape[0])
        x0, x1 = max(co, 0), min(co + W100, st.shape[1])
        sl_st = st[y0:y1, x0:x1]
        sl_fl = flag[y0:y1, x0:x1]
        target_valid[y0 - ro : y1 - ro, x0 - co : x1 - co] = (
            (sl_fl == 0) & (sl_st >= LST_RANGE_K[0]) & (sl_st <= LST_RANGE_K[1])
        )

    # ── S2 coarse reflectance range sanity (overview sample) ───────────
    for band in range(1, _S2_N_BANDS + 1):
        rng = _sample_band_range(scene.s2_cog, band)
        if rng is not None and not (
            S2_REFLECTANCE_RANGE[0] <= rng[0] <= S2_REFLECTANCE_RANGE[1]
            and S2_REFLECTANCE_RANGE[0] <= rng[1] <= S2_REFLECTANCE_RANGE[1]
        ):
            findings.append(
                f"{scene.scene_id}: S2 band {band} sampled range {rng[0]:.3f}.."
                f"{rng[1]:.3f} outside [0, 1]"
            )

    # ── present-count grid (edge-truncated 100 m cells) ────────────────
    H10, W10 = analysis_10.shape.y, analysis_10.shape.x
    iy = np.arange(H100, dtype=np.int64)[:, None]
    ix = np.arange(W100, dtype=np.int64)[None, :]
    present = np.minimum(10, H10 - 10 * iy) * np.minimum(10, W10 - 10 * ix)

    support = np.zeros((H100, W100), dtype=np.int32)
    layer_invalid: dict[str, int] = {
        "s2": 0,
        "era5": 0,
        "shadow_building": 0,
        "shadow_vegetation": 0,
    }
    for product in static_masks:
        layer_invalid[f"static_{product}"] = 0

    shb_uri = scene.dynamic.get("shadow_building", "")
    shv_uri = scene.dynamic.get("shadow_vegetation", "")
    if era5_cog and shb_uri and shv_uri:
        try:
            with (
                rasterio.open(scene.s2_flag) as s2f,
                rasterio.open(era5_cog) as era5,
                rasterio.open(shb_uri) as shb,
                rasterio.open(shv_uri) as shv,
            ):
                s2f_off = _window_offset(s2f, analysis_10, 10.0)
                era5_off = _window_offset(era5, analysis_10, 10.0)
                shb_off = _window_offset(shb, analysis_10, 10.0)
                shv_off = _window_offset(shv, analysis_10, 10.0)
                static_srcs = {product: rasterio.open(uri) for product, uri in static_masks.items()}
                try:
                    static_offs = {
                        product: _window_offset(src, analysis_10, 10.0)
                        for product, src in static_srcs.items()
                    }
                    for (r0, c0, r1, c1), _ in _tile_windows(analysis_10):
                        bh, bw = r1 - r0, c1 - c0

                        # ── S2 validity: flag == 0 ────────────────────
                        w_s2 = Window(c0 + s2f_off[0], r0 + s2f_off[1], bw, bh)  # type: ignore[call-arg]
                        s2_valid = s2f.read(1, window=w_s2) == 0
                        layer_invalid["s2"] += int(np.sum(~s2_valid))

                        # ── ERA5: finite + in range (8 bands) ─────────
                        era5_valid = np.ones((bh, bw), dtype=bool)
                        for i, (vmin, vmax) in enumerate(era5_ranges, 1):
                            w_e = Window(c0 + era5_off[0], r0 + era5_off[1], bw, bh)  # type: ignore[call-arg]
                            arr = era5.read(i, window=w_e).astype(np.float32)
                            era5_valid &= (arr >= vmin) & (arr <= vmax)
                        layer_invalid["era5"] += int(np.sum(~era5_valid))

                        # ── shadows: value in {0, 1} ──────────────────
                        w_b = Window(c0 + shb_off[0], r0 + shb_off[1], bw, bh)  # type: ignore[call-arg]
                        shb_valid = _shadow_valid(shb, w_b)
                        w_v = Window(c0 + shv_off[0], r0 + shv_off[1], bw, bh)  # type: ignore[call-arg]
                        shv_valid = _shadow_valid(shv, w_v)
                        layer_invalid["shadow_building"] += int(np.sum(~shb_valid))
                        layer_invalid["shadow_vegetation"] += int(np.sum(~shv_valid))

                        # ── static derived (cached masks) ─────────────
                        static_valid = np.ones((bh, bw), dtype=bool)
                        for product, src in static_srcs.items():
                            off = static_offs[product]
                            w_m = Window(c0 + off[0], r0 + off[1], bw, bh)  # type: ignore[call-arg]
                            m = src.read(1, window=w_m).astype(bool)
                            static_valid &= m
                            layer_invalid[f"static_{product}"] += int(np.sum(~m))

                        combined = s2_valid & era5_valid & shb_valid & shv_valid & static_valid

                        # ── aggregate to 100 m (exact nested 10x10) ───
                        ph, pw = -(-bh // 10) * 10, -(-bw // 10) * 10
                        if ph != bh or pw != bw:
                            m = np.zeros((ph, pw), dtype=np.uint8)
                            m[:bh, :bw] = combined
                        else:
                            m = combined.astype(np.uint8)
                        block_sum = (
                            m.reshape(ph // 10, 10, pw // 10, 10).sum(axis=(1, 3)).astype(np.int32)
                        )
                        br0, bc0 = r0 // 10, c0 // 10
                        support[br0 : br0 + ph // 10, bc0 : bc0 + pw // 10] += block_sum
                finally:
                    for src in static_srcs.values():
                        src.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{scene.scene_id}: pixel scan failed: {exc}")

    # ── support statistics over target-valid cells ────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(present > 0, support / np.maximum(present, 1), np.nan)
    cells = target_valid & (present > 0)
    target_valid_cells = int(np.sum(target_valid))
    all_100_cells = int(np.sum(target_valid & (present == 10 * 10) & (support == 100)))
    full_support_cells = int(np.sum(target_valid & (support == present)))
    support_fracs = frac[cells]
    support_mean = float(np.nanmean(support_fracs)) if support_fracs.size else 0.0
    hist: dict[str, int] = {}
    if support_fracs.size:
        idx = np.digitize(support_fracs, _SUPPORT_BINS) - 1
        counts = np.bincount(idx, minlength=len(_BUCKET_LABELS))
        for label, count in zip(_BUCKET_LABELS, counts, strict=False):
            if count:
                hist[label] = int(count)

    metrics = SceneMetrics(
        scene_id=scene.scene_id,
        year=scene.year,
        s2_scene_id=scene.s2_scene_id,
        geometry_id=scene.geometry_id,
        assessed=True,
        exclusion_reason=None,
        target_valid_cells=target_valid_cells,
        all_100_cells=all_100_cells,
        full_support_cells=full_support_cells,
        support_mean_frac=round(support_mean, 6),
        support_histogram=hist,
        errors=errors,
    )
    findings.extend(f"{scene.scene_id}: {e}" for e in errors)
    return metrics, layer_invalid, findings


# ── report serialisation ─────────────────────────────────────────────


def _summary_dict(report: Stage1Report) -> dict:
    return {
        "pipeline": "qa-stage1-raw",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "ok": report.ok,
        "grid": report.grid,
        "inputs": report.inputs,
        "fingerprints": report.fingerprints,
        "scenes": {
            "total_pairings": report.total_pairings,
            "assessed": report.assessed,
            "excluded": report.excluded,
            "exclusion_reasons": report.exclusion_reasons,
        },
        "aggregate": report.aggregate,
        "layers": report.layers,
        "findings": report.findings,
    }


def write_report(report: Stage1Report, output_root: str) -> dict[str, str]:
    """Persist summary.json, scenes.parquet, scenes.csv under the run prefix."""
    prefix = f"{output_root.rstrip('/')}/{report.run_id}"
    uris: dict[str, str] = {}

    summary_uri = f"{prefix}/summary.json"
    atomic_write(summary_uri, json_dumps(_summary_dict(report)), overwrite=True,
                 if_generation_match=0)
    uris["summary"] = summary_uri

    rows = [
        {
            "scene_id": s.scene_id,
            "year": s.year,
            "s2_scene_id": s.s2_scene_id,
            "geometry_id": s.geometry_id,
            "assessed": s.assessed,
            "exclusion_reason": s.exclusion_reason,
            "target_valid_cells": s.target_valid_cells,
            "all_100_cells": s.all_100_cells,
            "full_support_cells": s.full_support_cells,
            "support_mean_frac": s.support_mean_frac,
            "support_histogram": json_dumps(s.support_histogram),
            "errors": json_dumps(s.errors),
        }
        for s in report.scenes
    ]
    table = pa.Table.from_pylist(rows)
    parquet_uri = f"{prefix}/scenes.parquet"
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    atomic_write(parquet_uri, buf.getvalue().to_pybytes(), overwrite=True,
                 if_generation_match=0)
    uris["scenes_parquet"] = parquet_uri

    csv_uri = f"{prefix}/scenes.csv"
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "scene_id",
            "year",
            "s2_scene_id",
            "geometry_id",
            "assessed",
            "exclusion_reason",
            "target_valid_cells",
            "all_100_cells",
            "full_support_cells",
            "support_mean_frac",
            "support_histogram",
            "errors",
        ]
    )
    csv_buf = _io.StringIO()
    writer = _csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(csv_uri, csv_buf.getvalue().encode("utf-8"), overwrite=True,
                 if_generation_match=0)
    uris["scenes_csv"] = csv_uri
    return uris


def json_dumps(data) -> str:
    import json

    return json.dumps(data, indent=2, default=str)


# ── orchestration ────────────────────────────────────────────────────


def run_stage1_raw(cfg, *, run_id: str) -> Stage1Report:
    """Run the Stage-1 raw-input QA gate for the configured training universe.

    ``cfg`` carries: ``manifest_uri``, ``ard_root``, ``static_sources_root``,
    ``static_derived_root``, ``dynamic_root``, ``geometry_mapping_uri``,
    ``output_root`` (QA evidence root), optional ``bbox`` (smoke subset)
    and ``scene_ids`` (smoke restriction).
    """
    timestamp = datetime.now(UTC).isoformat()

    manifest_uri = str(cfg.manifest_uri)
    ard_root = str(cfg.ard_root)
    static_sources_root = str(cfg.static_sources_root)
    static_derived_root = str(cfg.static_derived_root)
    dynamic_root = str(cfg.dynamic_root)
    geometry_mapping_uri = str(cfg.geometry_mapping_uri)
    bbox = tuple(cfg.get("bbox", None)) if cfg.get("bbox") else None
    scene_ids = [str(s) for s in cfg.get("scene_ids", []) or []]

    log_event(_logger, logging.INFO, "stage1_start", run_id=run_id, manifest=manifest_uri)

    inventory = build_inventory(
        manifest_uri=manifest_uri,
        ard_root=ard_root,
        static_sources_root=static_sources_root,
        static_derived_root=static_derived_root,
        dynamic_root=dynamic_root,
        geometry_mapping_uri=geometry_mapping_uri,
        scene_ids=scene_ids,
    )
    if not inventory.ok:
        log_event(_logger, logging.ERROR, "stage1_inventory_failed", errors=inventory.errors)
        # Fail closed: a broken inventory cannot produce trustworthy QA.
        raise RuntimeError(f"Stage-1 inventory failed: {inventory.errors}")

    analysis_10 = analysis_grid_10m(bbox)
    analysis_100 = analysis_10.zoom_out(10)

    # ERA5 per-band ranges (declared in the dynamic contract).
    era5_ranges = [
        (spec.valid_range[0], spec.valid_range[1])
        for spec in contract_for_era5_scene().output_bands
        if spec.valid_range is not None
    ]
    if len(era5_ranges) != 8:
        raise RuntimeError(f"ERA5 contract must define 8 band ranges, got {len(era5_ranges)}")

    # ── static source metadata checks (once per run, vintage-fixed) ────
    static_source_findings: list[str] = []
    for key, cog_uri in sorted(inventory.static_sources.items()):
        for err in _check_metadata_10m(cog_uri, n_bands=None, dtype="float32"):
            static_source_findings.append(f"static source {key}: {err}")

    findings: list[str] = list(static_source_findings)
    scenes_out: list[SceneMetrics] = []
    agg_target = agg_all100 = agg_full = 0
    agg_hist: dict[str, int] = {}
    agg_layers: dict[str, int] = {}
    layer_scenes: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="qa-stage1-static-") as tmp_str:
        tmp_dir = Path(tmp_str)
        static_cache: dict[str, str] = {}

        for scene in inventory.scenes:
            if not scene.assessable:
                reason = scene.exclusion_reason or "excluded"
                if reason not in _EXPECTED_EXCLUSIONS:
                    findings.append(f"{scene.scene_id}: {reason}")
                scenes_out.append(
                    SceneMetrics(
                        scene_id=scene.scene_id,
                        year=scene.year,
                        s2_scene_id=scene.s2_scene_id,
                        geometry_id=scene.geometry_id,
                        assessed=False,
                        exclusion_reason=reason,
                    )
                )
                continue

            # Resolve static masks for this scene's geometry profile.
            static_masks: dict[str, str] = {}
            profile_key = scene.geometry_id
            for product, cog_uri in sorted(scene.static_derived.items()):
                key = f"{product}/{profile_key}"
                if key not in static_cache:
                    mask_path, mask_findings = _static_validity_mask(
                        product, cog_uri, profile_key, tmp_dir
                    )
                    findings.extend(mask_findings)
                    static_cache[key] = mask_path
                if static_cache[key]:
                    static_masks[product] = static_cache[key]

            metrics, layer_invalid, scene_findings = _scan_scene(
                scene, analysis_10, analysis_100, static_masks, era5_ranges
            )
            findings.extend(scene_findings)
            # Surface precise inventory-level diagnosis (e.g. missing
            # COG/flag path in the ARD ledger) alongside the scan findings.
            if scene.errors:
                findings.extend(f"{scene.scene_id}: {e}" for e in scene.errors)
            scenes_out.append(metrics)

            if metrics.assessed:
                agg_target += metrics.target_valid_cells
                agg_all100 += metrics.all_100_cells
                agg_full += metrics.full_support_cells
                for label, count in metrics.support_histogram.items():
                    agg_hist[label] = agg_hist.get(label, 0) + count
                for layer, count in layer_invalid.items():
                    agg_layers[layer] = agg_layers.get(layer, 0) + count
                    layer_scenes[layer] = layer_scenes.get(layer, 0) + 1

    # ── aggregate layer fractions ──────────────────────────────────────
    # ``invalid_px`` is the sum over all scanned scenes of the invalid
    # 10 m pixels of that layer, each scene scanned once over the analysis
    # grid. The fraction is normalized by the number of scenes where the
    # layer was actually scanned (optional morphology layers such as
    # vegetation_dsm exist only for the 2024 vintage), so it stays in
    # [0, 1] and is comparable across layers.
    assessed_scenes = [s for s in scenes_out if s.assessed]
    total_10m_px = analysis_10.shape.y * analysis_10.shape.x
    layers: dict[str, dict] = {}
    for layer, count in sorted(agg_layers.items()):
        scanned = max(layer_scenes.get(layer, 0), 1)
        layers[layer] = {
            "invalid_px": count,
            "invalid_frac": round(count / (scanned * total_10m_px), 6),
            "scanned_scenes": layer_scenes.get(layer, 0),
        }

    aggregate = {
        "analysis_grid": {
            "crs": _GRID_CRS,
            "res_10m": {"width": analysis_10.shape.x, "height": analysis_10.shape.y},
            "res_100m": {"width": analysis_100.shape.x, "height": analysis_100.shape.y},
            "origin": [analysis_10.transform.xoff, analysis_10.transform.yoff],
            "bbox_subset": bbox is not None,
        },
        "assessed_scenes": len(assessed_scenes),
        "target_valid_cells": agg_target,
        "all_100_cells": agg_all100,
        "full_support_cells": agg_full,
        "support_histogram": dict(sorted(agg_hist.items())),
    }

    report = Stage1Report(
        run_id=run_id,
        timestamp=timestamp,
        grid={
            "crs": _GRID_CRS,
            "res_10m": [analysis_10.shape.x, analysis_10.shape.y],
            "res_100m": [analysis_100.shape.x, analysis_100.shape.y],
            "origin": [analysis_10.transform.xoff, analysis_10.transform.yoff],
        },
        inputs={
            "manifest_uri": manifest_uri,
            "ard_root": ard_root,
            "static_sources_root": static_sources_root,
            "static_derived_root": static_derived_root,
            "dynamic_root": dynamic_root,
            "geometry_mapping_uri": geometry_mapping_uri,
        },
        fingerprints=inventory.fingerprints,
        total_pairings=inventory.total_pairings,
        assessed=inventory.assessed,
        excluded=inventory.excluded,
        exclusion_reasons=inventory.exclusion_reasons,
        aggregate=aggregate,
        layers=layers,
        findings=sorted(set(findings)),
        scenes=scenes_out,
    )

    log_event(
        _logger,
        logging.INFO,
        "stage1_done",
        run_id=run_id,
        assessed=inventory.assessed,
        excluded=inventory.excluded,
        findings=len(report.findings),
        ok=report.ok,
    )
    return report


__all__ = [
    "SceneMetrics",
    "Stage1Report",
    "analysis_grid_10m",
    "run_stage1_raw",
    "write_report",
]
