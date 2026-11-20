"""Berlin-wide 10 m inference for a frozen v6 U-Net.

The scene prior, the model-boundary affine, and the residual composition are
the ones the training reader and ``RealLSTTask.forward`` already use. Tiles
are size-agnostic: ``forward`` refuses anything but a 160×160 target, so this
path calls ``task.model`` and adds ``reconstruct_prior_kelvin`` itself.

Prior modes
-----------
``native100`` (default) is the inference regime in
``docs/pseudo-pair-tensor-contract.md``: each 10 m pixel takes its
native-valid 100 m Landsat cell. ``block1000`` is the training regime: the
global-lattice 1000 m block means of :func:`global_block_means`, expanded.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
from affine import Affine
from odc.geo.geobox import GeoBox
from rasterio.windows import Window
from torch import Tensor

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic
from berlin_lst_downscaling.data.features.paths import feature_cog, feature_mask_cog, ledger_path
from berlin_lst_downscaling.data.io import exists, log_event, read_bytes
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger
from berlin_lst_downscaling.data.training.contracts import CANON_GRID_ORIGIN_X, CANON_GRID_ORIGIN_Y
from berlin_lst_downscaling.data.training.paths import patch_index_parquet, scaler_json
from berlin_lst_downscaling.modeling.aggregation import AGGREGATION_ARITHMETIC
from berlin_lst_downscaling.modeling.channels import ActiveChannelSelection
from berlin_lst_downscaling.modeling.contracts import PRIOR_BLOCK_CELLS, REAL_PATCH_PX
from berlin_lst_downscaling.modeling.metrics import (
    clamp_eval_prediction,
    masked_abs_error_sums,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    ScalerSpec,
    _canonical_offset,
    global_block_means,
    load_landsat_uris,
    native_landsat_valid,
    prior_model_channel,
    scale_and_fill_features,
)
from berlin_lst_downscaling.modeling.real_task import reconstruct_prior_kelvin

_logger = logging.getLogger(__name__)

PRIOR_NATIVE100 = "native100"
PRIOR_BLOCK1000 = "block1000"
PRIOR_MODES: tuple[str, ...] = (PRIOR_NATIVE100, PRIOR_BLOCK1000)
PRIOR_PLACEHOLDER_K = 275.0
INFERENCE_CONFIGS: tuple[str, ...] = ("v6_l1_locked", "v6_l4_locked")
REPRODUCTION_RELATIVE_TOLERANCE = 1e-4
_FEATURE_SOURCE = "feature_stack"
_OFFSET_TOL_PX = 1e-6
_LATTICE_PX = 10
_SEED_IN_NAME = re.compile(r"(?:^|[-_])seed(\d+)(?:[-_]|$)")


def _lst_contract(height: int, width: int) -> Contract:
    """Float32 Kelvin COG. Block size stays 512 except when the raster is smaller.

    A 160 px patch window cannot hold a 512 px tile. GDAL rejects a block
    larger than the image, so the block shrinks to the largest multiple of
    16 that fits. Berlin scenes keep 512.
    """
    block = 512
    if height < block or width < block:
        fitted = min(height, width)
        if fitted >= 16:
            fitted = min(fitted, max(16, (fitted // 16) * 16))
        block = max(1, fitted)
    return Contract(
        source="lst10m-inference",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="lst",
                dtype="float32",
                nodata=float("nan"),
                description="10 m LST Kelvin",
                unit="K",
            ),
        ),
        tiling=TilingSpec(blocksize=block, compress="deflate", predictor=3),
        schema_version=1,
        flag_mode="none",
    )


@dataclass(frozen=True)
class TileWindow:
    """One inference tile and the scene pixels it alone is allowed to write."""

    row0: int
    col0: int
    out_row0: int
    out_row1: int
    out_col0: int
    out_col1: int


@dataclass(frozen=True)
class InferenceProvenance:
    """Identity recorded next to a scene GeoTIFF. Paths are the ones read."""

    checkpoint_uri: str
    checkpoint_sha256: str
    config_name: str
    arm: str
    seed: int
    code_commit: str
    scaler_sha256: str


def _is_epsg25833(crs: object) -> bool:
    if crs is None:
        return False
    to_epsg = getattr(crs, "to_epsg", None)
    if callable(to_epsg) and to_epsg() == 25833:
        return True
    return str(crs).upper() == "EPSG:25833"


def _exact_canonical_offset(transform: Affine, res: float) -> tuple[int, int]:
    """Return the global canonical index of pixel ``(0, 0)``.

    A fractional offset raises. Nothing is rounded onto the lattice.
    """
    col_f = (transform.xoff - CANON_GRID_ORIGIN_X) / res
    row_f = (CANON_GRID_ORIGIN_Y - transform.yoff) / res
    col0 = round(col_f)
    row0 = round(row_f)
    if abs(col_f - col0) > _OFFSET_TOL_PX or abs(row_f - row0) > _OFFSET_TOL_PX:
        raise ValueError(
            f"transform origin ({transform.xoff}, {transform.yoff}) is not an integer "
            f"offset of the canonical {res:g} m lattice (column {col_f}, row {row_f})"
        )
    return int(col0), int(row0)


def _assert_north_up_resolution(transform: Affine, res: float, *, label: str) -> None:
    if not np.isclose(transform.a, res) or not np.isclose(abs(transform.e), res):
        raise ValueError(
            f"{label}: resolution is ({transform.a}, {transform.e}), expected {res:g} m"
        )
    if transform.e > 0:
        raise ValueError(f"{label}: transform is south-up")
    if transform.b or transform.d:
        raise ValueError(f"{label}: transform is rotated")


def _assert_grid_10m(grid: GeoBox) -> tuple[int, int]:
    if not _is_epsg25833(grid.crs):
        raise ValueError(f"10 m grid CRS is {grid.crs!r}, expected EPSG:25833")
    _assert_north_up_resolution(grid.transform, 10.0, label="10 m grid")
    return _exact_canonical_offset(grid.transform, 10.0)


def _lookup(table: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample ``table`` at broadcast ``rows`` (H,) and ``cols`` (W,). Outside is NaN."""
    height = int(rows.shape[0])
    width = int(cols.shape[0])
    out = np.full((height, width), np.nan, dtype=np.float32)
    if table.size == 0 or height == 0 or width == 0:
        return out
    valid_r = (rows >= 0) & (rows < table.shape[0])
    valid_c = (cols >= 0) & (cols < table.shape[1])
    if not bool(valid_r.any()) or not bool(valid_c.any()):
        return out
    safe_r = np.clip(rows, 0, table.shape[0] - 1).astype(np.int64, copy=False)
    safe_c = np.clip(cols, 0, table.shape[1] - 1).astype(np.int64, copy=False)
    sampled = np.asarray(table, dtype=np.float32)[safe_r[:, None], safe_c[None, :]]
    inside = valid_r[:, None] & valid_c[None, :]
    out[inside] = sampled[inside]
    return out


def build_prior_10m(
    st_100m: np.ndarray,
    flag_100m: np.ndarray,
    transform_100m: Affine,
    grid_10m: GeoBox,
    mode: str = PRIOR_NATIVE100,
) -> np.ndarray:
    """Return the physical Kelvin prior on ``grid_10m``.

    ``native100`` copies each native-valid 100 m cell onto its 10 m pixels and
    leaves the rest NaN. ``block1000`` expands the global-lattice 1000 m block
    means; a block with no native-valid cell is NaN. The 100 m raster and the
    10 m grid must be EPSG:25833, north-up, and on the canonical lattice. The
    100 m CRS is not an argument: callers check the dataset CRS, and this
    function checks ``grid_10m`` plus the 100 m transform.
    """
    if mode not in PRIOR_MODES:
        raise ValueError(f"prior mode must be one of {PRIOR_MODES}, got {mode!r}")
    if st_100m.ndim != 2 or flag_100m.ndim != 2:
        raise ValueError(
            f"Landsat arrays must be 2-D, got LST {st_100m.shape} and flag {flag_100m.shape}"
        )
    valid = native_landsat_valid(st_100m, flag_100m)
    _assert_north_up_resolution(transform_100m, 100.0, label="100 m Landsat")
    col0_100, row0_100 = _exact_canonical_offset(transform_100m, 100.0)
    col0_10, row0_10 = _assert_grid_10m(grid_10m)
    height = int(grid_10m.shape[0])
    width = int(grid_10m.shape[1])
    cell_r = np.floor_divide(row0_10 + np.arange(height, dtype=np.int64), _LATTICE_PX)
    cell_c = np.floor_divide(col0_10 + np.arange(width, dtype=np.int64), _LATTICE_PX)
    if mode == PRIOR_NATIVE100:
        source = np.where(valid, st_100m, np.nan).astype(np.float32)
        return _lookup(source, cell_r - row0_100, cell_c - col0_100)
    block_means, block_row0, block_col0 = global_block_means(
        np.asarray(st_100m, dtype=np.float32),
        valid,
        col0=col0_100,
        row0=row0_100,
        method=AGGREGATION_ARITHMETIC,
    )
    block_r = np.floor_divide(cell_r, PRIOR_BLOCK_CELLS) - block_row0
    block_c = np.floor_divide(cell_c, PRIOR_BLOCK_CELLS) - block_col0
    return _lookup(block_means, block_r, block_c)


def _axis_origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    need = length - tile
    last = ((need + _LATTICE_PX - 1) // _LATTICE_PX) * _LATTICE_PX
    origins = list(range(0, last, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _axis_spans(length: int, tile: int, stride: int) -> list[tuple[int, int, int]]:
    """Return ``(origin, out_start, out_end)`` covering ``[0, length)`` once.

    The nominal centre is ``[origin + margin, origin + margin + stride)``.
    The first tile also owns the leading margin and the last tile the trailing
    margin. Where two centres both contain a pixel, the earlier tile keeps it
    through the end of its own centre.
    """
    margin = (tile - stride) // 2
    origins = _axis_origins(length, tile, stride)
    if len(origins) == 1:
        return [(origins[0], 0, length)]
    spans: list[list[int]] = []
    last_i = len(origins) - 1
    for i, origin in enumerate(origins):
        centre_lo = origin + margin
        centre_hi = origin + margin + stride
        lo = 0 if i == 0 else centre_lo
        hi = length if i == last_i else centre_hi
        spans.append([origin, lo, hi])
    for i in range(last_i):
        if spans[i][2] <= spans[i + 1][1]:
            if spans[i][2] < spans[i + 1][1]:
                spans[i][2] = spans[i + 1][1]
            continue
        boundary = spans[i][0] + margin + stride
        boundary = min(max(boundary, spans[i][1]), spans[i + 1][2])
        spans[i][2] = boundary
        spans[i + 1][1] = boundary
    if spans[0][1] != 0 or spans[-1][2] != length:
        raise RuntimeError("tile plan does not cover the axis")
    for left, right in zip(spans, spans[1:], strict=False):
        if left[2] != right[1]:
            raise RuntimeError("tile plan leaves a gap or an overlap")
    return [(origin, lo, hi) for origin, lo, hi in spans]


def plan_tiles(height: int, width: int, tile_px: int, stride_px: int) -> list[TileWindow]:
    """Plan overlapping tiles whose output regions partition the scene.

    Tile size is divisible by 16 (the U-Net depth) and by 10. Stride is a
    positive multiple of 10 no larger than the tile, and ``tile - stride`` is
    even so the centre is a whole pixel. Every origin is a multiple of 10 px.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"scene shape {(height, width)} is empty")
    if tile_px % 16 != 0:
        raise ValueError(f"tile_px must be divisible by 16, got {tile_px}")
    if tile_px % _LATTICE_PX != 0 or stride_px % _LATTICE_PX != 0:
        raise ValueError(
            f"tile_px and stride_px must be multiples of {_LATTICE_PX} px "
            f"(the 100 m lattice), got {tile_px} and {stride_px}"
        )
    if stride_px <= 0 or stride_px > tile_px or (tile_px - stride_px) % 2 != 0:
        raise ValueError(
            f"stride_px must be in (0, tile_px] and tile_px - stride_px even, "
            f"got tile_px={tile_px} stride_px={stride_px}"
        )
    rows = _axis_spans(height, tile_px, stride_px)
    cols = _axis_spans(width, tile_px, stride_px)
    tiles = [
        TileWindow(row0, col0, row_lo, row_hi, col_lo, col_hi)
        for row0, row_lo, row_hi in rows
        for col0, col_lo, col_hi in cols
    ]
    for tile in tiles:
        if tile.row0 % _LATTICE_PX or tile.col0 % _LATTICE_PX:
            raise RuntimeError(f"tile origin {(tile.row0, tile.col0)} is off the 100 m lattice")
    cover = np.zeros((height, width), dtype=np.int16)
    for tile in tiles:
        cover[tile.out_row0 : tile.out_row1, tile.out_col0 : tile.out_col1] += 1
    if tiles and (int(cover.min()) != 1 or int(cover.max()) != 1):
        raise RuntimeError("tile plan does not cover every pixel exactly once")
    return tiles


def _pad_spatial(array: np.ndarray, tile: int, fill: float) -> np.ndarray:
    if array.shape[-2] == tile and array.shape[-1] == tile:
        return array
    if array.ndim == 2:
        out = np.full((tile, tile), fill, dtype=np.float32)
        out[: array.shape[0], : array.shape[1]] = array
        return out
    out = np.full((array.shape[0], tile, tile), fill, dtype=np.float32)
    out[:, : array.shape[-2], : array.shape[-1]] = array
    return out


def _prepare_tile(
    features: np.ndarray,
    prior: np.ndarray,
    feature_valid: np.ndarray,
    tile: TileWindow,
    tile_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(model_input, valid, scene_rows, scene_cols)`` or None to skip.

    The returned arrays are the padded tile. ``valid`` is the output-valid
    mask in tile coordinates (False on padding).
    """
    row1 = min(tile.row0 + tile_px, prior.shape[0])
    col1 = min(tile.col0 + tile_px, prior.shape[1])
    prior_crop = prior[tile.row0 : row1, tile.col0 : col1]
    valid_crop = (feature_valid[tile.row0 : row1, tile.col0 : col1] == 1) & np.isfinite(prior_crop)
    if not bool(valid_crop.any()):
        return None
    finite = prior_crop[np.isfinite(prior_crop)]
    fill = float(finite.mean()) if finite.size else PRIOR_PLACEHOLDER_K
    filled = np.array(prior_crop, dtype=np.float32, copy=True)
    missing = ~np.isfinite(filled)
    if bool(missing.any()):
        filled[missing] = np.float32(fill)
    feat_crop = features[:, tile.row0 : row1, tile.col0 : col1]
    feat_pad = _pad_spatial(np.ascontiguousarray(feat_crop, dtype=np.float32), tile_px, 0.0)
    prior_pad = _pad_spatial(filled, tile_px, fill)
    valid_pad = _pad_spatial(valid_crop.astype(np.float32), tile_px, 0.0) > 0
    prior_ch = prior_model_channel(prior_pad)
    model_input = np.concatenate([feat_pad, prior_ch[None]], axis=0)
    return model_input, valid_pad, prior_pad


def predict_scene(
    task: Any,
    features_scaled_selected: np.ndarray,
    prior_k: np.ndarray,
    *,
    feature_valid: np.ndarray,
    tile_px: int = 160,
    stride_px: int = 80,
    device: torch.device | str,
    batch_size: int,
) -> np.ndarray:
    """Predict 10 m LST for one scene window.

    Input tiles are ``cat([selected scaled features, prior_model_channel])``
    through ``task.model``. When ``task.residual_prior`` is set, the Kelvin
    prior is added back with :func:`reconstruct_prior_kelvin`, the same
    composition as ``RealLSTTask.forward``. A NaN prior is a finite
    placeholder inside the tile (the mean of that tile's finite prior, or
    275 K) and the output pixel is NaN. An output pixel is valid only where
    ``feature_valid == 1`` and the prior is finite. A non-finite model value
    on a valid pixel raises.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if features_scaled_selected.ndim != 3:
        raise ValueError(f"features must be (C, H, W), got {features_scaled_selected.shape}")
    channels, height, width = features_scaled_selected.shape
    if prior_k.shape != (height, width) or feature_valid.shape != (height, width):
        raise ValueError(
            f"prior {prior_k.shape} and feature_valid {feature_valid.shape} "
            f"must match the feature grid {(height, width)}"
        )
    n_active = int(getattr(task, "n_active_channels", channels))
    if channels != n_active:
        raise ValueError(f"selected {channels} channels, task expects {n_active}")
    tiles = plan_tiles(height, width, tile_px, stride_px)
    device_t = torch.device(device)
    task.eval()
    task.to(device_t)
    output = np.full((height, width), np.nan, dtype=np.float32)
    pending: list[tuple[TileWindow, np.ndarray, np.ndarray]] = []

    def _flush() -> None:
        if not pending:
            return
        inputs = np.stack([item[1] for item in pending])
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs).to(device_t)
            delta = task.model(tensor)
            if task.residual_prior:
                pred = reconstruct_prior_kelvin(tensor[:, -1:]).float() + delta.float()
            else:
                pred = delta.float()
            pred_np = pred.detach().cpu().numpy()
        if pred_np.shape != (len(pending), 1, tile_px, tile_px):
            raise RuntimeError(
                f"model output shape {tuple(pred_np.shape)} != "
                f"({len(pending)}, 1, {tile_px}, {tile_px})"
            )
        for (window, _inputs, valid), tile_pred in zip(pending, pred_np[:, 0], strict=True):
            scene_h = min(tile_px, height - window.row0)
            scene_w = min(tile_px, width - window.col0)
            scene_pred = tile_pred[:scene_h, :scene_w]
            scene_valid = valid[:scene_h, :scene_w]
            if bool(scene_valid.any()) and not bool(np.isfinite(scene_pred[scene_valid]).all()):
                raise RuntimeError("non-finite model output on a valid pixel")
            row_lo = window.out_row0 - window.row0
            row_hi = window.out_row1 - window.row0
            col_lo = window.out_col0 - window.col0
            col_hi = window.out_col1 - window.col0
            owned_pred = scene_pred[row_lo:row_hi, col_lo:col_hi]
            owned_valid = scene_valid[row_lo:row_hi, col_lo:col_hi]
            written = np.full(owned_pred.shape, np.nan, dtype=np.float32)
            written[owned_valid] = owned_pred[owned_valid]
            output[window.out_row0 : window.out_row1, window.out_col0 : window.out_col1] = written
        pending.clear()

    for window in tiles:
        prepared = _prepare_tile(features_scaled_selected, prior_k, feature_valid, window, tile_px)
        if prepared is None:
            continue
        model_input, valid, _prior_pad = prepared
        pending.append((window, model_input, valid))
        if len(pending) == batch_size:
            _flush()
    _flush()
    return output


def _require_file(uri: str) -> None:
    if not exists(uri):
        raise FileNotFoundError(uri)


def _assert_feature_grid(transform: Affine, *, uri: str, count: int, expected_bands: int) -> None:
    _assert_north_up_resolution(transform, 10.0, label=uri)
    rounded = _canonical_offset(transform, 10.0)
    expected = _canonical_offset(canon_grid_10m().transform, 10.0)
    if rounded != expected:
        raise RuntimeError(
            f"{uri} is not aligned with the analysis grid (offset {rounded} != {expected})"
        )
    _exact_canonical_offset(transform, 10.0)
    if count != expected_bands:
        raise RuntimeError(f"{uri}: feature stack has {count} bands, expected {expected_bands}")


def _read_scene_inputs(
    features_uri: str,
    feature_valid_uri: str,
    *,
    expected_bands: int,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Read the feature window and the matching ``feature_valid`` window."""
    _require_file(features_uri)
    _require_file(feature_valid_uri)
    with rasterio.open(features_uri) as src, rasterio.open(feature_valid_uri) as mask_src:
        if not _is_epsg25833(src.crs):
            raise RuntimeError(f"{features_uri}: CRS is {src.crs!r}, expected EPSG:25833")
        if not _is_epsg25833(mask_src.crs):
            raise RuntimeError(f"{feature_valid_uri}: CRS is {mask_src.crs!r}, expected EPSG:25833")
        _assert_feature_grid(
            src.transform, uri=features_uri, count=src.count, expected_bands=expected_bands
        )
        _assert_north_up_resolution(mask_src.transform, 10.0, label=feature_valid_uri)
        feature_off = _exact_canonical_offset(src.transform, 10.0)
        if _exact_canonical_offset(mask_src.transform, 10.0) != feature_off:
            raise RuntimeError(f"{feature_valid_uri}: grid differs from the feature stack")
        if (mask_src.height, mask_src.width) != (src.height, src.width):
            raise RuntimeError(
                f"{feature_valid_uri}: shape {(mask_src.height, mask_src.width)} != "
                f"feature stack {(src.height, src.width)}"
            )
        if mask_src.count != 1:
            raise RuntimeError(f"{feature_valid_uri}: expected 1 band, got {mask_src.count}")
        col0_10, row0_10 = feature_off
        if bbox is None:
            col0, row0, width, height = 0, 0, src.width, src.height
        else:
            col0, row0, width, height = bbox
            if any(
                value % _LATTICE_PX != 0
                for value in (col0_10 + col0, row0_10 + row0, width, height)
            ):
                raise ValueError(
                    "bbox must be a canonical 10 m window aligned to the 100 m lattice "
                    f"(col0={col0}, row0={row0}, width={width}, height={height})"
                )
        if (
            col0 < 0
            or row0 < 0
            or width <= 0
            or height <= 0
            or col0 + width > src.width
            or row0 + height > src.height
        ):
            raise RuntimeError(
                f"feature window ({row0}, {col0}, {height}, {width}) is outside "
                f"{features_uri} ({src.height}, {src.width})"
            )
        window = Window(col0, row0, width, height)  # type: ignore[call-arg]
        features = src.read(window=window).astype(np.float32)
        mask = mask_src.read(1, window=window)
        transform = src.window_transform(window)
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise RuntimeError(f"{feature_valid_uri}: values outside {{0, 1}}: {sorted(unique)}")
    return features, mask, transform


def _read_landsat_scene(
    landsat_cog: str, landsat_flag: str
) -> tuple[np.ndarray, np.ndarray, Affine]:
    _require_file(landsat_cog)
    _require_file(landsat_flag)
    with rasterio.open(landsat_cog) as src:
        if not _is_epsg25833(src.crs):
            raise RuntimeError(f"{landsat_cog}: CRS is {src.crs!r}, expected EPSG:25833")
        _assert_north_up_resolution(src.transform, 100.0, label=landsat_cog)
        _exact_canonical_offset(src.transform, 100.0)
        st = src.read(1).astype(np.float32)
        transform = src.transform
        shape = (src.height, src.width)
        offset = _canonical_offset(src.transform, 100.0)
    with rasterio.open(landsat_flag) as src:
        if (src.height, src.width) != shape:
            raise RuntimeError(f"{landsat_flag}: shape differs from the LST COG")
        if _canonical_offset(src.transform, 100.0) != offset:
            raise RuntimeError(f"{landsat_flag}: grid differs from the LST COG")
        if not _is_epsg25833(src.crs):
            raise RuntimeError(f"{landsat_flag}: CRS is {src.crs!r}, expected EPSG:25833")
        flag = src.read(1)
    return st, flag, transform


def _complete_cell_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic 10×10 means and the mask of cells whose 100 pixels are finite."""
    height = values.shape[0] // _LATTICE_PX * _LATTICE_PX
    width = values.shape[1] // _LATTICE_PX * _LATTICE_PX
    if height == 0 or width == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, np.zeros((0, 0), dtype=bool)
    tiles = values[:height, :width].reshape(
        height // _LATTICE_PX, _LATTICE_PX, width // _LATTICE_PX, _LATTICE_PX
    )
    tiles = np.moveaxis(tiles, 1, 2).reshape(-1, _LATTICE_PX * _LATTICE_PX)
    ok = np.isfinite(tiles).all(axis=1)
    means = np.full(tiles.shape[0], np.nan, dtype=np.float64)
    if bool(ok.any()):
        means[ok] = tiles[ok].mean(axis=1)
    shape = (height // _LATTICE_PX, width // _LATTICE_PX)
    return means.reshape(shape).astype(np.float32), ok.reshape(shape)


def _mae_bias(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = mask & np.isfinite(prediction) & np.isfinite(reference)
    count = int(selected.sum())
    if count == 0:
        return {"mae": None, "bias": None, "count": 0}
    diff = prediction[selected].astype(np.float64) - reference[selected].astype(np.float64)
    return {
        "mae": float(np.abs(diff).mean()),
        "bias": float(diff.mean()),
        "count": count,
    }


def _native_reference(
    st: np.ndarray,
    flag: np.ndarray,
    transform_100m: Affine,
    transform_10m: Affine,
    shape_10m: tuple[int, ...],
) -> np.ndarray:
    """Native-valid 100 m values for each complete 10×10 cell of the 10 m window."""
    col0_10, row0_10 = _exact_canonical_offset(transform_10m, 10.0)
    col0_100, row0_100 = _exact_canonical_offset(transform_100m, 100.0)
    n_rows = shape_10m[0] // _LATTICE_PX
    n_cols = shape_10m[1] // _LATTICE_PX
    reference = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    if n_rows == 0 or n_cols == 0:
        return reference
    valid = native_landsat_valid(st, flag)
    rows = np.arange(n_rows, dtype=np.int64) + (row0_10 // _LATTICE_PX) - row0_100
    cols = np.arange(n_cols, dtype=np.int64) + (col0_10 // _LATTICE_PX) - col0_100
    source = np.where(valid, st, np.nan).astype(np.float32)
    return _lookup(source, rows, cols)


def scale_check_report(
    prediction: np.ndarray,
    prior: np.ndarray,
    st_100m: np.ndarray,
    flag_100m: np.ndarray,
    transform_100m: Affine,
    transform_10m: Affine,
) -> dict[str, object]:
    """MAE and bias of 10×10 means on cells whose 100 pixels are all output-valid."""
    pooled, complete = _complete_cell_means(prediction)
    prior_pooled, _prior_complete = _complete_cell_means(prior)
    native = _native_reference(st_100m, flag_100m, transform_100m, transform_10m, prediction.shape)
    if native.shape != complete.shape:
        raise RuntimeError(
            f"native reference shape {native.shape} != pooled shape {complete.shape}"
        )
    both = np.isfinite(prediction) & np.isfinite(prior)
    mean_abs = float(np.abs(prediction[both] - prior[both]).mean()) if bool(both.any()) else None
    return {
        "mean_abs_output_minus_prior_10m": mean_abs,
        "native_100m": _mae_bias(pooled, native, complete),
        "prior_100m": _mae_bias(pooled, prior_pooled, complete),
    }


def write_lst_cog(array: np.ndarray, transform: Affine, dst: str) -> str:
    """Write one float32 Kelvin COG (NaN nodata, EPSG:25833, tiled deflate)."""
    import rioxarray  # noqa: F401 — registers the rio accessor
    import xarray as xr

    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"LST array must be 2-D, got {values.shape}")
    data = xr.DataArray(values, dims=("y", "x"), name="lst")
    data.rio.write_crs("EPSG:25833", inplace=True)
    data.rio.write_transform(transform, inplace=True)
    contract = _lst_contract(int(values.shape[0]), int(values.shape[1]))
    return write_cog_atomic(xr.Dataset({"lst": data}), dst, contract, overwrite=True)


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    if str(path).startswith("gs://"):
        from berlin_lst_downscaling.data.io import atomic_write

        atomic_write(path, body, overwrite=True)
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)


def scene_product_paths(out_dir: str, scene_id: str) -> tuple[str, str]:
    """Return ``(<scene>.lst10m.tif, <scene>.inference.json)``."""
    root = f"{out_dir.rstrip('/')}/{scene_id}"
    return f"{root}/{scene_id}.lst10m.tif", f"{root}/{scene_id}.inference.json"


def run_scene(
    task: Any,
    *,
    scene_id: str,
    features_uri: str,
    feature_valid_uri: str,
    landsat_cog: str,
    landsat_flag: str,
    scaler: ScalerSpec,
    selection: ActiveChannelSelection,
    out_dir: str,
    provenance: InferenceProvenance,
    prior_mode: str = PRIOR_NATIVE100,
    bbox: tuple[int, int, int, int] | None = None,
    tile_px: int = 160,
    stride_px: int = 80,
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    mask_native_invalid: bool = False,
) -> dict[str, object]:
    """Read one scene, predict the 10 m LST window, and write the COG plus JSON.

    ``mask_native_invalid`` additionally sets the output to NaN on every 10 m
    pixel whose 100 m Landsat cell is not native-valid. With ``native100`` this
    is already the case; with ``block1000`` it removes the prediction under
    clouds and flagged cells that the 1000 m block mean would otherwise fill.
    """
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"prior mode must be one of {PRIOR_MODES}, got {prior_mode!r}")
    if selection.n_active != int(getattr(task, "n_active_channels", selection.n_active)):
        raise ValueError(
            f"selection has {selection.n_active} channels, task expects "
            f"{getattr(task, 'n_active_channels', None)}"
        )
    log_event(
        _logger,
        logging.INFO,
        "scene_inference_start",
        scene_id=scene_id,
        prior_mode=prior_mode,
        features_uri=features_uri,
        landsat_cog=landsat_cog,
    )
    features, feature_valid, transform_10m = _read_scene_inputs(
        features_uri,
        feature_valid_uri,
        expected_bands=len(scaler.channels),
        bbox=bbox,
    )
    if int(features.shape[0]) != len(scaler.channels):
        raise RuntimeError(
            f"{scene_id}: feature stack has {features.shape[0]} bands, "
            f"expected {len(scaler.channels)}"
        )
    scaled, _filled = scale_and_fill_features(scaler, features)
    selected = np.ascontiguousarray(scaled[list(selection.indices)])
    st, flag, transform_100m = _read_landsat_scene(landsat_cog, landsat_flag)
    grid = GeoBox((feature_valid.shape[0], feature_valid.shape[1]), transform_10m, "EPSG:25833")
    prior = build_prior_10m(st, flag, transform_100m, grid, prior_mode)
    prediction = predict_scene(
        task,
        selected,
        prior,
        feature_valid=feature_valid,
        tile_px=tile_px,
        stride_px=stride_px,
        device=device,
        batch_size=batch_size,
    )
    if mask_native_invalid:
        native = build_prior_10m(st, flag, transform_100m, grid, PRIOR_NATIVE100)
        prediction = np.where(np.isfinite(native), prediction, np.nan).astype(prediction.dtype)
    valid_output = np.isfinite(prediction)
    if bool(valid_output.any()) and not bool(np.isfinite(prediction[valid_output]).all()):
        raise RuntimeError(f"{scene_id}: non-finite model output on a valid pixel")
    finite = prediction[valid_output]
    report = scale_check_report(prediction, prior, st, flag, transform_100m, transform_10m)
    payload: dict[str, object] = {
        "scene_id": scene_id,
        "checkpoint_uri": provenance.checkpoint_uri,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "config_name": provenance.config_name,
        "arm": provenance.arm,
        "seed": provenance.seed,
        "prior_mode": prior_mode,
        "mask_native_invalid": mask_native_invalid,
        "tile_px": tile_px,
        "stride_px": stride_px,
        "code_commit": provenance.code_commit,
        "scaler_sha256": provenance.scaler_sha256,
        "features_uri": features_uri,
        "feature_valid_uri": feature_valid_uri,
        "landsat_cog": landsat_cog,
        "landsat_flag": landsat_flag,
        "bbox": None if bbox is None else list(bbox),
        "output_valid_px": int(valid_output.sum()),
        "feature_valid_px": int((feature_valid == 1).sum()),
        "prior_valid_px": int(np.isfinite(prior).sum()),
        "value_min_k": None if finite.size == 0 else float(finite.min()),
        "value_max_k": None if finite.size == 0 else float(finite.max()),
        **report,
    }
    tif_path, json_path = scene_product_paths(out_dir, scene_id)
    write_lst_cog(prediction, transform_10m, tif_path)
    _write_json(json_path, payload)
    log_event(
        _logger,
        logging.INFO,
        "scene_inference_done",
        scene_id=scene_id,
        output_valid_px=payload["output_valid_px"],
        lst_path=tif_path,
    )
    return payload


def scenes_with_done_features(features_root: str) -> list[str]:
    """Return scene ids whose features-ledger row is ``feature_stack`` / ``done``.

    A missing ledger or a duplicate scene id raises. The year filter is applied
    by the caller with :func:`berlin_lst_downscaling.modeling.mosaic.acquisition_year`.
    """
    uri = ledger_path(features_root)
    if not exists(uri):
        raise FileNotFoundError(uri)
    ledger = SecondaryLedger.open(uri)
    found: list[str] = []
    seen: set[str] = set()
    for row in ledger.items_for_source(_FEATURE_SOURCE):
        if row.status != "done":
            continue
        scene_id = str(row.period_or_vintage)
        if scene_id in seen:
            raise RuntimeError(f"duplicate done feature_stack row for scene {scene_id!r}")
        seen.add(scene_id)
        found.append(scene_id)
    found.sort()
    return found


def resolve_scene_landsat(ard_root: str, scene_id: str) -> tuple[str, str]:
    """Return ``(landsat_cog, landsat_flag)`` for one done ARD scene, or raise."""
    found = load_landsat_uris(ard_root, {scene_id})
    uris = found.get(scene_id)
    if uris is None:
        raise FileNotFoundError(f"no done Landsat ARD row for scene {scene_id!r} under {ard_root}")
    return uris


def assert_validation_split(split: str, *, patch_id: str) -> None:
    """Refuse every split other than validation. The 2025 test split stays unread."""
    if split != "validation":
        raise ValueError(
            f"{patch_id}: split {split!r} is not validation; "
            "inference reproduction refuses every other split"
        )


def require_validation_patches(patch_index_root: str, patch_ids: Sequence[str]) -> dict[str, str]:
    """Read the requested splits and refuse anything other than validation.

    Call this before a checkpoint is opened and before any feature window is
    read. A 2025 test id stops the reproduction gate here.
    """
    if not patch_ids:
        raise ValueError("at least one validation patch id is required")
    splits = read_patch_splits(patch_index_root, patch_ids)
    for patch_id in patch_ids:
        assert_validation_split(splits[str(patch_id)], patch_id=str(patch_id))
    return splits


def read_patch_splits(patch_index_root: str, patch_ids: Sequence[str]) -> dict[str, str]:
    """Return ``patch_id -> split`` for the requested ids. Missing ids raise."""
    uri = patch_index_parquet(patch_index_root)
    _require_file(uri)
    table = pq.read_table(io.BytesIO(read_bytes(uri)), columns=["patch_id", "split"])
    wanted = {str(patch_id) for patch_id in patch_ids}
    if len(wanted) != len(tuple(patch_ids)):
        raise ValueError("patch ids must be unique")
    found: dict[str, str] = {}
    ids = table.column("patch_id").to_pylist()
    splits = table.column("split").to_pylist()
    for patch_id, split in zip(ids, splits, strict=True):
        key = str(patch_id)
        if key in wanted:
            found[key] = str(split)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"patch ids not in the index: {sorted(missing)}")
    return found


def score_patch_like_export(
    prediction_10m: np.ndarray,
    target_100m: np.ndarray,
    mask_100m: np.ndarray,
) -> tuple[float, int]:
    """Return ``(abs_error_sum, valid_cells)`` the way ``score_batch_patches`` does.

    The 10 m prediction is pooled with arithmetic 10×10 means, clamped once
    with :func:`clamp_eval_prediction`, then reduced with
    :func:`masked_abs_error_sums` on the patch eligibility mask and the native
    target. SSIM is not part of the reproduction comparison.
    """
    pred = np.asarray(prediction_10m, dtype=np.float32)
    target = np.asarray(target_100m, dtype=np.float32)
    mask = np.asarray(mask_100m, dtype=bool)
    if pred.ndim == 2:
        pred_t = torch.from_numpy(pred)[None, None]
    elif pred.ndim == 3:
        pred_t = torch.from_numpy(pred)[None]
    else:
        raise ValueError(f"prediction must be (H, W) or (1, H, W), got {pred.shape}")
    if target.ndim == 2:
        target_t = torch.from_numpy(target)[None, None]
    elif target.ndim == 3:
        target_t = torch.from_numpy(target)[None]
    else:
        raise ValueError(f"target must be (H, W) or (1, H, W), got {target.shape}")
    if mask.ndim == 2:
        mask_t: Tensor = torch.from_numpy(mask)[None, None]
    elif mask.ndim == 3:
        mask_t = torch.from_numpy(mask)[None]
    else:
        raise ValueError(f"mask must be (H, W) or (1, H, W), got {mask.shape}")
    pooled = pool_10m_to_100m(pred_t, AGGREGATION_ARITHMETIC)
    pooled = clamp_eval_prediction(pooled)
    error_sum, count = masked_abs_error_sums(pooled, target_t, mask_t)
    return float(error_sum), int(count)


def relative_difference(got: float, reference: float) -> float:
    """Return ``|got - reference| / |reference|``. A zero reference matches only a zero."""
    if not math.isfinite(got) or not math.isfinite(reference):
        return math.inf
    scale = abs(reference)
    if scale == 0.0:
        return 0.0 if got == 0.0 else math.inf
    return abs(got - reference) / scale


def reproduction_mismatch(
    *,
    got_sum: float,
    got_cells: int,
    reference_sum: float,
    reference_cells: int,
    tolerance: float = REPRODUCTION_RELATIVE_TOLERANCE,
) -> str | None:
    """Return a failure reason, or None when the patch matches the export."""
    if got_cells != reference_cells:
        return f"valid_cells {got_cells} != {reference_cells}"
    difference = relative_difference(got_sum, reference_sum)
    if difference > tolerance:
        return f"relative abs_error_sum difference {difference} > {tolerance}"
    return None


def patch_bbox_10m(ref: PatchRef, *, col0_10: int, row0_10: int) -> tuple[int, int, int, int]:
    """Return the feature-COG window of one patch, the same one the reader uses."""
    col = ref.col * _LATTICE_PX - col0_10
    row = ref.row * _LATTICE_PX - row0_10
    return col, row, REAL_PATCH_PX, REAL_PATCH_PX


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def seed_from_checkpoint_name(uri: str) -> int:
    """Return the ``-seedN-`` token of a matrix checkpoint URI, else 0.

    The token is read from the whole URI. A matrix directory is
    ``v6m-…-l4-seed2-ep40-cos`` and the object name itself is ``best.ckpt``.
    """
    found = _SEED_IN_NAME.findall(uri)
    if not found:
        return 0
    if len(set(found)) != 1:
        raise ValueError(f"checkpoint URI names more than one seed: {uri}")
    return int(found[0])


def feature_uris(features_root: str, scene_id: str) -> tuple[str, str]:
    return feature_cog(features_root, scene_id), feature_mask_cog(features_root, scene_id)


def scaler_sha256(training_root: str) -> str:
    return sha256_bytes(read_bytes(scaler_json(training_root)))


__all__ = [
    "INFERENCE_CONFIGS",
    "InferenceProvenance",
    "PRIOR_BLOCK1000",
    "PRIOR_MODES",
    "PRIOR_NATIVE100",
    "PRIOR_PLACEHOLDER_K",
    "REPRODUCTION_RELATIVE_TOLERANCE",
    "TileWindow",
    "assert_validation_split",
    "build_prior_10m",
    "feature_uris",
    "patch_bbox_10m",
    "plan_tiles",
    "predict_scene",
    "read_patch_splits",
    "relative_difference",
    "require_validation_patches",
    "reproduction_mismatch",
    "resolve_scene_landsat",
    "run_scene",
    "scaler_sha256",
    "scale_check_report",
    "scenes_with_done_features",
    "scene_product_paths",
    "score_patch_like_export",
    "seed_from_checkpoint_name",
    "sha256_bytes",
    "sha256_file",
    "write_lst_cog",
]
