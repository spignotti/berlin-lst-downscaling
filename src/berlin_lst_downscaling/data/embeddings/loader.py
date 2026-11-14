"""Per-scene-year TESSERA embedding loader for the ablation training path.

The TESSERA arm keeps every v6 channel identical to the main arm except
the 6 morphology + 2 shadow channels, which are replaced by the annual
TESSERA embedding for each scene's year. This module is the training-data
path wiring:

- :func:`load_scene_year_embedding` reads the canonical per-year product
  (int8 COG + scale COG from ``product.py``) for one calendar year and
  returns the dequantized embedding with NaN at invalid pixels;
- :func:`select_kept_features` drops the replaced morphology/shadow
  channels from a v6 stack, keeping the other 14 channels identical;
- :func:`build_tessera_model_inputs` splits a full v6 patch into the
  kept hand-built tensor and the aligned embedding tensor consumed by
  :class:`TesseraUNet`.

All rasters are on the canonical 10 m grid; no resampling happens here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from berlin_lst_downscaling.data.embeddings.product import (
    product_cog_uri,
    product_scale_uri,
    product_valid_uri,
)
from berlin_lst_downscaling.data.embeddings.tessera import (
    TESSERA_N_CHANNELS,
    dequantize_tessera,
    validate_tessera_year,
)
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.modeling.tessera_adapter import (
    tessera_kept_channel_indices,
)


def _v6_width() -> int:
    """Width of the features/v6 roster (never a literal on this path)."""
    return feature_schema(FEATURE_SCHEMA_VERSION_V6).n_channels


def select_kept_features(features_v6: np.ndarray) -> np.ndarray:
    """Drop the replaced morphology/shadow channels from a v6 stack.

    Parameters
    ----------
    features_v6 :
        ``(C, H, W)`` or ``(H, W, C)`` v6 stack in roster order, where
        ``C`` is ``schema.n_channels``.

    Returns
    -------
    numpy.ndarray
        The 14 kept channels (spectral + index + ERA5), same layout.
    """
    arr = np.asarray(features_v6)
    width = _v6_width()
    kept = tessera_kept_channel_indices()
    if arr.ndim == 3 and arr.shape[0] == width:
        return arr[list(kept), :, :]
    if arr.ndim == 3 and arr.shape[2] == width:
        return arr[:, :, list(kept)]
    raise ValueError(f"expected a {width}-channel v6 stack, got shape {tuple(arr.shape)}")


def load_scene_year_embedding(
    embeddings_root: str,
    year: int,
    *,
    reader: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the dequantized annual embedding for one scene year.

    Reads the canonical per-year product (int8 COG + scale COG) written by
    :mod:`product`. ``reader`` injects a fake ``(quantized, scale)`` pair
    for offline tests; ``None`` reads local COG paths with rasterio
    (GCS URIs rely on GDAL's ``/vsigs/`` handler on the operator VM).

    Returns ``(embedding (H, W, 128) float32, valid (H, W) bool)`` with NaN
    at TESSERA-invalid pixels.
    """
    validate_tessera_year(int(year))
    if reader is not None:
        quantized, scale = reader(int(year))
        quantized_arr = np.asarray(quantized, dtype=np.int8)
        scale_arr = np.asarray(scale)
    else:
        import rasterio

        with rasterio.open(product_cog_uri(str(embeddings_root), int(year))) as src:
            if src.count != TESSERA_N_CHANNELS:
                raise ValueError(f"expected 128 int8 bands, got {src.count}")
            quantized_arr = src.read().astype(np.int8).transpose(1, 2, 0)
        with rasterio.open(product_scale_uri(str(embeddings_root), int(year))) as src:
            scale_arr = src.read(1)
    embedding = dequantize_tessera(
        np.ascontiguousarray(quantized_arr, dtype=np.int8),
        np.ascontiguousarray(scale_arr),
    )
    valid = np.all(np.isfinite(embedding), axis=2)
    embedding[~valid] = np.nan
    return embedding, np.asarray(valid, dtype=bool)


def build_tessera_model_inputs(
    features_v6: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a v6 patch and its year embedding for :class:`TesseraUNet`.

    Returns ``(kept (14, H, W), embedding_chw (128, H, W))`` with matching
    spatial extents; raises on mismatch so a misaligned year fails loudly.
    """
    kept = select_kept_features(np.asarray(features_v6))
    emb = np.asarray(embedding)
    if emb.ndim == 3 and emb.shape[2] == 128:
        emb_chw = np.moveaxis(emb, 2, 0)
    elif emb.ndim == 3 and emb.shape[0] == 128:
        emb_chw = emb
    else:
        raise ValueError(f"expected a 128-channel embedding, got {tuple(emb.shape)}")
    if kept.shape[1:] != emb_chw.shape[1:]:
        raise ValueError(
            f"spatial mismatch: kept {tuple(kept.shape[1:])} != "
            f"embedding {tuple(emb_chw.shape[1:])}"
        )
    return kept, emb_chw


_OPEN_YEAR: dict[tuple[str, int, str], Any] = {}


def _year_dataset(embeddings_root: str, year: int, kind: str) -> Any:
    """Reuse one open COG per year and kind inside this process."""
    key = (str(embeddings_root), int(year), kind)
    dataset = _OPEN_YEAR.get(key)
    if dataset is not None:
        return dataset
    import rasterio

    uri = {
        "int8": product_cog_uri(str(embeddings_root), int(year)),
        "scale": product_scale_uri(str(embeddings_root), int(year)),
        "valid": product_valid_uri(str(embeddings_root), int(year)),
    }[kind]
    dataset = rasterio.open(uri)
    _OPEN_YEAR[key] = dataset
    return dataset


def tessera_window_fully_valid(
    embeddings_root: str,
    year: int,
    *,
    row_100m: int,
    col_100m: int,
    patch_cells: int = 16,
) -> bool:
    """True when every 10 m pixel in the patch window is TESSERA-valid."""
    from rasterio.windows import Window

    validate_tessera_year(int(year))
    edge = int(patch_cells) * 10
    col_off, row_off = int(col_100m) * 10, int(row_100m) * 10
    window = Window.from_slices((row_off, row_off + edge), (col_off, col_off + edge))
    valid = _year_dataset(embeddings_root, int(year), "valid").read(1, window=window)
    return bool(valid.size) and bool(np.all(valid == 1))


def load_patch_quantized(
    embeddings_root: str,
    year: int,
    *,
    row_100m: int,
    col_100m: int,
    patch_cells: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the unchanged int8 HWC window and float32 per-pixel scale."""
    from rasterio.windows import Window

    validate_tessera_year(int(year))

    h = w = int(patch_cells) * 10
    col_off, row_off = int(col_100m) * 10, int(row_100m) * 10
    window = Window.from_slices((row_off, row_off + h), (col_off, col_off + w))
    quantized_arr = (
        _year_dataset(embeddings_root, int(year), "int8")
        .read(window=window)
        .astype(np.int8)
        .transpose(1, 2, 0)
    )
    scale_arr = _year_dataset(embeddings_root, int(year), "scale").read(1, window=window)
    if quantized_arr.shape != (h, w, TESSERA_N_CHANNELS) or scale_arr.shape != (h, w):
        raise ValueError("TESSERA patch window is outside the product geometry")
    if scale_arr.dtype != np.dtype("float32"):
        raise ValueError("TESSERA product scale must be float32")
    return np.ascontiguousarray(quantized_arr), np.ascontiguousarray(scale_arr)


def load_patch_embedding(
    embeddings_root: str,
    year: int,
    *,
    row_100m: int,
    col_100m: int,
    patch_cells: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Dequantize one canonical patch window, preserving NaN at gaps."""
    quantized_arr, scale_arr = load_patch_quantized(
        embeddings_root,
        year,
        row_100m=row_100m,
        col_100m=col_100m,
        patch_cells=patch_cells,
    )
    embedding = dequantize_tessera(
        np.ascontiguousarray(quantized_arr, dtype=np.int8),
        np.ascontiguousarray(scale_arr),
    )
    valid = np.all(np.isfinite(embedding), axis=2)
    embedding[~valid] = np.nan
    return embedding, np.asarray(valid, dtype=bool)


def collate_tessera_batch(samples, embeddings_root, *, kept_indices=None):
    """Collate kept hand-built channels with per-scene-year embeddings.

    Each sample carries ``features`` (full v6 patch, channel-first) and
    ``meta`` with ``year``/``row``/``col``. The loader reads that scene's
    annual embedding patch window (never the full-year raster), dequantizes
    it, and stacks ``(kept (B, 14, H, W), embedding (B, 128, H, W))`` for
    :class:`TesseraRealLSTTask`. Morphology + shadow channels are dropped
    here, never passed to the model.

    Window-read failures propagate (fail closed). There is deliberately
    no full-raster fallback: a 2.3 GB whole-product read inside the
    per-sample loop would violate the no-full-raster-loads rule.
    """
    import torch

    width = _v6_width()
    kept_idx = (
        list(kept_indices) if kept_indices is not None else list(tessera_kept_channel_indices())
    )
    kept_list: list = []
    emb_list: list = []
    for sample in samples:
        features = np.asarray(sample.features)
        if features.shape[0] != width:
            raise ValueError(f"expected {width}-channel v6 patch, got {tuple(features.shape)}")
        kept_list.append(features[kept_idx, :, :])
        meta = sample.meta
        row = int(getattr(meta, "row", 0))
        col = int(getattr(meta, "col", 0))
        quantized = getattr(sample, "tessera_int8", None)
        scale = getattr(sample, "tessera_scale", None)
        if quantized is not None and scale is not None:
            embedding = dequantize_tessera(quantized, scale)
            embedding[~np.all(np.isfinite(embedding), axis=2)] = np.nan
        elif quantized is not None or scale is not None:
            raise RuntimeError("incomplete cached TESSERA encoding")
        else:
            embedding, _ = load_patch_embedding(
                str(embeddings_root), int(meta.year), row_100m=row, col_100m=col
            )
        emb_list.append(np.moveaxis(embedding, 2, 0))
    kept_batch = torch.from_numpy(np.stack(kept_list).astype("float32"))
    emb_batch = torch.from_numpy(np.stack(emb_list).astype("float32"))
    return kept_batch, emb_batch


__all__ = [
    "build_tessera_model_inputs",
    "collate_tessera_batch",
    "load_patch_embedding",
    "load_scene_year_embedding",
    "select_kept_features",
]
