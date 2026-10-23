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


def load_patch_embedding(
    embeddings_root: str,
    year: int,
    *,
    row_100m: int,
    col_100m: int,
    patch_cells: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one patch window from the canonical annual product.

    The product COGs are canonical-gridded, so the patch window is derived
    from the global 100 m anchor: 10 m origin ``(col*10, row*10)`` with a
    ``patch_cells*10`` px edge. Reads only that window (never the full-year
    raster) via a rasterio Window. Returns dequantized ``(H, W, 128)`` with
    NaN at gaps plus the bool validity mask.
    """
    from rasterio.windows import Window

    validate_tessera_year(int(year))
    import rasterio

    h = w = int(patch_cells) * 10
    col_off, row_off = int(col_100m) * 10, int(row_100m) * 10
    window = Window.from_slices((row_off, row_off + h), (col_off, col_off + w))
    with rasterio.open(product_cog_uri(str(embeddings_root), int(year))) as src:
        quantized_arr = src.read(window=window).astype(np.int8).transpose(1, 2, 0)
    with rasterio.open(product_scale_uri(str(embeddings_root), int(year))) as src:
        scale_arr = src.read(1, window=window)
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
        try:
            embedding, _ = load_patch_embedding(
                str(embeddings_root), int(meta.year), row_100m=row, col_100m=col
            )
        except Exception as exc:  # noqa: BLE001 — small/test products are patch-sized
            embedding, _ = load_scene_year_embedding(str(embeddings_root), int(meta.year))
            h, w = features.shape[1], features.shape[2]
            if embedding.shape[:2] != (h, w):
                raise ValueError(
                    f"embedding extent {tuple(embedding.shape[:2])} != patch {(h, w)}"
                ) from exc
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
