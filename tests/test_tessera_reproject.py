"""TESSERA reprojection: explicit nearest onto the canonical 10 m grid."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.warp import Resampling

from berlin_lst_downscaling.data.embeddings.reproject import (
    reproject_quantized_to_canonical,
    reproject_to_canonical,
)


def _offset_canonical_grid() -> tuple[tuple[int, int], Affine]:
    """Return a small canonical-like grid and a half-pixel-offset source grid."""
    dst_transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    dst_shape = (8, 10)
    return dst_shape, dst_transform


def test_nearest_copies_values_without_blending() -> None:
    dst_shape, dst_transform = _offset_canonical_grid()
    # Source grid offset by 3 m east / 2 m north from the canonical origin:
    # nearest must snap each destination pixel to exactly one source value.
    src_transform = Affine(10.0, 0.0, 1003.0, 0.0, -10.0, 1998.0)
    h, w = 8, 10
    base = np.arange(h * w, dtype=np.float32).reshape(h, w)
    source = np.stack([base + c * 1000.0 for c in range(128)], axis=2)
    out = reproject_to_canonical(
        source,
        src_transform,
        "EPSG:32633",
        dst_shape=dst_shape,
        dst_transform=dst_transform,
        dst_crs="EPSG:25833",
    )
    assert out.shape == (h, w, 128)
    # Every output value comes from the source palette (no averaging).
    palette = set(np.unique(source[:, :, 0]).tolist())
    for value in np.unique(out[:, :, 0][np.isfinite(out[:, :, 0])]).tolist():
        assert value in palette
    # Channel separation is preserved (offset of 1000 per channel).
    finite = np.isfinite(out[:, :, 0]) & np.isfinite(out[:, :, 5])
    assert bool(np.any(finite))
    assert bool(np.all(out[:, :, 5][finite] - out[:, :, 0][finite] == 5000.0))


def test_non_nearest_resampling_refused() -> None:
    dst_shape, dst_transform = _offset_canonical_grid()
    source = np.zeros((4, 4, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="nearest"):
        reproject_to_canonical(
            source,
            Affine.identity(),
            "EPSG:32633",
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            resampling=Resampling.bilinear,
        )


def test_channel_first_layout_round_trips() -> None:
    dst_shape, dst_transform = _offset_canonical_grid()
    src_transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    source = np.zeros((128, 8, 10), dtype=np.float32)
    source[:, 2, 3] = 7.0
    out = reproject_to_canonical(
        source,
        src_transform,
        "EPSG:32633",
        dst_shape=dst_shape,
        dst_transform=dst_transform,
        dst_crs="EPSG:25833",
    )
    assert out.shape == (128, 8, 10)
    assert float(out[:, 2, 3].mean()) == 7.0


def test_quantized_reprojection_keeps_gaps_nan() -> None:
    dst_shape = (6, 6)
    dst_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 60.0)
    src_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 60.0)
    quantized = np.ones((6, 6, 128), dtype=np.int8)
    scale = np.full((6, 6), 0.25, dtype=np.float32)
    scale[0, 0] = np.nan
    q_out, s_out = reproject_quantized_to_canonical(
        quantized,
        scale,
        src_transform,
        "EPSG:32633",
        dst_shape=dst_shape,
        dst_transform=dst_transform,
    )
    assert q_out.shape == (6, 6, 128)
    assert s_out.shape == (6, 6)
    assert bool(np.isnan(s_out[0, 0]))
    assert bool(np.isfinite(s_out[1, 1]))


def test_int8_extreme_127_codes_are_preserved() -> None:
    dst_shape = (4, 4)
    dst_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 40.0)
    src_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 40.0)
    quantized = np.full((4, 4, 128), 127, dtype=np.int8)
    scale = np.full((4, 4), 0.5, dtype=np.float32)
    q_out, s_out = reproject_quantized_to_canonical(
        quantized,
        scale,
        src_transform,
        "EPSG:32633",
        dst_shape=dst_shape,
        dst_transform=dst_transform,
    )
    assert q_out.dtype == np.int8
    assert bool(np.all(q_out == 127))
    assert bool(np.all(np.isfinite(s_out)))
