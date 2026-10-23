"""TESSERA missing data: gaps become invalid/NaN, never zero-filled."""

from __future__ import annotations

import numpy as np

from berlin_lst_downscaling.data.embeddings.tessera import (
    dequantize_tessera,
    tessera_invalid_mask,
)


def test_below_threshold_pixels_are_invalid_not_zero() -> None:
    quantized = np.full((3, 3, 128), 10, dtype=np.int8)
    scale = np.full((3, 3), 0.5, dtype=np.float32)
    quality = np.full((3, 3), 0.8)
    quality[1, 1] = 0.01  # below threshold
    invalid = tessera_invalid_mask(scale, quality, quality_threshold=0.5)
    assert invalid.tolist()[1][1] is True
    embedding = dequantize_tessera(quantized, scale)
    embedding[invalid] = np.nan
    assert bool(np.all(np.isnan(embedding[1, 1])))
    # Valid pixels keep their dequantized values (10 * 0.5), not zero.
    assert float(embedding[0, 0, 0]) == 5.0


def test_nonfinite_scale_never_zero_filled() -> None:
    quantized = np.zeros((2, 2, 128), dtype=np.int8)
    scale = np.zeros((2, 2), dtype=np.float32)
    scale[0, 0] = np.nan
    out = dequantize_tessera(quantized, scale)
    # Even an all-zero code with NaN scale is NaN (gap), not a zero vector.
    assert bool(np.all(np.isnan(out[0, 0])))
    assert bool(np.all(out[1, 1] == 0))
