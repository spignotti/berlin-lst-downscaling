"""Explicit TESSERA reprojection EPSG:32633 → canonical EPSG:25833 10 m.

TESSERA is natively ``EPSG:32633`` on its own raster origin, which does not
align with the canonical Berlin grid (``common/grid.py``, ``EPSG:25833``
10 m). This module reprojects onto the canonical grid with **nearest**
resampling only — embeddings are representation vectors and must be copied,
never interpolated or blended.

The resampling is always explicit (``rasterio.warp.Resampling.nearest``);
callers that omit it fail closed. The canonical grid (shape, transform,
CRS) comes from :func:`canon_grid_10m` unless an explicit test grid is
injected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m


def canonical_10m_shape_transform() -> tuple[tuple[int, int], Any, str]:
    """Return the canonical 10 m grid ``((rows, cols), transform, crs)``."""
    geobox = canon_grid_10m()
    transform = geobox.transform
    shape = (int(geobox.shape.y), int(geobox.shape.x))
    return shape, transform, str(geobox.crs)


def reproject_to_canonical(
    source: np.ndarray,
    src_transform: Any,
    src_crs: str,
    *,
    dst_shape: tuple[int, int] | None = None,
    dst_transform: Any | None = None,
    dst_crs: str = "EPSG:25833",
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Reproject a TESSERA mosaic onto the canonical 10 m grid.

    Parameters
    ----------
    source :
        Native mosaic. ``(H, W, 128)`` channel-last (the
        :mod:`geotessera` convention) or ``(128, H, W)`` channel-first.
        Returned array matches the input layout.
    src_transform :
        Affine transform of the native read.
    src_crs :
        Native CRS (expected ``EPSG:32633``; other values are passed
        through to :func:`rasterio.warp.reproject` unchanged).
    dst_shape, dst_transform :
        Canonical destination grid; defaults to :func:`canon_grid_10m`.
    dst_crs :
        Destination CRS (default ``EPSG:25833``).
    resampling :
        Must be :attr:`Resampling.nearest`; any other value raises.

    Returns
    -------
    numpy.ndarray
        Reprojected mosaic on the canonical grid, same layout/dtype-kind
        as the input (NaN preserved, never blended).
    """
    if resampling != Resampling.nearest:
        raise ValueError(
            "TESSERA embeddings must use nearest resampling "
            f"(no interpolation of embeddings); got {resampling!r}"
        )
    arr = np.asarray(source)
    if arr.ndim != 3:
        raise ValueError(f"TESSERA mosaic must be 3-D, got shape {tuple(arr.shape)}")
    channel_last = arr.shape[2] == 128
    channel_first = arr.shape[0] == 128
    if not (channel_last ^ channel_first):
        raise ValueError(
            f"TESSERA mosaic must have a 128-channel axis (got shape {tuple(arr.shape)})"
        )
    work = arr if channel_first else np.moveaxis(arr, 2, 0)
    n_bands, src_h, src_w = work.shape

    if dst_shape is None or dst_transform is None:
        canon_shape, canon_transform, _ = canonical_10m_shape_transform()
        dst_shape = canon_shape if dst_shape is None else dst_shape
        dst_transform = canon_transform if dst_transform is None else dst_transform
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])

    out = np.full((n_bands, dst_h, dst_w), np.nan, dtype=np.float32)
    src = np.ascontiguousarray(work.astype(np.float32, copy=False))
    reproject(
        source=src,
        destination=out,
        src_transform=src_transform,
        src_crs=CRS.from_string(str(src_crs)),
        dst_transform=dst_transform,
        dst_crs=CRS.from_string(str(dst_crs)),
        resampling=Resampling.nearest,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
    )
    if channel_last:
        return np.moveaxis(out, 0, 2)
    return out


def reproject_quantized_to_canonical(
    quantized: np.ndarray,
    scale: np.ndarray,
    src_transform: Any,
    src_crs: str,
    *,
    dst_shape: tuple[int, int] | None = None,
    dst_transform: Any | None = None,
    dst_crs: str = "EPSG:25833",
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject int8 + scale onto the canonical grid with nearest resampling.

    The int8 codes and the float32 scale are reprojected as separate
    nearest-neighbour fields, then recombined by the caller with
    :func:`dequantize_tessera`. Scale pixels that fall outside the source
    footprint stay NaN (hence invalid), never zero. Validity rides on the
    scale alone: every int8 code (including the 127 extreme) is preserved
    verbatim, so no sentinel nodata is used for the code field.
    """
    q = np.asarray(quantized)
    if q.ndim != 3 or q.shape[2] != 128:
        raise ValueError(f"quantized must be (H, W, 128) int8, got {tuple(q.shape)}")
    # Nearest on int8 codes with no nodata masking: every code is a valid
    # embedding value (including 127), and invalid pixels are carried by
    # the reprojected scale (NaN → invalid after dequantization).
    q_work = np.moveaxis(q.astype(np.float32), 2, 0)
    s = np.asarray(scale)
    if s.ndim == 3:
        s = s[:, :, 0] if s.shape[2] == 1 else np.nanmean(s, axis=2)
    s_work = np.ascontiguousarray(s.astype(np.float32, copy=False)[np.newaxis, :, :])

    if dst_shape is None or dst_transform is None:
        canon_shape, canon_transform, _ = canonical_10m_shape_transform()
        dst_shape = canon_shape if dst_shape is None else dst_shape
        dst_transform = canon_transform if dst_transform is None else dst_transform
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])

    q_out = np.full((128, dst_h, dst_w), np.nan, dtype=np.float32)
    s_out = np.full((1, dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=q_work,
        destination=q_out,
        src_transform=src_transform,
        src_crs=CRS.from_string(str(src_crs)),
        dst_transform=dst_transform,
        dst_crs=CRS.from_string(str(dst_crs)),
        resampling=Resampling.nearest,
    )
    reproject(
        source=s_work,
        destination=s_out,
        src_transform=src_transform,
        src_crs=CRS.from_string(str(src_crs)),
        dst_transform=dst_transform,
        dst_crs=CRS.from_string(str(dst_crs)),
        resampling=Resampling.nearest,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
    )
    q_back = np.moveaxis(q_out, 0, 2)
    # Outside-footprint codes stay NaN from the pre-fill and become 0 here;
    # they are invalid via the NaN scale. Inside-footprint codes (including
    # the 127 extreme) round-trip exactly with no sentinel loss.
    q_int = np.where(np.isfinite(q_back), np.rint(q_back), 0).astype(np.int8)
    s_back = s_out[0]
    q_int[~np.isfinite(s_back)] = 0
    return q_int, s_back


__all__ = [
    "canonical_10m_shape_transform",
    "reproject_quantized_to_canonical",
    "reproject_to_canonical",
]
