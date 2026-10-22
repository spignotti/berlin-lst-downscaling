"""Temperature aggregation as implemented: prior means and 10x10 pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.modeling.aggregation import AggregationContractError
from berlin_lst_downscaling.modeling.metrics import pool_10m_to_100m
from berlin_lst_downscaling.modeling.patches import _build_scene_prior

_KELVIN_MIN = 150.0
_KELVIN_MAX = 400.0


def test_lst_range_is_the_kelvin_band() -> None:
    """Physical LST bounds are Kelvin, inclusive, and reject a Celsius-looking 25."""
    assert LST_RANGE_K == (_KELVIN_MIN, _KELVIN_MAX)
    assert not (_KELVIN_MIN <= 25.0 <= _KELVIN_MAX)


def _write_band(path: Path, array: np.ndarray, dtype: str) -> None:
    """Write one canonical 100 m EPSG:25833 raster."""
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs="EPSG:25833",
        transform=from_origin(369190, 5838410, 100, 100),
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _native_scene() -> tuple[np.ndarray, np.ndarray]:
    """Return a 10x20 LST/flag pair: one mixed block and one all-invalid block."""
    st = np.full((10, 20), 300.0, dtype=np.float32)
    flag = np.zeros((10, 20), dtype=np.uint8)
    st[0, 0] = np.nan
    st[0, 1] = 25.0
    st[0, 2] = 149.0
    st[0, 3] = 150.0
    st[0, 4] = 400.0
    st[0, 5] = 400.1
    st[0, 6] = 290.0
    flag[0, 6] = 1
    st[:, 10:] = 25.0
    return st, flag


def test_prior_block_mean_uses_native_valid_kelvin_only(tmp_path: Path) -> None:
    """Invalid, flagged, and Celsius-looking cells stay out of the 1000 m mean.

    The mean is the sum of native-valid Kelvin samples over their count. A
    block with no such sample is NaN, which is not a substituted zero.
    """
    st, flag = _native_scene()
    valid = np.isfinite(st) & (flag == 0) & (st >= _KELVIN_MIN) & (st <= _KELVIN_MAX)
    assert int(valid[:, :10].sum()) == 95
    assert int(valid[:, 10:].sum()) == 0
    expected = float(st[:, :10][valid[:, :10]].astype(np.float64).mean())

    lst_path = tmp_path / "lst.tif"
    flag_path = tmp_path / "flag.tif"
    _write_band(lst_path, st, "float32")
    _write_band(flag_path, flag, "uint8")
    prior = _build_scene_prior("scene", str(lst_path), str(flag_path))

    assert prior.block_means.shape == (1, 2)
    assert prior.native_valid_cells == 95
    np.testing.assert_allclose(prior.block_means[0, 0], np.float32(expected), atol=1e-5)
    assert np.isnan(prior.block_means[0, 1])
    assert not np.isfinite(prior.block_means[0, 1])


def test_pool_10m_to_100m_is_the_exact_nested_block_mean() -> None:
    """Pooling is the mean of each disjoint 10x10 block, with no resampling."""
    block = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    pooled = pool_10m_to_100m(block)
    assert tuple(pooled.shape) == (1, 1, 1, 1)
    assert float(pooled) == pytest.approx(49.5)

    image = torch.cat(
        [torch.full((1, 1, 10, 10), 2.0), torch.full((1, 1, 10, 10), 8.0)],
        dim=-1,
    )
    side_by_side = pool_10m_to_100m(image)
    assert tuple(side_by_side.shape) == (1, 1, 1, 2)
    assert float(side_by_side[0, 0, 0, 0]) == pytest.approx(2.0)
    assert float(side_by_side[0, 0, 0, 1]) == pytest.approx(8.0)

    dirty = block.clone()
    dirty[0, 0, 0, 0] = float("nan")
    assert torch.isnan(pool_10m_to_100m(dirty)).all()

    with pytest.raises(ValueError, match="not divisible"):
        pool_10m_to_100m(torch.zeros(1, 1, 10, 11))


def test_pool_nan_propagates_inside_its_block_only() -> None:
    """One NaN makes that 10×10 arithmetic block NaN and leaves the next block finite.

    Arithmetic ``mean`` propagates NaN. The NaN is not dropped from the block
    and is not replaced by zero, so the finite block beside it stays its mean.
    T4 refuses a non-finite contributing pixel instead of emitting a NaN block.
    """
    finite = torch.full((1, 1, 10, 10), 300.0)
    dirty = finite.clone()
    dirty[0, 0, 0, 0] = float("nan")
    image = torch.cat([dirty, finite], dim=-1)
    pooled = pool_10m_to_100m(image, "arithmetic")
    assert tuple(pooled.shape) == (1, 1, 1, 2)
    assert torch.isnan(pooled[0, 0, 0, 0])
    assert float(pooled[0, 0, 0, 1]) == pytest.approx(300.0)
    assert not torch.isnan(pooled[0, 0, 0, 1])

    with pytest.raises(AggregationContractError, match="finite"):
        pool_10m_to_100m(image, "t4")


def test_pool_excludes_incomplete_edge_blocks() -> None:
    """Leftover rows or columns past the last full 10×10 block are not pooled.

    A 10×20 map yields two complete blocks. A 10×19 or 19×10 map raises, so
    the incomplete edge is excluded instead of reduced as a partial block.
    """
    complete = torch.arange(200, dtype=torch.float32).reshape(1, 1, 10, 20)
    pooled = pool_10m_to_100m(complete)
    assert tuple(pooled.shape) == (1, 1, 1, 2)
    assert float(pooled[0, 0, 0, 0]) == pytest.approx(float(complete[0, 0, :, :10].mean()))
    assert float(pooled[0, 0, 0, 1]) == pytest.approx(float(complete[0, 0, :, 10:].mean()))

    with pytest.raises(ValueError, match="not divisible"):
        pool_10m_to_100m(torch.zeros(1, 1, 10, 19))
    with pytest.raises(ValueError, match="not divisible"):
        pool_10m_to_100m(torch.zeros(1, 1, 19, 10))
