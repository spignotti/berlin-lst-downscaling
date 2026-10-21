"""Canonical grid: EPSG:25833 origin, 10 m base, 100 m nested 10x10."""

from __future__ import annotations

from berlin_lst_downscaling.common.config import TARGET_CRS, TARGET_RESOLUTION
from berlin_lst_downscaling.common.grid import (
    canon_grid_10m,
    canon_grid_100m,
    canon_grid_for_resolution,
)
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
    cell_id,
)


def test_canonical_grids_share_epsg25833_origin() -> None:
    """10 m and 100 m grids use EPSG:25833 and the documented Berlin origin."""
    grid_10 = canon_grid_10m()
    grid_100 = canon_grid_100m()

    assert str(grid_10.crs) == TARGET_CRS == "EPSG:25833"
    assert str(grid_100.crs) == "EPSG:25833"
    assert TARGET_RESOLUTION == 10
    assert grid_10.affine.a == 10
    assert grid_10.affine.e == -10
    assert grid_100.affine.a == CELL_SIZE_M == 100
    assert grid_100.affine.e == -100
    assert grid_10.affine.c == CANON_GRID_ORIGIN_X == 369190
    assert grid_10.affine.f == CANON_GRID_ORIGIN_Y == 5838410
    assert grid_100.affine.c == grid_10.affine.c
    assert grid_100.affine.f == grid_10.affine.f
    assert canon_grid_for_resolution(10).affine == grid_10.affine
    assert canon_grid_for_resolution(100).affine == grid_100.affine


def test_100m_cell_is_a_10_by_10_block_of_the_10m_grid() -> None:
    """One 100 m cell covers exactly ten by ten 10 m pixels, same origin."""
    grid_10 = canon_grid_10m()
    grid_100 = canon_grid_100m()

    # Interior cell (column 3, row 4): its top-left is the 10 m pixel (30, 40).
    assert grid_100.affine.c + 3 * 100 == grid_10.affine.c + 30 * 10
    assert grid_100.affine.f + 4 * (-100) == grid_10.affine.f + 40 * (-10)
    assert grid_100.affine == grid_10.zoom_out(10).affine
    # The 10 m width is not divisible by 10 (4699). zoom_out rounds outward,
    # so the 100 m grid is one partial cell larger on each trailing edge.
    assert grid_100.shape.x == -(-grid_10.shape.x // 10)
    assert grid_100.shape.y == -(-grid_10.shape.y // 10)


def test_cell_id_encodes_the_100m_corner() -> None:
    """Cell ids are easting/northing of the cell's top-left on the 100 m lattice."""
    assert cell_id(0, 0) == "E369190N5838410"
    assert cell_id(0, 1) == "E369290N5838410"
    assert cell_id(2, 0) == "E369190N5838210"
