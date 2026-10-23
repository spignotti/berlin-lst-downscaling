"""TESSERA common cell set: both arms see exactly the same cells."""

from __future__ import annotations

import numpy as np

from berlin_lst_downscaling.data.embeddings.cells import (
    aggregate_tessera_valid_to_100m,
    common_cell_set,
    intersect_eligibility,
    main_arm_cell_set,
    select_common_cell_ids,
    tessera_arm_cell_set,
)


def test_strict_aggregation_requires_all_100_subpixels() -> None:
    valid_10m = np.ones((20, 20), dtype=bool)
    valid_10m[0, 0] = False  # one gap poisons its 100 m cell
    agg = aggregate_tessera_valid_to_100m(valid_10m)
    assert agg.shape == (2, 2)
    assert agg.tolist() == [[False, True], [True, True]]


def test_intersection_is_shared_by_both_arms() -> None:
    main = np.array([[True, True], [True, False]])
    tess = np.array([[True, False], [True, True]])
    common = intersect_eligibility(main, tess)
    assert common.tolist() == [[True, False], [True, False]]
    # Both arm entry points resolve the identical intersection.
    valid_10m = np.ones((20, 20), dtype=bool)
    valid_10m[0, 1] = False  # poisons cell (0, 0) only in the tessera mask
    main_100 = np.ones((2, 2), dtype=bool)
    main_cells = main_arm_cell_set(main_100, valid_10m)
    tessera_cells = tessera_arm_cell_set(main_100, valid_10m)
    assert np.array_equal(main_cells, tessera_cells)
    assert main_cells.tolist() == [[False, True], [True, True]]
    # The patch index for both arms admits the same sorted cell IDs.
    assert select_common_cell_ids(["a", "b", "c"], ["b", "c", "d"]) == ["b", "c"]


def test_common_cell_set_end_to_end() -> None:
    main_100 = np.ones((2, 2), dtype=bool)
    main_100[1, 1] = False
    valid_10m = np.ones((20, 20), dtype=bool)
    valid_10m[0, 0] = False  # poisons cell (0, 0)
    common = common_cell_set(main_100, valid_10m)
    assert common.tolist() == [[False, True], [True, False]]
