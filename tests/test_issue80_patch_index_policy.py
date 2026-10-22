"""Issue #80: split-aware patch-index admission policy."""

from __future__ import annotations

from berlin_lst_downscaling.data.training.patch_index import (
    MIN_ELIGIBLE_CELLS,
    MIN_ELIGIBLE_FRACTION,
    PatchIndexAdmissionPolicy,
    patch_index_policy_hash,
    patch_index_policy_hash_for,
)
from berlin_lst_downscaling.modeling.aggregation import min_eligible_cells_from_fraction


def test_legacy_hash_preserved_at_default_train_fraction() -> None:
    policy = PatchIndexAdmissionPolicy(train_min_eligible_fraction=MIN_ELIGIBLE_FRACTION)
    assert patch_index_policy_hash_for(policy) == patch_index_policy_hash()


def test_train_threshold_differs_from_val_test() -> None:
    policy = PatchIndexAdmissionPolicy(train_min_eligible_fraction=0.90)
    assert policy.min_eligible_cells_for_split("train") == min_eligible_cells_from_fraction(0.90)
    assert policy.min_eligible_cells_for_split("validation") == MIN_ELIGIBLE_CELLS
    assert policy.min_eligible_cells_for_split("test") == MIN_ELIGIBLE_CELLS
    assert patch_index_policy_hash_for(policy) != patch_index_policy_hash()


def test_train_threshold_configurable_val_test_fixed_at_244() -> None:
    """Train may use 244, 230, or 200 of 256. Validation and test stay at 244."""
    assert MIN_ELIGIBLE_CELLS == 244
    for cells in (244, 230, 200):
        policy = PatchIndexAdmissionPolicy(train_min_eligible_fraction=cells / 256)
        assert policy.min_eligible_cells_for_split("train") == cells
        assert policy.min_eligible_cells_for_split("validation") == 244
        assert policy.min_eligible_cells_for_split("test") == 244
        assert cells - 1 < policy.min_eligible_cells_for_split("train")
