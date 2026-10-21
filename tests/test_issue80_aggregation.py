"""Issue #80: Kelvin aggregation and locked-profile guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.aggregation import (
    AGGREGATION_ARITHMETIC,
    AggregationContractError,
    pool_10m_to_100m,
    pool_block_means_from_accumulators,
)
from berlin_lst_downscaling.modeling.guards import guard_modeling_config

_REPO = Path(__file__).resolve().parents[1]
_CONFIG = str(_REPO / "configs" / "modeling")


def test_t4_pooling_homogeneous_block_matches_closed_form() -> None:
    t = 300.0
    block = torch.full((1, 1, 10, 10), t)
    arith = float(pool_10m_to_100m(block, AGGREGATION_ARITHMETIC))
    t4 = float(pool_10m_to_100m(block, "t4"))
    assert arith == pytest.approx(t)
    assert t4 == pytest.approx(t)


def test_t4_pooling_is_at_least_arithmetic_within_one_block() -> None:
    warm = torch.full((50,), 310.0)
    cold = torch.full((50,), 290.0)
    block = torch.cat([warm, cold]).reshape(1, 1, 10, 10)
    arith = float(pool_10m_to_100m(block, AGGREGATION_ARITHMETIC))
    t4 = float(pool_10m_to_100m(block, "t4"))
    assert t4 >= arith


def test_t4_mask_ignores_non_positive_pixels_in_excluded_blocks() -> None:
    block = torch.full((1, 1, 10, 20), -5.0)
    block[0, 0, :, :10] = 300.0
    mask = torch.zeros((1, 1, 1, 2), dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    pooled = pool_10m_to_100m(block, "t4", mask)
    assert float(pooled[0, 0, 0, 0]) == pytest.approx(300.0)


def test_t4_mask_raises_for_non_positive_contributing_block() -> None:
    block = torch.full((1, 1, 10, 10), 300.0)
    block[0, 0, 0, 0] = -1.0
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    with pytest.raises(AggregationContractError):
        pool_10m_to_100m(block, "t4", mask)


def test_prior_t4_from_accumulators() -> None:
    counts = np.array([[2]], dtype=np.int64)
    sum_t = np.array([[600.0]], dtype=np.float64)
    sum_t4 = np.array([[2 * 300.0**4]], dtype=np.float64)
    out = pool_block_means_from_accumulators(sum_t, sum_t4, counts, method="t4")
    assert float(out[0, 0]) == pytest.approx(300.0, rel=1e-5)


def test_locked_configs_reject_t4_aggregation() -> None:
    with initialize_config_dir(config_dir=_CONFIG, version_base=None):
        cfg = compose(config_name="stage1_locked", overrides=["aggregation=t4"])
    with pytest.raises(ValueError, match="arithmetic"):
        guard_modeling_config(cfg, "stage1_locked")
