"""Masked loss and metrics: invalid cells leave both the numerator and the denominator."""

from __future__ import annotations

import pytest
import torch

from berlin_lst_downscaling.modeling.metrics import (
    MaskedMAE,
    MaskedSSIM,
    masked_l1_loss,
    masked_ssim_stats,
)


def test_masked_l1_drops_invalid_cells_including_nan() -> None:
    """A masked cell, even with a NaN target or a huge prediction, does not enter the loss."""
    prediction = torch.tensor([[[[10.0, 1.0e9]]]])
    target = torch.tensor([[[[0.0, float("nan")]]]])
    mask = torch.tensor([[[[True, False]]]])

    assert float(masked_l1_loss(prediction, target, mask)) == pytest.approx(10.0)

    poisoned = prediction.clone()
    poisoned[0, 0, 0, 1] = 0.0
    assert float(masked_l1_loss(poisoned, target, mask)) == pytest.approx(10.0)


def test_masked_l1_rejects_an_empty_or_non_finite_selection() -> None:
    """An all-invalid mask, or a NaN inside a valid cell, is an error."""
    prediction = torch.tensor([[[[10.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    with pytest.raises(ValueError, match="no valid"):
        masked_l1_loss(prediction, target, torch.zeros_like(mask))

    bad_target = target.clone()
    bad_target[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        masked_l1_loss(prediction, bad_target, mask)


def test_masked_mae_weights_cells_not_batches() -> None:
    """Epoch MAE is the summed error over the summed count, not the mean of batch means."""
    metric = MaskedMAE()
    metric.update(
        torch.tensor([[[[3.0, 0.0]]]]),
        torch.tensor([[[[0.0, float("nan")]]]]),
        torch.tensor([[[[True, False]]]]),
    )
    metric.update(
        torch.tensor([[[[1.0, 1.0], [1.0, 100.0]]]]),
        torch.tensor([[[[0.0, 0.0], [0.0, float("nan")]]]]),
        torch.tensor([[[[True, True], [True, False]]]]),
    )
    # Errors 3 + 1 + 1 + 1 over 4 cells. The mean of the batch means would be 2.
    assert float(metric.compute()) == pytest.approx(1.5)


def test_ssim_counts_only_fully_valid_windows() -> None:
    """Identical valid windows score 1. A window that touches an invalid cell is dropped."""
    prediction = torch.full((1, 1, 7, 7), 300.0)
    target = prediction.clone()
    mask = torch.ones_like(prediction, dtype=torch.bool)
    total, count = masked_ssim_stats(prediction, target, mask)
    assert float(count) == pytest.approx(1.0)
    assert float(total) == pytest.approx(1.0)

    mask[0, 0, 0, 0] = False
    target[0, 0, 0, 0] = float("nan")
    total_bad, count_bad = masked_ssim_stats(prediction, target, mask)
    assert float(count_bad) == pytest.approx(0.0)

    metric = MaskedSSIM()
    metric.update(total_bad, count_bad)
    assert torch.isnan(metric.compute())
