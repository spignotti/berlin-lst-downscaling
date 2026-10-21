"""Issue #80: Stage-5 thermal objective audit statistics."""

from __future__ import annotations

import torch

from berlin_lst_downscaling.modeling.metrics import thermal_objective_batch


def test_thermal_objective_batch_reports_support_counts() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = pred + 1.0
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[:, :, :2, :] = False
    batch = thermal_objective_batch(pred, target, mask)
    assert float(batch.valid_cells) > 0.0
    assert float(batch.l1_count) == float(batch.valid_cells)
    assert float(batch.ssim_candidate) > 0.0
    assert float(batch.grad_candidate) > 0.0
    assert torch.isfinite(batch.loss)
