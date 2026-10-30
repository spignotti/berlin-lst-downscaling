"""Stage-5 loss audit: SSIM numerics at Kelvin scale, support handling, and term scale."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from berlin_lst_downscaling.modeling.metrics import (
    SSIM_K1,
    SSIM_K2,
    SSIM_LEGACY_DATA_RANGE_K,
    THERMAL_AWARE_GRAD_WEIGHT,
    THERMAL_AWARE_SSIM_WEIGHT,
    masked_ssim_stats,
    pool_10m_to_100m,
    thermal_objective_batch,
)


def _smooth_field(
    generator: torch.Generator, shape: tuple[int, ...], std: float, dtype: torch.dtype
) -> Tensor:
    """Gaussian-smoothed noise rescaled to the requested standard deviation."""
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    taps = torch.exp(-0.5 * (torch.arange(-4, 5, dtype=torch.float64) / 2.0) ** 2)
    kernel = (taps[:, None] * taps[None, :]) / (taps.sum() ** 2)
    smooth = F.conv2d(F.pad(noise, (4, 4, 4, 4), mode="reflect"), kernel[None, None])
    smooth = (smooth - smooth.mean()) / smooth.std()
    return (std * smooth).to(dtype)


def _reference_ssim_fp64(prediction: Tensor, target: Tensor) -> float:
    c1 = (SSIM_K1 * SSIM_LEGACY_DATA_RANGE_K) ** 2
    c2 = (SSIM_K2 * SSIM_LEGACY_DATA_RANGE_K) ** 2
    x = prediction.double().flatten()
    y = target.double().flatten()
    mu_x, mu_y = x.mean(), y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den)


def test_ssim_fp32_matches_fp64_reference_at_kelvin_scale() -> None:
    """fp32 SSIM of one 7x7 window at ~300 K agrees with an fp64 reference."""
    generator = torch.Generator().manual_seed(0)
    max_error = 0.0
    for _ in range(40):
        target64 = 300.0 + _smooth_field(generator, (1, 1, 7, 7), 0.7, torch.float64)
        pert = _smooth_field(generator, (1, 1, 7, 7), 0.3, torch.float64)
        pred64 = target64 + pert
        prediction = pred64.float()
        target = target64.float()
        mask = torch.ones_like(prediction, dtype=torch.bool)
        total, count = masked_ssim_stats(
            prediction, target, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K
        )
        assert int(count) == 1
        expected = _reference_ssim_fp64(prediction, target)
        max_error = max(max_error, abs(float(total) - expected))
    print(f"max abs SSIM error: {max_error:.3e}")
    assert max_error < 1e-5


def test_ssim_ignores_non_finite_values_outside_supported_windows() -> None:
    """An unclamped huge prediction in no fully valid window cannot poison the sum."""
    mask = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    mask[..., :7, :7] = True
    mask[..., 15, 15] = True
    prediction = torch.full((1, 1, 16, 16), 300.0)
    prediction[..., 15, 15] = 1e20
    target = torch.full((1, 1, 16, 16), 300.0)

    total, count = masked_ssim_stats(
        prediction,
        target,
        mask,
        clamp_predictions=False,
        data_range_k=SSIM_LEGACY_DATA_RANGE_K,
    )

    assert int(count) == 1
    assert torch.isfinite(total)
    assert float(total) == 1.0


def _invalid_mask() -> Tensor:
    mask = torch.ones((2, 1, 16, 16), dtype=torch.bool)
    mask[0, :, 3, 4] = False
    mask[0, :, 10, 10:12] = False
    mask[1, :, :, 8:] = False
    return mask


def test_thermal_objective_ignores_invalid_cell_values() -> None:
    """Loss and gradient do not depend on values in invalid cells."""
    generator = torch.Generator().manual_seed(1)
    mask = _invalid_mask()
    target = 300.0 + _smooth_field(generator, (2, 1, 16, 16), 1.0, torch.float32)
    base = target + 0.5 * torch.randn((2, 1, 16, 16), generator=generator)

    def run(pred_invalid: float | None, target_invalid: float | None) -> tuple[Tensor, Tensor]:
        pred = base.clone()
        tgt = target.clone()
        if pred_invalid is not None:
            pred[~mask] = pred_invalid
        if target_invalid is not None:
            tgt[~mask] = target_invalid
        pred.requires_grad_(True)
        loss = thermal_objective_batch(pred, tgt, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K).loss
        loss.backward()
        assert pred.grad is not None
        return loss.detach(), pred.grad

    loss_a, grad_a = run(None, None)
    loss_b, grad_b = run(-5000.0, float("nan"))

    assert torch.isfinite(loss_b)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(grad_a, grad_b)
    assert torch.isfinite(grad_b).all()
    assert torch.all(grad_b[~mask] == 0.0)


def test_thermal_objective_without_supported_window_is_l1_plus_gradient() -> None:
    """With no fully valid 7x7 window the SSIM term drops out entirely."""
    generator = torch.Generator().manual_seed(2)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    mask[..., ::4, :] = False
    target = 300.0 + _smooth_field(generator, (1, 1, 16, 16), 1.0, torch.float32)
    prediction = target + 0.5 * torch.randn((1, 1, 16, 16), generator=generator)

    out = thermal_objective_batch(prediction, target, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K)

    assert float(out.ssim_supported) == 0.0
    assert float(out.ssim_penalty_unweighted) == 0.0
    expected = out.l1_unweighted + THERMAL_AWARE_GRAD_WEIGHT * out.gradient_unweighted
    assert torch.allclose(out.loss, expected)


def _grad_norm(term: Tensor, prediction: Tensor) -> float:
    (grad,) = torch.autograd.grad(term, prediction, retain_graph=True)
    return float(grad.norm())


def test_ssim_term_is_small_against_l1_on_realistic_patches() -> None:
    """Documents the audit finding: the weighted SSIM term is tiny next to L1."""
    generator = torch.Generator().manual_seed(3)
    target = 300.0 + _smooth_field(generator, (4, 1, 16, 16), 1.0, torch.float32)
    coarse = F.avg_pool2d(target, kernel_size=8).repeat_interleave(8, 2).repeat_interleave(8, 3)
    prediction = (coarse + 1.0).requires_grad_(True)
    mask = torch.ones_like(target, dtype=torch.bool)

    out = thermal_objective_batch(prediction, target, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K)

    ssim_term = THERMAL_AWARE_SSIM_WEIGHT * out.ssim_penalty_unweighted
    grad_term = THERMAL_AWARE_GRAD_WEIGHT * out.gradient_unweighted
    assert float(ssim_term) < 0.01 * float(out.l1_unweighted)

    g_l1 = _grad_norm(out.l1_unweighted, prediction)
    g_ssim = _grad_norm(ssim_term, prediction)
    g_grad = _grad_norm(grad_term, prediction)
    assert g_ssim < 0.01 * g_l1
    assert 0.01 * g_l1 < g_grad < 0.5 * g_l1


def test_thermal_objective_cannot_see_structure_inside_a_100m_block() -> None:
    """All terms act on the arithmetic 100 m pool, so a zero-mean 10 m pattern is free."""
    generator = torch.Generator().manual_seed(4)
    target = 300.0 + _smooth_field(generator, (1, 1, 16, 16), 1.0, torch.float32)
    mask = torch.ones_like(target, dtype=torch.bool)
    base_10m = target.repeat_interleave(10, 2).repeat_interleave(10, 3) + 0.5
    checker = (torch.arange(10)[:, None] + torch.arange(10)[None, :]) % 2 * 2.0 - 1.0
    pattern = 5.0 * checker.repeat(16, 16)[None, None]

    plain = thermal_objective_batch(
        pool_10m_to_100m(base_10m), target, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K
    )
    textured = thermal_objective_batch(
        pool_10m_to_100m(base_10m + pattern),
        target,
        mask,
        data_range_k=SSIM_LEGACY_DATA_RANGE_K,
    )

    assert torch.allclose(plain.loss, textured.loss, atol=1e-5)


def test_ssim_reference_shift_keeps_target_clamp_for_huge_targets() -> None:
    """A huge finite target is still clamped to the data range, as before the shift."""
    mask = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    prediction = torch.full((1, 1, 8, 8), 300.0)
    target = torch.full((1, 1, 8, 8), 300.0)
    target[..., 0, 0] = 1e25

    total, count = masked_ssim_stats(
        prediction, target, mask, data_range_k=SSIM_LEGACY_DATA_RANGE_K
    )

    assert int(count) == 4
    assert torch.isfinite(total)
