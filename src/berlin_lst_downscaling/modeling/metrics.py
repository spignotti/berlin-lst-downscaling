"""Stage-1 metrics — exact pooling, masked L1/MAE, masked SSIM (issues #18/#19/#80).

Single implementation of the comparison arithmetic, shared by the Lightning
task and the naive baseline so the two arms cannot drift apart. Semantics are
fixed by ``docs/pseudo-pair-tensor-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.modeling.aggregation import (
    AGGREGATION_ARITHMETIC,
    AggregationMethod,
)
from berlin_lst_downscaling.modeling.aggregation import (
    pool_10m_to_100m as _pool_10m_to_100m,
)

SSIM_WINDOW: int = 7
SSIM_PAD: int = SSIM_WINDOW // 2
SSIM_DATA_RANGE: tuple[float, float] = LST_RANGE_K
SSIM_K1: float = 0.01
SSIM_K2: float = 0.03


def pool_10m_to_100m(
    prediction_10m: Tensor,
    method: AggregationMethod = AGGREGATION_ARITHMETIC,
    mask_100m: Tensor | None = None,
) -> Tensor:
    """Exact nested 10×10 block reduction on the canonical 100 m grid."""
    return _pool_10m_to_100m(prediction_10m, method, mask_100m)


def masked_abs_error_sums(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sum |prediction - target| over valid cells, valid-cell count)``."""
    if prediction_100m.shape != target_100m.shape or prediction_100m.shape != mask_100m.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction_100m.shape)}, "
            f"target {tuple(target_100m.shape)}, mask {tuple(mask_100m.shape)}"
        )
    valid = mask_100m.bool()
    if valid.shape[1] != 1:
        raise ValueError(f"mask must have exactly 1 channel, got {valid.shape[1]}")
    pred_sel = prediction_100m[valid]
    target_sel = target_100m[valid]
    if pred_sel.numel() == 0:
        raise ValueError("no valid 100 m cell in the batch (count(m_100) must be > 0)")
    if not bool(torch.isfinite(pred_sel).all()) or not bool(torch.isfinite(target_sel).all()):
        raise ValueError("a valid 100 m cell carries a non-finite prediction or target")
    diff = (pred_sel - target_sel).abs()
    count = torch.tensor(float(pred_sel.numel()), dtype=diff.dtype, device=diff.device)
    return diff.sum(), count


def masked_l1_loss(prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor) -> Tensor:
    """Return the masked L1 over valid 100 m cells (the Stage-1 training loss)."""
    total, count = masked_abs_error_sums(prediction_100m, target_100m, mask_100m)
    return total / count


THERMAL_AWARE_SSIM_WEIGHT: float = 0.1
THERMAL_AWARE_GRAD_WEIGHT: float = 0.1


def _ssim_candidate_window_count(prediction_100m: Tensor) -> int:
    b, _, h, w = prediction_100m.shape
    if h < SSIM_WINDOW or w < SSIM_WINDOW:
        return 0
    return b * (h - SSIM_WINDOW + 1) * (w - SSIM_WINDOW + 1)


def _gradient_candidate_pair_count(prediction_100m: Tensor) -> int:
    b, _, h, w = prediction_100m.shape
    return b * (h * (w - 1) + (h - 1) * w)


def masked_gradient_stats(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(abs-diff sum, supported-pair count, candidate-pair count)``."""
    valid = mask_100m.bool()
    candidate = float(_gradient_candidate_pair_count(prediction_100m))
    candidate_t = prediction_100m.new_tensor(candidate)

    dx_valid = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    dy_valid = valid[:, :, 1:, :] & valid[:, :, :-1, :]

    parts: list[Tensor] = []
    if bool(dx_valid.any()):
        pred_dx = prediction_100m[:, :, :, 1:][dx_valid]
        pred_dx0 = prediction_100m[:, :, :, :-1][dx_valid]
        tgt_dx = target_100m[:, :, :, 1:][dx_valid]
        tgt_dx0 = target_100m[:, :, :, :-1][dx_valid]
        parts.append((pred_dx - pred_dx0 - (tgt_dx - tgt_dx0)).abs())
    if bool(dy_valid.any()):
        pred_dy = prediction_100m[:, :, 1:, :][dy_valid]
        pred_dy0 = prediction_100m[:, :, :-1, :][dy_valid]
        tgt_dy = target_100m[:, :, 1:, :][dy_valid]
        tgt_dy0 = target_100m[:, :, :-1, :][dy_valid]
        parts.append((pred_dy - pred_dy0 - (tgt_dy - tgt_dy0)).abs())

    if not parts:
        zero = prediction_100m.new_zeros(())
        return zero, zero, candidate_t

    selected = torch.cat(parts)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("a valid gradient cell carries a non-finite prediction or target")
    count = torch.tensor(float(selected.numel()), dtype=selected.dtype, device=selected.device)
    return selected.sum(), count, candidate_t


def _masked_gradient_l1(prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor) -> Tensor:
    """Masked L1 on finite differences where both adjacent cells are valid."""
    total, count, _ = masked_gradient_stats(prediction_100m, target_100m, mask_100m)
    if float(count) <= 0.0:
        return prediction_100m.new_zeros(())
    return total / count


@dataclass(frozen=True)
class ThermalObjectiveBatch:
    """Per-batch Stage-5 objective pieces and detached audit statistics."""

    loss: Tensor
    l1_unweighted: Tensor
    ssim_penalty_unweighted: Tensor
    gradient_unweighted: Tensor
    l1_sum: Tensor
    l1_count: Tensor
    ssim_sum: Tensor
    ssim_supported: Tensor
    ssim_candidate: Tensor
    grad_sum: Tensor
    grad_supported: Tensor
    grad_candidate: Tensor
    valid_cells: Tensor
    total_cells: Tensor


def thermal_objective_batch(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    *,
    ssim_weight: float = THERMAL_AWARE_SSIM_WEIGHT,
    grad_weight: float = THERMAL_AWARE_GRAD_WEIGHT,
) -> ThermalObjectiveBatch:
    """Compute the differentiable batch objective and detached support statistics."""
    l1_sum, l1_count = masked_abs_error_sums(prediction_100m, target_100m, mask_100m)
    l1_unweighted = l1_sum / l1_count

    # Training SSIM stays clamped. Patches with ssim_supported == 0 stay in L1
    # and are omitted from this mean (they are not scored as SSIM 0).
    ssim_sum, ssim_supported = masked_ssim_stats(
        prediction_100m, target_100m, mask_100m, clamp_predictions=True
    )
    ssim_candidate = prediction_100m.new_tensor(
        float(_ssim_candidate_window_count(prediction_100m))
    )
    if float(ssim_supported) > 0.0:
        ssim_penalty = 1.0 - (ssim_sum / ssim_supported)
        ssim_term = ssim_weight * ssim_penalty
    else:
        ssim_penalty = prediction_100m.new_zeros(())
        ssim_term = ssim_penalty

    grad_sum, grad_supported, grad_candidate = masked_gradient_stats(
        prediction_100m, target_100m, mask_100m
    )
    if float(grad_supported) > 0.0:
        gradient_unweighted = grad_sum / grad_supported
        grad_term = grad_weight * gradient_unweighted
    else:
        gradient_unweighted = prediction_100m.new_zeros(())
        grad_term = gradient_unweighted

    loss = l1_unweighted + ssim_term + grad_term
    valid_cells = mask_100m.bool().sum().to(dtype=l1_sum.dtype)
    total_cells = mask_100m.new_tensor(float(mask_100m.numel()))

    return ThermalObjectiveBatch(
        loss=loss,
        l1_unweighted=l1_unweighted,
        ssim_penalty_unweighted=ssim_penalty,
        gradient_unweighted=gradient_unweighted,
        l1_sum=l1_sum.detach(),
        l1_count=l1_count.detach(),
        ssim_sum=ssim_sum.detach(),
        ssim_supported=ssim_supported.detach(),
        ssim_candidate=ssim_candidate.detach(),
        grad_sum=grad_sum.detach(),
        grad_supported=grad_supported.detach(),
        grad_candidate=grad_candidate.detach(),
        valid_cells=valid_cells.detach(),
        total_cells=total_cells.detach(),
    )


def thermal_aware_loss(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    *,
    ssim_weight: float = THERMAL_AWARE_SSIM_WEIGHT,
    grad_weight: float = THERMAL_AWARE_GRAD_WEIGHT,
) -> Tensor:
    """Stage-5 loss: masked L1 plus structural and gradient penalties."""
    return thermal_objective_batch(
        prediction_100m,
        target_100m,
        mask_100m,
        ssim_weight=ssim_weight,
        grad_weight=grad_weight,
    ).loss


def _uniform_ssim_map(
    prediction_100m: Tensor,
    target_100m: Tensor,
    *,
    clamp_predictions: bool,
) -> Tensor:
    low, high = SSIM_DATA_RANGE
    data_range = high - low
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    # Training loss clamps predictions into the Kelvin data range. Evaluation
    # passes the pooled prediction through unclamped.
    preds = (
        torch.clamp(prediction_100m, min=low, max=high) if clamp_predictions else prediction_100m
    )
    target = torch.clamp(target_100m, min=low, max=high)
    channel = int(preds.shape[1])
    kernel = torch.ones(
        (channel, 1, SSIM_WINDOW, SSIM_WINDOW), dtype=preds.dtype, device=preds.device
    ) / float(SSIM_WINDOW * SSIM_WINDOW)
    mu_pred = F.conv2d(preds, kernel, groups=channel)
    mu_target = F.conv2d(target, kernel, groups=channel)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = torch.clamp(
        F.conv2d(preds * preds, kernel, groups=channel) - mu_pred_sq, min=0.0
    )
    sigma_target_sq = torch.clamp(
        F.conv2d(target * target, kernel, groups=channel) - mu_target_sq, min=0.0
    )
    sigma_pred_target = F.conv2d(preds * target, kernel, groups=channel) - mu_pred_target
    upper = 2.0 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2
    return ((2.0 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)


def masked_ssim_stats(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    *,
    clamp_predictions: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return ``(sum of supported local SSIM values, supported-window count)``.

    A patch with no fully valid SSIM window is left out of both the sum and the
    count. It does not enter the batch mean as a zero. ``clamp_predictions``
    is the training-loss default; evaluation sets it false.
    """
    valid = mask_100m.bool()
    if prediction_100m.shape[1] != 1 or valid.shape != prediction_100m.shape:
        raise ValueError("SSIM expects a single-channel prediction and a matching mask")
    if prediction_100m.shape[-1] < SSIM_WINDOW or prediction_100m.shape[-2] < SSIM_WINDOW:
        raise ValueError("patch is smaller than the SSIM window")
    device = prediction_100m.device
    dtype = prediction_100m.dtype
    if not valid.any():
        return torch.zeros((), dtype=dtype, device=device), torch.zeros((), device=device)
    with torch.autocast(device_type=device.type, enabled=False):
        safe_pred = torch.where(valid, prediction_100m, torch.zeros_like(prediction_100m)).float()
        safe_target = torch.where(valid, target_100m, torch.zeros_like(target_100m)).float()
        ssim_map = _uniform_ssim_map(safe_pred, safe_target, clamp_predictions=clamp_predictions)
    supported = F.avg_pool2d(valid.to(ssim_map.dtype), kernel_size=SSIM_WINDOW, stride=1) == 1.0
    per_count = supported.flatten(1).sum(dim=1)
    active = per_count > 0
    if not bool(active.any()):
        return torch.zeros((), dtype=ssim_map.dtype, device=device), torch.zeros((), device=device)
    values = ssim_map[supported]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("a supported SSIM window carries a non-finite value")
    per_sum = (ssim_map * supported.to(dtype=ssim_map.dtype)).flatten(1).sum(dim=1)
    total = per_sum[active].sum()
    count = per_count[active].sum().to(dtype=ssim_map.dtype)
    return total, count


def prediction_outside_lst_range_counts(prediction: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(count outside LST_RANGE_K, pixel count)`` for a prediction map.

    Non-finite pixels count as outside the physical range. The share is the
    ratio of the two counts.
    """
    low, high = LST_RANGE_K
    outside = (~torch.isfinite(prediction)) | (prediction < low) | (prediction > high)
    total = prediction.new_tensor(float(prediction.numel()))
    count = outside.sum().to(dtype=prediction.dtype)
    return count, total


def masked_ssim_support(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(ssim sum, supported windows, candidate windows)``."""
    total, supported = masked_ssim_stats(prediction_100m, target_100m, mask_100m)
    candidate = prediction_100m.new_tensor(float(_ssim_candidate_window_count(prediction_100m)))
    return total, supported, candidate


class MaskedMAE(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("abs_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_cells", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor) -> None:
        total, count = masked_abs_error_sums(prediction_100m, target_100m, mask_100m)
        self.abs_error_sum = self.abs_error_sum + total
        self.valid_cells = self.valid_cells + count

    def compute(self) -> Tensor:
        if float(self.valid_cells) <= 0.0:
            raise ValueError("masked MAE accumulated no valid cell")
        return self.abs_error_sum / self.valid_cells


class SupportWeightedSum(Metric):
    """Accumulate a numerator and denominator for support-weighted epoch means."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, numerator: Tensor, denominator: Tensor) -> None:
        self.numerator = self.numerator + numerator
        self.denominator = self.denominator + denominator

    def compute(self) -> Tensor:
        if float(self.denominator) <= 0.0:
            return torch.tensor(float("nan"), device=self.numerator.device)
        return self.numerator / self.denominator


class ScaledSupportWeightedSum(SupportWeightedSum):
    """Support-weighted mean scaled by a fixed objective weight."""

    def __init__(self, weight: float) -> None:
        super().__init__()
        self._weight = float(weight)

    def compute(self) -> Tensor:
        base = super().compute()
        if not bool(torch.isfinite(base).all()):
            return base
        return base * self._weight


class SupportCounter(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: Tensor) -> None:
        self.total = self.total + value

    def compute(self) -> Tensor:
        return self.total


class SupportRatio(Metric):
    """Epoch ratio of summed numerator to summed denominator."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, numerator: Tensor, denominator: Tensor) -> None:
        self.numerator = self.numerator + numerator
        self.denominator = self.denominator + denominator

    def compute(self) -> Tensor:
        if float(self.denominator) <= 0.0:
            return torch.tensor(float("nan"), device=self.numerator.device)
        return self.numerator / self.denominator


class MaskedSSIM(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("window_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, total: Tensor, count: Tensor) -> None:
        self.ssim_sum = self.ssim_sum + total
        self.window_count = self.window_count + count

    def compute(self) -> Tensor:
        if float(self.window_count) <= 0.0:
            return torch.tensor(float("nan"), device=self.ssim_sum.device)
        return self.ssim_sum / self.window_count


class SupportedWindows(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("window_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, count: Tensor) -> None:
        self.window_count = self.window_count + count

    def compute(self) -> Tensor:
        return self.window_count


class ValidCells(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("cell_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mask_100m: Tensor) -> None:
        self.cell_count = self.cell_count + mask_100m.sum()

    def compute(self) -> Tensor:
        return self.cell_count


__all__ = [
    "SSIM_DATA_RANGE",
    "SSIM_PAD",
    "SSIM_WINDOW",
    "THERMAL_AWARE_GRAD_WEIGHT",
    "THERMAL_AWARE_SSIM_WEIGHT",
    "MaskedMAE",
    "MaskedSSIM",
    "ScaledSupportWeightedSum",
    "SupportCounter",
    "SupportRatio",
    "SupportWeightedSum",
    "SupportedWindows",
    "ThermalObjectiveBatch",
    "ValidCells",
    "masked_abs_error_sums",
    "masked_gradient_stats",
    "masked_l1_loss",
    "masked_ssim_stats",
    "masked_ssim_support",
    "pool_10m_to_100m",
    "prediction_outside_lst_range_counts",
    "thermal_aware_loss",
    "thermal_objective_batch",
]
