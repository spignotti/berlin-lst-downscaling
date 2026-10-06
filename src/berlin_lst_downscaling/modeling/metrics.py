"""Stage-1 metrics — exact pooling, masked L1/MAE, masked SSIM (issues #18/#19).

Single implementation of the comparison arithmetic, shared by the Lightning
task and the naive baseline so the two arms cannot drift apart. Semantics are
fixed by ``docs/pseudo-pair-tensor-contract.md``.

- :func:`pool_10m_to_100m` — exact nested 10x10 block mean, no resampling.
- :func:`masked_abs_error_sums` — selects valid 100 m cells and returns
  ``(sum of absolute errors, count)``. Invalid cells are **selected out**, never
  multiplied by zero, so a non-finite value at an invalid cell cannot reach the
  numerator or the denominator.
- :class:`MaskedMAE` — cell-weighted epoch aggregation: the ratio of summed
  numerators to summed counts, never a mean of per-batch or per-patch values.
- :class:`MaskedSSIM` — secondary diagnostic only. Never a loss, never a
  checkpoint gate.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.modeling.contracts import POOL_FACTOR

# decision: local SSIM matches TorchMetrics 1.9.0 with ``gaussian_kernel=False``
# (uniform window, k1=0.01, k2=0.03, data_range clamp). TorchMetrics pads with
# reflection before the convolution; CUDA has no deterministic
# ``reflection_pad2d`` backward, so Stage-5 ``thermal_aware`` blew up under
# ``Trainer(deterministic=True)``. We compute only the valid (unpadded) window
# map — the same interior TorchMetrics keeps after crop — so training and the
# diagnostic share one pad-free path. TorchMetrics stays in the lock via
# Lightning but is not called for this map.
#
# Fixed SSIM configuration (both arms).
SSIM_WINDOW: int = 7
SSIM_PAD: int = SSIM_WINDOW // 2  # centres nearer the edge lack a full window
SSIM_DATA_RANGE: tuple[float, float] = LST_RANGE_K
SSIM_K1: float = 0.01
SSIM_K2: float = 0.03


def pool_10m_to_100m(prediction_10m: Tensor) -> Tensor:
    """Return the exact nested 10x10 block mean on the canonical 100 m grid.

    ``(B, C, H, W) -> (B, C, H // 10, W // 10)``. Blocks are the disjoint 10x10
    groups of the input grid, so the result is aligned with the native 100 m
    supervision by construction. Partial blocks are a geometry error, not a
    value to average over: the patch is exactly 160 px here.
    """
    b, c, h, w = prediction_10m.shape
    if h % POOL_FACTOR or w % POOL_FACTOR:
        raise ValueError(f"prediction extent ({h}, {w}) is not divisible by {POOL_FACTOR}")
    return prediction_10m.reshape(
        b, c, h // POOL_FACTOR, POOL_FACTOR, w // POOL_FACTOR, POOL_FACTOR
    ).mean(dim=(3, 5))


def masked_abs_error_sums(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sum |prediction - target| over valid cells, valid-cell count)``.

    Invalid cells are dropped by selection, so a non-finite target or
    prediction at an invalid cell never contributes. A selected cell that is
    non-finite raises: an eligible cell is required to carry a valid native
    target, so this is a contract violation rather than something to average.
    """
    if prediction_100m.shape != target_100m.shape or prediction_100m.shape != mask_100m.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction_100m.shape)}, "
            f"target {tuple(target_100m.shape)}, mask {tuple(mask_100m.shape)}"
        )
    valid = mask_100m.bool()
    if valid.shape[1] != 1:
        raise ValueError(f"mask must have exactly 1 channel, got {valid.shape[1]}")
    diff = (prediction_100m - target_100m).abs()
    selected = diff[valid]
    if selected.numel() == 0:
        raise ValueError("no valid 100 m cell in the batch (count(m_100) must be > 0)")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("a valid 100 m cell carries a non-finite prediction or target")
    count = torch.tensor(float(selected.numel()), dtype=selected.dtype, device=selected.device)
    return selected.sum(), count


def masked_l1_loss(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> Tensor:
    """Return the masked L1 over valid 100 m cells (the Stage-1 training loss)."""
    total, count = masked_abs_error_sums(prediction_100m, target_100m, mask_100m)
    return total / count


# Fixed Stage-5 thermal-aware mix (issue #57). Selection stays masked MAE;
# only the training objective changes. Weights are method constants, not tuned
# per ablation stage.
THERMAL_AWARE_SSIM_WEIGHT: float = 0.1
THERMAL_AWARE_GRAD_WEIGHT: float = 0.1


def _masked_gradient_l1(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> Tensor:
    """Masked L1 on finite differences where both adjacent cells are valid."""
    valid = mask_100m.bool()
    dx_pred = prediction_100m[:, :, :, 1:] - prediction_100m[:, :, :, :-1]
    dx_target = target_100m[:, :, :, 1:] - target_100m[:, :, :, :-1]
    dx_valid = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    dy_pred = prediction_100m[:, :, 1:, :] - prediction_100m[:, :, :-1, :]
    dy_target = target_100m[:, :, 1:, :] - target_100m[:, :, :-1, :]
    dy_valid = valid[:, :, 1:, :] & valid[:, :, :-1, :]

    parts: list[Tensor] = []
    if bool(dx_valid.any()):
        parts.append((dx_pred - dx_target).abs()[dx_valid])
    if bool(dy_valid.any()):
        parts.append((dy_pred - dy_target).abs()[dy_valid])
    if not parts:
        return prediction_100m.new_zeros(())
    selected = torch.cat(parts)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("a valid gradient cell carries a non-finite prediction or target")
    return selected.mean()


def thermal_aware_loss(
    prediction_100m: Tensor,
    target_100m: Tensor,
    mask_100m: Tensor,
    *,
    ssim_weight: float = THERMAL_AWARE_SSIM_WEIGHT,
    grad_weight: float = THERMAL_AWARE_GRAD_WEIGHT,
) -> Tensor:
    """Stage-5 loss: masked L1 plus structural and gradient penalties.

    The L1 term matches Stage 1–4. SSIM enters as ``1 - mean(supported local
    SSIM)`` when at least one window is supported, else zero. The gradient term
    is masked finite-difference L1. Checkpoint selection remains masked MAE.
    """
    loss = masked_l1_loss(prediction_100m, target_100m, mask_100m)
    ssim_sum, ssim_count = masked_ssim_stats(prediction_100m, target_100m, mask_100m)
    if float(ssim_count) > 0.0:
        loss = loss + ssim_weight * (1.0 - (ssim_sum / ssim_count))
    loss = loss + grad_weight * _masked_gradient_l1(prediction_100m, target_100m, mask_100m)
    return loss


def _uniform_ssim_map(prediction_100m: Tensor, target_100m: Tensor) -> Tensor:
    """Return the valid-convolution local SSIM map ``(B, 1, H-k+1, W-k+1)``.

    Matches TorchMetrics 1.9.0 with ``gaussian_kernel=False`` on the interior
    that remains after reflection-pad + crop. No padding, so CUDA backward stays
    compatible with ``torch.use_deterministic_algorithms(True)``.
    """
    low, high = SSIM_DATA_RANGE
    data_range = high - low
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    preds = torch.clamp(prediction_100m, min=low, max=high)
    target = torch.clamp(target_100m, min=low, max=high)
    channel = int(preds.shape[1])
    kernel = torch.ones(
        (channel, 1, SSIM_WINDOW, SSIM_WINDOW), dtype=preds.dtype, device=preds.device
    ) / float(SSIM_WINDOW * SSIM_WINDOW)
    # TorchMetrics packs the five moments into one grouped conv; keep the same
    # algebra with five calls so the graph stays obvious and pad-free.
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
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sum of supported local SSIM values, supported-window count)``.

    A centre counts only when its full ``SSIM_WINDOW`` neighbourhood lies inside
    the patch **and** every cell in that neighbourhood is valid. Invalid cells
    are replaced by zero in a temporary copy so the convolution never sees a
    NaN, and every window touching such a cell is then discarded by the support
    test, so the substitution cannot influence a reported value.

    Scalar reduction over unsupported centres is intentionally unused: the
    contract forbids averaging them in.
    """
    valid = mask_100m.bool()
    if prediction_100m.shape[1] != 1 or valid.shape != prediction_100m.shape:
        raise ValueError("SSIM expects a single-channel prediction and a matching mask")
    if prediction_100m.shape[-1] < SSIM_WINDOW or prediction_100m.shape[-2] < SSIM_WINDOW:
        raise ValueError("patch is smaller than the SSIM window")
    if not valid.any():
        return torch.zeros((), dtype=prediction_100m.dtype), torch.zeros(())
    # FP32 boundary: the SSIM convolution must execute in float32 with autocast
    # disabled. Kelvin-scale squared moments overflow float16, which produced
    # NaN SSIM under mixed-precision validation (J4 evidence). A bare
    # ``.float()`` cast is insufficient: convolution autocast would downcast
    # again inside an enabled region.
    with torch.autocast(device_type=prediction_100m.device.type, enabled=False):
        safe_pred = torch.where(valid, prediction_100m, torch.zeros_like(prediction_100m)).float()
        safe_target = torch.where(valid, target_100m, torch.zeros_like(target_100m)).float()
        ssim_map = _uniform_ssim_map(safe_pred, safe_target)
    # A centre is supported only when its whole window is valid; avg_pool with
    # the SSIM window already drops the incomplete border.
    supported = F.avg_pool2d(valid.to(ssim_map.dtype), kernel_size=SSIM_WINDOW, stride=1) == 1.0
    values = ssim_map[supported]
    if values.numel() == 0:
        return torch.zeros((), dtype=ssim_map.dtype, device=ssim_map.device), torch.zeros(())
    if not bool(torch.isfinite(values).all()):
        raise ValueError("a supported SSIM window carries a non-finite value")
    return values.sum(), torch.tensor(
        float(values.numel()), dtype=ssim_map.dtype, device=ssim_map.device
    )


class MaskedMAE(Metric):
    """Cell-weighted masked MAE across an epoch.

    Accumulates the numerator and the valid-cell count over every batch, so
    patches with unequal valid-cell counts do not carry equal weight.
    """

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


class MaskedSSIM(Metric):
    """Supported-window mean SSIM at 100 m, or NaN when nothing is supported.

    ``update`` takes the ``(sum, count)`` pair from a single
    :func:`masked_ssim_stats` call rather than recomputing the SSIM map, so the
    convolution runs once per batch while the metric still aggregates correctly
    across an epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("window_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, total: Tensor, count: Tensor) -> None:
        """Accumulate one batch's ``(SSIM sum, supported-window count)``."""
        self.ssim_sum = self.ssim_sum + total
        self.window_count = self.window_count + count

    def compute(self) -> Tensor:
        if float(self.window_count) <= 0.0:
            # Reported as unavailable, never as zero.
            return torch.tensor(float("nan"), device=self.ssim_sum.device)
        return self.ssim_sum / self.window_count


class SupportedWindows(Metric):
    """Count of SSIM windows with full valid support over an epoch.

    Reported next to the SSIM value so an unsupported (NaN) SSIM is
    distinguishable from a genuinely poor one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("window_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, count: Tensor) -> None:
        """Accumulate one batch's supported-window count."""
        self.window_count = self.window_count + count

    def compute(self) -> Tensor:
        return self.window_count


class ValidCells(Metric):
    """Sum of valid 100 m cells over an epoch — the masked MAE's support.

    The masked MAE's own ``valid_cells`` state is consumed (and reset) when the
    epoch value is computed, so it cannot be read back as run evidence. This
    separate accumulator keeps the support observable per epoch alongside the
    metric it weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("cell_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mask_100m: Tensor) -> None:
        """Accumulate one batch's valid-cell count."""
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
    "SupportedWindows",
    "ValidCells",
    "masked_abs_error_sums",
    "masked_l1_loss",
    "masked_ssim_stats",
    "pool_10m_to_100m",
    "thermal_aware_loss",
]
