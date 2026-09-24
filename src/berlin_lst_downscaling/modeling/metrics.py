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
from torchmetrics.functional.image import (
    structural_similarity_index_measure as _ssim,
)

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.modeling.contracts import POOL_FACTOR

# decision: SSIM comes from TorchMetrics, because it is already in the lock
# through Lightning and the pinned 1.9.0 exposes the full local map via
# ``return_full_image``. Alternatives rejected: a new scikit-image dependency,
# and a hand-rolled SSIM formula.
# searched for library: TorchMetrics and scikit-image were checked. Context7 was
# unavailable, so the version-pinned upstream source
# (github.com/Lightning-AI/torchmetrics, tag v1.9.0, functional/image/ssim.py) was
# read instead. TorchMetrics' SSIM has no mask argument, so the window-support
# test lives in this module, not in the library call.
#
# Fixed SSIM configuration (both arms).
SSIM_WINDOW: int = 7
SSIM_PAD: int = SSIM_WINDOW // 2  # centres nearer the edge lack a full window
SSIM_DATA_RANGE: tuple[float, float] = LST_RANGE_K


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


def masked_ssim_stats(
    prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sum of supported local SSIM values, supported-window count)``.

    A centre counts only when its full ``SSIM_WINDOW`` neighbourhood lies inside
    the patch **and** every cell in that neighbourhood is valid. Invalid cells
    are replaced by zero in a temporary copy so the library never sees a NaN,
    and every window touching such a cell is then discarded by the support
    test, so the substitution cannot influence a reported value.

    TorchMetrics' own scalar reduction is intentionally unused: it averages over
    unsupported centres, which the contract forbids.
    """
    valid = mask_100m.bool()
    if prediction_100m.shape[1] != 1 or valid.shape != prediction_100m.shape:
        raise ValueError("SSIM expects a single-channel prediction and a matching mask")
    if prediction_100m.shape[-1] < SSIM_WINDOW or prediction_100m.shape[-2] < SSIM_WINDOW:
        raise ValueError("patch is smaller than the SSIM window")
    if not valid.any():
        return torch.zeros((), dtype=prediction_100m.dtype), torch.zeros(())
    # Temporary computation copies: zero where invalid so the conv never sees NaN.
    safe_pred = torch.where(valid, prediction_100m, torch.zeros_like(prediction_100m))
    safe_target = torch.where(valid, target_100m, torch.zeros_like(target_100m))
    _, ssim_map = _ssim(
        safe_pred,
        safe_target,
        gaussian_kernel=False,
        kernel_size=SSIM_WINDOW,
        data_range=SSIM_DATA_RANGE,
        return_full_image=True,
    )
    # A centre is supported only when its whole window is valid, which also
    # excludes the pad-wide border whose window would fall outside the patch.
    supported = F.avg_pool2d(valid.to(ssim_map.dtype), kernel_size=SSIM_WINDOW, stride=1) == 1.0
    inner = ssim_map[
        :, :, SSIM_PAD : SSIM_PAD + supported.shape[-2], SSIM_PAD : SSIM_PAD + supported.shape[-1]
    ]
    values = inner[supported]
    if values.numel() == 0:
        return torch.zeros((), dtype=ssim_map.dtype, device=ssim_map.device), torch.zeros(())
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


__all__ = [
    "SSIM_DATA_RANGE",
    "SSIM_PAD",
    "SSIM_WINDOW",
    "MaskedMAE",
    "MaskedSSIM",
    "SupportedWindows",
    "masked_abs_error_sums",
    "masked_l1_loss",
    "masked_ssim_stats",
    "pool_10m_to_100m",
]
