"""Lightning regression task — the configured training lifecycle unit.

The task owns the train/validation loop semantics: forward the batch,
compute the configured loss, and log structured metrics. It deliberately
knows nothing about data layout, patch geometry, or target semantics —
those arrive as :class:`~berlin_lst_downscaling.modeling.contracts.Batch`
objects and a loss callable.

The default loss is plain MSE, which is **not** the future Stage-5
masked loss. The synthetic path uses an all-valid mask; the real masked
loss is a WB3a decision and arrives through the same
:class:`~berlin_lst_downscaling.modeling.contracts.MaskedLoss` surface.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import functional as F

from berlin_lst_downscaling.modeling.contracts import (
    Batch,
    MaskedLoss,
    validate_batch,
)
from berlin_lst_downscaling.modeling.unet import UNet


def _default_loss() -> MaskedLoss:
    """Plain MSE on the full map — the synthetic (all-valid) default."""
    return lambda output, target, mask: F.mse_loss(output, target)


class LSTRegressionTask(LightningModule):
    """Deterministic regression task around the fixed U-Net baseline.

    Parameters
    ----------
    model:
        The U-Net (``in_channels == n_active_channels``).
    n_active_channels:
        Number of active input channels (first-C of the fixed V3 order);
        validated against each batch at the task boundary.
    learning_rate:
        AdamW learning rate.
    weight_decay:
        AdamW weight decay.
    loss_factory:
        Returns a :class:`MaskedLoss` callable ``(output, target, mask) -> scalar``.
    """

    def __init__(
        self,
        model: UNet,
        n_active_channels: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss_factory: Callable[[], MaskedLoss] = _default_loss,
    ) -> None:
        super().__init__()
        if model.in_channels != n_active_channels:
            raise ValueError(
                f"model expects {model.in_channels} input channels, task configured "
                f"for {n_active_channels}"
            )
        self.model = model
        self.n_active_channels = n_active_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = loss_factory()

    # ── lifecycle ─────────────────────────────────────────────────────

    def forward(self, batch: Batch) -> Tensor:
        """Predict the 100 m regression map for a batch."""
        validate_batch(batch, n_active_channels=self.n_active_channels)
        return self.model(batch.features)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        loss = self._loss_for(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        loss = self._loss_for(batch)
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def _loss_for(self, batch: Batch) -> Tensor:
        output = self(batch)
        return self.loss(output, batch.target, batch.mask)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer  # type: ignore[return-value]


__all__ = ["LSTRegressionTask"]