"""Optional cosine learning-rate schedule with linear warmup.

``constant`` (the default) is exactly the historical behaviour: a bare AdamW and
no scheduler object. ``cosine`` wraps the same optimizer in a per-epoch
``LambdaLR`` that follows :func:`lr_at_epoch`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR

LR_SCHEDULE_NAMES = ("constant", "cosine")


@dataclass(frozen=True)
class LRScheduleSpec:
    """Validated ``trainer.lr_schedule`` block."""

    name: str
    warmup_epochs: int
    eta_min: float
    t_max: int

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "warmup_epochs": self.warmup_epochs,
            "eta_min": self.eta_min,
            "interval": "epoch",
            "t_max": self.t_max,
        }


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def lr_schedule_from_config(cfg: DictConfig) -> LRScheduleSpec:
    """Read and validate ``trainer.lr_schedule``; a missing block means constant."""
    max_epochs = OmegaConf.select(cfg, "trainer.max_epochs")
    if not _is_plain_int(max_epochs) or max_epochs < 1:
        raise ValueError(f"trainer.max_epochs must be an int >= 1, got {max_epochs!r}")
    block = OmegaConf.select(cfg, "trainer.lr_schedule")
    if block is None:
        return LRScheduleSpec("constant", 2, 0.0, int(max_epochs))
    unknown = sorted(set(block.keys()) - {"name", "warmup_epochs", "eta_min"})
    if unknown:
        raise ValueError(f"trainer.lr_schedule has unknown keys {unknown}")
    name = block.get("name", "constant")
    if name not in LR_SCHEDULE_NAMES:
        raise ValueError(
            f"trainer.lr_schedule.name must be one of {LR_SCHEDULE_NAMES}, got {name!r}"
        )
    warmup = block.get("warmup_epochs", 2)
    if not _is_plain_int(warmup) or warmup < 0:
        raise ValueError(f"trainer.lr_schedule.warmup_epochs must be an int >= 0, got {warmup!r}")
    eta_min = block.get("eta_min", 0.0)
    if (
        isinstance(eta_min, bool)
        or not isinstance(eta_min, (int, float))
        or not math.isfinite(eta_min)
        or eta_min < 0
    ):
        raise ValueError(
            f"trainer.lr_schedule.eta_min must be a finite number >= 0, got {eta_min!r}"
        )
    base_lr = OmegaConf.select(cfg, "trainer.learning_rate")
    if (
        isinstance(base_lr, bool)
        or not isinstance(base_lr, (int, float))
        or not math.isfinite(base_lr)
        or eta_min >= base_lr
    ):
        raise ValueError(
            f"trainer.lr_schedule.eta_min={eta_min!r} must be < trainer.learning_rate={base_lr!r}"
        )
    if name == "cosine" and max_epochs <= warmup:
        raise ValueError(
            f"trainer.max_epochs={max_epochs} must exceed trainer.lr_schedule.warmup_epochs="
            f"{warmup} for the cosine schedule"
        )
    return LRScheduleSpec(str(name), int(warmup), float(eta_min), int(max_epochs))


def lr_at_epoch(
    epoch: int, base_lr: float, max_epochs: int, warmup_epochs: int, eta_min: float
) -> float:
    """Learning rate used for every optimizer step of 0-indexed ``epoch``.

    Stepped per epoch. With ``W = warmup_epochs`` and ``T_max = max_epochs``
    (warmup is counted inside ``max_epochs``)::

        e <  W:  base_lr * (e + 1) / W
        e >= W:  eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * (e - W) / (T_max - W)))

    Early stopping does not shorten ``T_max``.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if max_epochs <= warmup_epochs:
        raise ValueError(f"max_epochs={max_epochs} must exceed warmup_epochs={warmup_epochs}")
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_config(
    optimizer: torch.optim.Optimizer, lr_schedule: Mapping[str, object] | None
) -> OptimizerLRScheduler:
    """Return the bare optimizer (constant) or optimizer plus per-epoch scheduler (cosine)."""
    if lr_schedule is None or lr_schedule.get("name") == "constant":
        return optimizer
    if lr_schedule.get("name") != "cosine":
        raise ValueError(f"unknown lr_schedule name {lr_schedule.get('name')!r}")
    base = float(optimizer.param_groups[0]["lr"])
    t_max = int(lr_schedule["t_max"])  # type: ignore[call-overload]
    warmup = int(lr_schedule["warmup_epochs"])  # type: ignore[call-overload]
    eta_min = float(lr_schedule["eta_min"])  # type: ignore[arg-type]
    scheduler = LambdaLR(optimizer, lambda e: lr_at_epoch(e, base, t_max, warmup, eta_min) / base)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr",
        },
    }


def current_lr(module: LightningModule) -> float:
    """Learning rate of the first param group of the attached trainer's optimizer."""
    return float(module.trainer.optimizers[0].param_groups[0]["lr"])


def log_current_lr(module: LightningModule) -> None:
    """Log the epoch's LR once as ``lr`` (call from ``on_train_epoch_start``)."""
    module.log("lr", current_lr(module), on_step=False, on_epoch=True, batch_size=1)


__all__ = [
    "LR_SCHEDULE_NAMES",
    "LRScheduleSpec",
    "current_lr",
    "log_current_lr",
    "lr_at_epoch",
    "lr_schedule_from_config",
    "optimizer_config",
]
