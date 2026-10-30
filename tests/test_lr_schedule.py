"""Optional cosine LR schedule with linear warmup: math, config, guards, Lightning wiring."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from berlin_lst_downscaling.modeling.channels import V6_ABLATION_LOCKED_CONFIG_NAMES
from berlin_lst_downscaling.modeling.guards import assert_ablation_lock, guard_modeling_config
from berlin_lst_downscaling.modeling.lr_schedule import (
    lr_at_epoch,
    lr_schedule_from_config,
    optimizer_config,
)
from berlin_lst_downscaling.modeling.real_task import RealLSTTask
from berlin_lst_downscaling.modeling.task import LSTRegressionTask

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_BASE = 1.0e-3


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):

        def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
            return compose(config_name=name, overrides=overrides or [])

        yield _compose
    GlobalHydra.instance().clear()


def _cfg(**block: Any) -> DictConfig:
    return OmegaConf.create(
        {"trainer": {"max_epochs": 40, "learning_rate": _BASE, "lr_schedule": block}}
    )


# ── math ──────────────────────────────────────────────────────────────


def test_warmup_values() -> None:
    assert lr_at_epoch(0, _BASE, 40, 2, 0.0) == pytest.approx(_BASE / 2)
    assert lr_at_epoch(1, _BASE, 40, 2, 0.0) == pytest.approx(_BASE)


def test_cosine_anchor_points() -> None:
    eta = 1.0e-5
    assert lr_at_epoch(2, _BASE, 40, 2, eta) == _BASE
    mid = lr_at_epoch(2 + 19, _BASE, 40, 2, eta)
    assert mid == pytest.approx((_BASE + eta) / 2)
    last = lr_at_epoch(39, _BASE, 40, 2, 0.0)
    assert last == pytest.approx(1.7e-6, rel=0.02)
    assert last > 0.0
    assert lr_at_epoch(39, _BASE, 40, 2, eta) > eta


def test_cosine_is_monotone_after_warmup() -> None:
    values = [lr_at_epoch(e, _BASE, 40, 2, 0.0) for e in range(2, 40)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_no_warmup_and_single_cosine_epoch() -> None:
    assert lr_at_epoch(0, _BASE, 10, 0, 0.0) == _BASE
    assert lr_at_epoch(2, _BASE, 3, 2, 0.0) == _BASE


def test_lr_at_epoch_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="epoch"):
        lr_at_epoch(-1, _BASE, 10, 2, 0.0)
    with pytest.raises(ValueError, match="max_epochs"):
        lr_at_epoch(0, _BASE, 2, 2, 0.0)


# ── config validation ─────────────────────────────────────────────────


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        lr_schedule_from_config(_cfg(name="cosine", warmup_epoch=5))


def test_missing_block_is_constant() -> None:
    cfg = OmegaConf.create({"trainer": {"max_epochs": 5, "learning_rate": _BASE}})
    spec = lr_schedule_from_config(cfg)
    assert (spec.name, spec.warmup_epochs, spec.eta_min, spec.t_max) == ("constant", 2, 0.0, 5)
    assert spec.payload() == {
        "name": "constant",
        "warmup_epochs": 2,
        "eta_min": 0.0,
        "interval": "epoch",
        "t_max": 5,
    }


def test_valid_cosine_block() -> None:
    spec = lr_schedule_from_config(_cfg(name="cosine", warmup_epochs=2, eta_min=1e-6))
    assert spec.name == "cosine"
    assert spec.t_max == 40
    assert spec.eta_min == 1e-6


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ({"name": "linear", "warmup_epochs": 2, "eta_min": 0.0}, "name"),
        ({"name": "cosine", "warmup_epochs": 2, "eta_min": -1e-6}, "eta_min"),
        ({"name": "cosine", "warmup_epochs": 2, "eta_min": _BASE}, "eta_min"),
        ({"name": "cosine", "warmup_epochs": 2, "eta_min": float("nan")}, "eta_min"),
        ({"name": "cosine", "warmup_epochs": 1.5, "eta_min": 0.0}, "warmup_epochs"),
        ({"name": "cosine", "warmup_epochs": True, "eta_min": 0.0}, "warmup_epochs"),
        ({"name": "cosine", "warmup_epochs": -1, "eta_min": 0.0}, "warmup_epochs"),
        ({"name": "cosine", "warmup_epochs": 40, "eta_min": 0.0}, "warmup_epochs"),
    ],
)
def test_invalid_blocks_raise(block: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=key):
        lr_schedule_from_config(_cfg(**block))


def test_constant_does_not_require_epochs_above_warmup() -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {
                "max_epochs": 1,
                "learning_rate": _BASE,
                "lr_schedule": {"name": "constant", "warmup_epochs": 2, "eta_min": 0.0},
            }
        }
    )
    assert lr_schedule_from_config(cfg).name == "constant"


# ── tasks: default off ────────────────────────────────────────────────

_COSINE = {"name": "cosine", "warmup_epochs": 2, "eta_min": 0.0, "interval": "epoch", "t_max": 5}
_CONSTANT = {**_COSINE, "name": "constant"}


def _real_task(**kwargs: Any) -> RealLSTTask:
    return RealLSTTask(
        n_active_channels=2,
        base_width=4,
        depth=1,
        ssim_data_range_k=10.0,
        **kwargs,
    )


@pytest.mark.parametrize("payload", [None, _CONSTANT])
def test_constant_returns_bare_adamw(payload: dict[str, object] | None) -> None:
    for task in (
        _real_task(lr_schedule=payload),
        LSTRegressionTask(n_active_channels=2, base_width=4, depth=1, lr_schedule=payload),
    ):
        opt = task.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)


def test_cosine_returns_epoch_scheduler() -> None:
    for task in (
        _real_task(lr_schedule=_COSINE),
        LSTRegressionTask(n_active_channels=2, base_width=4, depth=1, lr_schedule=_COSINE),
    ):
        out: Any = task.configure_optimizers()
        assert isinstance(out, dict)
        assert isinstance(out["optimizer"], torch.optim.AdamW)
        assert out["lr_scheduler"]["interval"] == "epoch"
        assert out["lr_scheduler"]["frequency"] == 1


def test_hparams_record_lr_schedule() -> None:
    assert _real_task(lr_schedule=_COSINE).hparams["lr_schedule"] == _COSINE
    task = LSTRegressionTask(n_active_channels=2, base_width=4, depth=1, lr_schedule=_COSINE)
    assert task.hparams["lr_schedule"] == _COSINE
    assert _real_task().hparams["lr_schedule"] is None


def test_optimizer_config_identity_and_unknown_name() -> None:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=_BASE)
    assert optimizer_config(opt, None) is opt
    assert optimizer_config(opt, _CONSTANT) is opt
    with pytest.raises(ValueError, match="unknown"):
        optimizer_config(opt, {"name": "step"})


# ── Lightning stepping and resume ─────────────────────────────────────


class _Tiny(LightningModule):
    def __init__(self, lr_schedule: dict[str, object] | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.layer = torch.nn.Linear(2, 1)
        self.lr_schedule = lr_schedule

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(self.parameters(), lr=_BASE)
        return optimizer_config(opt, self.lr_schedule)


class _Data(LightningDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(TensorDataset(torch.randn(4, 2), torch.randn(4, 1)), batch_size=2)


class _Record(Callback):
    def __init__(self) -> None:
        self.lrs: dict[int, float] = {}

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.lrs[trainer.current_epoch] = trainer.optimizers[0].param_groups[0]["lr"]


def _trainer(max_epochs: int, record: _Record) -> Trainer:
    return Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[record],
    )


def test_lightning_steps_per_epoch() -> None:
    record = _Record()
    _trainer(5, record).fit(_Tiny(_COSINE), datamodule=_Data())
    for epoch in range(5):
        assert record.lrs[epoch] == pytest.approx(lr_at_epoch(epoch, _BASE, 5, 2, 0.0))


def test_constant_lr_never_moves() -> None:
    record = _Record()
    _trainer(3, record).fit(_Tiny(_CONSTANT), datamodule=_Data())
    assert set(record.lrs.values()) == {_BASE}


def test_resume_keeps_scheduler_state(tmp_path: Path) -> None:
    first = _Record()
    trainer = _trainer(3, first)
    trainer.fit(_Tiny(_COSINE), datamodule=_Data())
    ckpt = tmp_path / "last.ckpt"
    trainer.save_checkpoint(ckpt)
    assert torch.load(ckpt, weights_only=False)["hyper_parameters"]["lr_schedule"] == _COSINE

    second = _Record()
    _trainer(5, second).fit(_Tiny(_COSINE), datamodule=_Data(), ckpt_path=str(ckpt))
    assert sorted(second.lrs) == [3, 4]
    for epoch in (3, 4):
        assert second.lrs[epoch] == pytest.approx(lr_at_epoch(epoch, _BASE, 5, 2, 0.0))


def test_task_logs_lr_each_epoch() -> None:
    from berlin_lst_downscaling.modeling.synthetic import SyntheticDataModule

    task = LSTRegressionTask(n_active_channels=2, base_width=4, depth=1, lr_schedule=_COSINE)
    data = SyntheticDataModule(
        n_active_channels=2, batch_size=2, patch_size=16, n_train=2, n_val=2, n_test=2, seed=0
    )
    trainer = Trainer(
        max_epochs=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(task, datamodule=data)
    assert "lr" in trainer.callback_metrics


# ── guards ────────────────────────────────────────────────────────────


def _v6(compose_modeling: Callable[..., DictConfig], *overrides: str) -> DictConfig:
    cfg = compose_modeling("v6_l4_locked", list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def test_v6_lock_accepts_constant_and_cosine(compose_modeling: Callable[..., DictConfig]) -> None:
    assert_ablation_lock(_v6(compose_modeling))
    assert_ablation_lock(_v6(compose_modeling, "trainer.lr_schedule.name=cosine"))
    guard_modeling_config(_v6(compose_modeling, "trainer.lr_schedule.name=cosine"), "v6_l4_locked")


@pytest.mark.parametrize(
    "override",
    ["trainer.lr_schedule.warmup_epochs=3", "trainer.lr_schedule.eta_min=1e-6"],
)
def test_v6_lock_pins_warmup_and_eta_min(
    compose_modeling: Callable[..., DictConfig], override: str
) -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        assert_ablation_lock(_v6(compose_modeling, override))


def test_v6_lock_rejects_missing_block(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = _v6(compose_modeling)
    del cfg.trainer["lr_schedule"]
    with pytest.raises(ValueError, match="lr_schedule block is missing"):
        assert_ablation_lock(cfg)


def test_v6_lock_rejects_bad_name(compose_modeling: Callable[..., DictConfig]) -> None:
    with pytest.raises(ValueError, match="lr_schedule.name"):
        assert_ablation_lock(_v6(compose_modeling, "trainer.lr_schedule.name=step"))


@pytest.mark.parametrize("name", ["stage1_locked", "stage2_locked", "stage1_probe"])
def test_v1_locks_reject_cosine(compose_modeling: Callable[..., DictConfig], name: str) -> None:
    cfg = compose_modeling(name, ["trainer.lr_schedule.name=cosine"])
    with pytest.raises(ValueError, match="cosine"):
        guard_modeling_config(cfg, name)


def test_v1_ablation_lock_reports_cosine(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = compose_modeling("stage2_locked", ["trainer.lr_schedule.name=cosine"])
    with pytest.raises(ValueError, match="cosine"):
        assert_ablation_lock(cfg)


@pytest.mark.parametrize("name", sorted(V6_ABLATION_LOCKED_CONFIG_NAMES))
def test_arm_files_do_not_set_lr_schedule(name: str) -> None:
    raw = OmegaConf.load(_CONFIG_DIR / f"{name}.yaml")
    assert OmegaConf.select(raw, "trainer.lr_schedule") is None


def test_math_sanity() -> None:
    assert math.isclose(lr_at_epoch(2, 1.0, 12, 2, 0.0), 1.0)
