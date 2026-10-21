"""Issue #80: thermal epoch audit metrics and recorder retention."""

from __future__ import annotations

from pathlib import Path

import torch

from berlin_lst_downscaling.modeling.metrics import SupportWeightedSum, thermal_objective_batch
from berlin_lst_downscaling.modeling.run import EpochMetricsRecorder


def test_thermal_support_metrics_reset_between_epochs() -> None:
    metric = SupportWeightedSum()
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    batch_a = thermal_objective_batch(
        torch.full((1, 1, 16, 16), 300.0),
        torch.full((1, 1, 16, 16), 299.0),
        mask,
    )
    metric.update(batch_a.l1_sum, batch_a.l1_count)
    first = float(metric.compute())
    metric.reset()
    batch_b = thermal_objective_batch(
        torch.full((1, 1, 16, 16), 310.0),
        torch.full((1, 1, 16, 16), 300.0),
        mask,
    )
    metric.update(batch_b.l1_sum, batch_b.l1_count)
    second = float(metric.compute())
    assert first != second


def test_epoch_metrics_recorder_reads_thermal_keys(tmp_path: Path) -> None:
    recorder = EpochMetricsRecorder(tmp_path / "epoch_metrics.json")

    class _Trainer:
        callback_metrics: dict[str, torch.Tensor]
        current_epoch: int
        precision_plugin: object

        def __init__(self) -> None:
            self.callback_metrics = {
                "train/thermal_l1_unweighted": torch.tensor(1.25),
                "validation/thermal_l1_unweighted": torch.tensor(2.5),
                "validation/ssim_coverage": torch.tensor(0.75),
            }
            self.current_epoch = 0
            self.precision_plugin = type("P", (), {"scaler": None})()

    trainer = _Trainer()
    recorder.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    recorder.on_train_epoch_end(trainer, None)  # type: ignore[arg-type]
    epoch = recorder.epochs[0]
    assert epoch["validation_thermal_l1_unweighted"] == 2.5
    assert epoch["train_thermal_l1_unweighted"] == 1.25
    assert epoch["validation_ssim_coverage"] == 0.75
