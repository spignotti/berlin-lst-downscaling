"""Real contract-conforming Lightning path (WB3, issue #18).

The model predicts at 10 m from the 28 frozen feature channels plus the
separate ``lst_prior`` channel. Training compares the exact 10x10 pooled
prediction against the native 100 m Landsat target under
``training_eligible@100m`` and selects checkpoints on the cell-weighted
masked MAE. SSIM at 100 m is logged as a secondary diagnostic only.

This module is additive: ``modeling/task.py`` and ``modeling/synthetic.py``
keep their synthetic lifecycle and its ``validation/loss`` smoke untouched.

``RealPatchDataModule`` lives here rather than in ``modeling/patches.py`` so
the reader stays free of Lightning and the naive baseline can reuse it
without importing the training framework.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    RealBatch,
    validate_real_batch,
)
from berlin_lst_downscaling.modeling.metrics import (
    MaskedMAE,
    MaskedSSIM,
    SupportedWindows,
    masked_l1_loss,
    masked_ssim_stats,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSourceConfig,
    collate_real_batch,
    load_patch_refs,
)
from berlin_lst_downscaling.modeling.unet import UNet

_REAL_SPLITS = ("train", "validation", "test")


class _BatchDataset(Dataset[RealBatch]):
    """Pre-collated batches for one split (deterministic order)."""

    def __init__(self, batches: list[RealBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> RealBatch:
        return self.batches[idx]


class RealPatchDataModule(LightningDataModule):
    """Bounded real patch batches for the contract-conforming path.

    The subset is read eagerly at ``setup`` so a smoke run is a single
    bounded pass over the published sources instead of repeated GCS reads.
    ``max_batches_per_split`` caps every split; ``scene_ids`` restricts the
    scene universe. Both default to unbounded, which is a full-run choice and
    is expected to be invoked explicitly, not by CI.
    """

    def __init__(
        self,
        source: RealSourceConfig,
        *,
        batch_size: int = 4,
        max_batches_per_split: int | None = None,
        scene_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.batch_size = batch_size
        self.max_batches_per_split = max_batches_per_split
        self.scene_ids = tuple(scene_ids) if scene_ids else None
        self.reader: RealPatchReader | None = None
        self._datasets: dict[str, _BatchDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        # Idempotent: the real read is expensive and the lifecycle may set the
        # module up explicitly (to record the read scope) before ``fit``.
        if self._datasets:
            return
        self.reader = RealPatchReader(self.source)
        refs = load_patch_refs(self.source, splits=_REAL_SPLITS, scene_ids=self.scene_ids)
        by_split: dict[str, list[PatchRef]] = {split: [] for split in _REAL_SPLITS}
        for ref in refs:
            by_split[ref.split].append(ref)
        for split, split_refs in by_split.items():
            self._datasets[split] = _BatchDataset(
                self._read_batches(self.reader, split_refs)
            )

    def stats(self) -> dict[str, object]:
        """Describe the real read: per-split batches, patch IDs, and exclusions.

        Retained as run evidence so a comparison can be audited: which patches
        entered each split, and which were dropped with which reason.
        """
        if self.reader is None:
            raise RuntimeError("setup() must run before stats()")
        patch_ids = {
            split: [meta.patch_id for batch in dataset.batches for meta in batch.metadata]
            for split, dataset in self._datasets.items()
        }
        return {
            "batches_per_split": {s: len(d) for s, d in self._datasets.items()},
            "patches_per_split": {s: len(ids) for s, ids in patch_ids.items()},
            "patch_ids": patch_ids,
            "exclusions": dict(self.reader.exclusions),
        }

    def _read_batches(self, reader: RealPatchReader, refs: list[PatchRef]) -> list[RealBatch]:
        """Read, filter exclusions, and collate in deterministic order."""
        cap = self.max_batches_per_split
        batches: list[RealBatch] = []
        chunk: list[RealSample] = []
        for sample in reader.iter_samples(refs):
            chunk.append(sample)
            if len(chunk) == self.batch_size:
                batches.append(collate_real_batch(chunk))
                chunk = []
                if cap is not None and len(batches) >= cap:
                    return batches
        if chunk:
            batches.append(collate_real_batch(chunk))
        return batches

    def _loader(self, split: str) -> DataLoader[RealBatch]:
        dataset = self._datasets.get(split)
        if dataset is None or len(dataset) == 0:
            raise RuntimeError(f"real path has no {split} batch to train on")
        return DataLoader(dataset, batch_size=None, shuffle=False)

    def train_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("validation")

    def test_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("test")


class RealLSTTask(LightningModule):
    """10 m LST downscaling task over the contract-conforming batch.

    Parameters
    ----------
    n_active_channels:
        Number of active feature channels (first-C of the V3 order). The U-Net
        receives one extra input channel for ``lst_prior``.
    base_width / depth:
        U-Net capacity. ``depth`` must satisfy ``160 % 2 ** depth == 0``; the
        contract's patch size requires ``depth <= 4``.
    learning_rate / weight_decay:
        AdamW optimizer settings.
    """

    def __init__(
        self,
        n_active_channels: int = N_FEATURE_CHANNELS,
        base_width: int = 32,
        depth: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = UNet(
            in_channels=n_active_channels + 1, base_width=base_width, depth=depth
        )
        self.n_active_channels = n_active_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_mae = MaskedMAE()
        self.val_mae = MaskedMAE()
        self.val_ssim = MaskedSSIM()
        self.val_ssim_windows = SupportedWindows()
        # Reconstructible model/loss configuration for load_from_checkpoint.
        self.save_hyperparameters()

    # ── lifecycle ─────────────────────────────────────────────────────

    def forward(self, batch: RealBatch) -> Tensor:
        """Predict the 10 m LST map from features concatenated with the prior."""
        validate_real_batch(batch, n_active_channels=self.n_active_channels)
        inputs = torch.cat([batch.features, batch.lst_prior], dim=1)
        return self.model(inputs)

    def training_step(self, batch: RealBatch, batch_idx: int) -> Tensor:
        prediction_100m = pool_10m_to_100m(self(batch))
        loss = masked_l1_loss(prediction_100m, batch.target_100m, batch.mask_100m)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_mae.update(prediction_100m, batch.target_100m, batch.mask_100m)
        self.log("train/mae_100m", self.train_mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: RealBatch, batch_idx: int) -> None:
        prediction_100m = pool_10m_to_100m(self(batch))
        self.val_mae.update(prediction_100m, batch.target_100m, batch.mask_100m)
        ssim_sum, ssim_count = masked_ssim_stats(
            prediction_100m, batch.target_100m, batch.mask_100m
        )
        self.val_ssim.update(ssim_sum, ssim_count)
        self.val_ssim_windows.update(ssim_count)
        # Masked MAE is the selection metric; SSIM is logged only.
        self.log(
            "validation/mae_100m", self.val_mae, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("validation/ssim_100m", self.val_ssim, on_step=False, on_epoch=True)
        self.log(
            "validation/ssim_windows", self.val_ssim_windows, on_step=False, on_epoch=True
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


__all__ = ["RealLSTTask", "RealPatchDataModule"]
