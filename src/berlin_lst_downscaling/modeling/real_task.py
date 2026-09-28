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

from collections import Counter
from collections.abc import Sequence
from typing import cast

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler

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


def _batch_count(n_patches: int, batch_size: int) -> int:
    """Number of batches a split of ``n_patches`` collates into (ceil)."""
    if n_patches == 0:
        return 0
    return (n_patches + batch_size - 1) // batch_size


class _BatchDataset(Dataset[RealBatch]):
    """Pre-collated batches for one split (deterministic order)."""

    def __init__(self, batches: list[RealBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> RealBatch:
        return self.batches[idx]


class RealPatchDataset(Dataset[RealSample]):
    """Lazy map-style dataset over admitted patch refs.

    Yields a :class:`RealSample` per indexed ref by reading its COG window on
    demand, so a split's feature tensors are never all resident at once. Refs
    are admitted before construction (see :meth:`RealPatchDataModule.setup`),
    so under stable published sources no unreadable window reaches
    ``__getitem__``; one that does raises rather than silently shrinking the
    comparison universe.

    The reader is created lazily per process, so a worker never inherits an
    open ``rasterio`` dataset from the parent and each worker opens its own on
    first access. This holds under both the ``fork`` and ``spawn`` start
    methods.
    """

    def __init__(self, source: RealSourceConfig, refs: Sequence[PatchRef]) -> None:
        self.source = source
        self.refs = list(refs)
        self._reader: RealPatchReader | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> RealSample:
        sample = self._reader_for_process().read_patch(self.refs[idx])
        if sample is None:
            ref = self.refs[idx]
            raise RuntimeError(
                f"admitted patch {ref.patch_id} is unreadable — published sources "
                f"changed between admission and read"
            )
        return sample

    def _reader_for_process(self) -> RealPatchReader:
        """Return this process's reader, opening one on first access.

        The scene ledger is preloaded once so worker access does not re-read it
        per scene; the reader caches it for the life of the process.
        """
        if self._reader is None:
            reader = RealPatchReader(self.source)
            reader.preload({ref.scene_id for ref in self.refs})
            self._reader = reader
        return self._reader


class RealPatchDataModule(LightningDataModule):
    """Contract-conforming real patch batches.

    ``setup`` admits every selected ref once — the same prior, footprint, and
    eligibility-mask checks the reader applies at read time — and records which
    patches entered each split and which were dropped, with a reason. That
    pre-fit accounting is what keeps the model and the naive baseline
    comparable; it reads no feature window.

    ``mode="eager"`` then pre-collates every admitted patch (a bounded smoke
    convenience). ``mode="stream"`` builds a lazy map-style dataset and reads
    each patch on demand in the loader workers, so a full split is never held
    in memory at once. ``max_patches_per_split`` bounds every split to its
    first N indexed rows — the same ref-level bound the naive baseline uses,
    so the two arms cover an identical requested universe. It defaults to
    unbounded, which is a full-run choice and is expected to be invoked
    explicitly, not by CI.
    """

    def __init__(
        self,
        source: RealSourceConfig,
        *,
        batch_size: int = 4,
        max_patches_per_split: int | None = None,
        scene_ids: Sequence[str] | None = None,
        mode: str = "eager",
        num_workers: int = 0,
        shuffle_train: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in ("eager", "stream"):
            raise ValueError(f"data mode {mode!r} is not one of 'eager', 'stream'")
        self.source = source
        self.batch_size = batch_size
        self.max_patches_per_split = max_patches_per_split
        self.scene_ids = tuple(scene_ids) if scene_ids else None
        self.mode = mode
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.seed = seed
        self.reader: RealPatchReader | None = None
        self._admitted: dict[str, list[PatchRef]] = {}
        self._skipped: dict[str, list[dict[str, str]]] = {split: [] for split in _REAL_SPLITS}
        self._datasets: dict[str, _BatchDataset | RealPatchDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        # Idempotent: admission is expensive and the lifecycle may set the
        # module up explicitly (to record the read scope) before ``fit``.
        if self._datasets:
            return
        self.reader = RealPatchReader(self.source)
        refs = load_patch_refs(self.source, splits=_REAL_SPLITS, scene_ids=self.scene_ids)
        # Bound the *refs* per split, before any exclusion, so the requested
        # universe matches the baseline's selection exactly.
        taken: Counter[str] = Counter()
        requested: dict[str, list[PatchRef]] = {split: [] for split in _REAL_SPLITS}
        for ref in refs:
            if (
                self.max_patches_per_split is not None
                and taken[ref.split] >= self.max_patches_per_split
            ):
                continue
            taken[ref.split] += 1
            requested[ref.split].append(ref)

        # Resolve every scene's Landsat COG/flag once, so the admission pass
        # reads the ARD ledger a single time rather than once per scene.
        self.reader.preload({ref.scene_id for ref in refs})
        for split, split_refs in requested.items():
            admitted = self._admit(self.reader, split, split_refs)
            self._admitted[split] = admitted
            if self.mode == "eager":
                self._datasets[split] = _BatchDataset(
                    self._read_batches(self.reader, admitted)
                )
            else:
                self._datasets[split] = RealPatchDataset(self.source, admitted)

    def _admit(
        self, reader: RealPatchReader, split: str, refs: list[PatchRef]
    ) -> list[PatchRef]:
        """Keep the readable refs in canonical order; record the skipped ones."""
        admitted: list[PatchRef] = []
        for ref in refs:
            reason = reader.admission_reason(ref)
            if reason is None:
                admitted.append(ref)
            else:
                self._skipped[split].append({"patch_id": ref.patch_id, "reason": reason})
        return admitted

    def stats(self) -> dict[str, object]:
        """Describe the real read: per-split patches, IDs, skips, and exclusions.

        Counts and IDs come from the admitted refs in canonical index order,
        never from loader iteration order, so a shuffled stream reports the
        same universe as the eager path. Retained as run evidence so a
        comparison can be audited: which patches entered each split, and which
        were dropped with which reason.
        """
        if self.reader is None:
            raise RuntimeError("setup() must run before stats()")
        return {
            "mode": self.mode,
            "batches_per_split": {
                split: _batch_count(len(refs), self.batch_size)
                for split, refs in self._admitted.items()
            },
            "patches_per_split": {split: len(refs) for split, refs in self._admitted.items()},
            "patch_ids": {
                split: [ref.patch_id for ref in refs]
                for split, refs in self._admitted.items()
            },
            "skipped_refs": self._skipped,
            "exclusions": dict(self.reader.exclusions),
            "train_shuffle_order": self._train_shuffle_order(),
        }

    def _train_shuffle_order(self) -> list[int] | None:
        """The seeded epoch-1 train index order, or ``None`` when unshuffled.

        # decision: record the shuffled order as run evidence, because the plan
        # requires a verifiable same-seed replay and a shuffled stream is
        # otherwise unobservable from canonical-order stats. Alternative:
        # assert it only inside the smoke step (rejected — not retained).
        """
        if self.mode != "stream" or not self.shuffle_train:
            return None
        dataset = self._datasets.get("train")
        if dataset is None or len(dataset) == 0:
            return None
        return list(RandomSampler(dataset, generator=torch.Generator().manual_seed(self.seed)))

    def _read_batches(self, reader: RealPatchReader, refs: list[PatchRef]) -> list[RealBatch]:
        """Read admitted patches and collate them in canonical order."""
        batches: list[RealBatch] = []
        chunk: list[RealSample] = []
        for ref in refs:
            sample = reader.read_patch(ref)
            if sample is None:
                raise RuntimeError(
                    f"admitted patch {ref.patch_id} became unreadable during the eager read"
                )
            chunk.append(sample)
            if len(chunk) == self.batch_size:
                batches.append(collate_real_batch(chunk))
                chunk = []
        if chunk:
            batches.append(collate_real_batch(chunk))
        return batches

    def _loader(self, split: str) -> DataLoader[RealBatch]:
        dataset = self._datasets.get(split)
        if dataset is None or len(dataset) == 0:
            raise RuntimeError(f"real path has no {split} batch to train on")
        if isinstance(dataset, _BatchDataset):
            return DataLoader(dataset, batch_size=None, shuffle=False)
        shuffle = self.shuffle_train and split == "train"
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_real_batch,
            num_workers=self.num_workers,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )
        # The dataset yields RealSample; ``collate_real_batch`` stacks each
        # batch into RealBatch, so the loader's yielded type is RealBatch.
        return cast(DataLoader[RealBatch], loader)

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
