"""Deterministic synthetic data module — the smoke lifecycle path.

Produces learnable, seeded batches through the formal
:class:`~berlin_lst_downscaling.modeling.contracts.Batch` interface so the
full training lifecycle can be exercised before real WB3 data exists.

The data is a deliberate toy: Gaussian feature noise plus a spatial
low-frequency drift, with the target a weighted combination of the
active feature channels plus noise. Training reduces the loss, which is
all the scaffold needs to prove the lifecycle; nothing here claims to
model LST or to pre-empt the patch/admission decisions (all masks are
valid by construction).

The generator is seeded once per module from ``seed``, so a configured
seed reproduces identical tensors, batches, and splits across runs.
"""

from __future__ import annotations

import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from berlin_lst_downscaling.data.training.contracts import cell_id
from berlin_lst_downscaling.modeling.contracts import (
    POOL_FACTOR,
    PRIOR_AFFINE_OFFSET_K,
    PRIOR_AFFINE_SCALE_K,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    Batch,
    RealBatch,
    RealSampleMeta,
    SampleMeta,
)
from berlin_lst_downscaling.modeling.metrics import pool_10m_to_100m

# Synthetic identifiers follow the published cell-ID convention
# (data/training/contracts.py) so metadata stays contract-shaped.
_CELL_TEMPLATE = "E{0}N{1}"
_SCENE_TEMPLATE = "SYNTH_SCENE_{0}"
_CONTRACT_SCENE_TEMPLATE = "SYNTH_CONTRACT_SCENE_{0}"

# The contract lifecycle trains on train/validation/test; the test split is
# read for parity with the real reader but never selected on.
_CONTRACT_SPLITS = ("train", "validation", "test")
_CONTRACT_SPLIT_YEARS = {"train": 2021, "validation": 2024, "test": 2025}


class _SyntheticDataset(Dataset[Batch]):
    """In-memory synthetic batches for one split."""

    def __init__(
        self,
        features: Tensor,
        target: Tensor,
        mask: Tensor,
        metadata: list[SampleMeta],
        batch_size: int,
    ) -> None:
        self.features = features
        self.target = target
        self.mask = mask
        self.metadata = metadata
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.features) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> Batch:
        start = idx * self.batch_size
        stop = min(start + self.batch_size, len(self.features))
        return Batch(
            features=self.features[start:stop],
            target=self.target[start:stop],
            mask=self.mask[start:stop],
            metadata=self.metadata[start:stop],
        )


class SyntheticDataModule(LightningDataModule):
    """Deterministic synthetic batches for smoke and CI.

    Parameters
    ----------
    n_active_channels:
        Number of active feature channels (first-C of the V3 order).
    batch_size:
        Batches per step.
    patch_size:
        Spatial extent (H == W) of each patch; must be divisible by
        ``2 ** depth`` of the configured U-Net.
    n_train / n_val / n_test:
        Sample counts per split.
    seed:
        Generator seed — a fixed seed reproduces identical data.
    """

    def __init__(
        self,
        n_active_channels: int = 28,
        batch_size: int = 8,
        patch_size: int = 32,
        n_train: int = 128,
        n_val: int = 32,
        n_test: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_active_channels = n_active_channels
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed
        self._datasets: dict[str, _SyntheticDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(self.seed)
        self._datasets = {
            "train": self._generate("train", self.n_train, generator),
            "val": self._generate("validation", self.n_val, generator),
            "test": self._generate("test", self.n_test, generator),
        }

    def _generate(self, split: str, n: int, generator: torch.Generator) -> _SyntheticDataset:
        patch = self.patch_size
        channels = self.n_active_channels
        # Gaussian noise input; the target below is a learnable combination.
        features = torch.randn(n, channels, patch, patch, generator=generator)
        # Deterministic spatial low-frequency drift (one per channel).
        # ``torch.linspace`` is deterministic by construction (no generator).
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, patch),
            torch.linspace(-1.0, 1.0, patch),
            indexing="ij",
        )
        drift = (yy**2 + xx**2).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        # Target: channel-weighted mean + drift + independent noise.
        # ``torch.linspace`` is deterministic (no generator).
        weights = torch.linspace(0.1, 1.0, channels).view(1, channels, 1, 1)
        signal = (features * weights).mean(dim=1, keepdim=True) + 0.25 * drift
        target = signal + 0.05 * torch.randn(n, 1, patch, patch, generator=generator)
        # All-valid mask — the synthetic path claims no masking semantics.
        mask = torch.ones_like(target, dtype=torch.bool)

        cells = [_CELL_TEMPLATE.format(369190 + 100 * i, 5838410 - 100 * i) for i in range(n)]
        metadata = [
            SampleMeta(
                cell_id=cells[i],
                scene_id=_SCENE_TEMPLATE.format(i),
                split=split,
                year=2024 if split == "validation" else 2021,
            )
            for i in range(n)
        ]
        return _SyntheticDataset(features, target, mask, metadata, self.batch_size)

    # Lightning hooks — DataLoader default samplers are deterministic
    # under a fixed generator and shuffle=False.

    def train_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(self._datasets["train"], batch_size=None, shuffle=False)

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(self._datasets["val"], batch_size=None, shuffle=False)

    def test_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(self._datasets["test"], batch_size=None, shuffle=False)


class _ContractSyntheticDataset(Dataset[RealBatch]):
    """Pre-collated contract-shaped batches for one split (deterministic order)."""

    def __init__(self, batches: list[RealBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> RealBatch:
        return self.batches[idx]


class ContractSyntheticDataModule(LightningDataModule):
    """Deterministic contract-shaped synthetic batches — the GCS-free gate.

    Same tensor contract as the published real path: ``(B, 28, 160, 160)``
    features, a prior channel normalized by the fixed contract affine, and
    ``(B, 1, 16, 16)`` target/mask. It exercises the masked L1 loss, the
    masked-MAE selection metric, and the checkpoint/reload lifecycle without
    reading GCS.

    Unlike :class:`SyntheticDataModule` (the all-valid MSE fixture), the mask
    here is partially valid by construction: the first row and column are
    ineligible, so the invalid cells must be selected out rather than counted.
    Invalid target cells carry ``NaN`` so a masking regression cannot hide.

    This is still a wiring fixture, not a model of LST. Nothing here claims
    training quality; the geometry and loss are the only contract claims.
    """

    def __init__(
        self,
        n_active_channels: int = 28,
        batch_size: int = 4,
        n_train: int = 8,
        n_val: int = 4,
        n_test: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_active_channels = n_active_channels
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed
        self._datasets: dict[str, _ContractSyntheticDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        counts = {"train": self.n_train, "validation": self.n_val, "test": self.n_test}
        generator = torch.Generator().manual_seed(self.seed)
        self._datasets = {
            split: self._generate(split, counts[split], generator)
            for split in _CONTRACT_SPLITS
        }

    def _generate(
        self, split: str, n: int, generator: torch.Generator
    ) -> _ContractSyntheticDataset:
        patch = REAL_PATCH_PX
        cells = REAL_PATCH_CELLS
        channels = self.n_active_channels

        features = torch.randn(n, channels, patch, patch, generator=generator)

        # Prior: a smooth low-frequency field at 100 m resolution, block-expanded
        # to 10 m and normalized by the same fixed affine the real reader uses,
        # so the model input channel is in the contract's [-1, 1] range.
        cell_noise = torch.randn(n, 1, cells, cells, generator=generator)
        prior_norm = 0.2 * torch.tanh(cell_noise)
        prior = prior_norm.repeat_interleave(POOL_FACTOR, dim=2).repeat_interleave(
            POOL_FACTOR, dim=3
        )

        # Target: a learnable function of the pooled features plus the prior,
        # placed inside the physical LST range so SSIM's data_range is honest.
        pooled = pool_10m_to_100m(features)
        weights = torch.linspace(0.1, 1.0, channels).view(1, channels, 1, 1)
        signal = (pooled * weights).mean(dim=1, keepdim=True)
        target = PRIOR_AFFINE_OFFSET_K + PRIOR_AFFINE_SCALE_K * (
            0.25 * torch.tanh(2.0 * signal) + 0.1 * prior_norm
        )

        # Partial mask: first row and column ineligible (a solid 15x15 valid
        # block, so masked SSIM still has supported windows). The invalid
        # cells carry NaN targets to prove the loss selects them out.
        mask = torch.ones(n, 1, cells, cells, dtype=torch.bool)
        mask[:, :, 0, :] = False
        mask[:, :, :, 0] = False
        target = target.masked_fill(~mask, float("nan"))

        year = _CONTRACT_SPLIT_YEARS[split]
        metadata = []
        for i in range(n):
            row = 5800 + i
            col = 3690 + i
            scene_id = _CONTRACT_SCENE_TEMPLATE.format(i)
            metadata.append(
                RealSampleMeta(
                    patch_id=f"{scene_id}:{cell_id(row, col)}",
                    scene_id=scene_id,
                    split=split,
                    year=year,
                    row=row,
                    col=col,
                    filled_feature_pixels=0,
                )
            )
        # Collate once, in deterministic order, so the Dataset mirrors the real
        # module's pre-collated shape and ``stats`` can read the batch list.
        batches = [
            RealBatch(
                features=features[start : start + self.batch_size],
                lst_prior=prior[start : start + self.batch_size],
                target_100m=target[start : start + self.batch_size],
                mask_100m=mask[start : start + self.batch_size],
                metadata=metadata[start : start + self.batch_size],
            )
            for start in range(0, n, self.batch_size)
        ]
        return _ContractSyntheticDataset(batches)

    def stats(self) -> dict[str, object]:
        """Describe the synthetic read in the same shape as the real module.

        Keeps the shared lifecycle's ``data_scope.json`` uniform across both
        contract-shaped paths; the synthetic path has no exclusions.
        """
        patch_ids = {
            split: [meta.patch_id for batch in dataset.batches for meta in batch.metadata]
            for split, dataset in self._datasets.items()
        }
        return {
            "batches_per_split": {s: len(d) for s, d in self._datasets.items()},
            "patches_per_split": {s: len(ids) for s, ids in patch_ids.items()},
            "patch_ids": patch_ids,
            "exclusions": {},
        }

    # Lightning hooks — default samplers are deterministic under a fixed
    # generator and shuffle=False.

    def train_dataloader(self) -> DataLoader[RealBatch]:
        return DataLoader(self._datasets["train"], batch_size=None, shuffle=False)

    def val_dataloader(self) -> DataLoader[RealBatch]:
        return DataLoader(self._datasets["validation"], batch_size=None, shuffle=False)

    def test_dataloader(self) -> DataLoader[RealBatch]:
        return DataLoader(self._datasets["test"], batch_size=None, shuffle=False)


__all__ = ["ContractSyntheticDataModule", "SyntheticDataModule"]
