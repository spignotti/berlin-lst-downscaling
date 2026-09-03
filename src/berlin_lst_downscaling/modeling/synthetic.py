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

from berlin_lst_downscaling.modeling.contracts import Batch, SampleMeta

# Synthetic identifiers follow the published cell-ID convention
# (data/training/contracts.py) so metadata stays contract-shaped.
_CELL_TEMPLATE = "E{0}N{1}"
_SCENE_TEMPLATE = "SYNTH_SCENE_{0}"


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


__all__ = ["SyntheticDataModule"]
