"""WB3 modeling contract — batch interface, reader/loss extension points.

The model boundary is deliberately small so that later real-data
integration can target a stable surface without committing to the
still-open WB3a decisions (patch geometry, pseudo-pair semantics,
target resampling, masked-loss behaviour).

Batch interface
---------------
A :class:`Batch` is the single currency between the data layer and the
Lightning task: ``features`` (the active feature channels, in the fixed
V3 28-channel order, first ``n_active_channels`` used), ``target`` (the
100 m regression target), ``mask`` (pixel-wise validity, shape-matched to
the target), and per-sample ``metadata``. The mask semantics are an
extension point: the synthetic path publishes an all-valid mask, the
future real path owns the true masked-loss semantics.

Extension points (protocols, not implementations)
--------------------------------------------------
- :class:`DatasetReader` — the future patch-index reader surface
  (``patch_index.parquet`` schema pending WB3a). It must yield
  :class:`Batch` objects; nothing here assumes a patch geometry.
- :class:`MaskedLoss` — the future Stage-5 loss surface. It receives the
  model output, target, and mask; the exact reduction and masked
  semantics are WB3a decisions, not encoded here.

The 28-channel default comes from the immutable Feature Release V3
interface (``data/features/contracts.py``); reducing the active channel
count for ablations is supported by the task wiring, never by reordering
or duplicating the published stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES

# ── fixed interface ───────────────────────────────────────────────────

# Feature Release V3 publishes exactly this many channels in a fixed order.
N_FEATURE_CHANNELS: int = len(FEATURE_CHANNEL_NAMES)

# ── batch surface ─────────────────────────────────────────────────────


@dataclass
class SampleMeta:
    """Per-sample provenance carried through training.

    Only fields grounded in the published WB2c-4 release are fixed; the
    schema may grow when the real patch index arrives.
    """

    cell_id: str  # stable canonical-grid cell ID (data/training/contracts.py)
    scene_id: str
    split: str  # train | validation | test | inference
    year: int


@dataclass
class Batch:
    """One training batch — the data/task boundary.

    Shapes
    ------
    features: ``(B, C, H, W)`` float — C active channels, first-C of the
        fixed V3 28-channel order.
    target:   ``(B, 1, H, W)`` float — regression target on the canonical
        100 m grid.
    mask:     ``(B, 1, H, W)`` bool/float — validity per target pixel.
    metadata: per-sample :class:`SampleMeta` records.
    """

    features: Tensor
    target: Tensor
    mask: Tensor
    metadata: list[SampleMeta]


# ── extension points ──────────────────────────────────────────────────


class DatasetReader(Protocol):
    """Surface a future patch-index dataset reader must satisfy.

    The real reader (patch geometry, admission threshold, index schema)
    is a WB3a decision; this protocol only fixes the return type so the
    task never depends on a concrete reader.
    """

    def __iter__(self) -> DatasetReader:
        """Iterate over batches (yields :class:`Batch`)."""
        ...

    def __len__(self) -> int:
        """Return the number of batches."""
        ...


class MaskedLoss(Protocol):
    """Surface the future masked-loss implementation must satisfy.

    ``output`` and ``target`` are shape ``(B, 1, H, W)``; ``mask`` is
    the validity mask from the batch. The exact loss semantics (reduction,
    masking, nodata policy) are WB3a decisions — this protocol only
    fixes the call signature.
    """

    def __call__(self, output: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Return a scalar loss tensor."""
        ...


# A plain loss callable (e.g. ``torch.nn.functional.mse_loss``) is a valid
# MaskedLoss; the synthetic path uses one directly.
type LossFactory = Callable[[], MaskedLoss]

# ── boundary validation ───────────────────────────────────────────────


def validate_batch(
    batch: Batch,
    *,
    n_active_channels: int,
    rank: int = 4,
) -> None:
    """Validate the batch contract at the task boundary.

    Checks tensor rank, batch-dimension agreement across features, target
    and mask, and the active-channel count. Raises ``ValueError`` on any
    violation so failures surface at the boundary, not deep in the model.
    """
    features = batch.features
    target = batch.target
    mask = batch.mask
    if features.ndim != rank:
        raise ValueError(f"features must have rank {rank}, got {features.ndim}")
    if target.ndim != rank:
        raise ValueError(f"target must have rank {rank}, got {target.ndim}")
    if mask.ndim != rank:
        raise ValueError(f"mask must have rank {rank}, got {mask.ndim}")
    if not (features.shape[0] == target.shape[0] == mask.shape[0]):
        raise ValueError("features/target/mask batch dimensions differ")
    if features.shape[1] != n_active_channels:
        raise ValueError(
            f"features carry {features.shape[1]} channels, expected {n_active_channels}"
        )
    if target.shape[1] != 1:
        raise ValueError(f"target must have exactly 1 channel, got {target.shape[1]}")
    if target.shape[1:] != mask.shape[1:]:
        raise ValueError("target and mask spatial shapes differ")
    if len(batch.metadata) != features.shape[0]:
        raise ValueError("metadata length does not match batch dimension")


__all__ = [
    "Batch",
    "DatasetReader",
    "LossFactory",
    "MaskedLoss",
    "N_FEATURE_CHANNELS",
    "SampleMeta",
    "validate_batch",
]
