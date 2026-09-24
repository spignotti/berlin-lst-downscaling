"""WB3 modeling contract — batch interface, reader/loss extension points.

The model boundary is deliberately small so that later real-data
integration can target a stable surface without committing to the
still-open WB3a decisions (patch geometry, pseudo-pair semantics,
target resampling, masked-loss behaviour).

Batch interfaces
----------------
Two batch surfaces exist side by side:

- :class:`Batch` — the **synthetic scaffold** currency used by
  ``modeling/synthetic.py`` and ``modeling/task.py``: ``features``, a
  shape-matched ``target``/``mask`` (both at the same grid), and per-sample
  ``metadata``. Its mask semantics are all-valid; it does not implement the
  pseudo-pair contract.
- :class:`RealBatch` — the **contract-conforming** currency of the real
  path (``modeling/patches.py``): 28-channel ``features`` and a separate
  ``lst_prior`` at 10 m, plus ``target_100m``/``mask_100m`` at 100 m. See
  ``docs/pseudo-pair-tensor-contract.md``.

Extension points (protocols, not implementations)
--------------------------------------------------
- :class:`DatasetReader` — the patch-index reader surface. It must yield
  :class:`Batch` objects; nothing here assumes a patch geometry.
- :class:`MaskedLoss` — the masked-loss surface. It receives the model
  output, target, and mask; the exact reduction and masked semantics are
  owned by ``modeling/metrics.py`` and the tensor contract.

The 28-channel default comes from the immutable Feature Release V3
interface (``data/features/contracts.py``); reducing the active channel
count for ablations is supported by the task wiring, never by reordering
or duplicating the published stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K

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

    def __iter__(self) -> Iterator[Batch]:
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

# ── real (contract-conforming) path ───────────────────────────────────
#
# Geometry, pooling, and prior-normalization constants shared by the real
# reader, the metrics, and the naive baseline so the model and the baseline
# cannot drift apart. Fixed by ``docs/pseudo-pair-tensor-contract.md``.

# 160 x 160 px at 10 m = 16 x 16 cells at 100 m (depth-4 U-Net edge).
REAL_PATCH_PX: int = 160
REAL_PATCH_CELLS: int = 16

# Exact nested block factors: 10 x 10 pixels of 10 m per 100 m cell, and
# 10 x 10 cells of 100 m per 1000 m prior block.
POOL_FACTOR: int = 10
PRIOR_BLOCK_CELLS: int = 10
PRIOR_BLOCK_PX_10M: int = POOL_FACTOR * PRIOR_BLOCK_CELLS  # 100

# Fixed affine prior normalization for the model input channel. Derived
# from the Landsat physical range so the two constants cannot drift:
# ``[LST_RANGE_K[0], LST_RANGE_K[1]]`` maps to ``[-1, 1]``.
# decision: a fixed constant (not a fitted statistic), because the prior is
# a physical input and a train-only z-score would make the model's input
# distribution depend on the split. Alternative: train-only z-score over
# native-valid cells (rejected - extra state with no Stage-1 benefit).
PRIOR_AFFINE_OFFSET_K: float = (LST_RANGE_K[0] + LST_RANGE_K[1]) / 2.0
PRIOR_AFFINE_SCALE_K: float = (LST_RANGE_K[1] - LST_RANGE_K[0]) / 2.0

# Reason recorded when a patch requires a 1000 m prior block that contains
# no native-valid Landsat cell (and therefore carries no prior value).
NO_PRIOR_REASON: str = "no_valid_prior_block"


@dataclass
class RealSampleMeta:
    """Per-patch provenance for the contract-conforming real path."""

    patch_id: str  # ``<scene_id>:<cell_id>`` from the patch index
    scene_id: str
    split: str  # train | validation | test
    year: int
    row: int  # global canonical 100 m anchor row
    col: int  # global canonical 100 m anchor col
    filled_feature_pixels: int  # invalid 10 m predictor pixels zero-filled


@dataclass
class RealBatch:
    """One contract-conforming training batch — the data/task boundary.

    Shapes (``H10 = W10 = REAL_PATCH_PX``, ``H100 = W100 = REAL_PATCH_CELLS``)
    -------------------------------------------------------------------------
    features:   ``(B, 28, H10, W10)`` float32 — V3 order, train-only scaled,
        finite (invalid predictor pixels zero-filled at the boundary).
    lst_prior:  ``(B, 1, H10, W10)`` float32 — 1000 m block-expanded prior,
        normalized by the fixed affine above.
    target_100m: ``(B, 1, H100, W100)`` float32 — native Landsat LST in K.
    mask_100m:  ``(B, 1, H100, W100)`` bool — ``training_eligible@100m``.
    metadata:   per-sample :class:`RealSampleMeta` records.

    The prior channel is normalized for the model; the naive baseline
    consumes the physical Kelvin prior through the same reader and reuses
    the same target/mask, so the two arms stay comparable.
    """

    features: Tensor
    lst_prior: Tensor
    target_100m: Tensor
    mask_100m: Tensor
    metadata: list[RealSampleMeta]


def validate_real_batch(batch: RealBatch, *, n_active_channels: int) -> None:
    """Validate the real batch contract at the task boundary.

    Enforces the fixed pseudo-pair geometry (a 160x160 10 m prediction
    footprint and a 16x16 100 m supervision footprint), the separate prior
    channel, batch-dimension agreement, and finiteness of the scaled model
    inputs. Raises ``ValueError`` so a geometry drift fails at the
    boundary instead of deep in the model.
    """
    features = batch.features
    prior = batch.lst_prior
    target = batch.target_100m
    mask = batch.mask_100m

    for name, tensor in (
        ("features", features),
        ("lst_prior", prior),
        ("target_100m", target),
        ("mask_100m", mask),
    ):
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have rank 4, got {tensor.ndim}")

    b = features.shape[0]
    if not (prior.shape[0] == target.shape[0] == mask.shape[0] == b):
        raise ValueError("features/lst_prior/target_100m/mask_100m batch dimensions differ")
    if features.shape[1] != n_active_channels:
        raise ValueError(
            f"features carry {features.shape[1]} channels, expected {n_active_channels}"
        )
    if features.shape[2:] != (REAL_PATCH_PX, REAL_PATCH_PX):
        raise ValueError(
            f"features must be {REAL_PATCH_PX}x{REAL_PATCH_PX} at 10 m, "
            f"got {tuple(features.shape[2:])}"
        )
    if prior.shape[1] != 1 or prior.shape[2:] != (REAL_PATCH_PX, REAL_PATCH_PX):
        raise ValueError(
            f"lst_prior must be 1x{REAL_PATCH_PX}x{REAL_PATCH_PX}, got {tuple(prior.shape[1:])}"
        )
    cells = (REAL_PATCH_CELLS, REAL_PATCH_CELLS)
    if target.shape[1] != 1 or tuple(target.shape[2:]) != cells:
        raise ValueError(f"target_100m must be 1x{cells}, got {tuple(target.shape[1:])}")
    if mask.shape[1] != 1 or tuple(mask.shape[2:]) != cells:
        raise ValueError(f"mask_100m must be 1x{cells}, got {tuple(mask.shape[1:])}")
    if len(batch.metadata) != b:
        raise ValueError("metadata length does not match batch dimension")
    if not bool(mask.any()):
        raise ValueError("mask_100m has no valid cell (count(m_100) must be > 0)")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features contain non-finite values (zero-fill at the boundary first)")
    if not bool(torch.isfinite(prior).all()):
        raise ValueError("lst_prior contains non-finite values")


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
    "NO_PRIOR_REASON",
    "POOL_FACTOR",
    "PRIOR_AFFINE_OFFSET_K",
    "PRIOR_AFFINE_SCALE_K",
    "PRIOR_BLOCK_CELLS",
    "PRIOR_BLOCK_PX_10M",
    "REAL_PATCH_CELLS",
    "REAL_PATCH_PX",
    "Batch",
    "DatasetReader",
    "LossFactory",
    "MaskedLoss",
    "N_FEATURE_CHANNELS",
    "RealBatch",
    "RealSampleMeta",
    "SampleMeta",
    "validate_batch",
    "validate_real_batch",
]
