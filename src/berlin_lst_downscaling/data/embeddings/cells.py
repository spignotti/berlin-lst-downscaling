"""Shared eligibility: intersection of the main arm and TESSERA validity.

Both arms (main and TESSERA) are trained and evaluated on the SAME cell
set: the intersection of the main arm's eligible cells and TESSERA-valid
cells. TESSERA gaps (pixels below the quality threshold, non-finite
scales) are not filled.

Grid convention
---------------
- The main-arm eligibility mask is at 100 m (``training_eligible@100m``):
  True where the Landsat target is valid and all 100 10 m feature
  subpixels are ``feature_valid``.
- TESSERA validity is natively at 10 m on the canonical grid (True =
  valid embedding). It is aggregated to 100 m with the same strict
  100/100 rule: a 100 m cell is TESSERA-valid only when all 100 of its
  10 m subpixels are valid. Edge-truncated cells (fewer than 100 present
  subpixels) can never be TESSERA-valid.

The intersection is element-wise AND at 100 m. Both arms point at the
identical patch/cell set by construction.
"""

from __future__ import annotations

import numpy as np


def aggregate_tessera_valid_to_100m(
    tessera_valid_10m: np.ndarray,
    *,
    present_10m: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate a canonical 10 m TESSERA-valid mask to strict 100 m support.

    Parameters
    ----------
    tessera_valid_10m :
        bool ``(H10, W10)`` on the canonical 10 m grid.
    present_10m :
        Optional bool ``(H10, W10)`` present-pixel mask; ``None`` means
        every 10 m pixel is present.

    Returns
    -------
    numpy.ndarray
        bool ``(H100, W100)``: True only where all 100 subpixels are
        present and TESSERA-valid.
    """
    valid = np.asarray(tessera_valid_10m, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"tessera_valid_10m must be 2-D, got {tuple(valid.shape)}")
    h10, w10 = valid.shape
    h100 = (h10 + 9) // 10
    w100 = (w10 + 9) // 10
    pad_h = h100 * 10 - h10
    pad_w = w100 * 10 - w10
    if present_10m is None:
        present = np.ones_like(valid, dtype=bool)
    else:
        present = np.asarray(present_10m, dtype=bool)
        if present.shape != valid.shape:
            raise ValueError("present_10m shape must match tessera_valid_10m")
    valid_padded = np.pad(valid, ((0, pad_h), (0, pad_w)), constant_values=False)
    present_padded = np.pad(present, ((0, pad_h), (0, pad_w)), constant_values=False)
    valid_blocks = valid_padded.reshape(h100, 10, w100, 10).sum(axis=(1, 3))
    present_blocks = present_padded.reshape(h100, 10, w100, 10).sum(axis=(1, 3))
    return (present_blocks == 100) & (valid_blocks == 100)


def intersect_eligibility(
    main_eligible_100m: np.ndarray,
    tessera_valid_100m: np.ndarray,
) -> np.ndarray:
    """Intersect main-arm eligibility with TESSERA validity at 100 m.

    Both inputs are bool ``(H100, W100)``; the output is their
    element-wise AND. Neither input is mutated.
    """
    main = np.asarray(main_eligible_100m, dtype=bool)
    tess = np.asarray(tessera_valid_100m, dtype=bool)
    if main.shape != tess.shape:
        raise ValueError(f"shape mismatch: main {tuple(main.shape)} != tessera {tuple(tess.shape)}")
    return main & tess


def common_cell_set(
    main_eligible_100m: np.ndarray,
    tessera_valid_10m: np.ndarray,
    *,
    present_10m: np.ndarray | None = None,
) -> np.ndarray:
    """Return the common 100 m cell set for both arms.

    Aggregates the 10 m TESSERA validity with the strict 100/100 rule,
    then intersects with the main-arm eligibility. Both arms train and
    evaluate on exactly this set.
    """
    tess_100 = aggregate_tessera_valid_to_100m(tessera_valid_10m, present_10m=present_10m)
    main = np.asarray(main_eligible_100m, dtype=bool)
    if main.shape != tess_100.shape:
        raise ValueError(
            f"shape mismatch: main {tuple(main.shape)} != aggregated "
            f"tessera {tuple(tess_100.shape)}"
        )
    return main & tess_100


def main_arm_cell_set(
    main_eligible_100m: np.ndarray,
    tessera_valid_10m: np.ndarray,
    *,
    present_10m: np.ndarray | None = None,
) -> np.ndarray:
    """Cell set the main arm trains/evaluates on when paired with TESSERA.

    Identical to :func:`common_cell_set` by construction: when the TESSERA
    arm is active, the main arm is pointed at the same intersection instead
    of its standalone eligibility, so both arms share one comparison
    universe.
    """
    return common_cell_set(main_eligible_100m, tessera_valid_10m, present_10m=present_10m)


def tessera_arm_cell_set(
    main_eligible_100m: np.ndarray,
    tessera_valid_10m: np.ndarray,
    *,
    present_10m: np.ndarray | None = None,
) -> np.ndarray:
    """Cell set the TESSERA arm trains/evaluates on (same intersection)."""
    return common_cell_set(main_eligible_100m, tessera_valid_10m, present_10m=present_10m)


def select_common_cell_ids(
    main_cell_ids: list[str],
    tessera_valid_cell_ids: list[str],
) -> list[str]:
    """Intersect main-arm and TESSERA-valid cell-ID sets (sorted, deduped).

    Both arms point their patch index at the returned IDs, so the admitted
    patch/cell set is identical by construction.
    """
    main_set = set(main_cell_ids)
    tess_set = set(tessera_valid_cell_ids)
    return sorted(main_set & tess_set)


__all__ = [
    "aggregate_tessera_valid_to_100m",
    "common_cell_set",
    "intersect_eligibility",
    "main_arm_cell_set",
    "select_common_cell_ids",
    "tessera_arm_cell_set",
]
