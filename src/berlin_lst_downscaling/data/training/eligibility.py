"""Training-eligibility computation — strict 100/100 support at 100 m.

Computes the per-scene ``training_eligible@100m`` mask on the canonical
EPSG:25833 100 m grid from:

- the Landsat target (100 m): the target cell is valid when the ARD flag
  is clear (``flag == 0``) and LST is within the physical range
  ``[150, 400] K`` — the same expression the Stage-1/Stage-2 gates use;
- the ``feature_valid`` mask (10 m): a 100 m cell is eligible only when
  **all 100** of its 10 m subpixels are feature-valid (strict support,
  user-mandated). Edge-truncated cells (fewer than 100 present subpixels)
  can never be eligible.

No resampling, no imputation, no extra input masks (user-mandated). The
computation reuses the Stage-1/Stage-2 grid/window primitives so the
target/support semantics cannot drift from the QA gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.qa.inventory import ResolvedScene

# decision: reuse the Stage-1/Stage-2 grid/window primitives directly —
# eligibility must share the exact target-valid expression and 10x10
# aggregation of the QA gates; duplicated math would drift.
from berlin_lst_downscaling.data.qa.stage1_raw import _tile_windows, _window_offset

# 2560x2560 10 m tiles: exact multiple of the COG block size and the 10x10
# aggregation factor (same choice as the Stage-2 gate).
_TILE = 2560


@dataclass
class EligibilityResult:
    """Per-scene eligibility computation result."""

    scene_id: str
    year: int
    s2_scene_id: str
    eligible: np.ndarray  # bool (H100, W100): 1 = training-eligible
    target_valid: np.ndarray  # bool (H100, W100)
    support: np.ndarray  # int32 (H100, W100): valid feature subpixels
    present: np.ndarray  # int32 (H100, W100): present (non-edge) subpixels
    target_valid_cells: int
    eligible_cells: int
    errors: list[str] = field(default_factory=list)


def compute_eligibility(
    *,
    scene: ResolvedScene,
    analysis_10: GeoBox,
    mask_uri: str,
) -> EligibilityResult:
    """Compute the training-eligibility mask for one assessable scene.

    Parameters
    ----------
    scene :
        Resolved scene with ``landsat_cog``/``landsat_flag`` URIs.
    analysis_10 :
        10 m analysis grid (full canonical grid or a canonical-aligned
        subset for smoke). ``zoom_out(10)`` yields the 100 m grid.
    mask_uri :
        Published ``feature_valid`` mask COG URI for the scene.
    """
    analysis_100: GeoBox = analysis_10.zoom_out(10)
    H100, W100 = analysis_100.shape.y, analysis_100.shape.x
    H10, W10 = analysis_10.shape.y, analysis_10.shape.x
    errors: list[str] = []

    # ── Landsat target (100 m, full read — tiny) ───────────────────────
    st = flag = None
    ls_off = (0, 0)
    try:
        with rasterio.open(scene.landsat_cog) as src:
            st = src.read(1).astype(np.float32)
            ls_off = _window_offset(src, analysis_100, 100.0)
        with rasterio.open(scene.landsat_flag) as src:
            flag = src.read(1).astype(np.uint8)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene.scene_id}: cannot read Landsat target: {exc}")

    target_valid = np.zeros((H100, W100), dtype=bool)
    if st is not None and flag is not None:
        co, ro = ls_off
        y0, y1 = max(ro, 0), min(ro + H100, st.shape[0])
        x0, x1 = max(co, 0), min(co + W100, st.shape[1])
        target_valid[y0 - ro : y1 - ro, x0 - co : x1 - co] = (
            (flag[y0:y1, x0:x1] == 0)
            & (st[y0:y1, x0:x1] >= LST_RANGE_K[0])
            & (st[y0:y1, x0:x1] <= LST_RANGE_K[1])
        )

    # ── present-count grid (edge-truncated 100 m cells) ────────────────
    iy = np.arange(H100, dtype=np.int64)[:, None]
    ix = np.arange(W100, dtype=np.int64)[None, :]
    present = np.minimum(10, H10 - 10 * iy) * np.minimum(10, W10 - 10 * ix)

    # ── support from the published feature_valid mask (10 m) ───────────
    support = np.zeros((H100, W100), dtype=np.int32)
    try:
        with rasterio.open(mask_uri) as msk:
            mask_off = _window_offset(msk, analysis_10, 10.0)
            for (r0, c0, r1, c1), _ in _tile_windows(analysis_10, _TILE):
                bh, bw = r1 - r0, c1 - c0
                win = Window.from_slices(
                    (r0 + mask_off[1], r0 + mask_off[1] + bh),
                    (c0 + mask_off[0], c0 + mask_off[0] + bw),
                )
                valid = msk.read(1, window=win) == 1

                # Exact nested 10x10 aggregation (same as Stage-2): pad the
                # edge tile to a multiple of 10 so reshape stays exact.
                ph, pw = -(-bh // 10) * 10, -(-bw // 10) * 10
                if ph != bh or pw != bw:
                    m = np.zeros((ph, pw), dtype=np.uint8)
                    m[:bh, :bw] = valid
                else:
                    m = valid.astype(np.uint8)
                block_sum = m.reshape(ph // 10, 10, pw // 10, 10).sum(axis=(1, 3)).astype(np.int32)
                br0, bc0 = r0 // 10, c0 // 10
                support[br0 : br0 + ph // 10, bc0 : bc0 + pw // 10] += block_sum
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene.scene_id}: feature_valid mask scan failed: {exc}")

    # Strict eligibility: valid target AND all 100 subpixels valid.
    eligible = target_valid & (present == 100) & (support == 100)

    return EligibilityResult(
        scene_id=scene.scene_id,
        year=scene.year,
        s2_scene_id=scene.s2_scene_id,
        eligible=eligible,
        target_valid=target_valid,
        support=support,
        present=present,
        target_valid_cells=int(np.sum(target_valid)),
        eligible_cells=int(np.sum(eligible)),
        errors=errors,
    )


__all__ = ["EligibilityResult", "compute_eligibility"]
