"""Train-only scaler fit — streaming per-channel statistics over eligible pixels.

The scaler statistics are derived **exclusively** from the training split
(2017-2023) and **exclusively** from training-eligible cells (user-mandated):
for every train scene, only the 10 m feature pixels that fall inside an
eligible 100 m cell contribute. Validation, test, and 2026 inference scenes
never contribute.

Transform policy (user-mandated, mirrored in ``contracts.py``):

- continuous channels: z-score ``(x - mean) / std``;
- the three precipitation channels: ``log1p`` then z-score;
- the two shadow channels: identity (0/1 unchanged, no mean/std).

Statistics are accumulated with a numerically stable streaming Welford
update (population variance, ``ddof=0``) over fixed-order tiles and scenes,
so the result is deterministic and mergeable. The feature COG is read
tiled (2560x2560 10 m) so peak memory stays bounded.
"""

from __future__ import annotations

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_CHANNELS,
)
from berlin_lst_downscaling.data.features.paths import feature_cog
from berlin_lst_downscaling.data.qa.stage1_raw import _window_offset
from berlin_lst_downscaling.data.training.contracts import (
    PRECIP_CHANNELS,
    SHADOW_CHANNELS,
)
from berlin_lst_downscaling.data.training.paths import eligibility_cog
from berlin_lst_downscaling.data.training.report import SceneTrainingResult

_TILE = 2560


class _Welford:
    """Streaming mean/variance accumulator (population, ddof=0)."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        """Accumulate a 1-D float64 array of valid samples."""
        if values.size == 0:
            return
        # Batch Welford update — equivalent to per-sample updates.
        n = values.size
        delta = values - self.mean
        self.mean += np.sum(delta) / (self.count + n)
        delta2 = values - self.mean
        self.m2 += float(np.sum(delta * delta2))
        self.count += n

    def update_count(self, n: int) -> None:
        """Accumulate samples without statistics (identity channels)."""
        self.count += n

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.m2 / self.count))


def _transform_for(channel_name: str) -> str:
    if channel_name in PRECIP_CHANNELS:
        return "log1p_zscore"
    if channel_name in SHADOW_CHANNELS:
        return "identity"
    return "zscore"


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values  # not accumulated below, but keep the contract explicit
    if transform == "log1p_zscore":
        return np.log1p(values)
    return values


def fit_scaler(
    results: list[SceneTrainingResult],
    *,
    features_root: str,
    output_root: str,
    grid_10m: GeoBox,
) -> dict:
    """Fit per-channel scaler statistics on train-eligible pixels only.

    Parameters
    ----------
    results :
        Per-scene run results (only ``split == "train"`` scenes with
        ``eligible_cells > 0`` contribute).
    features_root :
        Feature Release V3 root (feature COG URIs).
    output_root :
        Training release root (eligibility mask URIs).
    grid_10m :
        10 m analysis grid (full canonical grid or a canonical-aligned
        subset for smoke).

    Returns a scaler payload (``channel_order`` + per-channel stats).
    """
    n_channels = len(FEATURE_CHANNELS)
    acc = [_Welford() for _ in range(n_channels)]

    train_scenes = [s for s in results if s.split == "train" and (s.eligible_cells or 0) > 0]

    for scene in train_scenes:
        mask_uri = eligibility_cog(output_root, scene.scene_id)
        cog_uri = feature_cog(features_root, scene.scene_id)

        # Read the eligibility mask at 100 m (tiny), upsample to 10 m.
        with rasterio.open(mask_uri) as src:
            eligible_100 = src.read(1) == 1
        eligible_10 = np.repeat(np.repeat(eligible_100, 10, axis=0), 10, axis=1)

        cog_off = (0, 0)
        with rasterio.open(cog_uri) as cog:
            cog_off = _window_offset(cog, grid_10m, 10.0)
            for r0 in range(0, grid_10m.shape.y, _TILE):
                r1 = min(r0 + _TILE, grid_10m.shape.y)
                for c0 in range(0, grid_10m.shape.x, _TILE):
                    c1 = min(c0 + _TILE, grid_10m.shape.x)
                    bh, bw = r1 - r0, c1 - c0
                    win = Window.from_slices(
                        (r0 + cog_off[1], r0 + cog_off[1] + bh),
                        (c0 + cog_off[0], c0 + cog_off[0] + bw),
                    )
                    bands = cog.read(window=win)  # (28, bh, bw) float32
                    sel = eligible_10[r0:r1, c0:c1]
                    for i, spec in enumerate(FEATURE_CHANNELS):
                        transform = _transform_for(spec.name)
                        if transform == "identity":
                            # Shadows are 0/1 and stay unchanged — count the
                            # samples for the record but accumulate no stats.
                            acc[i].update_count(int(sel.sum()))
                            continue
                        vals = _apply_transform(bands[i][sel].astype(np.float64), transform)
                        acc[i].update(vals)

    channels = []
    for i, spec in enumerate(FEATURE_CHANNELS):
        transform = _transform_for(spec.name)
        entry: dict = {
            "channel_index": i + 1,
            "channel_name": spec.name,
            "transform": transform,
            "count": acc[i].count,
        }
        if transform == "identity":
            entry["mean"] = None
            entry["std"] = None
        else:
            entry["mean"] = round(float(acc[i].mean), 8)
            entry["std"] = round(acc[i].std(), 8)
        channels.append(entry)

    return {
        "channel_order": list(FEATURE_CHANNEL_NAMES),
        "statistics_grain": "10m feature pixels within train-eligible 100m cells",
        "variance": "population (ddof=0), Welford streaming",
        "channels": channels,
    }


__all__ = ["fit_scaler"]
