"""Re-scale the features of the published TESSERA cache for the spatial holdout.

The published prebuilt cache holds the 22-channel feature stack scaled with the
published scaler, with every non-finite pixel set to 0.0 afterwards
(``RealPatchReader._read_features``). The holdout needs the same features scaled
with a scaler refit on the remaining train anchors. Per kept channel the stored
value ``x`` is mapped as

    t = x * std_published + mean_published      # the (log1p) value before scaling
    y = (t - mean_refit) / std_refit            # as ``ScalerSpec.apply``

and a stored exact 0.0 stays 0.0 (it was a filled pixel). The result equals a
job-local build with the refit scaler up to float32 rounding, provided a stored
0.0 can only be a filled pixel. That holds when the published mean of every kept
channel is not a value the input can take (``assert_zero_means_filled``) and no
patch has more stored zeros than recorded filled pixels (``scan_patch``). Both
fail closed before training. Only the kept channels are remapped; morphology and
shadow channels, which the TESSERA arm replaces, are left as stored.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from berlin_lst_downscaling.modeling.contracts import REAL_PATCH_PX
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, ordered_bounded_map
from berlin_lst_downscaling.modeling.patches import (
    RealSample,
    ScalerChannel,
    ScalerSpec,
    merge_scaler_moments,
    scaler_from_moments,
    scaler_moments,
)

# The quotient ``(x - mean) / std`` must stay this far above zero for every input that is not the
# mean; float32 underflows to 0.0 below 1.4e-45 (denormal) and 1.2e-38 (normal).
MIN_NONZERO_QUOTIENT = 1e-30
_ULP_NEIGHBOURS = 3


def _channels(published: ScalerSpec, channel_indices: Sequence[int], columns: Sequence[int]):
    return [published.channels[channel_indices[column]] for column in columns]


def assert_zero_means_filled(
    published: ScalerSpec, channel_indices: Sequence[int], columns: Sequence[int]
) -> None:
    """Prove per kept channel that a stored 0.0 cannot be a genuine value.

    The reader evaluates ``(float64(log1p)(float32 x) - mean) / std`` in float64 and
    stores the quotient as float32. The quotient is 0.0 only if ``x`` maps exactly onto
    the mean or the quotient underflows. For ``zscore`` the inputs are the float32
    values, so the nearest one to the mean decides; for ``log1p_zscore`` the inputs
    are ``log1p`` of float32 values near ``expm1(mean)`` (``log1p`` is monotone and its
    float64 error is far below one float32 step). The smallest distance over these
    candidates divided by std must be nonzero and above ``MIN_NONZERO_QUOTIENT``.
    """
    for column, ch in zip(columns, _channels(published, channel_indices, columns), strict=True):
        if ch.transform not in ("zscore", "log1p_zscore"):
            raise RuntimeError(f"kept channel {ch.name!r} has transform {ch.transform!r}")
        if ch.mean is None or not ch.std:
            raise RuntimeError(f"kept channel {ch.name!r} has no published mean and std")
        mean, std = float(ch.mean), float(ch.std)
        centre = np.float32(np.expm1(mean) if ch.transform == "log1p_zscore" else mean)
        candidates = [centre]
        for direction in (np.float32(np.inf), np.float32(-np.inf)):
            value = centre
            for _ in range(_ULP_NEIGHBOURS):
                value = np.nextafter(value, direction)
                candidates.append(value)
        values = np.asarray(candidates, dtype=np.float64)
        if ch.transform == "log1p_zscore":
            values = np.log1p(values)
        gap = float(np.min(np.abs(values - mean)))
        if gap == 0.0 or gap / abs(std) < MIN_NONZERO_QUOTIENT:
            raise RuntimeError(
                f"kept channel {ch.name!r} (cache column {column}): the published mean {mean!r} "
                f"is hit or nearly hit by an input value (distance {gap:.3g}); a stored 0.0 "
                "could be a genuine value, so the cache cannot be re-scaled"
            )


@dataclass(frozen=True)
class FeatureRemap:
    """Per-column re-scaling of cached features from the published to the refit scaler."""

    columns: tuple[int, ...]
    published_mean: tuple[float, ...]
    published_std: tuple[float, ...]
    refit_mean: tuple[float, ...]
    refit_std: tuple[float, ...]

    def __call__(self, features: np.ndarray) -> np.ndarray:
        out = np.array(features, dtype=np.float32, copy=True)
        for j, column in enumerate(self.columns):
            stored = features[column]
            t = stored.astype(np.float64) * self.published_std[j] + self.published_mean[j]
            y = (t - self.refit_mean[j]) / self.refit_std[j]
            out[column] = np.where(stored == 0.0, 0.0, y).astype(np.float32)
        return out


def scan_patch(sample: RealSample, columns: Sequence[int]) -> tuple[int, int]:
    """Return ``(zeros, filled)`` of one cached patch; more zeros than fills raises.

    ``filled`` is the recorded ``filled_feature_pixels`` (non-finite pixel-channels over
    all 22 channels, shadows included); ``zeros`` counts the stored exact 0.0 in the kept
    channels. Since a stored 0.0 is never a genuine value, ``zeros <= filled`` always.
    """
    zeros = int((sample.features[list(columns)] == 0.0).sum())
    filled = sample.meta.filled_feature_pixels
    if zeros > filled:
        raise RuntimeError(
            f"patch {sample.meta.patch_id}: {zeros} stored zeros in the kept feature channels "
            f"but only {filled} recorded filled pixels; the cache cannot be re-scaled"
        )
    return zeros, filled


def fit_feature_remap(
    cache: PatchCache,
    *,
    published: ScalerSpec,
    channel_indices: Sequence[int],
    columns: Sequence[int],
    train_rows: Sequence[int],
    check_rows: Sequence[int],
    workers: int,
) -> tuple[FeatureRemap, ScalerSpec, dict[str, Any]]:
    """Check the cache, refit the kept channels on ``train_rows`` and return the remap.

    ``check_rows`` are all rows the run reads (train rows kept and validation rows);
    ``train_rows`` must be among them. Moments use the pixels of eligible cells whose
    stored value is not 0.0, the pixel set of ``scaler_moments`` on the raw features.
    Returns ``(remap, refit scaler, evidence)``; the scaler keeps the published entries
    of all channels that are not kept.
    """
    assert_zero_means_filled(published, channel_indices, columns)
    kept = list(columns)
    chans = _channels(published, channel_indices, kept)
    moment_spec = ScalerSpec(
        channel_order=tuple(ch.name for ch in chans),
        channels=tuple(dataclasses.replace(ch, transform="zscore") for ch in chans),
        policy_hash=published.policy_hash,
    )
    pub_std = np.asarray([ch.std for ch in chans], dtype=np.float64)[:, None, None]
    pub_mean = np.asarray([ch.mean for ch in chans], dtype=np.float64)[:, None, None]
    train = set(train_rows)

    def scan(row: int) -> tuple[int, int, np.ndarray | None]:
        sample = cache[row]
        zeros, filled = scan_patch(sample, kept)
        if row not in train:
            return zeros, filled, None
        stored = sample.features[kept]
        t = stored.astype(np.float64) * pub_std + pub_mean
        t[stored == 0.0] = np.nan
        eligible = np.repeat(np.repeat(sample.mask_100m[0], 10, axis=0), 10, axis=1)
        if eligible.shape != (REAL_PATCH_PX, REAL_PATCH_PX):
            raise RuntimeError(f"patch {sample.meta.patch_id}: unexpected mask shape")
        return zeros, filled, scaler_moments(t, eligible, moment_spec)

    total = None
    zeros_total = filled_total = 0
    parts = ordered_bounded_map(scan, list(check_rows), max_workers=workers)
    try:
        for zeros, filled, moments in parts:
            zeros_total += zeros
            filled_total += filled
            if moments is not None:
                total = merge_scaler_moments(total, moments)
    finally:
        parts.close()
    if total is None:
        raise RuntimeError("the holdout scaler fit read no train patch")
    fit_spec = ScalerSpec(
        channel_order=moment_spec.channel_order,
        channels=tuple(chans),
        policy_hash=published.policy_hash,
    )
    fitted = scaler_from_moments(fit_spec, total)
    if any(not ch.std for ch in fitted.channels):
        raise RuntimeError("the refit scaler has a kept channel without spread")
    replacements: dict[int, ScalerChannel] = {
        channel_indices[column]: fit for column, fit in zip(kept, fitted.channels, strict=True)
    }
    refit = ScalerSpec(
        channel_order=published.channel_order,
        channels=tuple(replacements.get(i, ch) for i, ch in enumerate(published.channels)),
        policy_hash=published.policy_hash,
    )
    remap = FeatureRemap(
        columns=tuple(kept),
        published_mean=tuple(float(ch.mean) for ch in chans if ch.mean is not None),
        published_std=tuple(float(ch.std) for ch in chans if ch.std is not None),
        refit_mean=tuple(float(ch.mean) for ch in fitted.channels if ch.mean is not None),
        refit_std=tuple(float(ch.std) for ch in fitted.channels if ch.std is not None),
    )
    evidence = {
        "source": "published prebuilt cache, features re-scaled with the refit scaler",
        "kept_channels": [ch.name for ch in chans],
        "patches_checked": len(check_rows),
        "patches_refit": len(train_rows),
        "filled_pixels_recorded": filled_total,
        "zero_pixels_kept_channels": zeros_total,
    }
    return remap, refit, evidence
