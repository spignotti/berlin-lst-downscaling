# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for the real WB3 patch reader.

Read-only probe over the published sources. For a deterministic sample of
accepted patches it re-derives, **with its own method**, what
``modeling/patches.py`` produces, and compares:

- raster placement: the Landsat target COG is on the global canonical
  EPSG:25833 100 m lattice, unrotated, and the eligibility mask shares the
  reader's analysis-grid alignment;
- the 100 m supervision: the native target window and the
  ``training_eligible@100m`` window recomputed from the COGs, plus the mask
  cell count against the index's ``n_eligible``;
- the 1000 m ``lst_prior``: each needed block is averaged by reading that
  block's own 100 m window directly (not the reader's global accumulation),
  and the expanded patch window is checked to be piecewise constant per
  block, including across a 1000 m boundary inside the patch;
- the model-input fill count, recomputed by applying the published scaler
  independently and counting non-finite pixels.

The geometry constants, the block arithmetic, and the scaler application are
mirrored here rather than imported, so a reader bug cannot validate itself.
Nothing is written and no canonical artifact is touched.

Usage
-----
    uv run python scripts/validators/validate_real_patches.py \
        --patch-index-root gs://berlin-lst-training-data/training/patch-index/v1 \
        --training-root gs://berlin-lst-training-data/training/v1 \
        --features-root gs://berlin-lst-training-data/features/v3 \
        --ard-root gs://berlin-lst-training-data/ard/full/2017-2026-cutoff-20260717T235959Z \
        --per-split 2
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import rasterio
from rasterio.windows import Window

from berlin_lst_downscaling.data.io import read_bytes, resolve_canonical_uri
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSourceConfig,
    load_patch_refs,
)

# Mirrored constants — the validator must not import the reader's geometry.
_CANON_X = 369190.0
_CANON_Y = 5838410.0
_CELL_100 = 100.0
_PATCH_CELLS = 16
_PATCH_PX = 160
_BLOCK_CELLS = 10  # 1000 m = 10 x 100 m cells
_BLOCK_PX = 100  # 1000 m = 100 x 10 m pixels
_LST_MIN = 150.0
_LST_MAX = 400.0
_FEATURE_CHANNELS = 28


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _canon_offset(transform, res: float) -> tuple[int, int]:
    col0 = round((transform.xoff - _CANON_X) / res)
    row0 = round((_CANON_Y - transform.yoff) / res)
    return col0, row0


def _feature_cog(root: str, scene_id: str) -> str:
    return f"{root.rstrip('/')}/{scene_id}/{scene_id}.tif"


# ── independent scaler ────────────────────────────────────────────────


def _independent_scaler(training_root: str) -> list[tuple[str, float | None, float | None]]:
    """Return ``(transform, mean, std)`` per channel, in published order."""
    payload = _read_json(f"{training_root.rstrip('/')}/scaler.json")
    order = list(payload["channel_order"])
    if len(order) != _FEATURE_CHANNELS:
        raise RuntimeError(f"scaler publishes {len(order)} channels, expected {_FEATURE_CHANNELS}")
    return [(str(c["transform"]), c.get("mean"), c.get("std")) for c in payload["channels"]]


def _apply_scaler_independently(
    bands: np.ndarray, spec: list[tuple[str, float | None, float | None]]
) -> np.ndarray:
    """Apply the mirrored scaler transforms and return the scaled copy."""
    out = bands.astype(np.float32, copy=True)
    for i, (transform, mean, std) in enumerate(spec):
        if transform == "identity":
            continue
        band = out[i]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        vals = band[finite].astype(np.float64)
        if transform == "log1p_zscore":
            vals = np.log1p(vals)
        if mean is not None and std:
            vals = (vals - mean) / std
        band[finite] = vals
    return out


# ── independent prior (local per-block windowed reads) ────────────────


class LandsatProbe:
    """Recompute 1000 m block means by reading each block's own 100 m window.

    Deliberately a different method from the reader's single global
    accumulation: here every needed block is averaged from a direct window
    read, which also cross-checks the reader's block indexing.
    """

    def __init__(self, cog: str, flag: str) -> None:
        with rasterio.open(cog) as src:
            if str(src.crs) != "EPSG:25833":
                raise RuntimeError(f"landsat_cog CRS {src.crs!r} != EPSG:25833")
            if src.transform.b or src.transform.d:
                raise RuntimeError("landsat_cog transform is rotated")
            if not math.isclose(src.transform.a, _CELL_100) or not math.isclose(
                abs(src.transform.e), _CELL_100
            ):
                raise RuntimeError("landsat_cog is not at 100 m")
            self.col0, self.row0 = _canon_offset(src.transform, _CELL_100)
            self.height, self.width = src.height, src.width
        with rasterio.open(flag) as fsrc:
            if (fsrc.height, fsrc.width) != (self.height, self.width):
                raise RuntimeError("landsat flag shape differs from the LST COG")
        self.cog = cog
        self.flag = flag
        self._cache: dict[tuple[int, int], float | None] = {}

    def block_mean(self, brow: int, bcol: int) -> float | None:
        """Mean over the block's native-valid 100 m cells, or None if none."""
        if (brow, bcol) in self._cache:
            return self._cache[(brow, bcol)]
        # Global block (brow, bcol) covers global 100 m rows/cols [x*10, x*10+10).
        r0 = brow * _BLOCK_CELLS - self.row0
        c0 = bcol * _BLOCK_CELLS - self.col0
        # Clip to the COG; cells outside it simply do not exist in the product.
        r_lo, r_hi = max(r0, 0), min(r0 + _BLOCK_CELLS, self.height)
        c_lo, c_hi = max(c0, 0), min(c0 + _BLOCK_CELLS, self.width)
        value: float | None = None
        if r_lo < r_hi and c_lo < c_hi:
            window = Window.from_slices((r_lo, r_hi), (c_lo, c_hi))
            with rasterio.open(self.cog) as src:
                st = src.read(1, window=window).astype(np.float64)
            with rasterio.open(self.flag) as fsrc:
                flag = fsrc.read(1, window=window)
            valid = np.isfinite(st) & (flag == 0) & (st >= _LST_MIN) & (st <= _LST_MAX)
            if valid.any():
                value = float(st[valid].mean())
        self._cache[(brow, bcol)] = value
        return value


def _independent_prior_window(probe: LandsatProbe, row: int, col: int) -> np.ndarray | None:
    """Expand independently recomputed block means to the 160x160 10 m prior."""
    rows = (row * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    cols = (col * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    rb_lo, rb_hi = int(rows.min()), int(rows.max())
    cb_lo, cb_hi = int(cols.min()), int(cols.max())
    grid = np.empty((rb_hi - rb_lo + 1, cb_hi - cb_lo + 1), dtype=np.float64)
    for br in range(rb_lo, rb_hi + 1):
        for bc in range(cb_lo, cb_hi + 1):
            mean = probe.block_mean(br, bc)
            if mean is None:
                return None
            grid[br - rb_lo, bc - cb_lo] = mean
    return grid[np.ix_(rows - rb_lo, cols - cb_lo)]


# ── per-patch comparison ──────────────────────────────────────────────


def _check_patch(
    ref: PatchRef, reader: RealPatchReader, sample: RealSample, errors: list[str]
) -> dict:
    """Compare one patch's reader output against independent recomputation."""
    prior = reader.scene_prior(ref.scene_id)
    if prior is None:
        errors.append(f"{ref.patch_id}: reader could not resolve the Landsat target")
        return {}

    probe = LandsatProbe(prior.landsat_cog, prior.landsat_flag)
    if (probe.col0, probe.row0) != (prior.col0, prior.row0):
        errors.append(
            f"{ref.patch_id}: Landsat grid offset differs (validator "
            f"{(probe.col0, probe.row0)} != reader {(prior.col0, prior.row0)})"
        )

    dcol10, drow10, dcol100, drow100 = reader.analysis_offsets

    # ── target + mask, recomputed from the COGs ───────────────────────
    r0 = ref.row - probe.row0
    c0 = ref.col - probe.col0
    with rasterio.open(prior.landsat_cog) as src:
        target = src.read(
            1, window=Window.from_slices((r0, r0 + _PATCH_CELLS), (c0, c0 + _PATCH_CELLS))
        ).astype(np.float32)
    with rasterio.open(resolve_canonical_uri(ref.eligibility_mask)) as msk:
        moff = _canon_offset(msk.transform, _CELL_100)
        if moff != (dcol100, drow100):
            errors.append(f"{ref.patch_id}: eligibility mask is not aligned with the analysis grid")
        mr0 = ref.row - moff[1]
        mc0 = ref.col - moff[0]
        mask = msk.read(
            1, window=Window.from_slices((mr0, mr0 + _PATCH_CELLS), (mc0, mc0 + _PATCH_CELLS))
        )
    n_mask = int((mask == 1).sum())
    if n_mask != ref.n_eligible:
        errors.append(
            f"{ref.patch_id}: mask holds {n_mask} eligible cells, index says {ref.n_eligible}"
        )
    if not np.array_equal(target[None], sample.target_100m):
        errors.append(f"{ref.patch_id}: reader target_100m differs from the native COG window")
    if not np.array_equal(mask == 1, sample.mask_100m[0]):
        errors.append(f"{ref.patch_id}: reader mask_100m differs from the published mask")

    # ── prior, recomputed and checked across block boundaries ─────────
    expected_prior = _independent_prior_window(probe, ref.row, ref.col)
    if expected_prior is None:
        errors.append(f"{ref.patch_id}: reader produced a prior for an unavailable block")
        return {"n_mask": n_mask, "filled": 0, "distinct_axes": (0, 0)}
    if not np.allclose(expected_prior, sample.lst_prior_k[0], atol=1e-4):
        errors.append(f"{ref.patch_id}: reader lst_prior differs from independent block means")

    row_block = (ref.row * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    col_block = (ref.col * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    prior_2d = sample.lst_prior_k[0]
    for rb in sorted(set(row_block.tolist())):
        for cb in sorted(set(col_block.tolist())):
            sub = prior_2d[np.ix_(row_block == rb, col_block == cb)]
            block_mean = probe.block_mean(int(rb), int(cb))
            if block_mean is None or not np.allclose(sub, block_mean, atol=1e-4):
                errors.append(
                    f"{ref.patch_id}: prior block ({rb}, {cb}) is not constant at the "
                    f"independently recomputed value"
                )

    # ── fill count, recomputed from the raw feature COG ───────────────
    spec = _independent_scaler(reader.cfg.training_root)
    fr0 = ref.row * 10 - drow10
    fc0 = ref.col * 10 - dcol10
    with rasterio.open(_feature_cog(reader.cfg.features_root, ref.scene_id)) as fsrc:
        raw = fsrc.read(
            window=Window.from_slices((fr0, fr0 + _PATCH_PX), (fc0, fc0 + _PATCH_PX))
        ).astype(np.float32)
    scaled = _apply_scaler_independently(raw, spec)
    expected_filled = int((~np.isfinite(scaled)).sum())
    if expected_filled != sample.meta.filled_feature_pixels:
        errors.append(
            f"{ref.patch_id}: reader reports {sample.meta.filled_feature_pixels} filled pixels, "
            f"independent recomputation gives {expected_filled}"
        )
    if not np.isfinite(sample.features).all():
        errors.append(f"{ref.patch_id}: reader features are not finite")
    distinct_axes = (len(set(row_block.tolist())), len(set(col_block.tolist())))
    return {"n_mask": n_mask, "filled": expected_filled, "distinct_axes": distinct_axes}


def _spans_block_boundary(row: int, col: int) -> bool:
    """True when the 160 px window covers more than one 1000 m block on an axis.

    A patch anchored on a multiple of 10 still spans a boundary: 16 cells is
    wider than a 10-cell block. This mirrors the block indexing used in the
    per-block constancy check.
    """
    rows = (row * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    cols = (col * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX
    return int(rows.min()) != int(rows.max()) or int(cols.min()) != int(cols.max())


def _select_refs(refs: list[PatchRef], per_split: int) -> list[PatchRef]:
    """Pick deterministically, guaranteeing a block-boundary-crossing patch."""
    by_split: dict[str, list[PatchRef]] = {}
    for ref in refs:
        by_split.setdefault(ref.split, []).append(ref)
    chosen: list[PatchRef] = []
    for split in sorted(by_split):
        split_refs = by_split[split]
        picked = split_refs[:per_split]
        if not any(_spans_block_boundary(r.row, r.col) for r in picked):
            boundary = next(
                (r for r in split_refs if _spans_block_boundary(r.row, r.col)), None
            )
            if boundary is not None:
                picked = [*picked[: max(0, per_split - 1)], boundary]
        chosen.extend(picked)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the real WB3 patch reader.")
    parser.add_argument("--patch-index-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--ard-root", required=True)
    parser.add_argument(
        "--per-split", type=int, default=2, help="patches to check per split (default 2)"
    )
    args = parser.parse_args()

    cfg = RealSourceConfig(
        patch_index_root=args.patch_index_root.rstrip("/"),
        training_root=args.training_root.rstrip("/"),
        features_root=args.features_root.rstrip("/"),
        ard_root=args.ard_root.rstrip("/"),
    )
    errors: list[str] = []
    print(f"Validating real patch reader against {cfg.patch_index_root}")

    refs = load_patch_refs(cfg)
    selected = _select_refs(refs, max(1, args.per_split))
    if not selected:
        print("FAIL: no accepted patches for the configured splits")
        return 1

    reader = RealPatchReader(cfg)
    filled_total = 0
    crossing = 0
    checked = 0
    for ref in selected:
        sample = reader.read_patch(ref)
        if sample is None:
            errors.append(f"{ref.patch_id}: reader returned no sample")
            continue
        info = _check_patch(ref, reader, sample, errors)
        checked += 1
        filled_total += int(info.get("filled", 0))
        axes = info.get("distinct_axes", (1, 1))
        crossing += int(axes[0] > 1 or axes[1] > 1)
        print(
            f"  checked {ref.patch_id} [{ref.split}] mask={info.get('n_mask')} "
            f"filled={info.get('filled')} prior_axes={axes}"
        )

    for e in errors:
        print(f"  ✗ {e}")
    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    print(
        f"OK: real patch reader valid ({checked} patches, {crossing} crossing a 1000 m "
        f"boundary, {filled_total} filled predictor pixels)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
