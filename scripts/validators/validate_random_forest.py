# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "scikit-learn>=1.9.1",
#     "skops>=0.16.0",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for a random-forest residual baseline report (issue #59).

Read-only probe over a written ``random_forest_report.json`` plus the run's
model/rows/probe artifacts and the published sources. It re-derives, with its
own code and mirrored geometry constants:

- provenance: the report fingerprints equal the published index completion
  marker, whose ``patches_total`` equals the index row count;
- the train selection: the rows file SHA-256 and the selection digest match
  the report, the keys are unique, inside the 2017-2023 train contract, and
  each lies in a published train window of its scene; the count respects the
  reservoir cap and the candidate/duplicate arithmetic;
- per-stage structure: the channel subset is the frozen Tag-11 ladder, the
  model artifact SHA-256 matches, the serialized replay is marked identical,
  and every per-split summary equals the sum of its per-patch records;
- coverage and split separation: the scored IDs of a split are the unique,
  in-order subsequence of the first ``requested`` indexed rows, every record
  carries the split's year (2024/2025) and the index ``n_eligible`` cell
  count, and all stages scored identical ID sets and cell counts;
- the comparison universe: against the naive prior-expand evidence, the same
  scored IDs, valid-cell counts and feature-fill counts in the covered window
  - and, where the naive scope covers the whole requested window, the same
  exclusion budget;
- numerics on a bounded patch sample: features, prior, target and mask
  rebuilt from the published COGs, the masked absolute-error sum recomputed
  through its own 10 m reconstruction and 10x10 block mean, and the SSIM
  support count;
- the probe fixture: the rebuilt arrays are compared with the stored ones and
  the saved forest is replayed on the stored features - the residual must be
  bitwise identical per stage.

The validator never writes and never touches a canonical artifact. Exit code 0
means every check passed.

Usage
-----
    uv run python scripts/validators/validate_random_forest.py \
        --report data/smoke/random-forest/random_forest_report.json \
        --max-patches 5
"""

from __future__ import annotations

import argparse
import io
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
import sklearn
from rasterio.windows import Window

from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.io import read_bytes, resolve_canonical_uri
from berlin_lst_downscaling.modeling.random_forest import load_model

_CANON_X = 369190.0
_CANON_Y = 5838410.0
_CELL_100 = 100.0
_CELL_10 = 10.0
_PATCH_CELLS = 16
_PATCH_PX = 160
_POOL = 10
_BLOCK_CELLS = 10  # 1000 m = 10 x 100 m cells
_BLOCK_PX = 100  # 1000 m = 100 x 10 m pixels
_LST_MIN = 150.0
_LST_MAX = 400.0
_SSIM_WINDOW = 7  # must match modeling/metrics.py
_STRIDE_CELLS = 16  # must match the published patch-index stride
_TRAIN_YEARS = frozenset(range(2017, 2024))
_SPLIT_YEARS = {"validation": 2024, "test": 2025}

# The frozen Tag-11 ladder as V3 channel indices (issue #57), in the order the
# ablation configs select: stage 3 shadows sit after morphology, and stage 4
# appends ERA5 after the shadows - a named subset, not the first-C prefix.
_STAGE_CHANNEL_INDICES: dict[int, tuple[int, ...]] = {
    1: tuple(range(0, 10)),
    2: tuple(range(0, 18)),
    3: tuple(range(0, 18)) + (26, 27),
    4: tuple(range(0, 18)) + (26, 27) + tuple(range(18, 26)),
}
_EXPECTED_FEATURE_ORDER = {
    1: "v3_first_c",
    2: "v3_first_c",
    3: "v3_named_subset",
    4: "v3_named_subset",
}

# The baseline validator's tolerance reasoning applies here too: the report
# accumulates float32 pooling while this validator accumulates float64, so a
# summed error over ~250 cells differs by ~1e-6 relative. A wrong row, mask,
# or model shifts a sum by whole Kelvin and cannot pass.
_SUM_RTOL = 1e-5
_SUM_ATOL = 1e-3
# Rebuilt native features must agree with the stored probe matrices.
_FEATURE_ATOL = 1e-3
_MAX_REPORTED = 8


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _canon_offset(transform, res: float) -> tuple[int, int]:
    return (
        round((transform.xoff - _CANON_X) / res),
        round((_CANON_Y - transform.yoff) / res),
    )


def _check_raster(handle, *, res: float, scene: str, what: str) -> tuple[int, int]:
    """Check CRS, resolution, rotation, and canonical alignment of a raster."""
    if str(handle.crs) != "EPSG:25833":
        raise RuntimeError(f"{scene}: {what} CRS is {handle.crs!r}, expected EPSG:25833")
    if handle.transform.b or handle.transform.d:
        raise RuntimeError(f"{scene}: {what} transform is rotated")
    if not np.isclose(handle.transform.a, res) or not np.isclose(abs(handle.transform.e), res):
        raise RuntimeError(f"{scene}: {what} resolution is not {res} m")
    offset = _canon_offset(handle.transform, res)
    if offset != (0, 0):
        raise RuntimeError(f"{scene}: {what} is not on the canonical grid (offset {offset})")
    return offset


# ── published lookups ─────────────────────────────────────────────────


def _load_index(index_root: str) -> tuple[list[dict], dict[str, dict]]:
    uri = f"{index_root.rstrip('/')}/patch_index.parquet"
    rows = pq.read_table(io.BytesIO(read_bytes(uri))).to_pylist()
    return rows, {str(r["patch_id"]): r for r in rows}


def _load_landsat(ard_root: str) -> dict[str, tuple[str, str]]:
    uri = f"{ard_root.rstrip('/')}/ledger.parquet"
    cols = pq.read_table(io.BytesIO(read_bytes(uri))).to_pydict()
    cog_col = cols.get("path_cog", [None] * len(cols["source"]))
    flag_col = cols.get("path_flag", [None] * len(cols["source"]))
    out: dict[str, tuple[str, str]] = {}
    for i, source in enumerate(cols["source"]):
        if str(source) != "landsat-c2-l2" or str(cols["status"][i]) != "done":
            continue
        out[str(cols["scene_id"][i])] = (
            resolve_canonical_uri(str(cog_col[i] or "")),
            resolve_canonical_uri(str(flag_col[i] or "")),
        )
    return out


def _load_scaler(training_root: str) -> dict:
    scaler = _read_json(resolve_canonical_uri(f"{training_root.rstrip('/')}/scaler.json"))
    order = tuple(str(name) for name in scaler.get("channel_order", ()))
    if order != FEATURE_CHANNEL_NAMES:
        raise RuntimeError(f"scaler channel order drift: {order}")
    if len(scaler.get("channels", ())) != len(FEATURE_CHANNEL_NAMES):
        raise RuntimeError("scaler channel count drift")
    return scaler


# ── independent recomputation ─────────────────────────────────────────


def _block_grid(landsat_cog: str, landsat_flag: str) -> tuple[dict, dict]:
    """Whole-COG 1000 m block means over native-valid cells, keyed globally."""
    with rasterio.open(landsat_cog) as src:
        _check_raster(src, res=_CELL_100, scene=landsat_cog, what="Landsat COG")
        st = src.read(1).astype(np.float64)
        col0, row0 = _canon_offset(src.transform, _CELL_100)
        shape = st.shape
    with rasterio.open(landsat_flag) as fsrc:
        if (fsrc.height, fsrc.width) != shape:
            raise RuntimeError("landsat flag shape differs from the LST COG")
        if _canon_offset(fsrc.transform, _CELL_100) != (col0, row0):
            raise RuntimeError("landsat flag grid differs from the LST COG")
        flag = fsrc.read(1)

    valid = np.isfinite(st) & (flag == 0) & (st >= _LST_MIN) & (st <= _LST_MAX)
    rows = np.arange(shape[0], dtype=np.int64)[:, None] + row0
    cols = np.arange(shape[1], dtype=np.int64)[None, :] + col0
    brow = rows // _BLOCK_CELLS
    bcol = cols // _BLOCK_CELLS
    brow0, bcol0 = int(brow.min()), int(bcol.min())
    nbr = int(brow.max()) - brow0 + 1
    nbc = int(bcol.max()) - bcol0 + 1
    flat = (((brow - brow0) * nbc) + (bcol - bcol0)).ravel()
    sel = valid.ravel()
    counts = np.bincount(flat[sel], minlength=nbr * nbc)
    sums = np.bincount(flat[sel], weights=st.ravel()[sel], minlength=nbr * nbc)
    means = np.full(nbr * nbc, np.nan, dtype=np.float64)
    ok = counts > 0
    means[ok] = sums[ok] / counts[ok]
    grid = {
        "means": means.reshape(nbr, nbc).astype(np.float32),
        "brow0": brow0,
        "bcol0": bcol0,
    }
    return grid, {"col0": col0, "row0": row0}


def _prior_cells(grid: dict, row: int, col: int) -> np.ndarray | None:
    """Return the per-cell 100 m physical prior, or None when unavailable.

    A cell lies inside one 1000 m block; the reader's block-expanded 10 m
    prior is constant on that block, so its pooled 100 m value is the block
    mean.
    """
    rows = (row + np.arange(_PATCH_CELLS)) // _BLOCK_CELLS - grid["brow0"]
    cols = (col + np.arange(_PATCH_CELLS)) // _BLOCK_CELLS - grid["bcol0"]
    means = grid["means"]
    if (
        rows.min() < 0
        or cols.min() < 0
        or rows.max() >= means.shape[0]
        or cols.max() >= means.shape[1]
    ):
        return None
    values = means[np.ix_(rows, cols)]
    if not np.isfinite(values).all():
        return None
    return values.astype(np.float32)


def _scale_bands(bands: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply the published train-only scaler exactly as the reader does.

    The reader applies the i-th published scaler channel to band i; the
    published ``channel_index`` is the 1-based band number, so it must equal
    ``i + 1`` for every entry.
    """
    out = bands.astype(np.float32, copy=True)
    for position, channel in enumerate(scaler["channels"]):
        if int(channel["channel_index"]) != position + 1:
            raise RuntimeError(
                f"scaler channel {position}: published index "
                f"{channel['channel_index']} is not the 1-based band position"
            )
        transform = str(channel["transform"])
        if transform == "identity":
            continue
        band = out[position]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        values = band[finite].astype(np.float64)
        if transform == "log1p_zscore":
            values = np.log1p(values)
        if channel.get("mean") is not None and channel.get("std"):
            values = (values - float(channel["mean"])) / float(channel["std"])
        band[finite] = values
    return out


def _pool_10m_cells_to_100m(values: np.ndarray) -> np.ndarray:
    """Mirror of the module's exact nested 10x10 block mean (float64 once)."""
    c, h, w = values.shape
    reshaped = values.astype(np.float64).reshape(c, h // _POOL, _POOL, w // _POOL, _POOL)
    return reshaped.mean(axis=(2, 4)).astype(np.float32)


def _read_features(
    scene_id: str, row: int, col: int, features_root: str, scaler: dict
) -> tuple[np.ndarray, int]:
    """Read, scale, and zero-fill one native 10 m feature window."""
    uri = f"{features_root.rstrip('/')}/{scene_id}/{scene_id}.tif"
    with rasterio.open(uri) as src:
        _check_raster(src, res=_CELL_10, scene=scene_id, what="feature stack")
        if src.count != len(FEATURE_CHANNEL_NAMES):
            raise RuntimeError(f"{scene_id}: feature stack has {src.count} bands")
        window = Window.from_slices(
            (row * 10, row * 10 + _PATCH_PX), (col * 10, col * 10 + _PATCH_PX)
        )
        bands = src.read(window=window).astype(np.float32)
    scaled = _scale_bands(bands, scaler)
    filled = int((~np.isfinite(scaled)).sum())
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled, filled


def _read_mask(index_row: dict) -> np.ndarray:
    """Read the published eligibility window of one patch."""
    row, col = int(index_row["row"]), int(index_row["col"])
    with rasterio.open(resolve_canonical_uri(str(index_row["eligibility_mask"]))) as msk:
        _check_raster(msk, res=_CELL_100, scene=str(index_row["scene_id"]), what="eligibility mask")
        window = Window.from_slices((row, row + _PATCH_CELLS), (col, col + _PATCH_CELLS))
        mask = msk.read(1, window=window)
    if int(mask.max(initial=0)) > 1:
        raise RuntimeError(f"{index_row['patch_id']}: eligibility mask has values outside {{0, 1}}")
    return mask == 1


def _read_target(index_row: dict, landsat: dict[str, tuple[str, str]]) -> np.ndarray:
    """Read the native Landsat LST window of one patch."""
    scene_id = str(index_row["scene_id"])
    uris = landsat.get(scene_id)
    if uris is None:
        raise RuntimeError(f"{scene_id}: no done Landsat ledger row")
    row, col = int(index_row["row"]), int(index_row["col"])
    with rasterio.open(uris[0]) as src:
        _check_raster(src, res=_CELL_100, scene=scene_id, what="Landsat COG")
        offset_c, offset_r = _canon_offset(src.transform, _CELL_100)
        window = Window.from_slices(
            (row - offset_r, row - offset_r + _PATCH_CELLS),
            (col - offset_c, col - offset_c + _PATCH_CELLS),
        )
        return src.read(1, window=window).astype(np.float32)


def _support_count(mask_100: np.ndarray) -> int:
    """Number of fully valid 7x7 SSIM windows inside the patch."""
    span = _PATCH_CELLS - _SSIM_WINDOW + 1
    return int(
        sum(
            bool(mask_100[i : i + _SSIM_WINDOW, j : j + _SSIM_WINDOW].all())
            for i in range(span)
            for j in range(span)
        )
    )


def _recompute_patch(
    index_row: dict,
    landsat: dict[str, tuple[str, str]],
    grids: dict[str, tuple[dict, dict]],
    scaler: dict,
    features_root: str,
) -> tuple[dict | None, str | None]:
    """Rebuild target/mask/prior/features of one patch, or return a reason."""
    scene_id = str(index_row["scene_id"])
    uris = landsat.get(scene_id)
    if uris is None:
        return None, "landsat_unresolved"
    if scene_id not in grids:
        grids[scene_id] = _block_grid(uris[0], uris[1])
    grid, _ = grids[scene_id]
    row, col = int(index_row["row"]), int(index_row["col"])
    prior = _prior_cells(grid, row, col)
    if prior is None:
        return None, "prior_unavailable"
    mask = _read_mask(index_row)
    target = _read_target(index_row, landsat)
    features, filled = _read_features(scene_id, row, col, features_root, scaler)
    return {
        "target": target,
        "mask": mask,
        "prior": prior,
        "features": features,
        "filled": filled,
    }, None


def _error_sum(
    prediction_100m: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[int, float]:
    selected = np.abs(prediction_100m.astype(np.float64) - target.astype(np.float64))[mask]
    return int(selected.size), float(selected.sum())


# ── train selection audit ─────────────────────────────────────────────


def _rows_digest(keys: list[tuple[str, int, int, int]]) -> str:
    """Mirror of ``random_forest.train_rows_digest``: sorted tab-separated lines."""
    ordered = sorted(keys)
    payload = "".join(f"{scene}\t{year}\t{row}\t{col}\n" for scene, year, row, col in ordered)
    return sha256(payload.encode("utf-8")).hexdigest()


def _check_train_rows(
    report: dict, report_dir: Path, index_rows: list[dict], errors: list[str]
) -> int:
    """Verify the selected row identifiers against the cap and the index."""
    pool = report["train_pool"]
    rows_uri = str(report_dir / str(pool["rows_file"]))
    blob = read_bytes(rows_uri)
    if sha256(blob).hexdigest() != str(pool["rows_sha256"]):
        errors.append("train rows file SHA-256 does not match the report")
        return 0
    table = pq.read_table(io.BytesIO(blob))
    keys = [
        (str(r["scene_id"]), int(r["year"]), int(r["row"]), int(r["col"]))
        for r in table.to_pylist()
    ]
    selected = int(pool["selected_cells"])
    if selected <= 0:
        errors.append("train selection must contain at least one cell")
    accounted = int(pool["read_patches"]) + sum(int(v) for v in pool["exclusions"].values())
    if accounted != int(pool["requested_patches"]):
        errors.append("train requested/read/exclusion accounting does not close")
    max_cells = int(pool["max_cells"])
    candidate = int(pool["candidate_cells"])
    duplicates = int(pool["duplicate_cells"])
    if len(keys) != selected:
        errors.append(f"train rows file holds {len(keys)} keys, report says {selected}")
    if selected > max_cells:
        errors.append(f"selected {selected} exceeds the reservoir cap {max_cells}")
    if selected != min(candidate - duplicates, max_cells):
        errors.append(
            f"selected {selected} != min(candidates {candidate} - duplicates "
            f"{duplicates}, cap {max_cells})"
        )
    if len(set(keys)) != len(keys):
        errors.append(f"train rows file holds {len(keys) - len(set(keys))} duplicate key(s)")
    if _rows_digest(keys) != str(pool["selection_sha256"]):
        errors.append("train selection digest does not match the report")

    anchors: dict[str, set[tuple[int, int]]] = {}
    scene_year: dict[str, int] = {}
    for row in index_rows:
        if str(row["split"]) != "train":
            continue
        scene_id = str(row["scene_id"])
        anchors.setdefault(scene_id, set()).add((int(row["row"]), int(row["col"])))
        scene_year[scene_id] = int(row["year"])
    bad_year = 0
    bad_window = 0
    for scene_id, year, row, col in keys:
        if year not in _TRAIN_YEARS:
            bad_year += 1
        anchor = (row - row % _STRIDE_CELLS, col - col % _STRIDE_CELLS)
        if anchor not in anchors.get(scene_id, set()):
            bad_window += 1
        elif scene_year.get(scene_id) != year:
            bad_year += 1
    if bad_year:
        errors.append(f"{bad_year} selected row(s) outside the 2017-2023 train years")
    if bad_window:
        errors.append(f"{bad_window} selected row(s) not inside a published train window")
    print(
        f"  train selection: {selected} rows (cap {max_cells}), {candidate} candidate cells, "
        f"{duplicates} duplicate; anchors/unique/year checks "
        f"{'OK' if not (bad_year or bad_window) else 'FAILED'}"
    )
    return len(keys)


# ── naive-universe comparison ─────────────────────────────────────────


def _ordered_subsequence_gaps(requested_ids: list[str], scored_ids: list[str]) -> list[str] | None:
    """Return the requested IDs missing from ``scored_ids``, or ``None``.

    ``scored_ids`` must be a unique, in-order subsequence of ``requested_ids``.
    """
    if len(set(scored_ids)) != len(scored_ids):
        return None
    remaining = iter(requested_ids)
    for pid in scored_ids:
        if not any(candidate == pid for candidate in remaining):
            return None
    present = set(scored_ids)
    return [pid for pid in requested_ids if pid not in present]


def _split_index_ids(index_rows: list[dict], split: str) -> list[str]:
    return [str(r["patch_id"]) for r in index_rows if str(r["split"]) == split]


def _compare_naive_split(
    split: str,
    requested: int,
    rf_summary: dict,
    rf_records: list[dict],
    index_ids: list[str],
    naive: dict,
    naive_by_split: dict[str, list[dict]],
    errors: list[str],
) -> None:
    """Compare one split's scored IDs/cells/fill against the naive evidence."""
    naive_summary = naive.get("splits", {}).get(split)
    naive_records = naive_by_split.get(split, [])
    if naive_summary is None or not naive_records:
        if naive.get("_require_full"):
            errors.append(f"{split}: full naive evidence is required")
        print(f"  naive {split:<11}: no evidence for this split; IDs/cells not cross-checked")
        return
    window = index_ids[:requested]
    window_set = set(window)
    rf_ids = [str(r["patch_id"]) for r in rf_records]
    naive_in_window = [
        str(r["patch_id"]) for r in naive_records if str(r["patch_id"]) in window_set
    ]
    naive_by_id = {str(r["patch_id"]): r for r in naive_records}
    covered = int(naive_summary["requested_patches"]) >= requested
    if naive.get("_require_full") and not covered:
        errors.append(f"{split}: naive evidence does not cover the full evaluation")
    if covered:
        if rf_ids != naive_in_window:
            errors.append(
                f"{split}: scored IDs ({len(rf_ids)}) != naive evidence in the requested "
                f"window ({len(naive_in_window)})"
            )
        naive_exclusions = {str(k): int(v) for k, v in naive_summary["exclusions"].items()}
        if int(naive_summary["requested_patches"]) == requested:
            # Identical scope: the whole exclusion budget is comparable.
            rf_exclusions = {str(k): int(v) for k, v in rf_summary["exclusions"].items()}
            if rf_exclusions != naive_exclusions:
                errors.append(
                    f"{split}: exclusion budget {rf_exclusions} != naive {naive_exclusions}"
                )
        else:
            # The naive scope reaches past this window, so only the window's
            # gap count is comparable (per-patch reasons are not recorded).
            rf_gaps = sum(int(v) for v in rf_summary["exclusions"].values())
            window_gaps = len(window) - len(naive_in_window)
            if rf_gaps != window_gaps:
                errors.append(
                    f"{split}: {rf_gaps} exclusion(s) in the window != {window_gaps} "
                    f"proven by the naive evidence (naive split budget {naive_exclusions})"
                )
    else:
        print(
            f"  naive {split:<11}: naive scope {int(naive_summary['requested_patches'])} < "
            f"requested {requested}; overlap-only comparison"
        )
    mismatched = 0
    for record in rf_records:
        pid = str(record["patch_id"])
        naive_record = naive_by_id.get(pid)
        if naive_record is None:
            continue
        if int(naive_record["valid_cells"]) != int(record["valid_cells"]):
            mismatched += 1
        if int(naive_record["filled_feature_pixels"]) != int(record["filled_feature_pixels"]):
            mismatched += 1
    if mismatched:
        errors.append(f"{split}: {mismatched} cell/fill mismatch(es) against the naive evidence")
    compared = sum(1 for r in rf_records if str(r["patch_id"]) in naive_by_id)
    print(
        f"  naive {split:<11}: {compared} scored patch(es) in the naive evidence, "
        f"{mismatched} mismatch(es)"
    )


# ── main ──────────────────────────────────────────────────────────────


def _find_naive_report(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    repo_root = Path(__file__).resolve().parents[2]
    local_smoke = repo_root / "data" / "smoke" / "baseline-real" / "baseline_report.json"
    candidates = sorted(
        (repo_root / "docs" / "results").glob("baseline-full-*/baseline_report.json")
    )
    if candidates:
        return str(candidates[-1])
    return str(local_smoke) if local_smoke.is_file() else None


def _native_inputs(features: np.ndarray, prior: np.ndarray, stage: int) -> np.ndarray:
    """Build native pixel rows independently, with the physical prior last."""
    if features.shape != (28, _PATCH_PX, _PATCH_PX):
        raise ValueError("native feature shape must be (28, 160, 160)")
    if prior.shape != (_PATCH_CELLS, _PATCH_CELLS):
        raise ValueError("physical prior shape must be (16, 16)")
    selected = (
        features[list(_STAGE_CHANNEL_INDICES[stage])]
        .reshape(len(_STAGE_CHANNEL_INDICES[stage]), -1)
        .T
    )
    expanded = np.repeat(np.repeat(prior, _POOL, axis=0), _POOL, axis=1)
    return np.ascontiguousarray(np.column_stack((selected, expanded.ravel())), dtype=np.float32)


def _prediction_100m(prior: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Add the native residual before independently pooling exact 10x10 blocks."""
    if residual.shape != (_PATCH_PX, _PATCH_PX):
        raise ValueError("native residual shape must be (160, 160)")
    expanded = np.repeat(np.repeat(prior, _POOL, axis=0), _POOL, axis=1)
    native = expanded.astype(np.float32) + residual.astype(np.float32)
    return _pool_10m_cells_to_100m(native[None])[0]


def _check_protocol(report: dict, index_rows: list[dict], errors: list[str]) -> None:
    """Reject weakened full runs and split/year drift before model loading."""
    scope = report.get("scope", {})
    profile = scope.get("profile")
    if profile not in ("full", "smoke"):
        errors.append("scope.profile must be full or smoke")
    for row in index_rows:
        split = str(row["split"])
        year = int(row["year"])
        if split == "train":
            valid = year in _TRAIN_YEARS
        else:
            valid = split in _SPLIT_YEARS and year == _SPLIT_YEARS.get(split)
        if not valid:
            errors.append(f"published index has invalid split/year: {split}/{year}")
            break
    forest = report.get("forest", {})
    frozen = {
        "min_samples_leaf": 5,
        "max_features": 1.0,
        "criterion": "squared_error",
        "bootstrap": True,
        "random_state": 0,
    }
    if profile == "full":
        frozen.update({"n_estimators": 200, "max_depth": 20})
    for name, expected in frozen.items():
        if forest.get(name) != expected:
            errors.append(f"forest.{name} must be {expected!r}")
    if forest.get("n_jobs") not in (1, 2):
        errors.append("forest.n_jobs must be bounded to 1 or 2")
    pool = report["train_pool"]
    if pool.get("seed") != 0:
        errors.append("train_pool.seed must be 0")
    if profile != "full":
        return
    if pool.get("max_cells") != 500_000:
        errors.append("full train_pool.max_cells must be 500000")
    if any(scope.get(key) is not None for key in ("max_train_patches", "max_patches_per_split")):
        errors.append("full run must not bound train or evaluation patches")
    train_count = sum(str(row["split"]) == "train" for row in index_rows)
    if pool.get("requested_patches") != train_count or scope.get("train_patches") != train_count:
        errors.append("full run must request every published train patch")
    stages = report.get("stages", {})
    for stage, payload in stages.items():
        if set(payload.get("splits", {})) != set(_SPLIT_YEARS):
            errors.append(f"full stage {stage} must evaluate validation and test")
        for split in _SPLIT_YEARS:
            count = len(_split_index_ids(index_rows, split))
            if payload.get("splits", {}).get(split, {}).get("requested_patches") != count:
                errors.append(f"full stage {stage} {split}: incomplete requested coverage")
            if scope.get("eval_splits", {}).get(split) != count:
                errors.append(f"full scope {split}: incomplete requested coverage")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a random forest baseline report.")
    parser.add_argument("--report", required=True, help="path to random_forest_report.json")
    parser.add_argument("--max-patches", type=int, default=5, help="patches to recompute per split")
    parser.add_argument("--naive-report", default=None, help="override the naive evidence path")
    parser.add_argument(
        "--skip-naive", action="store_true", help="skip the naive universe comparison"
    )
    args = parser.parse_args()

    errors: list[str] = []
    report_path = Path(args.report)
    report_dir = report_path.parent
    report = _read_json(str(report_path))
    print(f"Validating random forest report: {args.report}")

    if int(report["train_pool"]["max_cells"]) < 1:  # cheap shape probe
        errors.append("train_pool.max_cells is not positive")

    # ── provenance ────────────────────────────────────────────────────
    source = report["source"]
    index_root = resolve_canonical_uri(str(source["patch_index_root"]))
    index_rows, index_by_id = _load_index(index_root)
    _check_protocol(report, index_rows, errors)
    marker = _read_json(f"{index_root.rstrip('/')}/complete.json")
    for key in ("index_policy_hash", "source_policy_hash", "patches_total"):
        if str(report["fingerprints"].get(key, "")) != str(marker.get(key, "")):
            errors.append(
                f"fingerprint {key}: report {report['fingerprints'].get(key)!r} != "
                f"marker {marker.get(key)!r}"
            )
    if int(report["fingerprints"].get("patches_total") or -1) != len(index_rows):
        errors.append(
            f"marker patches_total {report['fingerprints'].get('patches_total')} != "
            f"{len(index_rows)} index rows"
        )
    if int(marker.get("patches_total") or -1) != len(index_rows):
        errors.append("marker patches_total disagrees with the index row count")
    print(f"  provenance: {len(index_rows)} indexed patches, fingerprints compared to the marker")

    # ── library versions ──────────────────────────────────────────────
    recorded = report.get("library_versions", {})
    if str(recorded.get("scikit-learn", "")) != sklearn.__version__:
        errors.append(
            f"scikit-learn {sklearn.__version__} != recorded "
            f"{recorded.get('scikit-learn')!r}: replay identity cannot be guaranteed"
        )
    if str(recorded.get("numpy", "")) != np.__version__:
        print(
            f"  note: numpy {np.__version__} != recorded {recorded.get('numpy')!r} (informational)"
        )

    # ── train selection ───────────────────────────────────────────────
    _check_train_rows(report, report_dir, index_rows, errors)

    # ── per-stage structure and aggregates ────────────────────────────
    stages = report.get("stages", {})
    if sorted(int(s) for s in stages) != sorted(_STAGE_CHANNEL_INDICES):
        errors.append(f"expected stages {sorted(_STAGE_CHANNEL_INDICES)}, got {sorted(stages)}")
    stage_ids: dict[int, dict[str, list[str]]] = {}
    stage_cells: dict[int, dict[str, dict[str, int]]] = {}
    for stage_key in sorted(stages, key=int):
        stage_id = int(stage_key)
        stage = stages[stage_key]
        expected_indices = _STAGE_CHANNEL_INDICES.get(stage_id)
        if expected_indices is None:
            errors.append(f"stage {stage_id}: not part of the frozen Tag-11 ladder")
            continue
        expected_names = [FEATURE_CHANNEL_NAMES[i] for i in expected_indices]
        if list(stage["channels"]) != expected_names:
            errors.append(f"stage {stage_id}: channels do not match the frozen selection")
        if int(stage["n_active_channels"]) != len(expected_indices):
            errors.append(f"stage {stage_id}: n_active_channels mismatch")
        if str(stage["feature_order"]) != _EXPECTED_FEATURE_ORDER[stage_id]:
            errors.append(f"stage {stage_id}: feature_order mismatch")
        model_path = report_dir / str(stage["model_file"])
        if not model_path.is_file():
            errors.append(f"stage {stage_id}: model artifact missing ({model_path})")
        else:
            digest = sha256(model_path.read_bytes()).hexdigest()
            if digest != str(stage["model_sha256"]):
                errors.append(f"stage {stage_id}: model SHA-256 does not match the report")
        if not bool(stage.get("replay", {}).get("identical")):
            errors.append(f"stage {stage_id}: serialized replay is not marked identical")

        records = list(stage.get("patches", []))
        by_split: dict[str, list[dict]] = {}
        for record in records:
            by_split.setdefault(str(record["split"]), []).append(record)
        stage_ids[stage_id] = {
            split: [str(r["patch_id"]) for r in split_records]
            for split, split_records in by_split.items()
        }
        stage_cells[stage_id] = {
            split: {str(r["patch_id"]): int(r["valid_cells"]) for r in split_records}
            for split, split_records in by_split.items()
        }
        for split, summary in sorted(stage["splits"].items()):
            split_records = by_split.get(split, [])
            if int(summary["evaluated_patches"]) != len(split_records):
                errors.append(f"stage {stage_id} {split}: evaluated_patches != record count")
            cells = sum(int(r["valid_cells"]) for r in split_records)
            total = sum(float(r["abs_error_sum"]) for r in split_records)
            windows = sum(int(r["ssim_windows"]) for r in split_records)
            filled = sum(int(r["filled_feature_pixels"]) for r in split_records)
            if cells != int(summary["valid_cells"]):
                errors.append(f"stage {stage_id} {split}: cells != record sum")
            if not np.isclose(
                total, float(summary["abs_error_sum"]), rtol=_SUM_RTOL, atol=_SUM_ATOL
            ):
                errors.append(f"stage {stage_id} {split}: error sum != record sum")
            if windows != int(summary["ssim_windows"]):
                errors.append(f"stage {stage_id} {split}: ssim windows != record sum")
            if filled != int(summary["filled_feature_pixels"]):
                errors.append(f"stage {stage_id} {split}: filled pixels != record sum")
            if summary["mae"] is not None and cells > 0:
                if not np.isclose(float(summary["mae"]), total / cells, rtol=1e-9, atol=1e-12):
                    errors.append(f"stage {stage_id} {split}: mae != error_sum/cells")
            if summary["ssim"] is not None and windows > 0:
                if not 0.0 <= float(summary["ssim"]) <= 1.0:
                    errors.append(f"stage {stage_id} {split}: ssim outside [0, 1]")
            accounted = int(summary["evaluated_patches"]) + sum(
                int(v) for v in summary["exclusions"].values()
            )
            if accounted != int(summary["requested_patches"]):
                errors.append(
                    f"stage {stage_id} {split}: accounted {accounted} != "
                    f"requested {summary['requested_patches']}"
                )

    # ── coverage, split separation, cross-stage identity ──────────────
    reference_stage = min(stage_ids, default=None)
    splits = sorted({split for per_split in stage_ids.values() for split in per_split})
    for split in splits:
        if split not in _SPLIT_YEARS:
            errors.append(f"{split}: not an eval split of the frozen contract")
            continue
        index_ids = _split_index_ids(index_rows, split)
        for stage_id in sorted(stage_ids):
            summary = stages[str(stage_id)]["splits"].get(split)
            if summary is None:
                errors.append(f"stage {stage_id}: no summary for split {split}")
                continue
            requested = int(summary["requested_patches"])
            scored = stage_ids[stage_id].get(split, [])
            if requested > len(index_ids):
                errors.append(f"stage {stage_id} {split}: requested exceeds published rows")
                continue
            window = index_ids[:requested]
            gaps = _ordered_subsequence_gaps(window, scored)
            if gaps is None:
                errors.append(
                    f"stage {stage_id} {split}: scored IDs are not a unique, in-order "
                    f"subsequence of the first {requested} indexed rows"
                )
            for record in [r for r in stages[str(stage_id)]["patches"] if str(r["split"]) == split]:
                index_row = index_by_id.get(str(record["patch_id"]))
                if index_row is None:
                    errors.append(f"{record['patch_id']}: not in the published index")
                    continue
                if int(index_row["year"]) != _SPLIT_YEARS[split]:
                    errors.append(
                        f"{record['patch_id']}: year {index_row['year']} outside the "
                        f"{split} contract"
                    )
                if int(record["valid_cells"]) != int(index_row["n_eligible"]):
                    errors.append(
                        f"{record['patch_id']}: valid cells {record['valid_cells']} != "
                        f"index n_eligible {index_row['n_eligible']}"
                    )
        if reference_stage is not None:
            reference_ids = stage_ids[reference_stage].get(split, [])
            reference_cells = stage_cells[reference_stage].get(split, {})
            for stage_id in sorted(stage_ids):
                if stage_id == reference_stage:
                    continue
                if stage_ids[stage_id].get(split, []) != reference_ids:
                    errors.append(f"stage {stage_id} {split}: scored IDs differ across stages")
                if stage_cells[stage_id].get(split, {}) != reference_cells:
                    errors.append(f"stage {stage_id} {split}: cell counts differ across stages")
    for split in splits:
        if reference_stage is None:
            continue
        summary = stages[str(reference_stage)]["splits"].get(split)
        if summary is None:
            continue
        scored = stage_ids[reference_stage].get(split, [])
        print(
            f"  coverage {split:<11}: requested {int(summary['requested_patches'])} | "
            f"scored {len(scored)} | exclusions {summary['exclusions']}"
        )

    # ── naive universe comparison ─────────────────────────────────────
    full = report.get("scope", {}).get("profile") == "full"
    if args.skip_naive and full:
        errors.append("full validation cannot skip the naive evidence comparison")
    if not args.skip_naive:
        naive_path = _find_naive_report(args.naive_report)
        if naive_path is None:
            errors.append("no naive baseline evidence found; pass --naive-report or --skip-naive")
        else:
            naive = _read_json(naive_path)
            naive["_require_full"] = full
            print(f"  naive evidence: {naive_path}")
            for key in ("index_policy_hash", "source_policy_hash", "patches_total"):
                if str(naive.get("fingerprints", {}).get(key, "")) != str(
                    report["fingerprints"].get(key, "")
                ):
                    errors.append(f"naive evidence fingerprint {key} differs from this report")
            naive_by_split: dict[str, list[dict]] = {}
            for record in naive.get("patches", []):
                naive_by_split.setdefault(str(record["split"]), []).append(record)
            for split in splits:
                if reference_stage is None:
                    continue
                summary = stages[str(reference_stage)]["splits"].get(split)
                records = [
                    r for r in stages[str(reference_stage)]["patches"] if str(r["split"]) == split
                ]
                if summary is None:
                    continue
                _compare_naive_split(
                    split,
                    int(summary["requested_patches"]),
                    summary,
                    records,
                    _split_index_ids(index_rows, split),
                    naive,
                    naive_by_split,
                    errors,
                )

    if errors:
        for error in errors[:_MAX_REPORTED]:
            print(f"  ✗ {error}")
        print(f"FAIL: {len(errors)} finding(s); model replay refused")
        return 1

    # ── numeric recomputation on a bounded patch sample ───────────────
    landsat = _load_landsat(resolve_canonical_uri(str(source["ard_root"])))
    scaler = _load_scaler(str(source["training_root"]))
    features_root = str(source["features_root"])
    grids: dict[str, tuple[dict, dict]] = {}
    models = {}
    for stage_key in sorted(stages, key=int):
        model_path = report_dir / str(stages[stage_key]["model_file"])
        if model_path.is_file():
            expected_hash = str(stages[stage_key]["model_sha256"])
            if sha256(model_path.read_bytes()).hexdigest() != expected_hash:
                continue
            model = load_model(str(model_path))
            stage_id = int(stage_key)
            if stage_id not in _STAGE_CHANNEL_INDICES:
                continue
            if getattr(model, "n_features_in_", None) != len(_STAGE_CHANNEL_INDICES[stage_id]) + 1:
                errors.append(f"stage {stage_id}: model must consume C+1 features")
                continue
            if model_path.suffix != ".skops":
                errors.append(f"stage {stage_id}: model artifact must use safe .skops persistence")
                continue
            actual = model.get_params()
            if any(actual.get(key) != value for key, value in report["forest"].items()):
                errors.append(f"stage {stage_id}: model hyperparameters differ from report")
                continue
            models[stage_id] = model
    checked = 0
    for split in splits:
        if reference_stage is None:
            continue
        records = [r for r in stages[str(reference_stage)]["patches"] if str(r["split"]) == split]
        for record in records[: max(1, args.max_patches)]:
            index_row = index_by_id.get(str(record["patch_id"]))
            if index_row is None:
                errors.append(f"{record['patch_id']}: not in the published index")
                continue
            rebuilt, reason = _recompute_patch(index_row, landsat, grids, scaler, features_root)
            if rebuilt is None:
                errors.append(f"{record['patch_id']}: recomputation unavailable ({reason})")
                continue
            mask_cells = int(rebuilt["mask"].sum())
            if mask_cells != int(record["valid_cells"]):
                errors.append(
                    f"{record['patch_id']}: mask cells {mask_cells} != artifact "
                    f"{record['valid_cells']}"
                )
            if int(rebuilt["filled"]) != int(record["filled_feature_pixels"]):
                errors.append(
                    f"{record['patch_id']}: filled pixels {rebuilt['filled']} != artifact "
                    f"{record['filled_feature_pixels']}"
                )
            if _support_count(rebuilt["mask"]) != int(record["ssim_windows"]):
                errors.append(f"{record['patch_id']}: SSIM support != artifact ssim_windows")
            for stage_id, model in sorted(models.items()):
                cells = _native_inputs(rebuilt["features"], rebuilt["prior"], stage_id)
                residual = np.asarray(model.predict(cells), dtype=np.float32).reshape(
                    _PATCH_PX, _PATCH_PX
                )
                prediction = _prediction_100m(rebuilt["prior"], residual)
                cells_count, total = _error_sum(prediction, rebuilt["target"], rebuilt["mask"])
                artifact = next(
                    r
                    for r in stages[str(stage_id)]["patches"]
                    if str(r["patch_id"]) == str(record["patch_id"])
                )
                if cells_count != int(artifact["valid_cells"]):
                    errors.append(
                        f"{record['patch_id']} stage {stage_id}: recomputed cells "
                        f"{cells_count} != artifact {artifact['valid_cells']}"
                    )
                elif not np.isclose(
                    total, float(artifact["abs_error_sum"]), rtol=_SUM_RTOL, atol=_SUM_ATOL
                ):
                    errors.append(
                        f"{record['patch_id']} stage {stage_id}: recomputed error "
                        f"{total:.6f} != artifact {artifact['abs_error_sum']:.6f}"
                    )
            checked += 1
            print(f"  recomputed {record['patch_id']} [{split}] cells={mask_cells}")

    # ── probe fixture: rebuild comparison and bitwise replay ──────────
    probes = report.get("probes", {})
    probe_checked = 0
    if not probes:
        errors.append("probe fixture is required")
    if probes:
        probe_uri = str(report_dir / str(probes["file"]))
        probe_blob = read_bytes(probe_uri)
        if sha256(probe_blob).hexdigest() != str(probes["sha256"]):
            errors.append("probe file SHA-256 does not match the report")
            print("FAIL: probe hash mismatch; replay refused")
            return 1
        probe = np.load(io.BytesIO(probe_blob), allow_pickle=False)
        n_probe = len(probe["patch_id"])
        if probe["features"].shape != (n_probe, 28, _PATCH_PX, _PATCH_PX):
            print("FAIL: probe features must retain the native 10 m grid")
            return 1
        for stage in stages:
            if probe[f"residual_stage{stage}"].shape != (n_probe, _PATCH_PX, _PATCH_PX):
                print(f"FAIL: stage {stage} probe residual must be native 10 m")
                return 1
        entries = [
            {"patch_id": str(pid), "split": str(split)}
            for pid, split in zip(probe["patch_id"], probe["split"], strict=True)
        ]
        if entries != probes.get("patches") or len({e["patch_id"] for e in entries}) != n_probe:
            errors.append("probe identities must match the report and be unique")
        if set(str(split) for split in probe["split"]) != set(splits):
            errors.append("probes must cover every evaluation split")
        if n_probe != len(probes.get("patches", [])):
            errors.append("probe file patch count != report entries")
        for k in range(n_probe):
            pid = str(probe["patch_id"][k])
            split = str(probe["split"][k])
            index_row = index_by_id.get(pid)
            if index_row is not None and str(index_row["split"]) != split:
                errors.append(f"{pid}: probe split differs from the published index")
                continue
            if index_row is None:
                errors.append(f"{pid}: probe patch not in the published index")
                continue
            rebuilt, reason = _recompute_patch(index_row, landsat, grids, scaler, features_root)
            if rebuilt is None:
                errors.append(f"{pid}: probe recomputation unavailable ({reason})")
                continue
            if not np.array_equal(probe["mask_100m"][k], rebuilt["mask"]):
                errors.append(f"{pid}: probe mask differs from the rebuilt mask")
            if not np.allclose(probe["target_100m"][k], rebuilt["target"], rtol=1e-6, atol=1e-6):
                errors.append(f"{pid}: probe target differs from the rebuilt target")
            if not np.allclose(probe["prior_100m"][k], rebuilt["prior"], rtol=1e-6, atol=1e-3):
                errors.append(f"{pid}: probe prior differs from the rebuilt prior")
            max_diff = float(np.max(np.abs(probe["features"][k] - rebuilt["features"])))
            if max_diff > _FEATURE_ATOL:
                errors.append(
                    f"{pid}: probe features differ from the rebuilt features "
                    f"(max abs {max_diff:.3g})"
                )
            if int(probe["filled_feature_pixels"][k]) != int(rebuilt["filled"]):
                errors.append(f"{pid}: probe filled pixels != rebuilt count")
            for stage_id, model in sorted(models.items()):
                cells = _native_inputs(probe["features"][k], probe["prior_100m"][k], stage_id)
                residual = np.asarray(model.predict(cells), dtype=np.float32).reshape(
                    _PATCH_PX, _PATCH_PX
                )
                stored = np.asarray(probe[f"residual_stage{stage_id}"][k], dtype=np.float32)
                if not np.array_equal(residual, stored):
                    errors.append(
                        f"{pid} stage {stage_id}: replay differs from the stored residual"
                    )
                artifact = next(
                    (
                        r
                        for r in stages[str(stage_id)]["patches"]
                        if str(r["patch_id"]) == pid and str(r["split"]) == split
                    ),
                    None,
                )
                if artifact is None:
                    errors.append(f"{pid} stage {stage_id}: no report record for the probe")
                    continue
                prediction = _prediction_100m(probe["prior_100m"][k], stored)
                _, total = _error_sum(prediction, rebuilt["target"], rebuilt["mask"])
                if not np.isclose(
                    total, float(artifact["abs_error_sum"]), rtol=_SUM_RTOL, atol=_SUM_ATOL
                ):
                    errors.append(
                        f"{pid} stage {stage_id}: probe error sum {total:.6f} != artifact "
                        f"{artifact['abs_error_sum']:.6f}"
                    )
            probe_checked += 1
            print(f"  probe {pid} [{split}] rebuilt and replayed per stage")

    for error in errors[:_MAX_REPORTED]:
        print(f"  ✗ {error}")
    if len(errors) > _MAX_REPORTED:
        print(f"  ✗ ... and {len(errors) - _MAX_REPORTED} more finding(s)")
    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    print(
        f"OK: random forest report valid ({checked} patches recomputed, "
        f"{probe_checked} probe(s) replayed, "
        f"{int(report['train_pool']['selected_cells'])} train rows audited)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
