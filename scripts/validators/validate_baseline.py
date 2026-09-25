# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for the naive prior-expand baseline artifact.

Read-only probe over a written ``baseline_report.json`` and the published
sources. It re-derives, with its own code, what the baseline claims:

- the selection: the report's first N patch IDs per split equal the first N
  indexed rows of that split, in the index's deterministic order;
- per patch (bounded sample): the 1000 m block means recomputed from
  native-valid Landsat pixels as a whole-COG accumulation (a different method
  from the reader's stored-grid lookup), the native target and the
  ``training_eligible@100m`` window, then the masked absolute-error sum and
  valid-cell count with plain NumPy;
- the SSIM support count: the number of 7x7 windows of the patch's 100 m mask
  that are entirely valid. The SSIM value itself is not recomputed here, since
  it comes from the shared TorchMetrics path; only its support is checked;
- internal consistency: split aggregates equal the sum of their per-patch
  records, MAE equals numerator/count, and evaluated + exclusions == requested.

Mirrored constants keep the geometry independent of the implementation. The
validator never writes and never touches a canonical artifact.

Usage
-----
    uv run python scripts/validators/validate_baseline.py \
        --report data/smoke/baseline-real/baseline_report.json \
        --max-patches 12
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from berlin_lst_downscaling.data.io import read_bytes, resolve_canonical_uri

_CANON_X = 369190.0
_CANON_Y = 5838410.0
_CELL_100 = 100.0
_PATCH_CELLS = 16
_PATCH_PX = 160
_BLOCK_CELLS = 10  # 1000 m = 10 x 100 m cells
_BLOCK_PX = 100  # 1000 m = 100 x 10 m pixels
_LST_MIN = 150.0
_LST_MAX = 400.0
_SSIM_WINDOW = 7  # must match modeling/metrics.py

# The baseline accumulates in float32 (torch) while this validator accumulates
# in float64, so an error sum over ~250 cells differs by ~1e-6 relative. The
# tolerance is tight enough that a wrong block, mask, or target — which shifts
# the sum by whole Kelvin — cannot pass.
_SUM_RTOL = 1e-5
_SUM_ATOL = 1e-3


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _canon_offset(transform, res: float) -> tuple[int, int]:
    return (
        round((transform.xoff - _CANON_X) / res),
        round((_CANON_Y - transform.yoff) / res),
    )


# ── published lookups (loaded once) ───────────────────────────────────


def _load_index(index_root: str) -> tuple[list[dict], dict[str, dict]]:
    rows = _read_table(f"{index_root.rstrip('/')}/patch_index.parquet").to_pylist()
    return rows, {str(r["patch_id"]): r for r in rows}


def _load_landsat(ard_root: str) -> dict[str, tuple[str, str]]:
    cols = _read_table(f"{ard_root.rstrip('/')}/ledger.parquet").to_pydict()
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


# ── independent recomputation ─────────────────────────────────────────


def _block_grid(landsat_cog: str, landsat_flag: str) -> tuple[dict, dict]:
    """Whole-COG 1000 m block means over native-valid cells, keyed globally."""
    with rasterio.open(landsat_cog) as src:
        if str(src.crs) != "EPSG:25833":
            raise RuntimeError(f"landsat_cog CRS {src.crs!r} != EPSG:25833")
        if src.transform.b or src.transform.d:
            raise RuntimeError("landsat_cog transform is rotated")
        st = src.read(1).astype(np.float64)
        col0, row0 = _canon_offset(src.transform, _CELL_100)
        shape = st.shape
    with rasterio.open(landsat_flag) as fsrc:
        if (fsrc.height, fsrc.width) != shape:
            raise RuntimeError("landsat flag shape differs from the LST COG")
        flag = fsrc.read(1)

    valid = np.isfinite(st) & (flag == 0) & (st >= _LST_MIN) & (st <= _LST_MAX)
    rows = np.arange(shape[0], dtype=np.int64)[:, None] + row0  # (H, 1)
    cols = np.arange(shape[1], dtype=np.int64)[None, :] + col0  # (1, W)
    brow = rows // _BLOCK_CELLS  # (H, 1)
    bcol = cols // _BLOCK_CELLS  # (1, W)
    brow0, bcol0 = int(brow.min()), int(bcol.min())
    nbr = int(brow.max()) - brow0 + 1
    nbc = int(bcol.max()) - bcol0 + 1
    # Broadcast to the full (H, W) block index grid before flattening.
    flat = (((brow - brow0) * nbc) + (bcol - bcol0)).ravel()
    sel = valid.ravel()
    counts = np.bincount(flat[sel], minlength=nbr * nbc)
    sums = np.bincount(flat[sel], weights=st.ravel()[sel], minlength=nbr * nbc)
    means = np.full(nbr * nbc, np.nan, dtype=np.float64)
    ok = counts > 0
    means[ok] = sums[ok] / counts[ok]
    grid = {"means": means.reshape(nbr, nbc), "brow0": brow0, "bcol0": bcol0}
    return grid, {"col0": col0, "row0": row0}


def _pooled_prior_100m(grid: dict, row: int, col: int) -> np.ndarray | None:
    """Expand the block means and pool back to 100 m (the baseline prediction)."""
    ri = (row * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX - grid["brow0"]
    ci = (col * 10 + np.arange(_PATCH_PX)) // _BLOCK_PX - grid["bcol0"]
    means = grid["means"]
    if (
        ri.min() < 0
        or ci.min() < 0
        or ri.max() >= means.shape[0]
        or ci.max() >= means.shape[1]
    ):
        return None
    values = means[np.ix_(ri, ci)]
    if not np.isfinite(values).all():
        return None
    return values.reshape(_PATCH_CELLS, _BLOCK_CELLS, _PATCH_CELLS, _BLOCK_CELLS).mean(
        axis=(1, 3)
    )


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
) -> tuple[int, float, int] | None:
    """Return ``(valid_cells, abs_error_sum, ssim_support)`` or None if unavailable."""
    scene_id = str(index_row["scene_id"])
    uris = landsat.get(scene_id)
    if uris is None:
        return None
    if scene_id not in grids:
        grids[scene_id] = _block_grid(uris[0], uris[1])
    grid, geometry = grids[scene_id]

    row, col = int(index_row["row"]), int(index_row["col"])
    prediction = _pooled_prior_100m(grid, row, col)
    if prediction is None:
        return None

    with rasterio.open(uris[0]) as src:
        target = src.read(
            1,
            window=Window.from_slices(
                (row - geometry["row0"], row - geometry["row0"] + _PATCH_CELLS),
                (col - geometry["col0"], col - geometry["col0"] + _PATCH_CELLS),
            ),
        ).astype(np.float64)
    mask_uri = resolve_canonical_uri(str(index_row["eligibility_mask"]))
    with rasterio.open(mask_uri) as msk:
        moff_c, moff_r = _canon_offset(msk.transform, _CELL_100)
        mask = (
            msk.read(
                1,
                window=Window.from_slices(
                    (row - moff_r, row - moff_r + _PATCH_CELLS),
                    (col - moff_c, col - moff_c + _PATCH_CELLS),
                ),
            )
            == 1
        )
    selected = np.abs(prediction - target)[mask]
    return int(selected.size), float(selected.sum()), _support_count(mask)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a baseline report.")
    parser.add_argument("--report", required=True, help="path to baseline_report.json")
    parser.add_argument("--patch-index-root", default=None, help="override the index root")
    parser.add_argument(
        "--max-patches", type=int, default=12, help="patches to recompute per split"
    )
    args = parser.parse_args()

    errors: list[str] = []
    report = _read_json(args.report)
    source = report["source"]
    index_root = resolve_canonical_uri(
        args.patch_index_root or str(source["patch_index_root"])
    )
    print(f"Validating baseline report: {args.report}")

    index_rows, index_by_id = _load_index(index_root)
    landsat = _load_landsat(resolve_canonical_uri(str(source["ard_root"])))

    by_split: dict[str, list[dict]] = {}
    for record in report["patches"]:
        by_split.setdefault(str(record["split"]), []).append(record)

    # ── selection fidelity: the report must be the index's prefix per split ──
    for split, split_records in sorted(by_split.items()):
        expected = [str(r["patch_id"]) for r in index_rows if str(r["split"]) == split]
        got = [str(r["patch_id"]) for r in split_records]
        if expected[: len(got)] != got:
            errors.append(
                f"{split}: report patch IDs are not the first {len(got)} indexed rows "
                f"of that split"
            )

    # ── per-patch recomputation over a bounded sample ─────────────────
    grids: dict[str, tuple[dict, dict]] = {}
    checked = 0
    for split in sorted(by_split):
        for record in by_split[split][: max(1, args.max_patches)]:
            patch_id = str(record["patch_id"])
            index_row = index_by_id.get(patch_id)
            if index_row is None:
                errors.append(f"{patch_id}: not present in the published index")
                continue
            recomputed = _recompute_patch(index_row, landsat, grids)
            if recomputed is None:
                errors.append(f"{patch_id}: recomputed prediction unavailable")
                continue
            cells, total, support = recomputed
            if cells != int(record["valid_cells"]):
                errors.append(
                    f"{patch_id}: valid cells {cells} != artifact {record['valid_cells']}"
                )
            elif not np.isclose(
                total, float(record["abs_error_sum"]), rtol=_SUM_RTOL, atol=_SUM_ATOL
            ):
                errors.append(
                    f"{patch_id}: abs error sum {total:.6f} != artifact "
                    f"{record['abs_error_sum']:.6f}"
                )
            if support != int(record["ssim_windows"]):
                errors.append(
                    f"{patch_id}: SSIM support {support} != artifact {record['ssim_windows']}"
                )
            checked += 1
            print(
                f"  checked {patch_id} [{split}] cells={cells} err_sum={total:.4f} "
                f"ssim_windows={support}"
            )

    # ── internal consistency of the aggregates ────────────────────────
    for split, summary in sorted(report["splits"].items()):
        split_records = by_split.get(split, [])
        if int(summary["evaluated_patches"]) != len(split_records):
            errors.append(
                f"{split}: evaluated_patches {summary['evaluated_patches']} != "
                f"{len(split_records)} records"
            )
        cells = sum(int(r["valid_cells"]) for r in split_records)
        total = sum(float(r["abs_error_sum"]) for r in split_records)
        windows = sum(int(r["ssim_windows"]) for r in split_records)
        if cells != int(summary["valid_cells"]):
            errors.append(f"{split}: cells {cells} != aggregate {summary['valid_cells']}")
        if not np.isclose(total, float(summary["abs_error_sum"]), rtol=_SUM_RTOL, atol=_SUM_ATOL):
            errors.append(
                f"{split}: error sum {total:.6f} != aggregate {summary['abs_error_sum']}"
            )
        if windows != int(summary["ssim_windows"]):
            errors.append(
                f"{split}: ssim windows {windows} != aggregate {summary['ssim_windows']}"
            )
        if summary["mae"] is not None and cells > 0:
            if not np.isclose(float(summary["mae"]), total / cells, rtol=1e-9, atol=1e-12):
                errors.append(f"{split}: mae {summary['mae']} != error_sum/cells {total / cells}")
        if summary["ssim"] is not None and windows > 0:
            if not 0.0 <= float(summary["ssim"]) <= 1.0:
                errors.append(f"{split}: ssim {summary['ssim']} outside [0, 1]")
        accounted = int(summary["evaluated_patches"]) + sum(summary["exclusions"].values())
        if accounted != int(summary["requested_patches"]):
            errors.append(
                f"{split}: accounted {accounted} != requested {summary['requested_patches']}"
            )

    # ── top-level exclusions must equal the per-split totals ──────────
    per_split: dict[str, int] = {}
    for summary in report["splits"].values():
        for reason, n in summary["exclusions"].items():
            per_split[reason] = per_split.get(reason, 0) + int(n)
    if per_split != {k: int(v) for k, v in report["exclusions"].items()}:
        errors.append(
            f"top-level exclusions {report['exclusions']} != per-split totals {per_split}"
        )

    for e in errors:
        print(f"  ✗ {e}")
    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    print(f"OK: baseline report valid ({checked} patches recomputed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
