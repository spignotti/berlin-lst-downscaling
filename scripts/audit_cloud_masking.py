# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "numpy",
#     "xarray",
#     "pystac>=1.11.0",
#     "pystac-client>=0.9.0",
#     "planetary-computer>=1.0.0",
#     "odc-stac>=0.4.0",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Sampled cloud-masking audit — descriptive comparison, never a policy.

Selects a deterministic, stratified sample of Landsat→S2 pairs from the
canonical manifest and compares **four diagnostic views** of cloud/shadow
handling without applying any of them:

  1. current published ARD mask      — flag fractions from the published
                                       ARD flag COGs (S2 at 10 m; Landsat
                                       at native 100 m, reported separately)
  2. conservative native-QA view     — Landsat low-confidence cloud/shadow/
                                       cirrus + dilated-cloud bits and S2
                                       SCL class 7, reported as *additional
                                       candidate* pixels only
  3. s2cloudless cloud probability   — `S2PixelCloudDetector` threshold
                                       sweep on a supplied B10-capable L1C
                                       source; printed as *unavailable* when
                                       no such source is supplied
  4. optional local overlays         — `--show` renders RGB/mask overlays
                                       for manual residual-cloud inspection;
                                       nothing is saved

The probe prints a per-pair table and aggregate quantiles. It is
explicitly descriptive: no pass/fail decision, no thresholds applied to
production, no mask rewrite, and it never writes a file, GCS object, or
figure.

Native QA bands are loaded on near-native canonical grids — Landsat
QA_PIXEL at 30 m and S2 SCL at 20 m (both exact multiples of the 10 m
canonical grid) — so the probe stays fast over the full Berlin extent;
the published ARD flags are read at their native resolution and, for the
S2 pixel-level comparison, max-pooled to the SCL 20 m grid.

Usage
-----
    uv run python scripts/audit_cloud_masking.py \
        --manifest gs://berlin-lst-data/manifests/v3/<bundle>-r2/manifest.parquet \
        --pairings gs://berlin-lst-data/manifests/v3/<bundle>-r2/pairings.parquet \
        --ledger gs://berlin-lst-data/ard/full/<cutoff>/ledger.parquet \
        --seed 42 --n-pairs 24

    # optional local overlay review (matplotlib, nothing saved):
    uv run --with matplotlib python scripts/audit_cloud_masking.py \
        --manifest ... --ledger ... --seed 42 --show
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from statistics import quantiles

import numpy as np
import pyarrow.parquet as pq

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.acquisition.pc_client import resolve_item_from_href, stac_load
from berlin_lst_downscaling.data.io import read_bytes

_GRID_10M = canon_grid_10m()
_GRID_LS = _GRID_10M.zoom_out(3)  # 30 m — Landsat QA_PIXEL near-native
_GRID_S2 = _GRID_10M.zoom_out(2)  # 20 m — S2 SCL near-native

# SCL classes excluded by the production S2 clear definition (mirrors
# ``data/selection/clear_frac.py`` — fill, saturated, shadow, cloud, cirrus,
# snow). Everything else, including class 7 (unclassified/urban), is clear.
_S2_NOT_CLEAR = {0, 1, 3, 8, 9, 10, 11}

# Landsat QA_PIXEL raw-flag bits for the conservative view (bit indices).
_QA_CLOUD_BIT = 3  # raw cloud flag, any confidence
_QA_DILATED_BIT = 1  # USGS dilated-cloud buffer


@dataclass
class PairDiagnostics:
    """Per-pair cloud-masking diagnostics (all fractions over the full grid)."""

    landsat_scene_id: str
    sentinel2_scene_id: str
    joint_clear_frac: float | None
    anchor_cloud_cover: float | None
    # Landsat (10 m native QA; ARD flag at 100 m reported separately)
    ls_native_clear_frac: float
    ls_extra_candidate_frac: float
    ls_ard_flag_clear_frac: float | None
    # S2 (10 m)
    s2_native_clear_frac: float
    s2_extra_candidate_frac: float
    s2_ard_flag_clear_frac: float | None
    s2_ard_missed_frac: float | None
    s2_ard_extra_shadow_frac: float | None
    s2cloudless_frac: float | None
    s2cloudless_note: str = ""
    load_error: str = ""


def _read_table(uri: str):
    """Read a Parquet table from a local path or GCS URI."""
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _load_manifest(args) -> tuple[dict, dict]:
    """Return ``(pairings, manifest_rows)`` as ``{scene_id: row}`` dicts."""
    manifest = _read_table(args.manifest)
    pairings = _read_table(args.pairings)
    man_cols = manifest.to_pydict()
    manifest_rows: dict[str, dict] = {}
    for i in range(manifest.num_rows):
        manifest_rows[str(man_cols["scene_id"][i])] = {
            "source": str(man_cols["source"][i]),
            "role": str(man_cols["role"][i]),
            "year": int(man_cols["year"][i]),
            "acquisition_datetime": man_cols["acquisition_datetime"][i],
            "cloud_cover": man_cols["cloud_cover"][i],
            "item_href": man_cols["item_href"][i],
        }
    pair_cols = pairings.to_pydict()
    pairs: dict[str, dict] = {}
    for i in range(pairings.num_rows):
        pairs[str(pair_cols["landsat_scene_id"][i])] = {
            "sentinel2_scene_id": str(pair_cols["sentinel2_scene_id"][i]),
            "joint_clear_frac": pair_cols["joint_clear_frac"][i],
            "dt_seconds": pair_cols["dt_seconds"][i],
        }
    return pairs, manifest_rows


def _load_ard_flags(ledger_uri: str) -> dict[str, str]:
    """Return ``{scene_id: path_flag}`` for done ARD scenes."""
    table = _read_table(ledger_uri)
    cols = table.to_pydict()
    flags: dict[str, str] = {}
    for i in range(table.num_rows):
        if cols["status"][i] == "done" and cols["path_flag"][i]:
            flags[str(cols["scene_id"][i])] = str(cols["path_flag"][i])
    return flags


def _eligible_pairs(
    pairs: dict[str, dict],
    manifest_rows: dict[str, dict],
    ard_flags: dict[str, str],
) -> list[tuple[str, str]]:
    """Return ``(anchor_id, s2_id)`` pairs loadable for the audit."""
    eligible: list[tuple[str, str]] = []
    for anchor_id, pair in pairs.items():
        s2_id = pair["sentinel2_scene_id"]
        anchor = manifest_rows.get(anchor_id)
        s2 = manifest_rows.get(s2_id)
        if anchor is None or s2 is None:
            continue
        if not anchor["item_href"] or not s2["item_href"]:
            continue  # native QA needs a resolvable STAC item
        if anchor_id not in ard_flags or s2_id not in ard_flags:
            continue
        eligible.append((anchor_id, s2_id))
    return eligible


def _stratify(pair: tuple[str, str], manifest_rows: dict[str, dict]) -> tuple[str, ...]:
    """Return a deterministic stratum key ``(platform, season, cloud)``."""
    anchor = manifest_rows[pair[0]]
    month = int(anchor["acquisition_datetime"].month)
    season = "early" if month in (5, 6) else "peak" if month == 7 else "late"
    cloud = anchor["cloud_cover"] or 0.0
    cloud_bin = "low" if cloud < 10 else "mid" if cloud < 30 else "high"
    return (str(anchor["source"]), season, cloud_bin)


def _sample_pairs(
    eligible: list[tuple[str, str]],
    manifest_rows: dict[str, dict],
    seed: int,
    n_pairs: int,
) -> list[tuple[str, str]]:
    """Deterministic stratified round-robin sample of ``n_pairs`` pairs.

    Strata are ``(platform, season-bin, cloud-cover-bin)``; each cell is
    shuffled with a cell-seeded RNG and pairs are drawn round-robin so the
    sample spreads across all cells. Unique S2 scene ids are enforced
    (deduplication across anchors), so re-running with the same seed
    yields exactly the same pairs.
    """
    import zlib

    cells: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for pair in eligible:
        cells.setdefault(_stratify(pair, manifest_rows), []).append(pair)
    for key, items in cells.items():
        salt = zlib.crc32("|".join(key).encode("utf-8"))
        rng = np.random.default_rng((seed * 1000003 + salt) % (2**32))
        items.sort(key=lambda p: p[0])
        perm = rng.permutation(len(items))
        items[:] = [items[i] for i in perm]

    sample: list[tuple[str, str]] = []
    seen_s2: set[str] = set()
    cell_order = sorted(cells)
    cursor = 0
    while len(sample) < n_pairs and cursor < max(len(v) for v in cells.values()):
        for key in cell_order:
            items = cells[key]
            if cursor >= len(items):
                continue
            pair = items[cursor]
            if pair[1] in seen_s2:
                continue
            sample.append(pair)
            seen_s2.add(pair[1])
            if len(sample) >= n_pairs:
                break
        cursor += 1
    return sample


def _load_native_qa(item, band: str, grid) -> np.ndarray:
    """Load a single resolved STAC item's native QA band onto *grid*."""
    ds = stac_load(
        items=[item],
        bands=[band],
        geobox=grid,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return ds[band].values[0]


def _with_retry(fn, *, attempts: int = 4, base_delay: float = 8.0):
    """Run *fn* with exponential backoff on transient failures.

    Planetary Computer's SAS token endpoint throttles bursts of signing
    requests with HTTP 429; a short paced retry absorbs that without
    touching the shared acquisition layer.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # transient network / rate-limit errors
            last_exc = exc
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError(f"load failed after {attempts} attempts: {last_exc}") from last_exc


def _landsat_diagnostics(qa: np.ndarray) -> tuple[float, float]:
    """Return ``(native_clear_frac, extra_candidate_frac)`` on the 30 m grid.

    Native clear uses the production definition
    (``data/ard.masking.landsat_qa_to_clear_bits``). Extra candidates are
    pixels the production mask keeps clear that carry a residual QA hint:
    the raw cloud flag (bit 3) at any confidence or the USGS dilated-cloud
    bit (bit 1). Low-confidence bits 8-15 are *not* treated as hints —
    USGS Collection 2 sets them to 01 as the baseline for clear pixels
    (canonical clear-land value 0x5540), so they carry no residual signal.
    """
    qa = qa.astype(np.uint16)
    from berlin_lst_downscaling.data.ard.masking import landsat_qa_to_clear_bits

    clear = landsat_qa_to_clear_bits(qa)
    cloud_hint = ((qa >> _QA_CLOUD_BIT) & 1) == 1
    dilated = ((qa >> _QA_DILATED_BIT) & 1) == 1
    suspect = cloud_hint | dilated
    extra = clear & suspect
    total = max(int(qa.size), 1)
    return float(np.sum(clear)) / total, float(np.sum(extra)) / total


def _scl_array(scl_raw: np.ndarray) -> np.ndarray:
    """Normalise the raw SCL float array to uint8 class values.

    NaN (outside the tile footprint within the Berlin grid) maps to class 0
    (no-data), which the production definition treats as not clear.
    """
    return np.round(np.nan_to_num(scl_raw, nan=0.0)).astype(np.uint8)


def _s2_diagnostics(scl_raw: np.ndarray) -> tuple[float, float]:
    """Return ``(native_clear_frac, extra_candidate_frac)`` for SCL.

    Extra candidates are SCL class 7 (unclassified / low-probability
    cloud) pixels that the production definition treats as clear.
    """
    scl = _scl_array(scl_raw)
    native_clear = ~np.isin(scl, list(_S2_NOT_CLEAR))
    total = max(int(scl.size), 1)
    return float(np.sum(native_clear)) / total, float(np.sum(scl == 7)) / total


def _flag_clear_frac(flag_uri: str) -> float | None:
    """Return the fraction of flag==0 (clear) pixels of a published ARD flag."""
    import rasterio

    try:
        with rasterio.open(flag_uri) as src:
            flag = src.read(1).astype(np.uint8)
    except Exception:
        return None
    return float(np.sum(flag == 0)) / max(int(flag.size), 1)


def _s2_ard_comparison(flag_uri: str, scl_raw: np.ndarray) -> tuple[float | None, float | None]:
    """Return ``(missed_frac, extra_shadow_frac)`` on the SCL 20 m grid.

    The published S2 ARD flag is 10 m; it is max-pooled to the SCL 20 m
    grid (a 20 m pixel counts as clear only if all four 10 m subpixels
    are clear). ``missed_frac``: fraction of ARD-clear pixels the native
    SCL view considers not clear. ``extra_shadow_frac``: fraction of
    ARD-flagged pixels the native SCL view considers clear (projected
    cloud shadow). Both are None when the ARD flag is unreadable.
    """
    import rasterio

    try:
        with rasterio.open(flag_uri) as src:
            flag = src.read(1).astype(np.uint8)
    except Exception:
        return None, None
    scl = _scl_array(scl_raw)
    native_clear = ~np.isin(scl, list(_S2_NOT_CLEAR))

    h, w = native_clear.shape
    clear = flag == 0
    # The canonical 10 m grid has odd width (4699 px), so the flag does not
    # tile exactly into 2×2 blocks; pad the edge column by duplication
    # before max-pooling (a 20 m pixel counts as clear only if all of its
    # four 10 m subpixels are clear).
    ph, pw = 2 * h, 2 * w
    pad_w = pw - clear.shape[1]
    if pad_w > 0:
        clear = np.pad(clear, ((0, 0), (0, pad_w)), mode="edge")
    clear_20m = clear[:ph, :pw].reshape(h, 2, w, 2).all(axis=(1, 3))

    n_ard_clear = max(int(np.sum(clear_20m)), 1)
    n_ard_flagged = max(int(np.sum(~clear_20m)), 1)
    missed = float(np.sum(clear_20m & ~native_clear)) / n_ard_clear
    extra_shadow = float(np.sum(~clear_20m & native_clear)) / n_ard_flagged
    return missed, extra_shadow


def _s2cloudless_proba(
    item,
    bands: list[str],
    l1c_root: str,
) -> tuple[np.ndarray | None, str]:
    """Run s2cloudless on a B10-capable source; return ``(proba, note)``.

    s2cloudless requires the ordered ten-band reflectance tensor
    ``[B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12]`` (scaled
    reflectance, ``DN / 10000``). The Planetary Computer ``sentinel-2-l2a``
    collection does not expose B10 (live catalog probe), so the comparator
    requires an exact-match L1C source per scene supplied via
    ``--l1c-root``; without it the view is reported unavailable.
    """
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError:
        return None, "s2cloudless not installed"

    scene_id = item.id
    b10_uri = f"{l1c_root.rstrip('/')}/{scene_id}/{scene_id}_B10.tif"
    try:
        import rasterio

        with rasterio.open(b10_uri) as src:
            b10 = src.read(1).astype(np.float32)
    except Exception:
        return None, "s2cloudless: unavailable — B10 source not supplied"

    # decision: mirror the historical s2cloudless integration (removed in
    # d505104 after a DN-scaling bug) — load all ten bands on the 10 m grid
    # and divide by 10000 for scaled reflectance; this rerun is evidence
    # gathering, not a production change.
    ds = stac_load(
        items=[item],
        bands=bands,
        geobox=_GRID_10M,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    if b10.shape != ds[bands[0]].values[0].shape:
        return None, "s2cloudless: B10 source shape does not match S2 bands"

    stack = np.stack(
        [ds[b].values[0].astype(np.float32) / 10000.0 for b in bands] + [b10 / 10000.0],
        axis=-1,
    )
    stack[np.isnan(stack)] = 0.0
    detector = S2PixelCloudDetector(threshold=0.4, average_over=1, dilation_size=1)
    proba = detector.get_cloud_probability_maps(stack[np.newaxis, ...])[0]
    return proba, ""


def _show_overlays(pairs: list[PairDiagnostics], manifest_rows: dict) -> None:
    """Render RGB/mask overlays for manual inspection (nothing saved)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(
            "  --show requires matplotlib: run with "
            "`uv run --with matplotlib python scripts/audit_cloud_masking.py --show`"
        )
        print(f"  ({exc})")
        return

    from berlin_lst_downscaling.data.acquisition.pc_client import resolve_item_from_href, stac_load

    print("\n--show: rendering overlays for manual inspection (nothing saved)")
    for diag in pairs[:6]:
        anchor = manifest_rows.get(diag.landsat_scene_id)
        if anchor is None or not anchor["item_href"]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        try:
            item = resolve_item_from_href(anchor["item_href"])
            ds = stac_load(
                items=[item],
                bands=["red", "green", "blue", "qa_pixel"],
                geobox=_GRID_10M,
                chunks={"x": 2048, "y": 2048},
                groupby="solar_day",
            )
            rgb = np.stack(
                [ds[b].values[0].astype(np.float32) / 10000.0 for b in ("red", "green", "blue")],
                axis=-1,
            )
            rgb = np.clip(rgb, 0, 1)
            axes[0].imshow(rgb, interpolation="nearest")
            axes[0].set_title(f"RGB {diag.landsat_scene_id}")
            axes[1].imshow(ds["qa_pixel"].values[0], cmap="viridis", interpolation="nearest")
            axes[1].set_title("QA_PIXEL")
            axes[2].text(
                0.5,
                0.5,
                f"native clear {diag.ls_native_clear_frac:.3f}\n"
                f"extra candidates {diag.ls_extra_candidate_frac:.3f}",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_title("summary")
        except Exception as exc:
            axes[0].set_title(f"load failed: {exc}")
        plt.show()
        plt.close(fig)


def _f(value: float | None) -> float:
    """Return *value* or NaN when None (0.0 is a valid fraction)."""
    return value if value is not None else float("nan")


def _quantile_line(label: str, values: list[float]) -> str:
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return f"{label:<38} n={len(finite):<4} (insufficient samples)"
    qs = quantiles(finite, n=4, method="inclusive")
    return (
        f"{label:<38} min={min(finite):.3f} p50={qs[1]:.3f} p90={qs[2]:.3f} max={max(finite):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sampled cloud-masking audit probe")
    parser.add_argument("--manifest", required=True, help="v3 manifest.parquet URI")
    parser.add_argument("--pairings", required=True, help="v3 pairings.parquet URI")
    parser.add_argument("--ledger", required=True, help="ARD ledger.parquet URI")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the sample")
    parser.add_argument("--n-pairs", type=int, default=24, help="target sample size")
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="pacing delay between scene loads (respects PC SAS rate limits)",
    )
    parser.add_argument("--show", action="store_true", help="render local overlays (no save)")
    parser.add_argument(
        "--l1c-root",
        default=None,
        help="root with per-scene B10 COGs for the s2cloudless comparator",
    )
    args = parser.parse_args()

    try:
        pairs, manifest_rows = _load_manifest(args)
        ard_flags = _load_ard_flags(args.ledger)
    except Exception as exc:
        print(f"Error: cannot read manifest/pairings/ledger: {exc}", file=sys.stderr)
        return 1

    eligible = _eligible_pairs(pairs, manifest_rows, ard_flags)
    if not eligible:
        print("Error: no eligible pairs (both item_hrefs and ARD flags required)", file=sys.stderr)
        return 1

    sample = _sample_pairs(eligible, manifest_rows, args.seed, args.n_pairs)
    if len(sample) < 2:
        print("Error: sample too small", file=sys.stderr)
        return 1

    # 10-band set for the s2cloudless comparator (B10 comes from --l1c-root).
    s2c_bands = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B11", "B12"]

    diagnostics: list[PairDiagnostics] = []
    print(
        f"Cloud-masking audit — {len(sample)} pairs, seed={args.seed}, "
        f"grids=EPSG:25833 30 m (Landsat) / 20 m (S2)"
    )
    print(f"  eligible pairs: {len(eligible)} | sampled: {len(sample)}")
    print()

    for i, (anchor_id, s2_id) in enumerate(sample, 1):
        anchor = manifest_rows[anchor_id]
        s2 = manifest_rows[s2_id]
        pair = pairs[anchor_id]
        diag = PairDiagnostics(
            landsat_scene_id=anchor_id,
            sentinel2_scene_id=s2_id,
            joint_clear_frac=float(pair["joint_clear_frac"]) if pair["joint_clear_frac"] else None,
            anchor_cloud_cover=float(anchor["cloud_cover"]) if anchor["cloud_cover"] else None,
            ls_native_clear_frac=float("nan"),
            ls_extra_candidate_frac=float("nan"),
            ls_ard_flag_clear_frac=None,
            s2_native_clear_frac=float("nan"),
            s2_extra_candidate_frac=float("nan"),
            s2_ard_flag_clear_frac=None,
            s2_ard_missed_frac=None,
            s2_ard_extra_shadow_frac=None,
            s2cloudless_frac=None,
        )
        print(f"  [{i}/{len(sample)}] {anchor_id[:21]} + {s2_id[:21]}", flush=True)
        try:
            ls_item = _with_retry(
                lambda href=anchor["item_href"]: resolve_item_from_href(str(href)),
                base_delay=args.sleep,
            )

            def _ls_load(item=ls_item):
                return _load_native_qa(item, "qa_pixel", _GRID_LS)

            ls_qa = _with_retry(_ls_load, base_delay=args.sleep)
            diag.ls_native_clear_frac, diag.ls_extra_candidate_frac = _landsat_diagnostics(ls_qa)
        except Exception as exc:
            diag.load_error = f"landsat load failed: {exc}"
        try:
            s2_item = _with_retry(
                lambda href=s2["item_href"]: resolve_item_from_href(str(href)),
                base_delay=args.sleep,
            )

            def _s2_load(item=s2_item):
                return _load_native_qa(item, "SCL", _GRID_S2)

            s2_scl = _with_retry(_s2_load, base_delay=args.sleep)
            diag.s2_native_clear_frac, diag.s2_extra_candidate_frac = _s2_diagnostics(s2_scl)
            diag.s2_ard_flag_clear_frac = _flag_clear_frac(ard_flags[s2_id])
            diag.s2_ard_missed_frac, diag.s2_ard_extra_shadow_frac = _s2_ard_comparison(
                ard_flags[s2_id], s2_scl
            )
        except Exception as exc:
            diag.load_error = f"s2 load failed: {exc}"

        diag.ls_ard_flag_clear_frac = _flag_clear_frac(ard_flags[anchor_id])

        if args.l1c_root:
            try:
                s2_item = _with_retry(
                    lambda href=s2["item_href"]: resolve_item_from_href(str(href)),
                    base_delay=args.sleep,
                )
                proba, note = _s2cloudless_proba(s2_item, s2c_bands, args.l1c_root)
                if proba is not None:
                    diag.s2cloudless_frac = float(np.sum(proba > 0.4)) / max(int(proba.size), 1)
                diag.s2cloudless_note = note
            except Exception as exc:
                diag.s2cloudless_note = f"s2cloudless failed: {exc}"
        else:
            diag.s2cloudless_note = "s2cloudless: unavailable — B10 source not supplied"

        diagnostics.append(diag)
        if i < len(sample):
            time.sleep(args.sleep)

    # ── per-pair table ────────────────────────────────────────────────────
    header = (
        f"{'anchor':<22} {'ls.clear':>8} {'ls.extra':>8} {'ls.ard':>7} "
        f"{'s2.clear':>8} {'s2.extra':>8} {'s2.ard':>7} {'missed':>7} {'shadow+':>7} {'s2c':>6}"
    )
    print(header)
    print("-" * len(header))
    for d in diagnostics:
        print(
            f"{d.landsat_scene_id[:21]:<22} "
            f"{d.ls_native_clear_frac:>8.3f} {d.ls_extra_candidate_frac:>8.3f} "
            f"{_f(d.ls_ard_flag_clear_frac):>7.3f} "
            f"{d.s2_native_clear_frac:>8.3f} {d.s2_extra_candidate_frac:>8.3f} "
            f"{_f(d.s2_ard_flag_clear_frac):>7.3f} "
            f"{_f(d.s2_ard_missed_frac):>7.3f} "
            f"{_f(d.s2_ard_extra_shadow_frac):>7.3f} "
            f"{_f(d.s2cloudless_frac):>6.3f}"
        )
    print()
    print("Quantiles across sampled pairs (fraction of grid pixels):")
    ls_clear = [d.ls_native_clear_frac for d in diagnostics]
    ls_extra = [d.ls_extra_candidate_frac for d in diagnostics]
    s2_clear = [d.s2_native_clear_frac for d in diagnostics]
    s2_extra = [d.s2_extra_candidate_frac for d in diagnostics]
    print(_quantile_line("Landsat native clear", ls_clear))
    print(_quantile_line("Landsat extra candidates", ls_extra))
    print(_quantile_line("S2 native clear", s2_clear))
    print(_quantile_line("S2 extra candidates (class 7)", s2_extra))
    s2_ard_clear = [
        d.s2_ard_flag_clear_frac for d in diagnostics if d.s2_ard_flag_clear_frac is not None
    ]
    if s2_ard_clear:
        print(_quantile_line("S2 ARD flag clear", s2_ard_clear))
    s2_missed = [d.s2_ard_missed_frac for d in diagnostics if d.s2_ard_missed_frac is not None]
    if s2_missed:
        print(_quantile_line("S2 ARD-clear but SCL not-clear", s2_missed))
    s2_extra_shadow = [
        d.s2_ard_extra_shadow_frac for d in diagnostics if d.s2_ard_extra_shadow_frac is not None
    ]
    if s2_extra_shadow:
        print(_quantile_line("S2 ARD-flagged but SCL clear (shadow+)", s2_extra_shadow))

    # s2cloudless availability summary (descriptive, never a gate).
    notes = [d.s2cloudless_note for d in diagnostics if d.s2cloudless_note]
    load_errors = [d.load_error for d in diagnostics if d.load_error]
    if args.l1c_root:
        ok = [d for d in diagnostics if d.s2cloudless_frac is not None]
        print(f"\ns2cloudless: {len(ok)}/{len(diagnostics)} pairs computed (threshold 0.4)")
        if ok:
            print(_quantile_line("s2cloudless > 0.4", [d.s2cloudless_frac for d in ok]))
    else:
        print("\ns2cloudless: unavailable — B10 source not supplied (see --l1c-root)")
    if load_errors:
        print("load errors:")
        for err in sorted(set(load_errors)):
            print(f"  - {err}")
    if notes:
        print("notes:")
        for n in sorted(set(notes)):
            print(f"  - {n}")

    if args.show:
        _show_overlays(diagnostics, manifest_rows)

    print("\nProbe complete — descriptive only; no masks changed, nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
