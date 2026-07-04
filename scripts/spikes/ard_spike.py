# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""ARD spike — load Landsat + S2 scenes via PC STAC and odc-stac.

Usage
-----
    uv run python scripts/spikes/ard_spike.py
    uv run python scripts/spikes/ard_spike.py --date 2024-06-29
    uv run python scripts/spikes/ard_spike.py --list-items           # fast search only
    uv run python scripts/spikes/ard_spike.py --out /tmp/spike.png
    uv run python scripts/spikes/ard_spike.py --verbose

Output
------
- Console: per-sensor item IDs, bands, coverage report, performance timing
- Side-by-side RGB composite PNG (mid-gray for nodata / invalid pixels)
- ``--list-items``: fast STAC search only, no ``odc.stac.load``
- Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.Dataset
from PIL import Image

from berlin_lst_downscaling.common.config import settings
from berlin_lst_downscaling.data.acquisition import (
    get_catalog,
    load_landsat_scene,
    load_s2_scene,
)

# ── helpers ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=None,
        help=f"ISO date (default: {settings.default_date})",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help='Comma-separated "minx,miny,maxx,maxy" WGS84 (default: Berlin)',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: data/tmp/ard_spike_<date>.png)",
    )
    parser.add_argument(
        "--list-items",
        action="store_true",
        help="Fast mode: list available items (no data loading)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostics",
    )
    return parser.parse_args()


def _parse_bbox(text: str | None) -> tuple[float, float, float, float]:
    if text is None:
        return settings.berlin_bbox
    parts = [float(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox requires 4 comma-separated values, got {len(parts)}")
    minx, miny, maxx, maxy = parts
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bbox: min must be < max ({minx}, {miny}) > ({maxx}, {maxy})")
    return (minx, miny, maxx, maxy)


def _clip_for_display(arr: np.ndarray) -> np.ndarray:
    """Clip array to 2nd/98th percentile and rescale to 0-255 uint8.

    NaN values are left as NaN — callers should overlay a nodata mask.
    """
    flat = arr[~np.isnan(arr)]
    if flat.size == 0:
        return np.full(arr.shape, np.nan, dtype=np.float32)

    p2, p98 = np.percentile(flat, (2, 98))
    if p98 - p2 < 1e-8:
        return np.full(arr.shape, 127, dtype=np.uint8)

    clipped = np.clip((arr - p2) / (p98 - p2), 0, 1)
    return (clipped * 255).astype(np.uint8)


def _rgb_panel(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> Image.Image:
    """Compose a 3-band RGB Image with optional nodata → mid-gray overlay."""
    r8 = _clip_for_display(r)
    g8 = _clip_for_display(g)
    b8 = _clip_for_display(b)

    img = np.stack([r8, g8, b8], axis=-1).astype(np.uint8)

    if nodata_mask is not None:
        img[nodata_mask] = (128, 128, 128)

    return Image.fromarray(img, mode="RGB")


def _coverage_report(ds, label: str, band: str, nodata_zero: bool = False):
    """Print spatial coverage stats for a Dataset."""
    arr = ds[band].values.squeeze()
    total = arr.size

    if nodata_zero:
        valid = ~np.isnan(arr) & (arr != 0)
    else:
        valid = ~np.isnan(arr)

    n_valid = valid.sum()
    frac = n_valid / total * 100

    y_coords, x_coords = np.where(valid)
    print(f"  {label}: {frac:.1f}% valid  ({n_valid:,} / {total:,})")
    print(f"    Y range: {y_coords.min()}–{y_coords.max()}  (out of 0–{arr.shape[0] - 1})")
    print(f"    X range: {x_coords.min()}–{x_coords.max()}  (out of 0–{arr.shape[1] - 1})")

    # Approximate geographic bbox of valid pixels
    if hasattr(ds, "rio") and ds.rio.crs:
        try:
            # pixel indices → CRS coordinates
            ys = y_coords[[0, -1]]
            xs = [x_coords.min(), x_coords.max()]
            xs_f = np.array(xs, dtype=float)
            ys_f = np.array(ys, dtype=float)
            xs_f, ys_f = ds.rio.transform() * (xs_f, ys_f)
            print(
                f"    Geo bbox (EPSG:{ds.rio.crs.to_epsg()}): "
                f"{xs_f[0]:.2f},{ys_f[0]:.2f} → {xs_f[1]:.2f},{ys_f[1]:.2f}"
            )
        except Exception as exc:
            print(f"    (geo bbox diagnostic skipped: {exc})", file=sys.stderr)


def _ls_nodata_mask(ds) -> np.ndarray:
    """Landsat nodata: all 3 RGB bands exactly 0 (swath-edge fill)."""
    return (
        (ds["red"].values.squeeze() == 0)
        & (ds["green"].values.squeeze() == 0)
        & (ds["blue"].values.squeeze() == 0)
    )


def _s2_nodata_mask(ds) -> np.ndarray:
    """S2 nodata: NaN in the red band."""
    return np.isnan(ds["B04"].values.squeeze())


def _merge_nodata_frac_ls(ds) -> str:
    """Fraction of Landsat pixels that are **not** nodata-filled."""
    mask = _ls_nodata_mask(ds)
    total = mask.size
    valid = (~mask).sum()
    frac = valid / total * 100
    return f"{frac:.1f}% ({valid:,} / {total:,})"


def _merge_nodata_frac_ls(ds) -> str:
    """Fraction of Landsat pixels that are **not** nodata-filled (value=0)."""
    mask = _ls_nodata_mask(ds)
    total = mask.size
    valid = (~mask).sum()
    frac = valid / total * 100
    return f"{frac:.1f}% ({valid:,} / {total:,})"


def _merge_nodata_frac_s2(ds) -> str:
    """Fraction of S2 pixels that are **not** NaN."""
    mask = _s2_nodata_mask(ds)
    total = mask.size
    valid = (~mask).sum()
    frac = valid / total * 100
    return f"{frac:.1f}% ({valid:,} / {total:,})"


def _list_items_mode(date: str, bbox) -> int:
    """Search items for both sensors and print an overview (no data loading)."""
    sep = "-" * 60
    catalog = get_catalog()

    for collection, label in [
        ("landsat-c2-l2", "Landsat C2-L2"),
        ("sentinel-2-l2a", "S2 L2A"),
    ]:
        print(sep)
        print(f"{label} — date={date}  bbox={bbox}")
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=date,
            max_items=50,
        )
        items = list(search.items())
        print(f"  Found {len(items)} items:")
        for it in items:
            props = it.properties
            cloud = props.get("eo:cloud_cover", "?")
            dt = props.get("datetime", "?")
            bbox_str = (
                f"{it.bbox[0]:.2f},{it.bbox[1]:.2f} → {it.bbox[2]:.2f},{it.bbox[3]:.2f}"
                if it.bbox
                else "?"
            )
            print(f"    {it.id}")
            print(f"      datetime={dt}, cloud={cloud}%, bbox={bbox_str}")

    print(sep)
    print("Use --date to change date, or omit --list-items to load+render.")
    return 0


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    date = args.date or settings.default_date
    bbox = _parse_bbox(args.bbox)
    verbose = args.verbose

    if args.list_items:
        return _list_items_mode(date=date, bbox=bbox)

    out_path = args.out or os.fspath(Path("data") / "tmp" / f"ard_spike_{date}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    sep = "-" * 60

    # ════════════════ Landsat ════════════════
    print(sep)
    print(f"Landsat  —  date={date}  bbox={bbox}")
    t_ls_start = time.perf_counter()

    try:
        ls_ds, ls_item_ids = load_landsat_scene(date=date, bbox=bbox)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1

    t_ls_end = time.perf_counter()

    print(f"  Loaded items: {len(ls_item_ids)}")
    for iid in ls_item_ids:
        print(f"    {iid}")
    print(f"  CRS         : {ls_ds.rio.crs}")
    print(f"  Shape       : {dict(ls_ds.sizes)}")
    print(f"  Bands       : {list(ls_ds.data_vars)}")
    ls_dtypes = {k: str(ls_ds[k].dtype) for k in ls_ds.data_vars}
    print(f"  Dtypes      : {ls_dtypes}")
    print(f"  Valid pixels: {_merge_nodata_frac_ls(ls_ds)}")
    print(f"  Load time   : {t_ls_end - t_ls_start:.2f}s")
    if verbose:
        _coverage_report(ls_ds, "Landsat cov.", "red", nodata_zero=True)

    # ════════════════ Sentinel-2 ════════════════
    print(sep)
    print(f"Sentinel-2  —  date={date}  bbox={bbox}")
    t_s2_start = time.perf_counter()

    try:
        s2_ds, s2_item_ids = load_s2_scene(date=date, bbox=bbox)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1

    t_s2_end = time.perf_counter()

    print(f"  Loaded items: {len(s2_item_ids)}")
    for iid in s2_item_ids:
        print(f"    {iid}")
    print(f"  CRS         : {s2_ds.rio.crs}")
    print(f"  Shape       : {dict(s2_ds.sizes)}")
    print(f"  Bands       : {list(s2_ds.data_vars)}")
    s2_dtypes = {k: str(s2_ds[k].dtype) for k in s2_ds.data_vars}
    print(f"  Dtypes      : {s2_dtypes}")
    print(f"  Valid pixels: {_merge_nodata_frac_s2(s2_ds)}")
    print(f"  Load time   : {t_s2_end - t_s2_start:.2f}s")
    if verbose:
        _coverage_report(s2_ds, "S2 cov.", "B04")

    # ════════════════ Render ════════════════
    print(sep)
    print("Rendering RGB composite …")

    t_render_start = time.perf_counter()

    try:
        ls_mask = _ls_nodata_mask(ls_ds)
        s2_mask = _s2_nodata_mask(s2_ds)

        ls_rgb = _rgb_panel(
            ls_ds["red"].values.squeeze(),
            ls_ds["green"].values.squeeze(),
            ls_ds["blue"].values.squeeze(),
            nodata_mask=ls_mask,
        )
        s2_rgb = _rgb_panel(
            s2_ds["B04"].values.squeeze(),
            s2_ds["B03"].values.squeeze(),
            s2_ds["B02"].values.squeeze(),
            nodata_mask=s2_mask,
        )
    except KeyError as exc:
        print(f"  Missing band for RGB: {exc}")
        return 1

    # Side-by-side composite
    max_h = max(ls_rgb.height, s2_rgb.height)
    total_w = ls_rgb.width + s2_rgb.width

    composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
    composite.paste(ls_rgb, (0, 0))
    composite.paste(s2_rgb, (ls_rgb.width, 0))

    composite.save(out_path)
    t_render_end = time.perf_counter()

    print(
        f"  Saved: {out_path} ({composite.width}×{composite.height} px, "
        f"{t_render_end - t_render_start:.1f}s)"
    )

    # ════════════════ Summary ════════════════
    t_total = time.perf_counter() - t_start
    print(sep)
    print(f"Spike OK — {t_total:.1f}s total.")
    print(f"  Landsat  : {len(ls_item_ids)} item(s)")
    print(f"  Sentinel2: {len(s2_item_ids)} item(s)")
    print(f"  Output   : {out_path}")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
