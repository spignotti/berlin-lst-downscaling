# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""ARD spike — load one Landsat + one S2 scene via PC STAC and odc-stac.

Usage
-----
    uv run python scripts/spikes/ard_spike.py
    uv run python scripts/spikes/ard_spike.py --date 2024-07-15
    uv run python scripts/spikes/ard_spike.py --out /tmp/spike.png

Output
------
- Console summary (item IDs, shape per DS, dtype, valid-pixel fractions)
- Side-by-side 2-panel RGB composite PNG at ``--out`` (default
  ``data/tmp/ard_spike_<date>.png``).
- Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.Dataset
from PIL import Image

from berlin_lst_downscaling.common.config import settings
from berlin_lst_downscaling.data.acquisition import load_landsat_scene, load_s2_scene


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
    """Clip array to 2nd/98th percentile and rescale to 0-255 uint8."""
    # Flatten for percentile computation, ignoring NaN
    flat = arr[~np.isnan(arr)]
    if flat.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    p2, p98 = np.percentile(flat, (2, 98))
    if p98 - p2 < 1e-8:
        return np.full(arr.shape, 127, dtype=np.uint8)

    clipped = np.clip((arr - p2) / (p98 - p2), 0, 1)
    return (clipped * 255).astype(np.uint8)


def _rgb_panel(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Image.Image:
    """Compose a 3-band RGB Image from band arrays."""
    r8 = _clip_for_display(r)
    g8 = _clip_for_display(g)
    b8 = _clip_for_display(b)
    return Image.fromarray(np.stack([r8, g8, b8], axis=-1), mode="RGB")


def _valid_frac_ls(ds) -> str:
    """Parse Landsat qa_pixel to report valid (not cloudy/high-confidence) fraction."""
    try:
        qa = ds["qa_pixel"].values
        mask = qa  # bit 3 = cloud shadow, bit 5 = cloud, bit 6/7 = high conf cirrus/cloud
        # Build mask: clear = none of those bits
        cloud = mask & (1 << 5) != 0
        shadow = mask & (1 << 3) != 0
        clear = ~(cloud | shadow)
        total = (~np.isnan(qa)).sum()
        if total == 0:
            return "— (no valid pixels)"
        frac = clear.sum() / total
        valid_pixels = clear.sum()
        return f"{frac:.1%} ({valid_pixels:,} / {total:,})"
    except (KeyError, ValueError, TypeError):
        return "—"


def _valid_frac_s2(ds) -> str:
    """Parse S2 SCL to report valid (vegetation/bare/water, not cloud/shadow) fraction."""
    try:
        scl = ds["SCL"].values
        # SCL values: 2=water, 4=vegetation, 5=bare, 6=dark veg, 7=unclassified
        valid_classes = {2, 4, 5, 6, 7}
        total = (~np.isnan(scl)).sum()
        if total == 0:
            return "— (no valid pixels)"
        valid = sum((scl[~np.isnan(scl)] == v).sum() for v in valid_classes)
        frac = valid / total
        return f"{frac:.1%} ({valid:,} / {total:,})"
    except (KeyError, ValueError, TypeError):
        return "—"


def _gray_panel(arr: np.ndarray) -> Image.Image:
    """Render a single-band array as grayscale image (2-98% stretch)."""
    grey = _clip_for_display(arr)
    return Image.fromarray(grey, mode="L")


def main() -> int:
    args = _parse_args()
    date = args.date or settings.default_date
    bbox = _parse_bbox(args.bbox)
    out_path = args.out or os.fspath(
        Path("data") / "tmp" / f"ard_spike_{date}.png"
    )

    # Ensure output dir exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    separator = "-" * 60

    # ---------- Landsat ----------
    print(separator)
    print(f"Landsat  —  date={date}  bbox={bbox}")
    try:
        ls_ds, ls_item_id = load_landsat_scene(date=date, bbox=bbox)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"  Item ID    : {ls_item_id}")
    print(f"  CRS        : {ls_ds.rio.crs}")
    print(f"  Shape      : {dict(ls_ds.sizes)}")
    print(f"  Bands      : {list(ls_ds.data_vars)}")
    ls_dtypes = {k: str(ls_ds[k].dtype) for k in ls_ds.data_vars}
    print(f"  Dtypes     : {ls_dtypes}")
    print(f"  Valid frac : {_valid_frac_ls(ls_ds)}")

    # ---------- Sentinel-2 ----------
    print(separator)
    print(f"Sentinel-2  —  date={date}  bbox={bbox}")
    try:
        s2_ds, s2_item_id = load_s2_scene(date=date, bbox=bbox)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"  Item ID    : {s2_item_id}")
    print(f"  CRS        : {s2_ds.rio.crs}")
    print(f"  Shape      : {dict(s2_ds.sizes)}")
    print(f"  Bands      : {list(s2_ds.data_vars)}")
    s2_dtypes = {k: str(s2_ds[k].dtype) for k in s2_ds.data_vars}
    print(f"  Dtypes     : {s2_dtypes}")
    print(f"  Valid frac : {_valid_frac_s2(s2_ds)}")

    # ---------- RGB composite ----------
    print(separator)
    print("Rendering RGB composite …")

    try:
        ls_rgb = _rgb_panel(
            ls_ds["red"].values.squeeze(),
            ls_ds["green"].values.squeeze(),
            ls_ds["blue"].values.squeeze(),
        )
        s2_rgb = _rgb_panel(
            s2_ds["B04"].values.squeeze(),  # S2 Red
            s2_ds["B03"].values.squeeze(),  # S2 Green
            s2_ds["B02"].values.squeeze(),  # S2 Blue
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
    print(f"  Saved: {out_path} ({composite.width}×{composite.height} px)")

    # ---------- Summary ----------
    print(separator)
    print("Spike OK — both sensors loaded successfully.")
    print(f"  Landsat  : {ls_item_id}")
    print(f"  Sentinel2: {s2_item_id}")
    print(separator)

    return 0


if __name__ == "__main__":
    sys.exit(main())
