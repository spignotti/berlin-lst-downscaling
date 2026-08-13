"""Display-only PNG panel rendering for the cloud-masking audit.

Renders S2 true-colour, SCL, and ARD-flag panels into a single PNG for
human inspection of saved audit evidence. Downsampling happens here and
only for display; the audit's diagnostic fractions are never resampled.
"""

from __future__ import annotations

import io

import numpy as np

# Display size for saved PNG panels (diagnostics are never resampled).
_PANEL_WIDTH = 800

# SCL display palette (class -> RGB).
_SCL_COLORS = {
    0: (20, 20, 20),  # no-data
    1: (120, 120, 120),  # saturated
    2: (35, 105, 35),  # dark features
    3: (60, 60, 90),  # cloud shadow
    4: (60, 120, 60),  # vegetation
    5: (90, 90, 60),  # bare soil
    6: (40, 100, 160),  # water
    7: (200, 200, 200),  # unclassified
    8: (200, 200, 200),  # cloud medium
    9: (220, 220, 220),  # cloud high
    10: (140, 180, 200),  # cirrus
    11: (230, 240, 250),  # snow / ice
}


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch of a float reflectance stack to uint8."""
    valid = rgb[np.isfinite(rgb)]
    if valid.size == 0:
        return np.zeros(rgb.shape[:2] + (3,), dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(rgb.shape[:2] + (3,), dtype=np.uint8)
    out = (rgb - lo) / (hi - lo)
    out = np.clip(out, 0, 1) * 255
    return np.nan_to_num(out, nan=0).astype(np.uint8)


def scl_rgb(scl: np.ndarray) -> np.ndarray:
    """Map SCL classes to a fixed display palette."""
    out = np.zeros(scl.shape + (3,), dtype=np.uint8)
    for cls, color in _SCL_COLORS.items():
        out[scl == cls] = color
    return out


def flag_rgb(flag: np.ndarray) -> np.ndarray:
    """Render the ARD flag bitmask as a coloured overlay.

    Clear (flag==0) is dark; each flag bit gets its own colour, later
    bits drawn on top of earlier ones.
    """
    from berlin_lst_downscaling.data.ard.contract import Contract

    base = np.full(flag.shape + (3,), 25, dtype=np.uint8)
    layers = (
        (Contract.FLAG_CLOUDY, (230, 60, 60)),
        (Contract.FLAG_SHADOW, (240, 200, 40)),
        (Contract.FLAG_CIRRUS, (200, 90, 220)),
        (Contract.FLAG_SATURATED, (250, 140, 40)),
        (Contract.FLAG_SNOW_ICE, (240, 245, 250)),
        (Contract.FLAG_FILL, (70, 70, 70)),
    )
    out = base.copy()
    for bit, color in layers:
        mask = (flag & bit) != 0
        out[mask] = color
    return out


def _downscale(arr: np.ndarray, max_width: int = _PANEL_WIDTH) -> np.ndarray:
    """Downscale a (H, W, 3) display array to *max_width* via PIL."""
    from PIL import Image

    h, w = arr.shape[:2]
    scale = max_width / w
    new_size = (max_width, max(1, int(round(h * scale))))
    img = Image.fromarray(arr)
    return np.asarray(img.resize(new_size, Image.LANCZOS))


def compose_png(rgb: np.ndarray, scl: np.ndarray, flag: np.ndarray) -> bytes:
    """Compose three display panels (S2 RGB, SCL, ARD flag) into one PNG."""
    from PIL import Image, ImageDraw

    panels = [_downscale(stretch_rgb(rgb)), _downscale(scl_rgb(scl)), _downscale(flag_rgb(flag))]
    width = max(p.shape[1] for p in panels)
    height = sum(p.shape[0] for p in panels) + 3 * 14
    canvas = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, panel in zip(("S2 RGB", "SCL", "ARD flag"), panels, strict=True):
        draw.text((6, y + 2), label, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(panel), (0, y + 14))
        y += panel.shape[0] + 14
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def load_s2_rgb(item, grid) -> np.ndarray:
    """Load the S2 true-colour stack (red/green/blue) on the 10 m grid."""
    from berlin_lst_downscaling.data.acquisition.pc_client import stac_load

    ds = stac_load(
        items=[item],
        bands=["red", "green", "blue"],
        geobox=grid,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return np.stack([ds[b].values[0].astype(np.float32) for b in ("red", "green", "blue")], axis=-1)
