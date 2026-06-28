"""Quicklook thumbnail and contact sheet generation for ARD COGs.

Generates 8-bit PNG thumbnails (RGB or grayscale with a linear stretch)
and contact sheet mosaics from processed COGs. Uses Pillow for image
composition and rasterio for band I/O.

Usage::

    from berlin_lst_downscaling.data.quicklook import generate_thumbnail

    generate_thumbnail(
        cog_path=Path("/tmp/output.tif"),
        thumbnail_path=Path("/tmp/thumb.png"),
        width=512,
    )

    from berlin_lst_downscaling.data.quicklook import generate_contact_sheet

    generate_contact_sheet(
        thumbnail_paths=[Path("/tmp/thumb1.png"), ...],
        output_path=Path("/tmp/contact_sheet.png"),
        cols=8,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

LANCZOS = Image.Resampling.LANCZOS

logger = logging.getLogger(__name__)


def generate_thumbnail(
    cog_path: Path,
    thumbnail_path: Path,
    width: int = 512,
) -> Path:
    """Generate an 8-bit PNG thumbnail from a COG.

    For multi-band COGs (3+ bands): uses the first 3 bands as RGB.
    For single/dual-band COGs: uses the first band as grayscale.
    A 2-98 percentile linear stretch is applied to the data.

    Args:
        cog_path: Path to the input COG.
        thumbnail_path: Path to the output PNG thumbnail.
        width: Target width in pixels (height auto-computed).

    Returns:
        Path to the generated thumbnail.
    """
    with rasterio.open(cog_path) as src:
        src_count = src.count
        bands: list[np.ndarray] = []
        for i in range(min(src_count, 3)):
            band = src.read(i + 1).astype(np.float64)
            bands.append(band)

    # Build 3-band RGB array
    if len(bands) >= 3:
        rgb = np.stack([
            _stretch_uint8(bands[0]),
            _stretch_uint8(bands[1]),
            _stretch_uint8(bands[2]),
        ], axis=-1)
    else:
        gray = _stretch_uint8(bands[0])
        rgb = np.stack([gray, gray, gray], axis=-1)

    # Compute height to preserve aspect ratio
    h, w = rgb.shape[:2]
    aspect = h / w
    height = max(1, round(width * aspect))

    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((width, height), LANCZOS)
    img.save(thumbnail_path)

    return thumbnail_path


def generate_contact_sheet(
    thumbnail_dir: Path,
    output_path: Path,
    cols: int = 8,
    thumb_width: int = 256,
) -> Path:
    """Generate a contact sheet from a directory of thumbnail PNGs.

    Arranges thumbnails in a grid (``cols`` wide), labels each with its
    filename (without extension), and writes a single PNG.

    Args:
        thumbnail_dir: Directory containing thumbnail PNGs.
        output_path: Path to write the contact sheet PNG.
        cols: Number of thumbnail columns in the grid.
        thumb_width: Width of each thumbnail in the grid.

    Returns:
        Path to the generated contact sheet.
    """
    png_files = sorted(thumbnail_dir.glob("*.png"))
    if not png_files:
        logger.warning("No thumbnails found in %s", thumbnail_dir)
        return output_path

    # Load thumbnails
    thumbnails: list[Image.Image] = []
    for p in png_files:
        try:
            thumbnails.append(Image.open(p))
        except Exception as exc:
            logger.warning("Skipping unreadable thumbnail %s: %s", p, exc)

    if not thumbnails:
        return output_path

    # Layout: uniform grid
    n = len(thumbnails)
    rows = (n + cols - 1) // cols
    label_h = 20  # pixels for filename label below each thumbnail

    cell_w = thumb_width
    cell_h = thumb_width + label_h

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for idx, thumb in enumerate(thumbnails):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h

        resized = thumb.resize((thumb_width, thumb_width), LANCZOS)
        canvas.paste(resized, (x, y))

        # Label
        label = png_files[idx].stem[:20]
        draw.text((x + 2, y + thumb_width + 2), label, fill=(200, 200, 200))

    canvas.save(output_path)
    logger.info("Contact sheet written: %s (%d thumbnails)", output_path, n)
    return output_path


def _stretch_uint8(band: np.ndarray) -> np.ndarray:
    """Apply a 2-98 percentile linear stretch and convert to uint8."""
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    p2, p98 = np.percentile(valid, (2, 98))
    if p98 <= p2:
        p98 = p2 + 1.0

    stretched = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def generate_contact_sheet_from_gcs(
    bucket: str,
    prefix: str,
    year: int,
    output_path: Path,
    temp_dir: Path | None = None,
    cols: int = 8,
    thumb_width: int = 256,
) -> Path:
    """Generate a contact sheet from thumbnails stored on GCS.

    Downloads thumbnails from ``gs://{bucket}/{prefix}/{year}/thumbnails/``
    to a temporary directory, composes a contact sheet, and writes the
    result to ``output_path``.

    Args:
        bucket: GCS bucket name.
        prefix: GCS output prefix (e.g. ``"ard/processed/sentinel2"``).
        year: Year.
        output_path: Path to write the contact sheet PNG.
        temp_dir: Temporary directory for downloads (default: system tmp).
        cols: Number of thumbnail columns in the grid.
        thumb_width: Width of each thumbnail in the grid.

    Returns:
        Path to the generated contact sheet.
    """
    import tempfile

    from google.cloud import storage

    client = storage.Client()
    thumb_prefix = f"{prefix}/{year}/thumbnails/"
    blobs = list(client.list_blobs(bucket, prefix=thumb_prefix))

    if not blobs:
        logger.warning("No thumbnails found at gs://%s/%s", bucket, thumb_prefix)
        return output_path

    local_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
    local_dir.mkdir(parents=True, exist_ok=True)

    local_thumbnails: list[Path] = []
    for blob in blobs:
        local_path = local_dir / blob.name.replace("/", "_")
        blob.download_to_filename(str(local_path))
        local_thumbnails.append(local_path)

    thumbnail_dir = local_dir / f"contact_{year}"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    for p in local_thumbnails:
        # Move or symlink into a flat directory for contact sheet generation
        dest = thumbnail_dir / p.name
        p.rename(dest)

    result = generate_contact_sheet(
        thumbnail_dir, output_path, cols=cols, thumb_width=thumb_width,
    )

    # Cleanup
    import shutil

    shutil.rmtree(local_dir, ignore_errors=True)

    return result
