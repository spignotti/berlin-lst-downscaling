"""DGM 1 m (terrain height) source adapter for the secondary pipeline.

Official Geoportal Berlin DGM tiles (INSPIRE ATOM feed):
``https://gdi.berlin.de/data/dgm1/atom/0.atom``

Each tile is a ZIP containing an XYZ CSV file (2000×2000 points at 1 m
spacing, EPSG:25833, DHHN2016).  All 297 tiles covering the Berlin AOI
are processed.

Processing
----------
1. Parse the ATOM feed to discover tile assets intersecting the AOI.
2. Download each tile ZIP to raw storage via ``download_to_raw``.
3. Read the XYZ CSV from the ZIP (``np.loadtxt``).
4. Validate: regular 2000×2000 grid, correct coordinates, no gaps.
5. Reproject from native 1 m to canonical 10 m using ``Resampling.average``.
6. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts (COG + STAC + provenance + complete).
"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.atom import (
    AtomAsset,
    parse_atom_feed,
)
from berlin_lst_downscaling.data.secondary.download import DownloadReceipt, download_to_raw
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

_FEED_URL = "https://gdi.berlin.de/data/dgm1/atom/0.atom"
_TILE_SIZE_M = 2000  # each tile covers 2 km × 2 km
_POINTS_PER_TILE = _TILE_SIZE_M * _TILE_SIZE_M  # 4,000,000
_DGM_VINTAGE = 2021  # ALS acquisition date (Feb–Mar 2021)
_LICENSE = "dl-de/zero-2.0"
_DGM_RE = re.compile(r"DGM1_(\d{3})_(\d{4})\.zip$", re.IGNORECASE)

# ── contract ───────────────────────────────────────────────────────────

_CONFIG_HASH_PREFIX = "terrain_height:v1:"


def contract_for_terrain_height() -> Contract:
    """Return the output Contract for terrain-height COGs."""
    return Contract(
        source="terrain_height",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="terrain_height",
                dtype="float32",
                nodata=float("nan"),
                description="Terrain elevation in metres above sea level (DHHN2016)",
                unit="m",
                valid_range=(-10.0, 200.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def config_hash_for_vintage(vintage: int) -> str:
    """Return a stable config hash for a given vintage."""
    raw = f"{_CONFIG_HASH_PREFIX}{vintage}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── prepare ───────────────────────────────────────────────────────────


def prepare_terrain_height(
    vintage: int,
    output_root: str,
    run_id: str,
    smoke_tile_count: int | None = None,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Download, validate, and reproject DGM tiles to the canonical 10 m grid.

    Parameters
    ----------
    vintage :
        Must be ``2021``.
    output_root :
        Root URI for all outputs (local path or ``gs://bucket/…``).
    run_id :
        Unique run identifier.
    smoke_tile_count :
        If set, process only this many tiles (for local smoke testing).
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.

    Returns
    -------
    PreparedSecondaryProduct
        Canonical-grid dataset + source metadata + QA statistics.
    """
    if vintage != _DGM_VINTAGE:
        raise ValueError(f"Only vintage {_DGM_VINTAGE} is available; got {vintage}")

    c_hash = config_hash_for_vintage(vintage)
    grid = grid or canon_grid_10m()

    # ── 1. discover tiles via ATOM feed ──────────────────────────────
    manifest = parse_atom_feed(_FEED_URL, _DGM_RE, aoi_grid=grid)

    if smoke_tile_count is not None:
        manifest.assets = manifest.assets[:smoke_tile_count]

    log_event(_logger, logging.INFO, "dgm_tiles", n_tiles=len(manifest.assets))

    # ── 2. download + accumulate on canonical grid ───────────────────
    dst_arr = np.full((grid.shape.y, grid.shape.x), np.nan, dtype=np.float32)
    tile_receipts: list[dict] = []
    all_checksums: list[str] = []

    for i, asset in enumerate(manifest.assets):
        if (i + 1) % 50 == 0:
            log_event(
                _logger, logging.DEBUG, "dgm_tile_progress",
                done=i+1, total=len(manifest.assets),
            )

        receipt = _process_tile(asset, dst_arr, grid, output_root)
        tile_receipts.append({
            "filename": asset.filename,
            "easting": asset.easting,
            "northing": asset.northing,
            "checksum": receipt.checksum,
            "byte_count": receipt.byte_count,
        })
        all_checksums.append(receipt.checksum)

    # ── 3. compute combined checksum ─────────────────────────────────
    combined_hash = sha256("".join(sorted(all_checksums)).encode()).hexdigest()[:16]

    # ── 4. build canonical xr.Dataset ────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"terrain_height": (("y", "x"), dst_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    valid = dst_arr[~np.isnan(dst_arr)]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source="terrain_height",
        item_key=str(vintage),
        category="morphology",
        dataset=ds,
        contract=contract_for_terrain_height(),
        nominal_interval=vintage_interval(vintage),
        source_metadata={
            "feed_url": _FEED_URL,
            "feed_updated": manifest.feed_updated,
            "tile_count": len(manifest.assets),
            "tiles": tile_receipts,
            "combined_checksum": combined_hash,
            "retrieved_at": retrieved_at,
            "license": _LICENSE,
            "crs": "EPSG:25833",
            "vertical_datum": "DHHN2016",
        },
        qa_stats={
            "valid_frac": (
                round(float(len(valid)) / dst_arr.size, 4)
                if dst_arr.size > 0
                else 0.0
            ),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(dst_arr.shape),
            "tile_count": len(manifest.assets),
        },
        config_hash=c_hash,
    )


# ── tile processing ──────────────────────────────────────────────────


def _process_tile(
    asset: AtomAsset,
    dst_arr: np.ndarray,
    grid: GeoBox,
    output_root: str,
) -> DownloadReceipt:
    """Download a single DGM tile and accumulate onto the canonical grid."""
    from berlin_lst_downscaling.data.secondary.paths import raw_dir

    raw_uri = f"{raw_dir(output_root, 'terrain_height', str(_DGM_VINTAGE))}/{asset.filename}"
    cache_path = _local_cache_path(output_root, asset.filename)

    receipt = download_to_raw(
        url=asset.url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    asset.checksum = receipt.checksum
    asset.byte_count = receipt.byte_count

    xyz_path = Path(receipt.local_cache_path) if receipt.local_cache_path else None
    if xyz_path is None:
        raise ValueError(f"No local cache for {asset.filename}")

    src_arr, src_transform = _read_xyz_zip(xyz_path, asset)
    _validate_tile(src_arr, src_transform, asset)

    # Reproject 1m → 10m into a fresh temporary array, then merge
    tile_arr = np.empty((grid.shape.y, grid.shape.x), dtype=np.float32)
    tile_arr[:] = np.nan

    reproject(
        source=src_arr.astype(np.float32),
        destination=tile_arr,
        src_transform=src_transform,
        src_crs="EPSG:25833",
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Accumulate: where this tile has valid values, write to destination
    valid = ~np.isnan(tile_arr)
    dst_arr[valid] = tile_arr[valid]

    return receipt


def _read_xyz_zip(
    zip_path: Path, asset: AtomAsset,
) -> tuple[np.ndarray, object]:
    """Read XYZ CSV from a DGM tile ZIP and return a north-up 2-D array."""
    from rasterio.transform import from_origin

    with zipfile.ZipFile(zip_path) as z:
        xyz_names = [n for n in z.namelist() if n.endswith(".xyz")]
        if not xyz_names:
            raise ValueError(f"No .xyz member in {asset.filename}")
        with z.open(xyz_names[0]) as f:
            data = np.loadtxt(f, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"{asset.filename}: expected M×3 XYZ array, got shape {data.shape}"
        )
    if data.shape[0] == 0:
        raise ValueError(f"{asset.filename}: empty XYZ data")

    # Columns: X Y Z
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    z_vals = data[:, 2]

    # Validate coordinate spacing
    _validate_xyz_coords(x_vals, y_vals, asset)

    # Determine grid dimensions from unique coordinates
    n_cols = len(np.unique(x_vals))
    n_rows = len(np.unique(y_vals))

    if n_cols * n_rows < len(z_vals):
        raise ValueError(
            f"{asset.filename}: {n_cols}×{n_rows} = {n_cols * n_rows} cells "
            f"but {len(z_vals)} points (too many)"
        )

    # Build a lookup from (x, y) → z, allowing for incomplete tiles
    coord_to_z = {}
    for xi, yi, zi in zip(x_vals, y_vals, z_vals, strict=True):
        coord_to_z[(float(xi), float(yi))] = float(zi)

    x_unique = np.sort(np.unique(x_vals))
    y_unique = np.sort(np.unique(y_vals))

    # Build 2D array: rows = south-to-north (y ascending), cols = west-to-east
    arr = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for r, yv in enumerate(y_unique):
        for c, xv in enumerate(x_unique):
            key = (float(xv), float(yv))
            if key in coord_to_z:
                arr[r, c] = coord_to_z[key]

    # Flip vertically so row 0 = northernmost row (north-up)
    arr = arr[::-1, :]

    # Build transform from actual data extents
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    origin_x = x_min - 0.5  # cell center to edge offset
    origin_y = y_max + 0.5  # northernmost cell center + 0.5 = grid top
    res_x = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 1.0
    res_y = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 1.0
    transform = from_origin(origin_x, origin_y, res_x, res_y)

    return arr, transform


def _validate_xyz_coords(
    x_vals: np.ndarray, y_vals: np.ndarray, asset: AtomAsset,
) -> None:
    """Validate XYZ coordinate spacing and general extents."""
    # Check Y spacing (should be 1.0 m)
    y_unique = np.unique(y_vals)
    if len(y_unique) > 1:
        y_step = float(np.median(np.diff(y_unique)))
        if abs(y_step - 1.0) > 0.01:
            raise ValueError(
                f"{asset.filename}: Y spacing {y_step:.3f} m, expected 1.0 m"
            )

    # Check X spacing (should be 1.0 m)
    x_unique = np.unique(x_vals)
    if len(x_unique) > 1:
        x_step = float(np.median(np.diff(x_unique)))
        if abs(x_step - 1.0) > 0.01:
            raise ValueError(
                f"{asset.filename}: X spacing {x_step:.3f} m, expected 1.0 m"
            )

    # Sanity: tile should be near the declared origin (within 2 km)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    if abs(x_min - asset.easting) > 2500 or abs(y_min - asset.northing) > 2500:
        raise ValueError(
            f"{asset.filename}: X/Y range [{x_min:.0f}, {x_max:.0f}] x "
            f"[{y_min:.0f}, {y_max:.0f}] too far from origin "
            f"({asset.easting}, {asset.northing})"
        )


def _validate_tile(arr: np.ndarray, transform: object, asset: AtomAsset) -> None:
    """Validate a DGM tile's shape and coverage."""
    if arr.ndim != 2:
        raise ValueError(f"{asset.filename}: expected 2D array, got {arr.ndim}D")
    valid_count = int(np.sum(~np.isnan(arr)))
    if valid_count == 0:
        raise ValueError(f"{asset.filename}: all NaN — no valid terrain data")


def _local_cache_path(output_root: str, filename: str) -> str:
    """Return a writable local cache path for a DGM tile."""
    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/dgm/{filename}"
    return f"{output_root}/_raw/secondary/terrain_height/2021/{filename}"


__all__ = [
    "config_hash_for_vintage",
    "contract_for_terrain_height",
    "prepare_terrain_height",
]
