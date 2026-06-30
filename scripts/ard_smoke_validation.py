#!/usr/bin/env python3
"""ARD smoke-test visual validation — data quality check for pipeline output.

Auto-discovers smoke-test scenes from GCS (``ard/processed/*/2023/``),
downloads them, and generates visual QC overlays for manual review.

Generates per-scene outputs in ``data/tmp/ard_smoke/<ts>/{landsat,sentinel2,ecostress}/``:

* ``<scene>_rgb.png``           — True-color RGB (linear stretch, fast render)
* ``<scene>_georef.png``        — Georeferenced RGB with Berlin boundary + AOI overlay + axes
* ``<scene>_lst_colored.png``   — LST with inferno colour map + colour bar (Landsat/ECOSTRESS)
* ``<scene>_rgb_masked.png``    — RGB + cloud_mask overlay + boundary/axes
* ``<scene>_mask.png``          — Cloud mask binary (red = cloud/shadow)
* ``<scene>_histograms.png``    — Per-band value distributions
* ``<scene>_stats.json``        — Band statistics, cloud %, grid alignment

At the top level:
* ``comparison_all.png``        — 3-panel (Landsat | S2 | ECOSTRESS) same extent + boundary

Uses the GCS client (Google ADC) instead of ``gsutil`` for GCS access.

Usage:
    uv run python scripts/ard_smoke_validation.py
    uv run python scripts/ard_smoke_validation.py --year 2023
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import lines as mlines
from rasterio.vrt import WarpedVRT

from berlin_lst_downscaling.data.ard_qa import compute_aoi_coverage_fraction
from berlin_lst_downscaling.data.gcs_client import download_blob
from berlin_lst_downscaling.data.grid_spec import GridSpec, get_spec

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("data/tmp/ard_smoke")
_GCS_BUCKET = "berlin-lst-data"

# Sources to validate
_SOURCES: dict[str, dict[str, Any]] = {
    "landsat": {
        "input_prefix": "ard/processed/landsat",
    },
    "sentinel2": {
        "input_prefix": "ard/processed/sentinel2",
    },
    "ecostress": {
        "input_prefix": "ard/processed/ecostress",
    },
}

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class BandInfo:
    index: int  # 0-based band index
    name: str  # band description
    data: np.ndarray  # 2D array


@dataclass
class SceneData:
    source: str
    scene_id: str
    cog_path: Path
    bands: list[BandInfo] = field(default_factory=list)
    cloud_mask: np.ndarray | None = None
    rgb_indices: tuple[int, int, int] | None = None  # R, G, B indices (0-based)


# ── Scene discovery ───────────────────────────────────────────────────────────


def discover_scenes(year: int = 2023) -> list[dict[str, Any]]:
    """Discover smoke-test scenes on GCS via the GCS client."""
    from berlin_lst_downscaling.data.gcs_client import list_blobs

    scenes: list[dict[str, Any]] = []
    for source, cfg in _SOURCES.items():
        prefix = f"{cfg['input_prefix']}/{year}/"
        for blob_name in list_blobs(_GCS_BUCKET, prefix=prefix):
            if not blob_name.endswith(".tif"):
                continue
            if blob_name.endswith("_COG.tif"):
                continue
            # Turn blob name into a gs:// URI for internal routing
            cog_uri = f"gs://{_GCS_BUCKET}/{blob_name}"
            scene_id = _parse_scene_id(blob_name)
            if scene_id:
                scenes.append(
                    {
                        "source": source,
                        "scene_id": scene_id,
                        "cog_uri": cog_uri,
                    }
                )
    return scenes


def _parse_scene_id(blob_name: str) -> str | None:
    """Extract scene ID from GCS blob name.

    Processed COGs follow the pattern:
        ard/processed/<source>/<year>/<scene_id>.tif
    """
    stem = Path(blob_name).stem
    return stem if stem else None


# ── Data loading ──────────────────────────────────────────────────────────────


def download_cogs(scenes: list[dict[str, Any]], temp_dir: Path) -> list[SceneData]:
    """Download COGs from GCS to local temp directory."""
    scene_data_list: list[SceneData] = []
    for scene in scenes:
        cog_uri = scene["cog_uri"]
        local_path = _download_from_gcs(cog_uri, temp_dir)
        if local_path is None:
            logger.warning("Failed to download %s", cog_uri)
            continue
        scene_data_list.append(
            SceneData(
                source=scene["source"],
                scene_id=scene["scene_id"],
                cog_path=local_path,
            )
        )
    return scene_data_list


def _download_from_gcs(gcs_uri: str, temp_dir: Path) -> Path | None:
    """Download a single GCS blob to temp dir via GCS client. Returns local path or None."""
    # Extract blob name from gs://bucket/blob.name
    if not gcs_uri.startswith("gs://"):
        return None
    parts = gcs_uri[5:].split("/", 1)  # strip gs://, split bucket/key
    if len(parts) < 2:
        return None
    blob_name = parts[1]
    stem = Path(blob_name).stem
    local_path = temp_dir / f"{stem}.tif"
    if local_path.exists():
        return local_path
    try:
        download_blob(_GCS_BUCKET, blob_name, local_path)
    except Exception:
        return None
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    return None


# ── Band analysis ─────────────────────────────────────────────────────────────


def load_scene_data(scene: SceneData) -> None:
    """Read all bands from a COG, identify cloud_mask by description."""
    with rasterio.open(scene.cog_path) as src:
        for i in range(src.count):
            desc = (src.descriptions[i] or f"band_{i + 1}").lower()
            band_data = src.read(i + 1).astype(np.float64)
            scene.bands.append(BandInfo(index=i, name=desc, data=band_data))

            if desc == "cloud_mask":
                scene.cloud_mask = band_data

    # Identify RGB bands by description
    descs = [b.name for b in scene.bands]
    scene.rgb_indices = _find_rgb_indices(descs, scene.source)


def _find_rgb_indices(
    descriptions: list[str],
    source: str,
) -> tuple[int, int, int] | None:
    """Return (R, G, B) 0-based band indices for true-color compositing.

    Sentinel-2: B4, B3, B2
    Landsat: no RGB bands (LST + mask only)
    ECOSTRESS: no RGB bands
    """
    desc_lower = [d.lower() for d in descriptions]

    # Sentinel-2: B4=R, B3=G, B2=B
    if "b4" in desc_lower and "b3" in desc_lower and "b2" in desc_lower:
        return (desc_lower.index("b4"), desc_lower.index("b3"), desc_lower.index("b2"))

    # Fallback: first 3 bands as RGB (usually works for most sensors)
    return None


# ── Visualization ─────────────────────────────────────────────────────────────


def _stretch_uint8(band: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    """Apply percentile linear stretch and convert to uint8."""
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_low, p_high = np.percentile(valid, (low_pct, high_pct))
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((band - p_low) / (p_high - p_low) * 255.0, 0, 255)
    stretched = np.nan_to_num(stretched, nan=0.0)
    return stretched.astype(np.uint8)


def _scale_lst_to_celsius(lst_k: np.ndarray) -> np.ndarray:
    """Convert Kelvin LST to Celsius, clipping to plausible range."""
    celsius = lst_k - 273.15
    return celsius


def _make_rgb_image(
    scene: SceneData,
    overlay_cloud: bool = False,
) -> np.ndarray:
    """Generate RGB image, optionally with cloud_mask overlay (red)."""
    if scene.rgb_indices:
        # True-color bands available
        ri, gi, bi = scene.rgb_indices
        r = _stretch_uint8(scene.bands[ri].data)
        g = _stretch_uint8(scene.bands[gi].data)
        b = _stretch_uint8(scene.bands[bi].data)
        rgb = np.stack([r, g, b], axis=-1)
    elif scene.source == "landsat" and len(scene.bands) >= 1:
        # Landsat: first band is ST_B10 (LST) → grayscale
        lst = scene.bands[0].data
        celsius = _scale_lst_to_celsius(lst)
        gray = _stretch_uint8(celsius)
        rgb = np.stack([gray, gray, gray], axis=-1)
    elif scene.source == "ecostress" and len(scene.bands) >= 1:
        # ECOSTRESS: first band is LST → colored
        lst = scene.bands[0].data
        celsius = _scale_lst_to_celsius(lst)
        gray = _stretch_uint8(celsius)
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        # Fallback: first band grayscale
        gray = _stretch_uint8(scene.bands[0].data)
        rgb = np.stack([gray, gray, gray], axis=-1)

    if overlay_cloud and scene.cloud_mask is not None:
        cm = scene.cloud_mask
        # Red overlay where cloud_mask < 0.5 (cloud/shadow), 40% alpha
        cloud_pixels = cm < 0.5
        overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        overlay[cloud_pixels, 0] = 255  # red
        overlay[cloud_pixels, 3] = 100  # ~40% alpha
        return overlay_alpha_blend(rgb, overlay)

    return rgb


def overlay_alpha_blend(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto an RGB image."""
    h, w = rgb.shape[:2]
    result = rgb.astype(np.float32)
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _cloud_mask_rgba(mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Convert cloud_mask to RGBA image.

    Args:
        mask: cloud_mask (1=clear, 0=cloud/shadow)
        invert: if True, show clear as transparent, cloud as red

    Returns:
        (H, W, 4) uint8 RGBA image.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if invert:
        # Cloud = red, clear = transparent
        cloud = mask < 0.5
        rgba[cloud, 0] = 255
        rgba[cloud, 3] = 180
    else:
        # Clear = green, cloud = transparent (fallback)
        clear = mask >= 0.5
        rgba[clear, 1] = 255
        rgba[clear, 3] = 128
    return rgba


def _make_histogram_figure(scene: SceneData) -> plt.Figure:
    """Create a figure with per-band histograms and cloud fraction."""
    n_bands = len(scene.bands)
    cols = min(4, n_bands)
    rows = (n_bands + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, band in enumerate(scene.bands):
        ax = axes[idx]
        data = band.data[~np.isnan(band.data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "NoData", transform=ax.transAxes, ha="center")
            ax.set_title(band.name.capitalize())
            continue
        ax.hist(data, bins=100, color="steelblue", alpha=0.7)
        ax.set_title(f"{band.name.capitalize()}  (μ={data.mean():.4f})")
        ax.set_ylabel("freq")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Hide unused subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{scene.source} — {scene.scene_id}", fontsize=14)
    fig.tight_layout()
    return fig


# ── Stats computation ─────────────────────────────────────────────────────────


def compute_stats(
    scene: SceneData,
    spec: GridSpec | None = None,
) -> dict[str, Any]:
    """Compute per-band statistics + cloud/coverage fractions.

    Coverage is delegated to ``ard_qa.compute_aoi_coverage_fraction`` so the
    smoke-validation path stays consistent with the production QA — including
    CRS-aware AOI reprojection for native-CRS sources (ECOSTRESS).
    """
    stats: dict[str, Any] = {
        "source": scene.source,
        "scene_id": scene.scene_id,
        "bands": {},
        "cloud_fraction": -1.0,
        "aoi_coverage_fraction": -1.0,
        "band_count": len(scene.bands),
    }
    for band in scene.bands:
        data = band.data[~np.isnan(band.data)]
        if data.size == 0:
            stats["bands"][band.name] = {"valid_pixels": 0}
        else:
            stats["bands"][band.name] = {
                "valid_pixels": int(data.size),
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "p1": float(np.percentile(data, 1)),
                "p5": float(np.percentile(data, 5)),
                "p25": float(np.percentile(data, 25)),
                "p50": float(np.percentile(data, 50)),
                "p75": float(np.percentile(data, 75)),
                "p95": float(np.percentile(data, 95)),
                "p99": float(np.percentile(data, 99)),
            }

    if scene.cloud_mask is not None:
        mask = scene.cloud_mask
        valid = ~np.isnan(mask)
        if valid.any():
            cloud_pixels = np.sum((mask < 0.5) & valid)
            total_valid = np.sum(valid)
            stats["cloud_fraction"] = float(cloud_pixels / total_valid)

    if spec is not None:
        stats["aoi_coverage_fraction"] = compute_aoi_coverage_fraction(scene.cog_path, spec)

    # Grid alignment check
    stats["grid"] = _check_grid(scene.cog_path)

    return stats


def _check_grid(cog_path: Path) -> dict[str, Any]:
    """Check CRS, origin, and resolution of the COG."""
    with rasterio.open(cog_path) as src:
        crs = str(src.crs)
        t = src.transform
        result = {
            "crs": crs,
            "origin_x": t.c,
            "origin_y": t.f,
            "resolution_x": abs(t.a),
            "resolution_y": abs(t.e),
            "expected_origin_x": 368000.0,
            "expected_origin_y": 5839000.0,
            "origin_match_x": abs(t.c - 368000) < 1e-3,
            "origin_match_y": abs(t.f - 5839000) < 1e-3,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }
        # ECOSTRESS has native CRS → different origin, skip check
        if "32632" in crs or "4326" in crs:
            result["native_crs_passthrough"] = True
        return result


# ── Boundary overlay ───────────────────────────────────────────────────────────


def _load_boundary_gdf() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the Berlin Landesgrenze and AOI rectangle as GeoDataFrames.

    Returns:
        (landesgrenze_gdf, aoi_gdf) — both in EPSG:25833.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    landesgrenze = gpd.read_file(data_dir / "berlin_landesgrenze.geojson")
    aoi = gpd.read_file(data_dir / "berlin_aoi.geojson")
    return landesgrenze, aoi


def _reproj_boundary(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """Reproject boundary geometry to *target_crs*.

    The result is a dissolved single polygon for clean contour plotting.
    """
    reproj = gdf.to_crs(target_crs)
    if len(reproj) > 1:
        reproj = reproj.dissolve()
    return reproj


def _add_boundary_overlay(
    ax: plt.Axes,
    boundary_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Draw Berlin Landesgrenze and AOI rectangle.
    onto an existing matplotlib axes that already shows an image with proper
    ``extent`` in CRS coordinates.
    """
    for geom in boundary_gdf.geometry:
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            if geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(
                        x, y, color="yellow", linewidth=1.2, alpha=0.9, label="Berlin Landesgrenze"
                    )
            else:
                x, y = geom.exterior.xy
                ax.plot(x, y, color="yellow", linewidth=1.2, alpha=0.9, label="Berlin Landesgrenze")

    if aoi_gdf is not None:
        for geom in aoi_gdf.geometry:
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                ax.plot(x, y, color="cyan", linewidth=1.2, alpha=0.95, linestyle="-", label="AOI")


# ── Save outputs ──────────────────────────────────────────────────────────────


def _scene_figure(
    scene: SceneData,
    boundary_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    overlay_cloud: bool = False,
    colormap: str | None = None,
    show_cbar: bool = False,
    coverage_fraction: float | None = None,
) -> plt.Figure:
    """Create a matplotlib figure of the scene with boundary overlay + axes.

    Uses the scene's native CRS extent for axis labels (metres). For
    non-EPSG:25833 scenes (ECOSTRESS), the boundary is reprojected to the
    scene's CRS and a note is added to the title.
    """
    with rasterio.open(scene.cog_path) as src:
        crs = str(src.crs)
        left, bottom, right, top = src.bounds

    # Build the display image
    if colormap and scene.source in ("landsat", "ecostress") and len(scene.bands) >= 1:
        # LST with colour map
        lst = scene.bands[0].data
        celsius = _scale_lst_to_celsius(lst)
        valid = ~np.isnan(celsius)
        if valid.any():
            vmin = np.percentile(celsius[valid], 2)
            vmax = np.percentile(celsius[valid], 98)
        else:
            vmin, vmax = -10, 50
        display_data = celsius
    elif scene.source == "sentinel2" and scene.rgb_indices:
        ri, gi, bi = scene.rgb_indices
        r = _stretch_uint8(scene.bands[ri].data)
        g = _stretch_uint8(scene.bands[gi].data)
        b = _stretch_uint8(scene.bands[bi].data)
        display_data = np.stack([r, g, b], axis=-1)

        if overlay_cloud and scene.cloud_mask is not None:
            rgb = display_data
            cm = scene.cloud_mask
            cloud_pixels = cm < 0.5
            overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
            overlay[cloud_pixels, 0] = 255
            overlay[cloud_pixels, 3] = 100
            display_data = overlay_alpha_blend(rgb, overlay)
    else:
        # Default grayscale from first band
        gray = _stretch_uint8(scene.bands[0].data)
        display_data = np.stack([gray, gray, gray], axis=-1)

    # Reproject boundary to scene CRS
    scene_boundary = _reproj_boundary(boundary_gdf, crs)
    scene_aoi = _reproj_boundary(aoi_gdf, crs)

    fig, ax = plt.subplots(figsize=(10, 8))

    if colormap and scene.source in ("landsat", "ecostress"):
        im = ax.imshow(
            display_data,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=(left, right, bottom, top),
        )
        if show_cbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("LST (°C)")
    else:
        ax.imshow(
            display_data,
            extent=(left, right, bottom, top),
        )

    _add_boundary_overlay(ax, scene_boundary, scene_aoi)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(f"{scene.source.upper()} — {scene.scene_id}  |  {crs}")
    if coverage_fraction is not None and coverage_fraction >= 0:
        ax.text(
            0.02,
            0.02,
            f"AOI coverage: {coverage_fraction:.1%}",
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3, "edgecolor": "none"},
        )
        if coverage_fraction < 0.8:
            fig.patch.set_edgecolor("red")
            fig.patch.set_linewidth(4)

    # Legend for boundary lines
    legend_handles = [
        mlines.Line2D([], [], color="yellow", linewidth=1.2, label="Berlin"),
        mlines.Line2D([], [], color="cyan", linewidth=1.2, linestyle="-", label="AOI"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig


def _save_overlays(
    scene: SceneData,
    output_dir: Path,
    boundary_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    stats: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save all visual overlays for a scene. Returns dict of output paths.

    Per-source subdirectory layout:
        ``output_dir/{landsat,sentinel2,ecostress}/<scene_id>_*.png``
    """
    out: dict[str, Path] = {}
    sid = scene.scene_id.replace("/", "_")
    source_dir = output_dir / scene.source
    source_dir.mkdir(parents=True, exist_ok=True)

    # --- RGB (raw, fast — no boundary overlay) ---
    rgb = _make_rgb_image(scene, overlay_cloud=False)
    rgb_path = source_dir / f"{sid}_rgb.png"
    plt.imsave(str(rgb_path), rgb)
    out["rgb"] = rgb_path

    # --- Scene figure with boundary overlay + axes ---
    if scene.source == "sentinel2" and scene.rgb_indices:
        fig = _scene_figure(
            scene,
            boundary_gdf,
            aoi_gdf,
            overlay_cloud=False,
            coverage_fraction=(stats or {}).get("aoi_coverage_fraction"),
        )
        georef_path = source_dir / f"{sid}_georef.png"
        fig.savefig(str(georef_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["georef"] = georef_path

    # --- LST coloured (inferno) for Landsat / ECOSTRESS ---
    if scene.source in ("landsat", "ecostress") and len(scene.bands) >= 1:
        fig = _scene_figure(
            scene,
            boundary_gdf,
            aoi_gdf,
            colormap="inferno",
            show_cbar=True,
            coverage_fraction=(stats or {}).get("aoi_coverage_fraction"),
        )
        lst_path = source_dir / f"{sid}_lst_colored.png"
        fig.savefig(str(lst_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["lst_colored"] = lst_path

    # --- RGB + cloud_mask overlay (with boundary + axes) ---
    if scene.cloud_mask is not None:
        if scene.rgb_indices:
            # S2 with cloud overlay
            fig = _scene_figure(
                scene,
                boundary_gdf,
                aoi_gdf,
                overlay_cloud=True,
                coverage_fraction=(stats or {}).get("aoi_coverage_fraction"),
            )
        else:
            # Grayscale + cloud overlay
            fig = _scene_figure(
                scene,
                boundary_gdf,
                aoi_gdf,
                coverage_fraction=(stats or {}).get("aoi_coverage_fraction"),
            )

        masked_path = source_dir / f"{sid}_rgb_masked.png"
        fig.savefig(str(masked_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["rgb_masked"] = masked_path

        # --- Cloud mask only (binary) ---
        mask_rgba = _cloud_mask_rgba(scene.cloud_mask, invert=True)
        mask_path = source_dir / f"{sid}_mask.png"
        plt.imsave(str(mask_path), mask_rgba)
        out["mask"] = mask_path

    # --- Histograms ---
    hist_fig = _make_histogram_figure(scene)
    hist_path = source_dir / f"{sid}_histograms.png"
    hist_fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
    plt.close(hist_fig)
    out["histograms"] = hist_path

    # --- Stats JSON ---
    stats = stats or compute_stats(scene)
    stats_path = source_dir / f"{sid}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    out["stats"] = stats_path

    return out


# ── Multi-source comparison ────────────────────────────────────────────────────


def _reproject_to_25833(
    cog_path: Path,
    bounds_25833: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Read a COG band and reproject it to EPSG:25833, clipping to *bounds*.

    Returns a 2D array in the AOI extent, or ``None`` on failure.
    """
    try:
        with rasterio.open(cog_path) as src:
            with WarpedVRT(
                src,
                crs="EPSG:25833",
                resampling=rasterio.enums.Resampling.bilinear,
                add_alpha=False,
            ) as vrt:
                window = vrt.window(*bounds_25833)
                if window is None:
                    return None
                array = vrt.read(1, window=window, boundless=True, fill_value=np.nan)
                return array.astype(np.float32)
    except Exception:
        return None


def _make_lst_panel(
    lst_array: np.ndarray,
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    title: str,
    vmin: float = -5,
    vmax: float = 45,
) -> plt.Axes:
    """Render an LST panel with inferno colour map + boundary-ready axes."""
    celsius = lst_array - 273.15 if np.nanmean(lst_array) > 200 else lst_array
    valid = ~np.isnan(celsius)
    if valid.any():
        vmin = float(np.percentile(celsius[valid], 2))
        vmax = float(np.percentile(celsius[valid], 98))
    im = ax.imshow(celsius, cmap="inferno", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    return im


def _make_rgb_panel(
    scene_data: SceneData,
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    title: str,
) -> None:
    """Render an RGB panel from SceneData."""
    if scene_data.rgb_indices:
        ri, gi, bi = scene_data.rgb_indices
        r = _stretch_uint8(scene_data.bands[ri].data)
        g = _stretch_uint8(scene_data.bands[gi].data)
        b = _stretch_uint8(scene_data.bands[bi].data)
        rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(rgb, extent=extent)
    else:
        ax.text(
            0.5, 0.5, "No RGB bands available", transform=ax.transAxes, ha="center", va="center"
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)


def _create_comparison_all(
    scene_data_list: list[SceneData],
    output_dir: Path,
    boundary_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    stats_by_source: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    """Create a 3-panel comparison figure (Landsat | S2 | ECOSTRESS).

    All panels share the AOI extent in EPSG:25833 and have Berlin boundary
    overlay. Missing sources result in a "Not available" placeholder.
    Output is written to ``output_dir/comparison_all.png``.
    Returns the output path or ``None`` if nothing could be rendered.
    """
    landsat_sd = next((s for s in scene_data_list if s.source == "landsat"), None)
    sentinel2_sd = next((s for s in scene_data_list if s.source == "sentinel2"), None)
    ecostress_sd = next((s for s in scene_data_list if s.source == "ecostress"), None)

    if not any([landsat_sd, sentinel2_sd, ecostress_sd]):
        return None

    # Determine shared extent from the AOI GeoJSON bounds (EPSG:25833)
    aoi_bounds = aoi_gdf.total_bounds  # [xmin, ymin, xmax, ymax]
    extent = (aoi_bounds[0], aoi_bounds[2], aoi_bounds[1], aoi_bounds[3])

    # Reproject boundary to EPSG:25833 (already in 25833, but ensure)
    scene_boundary = _reproj_boundary(boundary_gdf, "EPSG:25833")
    scene_aoi = _reproj_boundary(aoi_gdf, "EPSG:25833")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Panel 1: Landsat LST
    ax = axes[0]
    if landsat_sd and len(landsat_sd.bands) >= 1:
        lst = landsat_sd.bands[0].data
        # The Landsat scene is already in EPSG:25833; align with extent
        with rasterio.open(landsat_sd.cog_path) as src:
            ls_extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        cov = (stats_by_source or {}).get("landsat", {}).get("aoi_coverage_fraction", -1.0)
        title = f"Landsat LST ({landsat_sd.scene_id})"
        if cov >= 0:
            title += f"\nAOI coverage: {cov:.1%}"
        _make_lst_panel(lst, ax, ls_extent, title)
    else:
        ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Landsat LST")
    _add_boundary_overlay(ax, scene_boundary, scene_aoi)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Panel 2: Sentinel-2 RGB
    ax = axes[1]
    if sentinel2_sd:
        with rasterio.open(sentinel2_sd.cog_path) as src:
            s2_extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        cov = (stats_by_source or {}).get("sentinel2", {}).get("aoi_coverage_fraction", -1.0)
        title = f"Sentinel-2 RGB ({sentinel2_sd.scene_id})"
        if cov >= 0:
            title += f"\nAOI coverage: {cov:.1%}"
        _make_rgb_panel(sentinel2_sd, ax, s2_extent, title)
    else:
        ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Sentinel-2 RGB")
    _add_boundary_overlay(ax, scene_boundary, scene_aoi)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Panel 3: ECOSTRESS LST (reprojected to EPSG:25833)
    ax = axes[2]
    if ecostress_sd:
        ecs_array = _reproject_to_25833(ecostress_sd.cog_path, aoi_bounds)
        if ecs_array is not None and np.any(~np.isnan(ecs_array)):
            cov = (stats_by_source or {}).get("ecostress", {}).get("aoi_coverage_fraction", -1.0)
            title = f"ECOSTRESS LST ({ecostress_sd.scene_id})"
            if cov >= 0:
                title += f"\nAOI coverage: {cov:.1%}"
            _make_lst_panel(ecs_array, ax, extent, title)
        else:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("ECOSTRESS LST")
    else:
        ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("ECOSTRESS LST")
    _add_boundary_overlay(ax, scene_boundary, scene_aoi)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.suptitle("ARD Smoke Test — Scene Comparison (EPSG:25833)", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "comparison_all.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main(year: int = 2023) -> None:
    """Run smoke test validation for all sources."""
    print("═" * 60)
    print(f"  ARD Smoke Test Validation — {year}")
    print("═" * 60)

    # Step 0: Load boundary data
    print("\n── Loading Berlin boundary overlay ──")
    try:
        boundary_gdf, aoi_gdf = _load_boundary_gdf()
        print(f"  Berlin Landesgrenze: {len(boundary_gdf)} feature(s)")
        print(f"  AOI rectangle: {len(aoi_gdf)} feature(s)")
    except Exception as exc:
        print(f"  ❌ Failed to load boundary: {exc}")
        print("  Proceeding without boundary overlay.")
        boundary_gdf = gpd.GeoDataFrame()
        aoi_gdf = gpd.GeoDataFrame()

    # Step 0b: Load canonical grid spec (single source of truth for AOI bounds)
    spec: GridSpec | None = None
    if not boundary_gdf.empty:
        try:
            spec = get_spec()
            print(f"  Grid spec: {spec.crs}  origin=({spec.origin_x}, {spec.origin_y})")
        except FileNotFoundError:
            print("  ⚠️ AOI geojson missing — AOI coverage will be skipped.")
            spec = None

    # Step 1: Discover scenes
    print("\n── Discovering smoke scenes from GCS ──")
    scenes = discover_scenes(year=year)
    if not scenes:
        print("  No smoke scenes found. Run ard_export + ard_process first.")
        return
    print(f"  Found {len(scenes)} scene(s):")
    for s in scenes:
        print(f"    [{s['source']}] {s['scene_id']}")

    # Step 2: Download
    print("\n── Downloading COGs ──")
    temp_dir = Path(tempfile.mkdtemp(prefix="ard_smoke_"))
    print(f"  Temp dir: {temp_dir}")
    scene_data_list = download_cogs(scenes, temp_dir)
    print(f"  Downloaded {len(scene_data_list)}/{len(scenes)} COGs")

    if not scene_data_list:
        print("  No COGs available. Aborting.")
        return

    # Step 3: Analyze and visualize
    print("\n── Generating visual validation ──")
    output_dir = _OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}/")

    all_stats: list[dict[str, Any]] = []
    for sdata in scene_data_list:
        print(f"\n  [{sdata.source}] {sdata.scene_id}")

        try:
            load_scene_data(sdata)
            print(f"    Bands: {[b.name for b in sdata.bands]}")
            cloud = sdata.cloud_mask is not None
            print(f"    Cloud mask: {'✅' if cloud else '❌'}")

            stats = compute_stats(sdata, spec)
            out_paths = _save_overlays(sdata, output_dir, boundary_gdf, aoi_gdf, stats)
            for name, path in out_paths.items():
                print(f"      {name}: {path.name}")

            all_stats.append(stats)
            print(f"    Cloud fraction: {stats.get('cloud_fraction', -1):.1%}")
            print(f"    AOI coverage: {stats.get('aoi_coverage_fraction', -1):.1%}")
            print(f"    Bands: {stats['band_count']}")

        except Exception as exc:
            logger.error("Validation failed for %s: %s", sdata.scene_id, exc)
            print(f"    ❌ Error: {exc}")
            continue

    # Step 3b: Multi-source comparison panel
    print("\n── Generating comparison_all panel ──")
    if not boundary_gdf.empty:
        comp_path = _create_comparison_all(
            scene_data_list,
            output_dir,
            boundary_gdf,
            aoi_gdf,
            {s["source"]: s for s in all_stats},
        )
        if comp_path:
            print(f"    ✅ comparison_all: {comp_path.name}")
        else:
            print("    ⏭️  No scenes to compare")
    else:
        print("    ⏭️  Boundary data not available")

    # Step 4: Summary
    print("\n" + "═" * 60)
    print("  Summary")
    print("═" * 60)
    for stats in all_stats:
        sid = stats["scene_id"]
        cf = stats.get("cloud_fraction", -1)
        cf_str = f"{cf:.1%}" if cf >= 0 else "N/A"
        grid = stats.get("grid", {})
        origin_ok = grid.get("origin_match_x", False) and grid.get("origin_match_y", False)
        if origin_ok:
            origin_str = "✅"
        elif grid.get("native_crs_passthrough"):
            origin_str = "⏭️ (native CRS)"
        else:
            origin_str = "❌"
        coverage = stats.get("aoi_coverage_fraction", -1)
        coverage_str = f"{coverage:.1%}" if coverage >= 0 else "N/A"
        print(f"  [{stats['source']}] {sid}")
        print(
            f"    Cloud: {cf_str}  |  Coverage: {coverage_str}  |  "
            f"Grid origin: {origin_str}  |  Bands: {stats['band_count']}"
        )

    # Save combined report
    report_path = output_dir / "_report.json"
    with open(report_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Combined report: {report_path}")
    print(f"  Output directory: {output_dir.resolve()}/")
    print("\n  🔍 Visual QC files to review:")
    for p in sorted(output_dir.rglob("*.png")):
        print(f"    {p.relative_to(output_dir)}")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARD smoke test visual validation")
    parser.add_argument("--year", type=int, default=2023, help="Year to validate")
    args = parser.parse_args()
    main(year=args.year)
