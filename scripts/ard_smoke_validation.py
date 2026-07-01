#!/usr/bin/env python3
"""ARD smoke-test visual validation — consolidated comparison + per-scene stack.

Auto-discovers processed scenes from GCS (``ard/processed/*/{year}/``), reads
the per-scene QA stack (``*_qa.json``) directly from the bucket, and produces
a single 6-panel comparison figure plus an aggregated JSON report.

The per-scene QA stack is the source of truth for stats (cloud fraction,
city coverage, clear pixel count, etc.) — this script does **not** recompute.
The visualization only re-reads band data from the COG to render panels.

Output in ``data/tmp/ard_smoke/{ts}/``:

* ``comparison.png``  — 3×2 grid (3 sources × {data, cloud_mask})
* ``comparison.json`` — aggregated per-scene stack data from the bucket

Usage:
    uv run python scripts/ard_smoke_validation.py
    uv run python scripts/ard_smoke_validation.py --year 2023
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from berlin_lst_downscaling.data.gcs_client import download_blob

logger = logging.getLogger(__name__)

_GCS_BUCKET = "berlin-lst-data"
_OUTPUT_DIR = Path("data/tmp/ard_smoke")
KEEP_N_RUNS = 3  # number of recent smoke-validation runs to keep on disk

_SOURCES: dict[str, dict[str, Any]] = {
    "landsat": {"input_prefix": "ard/processed/landsat"},
    "sentinel2": {"input_prefix": "ard/processed/sentinel2"},
    "ecostress": {"input_prefix": "ard/processed/ecostress"},
}

_BOUNDARY_DIR = Path(__file__).resolve().parent.parent / "data" / "boundaries"
_LANDESGRENZE_FILE = _BOUNDARY_DIR / "berlin_landesgrenze.geojson"
_AOI_FILE = _BOUNDARY_DIR / "berlin_landesgrenze_2km_buffer.geojson"


# ── GCS helpers ────────────────────────────────────────────────────────────────


def _list_cog_blobs(prefix: str) -> list[str]:
    """List COG blob names under ``gs://{bucket}/{prefix}/`` (skip non-TIFs)."""
    from berlin_lst_downscaling.data.gcs_client import list_blobs

    return [
        name
        for name in list_blobs(_GCS_BUCKET, prefix=f"{prefix}/")
        if name.lower().endswith((".tif", ".tiff"))
    ]


def _read_qa_stack(cog_uri: str) -> dict[str, Any]:
    """Read the per-scene QA stack from GCS (source of truth for stats)."""
    from berlin_lst_downscaling.data.gcs_client import read_text

    qa_uri = cog_uri.replace(".tif", "_qa.json").replace(".TIF", "_qa.json")
    blob_name = qa_uri.removeprefix(f"gs://{_GCS_BUCKET}/")
    text = read_text(_GCS_BUCKET, blob_name)
    if text is None:
        return {}
    return json.loads(text)


def _discover_scenes(year: int) -> list[dict[str, str]]:
    """Discover processed COGs in the bucket for the given year."""
    scenes: list[dict[str, str]] = []
    for source, cfg in _SOURCES.items():
        for blob_name in _list_cog_blobs(f"{cfg['input_prefix']}/{year}"):
            stem = Path(blob_name).stem
            cog_uri = f"gs://{_GCS_BUCKET}/{blob_name}"
            scenes.append({"source": source, "scene_id": stem, "cog_uri": cog_uri})
    return scenes


# ── COG band readers ──────────────────────────────────────────────────────────


def _read_data_band(cog_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the first data band (LST for Landsat/ECOSTRESS, B4 for S2) and metadata."""
    with rasterio.open(cog_path) as src:
        band = src.read(1).astype(np.float32)
        meta = {
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "bounds": tuple(src.bounds),
            "descriptions": list(src.descriptions),
        }
    return band, meta


def _read_cloud_mask_band(cog_path: Path) -> np.ndarray | None:
    """Read the cloud_mask band by description, or None if absent.

    Falls back to the last band for legacy single-band sources without
    band descriptions (e.g. older ECOSTRESS exports).
    """
    with rasterio.open(cog_path) as src:
        for i in range(1, src.count + 1):
            desc = (src.descriptions[i - 1] or "").lower()
            if desc == "cloud_mask":
                return src.read(i).astype(np.float32)
        if src.count == 1:
            return None  # single-band LST without cloud info
        return src.read(src.count).astype(np.float32)


def _read_rgb_bands(cog_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read B4 (R), B3 (G), B2 (B) for Sentinel-2 RGB rendering."""
    with rasterio.open(cog_path) as src:
        descs = [d.lower() for d in src.descriptions]
        try:
            r = src.read(descs.index("b4") + 1).astype(np.float32)
            g = src.read(descs.index("b3") + 1).astype(np.float32)
            b = src.read(descs.index("b2") + 1).astype(np.float32)
        except ValueError:
            return None
    return r, g, b


# ── Reprojection ──────────────────────────────────────────────────────────────


def _reproject_band(
    cog_path: Path, band_index: int, bounds_25833: tuple[float, float, float, float],
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray | None:
    """Read a band and reproject to EPSG:25833, clipped to ``bounds_25833``."""
    try:
        with rasterio.open(cog_path) as src:
            with WarpedVRT(
                src, crs="EPSG:25833", resampling=resampling, add_alpha=False,
            ) as vrt:
                window = vrt.window(*bounds_25833)
                if window is None:
                    return None
                return vrt.read(band_index, window=window, boundless=True, fill_value=np.nan)
    except Exception:
        return None


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _stretch_percentiles(arr: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> np.ndarray:
    """Linear stretch to [0, 1] using percentiles, NaN-safe."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    vmin, vmax = np.percentile(valid, [p_lo, p_hi])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def _render_data_panel(
    ax: Axes, source: str, cog_path: Path, scene_id: str, qa: dict,
) -> None:
    """Render the data panel (LST for Landsat/ECOSTRESS, RGB for S2)."""
    title_lines = [f"{source} {scene_id}"]
    if qa.get("city_coverage_fraction") is not None:
        title_lines.append(
            f"city cov {qa['city_coverage_fraction']:.0%} · "
            f"clear {qa.get('clear_pixel_count', '?')} / {qa.get('city_total_pixels', '?')}"
        )
    if qa.get("aoi_coverage_fraction") is not None and source != "sentinel2":
        title_lines.append(f"aoi cov {qa['aoi_coverage_fraction']:.0%}")
    if qa.get("cloud_fraction") is not None and qa["cloud_fraction"] >= 0:
        title_lines.append(f"cloud {qa['cloud_fraction']:.0%}")
    if qa.get("qa_warnings"):
        title_lines.append(f"⚠ {', '.join(qa['qa_warnings'])}")
    ax.set_title("\n".join(title_lines), fontsize=9)

    if source == "sentinel2":
        rgb = _read_rgb_bands(cog_path)
        if rgb is None:
            ax.text(0.5, 0.5, "No RGB bands", ha="center", va="center", transform=ax.transAxes)
            return
        r, g, b = rgb
        with rasterio.open(cog_path) as src:
            bounds = tuple(src.bounds)
        rgb_arr = np.stack([_stretch_percentiles(b) for b in (r, g, b)], axis=-1)
        ax.imshow(rgb_arr, extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
    else:
        data, meta = _read_data_band(cog_path)
        bounds = meta["bounds"]
        # For LST in Kelvin, convert to Celsius
        if np.nanmean(data) > 200:
            display = data - 273.15
            unit = "°C"
        else:
            display = data
            unit = ""
        vmin, vmax = np.nanpercentile(display, [2, 98])
        im = ax.imshow(
            display, cmap="inferno", vmin=vmin, vmax=vmax,
            extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        )
        plt.colorbar(im, ax=ax, label=unit, fraction=0.046, pad=0.04)


def _render_cloud_panel(
    ax: Axes, source: str, cog_path: Path, scene_id: str, qa: dict,
    bounds_25833: tuple[float, float, float, float],
) -> None:
    """Render the cloud mask panel."""
    title_lines = [f"{source} cloud_mask", scene_id]
    if qa.get("clear_pixel_count") is not None:
        title_lines.append(
            f"clear {qa['clear_pixel_count']:,} / {qa.get('city_total_pixels', '?')}"
        )
    ax.set_title("\n".join(title_lines), fontsize=9)

    # ECOSTRESS native CRS needs reprojection; Landsat/S2 already in EPSG:25833
    is_ecostress = source == "ecostress"
    with rasterio.open(cog_path) as src:
        cloud_idx = None
        for i in range(1, src.count + 1):
            if (src.descriptions[i - 1] or "").lower() == "cloud_mask":
                cloud_idx = i
                break
        if cloud_idx is None:
            ax.text(
                0.5, 0.5, "no cloud band",
                ha="center", va="center", transform=ax.transAxes, color="grey",
            )
            return

        if is_ecostress:
            mask = _reproject_band(
                cog_path, cloud_idx, bounds_25833, resampling=Resampling.nearest
            )
            if mask is None:
                ax.text(0.5, 0.5, "reproj failed", ha="center", va="center", transform=ax.transAxes)
                return
            extent = (bounds_25833[0], bounds_25833[2], bounds_25833[1], bounds_25833[3])
        else:
            mask = src.read(cloud_idx).astype(np.float32)
            b = tuple(src.bounds)
            extent = (b[0], b[2], b[1], b[3])

    # Render: 1 = clear (green transparent), 0 = cloud (red)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    valid = ~np.isnan(mask)
    clear = (mask >= 0.5) & valid
    cloud = (mask < 0.5) & valid
    rgba[clear] = [0.0, 0.7, 0.0, 0.4]
    rgba[cloud] = [0.9, 0.0, 0.0, 0.5]
    ax.imshow(rgba, extent=extent)


def _add_boundary_overlay(
    ax: Axes, landesgrenze: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame,
) -> None:
    """Overlay Berlin Landesgrenze (yellow) and AOI buffer (cyan) on the panel."""
    for geom in landesgrenze.geometry:
        if geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="yellow", linewidth=1.2, alpha=0.9)
        elif geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, color="yellow", linewidth=1.2, alpha=0.9, label="Landesgrenze")
    for geom in aoi.geometry:
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, color="cyan", linewidth=0.8, alpha=0.7, label="AOI 2 km buffer")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def _prune_old_runs(output_dir: Path, keep: int) -> None:
    """Delete oldest run directories so only the most recent ``keep`` remain.

    Each validation run creates ``{output_dir}/{timestamp}/``. Over time these
    accumulate. Sorted by name (ISO timestamp) the oldest ones are removed.
    """
    if not output_dir.is_dir():
        return
    runs = sorted(p for p in output_dir.iterdir() if p.is_dir())
    if len(runs) <= keep:
        return
    import shutil

    for stale in runs[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)
        print(f"  Pruned old smoke run: {stale.name}")


def main(year: int = 2023) -> int:
    """Run smoke-test validation; produce comparison.png and comparison.json."""
    import time
    _prune_old_runs(_OUTPUT_DIR, keep=KEEP_N_RUNS)
    output_dir = _OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Smoke validation output: {output_dir}")

    # Load boundaries
    landesgrenze = gpd.read_file(_LANDESGRENZE_FILE)
    aoi = gpd.read_file(_AOI_FILE)
    bounds_25833 = tuple(aoi.total_bounds)  # (xmin, ymin, xmax, ymax)

    # Discover scenes
    scenes = _discover_scenes(year)
    if not scenes:
        print(f"No processed scenes found for {year}. Run ard_run.py smoke first.")
        return 1
    print(f"Found {len(scenes)} scene(s):")
    for s in scenes:
        print(f"  [{s['source']}] {s['scene_id']}")

    # Download COGs and load QA stacks
    temp_dir = Path(tempfile.mkdtemp(prefix="ard_smoke_"))
    print(f"Temp dir: {temp_dir}")
    local_scenes: list[dict[str, Any]] = []
    blob_prefix = f"gs://{_GCS_BUCKET}/"
    for s in scenes:
        local_path = temp_dir / Path(s["cog_uri"]).name
        blob_name = s["cog_uri"].removeprefix(blob_prefix)
        try:
            download_blob(_GCS_BUCKET, blob_name, local_path)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", s["cog_uri"], exc)
            continue
        qa_stack = _read_qa_stack(s["cog_uri"])
        local_scenes.append({**s, "cog_path": local_path, "qa_stack": qa_stack})

    # Render comparison — one row per available source, 2 columns (data + cloud).
    # Size: 1 row → 1×2, 2 rows → 2×2, 3 rows → 3×2.
    sources = ["landsat", "sentinel2", "ecostress"]
    available = [s for s in sources if any(ls["source"] == s for ls in local_scenes)]
    n_rows = max(1, len(available))
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 7 * n_rows), squeeze=False)
    for row, source in enumerate(available):
        scene = next(ls for ls in local_scenes if ls["source"] == source)
        qa = scene["qa_stack"]
        _render_data_panel(axes[row, 0], source, scene["cog_path"], scene["scene_id"], qa)
        _add_boundary_overlay(axes[row, 0], landesgrenze, aoi)
        _render_cloud_panel(
            axes[row, 1], source, scene["cog_path"], scene["scene_id"], qa, bounds_25833
        )
        _add_boundary_overlay(axes[row, 1], landesgrenze, aoi)

    fig.suptitle(
        f"ARD Smoke Test — {year} (Landesgrenze yellow, AOI 2 km buffer cyan)",
        fontsize=14,
    )
    fig.tight_layout()
    comparison_png = output_dir / "comparison.png"
    fig.savefig(str(comparison_png), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {comparison_png}")

    # Write aggregated comparison JSON
    comparison_json = {
        "year": year,
        "boundary_files": {
            "landesgrenze": str(_LANDESGRENZE_FILE.relative_to(Path.cwd())),
            "buffered_aoi": str(_AOI_FILE.relative_to(Path.cwd())),
        },
        "scenes": {},
    }
    for s in local_scenes:
        comparison_json["scenes"][s["source"]] = {
            "scene_id": s["scene_id"],
            "cog_uri": s["cog_uri"],
            "qa_stack": s["qa_stack"],
        }
    json_path = output_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_json, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()
    raise SystemExit(main(args.year))
