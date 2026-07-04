# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "rasterio", "shapely"]
# ///

"""RGB + Berlin boundary visualization for smoke-test output.

Generates ``qa/<scene_id>_rgb.png`` for each successfully processed
S2 scene, showing a true-colour composite with the Berlin Landesgrenze
boundary overlaid in black.

Usage
-----
    uv run python scripts/visualize_smoke.py data/tmp/smoke_ard_2024-06-29
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def _load_boundary(geojson_path: str):
    from shapely.geometry import shape

    with open(geojson_path) as f:
        fc = json.load(f)
    geom = shape(fc["features"][0]["geometry"])
    # Berlin boundary is a MultiPolygon — union all parts for the outline
    from shapely.ops import unary_union

    return unary_union(geom)


def _percentile_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.nanpercentile(valid, (low, high))
    return float(vmin), float(vmax)


def _linear_stretch(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return out


def visualize_scene(
    cog_uri: str,
    scene_id: str,
    source: str,
    boundary_geojson: str,
    out_png: str,
) -> None:
    """Render an RGB composite with the Berlin boundary overlaid.

    Parameters
    ----------
    cog_uri :
        Path to the scene COG (must contain B02/B03/B04 bands for S2).
    scene_id :
        Scene identifier used in the output filename.
    source :
        ``sentinel-2-l2a`` or ``landsat-c2-l2``.
    boundary_geojson :
        Path to ``berlin_landesgrenze.geojson``.
    out_png :
        Output PNG path.
    """
    boundary = _load_boundary(boundary_geojson)
    # Handle both Polygon and MultiPolygon
    if hasattr(boundary, "exterior"):
        bdy = [boundary.exterior]
    else:
        # MultiPolygon — collect all exterior rings
        bdy = [poly.exterior for poly in boundary.geoms]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if source == "sentinel-2-l2a":
        with rasterio.open(cog_uri) as src:
            if src.count < 3:
                print(f"  [WARN] {scene_id}: fewer than 3 bands, skipping RGB")
                plt.close(fig)
                return

            # Read B04 (red), B03 (green), B02 (blue)
            b04 = src.read(1).astype(np.float32)
            b03 = src.read(2).astype(np.float32)
            b02 = src.read(3).astype(np.float32)

            nodata_mask = (
                np.isnan(b04) | np.isnan(b03) | np.isnan(b02)
                if src.nodata is None
                else (b04 == src.nodata) | (b03 == src.nodata) | (b02 == src.nodata)
            )

            # Stretch each band independently
            r_vmin, r_vmax = _percentile_stretch(b04)
            g_vmin, g_vmax = _percentile_stretch(b03)
            b_vmin, b_vmax = _percentile_stretch(b02)

            r = _linear_stretch(b04, r_vmin, r_vmax)
            g = _linear_stretch(b03, g_vmin, g_vmax)
            b = _linear_stretch(b02, b_vmin, b_vmax)

            rgb = np.stack([r, g, b], axis=0).transpose(1, 2, 0)
            rgb[nodata_mask] = 0

            # Use COG bounds as extent
            extent = [
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top,
            ]

        ax.imshow(rgb, extent=extent, origin="upper", aspect="auto")

        title_bands = "B04 (Red) · B03 (Green) · B02 (Blue)"
        title_src = "Sentinel-2 L2A"

    elif source == "landsat-c2-l2":
        with rasterio.open(cog_uri) as src:
            arr = src.read(1).astype(np.float32)
            nodata_mask = np.isnan(arr) if src.nodata is None else (arr == src.nodata)
            vmin, vmax = _percentile_stretch(arr)
            grey = _linear_stretch(arr, vmin, vmax)
            grey[nodata_mask] = 0

            extent = [
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top,
            ]

        ax.imshow(grey, extent=extent, origin="upper", aspect="auto", cmap="Greys")

        title_bands = "ST (Surface Temperature)"
        title_src = "Landsat C2 L2"

    else:
        plt.close(fig)
        return

    for i, ring in enumerate(bdy):
        bx, by = ring.xy
        label = "Berlin Landesgrenze" if i == 0 else None
        ax.plot(bx, by, color="black", linewidth=3)  # thick outline
        ax.plot(bx, by, color="white", linewidth=1.5, label=label)  # white border

    ax.set_title(
        f"{scene_id}\n{title_src} — {title_bands}",
        color="white",
        fontsize=10,
        pad=12,
    )
    ax.set_xlabel("Easting (m, EPSG:25833)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Northing (m, EPSG:25833)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    legend = ax.legend(
        loc="lower right",
        facecolor="#2a2a4a",
        edgecolor="#444444",
        labelcolor="white",
        fontsize=8,
    )
    legend.get_frame().set_alpha(0.8)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out_png}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/visualize_smoke.py <output_root>")
        sys.exit(1)

    output_root = Path(sys.argv[1]).expanduser().resolve()
    boundary_geojson = (
        Path(__file__).parent.parent / "data" / "boundaries" / "berlin_landesgrenze.geojson"
    )
    qa_dir = output_root / "qa"
    qa_dir.mkdir(exist_ok=True)

    # Load ledger to find done scenes
    ledger_path = output_root / "ledger.parquet"
    if not ledger_path.exists():
        print(f"[ERROR] No ledger found at {ledger_path}")
        sys.exit(1)

    import pyarrow.parquet as pq

    tbl = pq.read_table(str(ledger_path))
    done_mask = tbl.filter(
        __import__("pyarrow.compute", fromlist=["equal"]).equal(tbl.column("status"), "done")
    )

    print(f"Visualizing {done_mask.num_rows} done scene(s) in {output_root} ...")

    for i in range(done_mask.num_rows):
        row = done_mask.slice(i, 1).to_pydict()
        scene_id = str(row["scene_id"][0])
        source = str(row["source"][0])
        path_cog = str(row["path_cog"][0]) if row["path_cog"][0] is not None else None

        if path_cog is None:
            continue

        # path_cog may be stored as absolute or relative to repo root.
        # Prefer it as-is first (absolute or relative from CWD), then
        # fall back to output-root-relative.
        for candidate in [Path(path_cog), output_root / path_cog]:
            if candidate.exists():
                cog_path = candidate
                break
        else:
            print(f"  [WARN] COG not found for {scene_id}: tried {path_cog}")
            continue

        out_png = qa_dir / f"{scene_id}_rgb.png"
        visualize_scene(
            str(cog_path),
            scene_id,
            source,
            str(boundary_geojson),
            str(out_png),
        )

    print("Done.")


if __name__ == "__main__":
    main()
