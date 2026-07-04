# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "rasterio", "shapely"]
# ///

"""6-panel QA visualisation for smoke-test output.

Generates one ``qa/<scene_id>.png`` per done scene containing:

Row 0 (Sentinel-2 L2A):
  Col 0 — True-colour RGB  (B04/B03/B02, percentile stretch)
  Col 1 — Cloud mask       (bit 1, red on white)
  Col 2 — Shadow mask      (bit 2, blue on white)

Row 1 (Landsat C2 L2):
  Col 0 — LST  (inferno, Kelvin)
  Col 1 — Cloud mask       (bit 1, red on white)
  Col 2 — Shadow mask      (bit 2, blue on white)

The Berlin Landesgrenze boundary is overlaid in lime on the RGB/LST
panels (Col 0 of each row).  Cloud (red) and shadow (blue) pixel
overlays are drawn on the RGB/LST panels as well.

Usage
-----
    uv run python scripts/visualize_smoke.py data/tmp/smoke_ard_2024-06-29
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
import rasterio.warp as rwarp
from matplotlib import pyplot as plt
from pyarrow.compute import equal
from shapely.geometry import shape
from shapely.ops import unary_union

# ── flag bit constants (must match contract.py) ─────────────────────────

_FLAG_CLOUDY = 1 << 1  # 2
_FLAG_SHADOW = 1 << 2  # 4


# ── boundary loader ────────────────────────────────────────────────────


def _load_boundary(geojson_path: str):
    with open(geojson_path) as f:
        fc = json.load(f)
    geom = unary_union(shape(fc["features"][0]["geometry"]))
    # Return as list of exteriors for consistent iteration
    if hasattr(geom, "exterior"):
        return [geom.exterior]
    return [poly.exterior for poly in geom.geoms]


# ── stretch helpers ──────────────────────────────────────────────────


def _percentile_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.nanpercentile(valid, (low, high))
    return float(vmin), float(vmax)


def _linear_stretch(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


# ── flag resampling ──────────────────────────────────────────────────


def _resample_flag(
    flag_data: np.ndarray,
    flag_profile: dict,
    dst_transform,
    dst_crs,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    """Resample flag band to destination grid (nearest-neighbour, categorical)."""
    dest = np.empty((dst_height, dst_width), dtype=flag_data.dtype)
    _, _ = rwarp.reproject(
        source=flag_data,
        src_transform=flag_profile["transform"],
        src_width=flag_profile["width"],
        src_height=flag_profile["height"],
        src_crs=flag_profile["crs"],
        destination=dest,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        resampling=rwarp.Resampling.nearest,
    )
    return dest


# ── overlay helpers ───────────────────────────────────────────────────


def _add_boundary(ax, boundary_exteriors):
    """Draw Berlin boundary in lime (thick) on a georeferenced axis.

    The axis uses CRS coordinates via ``extent=[left, right, bottom, top]``.
    The boundary rings are already in CRS coords (EPSG:25833), so they
    are plotted directly without transformation.
    """
    for ring in boundary_exteriors:
        bx, by = ring.xy
        ax.plot(bx, by, color="lime", linewidth=2.0, zorder=5)
        ax.plot(bx, by, color="black", linewidth=4.0, zorder=4)



# ── main renderer ──────────────────────────────────────────────────────


def visualize_scene(
    scene_id: str,
    source: str,
    cog_uri: str,
    flag_uri: str,
    boundary_exteriors: list,
    out_png: str,
) -> None:
    """Render a 6-panel (2×3) QA visualisation to *out_png*.

    Parameters
    ----------
    scene_id, source, cog_uri, flag_uri :
        Scene metadata and file paths.
    boundary_exteriors :
        List of shapely LinearRing objects for the Berlin boundary.
    out_png :
        Output PNG path.
    """
    with rasterio.open(cog_uri) as cog_src:
        cog_data = {i + 1: cog_src.read(i + 1) for i in range(cog_src.count)}
        cog_profile = cog_src.profile.copy()
        cog_crs = cog_src.crs
        cog_transform = cog_src.transform
        cog_bounds = cog_src.bounds
        width = cog_profile["width"]
        height = cog_profile["height"]

    # Resample flag to cog grid
    with rasterio.open(flag_uri) as flag_src:
        flag_data = flag_src.read(1)
        flag_profile = flag_src.profile.copy()

    flag_resampled = _resample_flag(
        flag_data,
        flag_profile,
        cog_transform,
        cog_crs,
        width,
        height,
    )

    cloud_mask = (flag_resampled & _FLAG_CLOUDY) != 0
    shadow_mask = (flag_resampled & _FLAG_SHADOW) != 0
    n_total = width * height
    n_cloud = int(np.sum(cloud_mask))
    n_shadow = int(np.sum(shadow_mask))
    cloud_pct = n_cloud / n_total * 100
    shadow_pct = n_shadow / n_total * 100

    # Georeference extent in CRS coords
    extent = [cog_bounds.left, cog_bounds.right, cog_bounds.bottom, cog_bounds.top]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#f2f2f2")
    BG = "#f2f2f2"

    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ── Row 0 — S2 RGB + masks ─────────────────────────────────────────
    if source == "sentinel-2-l2a":
        b04 = cog_data[1].astype(np.float32)
        b03 = cog_data[2].astype(np.float32)
        b02 = cog_data[3].astype(np.float32)

        nd = np.isnan if cog_profile.get("nodata") is None else (b04 == cog_profile["nodata"])
        rgb_nodata = nd  # all bands same nodata

        r = _linear_stretch(b04, *_percentile_stretch(b04))
        g = _linear_stretch(b03, *_percentile_stretch(b03))
        b = _linear_stretch(b02, *_percentile_stretch(b02))
        rgb = np.stack([r, g, b], axis=0).transpose(1, 2, 0)
        rgb[rgb_nodata] = 0

        ax_rgb = axes[0, 0]
        ax_rgb.imshow(rgb, extent=extent, origin="upper", aspect="auto", zorder=1)
        # Cloud = red overlay
        cloud_overlay = np.zeros((height, width, 4), dtype=np.float32)
        cloud_m = cloud_mask & ~rgb_nodata
        cloud_overlay[cloud_m] = (1.0, 0.0, 0.0, 0.80)
        ax_rgb.imshow(cloud_overlay, extent=extent, origin="upper", aspect="auto", zorder=2)
        # Shadow = blue overlay
        shadow_overlay = np.zeros((height, width, 4), dtype=np.float32)
        shadow_m = shadow_mask & ~rgb_nodata
        shadow_overlay[shadow_m] = (0.0, 0.3, 1.0, 0.75)
        ax_rgb.imshow(shadow_overlay, extent=extent, origin="upper", aspect="auto", zorder=2)
        _add_boundary(ax_rgb, boundary_exteriors)
        ax_rgb.set_title(
            "S2 L2A  "
            + scene_id
            + "\nTrue Colour  —  Cloud (red) / Shadow (blue)  —  Berlin bbox (lime)",
            fontsize=9,
            pad=5,
        )

        # Cloud mask panel
        ax_c = axes[0, 1]
        bg_cloud = np.ones((height, width, 3), dtype=np.float32) * 0.9
        cm = cloud_mask.copy().astype(np.float32)
        cm_r = np.zeros_like(cm)
        cm_r[cloud_mask] = 1.0
        cm_g = np.zeros_like(cm)
        cm_g[cloud_mask] = 0.0
        cm_b = np.zeros_like(cm)
        cm_b[cloud_mask] = 0.0
        cloud_panel = np.stack([cm_r, cm_g, cm_b], axis=0).transpose(1, 2, 0) * 0.8 + bg_cloud * 0.2
        ax_c.imshow(cloud_panel, extent=extent, origin="upper", aspect="auto")
        ax_c.set_title(f"Cloud  —  {n_cloud:,} px  ({cloud_pct:.1f}%)", fontsize=9, pad=5)

        # Shadow mask panel
        ax_s = axes[0, 2]
        bg_sh = np.ones((height, width, 3), dtype=np.float32) * 0.9
        sh_r = np.zeros_like(shadow_mask, dtype=np.float32)
        sh_r[shadow_mask] = 0.0
        sh_g = np.zeros_like(shadow_mask, dtype=np.float32)
        sh_g[shadow_mask] = 0.3
        sh_b = np.ones_like(shadow_mask, dtype=np.float32)
        sh_b[shadow_mask] = 1.0
        shadow_panel = np.stack([sh_r, sh_g, sh_b], axis=0).transpose(1, 2, 0) * 0.8 + bg_sh * 0.2
        ax_s.imshow(shadow_panel, extent=extent, origin="upper", aspect="auto")
        ax_s.set_title(f"Shadow  —  {n_shadow:,} px  ({shadow_pct:.1f}%)", fontsize=9, pad=5)

    # ── Row 1 — Landsat LST + masks ────────────────────────────────────
    if source == "landsat-c2-l2":
        lst = cog_data[1].astype(np.float32)
        lst_nodata = (
            np.isnan(lst) if cog_profile.get("nodata") is None else lst == cog_profile["nodata"]
        )
        vmin, vmax = _percentile_stretch(lst)
        lst_stretched = _linear_stretch(lst, vmin, vmax)

        ax_lst = axes[1, 0]
        ax_lst.imshow(
            lst_stretched,
            cmap="inferno",
            extent=extent,
            origin="upper",
            aspect="auto",
            vmin=0,
            vmax=1,
            zorder=1,
        )
        # Cloud = red overlay
        cloud_overlay = np.zeros((height, width, 4), dtype=np.float32)
        cloud_m = cloud_mask & ~lst_nodata
        cloud_overlay[cloud_m] = (1.0, 0.0, 0.0, 0.80)
        ax_lst.imshow(cloud_overlay, extent=extent, origin="upper", aspect="auto", zorder=2)
        # Shadow = blue overlay
        shadow_overlay = np.zeros((height, width, 4), dtype=np.float32)
        shadow_m = shadow_mask & ~lst_nodata
        shadow_overlay[shadow_m] = (0.0, 0.3, 1.0, 0.75)
        ax_lst.imshow(shadow_overlay, extent=extent, origin="upper", aspect="auto", zorder=2)
        _add_boundary(ax_lst, boundary_exteriors)
        ax_lst.set_title(
            "Landsat C2 L2  "
            + scene_id
            + "\nLST (inferno)  —  Cloud (red) / Shadow (blue)  —  Berlin bbox (lime)",
            fontsize=9,
            pad=5,
        )

        # Cloud mask panel
        ax_c = axes[1, 1]
        bg_cloud = np.ones((height, width, 3), dtype=np.float32) * 0.9
        cm_r = np.zeros_like(cloud_mask, dtype=np.float32)
        cm_r[cloud_mask] = 1.0
        cm_g = np.zeros_like(cloud_mask, dtype=np.float32)
        cm_g[cloud_mask] = 0.0
        cm_b = np.zeros_like(cloud_mask, dtype=np.float32)
        cm_b[cloud_mask] = 0.0
        cloud_panel = np.stack([cm_r, cm_g, cm_b], axis=0).transpose(1, 2, 0) * 0.8 + bg_cloud * 0.2
        ax_c.imshow(cloud_panel, extent=extent, origin="upper", aspect="auto")
        ax_c.set_title(f"Cloud  —  {n_cloud:,} px  ({cloud_pct:.1f}%)", fontsize=9, pad=5)

        # Shadow mask panel
        ax_s = axes[1, 2]
        bg_sh = np.ones((height, width, 3), dtype=np.float32) * 0.9
        sh_r = np.zeros_like(shadow_mask, dtype=np.float32)
        sh_r[shadow_mask] = 0.0
        sh_g = np.zeros_like(shadow_mask, dtype=np.float32)
        sh_g[shadow_mask] = 0.3
        sh_b = np.ones_like(shadow_mask, dtype=np.float32)
        sh_b[shadow_mask] = 1.0
        shadow_panel = np.stack([sh_r, sh_g, sh_b], axis=0).transpose(1, 2, 0) * 0.8 + bg_sh * 0.2
        ax_s.imshow(shadow_panel, extent=extent, origin="upper", aspect="auto")
        ax_s.set_title(f"Shadow  —  {n_shadow:,} px  ({shadow_pct:.1f}%)", fontsize=9, pad=5)

    plt.subplots_adjust(hspace=0.12, wspace=0.05)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out_png}")


# ── entry point ────────────────────────────────────────────────────────


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

    ledger_path = output_root / "ledger.parquet"
    if not ledger_path.exists():
        print(f"[ERROR] No ledger found at {ledger_path}")
        sys.exit(1)

    tbl = pq.read_table(str(ledger_path))
    done_mask = tbl.filter(equal(tbl.column("status"), "done"))

    boundary_exteriors = _load_boundary(str(boundary_geojson))

    print(f"Visualising {done_mask.num_rows} done scene(s) in {output_root} ...")

    for i in range(done_mask.num_rows):
        row = done_mask.slice(i, 1).to_pydict()
        scene_id = str(row["scene_id"][0])
        source = str(row["source"][0])
        path_cog = str(row["path_cog"][0]) if row["path_cog"][0] is not None else None

        if path_cog is None:
            continue

        # Resolve COG path
        for candidate in [Path(path_cog), output_root / path_cog]:
            if candidate.exists():
                cog_path = candidate
                break
        else:
            print(f"  [WARN] COG not found for {scene_id}")
            continue

        # Flag COG is alongside the main COG
        flag_path = cog_path.with_name(cog_path.name.replace(".tif", ".flag.tif"))
        if not flag_path.exists():
            print(f"  [WARN] Flag COG not found for {scene_id}: {flag_path}")
            continue

        out_png = qa_dir / f"{scene_id}.png"
        visualize_scene(
            scene_id=scene_id,
            source=source,
            cog_uri=str(cog_path),
            flag_uri=str(flag_path),
            boundary_exteriors=boundary_exteriors,
            out_png=str(out_png),
        )

    print("Done.")


if __name__ == "__main__":
    main()
