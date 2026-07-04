# /// script
# requires-python = ">=3.12"
# dependencies = ["earthaccess", "rasterio", "typer"]
# ///

"""Download ECOSTRESS L2T granules for a list of tile×date pairs.

Queries NASA CMR for each tile×date combination, verifies the granule's
footprint overlaps Berlin by ≥10%, then downloads the four layer COGs
(LST, cloud, water, QC) into ``{out}/{granule_id}/``.

This is a **bulk download script** — run once per tile×date list to seed
the fixture directory.  The scene-selection task (which tiles×dates to
download) is a separate Szenen-Selektion step.

Usage
-----
    # Single tile, single date (smoke default)
    uv run python scripts/download_ecostress_fixture.py --tile 33UVU --date 2018-07-30

    # Multiple tiles
    uv run python scripts/download_ecostress_fixture.py \
        --tile 33UVU --date 2018-07-30 \
        --tile 33UVD --date 2018-08-15

    # Custom output root
    uv run python scripts/download_ecostress_fixture.py \
        --tile 33UVU --date 2018-07-30 \
        --out data/ecostress/fixtures

Output layout
-------------
    {out}/
        ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01/
            ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01_LST.tif
            ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01_cloud.tif
            ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01_water.tif
            ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01_QC.tif
"""

from __future__ import annotations

from pathlib import Path

import earthaccess
import typer
from earthaccess.store import Store

# Berlin bounding box (WGS84) — used for footprint intersection checks
BERLIN_BBOX = (13.08, 52.34, 13.76, 52.68)
BERLIN_AREA = (BERLIN_BBOX[2] - BERLIN_BBOX[0]) * (BERLIN_BBOX[3] - BERLIN_BBOX[1])
MIN_OVERLAP_FRAC = 0.10  # granule must cover ≥10% of Berlin bbox

app = typer.Typer(help=__doc__)


def _earthdata_login() -> earthaccess.auth.Auth:
    """Authenticate with NASA Earthdata.  Exits with a clear message on failure."""
    try:
        return earthaccess.login()
    except Exception as exc:
        typer.echo(
            "ERROR: NASA Earthdata login failed.\n"
            "  Run: python -c 'import earthaccess; earthaccess.login()'\n"
            "  Then re-run this script.\n"
            f"  Detail: {exc}",
            err=True,
        )
        raise typer.Exit(1) from exc


def _bbox_overlap_frac(granule_west: float, granule_east: float,
                        granule_south: float, granule_north: float) -> float:
    """Return fraction of Berlin bbox covered by the granule's footprint [0, 1]."""
    iw = max(granule_west, BERLIN_BBOX[0])
    ie = min(granule_east, BERLIN_BBOX[2])
    is_ = max(granule_south, BERLIN_BBOX[1])
    in_ = min(granule_north, BERLIN_BBOX[3])
    if iw >= ie or is_ >= in_:
        return 0.0
    intersection_area = (ie - iw) * (in_ - is_)
    return intersection_area / BERLIN_AREA


@app.command()
def main(
    tile: list[str] = typer.Option(  # noqa: B008
        ..., "--tile", help="MGRS tile to download (e.g. 33UVU). Repeatable."
    ),
    date: list[str] = typer.Option(  # noqa: B008
        ..., "--date", help="Acquisition date YYYY-MM-DD. Repeatable, matched 1:1 with --tile."
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("data/ecostress/fixtures"), "--out",
        help="Output root directory."
    ),
) -> None:
    if len(tile) != len(date):
        typer.echo(
            f"ERROR: --tile ({len(tile)}) and --date ({len(date)}) counts must match.",
            err=True,
        )
        raise typer.Exit(1)

    auth = _earthdata_login()
    total_downloaded = 0
    total_skipped = 0

    for tile_id, date_str in zip(tile, date, strict=True):
        typer.echo(f"\nProcessing tile={tile_id} date={date_str} ...")

        # 1. CMR query via granule_name wildcard
        granules = _cmr_search(tile_id, date_str)
        if not granules:
            typer.echo(
                f"  ERROR: No ECO_L2T_LSTE.002 granules for tile={tile_id} date={date_str}.",
                err=True,
            )
            raise typer.Exit(1)

        granule = granules[0]
        granule_id: str = granule["meta"]["native-id"]
        typer.echo(f"  Found: {granule_id}")

        # 2. Footprint validation
        overlap = _footprint_overlap(granule)
        typer.echo(f"  Berlin overlap: {overlap:.1%}")
        if overlap < MIN_OVERLAP_FRAC:
            typer.echo(
                f"  ERROR: Granule overlaps Berlin by {overlap:.1%}"
                f" (< {MIN_OVERLAP_FRAC:.0%} threshold).",
                err=True,
            )
            raise typer.Exit(1)

        # 3. Skip or download
        out_dir = out / granule_id
        if _fixture_complete(out_dir):
            typer.secho(f"  Skipping (already complete): {out_dir}/", fg=typer.colors.YELLOW)
            total_skipped += 1
            continue

        typer.echo(f"  Downloading to {out_dir}/ ...")
        _download_granule(granule, out_dir, auth)
        total_downloaded += 1

    typer.secho(
        f"\nDone — {total_downloaded} downloaded, {total_skipped} skipped.",
        fg=typer.colors.GREEN,
    )


def _cmr_search(tile_id: str, date: str) -> list:
    """Query CMR for granules matching tile_id + date."""
    # CMR granule names use compact date: YYYYMMDDTHHMMSS (no dashes)
    date_compact = date.replace("-", "")
    try:
        return earthaccess.search_data(
            short_name="ECO_L2T_LSTE",
            version="002",
            granule_name=f"ECOv002_L2T_LSTE_*_{tile_id}_{date_compact}T*",
            count=5,
        )
    except Exception as exc:
        typer.echo(f"  ERROR: CMR query failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _footprint_overlap(granule) -> float:
    """Return Berlin-overlap fraction of the granule's CMR footprint."""
    try:
        rects = (
            granule["umm"]
            .get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
            .get("BoundingRectangles", [])
        )
        if not rects:
            typer.echo(
                "  WARNING: No bounding rectangles in CMR response"
                " — skipping overlap check.",
            )
            return 1.0  # permissive on missing metadata
        r = rects[0]
        return _bbox_overlap_frac(
            r["WestBoundingCoordinate"],
            r["EastBoundingCoordinate"],
            r["SouthBoundingCoordinate"],
            r["NorthBoundingCoordinate"],
        )
    except Exception as exc:
        typer.echo(f"  WARNING: Could not parse footprint ({exc}) — skipping overlap check.")
        return 1.0


def _fixture_complete(out_dir: Path) -> bool:
    """Return True if all 4 required layer COGs are present."""
    required = ["LST", "cloud", "water", "QC"]
    granule_id = out_dir.name
    return all((out_dir / f"{granule_id}_{layer}.tif").exists() for layer in required)


def _download_granule(granule, out_dir: Path, auth) -> None:
    """Download one granule's 4 layer COGs into out_dir."""
    try:
        downloaded = Store(auth=auth).get(
            [granule],
            local_path=str(out_dir),
            threads=4,
        )
    except Exception as exc:
        typer.echo(f"  ERROR: download failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    downloaded_paths = [Path(p) for p in downloaded if p]
    typer.echo(f"  Downloaded {len(downloaded_paths)} file(s):")
    for p in downloaded_paths:
        size_mb = p.stat().st_size / 1e6
        typer.echo(f"    {p.name}  ({size_mb:.1f} MB)")

    # Ensure layer files follow the naming expected by load_ecostress_scene():
    #   {granule_id}_LST.tif  |  {granule_id}_cloud.tif  |  ...
    _ensure_layer_names(out_dir, out_dir.name, downloaded_paths)


def _ensure_layer_names(granule_dir: Path, granule_id: str, downloaded: list[Path]) -> None:
    """Ensure downloaded files follow the {granule_id}_{layer}.tif naming."""
    for path in downloaded:
        for suffix in ("LST", "cloud", "water", "QC"):
            if f"_{suffix}.tif" in path.name:
                target = granule_dir / f"{granule_id}_{suffix}.tif"
                if path.name != target.name and not target.exists():
                    path.rename(target)
                break


if __name__ == "__main__":
    app()
