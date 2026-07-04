# /// script
# requires-python = ">=3.12"
# dependencies = ["earthaccess", "rasterio", "typer"]
# ///

"""Stage ECOSTRESS L2T granules into local or GCS storage.

Downloads granules from NASA Earthdata (S3 origin, via earthaccess) and
uploads them to a configurable staging URI.

This script is the **stage-seeder**: it populates a stage directory with
raw L2T COG files that the ARD pipeline consumes.  The stage is ephemeral —
the pipeline deletes it after processing.

Usage
-----
    # Stage locally (smoke test)
    uv run python scripts/download_ecostress_fixture.py \
        --tile 33UUU --date 2018-07-30 \
        --stage-dir data/tmp/ecostress_stage/abc123

    # Stage to GCS (cloud run)
    uv run python scripts/download_ecostress_fixture.py \
        --tile 33UUU --date 2018-07-30 \
        --stage-dir gs://berlin-lst-data/_staging/ecostress/abc123

Output layout
-------------
    {stage_dir}/
        ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01/
            ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01_LST.tif
            ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01_cloud.tif
            ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01_water.tif
            ECOv002_L2T_LSTE_00373_003_33UUU_20180730T193555_0712_01_QC.tif
            ...

Layer files (LST, cloud, water, QC) are renamed to follow the
``{granule_id}_{layer}.tif`` naming convention that
:func:`berlin_lst_downscaling.data.acquisition.ecostress.load_ecostress_scene`
expects.
"""

from __future__ import annotations

from pathlib import Path

import earthaccess
import typer
from earthaccess.store import Store
from tenacity import retry, stop_after_attempt, wait_exponential

from berlin_lst_downscaling.data.io.staging import StageManager

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


def _bbox_overlap_frac(
    granule_west: float,
    granule_east: float,
    granule_south: float,
    granule_north: float,
) -> float:
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
        ..., "--tile", help="MGRS tile to download (e.g. 33UUU). Repeatable."
    ),
    date: list[str] = typer.Option(  # noqa: B008
        ..., "--date", help="Acquisition date YYYY-MM-DD. Repeatable, matched 1:1 with --tile."
    ),
    stage_dir: str = typer.Option(
        ..., "--stage-dir",
        help=(
            "Stage root URI. Local POSIX path, gs://bucket/path, or ~/.mnt/path (FUSE). "
            "Granules are written to {stage_dir}/{run_id}/{granule_id}/."
        ),
    ),
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="Unique run identifier (auto-generated if not set).",
    ),
) -> None:
    if len(tile) != len(date):
        typer.echo(
            f"ERROR: --tile ({len(tile)}) and --date ({len(date)}) counts must match.",
            err=True,
        )
        raise typer.Exit(1)

    auth = _earthdata_login()

    # Build stage manager
    stage = StageManager(uri=stage_dir, run_id=run_id, persist=True)
    typer.echo(f"Stage URI  : {stage.uri}")
    typer.echo(f"Run ID     : {stage.run_id}")
    typer.echo(f"Scheme     : {stage.uri.scheme}")

    total_staged = 0
    total_skipped = 0
    total_failed = 0

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
                f"  SKIP: Granule overlaps Berlin by {overlap:.1%}"
                f" (< {MIN_OVERLAP_FRAC:.0%} threshold).",
            )
            total_skipped += 1
            continue

        # 3. Download from Earthdata S3 to a local tmp dir (earthaccess limitation)
        tmp_dir = Path("/var/folders/7r/mzy_klsd1mn9xrh7fln8mt2m0000gn/T")
        typer.echo("  Downloading to local tmp ...")
        try:
            local_paths = _download_to_tmp(granule, tmp_dir, auth)
        except Exception as exc:
            typer.echo(
                f"  ERROR: Download failed (after retries): {exc}",
                err=True,
            )
            total_failed += 1
            continue

        # 4. Upload into stage
        typer.echo(f"  Staging to {stage.uri} ...")
        try:
            for local_path in local_paths:
                # Determine layer from filename suffix
                key = f"{granule_id}/{local_path.name}"
                stage.put(local_path, key)
        except Exception as exc:
            typer.echo(f"  ERROR: Stage upload failed: {exc}", err=True)
            total_failed += 1
            # Clean up downloaded files before continuing
            for local_path in local_paths:
                local_path.unlink(missing_ok=True)
            continue

        total_staged += 1

        # Clean up local tmp
        for local_path in local_paths:
            local_path.unlink(missing_ok=True)

    typer.echo("\nSummary:")
    typer.echo(f"  Staged  : {total_staged}")
    typer.echo(f"  Skipped : {total_skipped} (footprint < {MIN_OVERLAP_FRAC:.0%})")
    typer.echo(f"  Failed  : {total_failed}")

    if total_staged == 0:
        typer.secho("No granules staged — check errors above.", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"\nDone — {total_staged} granule(s) staged to {stage.uri}",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        "  NOTE: Run the pipeline, then call stage.cleanup() to delete staged files."
        if not stage._persist
        else "  [persist=True] Stage was NOT cleaned up.",
    )


def _cmr_search(tile_id: str, date: str) -> list:
    """Query CMR for granules matching tile_id + date."""
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, max=30),
    reraise=True,
)
def _download_to_tmp(granule, tmp_dir: Path, auth) -> list[Path]:
    """Download one granule's 4 layer COGs into a local tmp directory.

    Returns the list of downloaded Paths (local files).
    earthaccess.Store.get() always downloads locally — this is the only
    supported path from NASA's S3 endpoint.
    """
    try:
        downloaded = Store(auth=auth).get(
            [granule],
            local_path=str(tmp_dir),
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

    return downloaded_paths


if __name__ == "__main__":
    app()
