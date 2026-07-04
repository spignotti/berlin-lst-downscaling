# /// script
# requires-python = ">=3.12"
# dependencies = ["earthaccess", "rasterio", "typer"]
# ///

"""Download one ECOSTRESS L2T granule fixture for smoke testing.

Queries NASA CMR for a granule covering Berlin on the smoke-test date
(2018-07-30), then downloads the four layer COGs (LST, cloud, water, QC)
into ``data/ecostress/fixtures/{granule_id}/``.

This is a **one-shot manual script** — run it once to seed the fixture
directory, then commit the files to Git LFS (or reference the GCS mount).

Usage
-----
    uv run python scripts/download_ecostress_fixture.py

The script requires a valid NASA Earthdata session (see
``scripts/build_manifest_ecostress.py`` for setup).

Output layout
-------------
    data/ecostress/fixtures/
        ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01/
            ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01_LST.tif
            ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01_cloud.tif
            ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01_water.tif
            ECOv002_L2T_LSTE_00372_009_32UQC_20180730T175918_0712_01_QC.tif
"""

from __future__ import annotations

from pathlib import Path

import earthaccess
import typer
from earthaccess.store import Store

# Berlin bounding box (WGS84)
BERLIN_BBOX = (13.08, 52.32, 13.76, 52.68)
SMOKE_DATE = "2018-07-30"
FIXTURE_ROOT = Path("data/ecostress/fixtures")

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


@app.command()
def main(
    date: str = typer.Option(SMOKE_DATE, "--date", help="Acquisition date (YYYY-MM-DD)"),
    bbox: tuple[float, float, float, float] | None = typer.Option(
        None, "--bbox", help="WGS84 bbox 'minx,miny,maxx,maxy' (overrides Berlin default)"
    ),
    out: Path | None = typer.Option(None, "--out", help="Fixture root directory"),  # noqa: B008
    limit: int = typer.Option(
        5, "--limit", help="Max granules to consider (downloads first valid)"
    ),
) -> None:
    out_path = FIXTURE_ROOT if out is None else out
    minx, miny, maxx, maxy = bbox if bbox else BERLIN_BBOX

    typer.echo("Searching CMR for ECO_L2T_LSTE.002 granules ...")
    typer.echo(f"  Date : {date}")
    typer.echo(f"  Bbox : ({minx}, {miny}, {maxx}, {maxy})")

    try:
        granules = earthaccess.search_data(
            short_name="ECO_L2T_LSTE",
            version="002",
            bounding_box=(minx, miny, maxx, maxy),
            temporal=(date, date),
            count=limit,
        )
    except Exception as exc:
        typer.echo(f"ERROR: CMR query failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not granules:
        typer.echo("No ECOSTRESS granules found for this date/bbox.")
        typer.echo("Hint: try --date 2018-07-30 (smoke-test date) or a different bbox.")
        raise typer.Exit(1)

    granule = granules[0]
    granule_id: str = granule["meta"]["native-id"]
    typer.echo(f"Selected granule: {granule_id}")

    out_dir = out_path / granule_id
    if out_dir.exists():
        existing = list(out_dir.glob("*.tif"))
        if existing:
            typer.secho(
                f"Fixture already exists at {out_dir}/ ({len(existing)} files).",
                fg=typer.colors.YELLOW,
            )
            typer.echo("Delete the directory to re-download.")
            raise typer.Exit(0)

    typer.echo(f"Downloading to {out_dir}/ ...")
    try:
        auth = _earthdata_login()
        downloaded = Store(auth=auth).get(
            [granule],
            local_path=str(out_dir),
            threads=4,
        )
    except Exception as exc:
        typer.echo(f"ERROR: download failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    downloaded_paths = [Path(p) for p in downloaded if p]
    typer.echo(f"  Downloaded {len(downloaded_paths)} file(s):")
    for p in downloaded_paths:
        size_mb = p.stat().st_size / 1e6
        typer.echo(f"    {p.name}  ({size_mb:.1f} MB)")

    # Ensure layer files follow the naming expected by load_ecostress_scene():
    #   {granule_id}_LST.tif  |  {granule_id}_cloud.tif  |  ...
    # LP DAAC ECO_L2T_LSTE.002 ships each layer as a separate file.
    _ensure_layer_names(out_dir, granule_id, downloaded_paths)

    typer.secho(
        f"Fixture ready at {out_dir}/\n"
        "Set the following in configs/ard/ecostress/smoke.yaml:\n"
        f"  ecostress:\n"
        f"    raw_dir: {out_path.resolve()}",
        fg=typer.colors.GREEN,
    )


def _ensure_layer_names(
    granule_dir: Path,
    granule_id: str,
    downloaded: list[Path],
) -> None:
    """Ensure downloaded files follow the {granule_id}_{layer}.tif naming.

    ECO_L2T_LSTE.002 layers are distributed as separate COG files.
    This renames any mis-named files to match the expected convention.
    """
    LAYER_SUFFIXES = ["LST", "cloud", "water", "QC"]
    for path in downloaded:
        for suffix in LAYER_SUFFIXES:
            if f"_{suffix}.tif" in path.name:
                target = granule_dir / f"{granule_id}_{suffix}.tif"
                if path.name != target.name and not target.exists():
                    path.rename(target)
                break


if __name__ == "__main__":
    _earthdata_login()
    app()
