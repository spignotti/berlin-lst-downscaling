# /// script
# requires-python = ">=3.12"
# dependencies = ["earthaccess", "pyarrow", "typer"]
# ///

"""Build an ARD manifest Parquet for ECOSTRESS granules.

Queries NASA CMR (LP DAAC) via ``earthaccess`` for
``ECO_L2T_LSTE.002`` granules covering the Berlin AOI within a
date range, filters out granules whose footprint overlaps Berlin by
<10%, then writes a manifest Parquet for the ARD pipeline
(``mode=full``).

Output schema (Parquet)
-----------------------
scene_id  : str   Granule ID, e.g. ECOv002_L2T_LSTE_00372_010_33UVU_20180730T180010_0712_01
source    : str   Always "ecostress"
year      : int   Acquisition year
date      : str   Acquisition date "YYYY-MM-DD"
mgrs_tile : str   MGRS tile, e.g. 33UVU

Usage
-----
    # Interactive (reads defaults from config)
    uv run python scripts/build_manifest_ecostress.py

    # Override date range or output path
    uv run python scripts/build_manifest_ecostress.py \
        --start 2018-07-01 \
        --end   2024-12-31 \
        --bbox  13.0 52.3 13.8 52.7 \
        --out   data/ard/manifest.ecostress.parquet

Prerequisites
-------------
A valid NASA Earthdata token (``EARTHDATA_TOKEN`` env var) or
logged-in session (``earthaccess.login()``).  Run ``earthaccess.login()``
interactively once to cache credentials.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import earthaccess
import pyarrow as pa
import pyarrow.parquet as pq
import typer

# Berlin bounding box (WGS84) — loosely covers the full city
DEFAULT_BBOX = (13.08, 52.32, 13.76, 52.68)
DEFAULT_START = "2018-07-01"
DEFAULT_END = "2024-12-31"
DEFAULT_OUT = "data/ard/manifest.ecostress.parquet"

app = typer.Typer(help=__doc__)


def _earthdata_login() -> None:
    """Authenticate with NASA Earthdata.

    Checks for a cached session first; if not found, prompts interactively.
    Exits with a clear message on auth failure.
    """
    try:
        earthaccess.login()
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
    start: str = typer.Option(DEFAULT_START, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(DEFAULT_END, "--end", help="End date (inclusive, YYYY-MM-DD)"),
    bbox: tuple[float, float, float, float] | None = typer.Option(
        None, "--bbox", help="WGS84 bbox as 'minx,miny,maxx,maxy' (space/comma separated)"
    ),
    out: Path = typer.Option(DEFAULT_OUT, "--out", help="Output Parquet path"),  # noqa: B008
) -> None:
    # Resolve bbox — use default Berlin AOI when not provided
    if bbox is None:
        minx, miny, maxx, maxy = DEFAULT_BBOX
    else:
        minx, miny, maxx, maxy = bbox

    typer.echo("Querying CMR for ECO_L2T_LSTE.002 granules ...")
    typer.echo(f"  Date range : {start} – {end}")
    typer.echo(f"  Bbox       : ({minx}, {miny}, {maxx}, {maxy})")
    typer.echo(f"  Output     : {out}")

    try:
        results = earthaccess.search_data(
            short_name="ECO_L2T_LSTE",
            version="002",
            bounding_box=(minx, miny, maxx, maxy),
            temporal=(start, end),
            count=500,
        )
    except Exception as exc:
        typer.echo(f"ERROR: CMR query failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"  Found      : {len(results)} granules")

    if not results:
        typer.echo("No granules found for this date/bbox.")
        raise typer.Exit(0)

    # Build manifest rows, filtering by Berlin footprint overlap
    rows: list[dict] = []
    dropped = 0
    for granule in results:
        granule_id: str = granule["meta"]["native-id"]
        mgrs_tile = _extract_tile(granule_id)
        overlap = _footprint_overlap(granule)
        if overlap < 0.10:
            dropped += 1
            continue
        date_str = _extract_date(granule_id)
        year = int(date_str[:4]) if date_str else None

        rows.append(
            {
                "scene_id": granule_id,
                "source": "ecostress",
                "year": year,
                "date": date_str,
                "mgrs_tile": mgrs_tile,
            }
        )

    if dropped:
        typer.echo(f"  Dropped    : {dropped} (Berlin overlap < 10%)")

    # Sort by scene_id (chronologically stable)
    rows.sort(key=lambda r: r["scene_id"])

    # Write Parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    pq.write_table(table, out)
    typer.secho(f"Wrote {out}  ({table.num_rows} rows)", fg=typer.colors.GREEN)


def _extract_date(granule_id: str) -> str | None:
    """Extract YYYY-MM-DD from an ECOSTRESS granule ID.

    Granule ID pattern:
        ECOv002_L2T_LSTE_<orbit>_<scene>_<MGRS>_<YYYYMMDDThhmmss>_...

    Returns ``None`` if parsing fails.
    """
    import re

    m = re.search(r"(\d{8}T\d{6})", granule_id)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_tile(granule_id: str) -> str | None:
    """Extract MGRS tile from a granule ID (e.g. 33UVU from the 6th underscore-separated field)."""
    parts = granule_id.split("_")
    if len(parts) >= 6:
        return parts[5]  # e.g. 33UVU
    return None


def _footprint_overlap(granule) -> float:
    """Return fraction [0,1] of Berlin bbox overlapped by granule's CMR footprint."""
    BERLIN_BBOX = (13.08, 52.34, 13.76, 52.68)
    BERLIN_AREA = (BERLIN_BBOX[2] - BERLIN_BBOX[0]) * (BERLIN_BBOX[3] - BERLIN_BBOX[1])
    try:
        rects = (
            granule["umm"]
            .get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
            .get("BoundingRectangles", [])
        )
        if not rects:
            return 1.0  # permissive on missing metadata
        r = rects[0]
        iw = max(r["WestBoundingCoordinate"], BERLIN_BBOX[0])
        ie = min(r["EastBoundingCoordinate"], BERLIN_BBOX[2])
        is_ = max(r["SouthBoundingCoordinate"], BERLIN_BBOX[1])
        in_ = min(r["NorthBoundingCoordinate"], BERLIN_BBOX[3])
        if iw >= ie or is_ >= in_:
            return 0.0
        return (ie - iw) * (in_ - is_) / BERLIN_AREA
    except Exception:
        return 1.0


MANIFEST_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("source", pa.string()),
        ("year", pa.int32()),
        ("date", pa.string()),
        ("mgrs_tile", pa.string()),
    ]
)


if __name__ == "__main__":
    _earthdata_login()
    app()
