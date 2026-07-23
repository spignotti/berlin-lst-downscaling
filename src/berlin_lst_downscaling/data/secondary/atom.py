"""INSPIRE ATOM feed parser for Geoportal Berlin tile datasets.

Parses the XML feed to discover download assets, extract tile coordinates
from filenames, and filter by AOI intersection.  Used by DGM and LoD2
adapters to build deterministic asset manifests without hardcoded URLs.

The parser is intentionally minimal — it reads only the elements needed
for asset discovery (``entry/link[@rel='section']``) and ignores the
rest of the ATOM/INSPIRE metadata.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

from odc.geo.geobox import GeoBox

# ── asset record ─────────────────────────────────────────────────────

@dataclass
class AtomAsset:
    """One downloadable tile from an ATOM feed."""

    url: str
    filename: str
    title: str
    easting: int  # tile origin easting (EPSG:25833)
    northing: int  # tile origin northing (EPSG:25833)
    checksum: str = ""  # SHA-256, filled on download
    byte_count: int = 0
    local_path: str | None = None  # set after download

@dataclass
class AtomManifest:
    """Immutable catalog of all assets in an ATOM feed, optionally AOI-filtered."""

    feed_url: str
    feed_updated: str
    feed_title: str
    assets: list[AtomAsset] = field(default_factory=list)

# ── coordinate extraction patterns ───────────────────────────────────

# DGM1: DGM1_{E}_{N}.zip  (e.g. DGM1_368_5808.zip)
_DGM_RE = re.compile(r"DGM1_(\d{3})_(\d{4})\.zip$", re.IGNORECASE)

# LoD2: LoD2_{E}_{N}.zip  (e.g. LoD2_371_5809.zip)
_LOD2_RE = re.compile(r"LoD2_(\d{3})_(\d{4})\.zip$", re.IGNORECASE)

def _extract_coords(filename: str, pattern: re.Pattern[str]) -> tuple[int, int] | None:
    """Extract (easting, northing) tile origin from a filename."""
    m = pattern.search(filename)
    if not m:
        return None
    return int(m.group(1)) * 1000, int(m.group(2)) * 1000

# ── ATOM XML parsing ─────────────────────────────────────────────────

_ATOM_NS = "{http://www.w3.org/2005/Atom}"

def parse_atom_feed(
    feed_url: str,
    coord_pattern: re.Pattern[str],
    aoi_grid: GeoBox | None = None,
    timeout: int = 30,
) -> AtomManifest:
    """Parse an INSPIRE ATOM feed and return an asset manifest.

    Parameters
    ----------
    feed_url :
        URL of the ATOM feed XML.
    coord_pattern :
        Regex to extract tile coordinates from asset filenames.
    aoi_grid :
        If given, only include assets whose tile intersects this grid.
    timeout :
        HTTP timeout in seconds for fetching the feed.
    """
    with urlopen(feed_url, timeout=timeout) as resp:  # noqa: S310
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)  # noqa: S314

    feed_title = _text(root, f"{_ATOM_NS}title", "")
    feed_updated = _text(root, f"{_ATOM_NS}updated", "")

    assets: list[AtomAsset] = []

    for entry in root.findall(f"{_ATOM_NS}entry"):
        for link in entry.findall(f"{_ATOM_NS}link"):
            rel = link.get("rel", "")
            href = link.get("href", "")
            title = link.get("title", "")
            if rel != "section" or not href:
                continue

            filename = href.rsplit("/", 1)[-1]
            coords = _extract_coords(filename, coord_pattern)
            if coords is None:
                continue

            easting, northing = coords

            # AOI filter: skip tiles that don't intersect the grid
            if aoi_grid is not None:
                tile_e = easting + 2000  # DGM tiles are 2 km
                tile_s = northing - 2000
                grid_left = aoi_grid.transform.xoff
                grid_right = grid_left + aoi_grid.shape.x * abs(aoi_grid.transform.a)
                grid_top = aoi_grid.transform.yoff
                grid_bottom = grid_top - aoi_grid.shape.y * abs(aoi_grid.transform.e)

                # No intersection if tile is completely outside grid
                if (
                    easting >= grid_right
                    or tile_e <= grid_left
                    or northing <= grid_bottom
                    or tile_s >= grid_top
                ):
                    continue

            assets.append(
                AtomAsset(
                    url=href,
                    filename=filename,
                    title=title,
                    easting=easting,
                    northing=northing,
                )
            )

    # Sort by northing (south-to-north), then easting (west-to-east)
    assets.sort(key=lambda a: (a.northing, a.easting))

    return AtomManifest(
        feed_url=feed_url,
        feed_updated=feed_updated,
        feed_title=feed_title,
        assets=assets,
    )

def _text(element: ET.Element, tag: str, default: str = "") -> str:
    """Extract text content from an XML element."""
    child = element.find(tag)
    return child.text.strip() if child is not None and child.text else default

# ── manifest persistence ─────────────────────────────────────────────

def save_manifest_json(manifest: AtomManifest, path: str) -> None:
    """Save manifest as JSON for reproducibility."""
    import json

    data = {
        "feed_url": manifest.feed_url,
        "feed_updated": manifest.feed_updated,
        "feed_title": manifest.feed_title,
        "asset_count": len(manifest.assets),
        "assets": [
            {
                "url": a.url,
                "filename": a.filename,
                "easting": a.easting,
                "northing": a.northing,
                "checksum": a.checksum,
                "byte_count": a.byte_count,
            }
            for a in manifest.assets
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))

__all__ = [
    "AtomAsset",
    "AtomManifest",
    "parse_atom_feed",
    "save_manifest_json",
]