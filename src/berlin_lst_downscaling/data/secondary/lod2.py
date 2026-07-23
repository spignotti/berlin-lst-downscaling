"""LoD2 building morphology source adapter for the secondary pipeline.

Official Geoportal Berlin LoD2 CityGML tiles (INSPIRE ATOM feed):
``https://gdi.berlin.de/data/a_lod2/atom/0.atom``

Each tile is a ZIP containing a CityGML v2.0 file with 3D building models.
Buildings are parsed to extract footprints and ``measuredHeight``, then
rasterized to the canonical 10 m grid as three morphology bands:

- ``building_height_mean`` — mean building height per 10 m cell (metres)
- ``building_height_std`` — standard deviation of heights within the cell
- ``building_coverage_ratio`` — fraction of cell area covered by footprints

Processing
----------
1. Parse the ATOM feed to discover tile assets intersecting the AOI.
2. Download each tile ZIP to raw storage via ``download_to_raw``.
3. Stream-parse CityGML XML, extracting ``Building`` elements with
   ``measuredHeight`` and ``GroundSurface`` polygons.
4. Rasterize footprints at 10 m resolution, accumulate per-cell
   height statistics (sum, sum², count, area) across all tiles.
5. Compute final morphology bands from accumulated statistics.
6. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts.

Qualification
-------------
The current ATOM feed (2026-03-26) is a future source for 2017–2025
scenes.  Historical vintages must be qualified before production use.
See ``docs/data-sources-and-contracts.md``.
"""

from __future__ import annotations

import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

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

_FEED_URL = "https://gdi.berlin.de/data/a_lod2/atom/0.atom"
_LICENSE = "dl-de/zero-2.0"
_LOD2_RE = re.compile(r"LoD2_(\d{3})_(\d{4})\.zip$", re.IGNORECASE)

# CityGML namespaces — we detect version from the document itself
_CITYGML_NS_MAP = {
    1: {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/1.0",
        "core": "http://www.opengis.net/citygml/core/1.0",
    },
    2: {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/2.0",
        "core": "http://www.opengis.net/citygml/core/2.0",
    },
}

# ── data classes ──────────────────────────────────────────────────────

@dataclass
class ParsedBuilding:
    """A single building extracted from CityGML."""

    building_id: str
    footprint: Polygon | MultiPolygon | None
    measured_height: float | None  # metres above ground

# ── contract ───────────────────────────────────────────────────────────

_CONFIG_HASH_PREFIX = "lod2_morphology:v1:"

def contract_for_lod2_morphology() -> Contract:
    """Return the output Contract for LoD2 morphology COGs (4 bands)."""
    return Contract(
        source="lod2_morphology",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="building_height_mean",
                dtype="float32",
                nodata=float("nan"),
                description="Mean building height in metres above ground",
                unit="m",
                valid_range=(0.0, 200.0),
            ),
            BandSpec(
                name="building_height_std",
                dtype="float32",
                nodata=float("nan"),
                description="Standard deviation of building heights within cell",
                unit="m",
                valid_range=(0.0, 100.0),
            ),
            BandSpec(
                name="building_coverage_ratio",
                dtype="float32",
                nodata=float("nan"),
                description="Fraction of cell area covered by building footprints [0, 1]",
                unit="",
                valid_range=(-0.01, 1.01),
            ),
            BandSpec(
                name="building_height_max",
                dtype="float32",
                nodata=float("nan"),
                description="Maximum building height in cell (for DSM derivation)",
                unit="m",
                valid_range=(0.0, 200.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=2,
        flag_mode="none",
    )

def config_hash_for_vintage(vintage: int) -> str:
    """Return a stable config hash for a given vintage."""
    raw = f"{_CONFIG_HASH_PREFIX}{vintage}"
    return sha256(raw.encode()).hexdigest()[:12]

# ── CityGML parsing ──────────────────────────────────────────────────

def _detect_citygml_version(content: bytes) -> int:
    """Detect CityGML version from document content."""
    if b"citygml/building/2.0" in content:
        return 2
    if b"citygml/building/1.0" in content:
        return 1
    # Try root element tag
    if b"core:CityModel" in content:
        # Check for any 2.0 namespace
        if b"/1.0" in content:
            return 1
        if b"/2.0" in content:
            return 2
    return 1  # default to 1.0

def _parse_gml_ring(coords_text: str, srs_dim: int = 2) -> list[tuple[float, float]]:
    """Parse a GML posList string into (x, y) tuples.

    Handles both 2D (x y) and 3D (x y z) coordinate lists.
    """
    tokens = coords_text.strip().split()
    coords = []
    step = srs_dim
    for i in range(0, len(tokens) - step + 1, step):
        try:
            x, y = float(tokens[i]), float(tokens[i + 1])
            coords.append((x, y))
        except (ValueError, IndexError):
            continue
    return coords

def _parse_polygon(polygon_elem: ET.Element, ns: dict[str, str]) -> Polygon | None:
    """Parse a gml:Polygon element into a Shapely Polygon."""
    gml_ns = ns["gml"]

    # Find exterior ring posList
    ext_ring = None
    for pos_list in polygon_elem.iter(f"{{{gml_ns}}}posList"):
        # Check if this posList is inside an exterior ring
        parent = polygon_elem.find(f".//{{{gml_ns}}}exterior//{{{gml_ns}}}posList")
        if parent is pos_list or parent is not None:
            ext_ring = pos_list
            break
    if ext_ring is None:
        # Fallback: find any posList inside the polygon
        ext_ring = polygon_elem.find(f".//{{{gml_ns}}}posList")
    if ext_ring is None or not ext_ring.text:
        return None

    srs_dim = int(ext_ring.get("srsDimension", "2"))
    ext_coords = _parse_gml_ring(ext_ring.text, srs_dim)
    if len(ext_coords) < 4:
        return None

    # Interior rings (holes)
    holes = []
    for interior in polygon_elem.findall(f".//{{{gml_ns}}}interior"):
        pos_list = interior.find(f".//{{{gml_ns}}}posList")
        if pos_list is not None and pos_list.text:
            srs_dim_i = int(pos_list.get("srsDimension", "2"))
            hole_coords = _parse_gml_ring(pos_list.text, srs_dim_i)
            if len(hole_coords) >= 4:
                holes.append(hole_coords)

    try:
        poly = Polygon(ext_coords, holes)
        if poly.is_valid and not poly.is_empty:
            return poly
        poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty and isinstance(poly, Polygon):
            return poly
    except Exception:  # noqa: S110
        pass
    return None

def _parse_buildings_from_tile(
    zip_path: Path,
    asset: AtomAsset,
) -> list[ParsedBuilding]:
    """Parse all buildings from a CityGML tile ZIP."""
    buildings: list[ParsedBuilding] = []

    with zipfile.ZipFile(zip_path) as z:
        gml_names = [n for n in z.namelist() if n.lower().endswith((".gml", ".xml"))]
        if not gml_names:
            return buildings

        for gml_name in gml_names:
            with z.open(gml_name) as f:
                raw = f.read()

            # Detect version from actual content
            version = _detect_citygml_version(raw)
            ns = _CITYGML_NS_MAP[version]

            root = ET.fromstring(raw)  # noqa: S314

            # Find all Building elements (not BuildingPart — those are children)
            for building_elem in root.iter(f"{{{ns['bldg']}}}Building"):
                building = _extract_building(building_elem, ns)
                if building is not None:
                    buildings.append(building)

    return buildings

def _extract_building(
    building_elem: ET.Element,
    ns: dict[str, str],
) -> ParsedBuilding | None:
    """Extract footprint and height from a Building element."""
    bid = building_elem.get(
        f"{{{ns.get('gml', '')}}}id",
        building_elem.get("gml:id", ""),
    )

    # measuredHeight — try both with and without namespace prefix
    height = None
    for tag in [
        f"{{{ns['bldg']}}}measuredHeight",
        "bldg:measuredHeight",
    ]:
        height_elem = building_elem.find(tag)
        if height_elem is not None and height_elem.text:
            try:
                height = float(height_elem.text)
            except ValueError:
                pass
            break

    if height is None or height <= 0:
        return None

    # Ground surfaces → footprint
    ground_surfaces: list[Polygon] = []
    # CityGML 1.0: bldg:boundedBy/bldg:GroundSurface/bldg:lod2MultiSurface/...
    # CityGML 2.0: bldg:boundedBy/bldg:GroundSurface/bldg:lod2MultiSurface/...
    for gs_elem in building_elem.iter(f"{{{ns['bldg']}}}GroundSurface"):
        for poly_elem in gs_elem.iter(f"{{{ns['gml']}}}Polygon"):
            poly = _parse_polygon(poly_elem, ns)
            if poly is not None:
                ground_surfaces.append(poly)

    if not ground_surfaces:
        return None

    try:
        merged = unary_union(ground_surfaces)
        if isinstance(merged, (Polygon, MultiPolygon)):
            footprint = merged
        else:
            return None
    except Exception:
        return None

    if footprint.is_empty or not footprint.is_valid:
        return None

    return ParsedBuilding(building_id=bid, footprint=footprint, measured_height=height)

# ── rasterization ─────────────────────────────────────────────────────

def _accumulate_buildings(
    buildings: list[ParsedBuilding],
    grid: GeoBox,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    count_arr: np.ndarray,
    area_arr: np.ndarray,
    max_arr: np.ndarray,
) -> int:
    """Rasterize building footprints and accumulate statistics.

    Returns the number of buildings processed.
    """
    transform = grid.transform
    shape = (grid.shape.y, grid.shape.x)

    for bldg in buildings:
        if bldg.footprint is None or bldg.measured_height is None:
            continue

        # Rasterize this building's footprint at 10m
        try:
            geom = mapping(bldg.footprint)
            mask_result = rasterize(
                [(geom, 1)],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            if mask_result is None:
                continue
            mask = mask_result
        except Exception:  # noqa: S112
            continue

        cells = mask > 0
        n_cells = int(np.sum(cells))
        if n_cells == 0:
            continue

        h = bldg.measured_height
        sum_arr[cells] += h
        sumsq_arr[cells] += h * h
        count_arr[cells] += 1

        # BCR: use actual footprint area (m²) divided by cell area
        footprint_area = bldg.footprint.area
        area_arr[cells] += footprint_area / n_cells  # distribute evenly across touched cells

        # Max height — in-place update
        np.maximum.at(max_arr, cells, h)

    return len(buildings)

# ── prepare ───────────────────────────────────────────────────────────

def prepare_lod2_morphology(
    vintage: int,
    output_root: str,
    run_id: str,
    smoke_tile_count: int | None = None,
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Download, parse, and rasterize LoD2 buildings to the canonical 10 m grid.

    Parameters
    ----------
    vintage :
        Vintage year (must be qualified — see docs/data-sources-and-contracts.md).
    output_root :
        Root URI for all outputs.
    run_id :
        Unique run identifier.
    smoke_tile_count :
        If set, process only this many tiles (for local smoke testing).
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.

    Returns
    -------
    PreparedSecondaryProduct
        Three-band canonical-grid dataset: mean height, std, BCR.
    """
    c_hash = config_hash_for_vintage(vintage)
    grid = grid or canon_grid_10m()

    # ── 1. discover tiles via ATOM feed ──────────────────────────────
    manifest = parse_atom_feed(_FEED_URL, _LOD2_RE, aoi_grid=grid)

    if smoke_tile_count is not None:
        manifest.assets = manifest.assets[:smoke_tile_count]

    log_event(_logger, logging.INFO, "lod2_tiles", n_tiles=len(manifest.assets))

    # ── 2. accumulate rasterization statistics ───────────────────────
    shape = (grid.shape.y, grid.shape.x)
    sum_arr = np.zeros(shape, dtype=np.float64)
    sumsq_arr = np.zeros(shape, dtype=np.float64)
    count_arr = np.zeros(shape, dtype=np.int32)
    area_arr = np.zeros(shape, dtype=np.float64)
    max_arr = np.zeros(shape, dtype=np.float32)

    tile_receipts: list[dict] = []
    all_checksums: list[str] = []
    total_buildings = 0

    for i, asset in enumerate(manifest.assets):
        if (i + 1) % 50 == 0:
            log_event(
                _logger,
                logging.DEBUG,
                "lod2_tile_progress",
                done=i + 1,
                total=len(manifest.assets),
                buildings=total_buildings,
            )

        receipt = _process_lod2_tile(
            asset,
            grid,
            sum_arr,
            sumsq_arr,
            count_arr,
            area_arr,
            max_arr,
            output_root,
        )
        tile_receipts.append(
            {
                "filename": asset.filename,
                "easting": asset.easting,
                "northing": asset.northing,
                "checksum": receipt.checksum,
                "byte_count": receipt.byte_count,
            }
        )
        all_checksums.append(receipt.checksum)

    # ── 3. compute final morphology bands ────────────────────────────
    count_f = count_arr.astype(np.float64)
    mean_arr = np.where(count_f > 0, sum_arr / count_f, np.nan).astype(np.float32)
    variance = np.where(
        count_f > 1,
        (sumsq_arr - (sum_arr * sum_arr) / count_f) / (count_f - 1),
        0.0,
    )
    std_arr = np.where(count_f > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan).astype(np.float32)
    cell_area = 100.0  # 10m × 10m
    bcr_arr = np.where(count_f > 0, area_arr / cell_area, np.nan).astype(np.float32)
    bcr_arr = np.clip(bcr_arr, 0.0, 1.0)
    max_arr = np.where(count_f > 0, max_arr, np.nan).astype(np.float32)

    # ── 4. build canonical xr.Dataset ────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {
            "building_height_mean": (("y", "x"), mean_arr),
            "building_height_std": (("y", "x"), std_arr),
            "building_coverage_ratio": (("y", "x"), bcr_arr),
            "building_height_max": (("y", "x"), max_arr),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    combined_hash = sha256("".join(sorted(all_checksums)).encode()).hexdigest()[:16]
    retrieved_at = datetime.now(UTC).isoformat()

    valid_mean = mean_arr[~np.isnan(mean_arr)]
    return PreparedSecondaryProduct(
        source="lod2_morphology",
        item_key=str(vintage),
        category="morphology",
        dataset=ds,
        contract=contract_for_lod2_morphology(),
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
            "total_buildings": total_buildings,
        },
        qa_stats={
            "valid_frac": (
                round(float(len(valid_mean)) / mean_arr.size, 4) if mean_arr.size > 0 else 0.0
            ),
            "min_height": float(valid_mean.min()) if len(valid_mean) > 0 else None,
            "max_height": float(valid_mean.max()) if len(valid_mean) > 0 else None,
            "mean_height": float(np.nanmean(valid_mean)) if len(valid_mean) > 0 else None,
            "shape": list(mean_arr.shape),
            "tile_count": len(manifest.assets),
            "total_buildings": total_buildings,
        },
        config_hash=c_hash,
    )

def _process_lod2_tile(
    asset: AtomAsset,
    grid: GeoBox,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    count_arr: np.ndarray,
    area_arr: np.ndarray,
    max_arr: np.ndarray,
    output_root: str,
) -> DownloadReceipt:
    """Download and rasterize a single LoD2 tile."""
    from berlin_lst_downscaling.data.secondary.paths import raw_dir

    raw_uri = f"{raw_dir(output_root, 'lod2_morphology', 'current')}/{asset.filename}"
    cache_path = _local_cache_path(output_root, asset.filename)

    receipt = download_to_raw(
        url=asset.url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    asset.checksum = receipt.checksum
    asset.byte_count = receipt.byte_count

    zip_path = Path(receipt.local_cache_path) if receipt.local_cache_path else None
    if zip_path is None:
        raise ValueError(f"No local cache for {asset.filename}")

    buildings = _parse_buildings_from_tile(zip_path, asset)
    n = _accumulate_buildings(
        buildings,
        grid,
        sum_arr,
        sumsq_arr,
        count_arr,
        area_arr,
        max_arr,
    )
    log_event(_logger, logging.DEBUG, "lod2_tile_done", filename=asset.filename, buildings=n)

    return receipt

def _local_cache_path(output_root: str, filename: str) -> str:
    """Return a writable local cache path for a LoD2 tile."""
    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/lod2/{filename}"
    return f"{output_root}/_raw/secondary/lod2_morphology/{filename}"

__all__ = [
    "config_hash_for_vintage",
    "contract_for_lod2_morphology",
    "prepare_lod2_morphology",
]