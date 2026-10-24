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

The CityGML parsing path is shared with the historical-vintage runner
via :mod:`berlin_lst_downscaling.data.secondary.citygml`.
"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.secondary.atom import (
    AtomAsset,
    parse_atom_feed,
)
from berlin_lst_downscaling.data.secondary.building_raster import (
    accumulate_building,
    morphology_bands,
    new_accumulators,
)
from berlin_lst_downscaling.data.secondary.citygml import (
    Building as ParsedBuilding,
)
from berlin_lst_downscaling.data.secondary.citygml import (
    Lod2ParseStats,
    parse_lod2_buildings_with_stats,
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

# ── contract ───────────────────────────────────────────────────────────

# v2: BuildingPart heights, geometry fallback, area-exact 10 m coverage.
_CONFIG_HASH_PREFIX = "lod2_morphology:v2:"


def contract_for_lod2_morphology() -> Contract:
    """Return the output Contract for LoD2 morphology COGs (4 bands).

    Height bounds admit physically real Berlin structures: the Fernsehturm
    (TV tower, 368 m incl. antenna, sphere at ~200–250 m) is part of the
    official LoD2 stock, so ``building_height_mean``/``max`` admit
    [0, 400] m and ``building_height_std`` admits [0, 200] m. No clamping:
    out-of-range finite values fail validation; negative/NaN stay invalid.
    """
    return Contract(
        source="lod2_morphology",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="building_height_mean",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Area-weighted mean building height in metres. "
                    "Weight is the footprint intersection with the 10 m cell."
                ),
                unit="m",
                valid_range=(0.0, 400.0),
            ),
            BandSpec(
                name="building_height_std",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Area-weighted population standard deviation of building "
                    "heights in the cell. Zero when one building intersects it."
                ),
                unit="m",
                valid_range=(0.0, 200.0),
            ),
            BandSpec(
                name="building_coverage_ratio",
                dtype="float32",
                nodata=float("nan"),
                description=(
                    "Fraction of the 10 m cell covered by summed footprint "
                    "intersections, clipped to [0, 1]"
                ),
                unit="",
                valid_range=(-0.01, 1.01),
            ),
            BandSpec(
                name="building_height_max",
                dtype="float32",
                nodata=float("nan"),
                description="Maximum building height in cell (for DSM derivation)",
                unit="m",
                valid_range=(0.0, 400.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=3,
        flag_mode="none",
    )


def config_hash_for_vintage(vintage: int) -> str:
    """Return a stable config hash for a given vintage."""
    raw = f"{_CONFIG_HASH_PREFIX}{vintage}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── CityGML parsing ──────────────────────────────────────────────────
#
# Heavy lifting lives in :mod:`berlin_lst_downscaling.data.secondary.citygml`
# and is shared with the historical-vintage runner.  This adapter keeps
# only the streaming ZIP iteration + rasterisation glue.


@dataclass
class Lod2IngestStats:
    """Buildings read from one tile run and how they were rasterized.

    ``total_buildings`` is the number of emitted units (buildings or
    parts with a footprint and a resolved height). It is not a sentinel
    left at zero.
    """

    total_buildings: int = 0
    buildings_rasterized: int = 0
    buildings_outside_grid: int = 0
    skipped_missing_height: int = 0
    parse: Lod2ParseStats = field(default_factory=Lod2ParseStats)

    def add_tile(
        self,
        *,
        emitted: int,
        rasterized: int,
        outside_grid: int,
        skipped_missing_height: int,
        parse: Lod2ParseStats,
    ) -> None:
        """Add one tile's counters."""
        self.total_buildings += emitted
        self.buildings_rasterized += rasterized
        self.buildings_outside_grid += outside_grid
        self.skipped_missing_height += skipped_missing_height
        self.parse.add(parse)


def _parse_buildings_from_tile(
    zip_path: Path,
    asset: AtomAsset,
) -> tuple[list[ParsedBuilding], Lod2ParseStats]:
    """Parse every building from a CityGML tile ZIP."""
    del asset  # the ZIP, not the ATOM record, carries the buildings
    buildings: list[ParsedBuilding] = []
    stats = Lod2ParseStats()

    with zipfile.ZipFile(zip_path) as z:
        xml_names = [n for n in z.namelist() if n.lower().endswith((".gml", ".xml"))]
        if not xml_names:
            return buildings, stats

        for member in xml_names:
            with z.open(member) as handle:
                raw = handle.read()
            parsed, member_stats = parse_lod2_buildings_with_stats(raw)
            buildings.extend(parsed)
            stats.add(member_stats)

    return buildings, stats


# ── rasterization ─────────────────────────────────────────────────────


def _accumulate_buildings(
    buildings: list[ParsedBuilding],
    grid: GeoBox,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    count_arr: np.ndarray,
    area_arr: np.ndarray,
    max_arr: np.ndarray,
) -> dict[str, int]:
    """Rasterize footprints by cell intersection and return counters.

    ``total_buildings`` is ``len(buildings)``: every emitted unit handed
    to the rasterizer, including one that misses the grid. Units with
    no positive height are not painted and are counted in
    ``skipped_missing_height`` instead of as 0 m.
    """
    transform = grid.transform
    shape = (grid.shape.y, grid.shape.x)
    rasterized = 0
    outside_grid = 0
    skipped = 0

    for bldg in buildings:
        if bldg.footprint is None or bldg.measured_height is None or bldg.measured_height <= 0:
            skipped += 1
            continue
        hit = accumulate_building(
            bldg.footprint,
            float(bldg.measured_height),
            transform=transform,
            shape=shape,
            sum_h=sum_arr,
            sum_h2=sumsq_arr,
            sum_area=area_arr,
            count=count_arr,
            max_h=max_arr,
        )
        if hit:
            rasterized += 1
        else:
            outside_grid += 1

    return {
        "total_buildings": len(buildings),
        "buildings_rasterized": rasterized,
        "buildings_outside_grid": outside_grid,
        "skipped_missing_height": skipped,
    }


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
    buffers = new_accumulators(shape)
    sum_arr = buffers["sum_h"]
    sumsq_arr = buffers["sum_h2"]
    count_arr = buffers["count"]
    area_arr = buffers["sum_area"]
    max_arr = buffers["max_h"]

    tile_receipts: list[dict] = []
    all_checksums: list[str] = []
    ingest = Lod2IngestStats()

    for i, asset in enumerate(manifest.assets):
        if (i + 1) % 50 == 0:
            log_event(
                _logger,
                logging.DEBUG,
                "lod2_tile_progress",
                done=i + 1,
                total=len(manifest.assets),
                buildings=ingest.total_buildings,
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
            ingest,
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
    bands = morphology_bands(sum_arr, sumsq_arr, area_arr, count_arr, max_arr)
    mean_arr = bands["building_height_mean"]
    std_arr = bands["building_height_std"]
    bcr_arr = bands["building_coverage_ratio"]
    max_out = bands["building_height_max"]

    # ── 4. build canonical xr.Dataset ────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {
            "building_height_mean": (("y", "x"), mean_arr),
            "building_height_std": (("y", "x"), std_arr),
            "building_coverage_ratio": (("y", "x"), bcr_arr),
            "building_height_max": (("y", "x"), max_out),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    combined_hash = sha256("".join(sorted(all_checksums)).encode()).hexdigest()[:16]
    retrieved_at = datetime.now(UTC).isoformat()

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
            "total_buildings": ingest.total_buildings,
            "parse": ingest.parse.as_dict(),
        },
        qa_stats=lod2_qa_stats(
            mean_arr=mean_arr,
            ingest=ingest,
            tile_count=len(manifest.assets),
        ),
        config_hash=c_hash,
    )


def lod2_qa_stats(
    *,
    mean_arr: np.ndarray,
    ingest: Lod2IngestStats,
    tile_count: int,
) -> dict:
    """Build the morphology QA payload, including the real building count."""
    valid_mean = mean_arr[~np.isnan(mean_arr)]
    return {
        "valid_frac": (
            round(float(len(valid_mean)) / mean_arr.size, 4) if mean_arr.size > 0 else 0.0
        ),
        "min_height": float(valid_mean.min()) if len(valid_mean) > 0 else None,
        "max_height": float(valid_mean.max()) if len(valid_mean) > 0 else None,
        "mean_height": float(np.nanmean(valid_mean)) if len(valid_mean) > 0 else None,
        "shape": list(mean_arr.shape),
        "tile_count": tile_count,
        "total_buildings": ingest.total_buildings,
        "buildings_rasterized": ingest.buildings_rasterized,
        "buildings_outside_grid": ingest.buildings_outside_grid,
        "skipped_missing_height": ingest.skipped_missing_height,
        "parse": ingest.parse.as_dict(),
    }


def _process_lod2_tile(
    asset: AtomAsset,
    grid: GeoBox,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    count_arr: np.ndarray,
    area_arr: np.ndarray,
    max_arr: np.ndarray,
    output_root: str,
    ingest: Lod2IngestStats,
) -> DownloadReceipt:
    """Download and rasterize a single LoD2 tile.

    The emitted-building count is added to *ingest*. Dropping that
    return value is what left ``total_buildings`` at 0.
    """
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

    buildings, parse_stats = _parse_buildings_from_tile(zip_path, asset)
    counted = _accumulate_buildings(
        buildings,
        grid,
        sum_arr,
        sumsq_arr,
        count_arr,
        area_arr,
        max_arr,
    )
    ingest.add_tile(
        emitted=counted["total_buildings"],
        rasterized=counted["buildings_rasterized"],
        outside_grid=counted["buildings_outside_grid"],
        skipped_missing_height=counted["skipped_missing_height"],
        parse=parse_stats,
    )
    log_event(
        _logger,
        logging.DEBUG,
        "lod2_tile_done",
        filename=asset.filename,
        buildings=counted["total_buildings"],
    )

    return receipt


def _local_cache_path(output_root: str, filename: str) -> str:
    """Return a writable local cache path for a LoD2 tile."""
    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/lod2/{filename}"
    return f"{output_root}/_raw/secondary/lod2_morphology/{filename}"


__all__ = [
    "Lod2IngestStats",
    "config_hash_for_vintage",
    "contract_for_lod2_morphology",
    "lod2_qa_stats",
    "prepare_lod2_morphology",
]
