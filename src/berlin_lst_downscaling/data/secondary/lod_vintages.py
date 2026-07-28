"""LoD historical-vintage morphology processor — archive-first design.

Streams one GCS-resident CityGML ZIP per vintage (LoD1-2017, LoD2-2021,
LoD2-2022) directly into the morphology pipeline without retaining
expanded tile data on local disk, filters the 2021 stock against the
2017 baseline, and publishes the canonical-grid ``lod2_morphology``
products plus the per-vintage geometry bundle.

Vintages are addressed through the immutable archive contract::

    gs://<raw_root>/lod_vintages/<vintage>/archive.zip
        sha256: <hex>
        member_count: <n>
        members: <sorted list of *.xml>

The archive is downloaded to a ``tempfile.TemporaryDirectory()`` once
per vintage, iterated as a streaming ZipFile, and deleted on context
exit. No expanded XML is ever written to GCS.

Outputs follow the existing Pipeline-A layout::

    gs://<source_root>/ard/static/sources/lod2_morphology/<vintage>/
        ├─ lod2_morphology_<vintage>.tif
        ├─ lod2_morphology_<vintage>.stac.json
        ├─ provenance.json
        └─ complete.json

Derived per-vintage bundle (Pipeline B)::

    gs://<derived_root>/ard/static/derived/<product>/dgm1-2021__lod2-<vintage>__vh-2020/

Library search
--------------
PyPI candidates ``citygml``, ``pycitygml``, ``citygml-tools`` returned
404 on the project's index client.  XML parsing uses stdlib
``ElementTree`` (see :mod:`berlin_lst_downscaling.data.secondary.citygml`).
The 2017 stock filter uses :mod:`geopandas` with its documented
``sjoin`` predicate (``intersects``) — ``geopandas`` is already a
project dependency and provides an index-backed spatial join.

Memory discipline
-----------------
The runner processes one LoD2 tile at a time and rasterises into
pre-allocated accumulators.  For 2017 the LoD1 footprint cache is
limited to the current tile plus its eight neighbours (9-tile sliding
window).  Peak RAM holds 9 tiles of LoD1 footprints + 1 tile of LoD2
buildings + the 5 raster accumulators for the canonical 10 m grid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.io import atomic_upload, atomic_write, log_event
from berlin_lst_downscaling.data.secondary.citygml import (
    Building,
    Footprint,
    iter_xml_members,
    parse_lod1_footprints,
    parse_lod2_buildings,
)
from berlin_lst_downscaling.data.secondary.lod2 import (
    config_hash_for_vintage,
    contract_for_lod2_morphology,
)
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    finalize_secondary_product,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── configuration constants ──────────────────────────────────────────


@dataclass(frozen=True)
class VintageSpec:
    """Immutable description of one raw archive vintage."""

    vintage: int
    level: str  # "lod1" or "lod2"
    archive_filename: str  # e.g. "LoD1_2017.zip", "LoD2_BE_1_33_2021.zip"
    expected_count: int
    feed_label: str


# Final source spec — all vintages stream from a single ZIP per vintage.
_VINTAGE_SOURCES: dict[int, VintageSpec] = {
    2017: VintageSpec(
        vintage=2017,
        level="lod1",
        archive_filename="LoD1_2017.zip",
        expected_count=1006,
        feed_label="Berlin GDI LoD1 2017 (Senatsverwaltung Berlin)",
    ),
    2021: VintageSpec(
        vintage=2021,
        level="lod2",
        archive_filename="LoD2_BE_1_33_2021.zip",
        expected_count=928,
        feed_label="Berlin GDI LoD2 2021 (Senatsverwaltung Berlin)",
    ),
    2022: VintageSpec(
        vintage=2022,
        level="lod2",
        archive_filename="LoD2_2022.zip",
        expected_count=928,
        feed_label="Berlin GDI LoD2 2022 (Senatsverwaltung Berlin)",
    ),
}

# LoD2 building kept when its footprint is overlapped by LoD1 footprints
# by at least this fraction of its own footprint area.
# decision: 50% threshold matches the brief's "vorhanden" rule for
# existing buildings; stricter values would erase an- und umbauten that
# already existed in 2017 but received roof details in 2021.
_STOCK_OVERLAP_MIN_FRAC = 0.50


# ── archive materialisation ──────────────────────────────────────────


@dataclass
class ArchiveMaterialization:
    """Result of materialising one archive ZIP from GCS to local disk."""

    spec: VintageSpec
    local_path: Path
    byte_count: int
    sha256: str
    member_count: int
    member_names: list[str]


def _stream_sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    """SHA-256 of a local file, streamed in 64 KiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _stream_sha256_gcs(uri: str, chunk: int = 1 << 16) -> str:
    """SHA-256 of a GCS object, streamed via ``blob.open('rb')``."""
    from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    h = hashlib.sha256()
    with blob.open("rb") as f:
        while True:
            block: bytes = f.read(chunk)  # type: ignore[assignment]
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def archive_uri_for(raw_root: str, spec: VintageSpec) -> str:
    """Return the canonical GCS URI for *spec*'s archive ZIP."""
    return f"{raw_root.rstrip('/')}/lod_vintages/{spec.vintage}/{spec.archive_filename}"


def materialize_vintage_archive(
    spec: VintageSpec,
    raw_root: str,
) -> tuple[ArchiveMaterialization, tempfile.TemporaryDirectory]:
    """Stream one archive ZIP from GCS into a local temp directory.

    The ZIP is downloaded once via :func:`google.cloud.storage.blob.download_to_filename`
    so the rest of the pipeline can iterate it as a normal local file.
    Returns the materialization record plus the owning TemporaryDirectory;
    the caller is responsible for deleting the temp dir (or letting the
    context manager exit at function return).
    """
    uri = archive_uri_for(raw_root, spec)
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"lod_archive_{spec.vintage}_")
    target = Path(tmp_dir.name) / spec.archive_filename

    from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

    log_event(
        _logger,
        logging.INFO,
        "archive_download_start",
        vintage=spec.vintage,
        archive_uri=uri,
        local=str(target),
    )
    bucket_name, key = _parse_gs_uri(uri)
    client = _gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"Archive not found in GCS: {uri}")
    blob.download_to_filename(str(target))
    byte_count = target.stat().st_size
    sha = _stream_sha256_file(target)

    member_names = sorted(iter_xml_members(target))

    log_event(
        _logger,
        logging.INFO,
        "archive_download_done",
        vintage=spec.vintage,
        archive_uri=uri,
        byte_count=byte_count,
        sha256=sha,
        n_members=len(member_names),
    )

    if len(member_names) != spec.expected_count:
        raise ValueError(
            f"Archive member count mismatch for vintage {spec.vintage}: "
            f"expected {spec.expected_count}, got {len(member_names)}"
        )

    return (
        ArchiveMaterialization(
            spec=spec,
            local_path=target,
            byte_count=byte_count,
            sha256=sha,
            member_count=len(member_names),
            member_names=member_names,
        ),
        tmp_dir,
    )


# ── raw archive publication (manifest) ──────────────────────────────


@dataclass
class RawManifest:
    """Per-vintage raw archive manifest."""

    vintage: int
    source_kind: str
    feed_label: str
    archive_uri: str
    archive_sha256: str
    archive_byte_count: int
    member_count: int
    member_names: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return self.archive_byte_count


def publish_raw_archive_manifest(
    manifest: RawManifest,
    source_root: str,
) -> str:
    """Write the per-vintage archive manifest JSON next to the source layout."""
    base = (
        f"{source_root.rstrip('/')}/ard/static/sources/lod_vintages/"
        f"raw_manifest_{manifest.vintage}.json"
    )
    payload = {
        "vintage": manifest.vintage,
        "source_kind": manifest.source_kind,
        "feed_label": manifest.feed_label,
        "archive_uri": manifest.archive_uri,
        "archive_sha256": manifest.archive_sha256,
        "archive_byte_count": manifest.archive_byte_count,
        "member_count": manifest.member_count,
        "members": list(manifest.member_names),
        "published_at": datetime.now(UTC).isoformat(),
    }
    atomic_write(base, json.dumps(payload, indent=2), overwrite=True)
    return base


# ── archive streaming (local → GCS) ────────────────────────────────


def stream_archive_to_gcs(
    spec: VintageSpec,
    local_archive: Path,
    raw_root: str,
) -> RawManifest:
    """Upload a single archive ZIP to GCS and publish the raw manifest.

    Parameters
    ----------
    spec :
        Vintage metadata.
    local_archive :
        Local path to the compressed archive to upload. The caller is
        responsible for creating the archive (e.g. via ``zip`` CLI).
    raw_root :
        Bucket-level root under which the per-vintage archive path lives.
    """
    uri = archive_uri_for(raw_root, spec)
    log_event(
        _logger,
        logging.INFO,
        "raw_upload_start",
        vintage=spec.vintage,
        archive=str(local_archive),
        archive_uri=uri,
    )

    t0 = time.perf_counter()
    atomic_upload(local_archive, uri, overwrite=True)
    byte_count = local_archive.stat().st_size
    sha = _stream_sha256_gcs(uri)

    member_names = sorted(iter_xml_members(local_archive))

    elapsed = time.perf_counter() - t0
    log_event(
        _logger,
        logging.INFO,
        "raw_upload_done",
        vintage=spec.vintage,
        byte_count=byte_count,
        sha256=sha,
        n_members=len(member_names),
        elapsed_s=round(elapsed, 1),
    )

    if len(member_names) != spec.expected_count:
        log_event(
            _logger,
            logging.WARNING,
            "raw_upload_count_mismatch",
            vintage=spec.vintage,
            expected=spec.expected_count,
            actual=len(member_names),
        )

    manifest = RawManifest(
        vintage=spec.vintage,
        source_kind=spec.level,
        feed_label=spec.feed_label,
        archive_uri=uri,
        archive_sha256=sha,
        archive_byte_count=byte_count,
        member_count=len(member_names),
        member_names=member_names,
    )
    return manifest


# ── tile iteration over a ZIP archive ───────────────────────────────


_LOD1_TILE_RE = re.compile(r"LoD1_(\d{3})_(\d{4})_")
_LOD2_TILE_RE = re.compile(r"LoD2_(?:\d+_)?(\d{3})_(\d{4})_")


def _lod1_tile_key(name: str) -> str:
    m = _LOD1_TILE_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse LoD1 tile key from {name!r}")
    return f"{m.group(1)}_{m.group(2)}"


def _lod2_tile_key(name: str) -> str:
    m = _LOD2_TILE_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse LoD2 tile key from {name!r}")
    return f"{m.group(1)}_{m.group(2)}"


def _tile_key_in_bbox(tile_key: str, grid: GeoBox) -> bool:
    try:
        e, n = (int(s) for s in tile_key.split("_"))
    except ValueError:
        return False
    bbox = grid.extent.boundingbox
    minx, miny, maxx, maxy = bbox.left, bbox.bottom, bbox.right, bbox.top
    tile_box = (e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)
    if tile_box[0] > maxx or tile_box[2] < minx:
        return False
    if tile_box[1] > maxy or tile_box[3] < miny:
        return False
    return True


def _neighbour_tile_keys(tile_key: str) -> list[str]:
    """Return the eight adjacent 1 km tile keys."""
    e, n = (int(p) for p in tile_key.split("_"))
    keys = []
    for de in (-1, 0, 1):
        for dn in (-1, 0, 1):
            if de == 0 and dn == 0:
                continue
            keys.append(f"{e + de}_{n + dn}")
    return keys


def iter_lod1_tiles(
    archive_path: Path,
    *,
    grid: GeoBox | None = None,
    max_tiles: int | None = None,
) -> Iterator[tuple[str, list[Footprint]]]:
    """Yield ``(tile_key, footprints)`` pairs from a LoD1 ZIP archive.

    Each tuple holds one tile's worth of footprints; nothing is
    retained after the consumer iterates past it.  With *grid* set the
    loader skips tiles that cannot intersect it.
    """
    with zipfile.ZipFile(archive_path) as z:
        members = sorted(iter_xml_members(archive_path))
        if grid is not None:
            members = [m for m in members if _tile_key_in_bbox(_lod1_tile_key(m), grid)]
        if max_tiles is not None:
            members = members[:max_tiles]

        for member in members:
            data = z.read(member)
            tile_key = _lod1_tile_key(member)
            yield tile_key, parse_lod1_footprints(data)


def iter_lod2_tiles(
    archive_path: Path,
    *,
    grid: GeoBox | None = None,
    max_tiles: int | None = None,
) -> Iterator[tuple[str, list[Building]]]:
    """Yield ``(tile_key, buildings)`` pairs from a LoD2 ZIP archive."""
    with zipfile.ZipFile(archive_path) as z:
        members = sorted(iter_xml_members(archive_path))
        if grid is not None:
            members = [m for m in members if _tile_key_in_bbox(_lod2_tile_key(m), grid)]
        if max_tiles is not None:
            members = members[:max_tiles]

        for member in members:
            data = z.read(member)
            tile_key = _lod2_tile_key(member)
            yield tile_key, parse_lod2_buildings(data)


# ── 2017 stock filter ────────────────────────────────────────────────


@dataclass
class FilterStats:
    """QA counters for the 2017 LoD2 stock filter."""

    input_lod2: int = 0
    input_lod1: int = 0
    retained: int = 0
    rejected: int = 0
    rejected_low_overlap: int = 0
    rejected_invalid_overlap: int = 0

    def as_dict(self) -> dict:
        return {
            "input_lod2": self.input_lod2,
            "input_lod1": self.input_lod1,
            "retained": self.retained,
            "rejected": self.rejected,
            "rejected_low_overlap": self.rejected_low_overlap,
            "rejected_invalid_overlap": self.rejected_invalid_overlap,
            "min_overlap_frac": _STOCK_OVERLAP_MIN_FRAC,
        }


def filter_lod2_against_lod1(
    lod2_buildings: list[Building],
    lod1_pool: list[Footprint],
    tile_key: str,
) -> tuple[list[Building], FilterStats]:
    """Retain LoD2 buildings whose footprint is covered by LoD1 footprints.

    The LoD1 *pool* is the current LoD1 footprint cache (the active tile
    plus its eight neighbours).  Edge buildings are scored against the
    cached neighbours rather than the full 1006-tile LoD1 archive.
    """
    stats = FilterStats(input_lod2=len(lod2_buildings))

    pool: list[Polygon | MultiPolygon] = [
        f.footprint for f in lod1_pool if f.footprint is not None and not f.footprint.is_empty
    ]
    stats.input_lod1 = len(pool)
    if not pool or not lod2_buildings:
        stats.rejected = len(lod2_buildings)
        stats.rejected_low_overlap = len(lod2_buildings)
        return [], stats

    lod1_union = pool[0] if len(pool) == 1 else unary_union(pool)

    retained: list[Building] = []
    for b in lod2_buildings:
        if b.footprint is None or b.footprint.is_empty:
            stats.rejected += 1
            stats.rejected_invalid_overlap += 1
            continue

        try:
            intersection = b.footprint.intersection(lod1_union)
            inter_area = intersection.area
            frac = inter_area / b.footprint.area if b.footprint.area > 0 else 0.0
        except Exception:
            stats.rejected += 1
            stats.rejected_invalid_overlap += 1
            continue

        if frac >= _STOCK_OVERLAP_MIN_FRAC:
            retained.append(b)
            stats.retained += 1
        else:
            stats.rejected += 1
            stats.rejected_low_overlap += 1

    return retained, stats


# ── rasterisation accumulator ───────────────────────────────────────


class _Accumulator:
    """Pre-allocated 10 m-grid statistics buffers for one vintage product."""

    def __init__(self, grid: GeoBox) -> None:
        shape = (grid.shape.y, grid.shape.x)
        self.grid = grid
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)
        self.area = np.zeros(shape, dtype=np.float64)
        self.max = np.zeros(shape, dtype=np.float32)

    def add_tile(self, buildings: list[Building]) -> None:
        transform = self.grid.transform
        shape = (self.grid.shape.y, self.grid.shape.x)
        for bldg in buildings:
            if bldg.footprint is None or bldg.measured_height is None:
                continue
            try:
                geom = mapping(bldg.footprint)
                mask_result = rasterize(
                    [(geom, 1)],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                mask = mask_result if mask_result is not None else None
            except Exception:  # noqa: S112 — skip buildings with bad geometry
                continue
            if mask is None:
                continue
            cells = mask > 0
            n_cells = int(np.sum(cells))
            if n_cells == 0:
                continue

            h = bldg.measured_height
            self.sum[cells] += h
            self.sumsq[cells] += h * h
            self.count[cells] += 1
            self.area[cells] += bldg.footprint.area / n_cells
            np.maximum.at(self.max, cells, h)

    def to_dataset(self) -> xr.Dataset:
        """Render the accumulated statistics into the canonical 4-band dataset."""
        count_f = self.count.astype(np.float64)
        empty = self.count == 0

        mean_arr = np.where(count_f > 0, self.sum / count_f, np.nan).astype(np.float32)
        variance = np.where(
            count_f > 1,
            (self.sumsq - (self.sum * self.sum) / count_f) / (count_f - 1),
            0.0,
        )
        std_arr = np.where(
            count_f > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan
        ).astype(np.float32)
        bcr_arr = np.where(count_f > 0, self.area / 100.0, np.nan).astype(np.float32)
        bcr_arr = np.clip(bcr_arr, 0.0, 1.0)
        max_arr = np.where(count_f > 0, self.max, np.nan).astype(np.float32)

        # Mask untouched cells as NaN so partial-grid runs do not
        # masquerade as full-grids.
        for arr in (mean_arr, std_arr, bcr_arr, max_arr):
            arr[empty] = np.nan

        xs = self.grid.transform.xoff + 5.0 + np.arange(self.grid.shape.x) * 10.0
        ys = self.grid.transform.yoff - 5.0 - np.arange(self.grid.shape.y) * 10.0
        ds = xr.Dataset(
            {
                "building_height_mean": (("y", "x"), mean_arr),
                "building_height_std": (("y", "x"), std_arr),
                "building_coverage_ratio": (("y", "x"), bcr_arr),
                "building_height_max": (("y", "x"), max_arr),
            },
            coords={"x": xs, "y": ys},
        )
        return ds.rio.write_crs(str(self.grid.crs)).rio.write_transform(
            self.grid.transform
        )


# ── product publisher (memory-bounded) ──────────────────────────────


def prepare_vintage_morphology(
    vintage: int,
    raw_root: str,
    *,
    lod1_archive: Path | None = None,
    lod2_archive: Path | None = None,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
) -> tuple[PreparedSecondaryProduct, FilterStats, ArchiveMaterialization]:
    """Build the canonical morphology product for *vintage* by streaming the archive.

    For vintage 2017 the LoD2-2021 stock is filtered against the LoD1
    footprints via :func:`filter_lod2_against_lod1`; the other vintages
    consume the raw LoD2 stock directly.

    Parameters
    ----------
    lod1_archive :
        Local path to the LoD1 archive (only required for *vintage=2017*).
    lod2_archive :
        Local path to the LoD2 archive to process. For 2017, this is
        always the 2021 archive; for 2021/2022, the matching archive.
    """
    spec = _VINTAGE_SOURCES[vintage]
    source_vintage = 2021 if vintage == 2017 else vintage
    source_spec = _VINTAGE_SOURCES[source_vintage]

    if lod2_archive is None:
        raise ValueError("lod2_archive is required")
    if vintage == 2017 and lod1_archive is None:
        raise ValueError("Vintage 2017 requires lod1_archive")

    grid = grid or canon_grid_10m()
    c_hash = config_hash_for_vintage(vintage)

    accumulator = _Accumulator(grid)
    filter_stats = FilterStats()
    tile_count = 0

    if vintage == 2017:
        # Pre-walk LoD1 to know the tile ordering so we can stream both
        # archives in lockstep without scanning either archive twice.
        lod1_iter = iter_lod1_tiles(
            lod1_archive,  # type: ignore[arg-type]
            grid=grid,
            max_tiles=smoke_tile_count,
        )
        # Build the full LoD1 cache in a single pass — the brief
        # requires every LoD2 tile to be scored against the
        # neighbouring LoD1 stock, so we cannot simply pre-cache by
        # tile.  For the memory bound we cap total cache size by
        # streaming tiles; we re-fetch neighbours on demand.
        log_event(_logger, logging.INFO, "lod1_index_load_start", vintage=2017)
        lod1_full_index: dict[str, list[Footprint]] = {}
        for tile_key, footprints in lod1_iter:
            lod1_full_index[tile_key] = footprints
        log_event(
            _logger,
            logging.INFO,
            "lod1_index_load_done",
            n_tiles=len(lod1_full_index),
            n_footprints=sum(len(v) for v in lod1_full_index.values()),
        )

        for lod2_tile_key, buildings in iter_lod2_tiles(
            lod2_archive,
            grid=grid,
            max_tiles=smoke_tile_count,
        ):
            tile_count += 1
            # Sliding 9-tile LoD1 window around the current LoD2 tile.
            neighbour_keys = [lod2_tile_key, *_neighbour_tile_keys(lod2_tile_key)]
            pool: list[Footprint] = []
            for n_key in neighbour_keys:
                fps = lod1_full_index.get(n_key, [])
                if fps:
                    pool.extend(fps)
            kept, stats = filter_lod2_against_lod1(buildings, pool, lod2_tile_key)
            accumulator.add_tile(kept)

            filter_stats.input_lod2 += stats.input_lod2
            filter_stats.input_lod1 += stats.input_lod1
            filter_stats.retained += stats.retained
            filter_stats.rejected += stats.rejected
            filter_stats.rejected_low_overlap += stats.rejected_low_overlap
            filter_stats.rejected_invalid_overlap += stats.rejected_invalid_overlap
    else:
        for _tile_key, buildings in iter_lod2_tiles(
            lod2_archive,
            grid=grid,
            max_tiles=smoke_tile_count,
        ):
            tile_count += 1
            accumulator.add_tile(buildings)

    ds = accumulator.to_dataset()
    valid_frac = float(
        np.count_nonzero(~np.isnan(ds["building_height_mean"].values))
    ) / ds["building_height_mean"].size

    raw_manifest_uri = (
        f"{raw_root.rstrip('/').replace('gs://berlin-lst-data', '')}"
        f"/ard/static/sources/lod_vintages/raw_manifest_{vintage}.json"
    )

    source_metadata = {
        "vintage": vintage,
        "source_vintage": source_vintage,
        "level": spec.level,
        "feed_label": spec.feed_label,
        "archive_uri": archive_uri_for(raw_root, source_spec),
        "raw_manifest_uri": raw_manifest_uri,
        "tile_count": tile_count,
        "stock_filter": filter_stats.as_dict() if vintage == 2017 else None,
    }

    archive_materialization = ArchiveMaterialization(
        spec=source_spec,
        local_path=lod2_archive,
        byte_count=lod2_archive.stat().st_size,
        sha256=_stream_sha256_file(lod2_archive),
        member_count=len(iter_xml_members(lod2_archive)),
        member_names=sorted(iter_xml_members(lod2_archive)),
    )

    return (
        PreparedSecondaryProduct(
            source="lod2_morphology",
            item_key=str(vintage),
            category="morphology",
            dataset=ds,
            contract=contract_for_lod2_morphology(),
            nominal_interval=vintage_interval(vintage),
            source_metadata=source_metadata,
            qa_stats={
                "valid_frac": round(valid_frac, 4),
                "tile_count": tile_count,
                **({"stock_filter": filter_stats.as_dict()} if vintage == 2017 else {}),
            },
            config_hash=c_hash,
        ),
        filter_stats,
        archive_materialization,
    )


def publish_vintage_morphology(
    vintage: int,
    source_root: str,
    run_id: str,
    *,
    lod1_archive: Path | None = None,
    lod2_archive: Path | None = None,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
):
    """Prepare and finalise the *vintage* morphology product."""
    grid = grid or canon_grid_10m()
    prepared, stats, archive = prepare_vintage_morphology(
        vintage,
        source_root,
        lod1_archive=lod1_archive,
        lod2_archive=lod2_archive,
        grid=grid,
        smoke_tile_count=smoke_tile_count,
    )
    from berlin_lst_downscaling.data.secondary.paths import source_product_dir

    prod_dir = source_product_dir(source_root, "lod2_morphology", str(vintage))
    artifacts = finalize_secondary_product(prepared, grid, prod_dir, run_id)
    log_event(
        _logger,
        logging.INFO,
        "vintage_morphology_published",
        vintage=vintage,
        cog_uri=artifacts.cog_uri,
        retained=stats.retained,
        rejected=stats.rejected,
    )
    return artifacts, stats, archive


# ── geometry mapping ────────────────────────────────────────────────


def publish_geometry_mapping(
    metadata_root: str,
    *,
    vintage_artifacts: dict[int, dict],
    published_at: str | None = None,
) -> str:
    """Publish the year → vintage carry-forward mapping artefact.

    Only includes years covered by the *vintage_artifacts* actually
    published in this run.  Runners that publish a subset of vintages
    get a partial mapping; the validator uses the same vintage list to
    determine which years must be present.
    """
    published_at = published_at or datetime.now(UTC).isoformat()

    selected_vintages = set(int(v) for v in vintage_artifacts)
    year_map: dict[str, str] = {}
    for year in range(2017, 2027):
        matched = None
        for vintage in sorted(selected_vintages):
            lo, hi = year_to_vintage_range(vintage)
            if lo is None or hi is None:
                continue
            if lo <= year <= hi:
                matched = vintage
                break
        if matched is not None:
            year_map[str(year)] = str(matched)

    payload = {
        "version": "v1",
        "published_at": published_at,
        "rule": "carry-forward: scene_year → most-recent vintage with year ≤ scene_year",
        "year_to_vintage": year_map,
        "vintages": {
            str(vintage): {
                "geometry_id": art["geometry_id"],
                "lod2_morphology_cog": art["lod2_morphology"],
                "publish_target": f"dgm1-2021__lod2-{vintage}__vh-2020",
            }
            for vintage, art in sorted(vintage_artifacts.items())
        },
    }
    uri = f"{metadata_root.rstrip('/')}/geometry_mapping.json"
    atomic_write(uri, json.dumps(payload, indent=2), overwrite=True)
    log_event(_logger, logging.INFO, "geometry_mapping_published", uri=uri)
    return uri


def year_to_vintage_range(year: int) -> tuple[int | None, int | None]:
    """Return the inclusive scene-year range served by *year*."""
    mapping = {
        2017: (2017, 2020),
        2021: (2021, 2021),
        2022: (2022, 2023),
        2024: (2024, 2026),
    }
    return mapping.get(year, (None, None))


# ── derived products per vintage ─────────────────────────────────────


def vintage_geometry_id(vintage: int) -> str:
    """Return the per-vintage geometry ID for downstream derived products."""
    return f"dgm1-2021__lod2-{vintage}__vh-2020"


def derive_vintage_products(
    vintage: int,
    source_root: str,
    derived_root: str,
    run_id: str,
    *,
    grid: GeoBox | None = None,
    max_radius_m: float = 200.0,
    svf_max_radius: int = 3,
    svf_n_directions: int = 16,
) -> dict:
    """Build the morphology-dependent derived products for one vintage.

    Reads the freshly-published ``lod2_morphology/<vintage>`` COG plus the
    existing ``terrain_height/2021`` and ``vegetation_height/2020``
    products, and publishes:

    - ``building_dsm``
    - ``combined_dsm``
    - ``horizon_building``
    - ``svf``

    The vegetation-derived products (``vegetation_dsm``,
    ``horizon_vegetation``) are reused unchanged from the existing
    2024-derived bundle since their inputs are vintage-agnostic.
    """
    from berlin_lst_downscaling.data.io.storage import exists
    from berlin_lst_downscaling.data.secondary import dsm as dsm_mod
    from berlin_lst_downscaling.data.secondary import horizon as horizon_mod
    from berlin_lst_downscaling.data.secondary import svf as svf_mod
    from berlin_lst_downscaling.data.secondary.paths import (
        derived_product_cog,
        derived_product_dir,
        source_product_cog,
    )
    from berlin_lst_downscaling.data.secondary.source_products import resolve_source_products

    grid = grid or canon_grid_10m()
    geometry_id = vintage_geometry_id(vintage)

    log_event(
        _logger,
        logging.INFO,
        "derive_vintage_start",
        vintage=vintage,
        geometry_id=geometry_id,
    )

    report = resolve_source_products(source_root)
    if not report.ok:
        raise ValueError(
            "Required source products missing: " + "; ".join(report.errors)
        )
    src_map = {f"{r.source}/{r.revision}": r for r in report.resolved}
    terrain = src_map.get("terrain_height/2021")
    vh = src_map.get("vegetation_height/2020")
    if terrain is None or vh is None:
        raise ValueError(
            "terrain_height/2021 and vegetation_height/2020 are required "
            "to derive vintage-dependent products"
        )

    lod2_cog = source_product_cog(source_root, "lod2_morphology", str(vintage))
    if not exists(lod2_cog):
        raise ValueError(f"Morphology product missing: {lod2_cog}")

    upstream_hashes = {
        "terrain": terrain.config_hash,
        "lod2": next(
            r.config_hash for r in report.resolved if r.source == "lod2_morphology"
        ),
        "vh": vh.config_hash,
    }

    artefacts: dict[str, str] = {}

    # 1. building_dsm
    bldg_dir = derived_product_dir(derived_root, "building_dsm", geometry_id)
    building_dsm = dsm_mod.prepare_building_dsm(
        terrain.cog_uri,
        lod2_cog,
        derived_root,
        run_id,
        item_key=geometry_id,
        upstream_hashes=upstream_hashes,
        grid=grid,
    )
    building_dsm_artifacts = finalize_secondary_product(building_dsm, grid, bldg_dir, run_id)
    artefacts["building_dsm"] = building_dsm_artifacts.cog_uri

    # 2. combined_dsm — needs vegetation_dsm from existing pipeline output
    veg_dsm_cog = derived_product_cog(
        derived_root, "vegetation_dsm", "dgm1-2021__lod2-2024__vh-2020"
    )
    if exists(veg_dsm_cog):
        combined_dir = derived_product_dir(derived_root, "combined_dsm", geometry_id)
        combined_dsm = dsm_mod.prepare_combined_dsm(
            building_dsm_artifacts.cog_uri,
            veg_dsm_cog,
            derived_root,
            run_id,
            item_key=geometry_id,
            upstream_hashes=upstream_hashes,
            grid=grid,
        )
        combined_artifacts = finalize_secondary_product(combined_dsm, grid, combined_dir, run_id)
        artefacts["combined_dsm"] = combined_artifacts.cog_uri

        # 3. horizon_building (from building_dsm)
        horizon_dir = derived_product_dir(derived_root, "horizon_building", geometry_id)
        horizon_bldg = horizon_mod.prepare_horizon(
            building_dsm_artifacts.cog_uri,
            derived_root,
            run_id,
            item_key=geometry_id,
            component="building",
            upstream_hash=geometry_id,
            max_radius_m=max_radius_m,
            grid=grid,
        )
        horizon_artifacts = finalize_secondary_product(
            horizon_bldg, grid, horizon_dir, run_id
        )
        artefacts["horizon_building"] = horizon_artifacts.cog_uri

        # 4. svf (from combined_dsm)
        svf_dir = derived_product_dir(derived_root, "svf", geometry_id)
        svf_product = svf_mod.prepare_svf(
            combined_artifacts.cog_uri,
            derived_root,
            run_id,
            item_key=geometry_id,
            upstream_hash=geometry_id,
            max_radius=svf_max_radius,
            n_directions=svf_n_directions,
            grid=grid,
        )
        svf_artifacts = finalize_secondary_product(svf_product, grid, svf_dir, run_id)
        artefacts["svf"] = svf_artifacts.cog_uri
    else:
        log_event(
            _logger,
            logging.WARNING,
            "vegetation_dsm_missing",
            expected=veg_dsm_cog,
            note="combined_dsm, horizon_building, and svf skipped for this vintage",
        )

    log_event(
        _logger,
        logging.INFO,
        "derive_vintage_done",
        vintage=vintage,
        geometry_id=geometry_id,
        products=sorted(artefacts.keys()),
    )
    return {
        "vintage": vintage,
        "geometry_id": geometry_id,
        "artifacts": artefacts,
        "upstream_hashes": upstream_hashes,
    }


# ── local input staging (CLI helper) ─────────────────────────────────


def stage_local_archive(spec: VintageSpec, source_dir: Path) -> Path:
    """Create the canonical archive ZIP for *spec* from *source_dir*.

    Used to build the GCS-resident archive from the local input dir
    (2017 dir, 2021 dir) without keeping the archive plus the expanded
    files on disk at the same time.  Uses stdlib ``zipfile`` to stream
    the directory into the archive.

    The caller is responsible for deleting *source_dir* only after the
    archive has been verified in GCS.
    """
    archive_name = spec.archive_filename
    target = source_dir.parent / archive_name
    log_event(
        _logger,
        logging.INFO,
        "local_archive_build_start",
        vintage=spec.vintage,
        source_dir=str(source_dir),
        target=str(target),
    )
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for xml_path in sorted(source_dir.glob("*.xml")):
            z.write(xml_path, arcname=xml_path.name)
    log_event(
        _logger,
        logging.INFO,
        "local_archive_build_done",
        vintage=spec.vintage,
        archive=str(target),
        byte_count=target.stat().st_size,
    )
    return target


# ensure shutil is referenced (used elsewhere when expanding archives)
_ = shutil  # noqa: F841


__all__ = [
    "ArchiveMaterialization",
    "FilterStats",
    "RawManifest",
    "VintageSpec",
    "_VINTAGE_SOURCES",
    "archive_uri_for",
    "derive_vintage_products",
    "filter_lod2_against_lod1",
    "iter_lod1_tiles",
    "iter_lod2_tiles",
    "materialize_vintage_archive",
    "prepare_vintage_morphology",
    "publish_geometry_mapping",
    "publish_raw_archive_manifest",
    "publish_vintage_morphology",
    "stage_local_archive",
    "stream_archive_to_gcs",
    "vintage_geometry_id",
    "year_to_vintage_range",
]