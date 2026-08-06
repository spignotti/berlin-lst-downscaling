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
exit. No expanded XML ever appears in the bucket.

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
The 2017 stock filter uses :mod:`shapely.strtree.STRtree` with a
vectorized intersection/area check — ``geopandas`` and ``shapely`` are
already project dependencies and the R-tree plus batch geometry math
avoids per-row Python loops.

Memory discipline
-----------------
The runner processes one LoD2 tile at a time and rasterises into
pre-allocated local tile windows inside the canonical 10 m grid.
For 2017 the LoD1 footprint cache is the full in-memory index rather
than a 9-tile sliding window; the cache is built once and reused across
all LoD2 tiles so that edge buildings are always scored against their
true neighbours.  Peak RAM therefore holds the full LoD1 index plus one
LoD2 tile plus one local raster window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import tempfile
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio.windows
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import mapping

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

# Number of LoD2 tiles processed between ``lod2_tile_progress`` log events.
_LOG_PROGRESS_EVERY = 50


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


def _gcs_download_with_retry(blob, target: str) -> None:
    """Download a GCS blob to a local file with retries for transient failures."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _do_download():
        blob.download_to_filename(target)

    _do_download()


def archive_uri_for(raw_root: str, spec: VintageSpec) -> str:
    """Return the canonical GCS URI for *spec*'s archive ZIP."""
    return f"{raw_root.rstrip('/')}/lod_vintages/{spec.vintage}/{spec.archive_filename}"


@contextmanager
def materialize_vintage_archive(
    spec: VintageSpec,
    raw_root: str,
):
    """Context manager: yield the materialised archive ZIP, then delete it.

    Streams the GCS archive ZIP once into a per-vintage
    ``TemporaryDirectory`` so the rest of the pipeline can iterate it as
    a regular local file.  On exit the downloaded ZIP and its directory
    are removed unconditionally; the bucket remains the immutable source
    of truth and the published raw manifest is the only persisted
    provenance for the archive.

    Yields
    ------
    ArchiveMaterialization
        Live materialisation record; ``local_path`` is bound to the
        temp-dir backing file and is deleted when the context exits.
    """
    uri = archive_uri_for(raw_root, spec)
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"lod_archive_{spec.vintage}_")
    target = Path(tmp_dir.name) / spec.archive_filename

    log_event(
        _logger,
        logging.INFO,
        "archive_download_start",
        vintage=spec.vintage,
        archive_uri=uri,
        local=str(target),
    )
    try:
        from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

        bucket_name, key = _parse_gs_uri(uri)
        client = _gcs_client()
        blob = client.bucket(bucket_name).blob(key)
        if not blob.exists():
            raise FileNotFoundError(f"Archive not found in GCS: {uri}")

        _gcs_download_with_retry(blob, str(target))
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

        materialization = ArchiveMaterialization(
            spec=spec,
            local_path=target,
            byte_count=byte_count,
            sha256=sha,
            member_count=len(member_names),
            member_names=member_names,
        )
        yield materialization
    finally:
        try:
            tmp_dir.cleanup()
        except FileNotFoundError:
            pass
        log_event(
            _logger,
            logging.INFO,
            "archive_download_cleaned",
            vintage=spec.vintage,
            archive_uri=uri,
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

    Builds an R-tree over the LoD1 neighbourhood pool, runs a single
    :meth:`STRtree.query` with the ``intersects`` predicate, and computes
    exact intersection areas through Shapely's vectorised
    :func:`shapely.intersection` over the candidate pairs.  This is one
    C-level call per pair (no per-row Python ``iterrows``) and a single
    spatial index per tile.

    Union semantics: when an LoD2 footprint touches multiple LoD1
    footprints the areas accumulate, matching the ``unary_union`` previous
    behaviour in spirit (the *raw* area sum is what the filter cares about,
    not the union polygon).
    """
    from shapely import STRtree
    from shapely import area as shapely_area
    from shapely import intersection as shapely_intersection
    from shapely.geometry import MultiPolygon, Polygon

    stats = FilterStats(input_lod2=len(lod2_buildings))

    lod1_geoms: list[Polygon | MultiPolygon] = []
    for f in lod1_pool:
        if f.footprint is not None and not f.footprint.is_empty:
            lod1_geoms.append(f.footprint)
    stats.input_lod1 = len(lod1_geoms)

    valid_lod2: list[Building] = []
    for b in lod2_buildings:
        if b.footprint is None or b.footprint.is_empty:
            stats.rejected += 1
            stats.rejected_invalid_overlap += 1
            continue
        valid_lod2.append(b)

    if not lod1_geoms or not valid_lod2:
        stats.rejected += len(valid_lod2)
        stats.rejected_low_overlap += len(valid_lod2)
        return [], stats

    lod1_array = np.array(lod1_geoms, dtype=object)
    lod2_array = np.array(
        [b.footprint for b in valid_lod2], dtype=object
    )

    tree = STRtree(lod1_array)
    pair_indices = tree.query(lod2_array, predicate="intersects")
    if pair_indices.shape[1] == 0:
        # No LoD1 footprint touches any LoD2 building on this tile.
        for _ in valid_lod2:
            stats.rejected += 1
            stats.rejected_low_overlap += 1
        return [], stats

    lod2_pair_idx = pair_indices[0]
    lod1_pair_idx = pair_indices[1]
    lod2_geoms_for_pairs = lod2_array[lod2_pair_idx]
    lod1_geoms_for_pairs = lod1_array[lod1_pair_idx]

    inter_geoms = shapely_intersection(
        lod2_geoms_for_pairs, lod1_geoms_for_pairs
    )
    inter_areas = shapely_area(inter_geoms)
    areas = np.asarray(inter_areas, dtype=np.float64)
    np.nan_to_num(areas, copy=False, nan=0.0)

    accum = np.zeros(len(valid_lod2), dtype=np.float64)
    np.add.at(accum, lod2_pair_idx, areas)

    lod2_areas = np.asarray(
        [
            float(b.footprint.area) if b.footprint is not None else 0.0
            for b in valid_lod2
        ],
        dtype=np.float64,
    )
    np.divide(accum, lod2_areas, out=accum, where=lod2_areas > 0)
    keep_mask = accum >= _STOCK_OVERLAP_MIN_FRAC

    retained: list[Building] = []
    for i, b in enumerate(valid_lod2):
        if keep_mask[i]:
            retained.append(b)
            stats.retained += 1
        else:
            stats.rejected += 1
            stats.rejected_low_overlap += 1

    log_event(
        _logger,
        logging.DEBUG,
        "filter_tile_done",
        tile_key=tile_key,
        candidate_pairs=int(pair_indices.shape[1]),
        retained=stats.retained,
        rejected=stats.rejected,
    )
    return retained, stats


# ── rasterisation accumulator ───────────────────────────────────────


class _Accumulator:
    """Pre-allocated 10 m-grid statistics buffers for one vintage product.

    Rasterisation is performed into a *local* window per CityGML tile
    rather than against the full canonical Berlin grid.  Each building is
    still rasterised individually so per-building ``n_cells`` and the
    existing height/area accumulation semantics are preserved exactly.
    The global grid stays in memory; only the per-tile allocations are
    tiny.
    """

    def __init__(self, grid: GeoBox) -> None:
        shape = (grid.shape.y, grid.shape.x)
        self.grid = grid
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)
        self.area = np.zeros(shape, dtype=np.float64)
        self.max = np.zeros(shape, dtype=np.float32)

    def add_tile(self, buildings: list[Building]) -> None:
        # Decide the local window once per CityGML tile from the union of
        # building bounds; only build a window if the tile intersects the
        # grid at all.
        window = _local_window_for_buildings(buildings, self.grid)
        if window is None:
            return

        local_transform = rasterio.windows.transform(window, self.grid.transform)
        local_h = int(window.height)
        local_w = int(window.width)
        if local_h <= 0 or local_w <= 0:
            return

        row_off = int(window.row_off)
        col_off = int(window.col_off)
        sum_slice = self.sum[row_off : row_off + local_h, col_off : col_off + local_w]
        sumsq_slice = self.sumsq[
            row_off : row_off + local_h, col_off : col_off + local_w
        ]
        count_slice = self.count[
            row_off : row_off + local_h, col_off : col_off + local_w
        ]
        area_slice = self.area[
            row_off : row_off + local_h, col_off : col_off + local_w
        ]
        max_slice = self.max[row_off : row_off + local_h, col_off : col_off + local_w]

        for bldg in buildings:
            if bldg.footprint is None or bldg.measured_height is None:
                continue
            try:
                geom = mapping(bldg.footprint)
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(local_h, local_w),
                    transform=local_transform,
                    fill=0,
                    dtype=np.uint8,
                )
                if mask is None:
                    continue
            except Exception:  # noqa: S112 — skip buildings with bad geometry
                continue
            cells = mask > 0
            n_cells = int(np.sum(cells))
            if n_cells == 0:
                continue

            h = bldg.measured_height
            sum_slice[cells] += h
            sumsq_slice[cells] += h * h
            count_slice[cells] += 1
            area_slice[cells] += bldg.footprint.area / n_cells
            np.maximum.at(max_slice, cells, h)

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


def _local_window_for_buildings(
    buildings: list[Building], grid: GeoBox
) -> Window | None:
    """Compute a minimal canonical-grid window covering *buildings*.

    Returns ``None`` when the buildings do not intersect the grid.
    The window covers the union of building footprints extended by one
    pixel of safety, clipped to the grid extent, so neighbouring tiles
    see overlapping footprints but each building still rasterises exactly
    once (the caller only invokes this function once per source tile).
    """
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    found = False
    for b in buildings:
        if b.footprint is None or b.footprint.is_empty or not b.footprint.is_valid:
            continue
        bx0, by0, bx1, by1 = b.footprint.bounds
        if bx0 < minx:
            minx = bx0
        if by0 < miny:
            miny = by0
        if bx1 > maxx:
            maxx = bx1
        if by1 > maxy:
            maxy = by1
        found = True
    if not found:
        return None

    bbox = grid.extent.boundingbox
    grid_minx = bbox.left
    grid_miny = bbox.bottom
    grid_maxx = bbox.right
    grid_maxy = bbox.top
    if maxx <= grid_minx or minx >= grid_maxx or maxy <= grid_miny or miny >= grid_maxy:
        return None

    win = rasterio.windows.from_bounds(
        max(minx, grid_minx),
        max(miny, grid_miny),
        min(maxx, grid_maxx),
        min(maxy, grid_maxy),
        transform=grid.transform,
    )
    win = win.round_offsets().round_lengths()
    if win.height <= 0 or win.width <= 0:
        return None
    row0 = max(0, int(win.row_off))
    col0 = max(0, int(win.col_off))
    row1 = min(grid.shape.y, int(win.row_off + win.height))
    col1 = min(grid.shape.x, int(win.col_off + win.width))
    if row1 <= row0 or col1 <= col0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)  # type: ignore[call-arg]


# ── product publisher (memory-bounded) ──────────────────────────────


@dataclass
class MorphologyArtifacts:
    """Published artefacts of one historical vintage run."""

    cog_uri: str
    stac_uri: str
    provenance_uri: str
    completion_uri: str
    archive_uri: str
    archive_sha256: str
    archive_byte_count: int
    raw_manifest_uri: str


def prepare_vintage_morphology(
    vintage: int,
    raw_root: str,
    *,
    lod1_mat: ArchiveMaterialization | None = None,
    lod2_mat: ArchiveMaterialization,
    source_root: str | None = None,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
) -> tuple[PreparedSecondaryProduct, FilterStats]:
    """Build the canonical morphology product for *vintage* by streaming archives.

    The archive ZIPs are consumed as live :class:`ArchiveMaterialization`
    objects coming out of :func:`materialize_vintage_archive`; no extra
    hashing or member-listing is performed inside this function.

    For vintage 2017 the LoD2-2021 stock is filtered against the LoD1
    footprints via :func:`filter_lod2_against_lod1`; the other vintages
    consume the raw LoD2 stock directly.

    Parameters
    ----------
    source_root :
        Optional explicit root under which the raw archive manifest
        JSON lives (``<source_root>/ard/static/sources/lod_vintages/raw_manifest_<vintage>.json``).
        Defaults to ``raw_root``.
    """
    spec = _VINTAGE_SOURCES[vintage]
    source_vintage = 2021 if vintage == 2017 else vintage

    if lod2_mat is None:
        raise ValueError("lod2_mat is required")
    if vintage == 2017 and lod1_mat is None:
        raise ValueError("Vintage 2017 requires lod1_mat")

    manifest_root = source_root or raw_root
    grid = grid or canon_grid_10m()
    c_hash = config_hash_for_vintage(vintage)

    accumulator = _Accumulator(grid)
    filter_stats = FilterStats()
    tile_count = 0
    log_progress_step = max(1, _LOG_PROGRESS_EVERY)

    lod1_full_index: dict[str, list[Footprint]] = {}
    if vintage == 2017:
        assert lod1_mat is not None  # noqa: S101 — invariant: caller provides for 2017
        log_event(_logger, logging.INFO, "lod1_index_load_start", vintage=2017)
        for tile_key, footprints in iter_lod1_tiles(
            lod1_mat.local_path,
            grid=grid,
            max_tiles=smoke_tile_count,
        ):
            lod1_full_index[tile_key] = footprints
        log_event(
            _logger,
            logging.INFO,
            "lod1_index_load_done",
            n_tiles=len(lod1_full_index),
            n_footprints=sum(len(v) for v in lod1_full_index.values()),
        )

    log_event(
        _logger,
        logging.INFO,
        "lod2_tile_iter_start",
        vintage=vintage,
        archive_uri=archive_uri_for(raw_root, lod2_mat.spec),
    )
    t_iter = time.perf_counter()
    for lod2_tile_key, buildings in iter_lod2_tiles(
        lod2_mat.local_path,
        grid=grid,
        max_tiles=smoke_tile_count,
    ):
        tile_count += 1
        if vintage == 2017:
            neighbour_keys = [
                lod2_tile_key,
                *_neighbour_tile_keys(lod2_tile_key),
            ]
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
            accumulator.add_tile(buildings)

        if tile_count % log_progress_step == 0:
            log_event(
                _logger,
                logging.INFO,
                "lod2_tile_progress",
                vintage=vintage,
                tile_count=tile_count,
                retained=filter_stats.retained,
                rejected=filter_stats.rejected,
                elapsed_s=round(time.perf_counter() - t_iter, 1),
            )
    log_event(
        _logger,
        logging.INFO,
        "lod2_tile_iter_done",
        vintage=vintage,
        tile_count=tile_count,
        elapsed_s=round(time.perf_counter() - t_iter, 1),
    )

    ds = accumulator.to_dataset()
    valid_frac = float(
        np.count_nonzero(~np.isnan(ds["building_height_mean"].values))
    ) / ds["building_height_mean"].size

    archive_uri = archive_uri_for(raw_root, lod2_mat.spec)
    raw_manifest_uri = (
        f"{manifest_root.rstrip('/')}/ard/static/sources/lod_vintages/"
        f"raw_manifest_{vintage}.json"
    )

    source_metadata = {
        "vintage": vintage,
        "source_vintage": source_vintage,
        "level": spec.level,
        "feed_label": spec.feed_label,
        "archive": {
            "uri": archive_uri,
            "sha256": lod2_mat.sha256,
            "byte_count": lod2_mat.byte_count,
            "member_count": lod2_mat.member_count,
        },
        "raw_manifest_uri": raw_manifest_uri,
        "tile_count": tile_count,
        "stock_filter": filter_stats.as_dict() if vintage == 2017 else None,
    }

    if vintage == 2017 and lod1_mat is not None:
        source_metadata["lod1_archive"] = {
            "uri": archive_uri_for(raw_root, lod1_mat.spec),
            "sha256": lod1_mat.sha256,
            "byte_count": lod1_mat.byte_count,
            "member_count": lod1_mat.member_count,
        }

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
    )


def publish_vintage_morphology(
    vintage: int,
    source_root: str,
    run_id: str,
    *,
    lod1_mat: ArchiveMaterialization | None = None,
    lod2_mat: ArchiveMaterialization,
    raw_root: str,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
) -> tuple[MorphologyArtifacts, FilterStats]:
    """Prepare and finalise the *vintage* morphology product."""
    grid = grid or canon_grid_10m()
    prepared, stats = prepare_vintage_morphology(
        vintage,
        raw_root,
        lod1_mat=lod1_mat,
        lod2_mat=lod2_mat,
        source_root=source_root,
        grid=grid,
        smoke_tile_count=smoke_tile_count,
    )
    from berlin_lst_downscaling.data.secondary.paths import source_product_dir

    prod_dir = source_product_dir(source_root, "lod2_morphology", str(vintage))
    artifacts = finalize_secondary_product(prepared, grid, prod_dir, run_id)

    raw_manifest_uri = prepared.source_metadata["raw_manifest_uri"]
    archive_meta = prepared.source_metadata["archive"]
    morph_artifacts = MorphologyArtifacts(
        cog_uri=artifacts.cog_uri,
        stac_uri=artifacts.stac_uri,
        provenance_uri=artifacts.provenance_uri,
        completion_uri=artifacts.completion_uri,
        archive_uri=archive_meta["uri"],
        archive_sha256=archive_meta["sha256"],
        archive_byte_count=archive_meta["byte_count"],
        raw_manifest_uri=raw_manifest_uri,
    )
    log_event(
        _logger,
        logging.INFO,
        "vintage_morphology_published",
        vintage=vintage,
        cog_uri=artifacts.cog_uri,
        retained=stats.retained,
        rejected=stats.rejected,
    )
    return morph_artifacts, stats


# ── geometry mapping ────────────────────────────────────────────────


def publish_geometry_mapping(
    metadata_root: str,
    *,
    vintage_artifacts: dict[int, dict],
    legacy_source_root: str = "gs://berlin-lst-data/static/sources/full",
    legacy_derived_root: str = "gs://berlin-lst-data/static/derived/full",
    published_at: str | None = None,
) -> str:
    """Publish the year → vintage carry-forward mapping artefact.

    Merges explicitly published historical vintage artifacts **and** the
    existing legacy 2024 source/derived bundle so the final mapping
    always covers the full 2017–2026 contract.

    Requires finalized 2024 source + derived artifacts to exist before
    publishing.  Smoke runs and ``--skip-derived`` processing should
    skip this call rather than produce a partial map.
    """
    from berlin_lst_downscaling.data.io.storage import exists
    from berlin_lst_downscaling.data.secondary.paths import (
        derived_product_cog,
        source_product_cog,
    )

    published_at = published_at or datetime.now(UTC).isoformat()

    legacy_geometry_id = "dgm1-2021__lod2-2024__vh-2020"

    legacy_missing = [
        uri
        for uri in [
            source_product_cog(legacy_source_root, "lod2_morphology", "2024"),
            f"{legacy_source_root.rstrip('/')}/ard/static/sources/lod2_morphology/2024/complete.json",
            derived_product_cog(legacy_derived_root, "building_dsm", legacy_geometry_id),
            f"{legacy_derived_root.rstrip('/')}/ard/static/derived/building_dsm/{legacy_geometry_id}/complete.json",
            derived_product_cog(legacy_derived_root, "combined_dsm", legacy_geometry_id),
            f"{legacy_derived_root.rstrip('/')}/ard/static/derived/combined_dsm/{legacy_geometry_id}/complete.json",
            derived_product_cog(legacy_derived_root, "horizon_building", legacy_geometry_id),
            f"{legacy_derived_root.rstrip('/')}/ard/static/derived/horizon_building/{legacy_geometry_id}/complete.json",
            derived_product_cog(legacy_derived_root, "svf", legacy_geometry_id),
            f"{legacy_derived_root.rstrip('/')}/ard/static/derived/svf/{legacy_geometry_id}/complete.json",
        ]
        if not exists(uri)
    ]
    if legacy_missing:
        raise FileNotFoundError(
            "Geometry mapping requires published legacy 2024 bundle but "
            f"these artifacts are missing: {legacy_missing}"
        )

    legacy_uri = source_product_cog(legacy_source_root, "lod2_morphology", "2024")
    legacy_src_dir = (
        f"{legacy_source_root.rstrip('/')}/ard/static/sources/lod2_morphology/2024"
    )

    merged_vintage_artifacts = dict(vintage_artifacts)
    merged_vintage_artifacts[2024] = {
        "geometry_id": legacy_geometry_id,
        "lod2_morphology": legacy_uri,
        "stac": f"{legacy_src_dir}/lod2_morphology_2024.stac.json",
        "provenance": f"{legacy_src_dir}/provenance.json",
        "completion": f"{legacy_src_dir}/complete.json",
        "legacy": True,
    }

    selected_vintages = set(int(v) for v in merged_vintage_artifacts)
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
                **({"legacy_bundle": True} if art.get("legacy") else {}),
            }
            for vintage, art in sorted(merged_vintage_artifacts.items())
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


_DERIVED_PRODUCTS = ("building_dsm", "combined_dsm", "horizon_building", "svf")


def derive_vintage_products(
    vintage: int,
    source_root: str,
    derived_root: str,
    run_id: str,
    *,
    upstream_source_root: str | None = None,
    upstream_derived_root: str | None = None,
    grid: GeoBox | None = None,
    max_radius_m: float = 200.0,
    svf_max_radius: int = 3,
    svf_n_directions: int = 16,
    products: list[str] | None = None,
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

    When ``products`` is given, only those products are computed.
    ``building_dsm`` and ``horizon_building`` are then resolved from the
    existing finalised artefacts instead of being recomputed, so a
    derived-only repair never touches building outputs.  ``svf`` always
    fingerprints the actual ``combined_dsm`` config hash rather than the
    geometry ID, so it rebuilds when the vegetation input changes.

    Parameters
    ----------
    upstream_source_root :
        Optional override for ``source_root`` used to resolve the
        finalised terrain and vegetation-height products.  Defaults to
        *source_root*.
    upstream_derived_root :
        Optional override for ``derived_root`` used to locate the
        vintage-agnostic ``vegetation_dsm`` predecessor.  Defaults to
        *derived_root*.
    products :
        Optional subset of ``_DERIVED_PRODUCTS`` to compute.  Defaults to
        all four.
    """
    from berlin_lst_downscaling.data.io.storage import exists
    from berlin_lst_downscaling.data.secondary import dsm as dsm_mod
    from berlin_lst_downscaling.data.secondary import horizon as horizon_mod
    from berlin_lst_downscaling.data.secondary import svf as svf_mod
    from berlin_lst_downscaling.data.secondary.paths import (
        derived_product_cog,
        derived_product_dir,
        source_product_cog,
        source_product_provenance,
    )
    from berlin_lst_downscaling.data.secondary.source_products import (
        resolve_source_products,
    )

    grid = grid or canon_grid_10m()
    geometry_id = vintage_geometry_id(vintage)
    upstream_src = upstream_source_root or source_root
    veg_dsm_root = upstream_derived_root or derived_root

    log_event(
        _logger,
        logging.INFO,
        "derive_vintage_start",
        vintage=vintage,
        geometry_id=geometry_id,
        source_root=source_root,
        upstream_source_root=upstream_src,
        upstream_derived_root=veg_dsm_root,
    )

    # 1. Resolve vintage-agnostic upstream products from upstream roots.
    upstream_report = resolve_source_products(
        upstream_src,
        {
            "terrain_height": "2021",
            "vegetation_height": "2020",
        },
    )
    if not upstream_report.ok:
        raise ValueError(
            "Upstream source products missing: "
            + "; ".join(upstream_report.errors)
        )
    upstream_map = {
        f"{r.source}/{r.revision}": r for r in upstream_report.resolved
    }
    terrain = upstream_map.get("terrain_height/2021")
    vh = upstream_map.get("vegetation_height/2020")
    if terrain is None or vh is None:
        missing = [
            s
            for s, v in [
                ("terrain_height/2021", terrain),
                ("vegetation_height/2020", vh),
            ]
            if v is None
        ]
        raise ValueError(f"Upstream source products missing: {missing}")

    # 2. The freshly-published morphology lives in source_root.
    lod2_cog = source_product_cog(source_root, "lod2_morphology", str(vintage))
    if not exists(lod2_cog):
        raise ValueError(f"Morphology product missing: {lod2_cog}")
    lod2_prov_uri = source_product_provenance(source_root, "lod2_morphology", str(vintage))
    lod2_config_hash = _read_config_hash(lod2_prov_uri)

    upstream_hashes = {
        "terrain": terrain.config_hash,
        "lod2": lod2_config_hash,
        "vh": vh.config_hash,
    }

    requested = set(products) if products is not None else set(_DERIVED_PRODUCTS)
    unknown = requested - set(_DERIVED_PRODUCTS)
    if unknown:
        raise ValueError(f"Unsupported derived products: {sorted(unknown)}")

    artefacts: dict[str, str] = {}
    bldg_cog_uri = ""

    # 3. building_dsm — computed when requested; otherwise resolved from the
    #    existing finalised product so repair runs never touch building data.
    if "building_dsm" in requested:
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
        bldg_cog_uri = building_dsm_artifacts.cog_uri
        artefacts["building_dsm"] = bldg_cog_uri
    else:
        bldg_cog_uri = derived_product_cog(derived_root, "building_dsm", geometry_id)
        bldg_complete = (
            f"{derived_product_dir(derived_root, 'building_dsm', geometry_id)}/complete.json"
        )
        if not exists(bldg_cog_uri) or not exists(bldg_complete):
            raise ValueError(
                "Existing building_dsm is required but incomplete at "
                f"{bldg_cog_uri}; combined_dsm cannot be produced."
            )

    combined_dsm = None
    combined_artifacts = None

    # 4. combined_dsm — uses vegetation_dsm/2024 from upstream derived root
    if "combined_dsm" in requested:
        veg_dsm_geom = "dgm1-2021__lod2-2024__vh-2020"
        veg_dsm_cog = derived_product_cog(veg_dsm_root, "vegetation_dsm", veg_dsm_geom)
        veg_dsm_complete = (
            f"{derived_product_dir(veg_dsm_root, 'vegetation_dsm', veg_dsm_geom)}/complete.json"
        )
        if not exists(veg_dsm_cog) or not exists(veg_dsm_complete):
            raise ValueError(
                "Required vegetation_dsm/2024 is missing at "
                f"{veg_dsm_cog}; combined_dsm cannot be produced."
            )

        combined_dir = derived_product_dir(derived_root, "combined_dsm", geometry_id)
        combined_dsm = dsm_mod.prepare_combined_dsm(
            bldg_cog_uri,
            veg_dsm_cog,
            derived_root,
            run_id,
            item_key=geometry_id,
            upstream_hashes=upstream_hashes,
            grid=grid,
        )
        combined_artifacts = finalize_secondary_product(
            combined_dsm, grid, combined_dir, run_id
        )
        artefacts["combined_dsm"] = combined_artifacts.cog_uri

    # 5. horizon_building (from building_dsm)
    if "horizon_building" in requested:
        horizon_dir = derived_product_dir(derived_root, "horizon_building", geometry_id)
        horizon_bldg = horizon_mod.prepare_horizon(
            bldg_cog_uri,
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

    # 6. svf (from combined_dsm)
    if "svf" in requested:
        if combined_dsm is None or combined_artifacts is None:
            raise ValueError("svf requires combined_dsm in requested products")
        svf_dir = derived_product_dir(derived_root, "svf", geometry_id)
        svf_product = svf_mod.prepare_svf(
            combined_artifacts.cog_uri,
            derived_root,
            run_id,
            item_key=geometry_id,
            # lineage: fingerprint the actual combined DSM input, so a
            # corrected vegetation source rebuilds this vintage's SVF.
            upstream_hash=combined_dsm.config_hash,
            max_radius=svf_max_radius,
            n_directions=svf_n_directions,
            grid=grid,
        )
        svf_artifacts = finalize_secondary_product(svf_product, grid, svf_dir, run_id)
        artefacts["svf"] = svf_artifacts.cog_uri

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


def _read_config_hash(provenance_uri: str) -> str:
    """Read ``config_hash`` out of a published provenance.json; return "" on miss."""
    from berlin_lst_downscaling.data.io import read_bytes

    try:
        payload = json.loads(read_bytes(provenance_uri))
        return str(payload.get("config_hash", ""))
    except Exception:  # noqa: S110 — fall through, surface as missing hash
        return ""


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


__all__ = [
    "ArchiveMaterialization",
    "FilterStats",
    "MorphologyArtifacts",
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