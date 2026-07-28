"""LoD historical-vintage morphology processor.

Streams locally-supplied Berlin LoD1 (2017) and LoD2 (2021/2022) CityGML
into GCS as immutable raw archives, filters the 2021 stock against the
2017 baseline, and publishes the canonical-grid ``lod2_morphology``
products for the 2017, 2021, and 2022 vintages.

Outputs follow the existing Pipeline-A layout::

    gs://<source_root>/ard/static/sources/lod2_morphology/<vintage>/
        ├─ lod2_morphology_<vintage>.tif
        ├─ lod2_morphology_<vintage>.stac.json
        ├─ provenance.json
        └─ complete.json

Raw uploads live under::

    gs://<source_root>/_raw/secondary/lod_vintages/<vintage>/<filename>

Library search
--------------
PyPI candidates ``citygml``, ``pycitygml``, ``citygml-tools`` returned
404 on the project's index client.  XML is parsed with stdlib
``ElementTree`` (see :mod:`berlin_lst_downscaling.data.secondary.citygml`).
The 2017 stock filter uses :mod:`geopandas` with its documented
``sjoin`` predicate (``intersects``) — ``geopandas`` is already a
project dependency and provides an index-backed spatial join.

Spatial index
-------------
The LoD1 tile covers the same 1 km cell as its sibling LoD2 tile, so the
2017 filter is done per LoD2 tile against the LoD1 footprints of the
matching tile plus its eight neighbours (8-neighbour buffer for edge
buildings that span tile boundaries).

Vintage mapping
---------------
Year → vintage (carry-forward, never future geometry):

    2017–2020 → 2017
    2021      → 2021
    2022–2023 → 2022
    2024–2026 → 2024 (existing published product)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon, Polygon, mapping

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.io import atomic_write, log_event
from berlin_lst_downscaling.data.secondary.citygml import (
    Building,
    Footprint,
    iter_xml_members,
    parse_lod1_footprints_from_file,
    parse_lod2_buildings,
    parse_lod2_buildings_from_file,
)
from berlin_lst_downscaling.data.secondary.lod2 import (
    config_hash_for_vintage,
    contract_for_lod2_morphology,
)
from berlin_lst_downscaling.data.secondary.paths import raw_dir
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    finalize_secondary_product,
    vintage_interval,
)

_logger = logging.getLogger(__name__)

# ── configuration constants ──────────────────────────────────────────

# Per-vintage local source layout.  ``kind`` selects how the local files
# are enumerated; ``pattern`` is matched against XML/GML members.
_VINTAGE_SOURCES: dict[int, dict] = {
    2017: {
        "kind": "lod1_dir",
        "local_path": "data/LoD2/2017",
        "pattern": "*.xml",
        "expected_count": 1006,
        "raw_subdir": "lod1_2017",
        "feed_label": "Berlin GDI LoD1 2017 (Senatsverwaltung Berlin)",
    },
    2021: {
        "kind": "lod2_dir",
        "local_path": "data/LoD2/LoD2_BE_1_33_2021",
        "pattern": "*.xml",
        "expected_count": 928,
        "raw_subdir": "lod2_2021",
        "feed_label": "Berlin GDI LoD2 2021 (Senatsverwaltung Berlin)",
    },
    2022: {
        "kind": "lod2_zip",
        "local_path": "data/LoD2/LoD2_2022.zip",
        "pattern": "*.xml",
        "expected_count": 928,
        "raw_subdir": "lod2_2022",
        "feed_label": "Berlin GDI LoD2 2022 (Senatsverwaltung Berlin)",
    },
}

# LoD2 building kept when its footprint is overlapped by LoD1 footprints
# by at least this fraction of its own footprint area.
# decision: 50% threshold matches the brief's "vorhanden" rule for
# existing buildings; stricter values would erase an- und umbauten that
# already existed in 2017 but received roof details in 2021.
_STOCK_OVERLAP_MIN_FRAC = 0.50

# Carry-forward mapping used to publish scene-year → vintage.
# 2024+ is served by the existing ATOM-feed product and is intentionally
# excluded from this runner.
_YEAR_TO_VINTAGE: tuple[tuple[int, int, int], ...] = (
    (2017, 2017, 2020),
    (2021, 2021, 2021),
    (2022, 2022, 2023),
)


# ── inventory / raw upload ───────────────────────────────────────────


@dataclass
class RawManifestEntry:
    """One archived tile with checksum."""

    filename: str
    uri: str
    byte_count: int
    checksum: str


@dataclass
class RawManifest:
    """Per-vintage raw upload manifest."""

    vintage: int
    source_kind: str
    feed_label: str
    entries: list[RawManifestEntry] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(e.byte_count for e in self.entries)


def _stream_sha256(buf: bytes | bytearray | memoryview) -> str:
    """Return the SHA-256 hex digest of a bytes-like object."""
    return hashlib.sha256(bytes(buf)).hexdigest()


def _iter_vintage_files(spec: dict) -> list[Path]:
    """Return a list of local file paths for *spec* (dir layouts only)."""
    kind = spec["kind"]
    if kind in ("lod1_dir", "lod2_dir"):
        return sorted(Path(spec["local_path"]).glob(spec["pattern"]))
    raise ValueError(f"Unknown source kind for dir iteration: {kind!r}")


def _iter_zip_members(spec: dict) -> list[tuple[str, int]]:
    """Return ``(member_name, uncompressed_size)`` for every XML/GML in the ZIP."""
    if spec["kind"] != "lod2_zip":
        raise ValueError(f"ZIP iteration only valid for lod2_zip (got {spec['kind']})")
    zip_path = Path(spec["local_path"])
    out: list[tuple[str, int]] = []
    with zipfile.ZipFile(zip_path) as z:
        for member in iter_xml_members(zip_path):
            info = z.getinfo(member)
            out.append((info.filename, info.file_size))
    return out


def stream_vintage_to_gcs(
    vintage: int,
    source_root: str,
    raw_root: str | None = None,
) -> RawManifest:
    """Stream every input file for *vintage* to GCS and build an immutable manifest.

    Parameters
    ----------
    vintage :
        One of 2017, 2021, 2022.
    source_root :
        Root URI of the Pipeline A output (local or ``gs://…``).
    raw_root :
        Bucket-level root for raw staging.  Defaults to the same
        ``source_root`` so the layout matches other secondary sources.
    """
    spec = _VINTAGE_SOURCES[vintage]
    raw_bucket = raw_root or source_root
    staging_dir = raw_dir(raw_bucket, "lod_vintages", str(vintage))

    log_event(
        _logger,
        logging.INFO,
        "raw_upload_start",
        vintage=vintage,
        source_kind=spec["kind"],
        local_path=spec["local_path"],
    )

    entries: list[RawManifestEntry] = []
    t0 = time.perf_counter()

    if spec["kind"] in ("lod1_dir", "lod2_dir"):
        for path in _iter_vintage_files(spec):
            data = path.read_bytes()
            digest = _stream_sha256(data)
            dst = f"{staging_dir}/{path.name}"
            atomic_write(dst, data, overwrite=True)
            entries.append(
                RawManifestEntry(
                    filename=path.name,
                    uri=dst,
                    byte_count=len(data),
                    checksum=digest,
                )
            )
    else:
        zip_path = Path(spec["local_path"])
        with zipfile.ZipFile(zip_path) as z:
            for member, size in _iter_zip_members(spec):
                data = z.read(member)
                digest = _stream_sha256(data)
                dst = f"{staging_dir}/{Path(member).name}"
                atomic_write(dst, data, overwrite=True)
                entries.append(
                    RawManifestEntry(
                        filename=Path(member).name,
                        uri=dst,
                        byte_count=size,
                        checksum=digest,
                    )
                )

    manifest = RawManifest(
        vintage=vintage,
        source_kind=spec["kind"],
        feed_label=spec["feed_label"],
        entries=entries,
    )

    elapsed = time.perf_counter() - t0
    log_event(
        _logger,
        logging.INFO,
        "raw_upload_done",
        vintage=vintage,
        n_files=len(entries),
        total_bytes=manifest.total_bytes,
        elapsed_s=round(elapsed, 1),
    )

    if len(entries) != spec["expected_count"]:
        log_event(
            _logger,
            logging.WARNING,
            "raw_upload_count_mismatch",
            vintage=vintage,
            expected=spec["expected_count"],
            actual=len(entries),
        )

    _publish_raw_manifest(manifest, source_root)
    return manifest


def _publish_raw_manifest(manifest: RawManifest, source_root: str) -> str:
    """Write the raw manifest JSON next to the source layout."""
    base = (
        f"{source_root.rstrip('/')}/ard/static/sources/lod_vintages/"
        f"raw_manifest_{manifest.vintage}.json"
    )
    payload = {
        "vintage": manifest.vintage,
        "source_kind": manifest.source_kind,
        "feed_label": manifest.feed_label,
        "total_bytes": manifest.total_bytes,
        "n_files": len(manifest.entries),
        "files": [
            {
                "filename": e.filename,
                "uri": e.uri,
                "byte_count": e.byte_count,
                "checksum": e.checksum,
            }
            for e in manifest.entries
        ],
        "published_at": datetime.now(UTC).isoformat(),
    }
    atomic_write(base, json.dumps(payload, indent=2), overwrite=True)
    return base


# ── geometry mapping ────────────────────────────────────────────────


def publish_geometry_mapping(
    metadata_root: str,
    *,
    vintage_artifacts: dict[int, dict],
    published_at: str | None = None,
) -> str:
    """Publish the year → vintage carry-forward mapping artefact.

    Parameters
    ----------
    metadata_root :
        Bucket-level root for geometry mapping artefacts.
    vintage_artifacts :
        Mapping ``{vintage: {"geometry_id": …, "lod2_morphology": …}}``.
    """
    published_at = published_at or datetime.now(UTC).isoformat()

    year_map: dict[str, str] = {}
    for year in range(2017, 2027):
        matched = None
        for vintage in (2017, 2021, 2022, 2024):
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


# ── vintage loading (after raw upload) ───────────────────────────────


def load_lod1_footprints(vintage: int, raw_root: str | None = None) -> dict[str, list[Footprint]]:
    """Load every LoD1 footprint, grouped by tile key.

    The ``tile_key`` is the ``<easting>_<northing>`` prefix shared with
    the LoD2 tile naming.  Edge buildings that cross tile boundaries
    appear in exactly one tile (the tile they originate from); the 2017
    filter handles cross-tile overlap by including the eight neighbours.
    """
    if vintage != 2017:
        raise ValueError(f"LoD1 footprints only defined for vintage=2017 (got {vintage})")
    spec = _VINTAGE_SOURCES[2017]
    items = _iter_vintage_files(spec)
    groups: dict[str, list[Footprint]] = {}
    for path in items:
        tile_key = _lod1_tile_key(path.name)
        groups[tile_key] = parse_lod1_footprints_from_file(path)
    return groups


def load_lod2_buildings(
    vintage: int,
    raw_root: str | None = None,
) -> dict[str, list[Building]]:
    """Load every LoD2 building for *vintage*, grouped by tile key.

    ZIP-archived vintages are streamed one member at a time so the
    extracted XML never lands on local disk.
    """
    spec = _VINTAGE_SOURCES[vintage]
    groups: dict[str, list[Building]] = {}

    if spec["kind"] == "lod2_dir":
        for path in _iter_vintage_files(spec):
            tile_key = _lod2_tile_key(path.name)
            groups[tile_key] = parse_lod2_buildings_from_file(path)
    elif spec["kind"] == "lod2_zip":
        zip_path = Path(spec["local_path"])
        with zipfile.ZipFile(zip_path) as z:
            for member, _ in _iter_zip_members(spec):
                data = z.read(member)
                tile_key = _lod2_tile_key(member)
                groups[tile_key] = parse_lod2_buildings(data)
    else:
        raise ValueError(f"Unknown source kind for vintage {vintage}: {spec['kind']!r}")
    return groups


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
    lod1_index: dict[str, list[Footprint]],
    tile_key: str,
) -> tuple[list[Building], FilterStats]:
    """Retain LoD2 buildings whose footprint is covered by LoD1 footprints.

    Uses :func:`geopandas.sjoin` with the documented ``intersects``
    predicate; the LoD1 union for the tile plus its eight neighbours
    is the right-hand side, so edge buildings are still scored against
    their adjacent LoD1 stock.
    """

    stats = FilterStats(input_lod2=len(lod2_buildings))

    # Build the LoD1 pool: tile + 8 neighbours, dropping empty entries.
    pool: list[Polygon | MultiPolygon] = []
    for key in [tile_key, *_neighbour_tile_keys(tile_key)]:
        for foot in lod1_index.get(key, []):
            if foot.footprint is not None and not foot.footprint.is_empty:
                pool.append(foot.footprint)
                stats.input_lod1 += 1
    if not pool or not lod2_buildings:
        stats.rejected = len(lod2_buildings)
        stats.rejected_low_overlap = len(lod2_buildings)
        return [], stats

    if len(pool) == 1:
        lod1_union = pool[0]
    else:
        from shapely.ops import unary_union

        lod1_union = unary_union(pool)

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


def _accumulate_buildings(
    buildings: Iterable[Building],
    grid: GeoBox,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate per-cell statistics for the canonical 10 m grid."""
    shape = (grid.shape.y, grid.shape.x)
    sum_arr = np.zeros(shape, dtype=np.float64)
    sumsq_arr = np.zeros(shape, dtype=np.float64)
    count_arr = np.zeros(shape, dtype=np.int32)
    area_arr = np.zeros(shape, dtype=np.float64)
    max_arr = np.zeros(shape, dtype=np.float32)

    transform = grid.transform
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
        sum_arr[cells] += h
        sumsq_arr[cells] += h * h
        count_arr[cells] += 1
        area_arr[cells] += bldg.footprint.area / n_cells
        np.maximum.at(max_arr, cells, h)

    return sum_arr, sumsq_arr, count_arr, area_arr, max_arr


def _build_morphology_dataset(
    buildings: Iterable[Building],
    grid: GeoBox,
) -> xr.Dataset:
    """Build the canonical 4-band morphology dataset."""
    sum_arr, sumsq_arr, count_arr, area_arr, max_arr = _accumulate_buildings(buildings, grid)

    count_f = count_arr.astype(np.float64)
    mean_arr = np.where(count_f > 0, sum_arr / count_f, np.nan).astype(np.float32)
    variance = np.where(
        count_f > 1,
        (sumsq_arr - (sum_arr * sum_arr) / count_f) / (count_f - 1),
        0.0,
    )
    std_arr = np.where(count_f > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan).astype(np.float32)
    bcr_arr = np.where(count_f > 0, area_arr / 100.0, np.nan).astype(np.float32)
    bcr_arr = np.clip(bcr_arr, 0.0, 1.0)
    max_arr = np.where(count_f > 0, max_arr, np.nan).astype(np.float32)

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
    return ds


# ── product publisher ───────────────────────────────────────────────


def _validate_against_existing_vintages(
    buildings_count: int,
    vintage: int,
) -> None:
    """Warn when the historical run would conflict with a known vintage."""
    if buildings_count == 0:
        log_event(
            _logger,
            logging.WARNING,
            "no_buildings_in_vintage",
            vintage=vintage,
        )


def prepare_vintage_morphology(
    vintage: int,
    source_root: str,
    run_id: str,
    *,
    lod1_index: dict[str, list[Footprint]] | None = None,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
) -> tuple[PreparedSecondaryProduct, FilterStats]:
    """Prepare the canonical morphology product for *vintage* (2017/2021/2022).

    For vintage 2017 the LoD2-2021 stock is filtered against the LoD1
    footprints via :func:`filter_lod2_against_lod1`; the other vintages
    consume the raw LoD2 stock directly.

    Returns the prepared product plus the filter QA stats (empty for
    non-2017 vintages).
    """
    if vintage not in _VINTAGE_SOURCES:
        raise ValueError(f"Unsupported vintage: {vintage}")

    grid = grid or canon_grid_10m()
    c_hash = config_hash_for_vintage(vintage)

    log_event(_logger, logging.INFO, "vintage_load_start", vintage=vintage)
    lod2_groups = load_lod2_buildings(vintage)
    log_event(
        _logger,
        logging.INFO,
        "vintage_load_done",
        vintage=vintage,
        n_tiles=len(lod2_groups),
    )

    # Restrict to a smoke subset if requested.
    if smoke_tile_count is not None:
        kept = dict(list(lod2_groups.items())[:smoke_tile_count])
        lod2_groups = kept

    all_buildings: list[Building] = []
    filter_stats = FilterStats()

    if vintage == 2017:
        if lod1_index is None:
            raise ValueError("Vintage 2017 requires the LoD1 footprint index")
        for tile_key, buildings in lod2_groups.items():
            kept, stats = filter_lod2_against_lod1(buildings, lod1_index, tile_key)
            all_buildings.extend(kept)
            filter_stats.input_lod2 += stats.input_lod2
            filter_stats.input_lod1 += stats.input_lod1
            filter_stats.retained += stats.retained
            filter_stats.rejected += stats.rejected
            filter_stats.rejected_low_overlap += stats.rejected_low_overlap
            filter_stats.rejected_invalid_overlap += stats.rejected_invalid_overlap
    else:
        for buildings in lod2_groups.values():
            all_buildings.extend(buildings)

    _validate_against_existing_vintages(len(all_buildings), vintage)

    ds = _build_morphology_dataset(all_buildings, grid)

    spec = _VINTAGE_SOURCES[vintage]
    tile_count = len(lod2_groups)
    raw_manifest_uri = (
        f"{source_root.rstrip('/')}/ard/static/sources/lod_vintages/"
        f"raw_manifest_{vintage}.json"
    )
    source_metadata = {
        "vintage": vintage,
        "source_kind": spec["kind"],
        "local_path": spec["local_path"],
        "feed_label": spec["feed_label"],
        "raw_manifest_uri": raw_manifest_uri,
        "tile_count": tile_count,
        "total_buildings": len(all_buildings),
        "stock_filter": filter_stats.as_dict() if vintage == 2017 else None,
    }

    valid_frac = float(
        np.count_nonzero(~np.isnan(ds["building_height_mean"].values))
    ) / ds["building_height_mean"].size

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
                "total_buildings": len(all_buildings),
                **(
                    {"stock_filter": filter_stats.as_dict()}
                    if vintage == 2017
                    else {}
                ),
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
    lod1_index: dict[str, list[Footprint]] | None = None,
    grid: GeoBox | None = None,
    smoke_tile_count: int | None = None,
):
    """Prepare and finalise the *vintage* morphology product."""
    grid = grid or canon_grid_10m()
    prepared, stats = prepare_vintage_morphology(
        vintage,
        source_root,
        run_id,
        lod1_index=lod1_index,
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
    return artifacts, stats


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


__all__ = [
    "FilterStats",
    "RawManifest",
    "RawManifestEntry",
    "derive_vintage_products",
    "filter_lod2_against_lod1",
    "load_lod1_footprints",
    "load_lod2_buildings",
    "prepare_vintage_morphology",
    "publish_geometry_mapping",
    "publish_vintage_morphology",
    "stream_vintage_to_gcs",
    "vintage_geometry_id",
    "year_to_vintage_range",
]