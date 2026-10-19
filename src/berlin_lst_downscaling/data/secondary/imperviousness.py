"""Imperviousness (Versiegelung) source adapter for the secondary pipeline.

Official Umweltatlas Berlin raster products for 2016 and 2021.
Each vintage is a single ZIP with a GeoTIFF (EPSG:25833, 2.5m, uint8 class codes).

Processing
----------
1. Stream the ZIP to raw storage via ``download_to_raw`` (no full-RAM load).
2. Open the inner GeoTIFF via ``rasterio.band(src, 1)`` over a
   ``zip://`` VFS path — no extraction step.
3. Convert uint8 class codes to float32 sealing percent [0, 100].
   Direct codes keep their percent. Code 103 is 0 on water, the street
   segment ``vg`` on street land, or a block donor mean. Code 101 uses
   the block donor mean. Code 110 is 100. Unknown codes and leftover
   103 are NaN and counted. See ``imperviousness_assign``.
4. Reproject from native 2.5m grid to the canonical 10m EPSG:25833 grid
   using ``Resampling.average`` — averaging is correct **only** after
   class-code-to-percent conversion.
5. Return a :class:`PreparedSecondaryProduct`; the pipeline finaliser
   writes the four final artifacts (COG + STAC + provenance + complete).
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.common.util import sha256_file
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.io import atomic_upload, exists
from berlin_lst_downscaling.data.secondary.download import download_to_raw
from berlin_lst_downscaling.data.secondary.imperviousness_assign import (
    MAPPING_RULES_ID,
    apply_mapping,
    built_frame,
    donor_means,
    fill_empty_blocks_from_ring,
    fill_shadows_outside_blocks,
    rasterize_blocks,
    rasterize_streets,
    sealing_coverage_stats,
    target_counts,
)
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

# ── official URLs (verified live 2026-07-14) ──────────────────────────

IMPERVIOUSNESS_URLS: dict[int, str] = {
    2016: ("https://gdi.berlin.de/data/ua_versiegelung_2016/atom/Versiegelung_Raster_2016.zip"),
    2021: ("https://gdi.berlin.de/data/ua_versiegelung_2021/atom/Versiegelung_Raster_2021.zip"),
}

# ── auxiliary sources for codes 101 and 103 ───────────────────────────

# Umweltatlas 01.02 (2021), street layer. ``vg`` counts track ballast as
# sealed. The format description dates the street survey at 2014. Both
# class vintages use this layer: the 2016 streets product published
# district means only.
STREET_WFS_URL = (
    "https://gdi.berlin.de/services/wfs/ua_versiegelung_2021"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=ua_versiegelung_2021:versieg2021_str"
    "&SRSNAME=EPSG:25833&OUTPUTFORMAT=geopackage"
)
STREET_LAYER = "ua_versiegelung_2021:versieg2021_str"
STREET_FIELD = "vg"
# Stadtstruktur v1, vintage 2024. ``typ`` 100 is the water mask.
BLOCKS_SOURCE_URI = (
    "gs://berlin-lst-training-data/evaluation/stadtstruktur/v1/source/2024/blocks.parquet"
)
FORMAT_DOC_URL = (
    "https://gdi.berlin.de/data/ua_versiegelung_2021/docs/"
    "Datenformatbeschreibung_versiegelung2021.pdf"
)


@dataclass(frozen=True)
class ImperviousnessAux:
    """Local copies of the street layer and the Stadtstruktur blocks."""

    streets_path: str
    blocks_path: str
    street_sha256: str
    blocks_sha256: str
    street_uri: str
    blocks_source_uri: str

def contract_for_imperviousness() -> Contract:
    """Return the output Contract for imperviousness COGs."""
    return Contract(
        source="imperviousness",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="imperviousness",
                dtype="float32",
                nodata=float("nan"),
                valid_range=(-0.01, 100.01),
                description="Sealing degree in percent (0–100)",
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )

def config_hash_for_vintage(
    vintage: int,
    *,
    street_sha256: str,
    blocks_sha256: str,
) -> str:
    """Return a stable config hash for one vintage.

    The payload is the mapping, the source identifiers, and the checksums
    of the street GeoPackage and the Stadtstruktur blocks. A changed rule
    or a changed source file changes the hash.
    """
    payload = {
        "rules": MAPPING_RULES_ID,
        "vintage": vintage,
        "archive_url": IMPERVIOUSNESS_URLS[vintage],
        "direct": {"0": 0, "5_to_95_step_10": "code", "100": 100, "102": 100, "110": 100},
        "code_110": "map_rendering_100_rule_set_would_be_0",
        "code_101": "block_donor_mean_or_nearest_block_within_100m",
        "code_103": "water_0_then_street_vg_then_block_donor_mean_else_nan",
        "unknown_codes": "nan_counted",
        "water": {
            "source": BLOCKS_SOURCE_URI,
            "vintage": 2024,
            "typ": 100,
            "sha256": blocks_sha256,
        },
        "streets": {
            "wfs": "https://gdi.berlin.de/services/wfs/ua_versiegelung_2021",
            "layer": STREET_LAYER,
            "field": STREET_FIELD,
            "trackbed": "sealed",
            "survey_status": "2014",
            "sha256": street_sha256,
            "used_for_vintages": [2016, 2021],
        },
        "block_fill": {
            "donors": [0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 102, 110],
            "reducer": "equal_area_mean_of_pixel_centers",
            "empty_block": "100m_exterior_ring_mean",
            "shadow_outside_block": "nearest_finite_block_within_100m",
        },
        "resample": "canonical_10m_epsg25833_average",
        "format_doc": FORMAT_DOC_URL,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()[:12]

# ── scene-year → vintage mapping ──────────────────────────────────────

def vintage_for_scene_year(year: int) -> int:
    """Return the imperviousness vintage that applies to a scene year.

    2017–2019 → 2016
    2020–2025 → 2021
    """
    # decision: piecewise constant; no years earlier than 2017 are
    # included in the training window (May–Sep 2017–2025).
    if year <= 2019:
        return 2016
    return 2021

# ── prepare ───────────────────────────────────────────────────────────

def prepare_imperviousness(
    vintage: int,
    output_root: str,
    run_id: str,
    *,
    grid=None,
    aux: ImperviousnessAux | None = None,
) -> PreparedSecondaryProduct:
    """Download, convert, and reproject an imperviousness vintage.

    Parameters
    ----------
    vintage :
        Either ``2016`` or ``2021``.
    output_root :
        Root URI for all outputs (local path or ``gs://bucket/...``).
    run_id :
        Unique run identifier (for provenance and staging).
    grid :
        Optional output GeoBox.  Defaults to the full canonical 10 m grid.

    Returns
    -------
    PreparedSecondaryProduct
        Canonical-grid dataset + source metadata + QA statistics.
        The pipeline finaliser writes the four final artifacts.
    """
    url = IMPERVIOUSNESS_URLS[vintage]
    raw_uri = _raw_zip_uri(output_root, vintage)
    cache_path = _raw_zip_cache_uri(output_root, vintage)
    aux = aux or load_imperviousness_auxiliaries(output_root)
    c_hash = config_hash_for_vintage(
        vintage,
        street_sha256=aux.street_sha256,
        blocks_sha256=aux.blocks_sha256,
    )

    # ── 1. materialise raw ZIP via shared downloader (no full-RAM load) ──
    receipt = download_to_raw(
        url=url,
        destination=raw_uri,
        local_cache_path=cache_path,
    )
    archive_path = Path(receipt.local_cache_path)  # type: ignore[arg-type]

    # ── 2. read TIFF in place over zip:// VFS path ───────────────────
    with _zip_tiff_open(archive_path) as src:
        src_uint8 = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

        observed = sorted(int(v) for v in np.unique(src_uint8))
        src_pct, mapping_stats = _map_codes(
            src_uint8,
            src_crs=src_crs,
            src_transform=src_transform,
            aux=aux,
        )

    # ── 3. reproject to canonical 10m grid (average resampling) ──────
    grid = grid or canon_grid_10m()
    dst_arr = np.empty((grid.shape.y, grid.shape.x), dtype=np.float32)

    reproject(
        source=src_pct,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # ── 4. build canonical xr.Dataset ────────────────────────────────
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {"imperviousness": (("y", "x"), dst_arr)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    valid = dst_arr[~np.isnan(dst_arr)]
    retrieved_at = datetime.now(UTC).isoformat()

    return PreparedSecondaryProduct(
        source="imperviousness",
        item_key=str(vintage),
        category="morphology",
        dataset=ds,
        contract=contract_for_imperviousness(),
        nominal_interval=vintage_interval(vintage),
        source_metadata={
            "archive_url": url,
            "archive_sha256": receipt.checksum,
            "raw_uri": raw_uri,
            "retrieved_at": retrieved_at,
            "license": "dl-de/zero-2.0",
            "native_codes_observed": observed,
            "mapping_rules": MAPPING_RULES_ID,
            "street_layer": STREET_LAYER,
            "street_field": STREET_FIELD,
            "street_sha256": aux.street_sha256,
            "street_uri": aux.street_uri,
            "blocks_source_uri": aux.blocks_source_uri,
            "blocks_sha256": aux.blocks_sha256,
            "water_typ": 100,
            "format_doc": FORMAT_DOC_URL,
            "code_110": "100; rule set would be 0, map rendering is 100",
        },
        qa_stats={
            "valid_frac": (round(float(len(valid)) / dst_arr.size, 4) if dst_arr.size > 0 else 0.0),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(dst_arr.shape),
            **_rounded_mapping_stats(mapping_stats),
        },
        config_hash=c_hash,
    )

# ── helpers ───────────────────────────────────────────────────────────

def _raw_zip_uri(output_root: str, vintage: int) -> str:
    """Return the raw ZIP URI for a vintage."""
    return (
        f"{output_root.rstrip('/')}/_raw/secondary/imperviousness/"
        f"{vintage}/Versiegelung_Raster_{vintage}.zip"
    )

def _raw_zip_cache_uri(output_root: str, vintage: int) -> str:
    """Return a writable local cache path even when output_root is GCS.

    For local ``output_root`` we reuse the raw URI directly.  For
    ``gs://`` we stage the cache under ``$TMPDIR`` so we never write
    large archives into the CWD.
    """
    import tempfile

    if output_root.startswith("gs://"):
        return f"{tempfile.gettempdir()}/berlin_lst/imperviousness_{vintage}.zip"
    return _raw_zip_uri(output_root, vintage)

@contextlib.contextmanager
def _zip_tiff_open(zip_path: Path):
    """Open the first GeoTIFF member inside *zip_path* via ``zip://`` VFS."""
    import zipfile as _zf

    with _zf.ZipFile(zip_path) as z:
        tif_names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(f"No GeoTIFF member found in {zip_path}. Members: {z.namelist()}")
        if len(tif_names) != 1:
            raise ValueError(
                f"Expected exactly one GeoTIFF member; got {len(tif_names)}: {tif_names}"
            )
        member = tif_names[0]
    with rasterio.open(f"zip://{zip_path}!/{member}") as src:
        yield src

def load_imperviousness_auxiliaries(output_root: str) -> ImperviousnessAux:
    """Download the street GeoPackage and the Stadtstruktur blocks.

    The street request is the WFS GetFeature for ``versieg2021_str``. The
    blocks are the published Stadtstruktur v1 vintage-2024 parquet. Both
    are copied under ``output_root`` so the run keeps the bytes it hashed.
    """
    cache_root = Path(tempfile.gettempdir()) / "berlin_lst"
    cache_root.mkdir(parents=True, exist_ok=True)
    street_cache = cache_root / "versieg2021_str.gpkg"
    street_uri = (
        f"{output_root.rstrip('/')}/_raw/secondary/imperviousness/aux/versieg2021_str.gpkg"
    )
    receipt = download_to_raw(
        url=STREET_WFS_URL,
        destination=street_uri,
        local_cache_path=str(street_cache),
    )
    if receipt.local_cache_path is None:
        raise RuntimeError("street layer download did not leave a local cache")

    blocks_cache = cache_root / "stadtstruktur_2024_blocks.parquet"
    _ensure_local_gcs(BLOCKS_SOURCE_URI, blocks_cache)
    blocks_copy = (
        f"{output_root.rstrip('/')}/_raw/secondary/imperviousness/aux/blocks_2024.parquet"
    )
    if not exists(blocks_copy):
        atomic_upload(blocks_cache, blocks_copy)

    return ImperviousnessAux(
        streets_path=receipt.local_cache_path,
        blocks_path=str(blocks_cache),
        street_sha256=sha256_file(receipt.local_cache_path),
        blocks_sha256=sha256_file(str(blocks_cache)),
        street_uri=street_uri,
        blocks_source_uri=BLOCKS_SOURCE_URI,
    )


def _ensure_local_gcs(uri: str, cache_path: Path) -> None:
    """Download one GCS object to ``cache_path`` unless the cache is non-empty."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    from google.cloud import storage  # type: ignore[import-untyped]

    bucket_name, key = uri.removeprefix("gs://").split("/", 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(key).download_to_filename(str(cache_path))


def _is_utm33n(crs) -> bool:
    """True for EPSG:25833 and for the 2016 WKT that omits the authority code.

    The 2016 GeoTIFF names ETRS89 / UTM zone 33N with datum IRENET95 and no
    EPSG code. ``to_epsg()`` is then null. The pixel coordinates are still
    UTM 33N metres, which is the frame the street and block polygons use.
    """
    if crs is None:
        return False
    if crs.to_epsg() == 25833:
        return True
    wkt = crs.to_wkt().lower()
    return "utm zone 33n" in wkt and "transverse_mercator" in wkt and "metre" in wkt


def _map_codes(
    codes: np.ndarray,
    *,
    src_crs,
    src_transform,
    aux: ImperviousnessAux,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rasterize water, streets, and blocks, then apply the sealing rules."""
    if not _is_utm33n(src_crs):
        raise ValueError(f"class raster CRS must be UTM zone 33N metres, got {src_crs}")
    streets = gpd.read_file(aux.streets_path)
    blocks = gpd.read_parquet(aux.blocks_path)
    if streets.crs is None or streets.crs.to_epsg() != 25833:
        raise ValueError(f"street layer CRS must be EPSG:25833, got {streets.crs}")
    if blocks.crs is None or blocks.crs.to_epsg() != 25833:
        blocks = blocks.to_crs("EPSG:25833")
    if STREET_FIELD not in streets.columns:
        raise ValueError(f"street layer is missing {STREET_FIELD}")

    shape = (int(codes.shape[0]), int(codes.shape[1]))
    street_sum, street_count = rasterize_streets(streets, shape, src_transform)
    water, block_id, n_blocks = rasterize_blocks(blocks, shape, src_transform)
    built = built_frame(blocks)
    means = donor_means(codes, block_id, n_blocks)
    targets = target_counts(codes, block_id, water, street_count, n_blocks)
    ring_ids = fill_empty_blocks_from_ring(
        codes, built, shape, src_transform, means, targets
    )
    percent, stats = apply_mapping(
        codes,
        street_vg_sum=street_sum,
        street_count=street_count,
        water=water,
        block_id=block_id,
        block_fill=means,
        ring_blocks=ring_ids,
    )
    nearest = fill_shadows_outside_blocks(codes, percent, block_id, means)
    stats["nearest_101_px"] = float(nearest)
    stats["ring_blocks"] = float(ring_ids.size)
    stats.update(sealing_coverage_stats(codes, percent, water, street_count))
    return percent, stats


def _rounded_mapping_stats(stats: dict[str, float]) -> dict[str, float | None]:
    """Round coverage sums for provenance. Pixel counts stay integral."""
    out: dict[str, float | None] = {}
    for key, value in stats.items():
        if key.endswith("_sum"):
            out[key] = round(value, 3)
        else:
            out[key] = float(int(value))
    water_n = stats.get("water_finite_px", 0.0)
    street_n = stats.get("street_finite_px", 0.0)
    out["water_sealing_mean_pct"] = (
        round(stats["water_sealing_sum"] / water_n, 4) if water_n else None
    )
    out["street_sealing_mean_pct"] = (
        round(stats["street_sealing_sum"] / street_n, 4) if street_n else None
    )
    return out


__all__ = [
    "BLOCKS_SOURCE_URI",
    "IMPERVIOUSNESS_URLS",
    "ImperviousnessAux",
    "STREET_LAYER",
    "config_hash_for_vintage",
    "contract_for_imperviousness",
    "load_imperviousness_auxiliaries",
    "prepare_imperviousness",
    "vintage_for_scene_year",
]