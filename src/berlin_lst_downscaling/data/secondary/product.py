"""Secondary product finalisation — STAC, provenance, completion marker.

Each secondary product (static vintage or future dynamic scene) produces
four artifacts under its product directory:

- ``{source}_{key}.tif``            — final COG
- ``{source}_{key}.stac.json``      — STAC Item
- ``provenance.json``               — source/archive provenance
- ``complete.json``                 — publication marker (written last)

``finalize_secondary_product`` is the single entry point that source
runners call after preparing the canonical dataset.  It writes the COG
atomically, validates it, and then emits the STAC item, provenance, and
completion marker.  The completion marker is written last because GCS
cannot atomically publish multiple blobs — its absence means the
product is not considered final by ``reconcile()``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic
from berlin_lst_downscaling.data.io import atomic_write
from berlin_lst_downscaling.data.secondary.paths import (
    product_cog_path,
    product_completion_path,
    product_provenance_path,
    product_stac_path,
)
from berlin_lst_downscaling.data.secondary.validate import validate_secondary_cog

# STAC extension schema URLs (Projection v2.0.0, Raster v1.1.0).
# decision: pin to released schema URLs rather than bare extension names
# so consumers can validate with standard STAC tooling.
_PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
_RASTER_EXT = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
_STAC_VERSION = "1.0.0"


@dataclass
class PreparedSecondaryProduct:
    """Payload produced by a source adapter and handed to finalisation.

    Source adapters are responsible for the canonical-grid ``dataset``,
    source metadata, and QA statistics.  This dataclass is the contract
    between source adapters and the finalisation path.
    """

    source: str
    item_key: str  # vintage (e.g. "2016") or scene_id
    category: str  # e.g. "morphology" — used in the output directory
    dataset: xr.Dataset  # canonical grid with crs + transform written
    contract: Contract
    nominal_interval: tuple[str, str]  # (start_datetime, end_datetime) RFC 3339
    source_metadata: dict  # archive URL, checksum, license, native meta, …
    qa_stats: dict  # valid_frac, min, max, shape, …
    config_hash: str
    extra_provenance: dict = field(default_factory=dict)
    # dynamic (scene-timed) product fields — None for static vintage products
    acquisition_datetime: datetime | None = None
    stac_properties: dict | None = None


@dataclass
class ProductArtifacts:
    """The four URIs emitted by finalisation."""

    cog_uri: str
    stac_uri: str
    provenance_uri: str
    completion_uri: str


# ── STAC builder ──────────────────────────────────────────────────────


def build_secondary_stac_item(
    prepared: PreparedSecondaryProduct,
    grid: GeoBox,
    cog_href: str,
    provenance_href: str,
) -> dict:
    """Build a minimal STAC Item dict for a secondary product.

    The Item describes the canonical-grid COG and points to the
    provenance sidecar as a metadata asset.  ``datetime`` is null
    because the item is a nominal product (vintage/scene aggregate)
    rather than a point-acquisition — ``start_datetime`` /
    ``end_datetime`` carry the nominal interval instead.
    """
    band = list(prepared.dataset.data_vars)[0]
    height, width = prepared.dataset[band].shape[-2:]
    transform = prepared.dataset.rio.transform()
    bounds_native = array_bounds(height, width, transform)
    bbox_4326 = transform_bounds(str(grid.crs), "EPSG:4326", *bounds_native)

    start_dt, end_dt = prepared.nominal_interval

    # Build per-band raster:bands from contract specs
    raster_bands = []
    for spec in prepared.contract.output_bands:
        # Raster Extension 1.1: float nodata must be "nan" string, not JSON null
        if spec.nodata is not None and _is_nan(spec.nodata):
            nodata_stac = "nan"
        else:
            nodata_stac = spec.nodata
        band_entry: dict = {
            "data_type": spec.dtype,
            "nodata": nodata_stac,
            "spatial_resolution": abs(transform.a),
        }
        if spec.unit:
            band_entry["unit"] = spec.unit
        raster_bands.append(band_entry)

    first_spec = prepared.contract.output_bands[0]
    item: dict = {
        "stac_version": _STAC_VERSION,
        "stac_extensions": [_PROJ_EXT, _RASTER_EXT],
        "type": "Feature",
        "id": f"{prepared.source}-{prepared.item_key}",
        "bbox": list(bbox_4326),
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox_4326[0], bbox_4326[1]],
                    [bbox_4326[2], bbox_4326[1]],
                    [bbox_4326[2], bbox_4326[3]],
                    [bbox_4326[0], bbox_4326[3]],
                    [bbox_4326[0], bbox_4326[1]],
                ]
            ],
        },
        "properties": {
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "proj:code": str(grid.crs),
            "proj:shape": [height, width],
            "proj:transform": list(transform),
            "secondary:category": prepared.category,
            "secondary:config_hash": prepared.config_hash,
        },
        "assets": {
            "data": {
                "href": cog_href,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": first_spec.description or prepared.source,
                "roles": ["data"],
                "raster:bands": raster_bands,
            },
            "provenance": {
                "href": provenance_href,
                "type": "application/json",
                "title": "Source/archive provenance",
                "roles": ["metadata"],
            },
        },
        "links": [],
    }

    # Dynamic (scene-timed) product: set actual acquisition datetime
    if prepared.acquisition_datetime is not None:
        item["properties"]["datetime"] = prepared.acquisition_datetime.isoformat()
        # Remove nominal interval for point-acquisition products
        item["properties"].pop("start_datetime", None)
        item["properties"].pop("end_datetime", None)

    # Merge any extra STAC properties (solar geometry, vintage info, etc.)
    if prepared.stac_properties:
        item["properties"].update(prepared.stac_properties)

    return item


def build_provenance(
    prepared: PreparedSecondaryProduct,
    run_id: str,
    completed_at: str,
) -> dict:
    """Build the provenance payload for a secondary product."""
    prov = {
        "source": prepared.source,
        "item_key": prepared.item_key,
        "category": prepared.category,
        "config_hash": prepared.config_hash,
        "run_id": run_id,
        "completed_at": completed_at,
        "nominal_interval": {
            "start_datetime": prepared.nominal_interval[0],
            "end_datetime": prepared.nominal_interval[1],
        },
        "qa_stats": prepared.qa_stats,
        "source_metadata": prepared.source_metadata,
        **prepared.extra_provenance,
    }
    # Include acquisition datetime for scene-timed products
    if prepared.acquisition_datetime is not None:
        prov["acquisition_datetime"] = prepared.acquisition_datetime.isoformat()
    return prov


# ── finalisation ──────────────────────────────────────────────────────


def finalize_secondary_product(
    prepared: PreparedSecondaryProduct,
    grid: GeoBox,
    output_root: str,
    run_id: str,
    product_dir_override: str | None = None,
) -> ProductArtifacts:
    """Write the four final artifacts for a secondary product.

    Order of writes:

    1. COG (atomic) at the final product path.
    2. Validate the COG against the contract + canonical grid.
    3. Provenance JSON (atomic) at the final product path.
    4. STAC Item JSON (atomic) at the final product path.
    5. Completion marker JSON (atomic) — **written last**.

    ``product_dir_override`` lets Pipeline A write to its dedicated layout
    instead of the default ``ard/static/{category}/{source}/{item_key}``.

    Raises on COG validation failure. Callers should catch and mark the
    ledger row as ``failed``.
    """
    completed_at = datetime.now(UTC).isoformat()

    if product_dir_override is not None:
        base = product_dir_override.rstrip("/")
        cog_uri = f"{base}/{prepared.source}_{prepared.item_key}.tif"
        provenance_uri = f"{base}/provenance.json"
        stac_uri = f"{base}/{prepared.source}_{prepared.item_key}.stac.json"
        completion_uri = f"{base}/complete.json"
    else:
        cog_uri = product_cog_path(
            output_root,
            prepared.category,
            prepared.source,
            prepared.item_key,
        )
        provenance_uri = product_provenance_path(
            output_root,
            prepared.category,
            prepared.source,
            prepared.item_key,
        )
        stac_uri = product_stac_path(
            output_root,
            prepared.category,
            prepared.source,
            prepared.item_key,
        )
        completion_uri = product_completion_path(
            output_root,
            prepared.category,
            prepared.source,
            prepared.item_key,
        )

    # ── 1. write COG ─────────────────────────────────────────────────
    write_cog_atomic(prepared.dataset, cog_uri, prepared.contract, overwrite=True)

    # ── 2. validate ──────────────────────────────────────────────────
    vig = validate_secondary_cog(cog_uri, prepared.contract, grid)
    if not vig.ok:
        raise ValueError(
            f"COG validation failed for {prepared.source} "
            f"{prepared.item_key}: {'; '.join(vig.errors)}"
        )

    # ── 3. provenance ────────────────────────────────────────────────
    provenance = build_provenance(prepared, run_id, completed_at)
    atomic_write(
        provenance_uri,
        json.dumps(provenance, indent=2),
        overwrite=True,
    )

    # ── 4. STAC item ─────────────────────────────────────────────────
    stac_item = build_secondary_stac_item(
        prepared,
        grid,
        cog_uri,
        provenance_uri,
    )
    atomic_write(
        stac_uri,
        json.dumps(stac_item, indent=2),
        overwrite=True,
    )

    # ── 5. completion marker (last) ──────────────────────────────────
    atomic_write(
        completion_uri,
        json.dumps({"published_at": completed_at, "run_id": run_id}, indent=2),
        overwrite=True,
    )

    return ProductArtifacts(
        cog_uri=cog_uri,
        stac_uri=stac_uri,
        provenance_uri=provenance_uri,
        completion_uri=completion_uri,
    )


def vintage_interval(vintage: int) -> tuple[str, str]:
    """Return the nominal RFC 3339 interval for a vintage year."""
    return (f"{vintage}-01-01T00:00:00Z", f"{vintage}-12-31T23:59:59Z")


def _is_nan(value: float) -> bool:
    """Return True if *value* is NaN (safe for non-float inputs)."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


__all__ = [
    "PreparedSecondaryProduct",
    "ProductArtifacts",
    "build_secondary_stac_item",
    "build_provenance",
    "finalize_secondary_product",
    "vintage_interval",
]
