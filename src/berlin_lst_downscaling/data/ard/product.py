"""ARD product finalisation — STAC, provenance, completion marker.

Every published ARD scene produces four co-located files:

- ``{scene_id}.tif``         — Cloud-Optimised GeoTIFF (data bands)
- ``{scene_id}.flag.tif``    — uint8 quality-flag COG (optional, separate)
- ``{scene_id}.stac.json``   — STAC Item (core 1.0.0 + Projection + Raster)
- ``provenance.json``        — source/transform lineage
- ``complete.json``          — written last, publication gate

``finalize_ard_product`` is the single entry point the pipeline calls.
It writes sidecars only — COGs and flags are assumed to have been
written and validated beforehand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import xarray as xr
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.io import atomic_write

# STAC extension schema URLs (Projection v2.0.0, Raster v1.1.0).
# decision: pin to released schema URLs rather than bare extension names
# so consumers can validate with standard STAC tooling.
_PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
_RASTER_EXT = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
_STAC_VERSION = "1.0.0"


def _is_nan(value: float) -> bool:
    """Return True if *value* is NaN (safe for non-float inputs)."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def build_ard_stac_item(
    scene_id: str,
    source: str,
    year: int,
    masked: xr.Dataset,
    contract: Contract,
    cog_href: str,
    target_resolution: int,
    flag_href: str | None = None,
    provenance_href: str | None = None,
) -> dict[str, Any]:
    """Build a standards-compliant STAC item for one ARD scene.

    Complies with STAC 1.0.0, Projection 2.0.0, and Raster 1.1.0.
    Float NaN nodata is serialized as ``"nan"`` (Raster 1.1 § nodata).
    The flag asset omits ``nodata`` rather than writing JSON ``null``.
    """
    crs = masked.rio.crs
    geo_transform = masked.rio.transform()

    first_band = list(masked.data_vars)[0]
    height, width = masked[first_band].shape[-2:]

    bounds = array_bounds(height, width, geo_transform)
    bbox_4326 = transform_bounds(crs, "EPSG:4326", *bounds)

    assets: dict[str, Any] = {}
    for spec in contract.output_bands:
        # Raster Extension 1.1: float nodata must be "nan" string, not JSON null
        if spec.nodata is not None and _is_nan(spec.nodata):
            nodata_stac = "nan"
        else:
            nodata_stac = spec.nodata
        assets[spec.name] = {
            "href": cog_href,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": spec.description,
            "roles": ["data"],
            "raster:bands": [
                {
                    "data_type": spec.dtype,
                    "nodata": nodata_stac,
                    "spatial_resolution": target_resolution,
                }
            ],
        }

    # Flag band as separate asset — no nodata field (uint8 0=clear is valid)
    if flag_href is not None and contract.flag_mode == "separate":
        assets["flag"] = {
            "href": flag_href,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "title": "Quality flag (bitmask: fill, cloudy, shadow, cirrus, saturated, snow/ice)",
            "roles": ["quality"],
            "raster:bands": [
                {
                    "data_type": "uint8",
                    "spatial_resolution": target_resolution,
                }
            ],
        }

    # Provenance metadata asset
    if provenance_href is not None:
        assets["provenance"] = {
            "href": provenance_href,
            "type": "application/json",
            "title": "Source and processing provenance",
            "roles": ["metadata"],
        }

    # Acquisition datetime from dataset
    acq_dt = _acquisition_datetime(masked, year)

    item: dict[str, Any] = {
        "stac_version": _STAC_VERSION,
        "stac_extensions": [_PROJ_EXT, _RASTER_EXT],
        "type": "Feature",
        "id": f"{source}-{scene_id}",
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
            "datetime": (
                acq_dt.isoformat()
                if acq_dt
                else f"{year}-01-01T00:00:00Z"
            ),
            "proj:code": str(crs),
            "proj:shape": [height, width],
            "proj:transform": list(geo_transform),
            "ard:schema_version": contract.schema_version_str(),
            "ard:source": source,
            "ard:scene_id": scene_id,
        },
        "assets": assets,
        "links": [],
    }

    return item


def build_ard_provenance(
    scene_id: str,
    source: str,
    year: int,
    contract: Contract,
    run_id: str,
    *,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance payload for an ARD product."""
    provenance = {
        "source": source,
        "scene_id": scene_id,
        "year": year,
        "schema_version": contract.schema_version,
        "run_id": run_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "output_bands": [s.name for s in contract.output_bands],
    }
    if source_metadata:
        provenance["source_metadata"] = source_metadata
    return provenance


@dataclass
class ARDArtifacts:
    """Deterministic artifact URIs produced by an ARD scene."""

    scene_id: str
    source: str
    year: int
    root: str
    stac_uri: str
    provenance_uri: str
    completion_uri: str


def finalize_ard_product(
    scene_id: str,
    source: str,
    year: int,
    root: str,
    contract: Contract,
    masked: xr.Dataset,
    run_id: str,
    *,
    flag_uri: str | None = None,
    target_resolution: int,
    source_metadata: dict[str, Any] | None = None,
) -> ARDArtifacts:
    """Write ARD sidecars and return deterministic artifact URIs.

    This function does NOT write COGs or flags — it writes provenance,
    STAC, and completion marker against already-validated raster assets.

    Publication order:
    1. Validate COG/flag (caller's responsibility — done before calling).
    2. Write provenance.json (atomic).
    3. Write and validate STAC item (atomic).
    4. Write complete.json last (atomic) — publication gate.
    """
    from berlin_lst_downscaling.data.ard.paths import (
        completion_path as _completion_path,
    )
    from berlin_lst_downscaling.data.ard.paths import (
        provenance_path as _provenance_path,
    )
    from berlin_lst_downscaling.data.ard.paths import (
        stac_path as _stac_path,
    )

    stac_dst = _stac_path(root, source, year, scene_id)
    prov_dst = _provenance_path(root, source, year, scene_id)
    comp_dst = _completion_path(root, source, year, scene_id)

    # Write provenance
    provenance = build_ard_provenance(
        scene_id, source, year, contract, run_id,
        source_metadata=source_metadata,
    )
    atomic_write(prov_dst, json.dumps(provenance, indent=2), overwrite=True)

    # Build STAC item (using relative provenance href within the scene dir)
    prov_href = "provenance.json"
    flag_href = flag_uri.split("/")[-1] if flag_uri else None
    stac_item = build_ard_stac_item(
        scene_id, source, year, masked, contract,
        cog_href=f"{scene_id}.tif",
        target_resolution=target_resolution,
        flag_href=flag_href,
        provenance_href=prov_href,
    )
    json_bytes = json.dumps(stac_item, indent=2).encode("utf-8")
    atomic_write(stac_dst, json_bytes, overwrite=True)

    # Write completion marker last
    completed_at = datetime.now(UTC).isoformat()
    atomic_write(
        comp_dst,
        json.dumps({"published_at": completed_at, "run_id": run_id}, indent=2),
        overwrite=True,
    )

    return ARDArtifacts(
        scene_id=scene_id,
        source=source,
        year=year,
        root=root,
        stac_uri=stac_dst,
        provenance_uri=prov_dst,
        completion_uri=comp_dst,
    )


# ── helpers ──────────────────────────────────────────────────────────


def _acquisition_datetime(masked: xr.Dataset, year: int) -> datetime | None:
    """Extract acquisition datetime from the dataset time coordinate."""
    try:
        dt64 = masked.time.values[0]
        ts = dt64.astype("datetime64[us]").tolist()
        return datetime.fromtimestamp(ts.timestamp(), tz=UTC)
    except (IndexError, AttributeError, ValueError):
        return None


__all__ = [
    "ARDArtifacts",
    "build_ard_provenance",
    "build_ard_stac_item",
    "finalize_ard_product",
]
