"""Structural inspection and COG validation for WB2c-1 profiling.

Validates every expected COG against its canonical grid, runs strict
rio-cogeo validation, and records structural findings.
"""

from __future__ import annotations

import rasterio
from odc.geo.geobox import GeoBox
from rio_cogeo.cogeo import cog_validate

from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.io.storage import exists
from berlin_lst_downscaling.data.profiling.models import ProfileAsset, ProfileRow


def gdal_uri(uri: str) -> str:
    """Convert gs:// URI to /vsigs/ for GDAL/rasterio/rio-cogeo reads.

    Retains original URIs in all records; this adapter is used only
    for read operations that require GDAL VSI GCS support.
    """
    if uri.startswith("gs://"):
        path = uri[5:]  # remove gs://
        bucket, _, key = path.partition("/")
        return f"/vsigs/{bucket}/{key}"
    return uri


def inspect_asset(asset: ProfileAsset) -> ProfileRow:
    """Inspect a single asset for structural validity.

    Returns a ProfileRow with structural checks completed.
    """
    row = ProfileRow(
        item_id=asset.item_id,
        source=asset.source,
        cog_uri=asset.cog_uri,
        partition=asset.partition,
        year=asset.year,
        season=asset.season,
        resolution_m=asset.resolution_m,
    )

    # Check COG existence
    if not exists(asset.cog_uri):
        row.cog_exists = False
        row.has_hard_failure = True
        row.failure_reasons.append(f"COG not found: {asset.cog_uri}")
        return row

    row.cog_exists = True

    # Check sidecar existence
    if asset.stac_uri:
        row.stac_exists = exists(asset.stac_uri)
    if asset.provenance_uri:
        row.provenance_exists = exists(asset.provenance_uri)
    if asset.completion_uri:
        row.completion_exists = exists(asset.completion_uri)

    # Validate COG structure
    grid = canon_grid_for_resolution(asset.resolution_m or 10)
    errors = validate_cog_structure(asset.cog_uri, asset, grid)

    if errors:
        row.cog_valid = False
        row.cog_errors = errors
        row.has_hard_failure = True
        row.failure_reasons.extend(errors)
    else:
        row.cog_valid = True

    # Validate COG internal structure with rio-cogeo (in-process)
    cogeo_errors = validate_cogeo_internal(asset.cog_uri)
    if cogeo_errors:
        row.cog_errors.extend(cogeo_errors)
        row.has_hard_failure = True
        row.failure_reasons.extend(cogeo_errors)

    return row


def validate_cog_structure(
    uri: str,
    asset: ProfileAsset,
    expected_grid: GeoBox,
) -> list[str]:
    """Validate COG structural properties against expected grid."""
    errors: list[str] = []

    try:
        with rasterio.open(gdal_uri(uri)) as src:
            # CRS check
            crs_str = str(src.crs).upper() if src.crs else "None"
            if crs_str != asset.expected_crs.upper():
                errors.append(f"CRS mismatch: got {crs_str!r}, expected {asset.expected_crs!r}")

            # Resolution check
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            expected_res = expected_grid.transform.a
            if abs(res_x - expected_res) > 0.1 or abs(res_y - expected_res) > 0.1:
                errors.append(
                    f"Resolution mismatch: got ({res_x:.2f}, {res_y:.2f}), "
                    f"expected ({expected_res:.2f}, {expected_res:.2f})"
                )

            # Transform origin check
            x_off = src.transform.c
            y_off = src.transform.f
            expected_x_off = expected_grid.transform.xoff
            expected_y_off = expected_grid.transform.yoff
            if abs(x_off - expected_x_off) > 0.01 or abs(y_off - expected_y_off) > 0.01:
                errors.append(
                    f"Origin mismatch: got ({x_off:.1f}, {y_off:.1f}), "
                    f"expected ({expected_x_off:.1f}, {expected_y_off:.1f})"
                )

            # Shape check
            width, height = src.width, src.height
            expected_width, expected_height = expected_grid.shape.x, expected_grid.shape.y
            if width != expected_width or height != expected_height:
                errors.append(
                    f"Shape mismatch: got ({width}, {height}), "
                    f"expected ({expected_width}, {expected_height})"
                )

            # Band count check
            if src.count != asset.expected_bands:
                errors.append(
                    f"Band count mismatch: got {src.count}, expected {asset.expected_bands}"
                )

    except Exception as exc:
        errors.append(f"Cannot open COG: {exc}")

    return errors


def validate_cogeo_internal(uri: str) -> list[str]:
    """Validate COG internal structure with rio-cogeo (in-process)."""
    try:
        valid, errors, warnings = cog_validate(gdal_uri(uri), strict=True)
        result_errors: list[str] = []
        for err in errors:
            result_errors.append(f"COG validation: {err}")
        for warn in warnings:
            result_errors.append(f"COG validation warning: {warn}")
        return result_errors
    except FileNotFoundError:
        return ["rio-cogeo not found in PATH"]
    except Exception as exc:
        return [f"COG validation failed: {exc}"]


__all__ = [
    "gdal_uri",
    "inspect_asset",
    "validate_cog_structure",
    "validate_cogeo_internal",
]
