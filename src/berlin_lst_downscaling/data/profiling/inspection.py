"""Structural inspection and COG validation for WB2c-1 profiling.

Validates every expected COG against its canonical grid, runs strict
rio-cogeo validation, and records structural findings.
"""

from __future__ import annotations

import math

import rasterio
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.ard.contract import BandSpec
from berlin_lst_downscaling.data.io.storage import exists
from berlin_lst_downscaling.data.profiling.models import (
    ContractCheckResult,
    ProfileAsset,
    ProfileRow,
)


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

    # Check sidecar existence — missing expected sidecars are hard failures
    for label, uri in [
        ("STAC", asset.stac_uri),
        ("provenance", asset.provenance_uri),
        ("completion", asset.completion_uri),
    ]:
        if uri:
            present = exists(uri)
            if not present:
                row.has_hard_failure = True
                row.failure_reasons.append(f"{label} sidecar not found: {uri}")
            if label == "STAC":
                row.stac_exists = present
            elif label == "provenance":
                row.provenance_exists = present
            else:
                row.completion_exists = present

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
    cogeo_result = validate_cogeo_internal(asset.cog_uri)
    if not cogeo_result.valid:
        all_issues = list(cogeo_result.errors) + list(cogeo_result.warnings)
        row.cog_errors.extend(all_issues)
        row.has_hard_failure = True
        row.failure_reasons.extend(all_issues)

    # Validate band contracts (dtype, nodata, channel order)
    if asset.expected_band_contracts and row.cog_valid:
        contract_result = validate_band_contracts(
            asset.cog_uri, asset.expected_band_contracts
        )
        row.contract_check = contract_result
        if not contract_result.ok:
            row.has_hard_failure = True
            row.failure_reasons.extend(
                contract_result.dtype_mismatches
                + contract_result.nodata_mismatches
                + contract_result.channel_order_mismatches
            )

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


def _isnan(value: float | None) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _fmt_nodata(value: float) -> str:
    if _isnan(value):
        return "NaN"
    return str(value)


def validate_band_contracts(
    uri: str,
    expected_contracts: tuple[BandSpec, ...],
) -> ContractCheckResult:
    """Validate per-band dtype, nodata, and channel order against contracts.

    Channel order is validated against ``BandSpec.name`` (the persisted
    channel description). Prose descriptions and units are not persisted
    by the writer; their absence is a documented limitation, not a
    validation failure.
    """
    result = ContractCheckResult()

    try:
        with rasterio.open(gdal_uri(uri)) as src:
            n_bands = min(src.count, len(expected_contracts))

            for i in range(n_bands):
                spec = expected_contracts[i]
                band_idx = i + 1

                # dtype check
                if src.dtypes[i] != spec.dtype:
                    result.dtype_mismatches.append(
                        f"Band {band_idx} ({spec.name}): dtype "
                        f"got {src.dtypes[i]}, expected {spec.dtype}"
                    )

                # nodata check (per-band nodatavals)
                actual_nodata = src.nodatavals[i]
                if spec.nodata is not None:
                    if actual_nodata is None:
                        result.nodata_mismatches.append(
                            f"Band {band_idx} ({spec.name}): nodata "
                            f"got None, expected {_fmt_nodata(spec.nodata)}"
                        )
                    elif _isnan(spec.nodata):
                        if not (_isnan(actual_nodata) if actual_nodata is not None else False):
                            result.nodata_mismatches.append(
                                f"Band {band_idx} ({spec.name}): nodata "
                                f"got {actual_nodata}, expected NaN"
                            )
                    elif actual_nodata != spec.nodata:
                        result.nodata_mismatches.append(
                            f"Band {band_idx} ({spec.name}): nodata "
                            f"got {actual_nodata}, expected {_fmt_nodata(spec.nodata)}"
                        )

            # channel order: persisted descriptions carry the channel name
            if src.descriptions and len(expected_contracts) > 1:
                actual_names = [
                    (d.split(";")[0].strip() if d else "")
                    for d in src.descriptions[:n_bands]
                ]
                expected_names = [s.name for s in expected_contracts[:n_bands]]
                if actual_names != expected_names:
                    result.channel_order_mismatches.append(
                        f"Channel order: got {actual_names}, expected {expected_names}"
                    )

            # documented writer limitations, not failures
            for i, spec in enumerate(expected_contracts[:n_bands], 1):
                if spec.description:
                    result.prose_description_absent.append(
                        f"Band {i} ({spec.name}): prose description not persisted by writer"
                    )
                if spec.unit:
                    result.unit_absent.append(
                        f"Band {i} ({spec.name}): "
                        f"unit '{spec.unit}' not persisted by writer"
                    )

    except Exception as exc:
        result.dtype_mismatches.append(f"Could not read COG metadata: {exc}")

    return result


def validate_cogeo_internal(uri: str):
    """Validate COG internal structure with shared strict validation.

    Returns a StrictCogResult. Warnings are failures.
    """
    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    return validate_strict_cog(uri)


__all__ = [
    "gdal_uri",
    "inspect_asset",
    "validate_cog_structure",
    "validate_band_contracts",
    "validate_cogeo_internal",
]
