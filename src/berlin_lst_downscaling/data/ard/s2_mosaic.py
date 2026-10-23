"""Deterministic, whole-donor mosaics of individually harmonized S2 items."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from rasterio.warp import Resampling, reproject

from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.ard.contract import Contract
from berlin_lst_downscaling.data.io import read_bytes

DONOR_RULE = "first_scl_nonfill_by_sorted_item_id_all_bands_and_flag"
_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "flag")
_INPUT_HASH_PREFIX = "s2-mosaic-v10:sha256:"


def recorded_s2_members(scene_id: str, meta: dict) -> list[dict[str, str]]:
    """Validate recorded v4 membership without resolving or loading any item."""
    recorded = meta.get("s2_items")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(f"scene {scene_id}: schema-10 mosaic requires v4 recorded s2_items")
    group = meta.get("s2_mosaic_group")
    if not isinstance(group, dict) or any(
        not isinstance(group.get(key), str) or not group[key].strip()
        for key in ("solar_day", "datatake_id")
    ):
        raise ValueError(f"scene {scene_id}: recorded s2_items requires s2_mosaic_group")
    for member in recorded:
        if not isinstance(member, dict) or any(
            not isinstance(member.get(key), str) or not member[key].strip()
            for key in ("item_id", "item_href")
        ):
            raise ValueError(f"scene {scene_id}: invalid recorded s2_items member")
    ids = [member["item_id"] for member in recorded]
    if len(set(ids)) != len(ids) or scene_id not in ids:
        raise ValueError(f"scene {scene_id}: duplicate members or scene missing from s2_items")
    return [
        {"item_id": member["item_id"], "item_href": member["item_href"]}
        for member in sorted(recorded, key=lambda member: member["item_id"])
    ]


def aoi_mask_uri(cfg: DictConfig) -> str:
    return f"{str(cfg.aoi.mask_base).rstrip('/')}/aoi_{int(cfg.target_resolution_high)}m.tif"


def s2_aoi_identity(cfg: DictConfig) -> dict[str, str]:
    """Bind the AOI to both its location and bytes, including same-URI replacement."""
    uri = aoi_mask_uri(cfg)
    return {"uri": uri, "sha256": hashlib.sha256(read_bytes(uri)).hexdigest()}


def s2_input_contract(
    scene_id: str, year: int, meta: dict, cfg: DictConfig, aoi_identity: dict[str, str]
) -> dict[str, Any]:
    """Canonical per-scene v10 inputs and policy, independent of run/output paths.

    Policy revision must change when the implementation of these processing
    rules changes. Exact STAC references identify immutable source items.
    """
    members = recorded_s2_members(scene_id, meta)
    resolution = int(cfg.target_resolution_high)
    if resolution != 10:
        raise ValueError("Sentinel-2 schema-10 mosaics require the canonical 10m grid")
    threshold = validate_missing_fraction(
        cfg.sentinel2.mosaic.get("max_missing_aoi_fraction", 0.005)
    )
    sentinel2 = OmegaConf.to_container(cfg.sentinel2, resolve=True)
    if not isinstance(sentinel2, dict) or not isinstance(sentinel2.get("mosaic"), dict):
        raise ValueError("Sentinel-2 schema-10 mosaic policy is missing")
    sentinel2["schema_version"] = 10
    sentinel2["mosaic"]["max_missing_aoi_fraction"] = threshold
    grids = {}
    for res in (10, 20):
        grid = canon_grid_for_resolution(res)
        grids[str(res)] = {
            "crs": str(grid.crs),
            "shape": list(grid.shape),
            "transform": list(grid.transform),
        }
    acquired = meta.get("acquisition_datetime") or cfg.get("scene_date")
    if acquired is not None:
        dt = datetime.fromisoformat(str(acquired).replace("Z", "+00:00"))
        acquired = (dt.astimezone(UTC) if dt.tzinfo else dt).isoformat()
    return {
        "input_contract_version": 1,
        "source": "sentinel-2-l2a",
        "schema_version": 10,
        "scene_id": scene_id,
        "year": int(year),
        "acquisition_datetime": acquired,
        "s2_items": members,
        "s2_mosaic_group": dict(meta["s2_mosaic_group"]),
        "policy": {
            "revision": 1,
            "sentinel2": sentinel2,
            "donor_rule": DONOR_RULE,
            "footprint_rule": "finite_scl_nonzero_before_cloud_masking",
            "reflectance_rule": "per_item_boa_offset_then_clip_0_1",
            "swir_rule": "native_20m_nonzero_flag_mask_then_bilinear_to_10m",
            "cloud_base_height_m": float(cfg.get("cloud_base_height_m", 1000)),
            "cloud_dilation_px": int(cfg.get("cloud_dilation_px", 2)),
            "bbox": list(cfg.bbox),
            "target_resolution_high": resolution,
            "target_crs": str(cfg.get("target_crs", "EPSG:25833")),
            "canonical_grids": grids,
            "scene_date_fallback": cfg.get("scene_date"),
            "max_missing_aoi_fraction": threshold,
            "aoi": dict(aoi_identity),
        },
    }


def s2_input_contract_hash(input_contract: dict) -> str:
    encoded = json.dumps(input_contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _INPUT_HASH_PREFIX + hashlib.sha256(encoded.encode()).hexdigest()


def validate_missing_fraction(value: Any) -> float:
    """Require a finite fraction in [0, 1]."""
    if isinstance(value, bool):
        raise ValueError("max_missing_aoi_fraction must be a finite fraction in [0, 1]")
    try:
        threshold = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("max_missing_aoi_fraction must be a finite fraction in [0, 1]") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("max_missing_aoi_fraction must be a finite fraction in [0, 1]")
    return threshold


def load_aoi_on_grid(aoi_uri: str, target: xr.DataArray) -> np.ndarray:
    """Read the pre-baked AOI and align it to the mosaic grid using nearest."""
    destination = np.zeros((target.sizes["y"], target.sizes["x"]), dtype=np.uint8)
    if target.rio.crs is None:
        raise ValueError("Sentinel-2 mosaic grid has no CRS")
    with rasterio.open(aoi_uri) as src:
        reproject(
            source=src.read(1, masked=True).filled(0),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            destination=destination,
            dst_transform=target.rio.transform(),
            dst_crs=target.rio.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    inside = destination == 1
    if not inside.any():
        raise ValueError(f"Sentinel-2 mosaic AOI is empty on the target grid: {aoi_uri}")
    return inside


def coverage_metrics(
    footprint: np.ndarray, aoi: np.ndarray, max_missing_aoi_fraction: float = 0.005
) -> dict[str, Any]:
    """Measure pre-cloud source coverage; clouds are footprint-valid."""
    threshold = validate_missing_fraction(max_missing_aoi_fraction)
    inside = np.asarray(aoi)
    if inside.dtype != np.bool_ or inside.shape != footprint.shape:
        raise ValueError("Sentinel-2 mosaic AOI must be a boolean mask matching the footprint")
    total = int(np.count_nonzero(inside))
    if total == 0:
        raise ValueError("Sentinel-2 mosaic AOI is empty")
    missing = int(np.count_nonzero(inside & ~footprint))
    fraction = missing / total
    return {
        "aoi_total_px": total,
        "aoi_covered_px": total - missing,
        "aoi_missing_px": missing,
        "missing_aoi_fraction": fraction,
        "max_missing_aoi_fraction": threshold,
        "coverage_passed": fraction <= threshold,
        "footprint_rule": "finite_scl_nonzero_before_cloud_masking",
    }


def mosaic_s2_items(
    members: Sequence[tuple[str, xr.Dataset, np.ndarray]],
) -> tuple[xr.Dataset, np.ndarray]:
    """Select the first footprint-valid donor for all six bands and its flag.

    Members are (item ID, output of existing mask_s2, pre-cloud SCL != 0).
    Pixel selection never examines band finiteness or clear/cloud flags.
    Spatial grids must match exactly; singleton time coordinates may differ.
    """
    if not members:
        raise ValueError("Sentinel-2 mosaic requires recorded members")
    ordered = sorted(members, key=lambda member: member[0])
    ids = [member[0] for member in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Sentinel-2 mosaic has duplicate item IDs")
    template = ordered[0][1]
    shape = (template.sizes["y"], template.sizes["x"])
    union = np.zeros(shape, dtype=bool)
    arrays = {
        band: np.full(
            template[band].shape,
            Contract.FLAG_FILL if band == "flag" else np.nan,
            dtype=np.uint8 if band == "flag" else np.float32,
        )
        for band in _BANDS
    }
    provenance = []
    for item_id, masked, footprint in ordered:
        if (
            masked.rio.crs != template.rio.crs
            or masked.rio.transform() != template.rio.transform()
            or any(not masked[axis].equals(template[axis]) for axis in ("x", "y"))
        ):
            raise ValueError(f"Sentinel-2 mosaic member {item_id} has a mismatched grid")
        if footprint.shape != shape or footprint.dtype != np.bool_:
            raise ValueError(f"Sentinel-2 mosaic member {item_id} has an invalid footprint")
        donor = footprint & ~union
        for band in _BANDS:
            if masked[band].dims != template[band].dims or masked[band].shape != arrays[band].shape:
                raise ValueError(
                    f"Sentinel-2 mosaic member {item_id} has mismatched {band} dimensions"
                )
            arrays[band] = np.where(donor, masked[band].values, arrays[band])
        union |= footprint
        provenance.append({"item_id": item_id, **masked.attrs["s2_reflectance"]})
    out = template[list(_BANDS)].copy(deep=False)
    for band in _BANDS:
        out[band] = template[band].copy(data=arrays[band])
    out.attrs = {
        **template.attrs,
        "s2_reflectance": {
            "reflectance_harmonized": True,
            "item_ids": ids,
            "items": provenance,
            "donor_rule": DONOR_RULE,
            "source_footprint_px": int(union.sum()),
        },
    }
    return out, union
