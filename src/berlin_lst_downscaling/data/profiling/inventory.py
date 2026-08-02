"""Build expected asset inventory from manifest and ledger families.

Reads the canonical manifest, ARD ledger, static source/derived ledgers,
and dynamic ledger to produce the complete set of expected COGs.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from berlin_lst_downscaling.common.grid import canon_grid_for_resolution
from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.profiling.contracts import get_histogram_spec
from berlin_lst_downscaling.data.profiling.models import HistogramSpec, ProfileAsset
from berlin_lst_downscaling.data.selection.validate import load_bundle

_logger = logging.getLogger(__name__)

# Canonical GCS roots (from docs/delivered-implementation.md)
_MANIFEST_URI = (
    "gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
)
_ARD_LEDGER = "gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet"
_STATIC_SOURCES_ROOT = "gs://berlin-lst-data/static/sources/full"
_STATIC_DERIVED_ROOT = "gs://berlin-lst-data/static/derived/full"
_DYNAMIC_FULL_ROOT = "gs://berlin-lst-data/dynamic/full"
_DYNAMIC_INFERENCE_ROOT = "gs://berlin-lst-data/dynamic/inference/2026"

# Dynamic source band counts and expected grid
_DYNAMIC_BAND_COUNTS = {
    "era5_land": 8,
    "shadow_building": 1,
    "shadow_vegetation": 1,
}

# Training/inference split
_TRAINING_YEARS = set(range(2017, 2026))
_INFERENCE_YEAR = 2026


def _read_parquet_table(uri: str) -> pa.Table:
    """Read a Parquet table from local path or GCS."""
    from berlin_lst_downscaling.data.io.storage import read_bytes

    if uri.startswith("gs://"):
        return pq.read_table(io.BytesIO(read_bytes(uri)))
    return pq.read_table(uri)


def build_ard_assets(
    manifest_uri: str = _MANIFEST_URI,
    ledger_uri: str = _ARD_LEDGER,
) -> list[ProfileAsset]:
    """Build expected ARD assets from manifest and ledger."""
    bundle, _ = load_bundle(manifest_uri, require_item_href=False)
    manifest = bundle.manifest_table

    # Load ledger for sidecar paths
    ledger = _read_parquet_table(ledger_uri)

    # Build ledger lookup: (scene_id, source) -> row
    ledger_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(ledger.num_rows):
        row = ledger.slice(i, 1).to_pydict()
        key = (row["scene_id"][0], row["source"][0])
        ledger_lookup[key] = {k: v[0] for k, v in row.items()}

    assets: list[ProfileAsset] = []
    for i in range(manifest.num_rows):
        row = manifest.slice(i, 1).to_pydict()
        scene_id = row["scene_id"][0]
        source = row["source"][0]
        year = int(row["year"][0])
        acquisition_dt = row["acquisition_datetime"][0]

        # Determine season (May–September policy)
        if hasattr(acquisition_dt, "month"):
            month = acquisition_dt.month
        else:
            _logger.warning("Missing month in acquisition datetime for %s", scene_id)
            month = 5  # Default to summer for May–September policy
        season = "summer" if 5 <= month <= 9 else "shoulder"

        # Partition
        partition = "training" if year in _TRAINING_YEARS else "inference"

        # Get ledger row for sidecar paths
        ledger_row = ledger_lookup.get((scene_id, source), {})

        # Build expected COG path
        contract = contract_for_source(source)
        grid = canon_grid_for_resolution(10 if source != "ecostress" else 70)

        # Expected bands from contract
        band_specs = tuple(b.name for b in contract.output_bands)
        histogram_specs = tuple(_histogram_spec_for_band(b.name) for b in contract.output_bands)

        assets.append(
            ProfileAsset(
                item_id=scene_id,
                source=source,
                cog_uri=ledger_row.get("path_cog", ""),
                stac_uri=ledger_row.get("path_stac"),
                provenance_uri=f"{ledger_row.get('path_cog', '').rsplit('/', 1)[0]}/provenance.json"
                if ledger_row.get("path_cog")
                else None,
                completion_uri=f"{ledger_row.get('path_cog', '').rsplit('/', 1)[0]}/complete.json"
                if ledger_row.get("path_cog")
                else None,
                partition=partition,
                year=year,
                season=season,
                expected_crs="EPSG:25833",
                expected_resolution=grid.transform.a,
                expected_shape=(grid.shape.x, grid.shape.y),
                expected_bands=len(contract.output_bands),
                expected_band_specs=band_specs,
                expected_histogram_specs=histogram_specs,
                resolution_m=10 if source != "ecostress" else 70,
            )
        )

    return assets


def build_static_assets() -> list[ProfileAsset]:
    """Build expected static source and derived assets."""
    assets: list[ProfileAsset] = []

    # Static sources
    try:
        source_ledger = _read_parquet_table(f"{_STATIC_SOURCES_ROOT}/ledger.parquet")
        for i in range(source_ledger.num_rows):
            row = source_ledger.slice(i, 1).to_pydict()
            item_id = row["item_id"][0]
            source = row["source"][0]
            period = row["period_or_vintage"][0]
            output_uri = row.get("output_uri", [None])[0]

            if not output_uri:
                continue

            # Parse year from period if possible
            try:
                year = int(period)
            except (ValueError, TypeError):
                year = None

            # Static products are always shared_static
            assets.append(
                ProfileAsset(
                    item_id=item_id,
                    source=source,
                    cog_uri=output_uri,
                    stac_uri=row.get("stac_uri", [None])[0],
                    provenance_uri=row.get("provenance_uri", [None])[0],
                    completion_uri=row.get("completion_uri", [None])[0],
                    partition="shared_static",
                    year=year,
                    expected_crs="EPSG:25833",
                    expected_resolution=10.0,
                    expected_shape=(
                        canon_grid_for_resolution(10).shape.x,
                        canon_grid_for_resolution(10).shape.y,
                    ),
                    expected_bands=1,  # Will be overridden by contract lookup
                    resolution_m=10,
                )
            )
    except Exception as e:
        _logger.warning("Failed to load static sources ledger: %s", e)

    # Static derived
    try:
        derived_ledger = _read_parquet_table(
            f"{_STATIC_DERIVED_ROOT}/_state/static/derived/ledger.parquet"
        )
        for i in range(derived_ledger.num_rows):
            row = derived_ledger.slice(i, 1).to_pydict()
            item_id = row["item_id"][0]
            source = row["source"][0]
            output_uri = row.get("output_uri", [None])[0]

            if not output_uri:
                continue

            assets.append(
                ProfileAsset(
                    item_id=item_id,
                    source=source,
                    cog_uri=output_uri,
                    stac_uri=row.get("stac_uri", [None])[0],
                    provenance_uri=row.get("provenance_uri", [None])[0],
                    completion_uri=row.get("completion_uri", [None])[0],
                    partition="shared_static",
                    expected_crs="EPSG:25833",
                    expected_resolution=10.0,
                    expected_shape=(
                        canon_grid_for_resolution(10).shape.x,
                        canon_grid_for_resolution(10).shape.y,
                    ),
                    expected_bands=1,  # Will be overridden by contract lookup
                    resolution_m=10,
                )
            )
    except Exception as e:
        _logger.warning("Failed to load static derived ledger: %s", e)

    return assets


def build_dynamic_assets(
    dynamic_root: str = _DYNAMIC_FULL_ROOT,
    inference_root: str = _DYNAMIC_INFERENCE_ROOT,
) -> list[ProfileAsset]:
    """Build expected dynamic assets from ledger."""
    assets: list[ProfileAsset] = []

    for root, default_partition, default_role in [
        (dynamic_root, "training", "anchor"),
        (inference_root, "inference", "inference"),
    ]:
        try:
            ledger = _read_parquet_table(f"{root}/_state/dynamic/ledger.parquet")
        except Exception as e:
            _logger.warning("Failed to load dynamic ledger from %s: %s", root, e)
            continue

        for i in range(ledger.num_rows):
            row = ledger.slice(i, 1).to_pydict()
            item_id = row["item_id"][0]
            source = row["source"][0]
            status = row["status"][0]
            role = row.get("role", [default_role])[0]

            if status != "done":
                continue

            # Extract scene_id from item_id (format: "{scene_id}__{source}")
            scene_id = item_id.split("__")[0] if "__" in item_id else item_id

            # Determine year from scene_id (format: "LC09_..._YYYYMMDD_...")
            try:
                year = int(scene_id.split("_")[3][:4])
            except (IndexError, ValueError):
                year = None

            # Build expected paths
            product_dir = f"{root.rstrip('/')}/ard/dynamic/{source}/{scene_id}"
            cog_path = f"{product_dir}/{source}_{scene_id}.tif"
            stac_path = f"{product_dir}/{source}_{scene_id}.stac.json"
            provenance_path = f"{product_dir}/provenance.json"
            completion_path = f"{product_dir}/complete.json"

            # Determine grid based on source
            if source == "era5_land":
                grid = canon_grid_for_resolution(10)
                band_count = 8
                band_specs = (
                    "t2m_scene",
                    "ssrd_scene",
                    "ssrd_antecedent_72h_mean",
                    "vpd_scene",
                    "wind_speed_10m_scene",
                    "tp_0_24h",
                    "tp_24_48h",
                    "tp_48_72h",
                )
            elif source in ("shadow_building", "shadow_vegetation"):
                grid = canon_grid_for_resolution(10)
                band_count = 1
                band_specs = (source,)
            else:
                continue

            # Partition - explicit assignment for type safety
            if role == "inference":
                partition: Literal["training", "inference", "shared_static"] = "inference"
            else:
                partition = "training" if default_partition == "training" else "inference"

            assets.append(
                ProfileAsset(
                    item_id=item_id,
                    source=source,
                    cog_uri=cog_path,
                    stac_uri=stac_path,
                    provenance_uri=provenance_path,
                    completion_uri=completion_path,
                    partition=partition,
                    year=year,
                    expected_crs="EPSG:25833",
                    expected_resolution=grid.transform.a,
                    expected_shape=(grid.shape.x, grid.shape.y),
                    expected_bands=band_count,
                    expected_band_specs=band_specs,
                    resolution_m=10,
                )
            )

    return assets


def build_all_assets(
    manifest_uri: str = _MANIFEST_URI,
    ard_ledger_uri: str = _ARD_LEDGER,
    dynamic_root: str = _DYNAMIC_FULL_ROOT,
    inference_root: str = _DYNAMIC_INFERENCE_ROOT,
) -> list[ProfileAsset]:
    """Build the complete expected asset inventory."""
    assets: list[ProfileAsset] = []
    assets.extend(build_ard_assets(manifest_uri, ard_ledger_uri))
    assets.extend(build_static_assets())
    assets.extend(build_dynamic_assets(dynamic_root, inference_root))
    return assets


def _histogram_spec_for_band(
    band_name: str,
) -> HistogramSpec | None:
    """Return the HistogramSpec for a given band, or None if unmapped."""
    return get_histogram_spec(band_name)


__all__ = [
    "build_ard_assets",
    "build_static_assets",
    "build_dynamic_assets",
    "build_all_assets",
]
