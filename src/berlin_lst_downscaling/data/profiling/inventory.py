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
from berlin_lst_downscaling.data.ard.contract import BandSpec, contract_for_source
from berlin_lst_downscaling.data.profiling.contracts import get_histogram_spec
from berlin_lst_downscaling.data.profiling.models import (
    CompletenessResult,
    CoverageResult,
    HistogramSpec,
    ProfileAsset,
)
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

# Source → resolution mapping
_SOURCE_RESOLUTION = {
    "landsat-c2-l2": 100,
    "sentinel-2-l2a": 10,
    "ecostress": 70,
    "era5_land": 10,
    "shadow_building": 10,
    "shadow_vegetation": 10,
    "terrain_height": 10,
    "vegetation_height": 10,
    "building_dsm": 10,
    "vegetation_dsm": 10,
    "combined_dsm": 10,
    "lod2_morphology": 10,
    "imperviousness": 10,
    "svf": 10,
    "horizon_building": 10,
    "horizon_vegetation": 10,
}


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
        grid = canon_grid_for_resolution(_SOURCE_RESOLUTION[source])

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
                expected_band_contracts=contract.output_bands,
                expected_histogram_specs=histogram_specs,
                resolution_m=_SOURCE_RESOLUTION[source],
            )
        )

    return assets


# Flag band contract (separate uint8 COG, 255=nodata)
_FLAG_BAND = BandSpec(
    name="flag",
    dtype="uint8",
    nodata=255,
    description="Quality flag bitmask",
)


def build_ard_flag_assets(
    ledger_uri: str = _ARD_LEDGER,
) -> list[ProfileAsset]:
    """Build expected ARD flag assets from ledger."""
    ledger = _read_parquet_table(ledger_uri)

    assets: list[ProfileAsset] = []
    for i in range(ledger.num_rows):
        row = ledger.slice(i, 1).to_pydict()
        scene_id = row["scene_id"][0]
        source = row["source"][0]
        flag_uri = row.get("path_flag", [None])[0]

        if not flag_uri:
            continue

        year = int(row["year"][0])
        partition = "training" if year in _TRAINING_YEARS else "inference"

        grid = canon_grid_for_resolution(_SOURCE_RESOLUTION.get(source, 10))

        assets.append(
            ProfileAsset(
                item_id=f"{scene_id}__flag",
                source=f"{source}__flag",
                cog_uri=flag_uri,
                partition=partition,
                year=year,
                expected_crs="EPSG:25833",
                expected_resolution=grid.transform.a,
                expected_shape=(grid.shape.x, grid.shape.y),
                expected_bands=1,
                expected_band_specs=("flag",),
                expected_band_contracts=(_FLAG_BAND,),
                resolution_m=_SOURCE_RESOLUTION.get(source, 10),
            )
        )

    return assets


def build_static_assets(
    sources_root: str = _STATIC_SOURCES_ROOT,
    derived_root: str = _STATIC_DERIVED_ROOT,
) -> list[ProfileAsset]:
    """Build expected static source and derived assets."""
    assets: list[ProfileAsset] = []

    # Static sources
    try:
        source_ledger = _read_parquet_table(f"{sources_root}/ledger.parquet")
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

            # Get contract for this source
            contract = _contract_for_static_source(source)
            if contract is None:
                _logger.warning("No contract for static source %s, skipping", source)
                continue

            grid = canon_grid_for_resolution(10)
            band_specs = tuple(b.name for b in contract.output_bands)
            histogram_specs = tuple(_histogram_spec_for_band(b.name) for b in contract.output_bands)

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
                    expected_resolution=grid.transform.a,
                    expected_shape=(grid.shape.x, grid.shape.y),
                    expected_bands=len(contract.output_bands),
                    expected_band_specs=band_specs,
                    expected_band_contracts=contract.output_bands,
                    expected_histogram_specs=histogram_specs,
                    resolution_m=10,
                )
            )
    except Exception as e:
        _logger.warning("Failed to load static sources ledger: %s", e)

    # Static derived
    try:
        derived_ledger = _read_parquet_table(
            f"{derived_root}/_state/static/derived/ledger.parquet"
        )
        for i in range(derived_ledger.num_rows):
            row = derived_ledger.slice(i, 1).to_pydict()
            item_id = row["item_id"][0]
            source = row["source"][0]
            output_uri = row.get("output_uri", [None])[0]

            if not output_uri:
                continue

            # Get contract for this source
            contract = _contract_for_static_source(source)
            if contract is None:
                _logger.warning("No contract for static derived source %s, skipping", source)
                continue

            grid = canon_grid_for_resolution(10)
            band_specs = tuple(b.name for b in contract.output_bands)
            histogram_specs = tuple(_histogram_spec_for_band(b.name) for b in contract.output_bands)

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
                    expected_resolution=grid.transform.a,
                    expected_shape=(grid.shape.x, grid.shape.y),
                    expected_bands=len(contract.output_bands),
                    expected_band_specs=band_specs,
                    expected_band_contracts=contract.output_bands,
                    expected_histogram_specs=histogram_specs,
                    resolution_m=10,
                )
            )
    except Exception as e:
        _logger.warning("Failed to load static derived ledger: %s", e)

    return assets


def _contract_for_static_source(source: str) -> Any:
    """Return the Contract for a static source, or None if unmapped."""
    from berlin_lst_downscaling.data.ard.contract import Contract  # noqa: F401
    from berlin_lst_downscaling.data.secondary.dgm import contract_for_terrain_height
    from berlin_lst_downscaling.data.secondary.dsm import (
        contract_for_building_dsm,
        contract_for_combined_dsm,
        contract_for_vegetation_dsm,
    )
    from berlin_lst_downscaling.data.secondary.horizon import contract_for_horizon
    from berlin_lst_downscaling.data.secondary.imperviousness import (
        contract_for_imperviousness,
    )
    from berlin_lst_downscaling.data.secondary.lod2 import contract_for_lod2_morphology
    from berlin_lst_downscaling.data.secondary.svf import contract_for_svf
    from berlin_lst_downscaling.data.secondary.vegetation_height import (
        contract_for_vegetation_height,
    )

    factories: dict[str, Any] = {
        "terrain_height": contract_for_terrain_height,
        "vegetation_height": contract_for_vegetation_height,
        "building_dsm": contract_for_building_dsm,
        "vegetation_dsm": contract_for_vegetation_dsm,
        "combined_dsm": contract_for_combined_dsm,
        "lod2_morphology": contract_for_lod2_morphology,
        "imperviousness": contract_for_imperviousness,
        "svf": contract_for_svf,
    }

    if source in factories:
        return factories[source]()

    # Horizon sources: horizon_building, horizon_vegetation
    if source.startswith("horizon_"):
        component = source.split("_", 1)[1]
        return contract_for_horizon(component)

    return None


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

            # Extract scene_id from item_id (format: "{source}_{scene_id}")
            # e.g., "era5_land_LC08_L2SP_192023_20170526_02_T1"
            if item_id.startswith(f"{source}_"):
                scene_id = item_id[len(source) + 1 :]
            else:
                scene_id = item_id

            # Determine year from scene_id (format: "LC09_..._YYYYMMDD_...")
            try:
                # Try to find the date part (YYYYMMDD) in the scene_id
                parts = scene_id.split("_")
                year = None
                for part in parts:
                    if len(part) == 8 and part.isdigit():
                        year = int(part[:4])
                        break
            except (IndexError, ValueError):
                year = None

            # Build expected paths
            product_dir = f"{root.rstrip('/')}/ard/dynamic/{source}/{scene_id}"
            cog_path = f"{product_dir}/{source}_{scene_id}.tif"
            stac_path = f"{product_dir}/{source}_{scene_id}.stac.json"
            provenance_path = f"{product_dir}/provenance.json"
            completion_path = f"{product_dir}/complete.json"

            # Get contract for dynamic source
            contract = _contract_for_dynamic_source(source)
            if contract is None:
                _logger.warning("No contract for dynamic source %s, skipping", source)
                continue

            # Skip if year cannot be determined (required for non-static partitions)
            if year is None:
                _logger.warning("Cannot determine year for %s, skipping", item_id)
                continue

            grid = canon_grid_for_resolution(10)
            band_specs = tuple(b.name for b in contract.output_bands)
            histogram_specs = tuple(_histogram_spec_for_band(b.name) for b in contract.output_bands)

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
                    expected_bands=len(contract.output_bands),
                    expected_band_specs=band_specs,
                    expected_band_contracts=contract.output_bands,
                    expected_histogram_specs=histogram_specs,
                    resolution_m=10,
                )
            )

    return assets


def _contract_for_dynamic_source(source: str) -> Any:
    """Return the Contract for a dynamic source, or None if unmapped."""
    from berlin_lst_downscaling.data.ard.contract import Contract  # noqa: F401
    from berlin_lst_downscaling.data.dynamic.era5 import contract_for_era5_scene
    from berlin_lst_downscaling.data.dynamic.shadows import contract_for_shadow

    if source == "era5_land":
        return contract_for_era5_scene()
    elif source.startswith("shadow_"):
        component = source.split("_", 1)[1]
        return contract_for_shadow(component)
    return None


def build_all_assets(
    manifest_uri: str = _MANIFEST_URI,
    ard_ledger_uri: str = _ARD_LEDGER,
    dynamic_root: str = _DYNAMIC_FULL_ROOT,
    inference_root: str = _DYNAMIC_INFERENCE_ROOT,
    static_sources_root: str = _STATIC_SOURCES_ROOT,
    static_derived_root: str = _STATIC_DERIVED_ROOT,
) -> list[ProfileAsset]:
    """Build the complete expected asset inventory."""
    assets: list[ProfileAsset] = []
    assets.extend(build_ard_assets(manifest_uri, ard_ledger_uri))
    assets.extend(build_ard_flag_assets(ard_ledger_uri))
    assets.extend(build_static_assets(static_sources_root, static_derived_root))
    assets.extend(build_dynamic_assets(dynamic_root, inference_root))
    return assets


def select_assets(
    assets: list[ProfileAsset],
    limit_per_source_partition: int | None = None,
) -> list[ProfileAsset]:
    """Select a subset of assets for smoke runs.

    If limit_per_source_partition is set, keeps at most that many assets
    per (source, partition) group, sorted by stable asset key.
    """
    if limit_per_source_partition is None:
        return assets

    # Group by (source, partition)
    groups: dict[tuple[str, str], list[ProfileAsset]] = {}
    for asset in assets:
        key = (asset.source, asset.partition)
        groups.setdefault(key, []).append(asset)

    selected: list[ProfileAsset] = []
    for group_assets in groups.values():
        # Sort by item_id for determinism
        sorted_assets = sorted(group_assets, key=lambda a: a.item_id)
        selected.extend(sorted_assets[:limit_per_source_partition])

    return selected


def _histogram_spec_for_band(
    band_name: str,
) -> HistogramSpec | None:
    """Return the HistogramSpec for a given band, or None if unmapped."""
    return get_histogram_spec(band_name)


def check_manifest_ledger_completeness(
    manifest_uri: str = _MANIFEST_URI,
    ledger_uri: str = _ARD_LEDGER,
) -> CompletenessResult:
    """Check manifest↔ARD-ledger completeness over ``done`` rows.

    Only ledger rows in ``status == done`` count as published COGs. Any
    missing, extra, or duplicate key renders the inventory incomplete.
    """
    bundle, _ = load_bundle(manifest_uri, require_item_href=False)
    manifest = bundle.manifest_table

    manifest_keys: list[str] = []
    for i in range(manifest.num_rows):
        row = manifest.slice(i, 1).to_pydict()
        manifest_keys.append(f"{row['scene_id'][0]}|{row['source'][0]}")

    ledger = _read_parquet_table(ledger_uri)
    ledger_keys: list[str] = []
    status_col = ledger.column("status").to_pylist()
    scene_col = ledger.column("scene_id").to_pylist()
    source_col = ledger.column("source").to_pylist()
    for i in range(ledger.num_rows):
        if status_col[i] == "done":
            ledger_keys.append(f"{scene_col[i]}|{source_col[i]}")

    manifest_set = set(manifest_keys)
    ledger_set = set(ledger_keys)

    seen: set[str] = set()
    duplicates: list[str] = []
    for k in manifest_keys:
        if k in seen:
            duplicates.append(k)
        seen.add(k)

    return CompletenessResult(
        manifest_key_count=len(manifest_keys),
        ledger_key_count=len(ledger_keys),
        missing_in_ledger=sorted(manifest_set - ledger_set),
        extra_in_ledger=sorted(ledger_set - manifest_set),
        duplicate_keys=sorted(set(duplicates)),
    )


_DYNAMIC_SOURCES = ("era5_land", "shadow_building", "shadow_vegetation")


def check_dynamic_coverage(
    manifest_uri: str = _MANIFEST_URI,
    dynamic_root: str = _DYNAMIC_FULL_ROOT,
    inference_root: str = _DYNAMIC_INFERENCE_ROOT,
) -> list[CoverageResult]:
    """Check dynamic COG coverage against manifest Landsat anchors.

    Every Landsat anchor scene is expected to publish one COG per dynamic
    source. Returns one ``CoverageResult`` per partition (training,
    inference). Only ``done`` ledger rows count as published.
    """
    bundle, _ = load_bundle(manifest_uri, require_item_href=False)
    manifest = bundle.manifest_table

    expected: dict[str, set[str]] = {"training": set(), "inference": set()}
    src_col = manifest.column("source").to_pylist()
    role_col = manifest.column("role").to_pylist()
    scene_col = manifest.column("scene_id").to_pylist()
    year_col = manifest.column("year").to_pylist()
    for i in range(manifest.num_rows):
        if src_col[i] != "landsat-c2-l2" or role_col[i] != "anchor":
            continue
        year = int(year_col[i])
        partition = "training" if year in _TRAINING_YEARS else "inference"
        scene_id = scene_col[i]
        for source in _DYNAMIC_SOURCES:
            expected[partition].add(f"{source}|{scene_id}")

    found: dict[str, set[str]] = {"training": set(), "inference": set()}
    for root, default_partition in [
        (dynamic_root, "training"),
        (inference_root, "inference"),
    ]:
        try:
            ledger = _read_parquet_table(f"{root}/_state/dynamic/ledger.parquet")
        except Exception as e:
            raise RuntimeError(f"Failed to read dynamic ledger at {root}: {e}") from e
        status_col = ledger.column("status").to_pylist()
        item_col = ledger.column("item_id").to_pylist()
        source_col = ledger.column("source").to_pylist()
        for i in range(ledger.num_rows):
            if status_col[i] != "done":
                continue
            source = source_col[i]
            item_id = item_col[i]
            if item_id.startswith(f"{source}_"):
                scene_id = item_id[len(source) + 1 :]
            else:
                _logger.warning(
                    "Unexpected dynamic item_id format (skipping coverage entry): %s",
                    item_id,
                )
                continue
            found[default_partition].add(f"{source}|{scene_id}")

    results: list[CoverageResult] = []
    for partition in ("training", "inference"):
        exp = expected[partition]
        fnd = found[partition]
        results.append(
            CoverageResult(
                partition=partition,
                expected=len(exp),
                found=len(fnd),
                missing=sorted(exp - fnd),
                extra=sorted(fnd - exp),
            )
        )
    return results


__all__ = [
    "build_ard_assets",
    "build_ard_flag_assets",
    "build_static_assets",
    "build_dynamic_assets",
    "build_all_assets",
    "select_assets",
    "check_manifest_ledger_completeness",
    "check_dynamic_coverage",
]
