"""Feature-stack finalisation — data COG, validity-mask COG, sidecars.

Each scene publishes five co-located artifacts under its product directory
(in this exact order, because GCS cannot publish multiple blobs
atomically — the completion marker is the final visibility gate):

1. ``<scene_id>.tif``                 — 28-band float32 feature COG
2. ``<scene_id>.feature_valid.tif``   — uint8 0/1 validity-mask COG
3. ``provenance.json``                — inputs, policy, coverage
4. ``<scene_id>.stac.json``           — STAC Item (data + mask assets)
5. ``complete.json``                  — written last

The data COG is validated against the canonical grid before any sidecar
is written; the mask COG is validated structurally by the writer's strict
COG check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor on xr.Dataset
import xarray as xr
from google.api_core.exceptions import GoogleAPIError
from odc.geo.geobox import GeoBox
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.validate import validate_cog
from berlin_lst_downscaling.data.ard.writer import write_cog_atomic, write_flag_cog_atomic
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNELS,
    FEATURE_SCHEMA_VERSION,
)
from berlin_lst_downscaling.data.io import atomic_write, exists, read_bytes

_logger = logging.getLogger(__name__)

# STAC extension schema URLs (Projection v2.0.0, Raster v1.1.0) — pinned
# like the secondary product builder (data/secondary/product.py).
_PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
_RASTER_EXT = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
_STAC_VERSION = "1.0.0"


@dataclass
class PreparedFeatureProduct:
    """Payload produced by the composer and handed to finalisation."""

    scene_id: str
    dataset: xr.Dataset  # 28 float32 bands on the canonical grid
    mask: np.ndarray  # uint8 (H, W), 1 = valid
    config_hash: str
    acquisition_datetime: str  # RFC 3339 (S2 acquisition)
    source_metadata: dict  # resolved input URIs + policy
    coverage: dict  # composer coverage metrics


@dataclass
class FeatureArtifacts:
    """The five URIs emitted by finalisation."""

    cog_uri: str
    mask_uri: str
    provenance_uri: str
    stac_uri: str
    completion_uri: str


# The feature stack is a first-class output contract: one BandSpec per
# channel, so ``validate_cog`` checks band count, dtypes are float32, and
# valid_range is carried for downstream QA.
def _feature_ard_contract() -> Contract:
    bands = tuple(
        BandSpec(
            name=ch.name,
            dtype="float32",
            nodata=float("nan"),
            description=ch.description,
            unit=ch.unit,
            valid_range=ch.valid_range,
        )
        for ch in FEATURE_CHANNELS
    )
    return Contract(
        source="feature_stack",
        target_crs="EPSG:25833",
        output_bands=bands,
        tiling=TilingSpec(),
        schema_version=1,
    )


_FEATURE_CONTRACT = _feature_ard_contract()


def finalize_feature_product(
    prepared: PreparedFeatureProduct,
    grid: GeoBox,
    product_dir: str,
    run_id: str,
) -> FeatureArtifacts:
    """Write the five final artifacts for one feature stack."""
    completed_at = datetime.now(UTC).isoformat()
    base = product_dir.rstrip("/")
    cog_uri = f"{base}/{prepared.scene_id}.tif"
    mask_uri = f"{base}/{prepared.scene_id}.feature_valid.tif"
    provenance_uri = f"{base}/provenance.json"
    stac_uri = f"{base}/{prepared.scene_id}.stac.json"
    completion_uri = f"{base}/complete.json"

    # A completed scene is immutable: refuse to touch any artifact of a
    # scene whose completion marker already exists (guards against an
    # accidental re-run overwriting a published stack before the
    # create-only marker write below would fail).
    if exists(completion_uri):
        raise FileExistsError(
            f"scene {prepared.scene_id} already published: {completion_uri}"
        )

    # Per-scene publish lock (atomic create-only): exactly one publisher
    # may hold the lock; a concurrent publisher aborts here instead of
    # interleaving artifact writes with the winner. Released in ``finally``
    # after the create-only marker commits the scene. A stale lock after a
    # hard kill is an explicit operator state (inspect, then delete).
    lock_uri = f"{base}/.publish.lock"
    if lock_uri.startswith("gs://"):
        try:
            atomic_write(
                lock_uri,
                json.dumps({"run_id": run_id, "started_at": completed_at}, indent=2),
                overwrite=False,
                if_generation_match=0,
            )
        except FileExistsError:
            raise RuntimeError(
                f"scene {prepared.scene_id} is being published by another run "
                f"(lock {lock_uri})"
            ) from None
    else:
        import os

        lock_path = os.path.expanduser(lock_uri)
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(
                    json.dumps({"run_id": run_id, "started_at": completed_at}, indent=2)
                )
        except FileExistsError:
            raise RuntimeError(
                f"scene {prepared.scene_id} is being published by another run "
                f"(lock {lock_uri})"
            ) from None

    try:
        return _finalize_locked(
            prepared, grid, base, run_id, completed_at, cog_uri, mask_uri,
            provenance_uri, stac_uri, completion_uri,
        )
    finally:
        _delete_uri(lock_uri)


def _delete_uri(uri: str) -> None:
    """Delete one object best-effort (publish-lock cleanup)."""
    if uri.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

        bucket_name, key = _parse_gs_uri(uri)
        try:
            _gcs_client().bucket(bucket_name).blob(key).delete()
        except GoogleAPIError as exc:
            _logger.warning("publish-lock cleanup failed for %s: %s", uri, exc)
    else:
        import os

        try:
            os.remove(os.path.expanduser(uri))
        except OSError as exc:
            _logger.warning("publish-lock cleanup failed for %s: %s", uri, exc)


def _finalize_locked(
    prepared: PreparedFeatureProduct,
    grid: GeoBox,
    base: str,
    run_id: str,
    completed_at: str,
    cog_uri: str,
    mask_uri: str,
    provenance_uri: str,
    stac_uri: str,
    completion_uri: str,
) -> FeatureArtifacts:
    """Write the five artifacts while holding the per-scene publish lock."""
    write_cog_atomic(prepared.dataset, cog_uri, _FEATURE_CONTRACT, overwrite=True)

    # Remote validation reads (data COG validation + mask pair scan) run
    # with the GDAL directory listing disabled. GDAL caches a process-wide
    # listing of a scene folder when the data COG is first opened; the mask
    # COG is uploaded into that same folder afterwards, so a subsequent open
    # would consult the stale listing and report "does not exist" even though
    # the object is present (GCS is strongly consistent). See
    # https://github.com/OSGeo/gdal/issues/11351.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        # The generic minimum-valid-fraction gate is disabled: sparse or all-zero
        # stacks are structurally valid (their validity is governed by the
        # companion mask, not by bulk non-NaN density). All other COG checks
        # (CRS, grid, band count, dtype, strict-COG) still apply.
        vig = validate_cog(
            cog_uri,
            _FEATURE_CONTRACT,
            grid,
            require_min_valid_fraction=False,
        )
        if not vig.ok:
            raise ValueError(
                f"COG validation failed for {prepared.scene_id}: {'; '.join(vig.errors)}"
            )

        # ── 2. validity-mask COG ───────────────────────────────────────────
        mask_da = xr.DataArray(
            prepared.mask,
            dims=["y", "x"],
            coords={
                "x": prepared.dataset.x,
                "y": prepared.dataset.y,
            },
        )
        mask_da = mask_da.rio.write_crs(str(grid.crs))
        mask_da = mask_da.rio.write_transform(grid.transform)
        write_flag_cog_atomic(mask_da, mask_uri, _FEATURE_CONTRACT, overwrite=True)

        # ── 2b. pair validation ────────────────────────────────────────────
        # mask is uint8 {0,1} and mask == 1 ⇔ all 28 data bands finite.
        # An all-zero mask is valid (a fully sparse stack is still a coherent
        # product). Runs after both COGs are written, before any sidecar.
        _validate_mask_pair(prepared, mask_uri, cog_uri)

    # ── 3. provenance ──────────────────────────────────────────────────
    provenance = {
        "pipeline": "features",
        "scene_id": prepared.scene_id,
        "config_hash": prepared.config_hash,
        "run_id": run_id,
        "completed_at": completed_at,
        "acquisition_datetime": prepared.acquisition_datetime,
        "channel_order": [ch.name for ch in FEATURE_CHANNELS],
        "schema_version": FEATURE_SCHEMA_VERSION,
        "vegetation_height_policy": prepared.source_metadata["vegetation_height_policy"],
        "aoi_uri": prepared.source_metadata["aoi_uri"],
        "aoi_fingerprint": prepared.source_metadata["aoi_fingerprint"],
        "coverage": prepared.coverage,
        "inputs": prepared.source_metadata["inputs"],
        "lod_vintage": prepared.source_metadata.get("lod_vintage"),
        "lod_coverage": prepared.source_metadata.get("lod_coverage", {}),
        "mask_semantics": (
            "feature_valid == 1 iff inside Berlin AOI, S2 flag == 0, and all 28 "
            "channels finite and in-range; availability is per channel (only "
            "unavailable bands are NaN), all channels are NaN only outside the AOI"
        ),
    }
    atomic_write(provenance_uri, json.dumps(provenance, indent=2), overwrite=True)

    # ── 3b. provenance read-back (immutability hardening) ─────────────
    # The create-only marker below atomically commits the scene. Verify
    # the artifacts just written carry THIS run's identity before that
    # commit: a concurrent publisher that clobbered any artifact would
    # leave a differing config hash / scene id, and committing would
    # produce a mixed-identity immutable scene. Abort instead.
    written = json.loads(read_bytes(provenance_uri))
    if (
        written.get("config_hash") != prepared.config_hash
        or written.get("scene_id") != prepared.scene_id
    ):
        raise ValueError(
            f"scene {prepared.scene_id}: written provenance does not match this run "
            f"(config_hash {written.get('config_hash')!r}, scene {written.get('scene_id')!r})"
        )

    # ── 4. STAC Item ───────────────────────────────────────────────────
    stac_item = _build_feature_stac_item(prepared, grid, cog_uri, mask_uri, provenance_uri)
    atomic_write(stac_uri, json.dumps(stac_item, indent=2), overwrite=True)

    # ── 5. completion marker (last, create-only) ───────────────────────
    atomic_write(
        completion_uri,
        json.dumps({"published_at": completed_at, "run_id": run_id}, indent=2),
        overwrite=False,
        if_generation_match=0,
    )

    return FeatureArtifacts(
        cog_uri=cog_uri,
        mask_uri=mask_uri,
        provenance_uri=provenance_uri,
        stac_uri=stac_uri,
        completion_uri=completion_uri,
    )


def _validate_mask_pair(
    prepared: PreparedFeatureProduct,
    mask_uri: str,
    cog_uri: str,
) -> None:
    """Verify mask semantics on the written COGs before any sidecar.

    Checks that the mask COG holds uint8 values in {0, 1}, matches the
    composed in-memory mask, and that, pixelwise, ``mask == 1`` implies
    all 28 bands are finite and within their declared ranges. The
    converse is *not* required: availability is per channel, so
    ``mask == 0`` may legitimately hold finite values in any channel
    (only the unavailable bands are NaN). An all-zero mask is valid.
    Raises ``ValueError`` on any violation so a bad pair never reaches
    the completion marker.

    Scanned blockwise (1024 px tiles) so the peak memory stays bounded
    inside the per-scene subprocess — a full 28-band read would re-introduce
    the large in-memory footprint the per-scene isolation exists to avoid.

    Callers must wrap remote reads in
    ``rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")`` so a freshly
    uploaded sibling COG is not hidden by GDAL's cached folder listing.
    """
    from rasterio.windows import Window

    _TILE = 1024
    n_expected = len(FEATURE_CHANNELS)
    with rasterio.open(cog_uri) as src:
        if src.count != n_expected:
            raise ValueError(f"pair validation: expected {n_expected} data bands, got {src.count}")
        h, w = src.height, src.width
    with rasterio.open(mask_uri) as src:
        if src.dtypes[0] != "uint8":
            raise ValueError(f"pair validation: mask dtype {src.dtypes[0]!r}, expected 'uint8'")
        if (src.height, src.width) != (h, w):
            raise ValueError(f"pair validation: shape mismatch mask {(src.height, src.width)} "
                             f"vs data {(h, w)}")

    seen_values: set[int] = set()
    n_bad_mask = 0  # mask==1 but not all-finite-and-in-range
    n_written_mismatch = 0  # written COG mask differs from the composed mask
    with rasterio.open(cog_uri) as cog, rasterio.open(mask_uri) as msk:
        for r0 in range(0, h, _TILE):
            r1 = min(r0 + _TILE, h)
            for c0 in range(0, w, _TILE):
                c1 = min(c0 + _TILE, w)
                win = Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore[call-arg]
                data = cog.read(window=win)  # (28, bh, bw)
                mask = msk.read(1, window=win)
                seen_values.update(np.unique(mask).tolist())
                claim = mask == 1
                finite = np.isfinite(data)
                complete = np.all(finite, axis=0)
                for i, spec in enumerate(FEATURE_CHANNELS):
                    if spec.valid_range is None:
                        continue
                    lo, hi = spec.valid_range
                    complete &= (data[i] >= lo) & (data[i] <= hi)
                n_bad_mask += int(np.sum(claim & ~complete))
                expected = prepared.mask[r0:r1, c0:c1].astype(np.uint8)
                n_written_mismatch += int(np.sum(mask != expected))

    if not seen_values.issubset({0, 1}):
        raise ValueError(
            f"pair validation: mask has unexpected values {sorted(seen_values)}, "
            "expected only 0/1"
        )
    if n_bad_mask:
        raise ValueError(
            f"pair validation: mask==1 with non-finite/out-of-range values on "
            f"{n_bad_mask} pixels for {prepared.scene_id}"
        )
    if n_written_mismatch:
        raise ValueError(
            f"pair validation: written mask disagrees with composed mask on "
            f"{n_written_mismatch} pixels for {prepared.scene_id}"
        )


def _build_feature_stac_item(
    prepared: PreparedFeatureProduct,
    grid: GeoBox,
    cog_uri: str,
    mask_uri: str,
    provenance_uri: str,
) -> dict:
    """Build the STAC 1.0.0 Item with data + mask assets and 28 raster bands."""
    height, width = grid.shape.y, grid.shape.x
    transform = grid.transform
    bounds_native = array_bounds(height, width, transform)
    bbox_4326 = transform_bounds(str(grid.crs), "EPSG:4326", *bounds_native)

    raster_bands = []
    for spec in FEATURE_CHANNELS:
        band_entry: dict = {
            "data_type": "float32",
            "nodata": "nan",
            "spatial_resolution": abs(transform.a),
        }
        if spec.unit and spec.unit != "1":
            band_entry["unit"] = spec.unit
        raster_bands.append(band_entry)

    item: dict = {
        "stac_version": _STAC_VERSION,
        "stac_extensions": [_PROJ_EXT, _RASTER_EXT],
        "type": "Feature",
        "id": f"feature_stack-{prepared.scene_id}",
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
            "datetime": prepared.acquisition_datetime,
            "proj:code": str(grid.crs),
            "proj:shape": [height, width],
            "proj:transform": list(transform),
            "features:config_hash": prepared.config_hash,
            "features:channel_order": [ch.name for ch in FEATURE_CHANNELS],
        },
        "assets": {
            "data": {
                "href": cog_uri,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": "28-band scene feature stack",
                "roles": ["data"],
                "raster:bands": raster_bands,
            },
            "feature_valid": {
                "href": mask_uri,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": "Feature validity mask (1 = all channels valid)",
                "roles": ["mask"],
                "raster:bands": [
                    {"data_type": "uint8", "nodata": None, "spatial_resolution": abs(transform.a)}
                ],
            },
            "provenance": {
                "href": provenance_uri,
                "type": "application/json",
                "title": "Feature-stack provenance",
                "roles": ["metadata"],
            },
        },
        "links": [],
    }
    return item


__all__ = [
    "FeatureArtifacts",
    "PreparedFeatureProduct",
    "finalize_feature_product",
]
