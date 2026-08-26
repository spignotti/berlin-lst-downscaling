"""Training-eligibility publication — mask COG, provenance, completion marker.

Each assessable scene publishes three co-located artifacts under its
product directory (in this order — GCS cannot publish multiple blobs
atomically, so the completion marker is the final visibility gate):

1. ``<scene_id>.training_eligible_100m.tif`` — uint8 0/1 mask COG
2. ``provenance.json`` — policy hash, V3 config hash, counts, inputs
3. ``complete.json`` — written last (create-only)

The publication is guarded exactly like the feature stacks: an existing
completion marker makes the scene immutable; a per-scene create-only
publish lock serialises concurrent publishers; the provenance is
read back and identity-checked before the marker commits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from google.api_core.exceptions import GoogleAPIError
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.ard.writer import write_flag_cog_atomic
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.io import atomic_write, exists, read_bytes
from berlin_lst_downscaling.data.training.contracts import (
    TRAINING_SCHEMA_VERSION,
)
from berlin_lst_downscaling.data.training.eligibility import EligibilityResult
from berlin_lst_downscaling.data.training.paths import (
    eligibility_cog,
    eligibility_completion,
    eligibility_provenance,
)

_logger = logging.getLogger(__name__)


@dataclass
class EligibilityArtifacts:
    """The three URIs emitted by per-scene eligibility publication."""

    cog_uri: str
    provenance_uri: str
    completion_uri: str


# The eligibility mask is a first-class output contract: a single uint8
# band on the canonical 100 m grid, no nodata (0/1 are both meaningful).
def _eligibility_contract() -> Contract:
    return Contract(
        source="training_eligibility",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="training_eligible",
                dtype="uint8",
                nodata=None,
                description=(
                    "100 m training-eligibility mask: 1 iff valid Landsat target "
                    "and all 100 feature subpixels valid"
                ),
                valid_range=(0.0, 1.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=TRAINING_SCHEMA_VERSION,
    )


_ELIGIBILITY_CONTRACT = _eligibility_contract()


def publish_eligibility(
    *,
    result: EligibilityResult,
    grid_100m: GeoBox,
    output_root: str,
    run_id: str,
    policy_hash: str,
    v3_config_hash: str,
    feature_valid_uri: str,
    feature_provenance_uri: str,
    landsat_cog_uri: str,
    landsat_flag_uri: str,
    geometry_id: str,
) -> EligibilityArtifacts:
    """Publish one scene's eligibility mask, provenance, and completion marker."""
    completed_at = datetime.now(UTC).isoformat()
    base = output_root.rstrip("/")
    cog_uri = eligibility_cog(output_root, result.scene_id)
    provenance_uri = eligibility_provenance(output_root, result.scene_id)
    completion_uri = eligibility_completion(output_root, result.scene_id)

    # A completed scene is immutable: refuse to touch a scene whose
    # completion marker already exists (guards against an accidental
    # re-run overwriting a published artifact before the create-only
    # marker write below would fail).
    if exists(completion_uri):
        raise FileExistsError(f"scene {result.scene_id} already published: {completion_uri}")

    # Per-scene publish lock (atomic create-only) — exactly one publisher
    # may hold the lock; a concurrent publisher aborts here. Released in
    # ``finally`` after the create-only marker commits the scene. A stale
    # lock after a hard kill is an explicit operator state.
    lock_uri = f"{base}/{result.scene_id}/.publish.lock"
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
                f"scene {result.scene_id} is being published by another run (lock {lock_uri})"
            ) from None
    else:
        import os

        lock_path = os.path.expanduser(lock_uri)
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"run_id": run_id, "started_at": completed_at}, indent=2))
        except FileExistsError:
            raise RuntimeError(
                f"scene {result.scene_id} is being published by another run (lock {lock_uri})"
            ) from None

    try:
        return _publish_locked(
            result=result,
            grid_100m=grid_100m,
            run_id=run_id,
            policy_hash=policy_hash,
            v3_config_hash=v3_config_hash,
            feature_valid_uri=feature_valid_uri,
            feature_provenance_uri=feature_provenance_uri,
            landsat_cog_uri=landsat_cog_uri,
            landsat_flag_uri=landsat_flag_uri,
            geometry_id=geometry_id,
            completed_at=completed_at,
            cog_uri=cog_uri,
            provenance_uri=provenance_uri,
            completion_uri=completion_uri,
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


def _publish_locked(
    *,
    result: EligibilityResult,
    grid_100m: GeoBox,
    run_id: str,
    policy_hash: str,
    v3_config_hash: str,
    feature_valid_uri: str,
    feature_provenance_uri: str,
    landsat_cog_uri: str,
    landsat_flag_uri: str,
    geometry_id: str,
    completed_at: str,
    cog_uri: str,
    provenance_uri: str,
    completion_uri: str,
) -> EligibilityArtifacts:
    """Write the three artifacts while holding the per-scene publish lock."""
    # ── 1. eligibility mask COG (uint8, canonical 100 m grid) ──────────
    mask_da = xr.DataArray(
        result.eligible.astype(np.uint8),
        dims=["y", "x"],
        coords={
            "x": grid_100m.transform.xoff + (np.arange(grid_100m.shape.x) + 0.5) * 100.0,
            "y": grid_100m.transform.yoff - (np.arange(grid_100m.shape.y) + 0.5) * 100.0,
        },
    )
    mask_da = mask_da.rio.write_crs(str(grid_100m.crs))
    mask_da = mask_da.rio.write_transform(grid_100m.transform)
    write_flag_cog_atomic(mask_da, cog_uri, _ELIGIBILITY_CONTRACT, overwrite=True)

    # ── 2. provenance ──────────────────────────────────────────────────
    # Chain the immutable source: the scene's feature-stack provenance
    # (config hash + channel order) is verified and recorded, so the
    # training artifact is provably derived from V3.
    source_prov = json.loads(read_bytes(feature_provenance_uri))
    if source_prov.get("config_hash") != v3_config_hash or list(
        source_prov.get("channel_order", [])
    ) != list(FEATURE_CHANNEL_NAMES):
        raise ValueError(
            f"scene {result.scene_id}: source feature provenance does not match "
            f"Feature Release V3 (config_hash {source_prov.get('config_hash')!r}, "
            f"channel order {len(source_prov.get('channel_order', []))})"
        )

    provenance = {
        "pipeline": "training-data",
        "scene_id": result.scene_id,
        "year": result.year,
        "s2_scene_id": result.s2_scene_id,
        "geometry_id": geometry_id,
        "policy_hash": policy_hash,
        "v3_config_hash": v3_config_hash,
        "run_id": run_id,
        "completed_at": completed_at,
        "grid": {
            "crs": str(grid_100m.crs),
            "shape": [grid_100m.shape.y, grid_100m.shape.x],
            "origin": [grid_100m.transform.xoff, grid_100m.transform.yoff],
        },
        "counts": {
            "target_valid_cells": result.target_valid_cells,
            "eligible_cells": result.eligible_cells,
        },
        "inputs": {
            "feature_valid": feature_valid_uri,
            "feature_provenance": feature_provenance_uri,
            "feature_config_hash": source_prov.get("config_hash"),
            "feature_channel_order": list(source_prov.get("channel_order", [])),
            "landsat_cog": landsat_cog_uri,
            "landsat_flag": landsat_flag_uri,
        },
        "mask_semantics": (
            "training_eligible == 1 iff Landsat target valid (flag==0 and LST in "
            "[150, 400] K) and all 100 corresponding 10m feature_valid subpixels "
            "are valid (strict 100/100 support; edge-truncated cells excluded)"
        ),
    }
    atomic_write(provenance_uri, json.dumps(provenance, indent=2), overwrite=True)

    # ── 2b. provenance read-back (immutability hardening) ──────────────
    # The create-only marker below atomically commits the scene. Verify
    # the provenance just written carries THIS run's policy hash, scene
    # identity, and the chained source config hash before committing: a
    # concurrent publisher that clobbered the artifact would produce a
    # mixed-identity immutable scene.
    written = json.loads(read_bytes(provenance_uri))
    written_inputs = written.get("inputs", {})
    if (
        written.get("policy_hash") != policy_hash
        or written.get("scene_id") != result.scene_id
        or written.get("v3_config_hash") != v3_config_hash
        or written_inputs.get("feature_config_hash") != v3_config_hash
    ):
        raise ValueError(
            f"scene {result.scene_id}: written provenance does not match this run "
            f"(policy_hash {written.get('policy_hash')!r}, "
            f"v3_config_hash {written.get('v3_config_hash')!r}, "
            f"feature_config_hash {written_inputs.get('feature_config_hash')!r}, "
            f"scene {written.get('scene_id')!r})"
        )

    # ── 3. completion marker (last, create-only) ───────────────────────
    atomic_write(
        completion_uri,
        json.dumps({"published_at": completed_at, "run_id": run_id}, indent=2),
        overwrite=False,
        if_generation_match=0,
    )

    return EligibilityArtifacts(
        cog_uri=cog_uri,
        provenance_uri=provenance_uri,
        completion_uri=completion_uri,
    )


__all__ = ["EligibilityArtifacts", "publish_eligibility"]
