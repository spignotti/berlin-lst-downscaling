#!/usr/bin/env python3
"""Targeted S2 snow/ice flag repair — one-off production maintenance.

Source: published S2 ARD flag COGs that predate ``FLAG_SNOW_ICE``.
Purpose: OR bit 5 (snow/ice) into the existing flag COG for exactly the
75 audited scenes with SCL=11, without touching reflectance COGs or the
main ARD pipeline. Grain: one report row per target S2 scene.

Modes
-----
- Dry-run (default): full preflight, candidate staging, GCS snapshots,
  and a local receipt. Zero GCS writes.
- Apply: ``--apply`` — lock, immutable backup, generation-guarded
  publication of flags/sidecars/ledger, receipts under the evidence
  prefix.
- Restore: ``--restore <repair-id> --apply`` — restore every backed-up
  object byte-identically from the repair evidence prefix (ledger
  first, then each scene's flag/provenance/STAC/completion as a set).

Usage
-----
    # Dry-run (default)
    uv run python scripts/repair_s2_snow_ice_flags.py --config configs/repair/s2_snow_ice.yaml

    # Apply (after explicit manual approval of the dry-run receipt)
    uv run python scripts/repair_s2_snow_ice_flags.py \\
        --config configs/repair/s2_snow_ice.yaml --apply

    # Restore a repair from its evidence prefix
    uv run python scripts/repair_s2_snow_ice_flags.py --config configs/repair/s2_snow_ice.yaml \
        --restore <repair-id> --apply
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr
import yaml

_logger = logging.getLogger(__name__)

SNOW_BIT = 1 << 5  # Contract.FLAG_SNOW_ICE
_CANONICAL_CRS = "EPSG:25833"

# ── CLI / config ──────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted S2 snow/ice flag repair")
    parser.add_argument(
        "--config",
        default="configs/repair/s2_snow_ice.yaml",
        help="Repair configuration YAML",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write production GCS objects (default: read-only dry-run)",
    )
    parser.add_argument(
        "--restore",
        metavar="REPAIR_ID",
        default=None,
        help="Restore a previously applied repair from its evidence prefix",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    required = [
        "manifest_uri",
        "ard_ledger_uri",
        "ard_root",
        "aoi_mask_uri",
        "audit_root",
        "audit_run_id",
        "evidence_prefix",
        "source",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Config {path!r} missing required keys: {missing}")
    return cfg


# ── GCS helpers ───────────────────────────────────────────────────────


def _gcs_client():
    from google.cloud import storage

    return storage.Client()


def _parse_gs(uri: str) -> tuple[str, str]:
    path = uri.removeprefix("gs://")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return parts[0], parts[1]


def _blob(client, uri: str):
    bucket, key = _parse_gs(uri)
    return client.bucket(bucket).blob(key)


def snapshot_object(client, uri: str) -> dict[str, Any] | None:
    """Capture generation/metageneration/crc32c/size for one GCS object."""
    blob = _blob(client, uri)
    if not blob.exists():
        return None
    blob.reload()
    return {
        "uri": uri,
        "generation": blob.generation,
        "metageneration": blob.metageneration,
        "crc32c": blob.crc32c,
        "size": blob.size,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_backup(
    client,
    src_uri: str,
    src_snapshot: dict[str, Any],
    dst_uri: str,
) -> None:
    """Server-side copy of one exact source generation into the backup."""
    bucket, key = _parse_gs(src_uri)
    dst_bucket, dst_key = _parse_gs(dst_uri)
    src = client.bucket(bucket).blob(key)
    client.bucket(bucket).copy_blob(
        src,
        client.bucket(dst_bucket),
        new_name=dst_key,
        source_generation=src_snapshot["generation"],
        if_generation_match=0,
    )


def upload_bytes_generation(
    client,
    data: bytes,
    dst_uri: str,
    if_generation_match: int | None,
) -> None:
    """Upload bytes to GCS with a destination generation precondition."""
    blob = _blob(client, dst_uri)
    blob.upload_from_string(
        data,
        if_generation_match=if_generation_match,
        checksum="auto",
    )


def upload_file_generation(
    client,
    local_path: str | Path,
    dst_uri: str,
    if_generation_match: int | None,
) -> None:
    blob = _blob(client, dst_uri)
    blob.upload_from_filename(
        str(local_path),
        if_generation_match=if_generation_match,
        checksum="auto",
    )


def delete_generation(client, uri: str, if_generation_match: int) -> None:
    blob = _blob(client, uri)
    blob.delete(if_generation_match=if_generation_match)


def read_gcs_bytes(client, uri: str) -> bytes:
    return _blob(client, uri).download_as_bytes()


# ── candidate flag logic (pure) ───────────────────────────────────────


def build_candidate_flag(old_flag: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """OR snow/ice bit into existing flags exactly where source SCL==11."""
    candidate = old_flag.copy()
    candidate[scl == 11] |= SNOW_BIT
    return candidate


def assert_candidate(old_flag: np.ndarray, candidate: np.ndarray, scl: np.ndarray) -> list[str]:
    """Return a list of invariant violations (empty == valid candidate)."""
    errors: list[str] = []
    if not np.array_equal(candidate & ~SNOW_BIT, old_flag & ~SNOW_BIT):
        errors.append("non-snow bits changed by candidate")
    outside = scl != 11
    if np.any((candidate[outside] & SNOW_BIT) != (old_flag[outside] & SNOW_BIT)):
        errors.append("snow bit changed on pixels without SCL=11")
    on_snow = candidate[scl == 11]
    if on_snow.size and not np.all((on_snow & SNOW_BIT) != 0):
        errors.append("snow bit missing on some SCL=11 pixel")
    return errors


# ── scene data model ──────────────────────────────────────────────────


@dataclass
class SceneCandidate:
    """Staged, validated repair artifacts for one S2 scene."""

    scene_id: str
    year: int
    flag_uri: str
    cog_uri: str
    stac_uri: str
    prov_uri: str
    comp_uri: str
    old_flag_sha: str
    candidate_flag_sha: str
    local_flag_path: str
    scl11_px: int
    unflagged_px: int
    added_snow_px: int
    stac_payload: dict[str, Any]
    prov_payload: dict[str, Any]
    comp_payload: dict[str, Any]
    aoi: dict[str, Any]
    scl_mask_sha: str
    item_json_sha: str


@dataclass
class RepairPlan:
    """Full staged plan for the 75 target scenes."""

    targets: list[SceneCandidate]
    ledger_local_path: str
    ledger_payload: bytes
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)


# ── loading published state ───────────────────────────────────────────


def _git_head() -> str:
    """Return the git HEAD supplied by the builder, if any.

    The builder sets ``GIT_HEAD`` (verified against the pinned commits)
    before invoking ``--apply``; this keeps the repair tool free of
    subprocess and lets the caller control provenance.
    """
    return os.environ.get("GIT_HEAD", "")


def _load_audit(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from berlin_lst_downscaling.data.io.storage import read_bytes
    from berlin_lst_downscaling.data.qa.s2_snow_ice import (
        audit_summary_path,
        scene_audit_parquet_path,
    )

    root = str(cfg["audit_root"])
    summary = json.loads(read_bytes(audit_summary_path(root)))
    if summary.get("run_id") != cfg.get("audit_run_id"):
        raise RuntimeError(
            f"audit run mismatch: expected {cfg.get('audit_run_id')!r}, "
            f"got {summary.get('run_id')!r}"
        )
    if summary.get("scenes_failed", 0) != 0:
        raise RuntimeError(f"audit has failed scenes: {summary.get('scenes_failed')}")
    table = pq.read_table(io.BytesIO(read_bytes(scene_audit_parquet_path(root))))
    rows = table.to_pylist()
    if summary.get("scenes_total") != cfg.get("expected_scenes_total"):
        raise RuntimeError("audit scenes_total != expected")
    return summary, rows


def _select_targets(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = [r for r in rows if r.get("ok") and (r.get("scl11_px") or 0) > 0]
    if len(targets) != cfg.get("expected_scenes_with_scl11"):
        raise RuntimeError(
            f"target count {len(targets)} != expected "
            f"{cfg.get('expected_scenes_with_scl11')}"
        )
    if len({t["scene_id"] for t in targets}) != len(targets):
        raise RuntimeError("duplicate scene ids in audit rows")
    return targets


def _grid_check(src: rasterio.io.DatasetReader, gbox) -> list[str]:
    errors: list[str] = []
    if str(src.crs).upper() != _CANONICAL_CRS:
        errors.append(f"flag CRS {src.crs} != {_CANONICAL_CRS}")
    if (src.height, src.width) != (gbox.shape.y, gbox.shape.x):
        errors.append(f"flag shape ({src.height},{src.width}) != canonical")
    for got, exp in zip(src.transform, gbox.transform, strict=True):
        if abs(got - exp) > 0.01:
            errors.append(f"flag transform coefficient mismatch: {got} != {exp}")
    return errors


def _load_aoi_mask(aoi_mask_uri: str, gbox) -> np.ndarray:
    import rasterio.warp as rwarp

    with rasterio.open(aoi_mask_uri) as src:
        aoi_data = src.read(1)
        destination = np.empty((gbox.shape.y, gbox.shape.x), dtype=aoi_data.dtype)
        aoi_data, _ = rwarp.reproject(
            source=aoi_data,
            src_transform=src.transform,
            src_width=src.width,
            src_height=src.height,
            src_crs=src.crs,
            destination=destination,
            dst_crs=_CANONICAL_CRS,
            dst_transform=gbox.transform,
            resampling=rwarp.Resampling.nearest,
        )
    return aoi_data == 1


# ── per-scene candidate construction ──────────────────────────────────


def build_scene_candidate(
    cfg: dict[str, Any],
    audit_row: dict[str, Any],
    manifest_row: dict[str, Any],
    ledger_row: dict[str, Any],
    gbox,
    aoi_mask: np.ndarray,
    contract,
    tmpdir: Path,
    run_id: str,
    git_head: str,
) -> SceneCandidate:
    """Stage and validate the full repair candidate for one scene."""
    from berlin_lst_downscaling.data.acquisition.pc_client import resolve_item_from_href
    from berlin_lst_downscaling.data.acquisition.sentinel2 import load_s2_scene
    from berlin_lst_downscaling.data.ard.aoi import compute_aoi_metrics
    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    from berlin_lst_downscaling.data.ard.paths import (
        cog_path,
        completion_path,
        provenance_path,
        stac_path,
    )
    from berlin_lst_downscaling.data.ard.product import (
        build_ard_provenance,
        build_ard_stac_item,
    )
    from berlin_lst_downscaling.data.ard.validate import validate_flag_cog
    from berlin_lst_downscaling.data.ard.writer import write_flag_cog_atomic
    from berlin_lst_downscaling.data.io.storage import read_bytes
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    scene_id = str(audit_row["scene_id"])
    year = int(ledger_row["year"])

    # ── exact source SCL from the manifest item ──────────────────────
    item_href = str(manifest_row["item_href"])
    if not item_href:
        raise RuntimeError(f"{scene_id}: manifest item_href missing")
    item = resolve_item_from_href(item_href, expected_id=scene_id)
    item_json_sha = sha256_bytes(json.dumps(item.to_dict(), sort_keys=True).encode("utf-8"))
    ds, _ = load_s2_scene(items=[item], bands=["SCL"], resolution=10)
    scl = np.round(ds["SCL"].values.squeeze()).astype(np.uint8)

    # ── published flag COG ───────────────────────────────────────────
    flag_uri = str(ledger_row["path_flag"])
    if not flag_uri:
        raise RuntimeError(f"{scene_id}: ledger has no flag path")
    with rasterio.open(gdal_uri(flag_uri)) as src:
        grid_errors = _grid_check(src, gbox)
        if grid_errors:
            raise RuntimeError(f"{scene_id}: flag grid mismatch: {grid_errors}")
        old_flag = src.read(1)
    if old_flag.shape != scl.shape:
        raise RuntimeError(f"{scene_id}: flag shape {old_flag.shape} != SCL {scl.shape}")

    # ── baseline counts must match the audit row ─────────────────────
    inside = aoi_mask
    scl11 = scl == 11
    scl11_px = int(np.sum(scl11 & inside))
    unflagged_px = int(np.sum(scl11 & (old_flag == 0) & inside))
    if scl11_px != int(audit_row["scl11_px"]):
        raise RuntimeError(
            f"{scene_id}: SCL11 count drift {scl11_px} != audit {audit_row['scl11_px']}"
        )
    if unflagged_px != int(audit_row["scl11_unflagged_px"]):
        raise RuntimeError(
            f"{scene_id}: unflagged drift {unflagged_px} != audit "
            f"{audit_row['scl11_unflagged_px']}"
        )

    # ── candidate flag ───────────────────────────────────────────────
    candidate = build_candidate_flag(old_flag, scl)
    violations = assert_candidate(old_flag, candidate, scl)
    if violations:
        raise RuntimeError(f"{scene_id}: candidate invariants violated: {violations}")
    added_snow_px = int(np.sum((candidate & SNOW_BIT) != 0) - np.sum((old_flag & SNOW_BIT) != 0))
    if added_snow_px < unflagged_px:
        raise RuntimeError(
            f"{scene_id}: added snow bits {added_snow_px} < unflagged {unflagged_px}"
        )

    # ── stage local flag COG + validate ──────────────────────────────
    flag_da = xr.DataArray(candidate, dims=("y", "x"))
    flag_da.rio.write_crs(_CANONICAL_CRS, inplace=True)
    flag_da.rio.write_transform(gbox.transform, inplace=True)
    local_flag = tmpdir / f"{scene_id}.flag.tif"
    write_flag_cog_atomic(flag_da, str(local_flag), contract, overwrite=True)
    strict = validate_strict_cog(str(local_flag))
    if not strict.valid:
        raise RuntimeError(f"{scene_id}: strict COG failed: {strict.errors + strict.warnings}")
    vflag = validate_flag_cog(str(local_flag), gbox)
    if not vflag.ok:
        raise RuntimeError(f"{scene_id}: flag validation failed: {vflag.errors}")

    # ── AOI metrics from the candidate ───────────────────────────────
    aoi = compute_aoi_metrics(str(local_flag), str(cfg["aoi_mask_uri"]), contract)
    for key, expect in (
        ("aoi_cloudy_px", ledger_row.get("aoi_cloudy_px")),
        ("aoi_shadow_px", ledger_row.get("aoi_shadow_px")),
        ("aoi_cirrus_px", ledger_row.get("aoi_cirrus_px")),
        ("aoi_saturated_px", ledger_row.get("aoi_saturated_px")),
        ("aoi_fill_px", ledger_row.get("aoi_fill_px")),
        ("aoi_total_px", ledger_row.get("aoi_total_px")),
    ):
        if expect is not None and int(aoi[key]) != int(expect):
            raise RuntimeError(
                f"{scene_id}: AOI {key} changed {aoi[key]} != ledger {expect}"
            )
    expected_clear = None
    if ledger_row.get("aoi_clear_px") is not None:
        expected_clear = int(ledger_row["aoi_clear_px"]) - unflagged_px
        if int(aoi["aoi_clear_px"]) != expected_clear:
            raise RuntimeError(
                f"{scene_id}: AOI clear {aoi['aoi_clear_px']} != expected {expected_clear}"
            )

    # ── sidecar candidates (no GCS writes) ───────────────────────────
    root = str(cfg["ard_root"])
    prov_uri = provenance_path(root, str(cfg["source"]), year, scene_id)
    existing_prov = json.loads(read_bytes(prov_uri))
    source_metadata = existing_prov.get("source_metadata")
    prov_payload = build_ard_provenance(
        scene_id,
        str(cfg["source"]),
        year,
        contract,
        run_id,
        repair=True,
        repair_commit=git_head or "unknown",
        source_metadata=source_metadata,
    )
    stac_payload = build_ard_stac_item(
        scene_id,
        str(cfg["source"]),
        year,
        ds,
        contract,
        cog_href=f"{scene_id}.tif",
        target_resolution=10,
        flag_href=f"{scene_id}.flag.tif",
        provenance_href="provenance.json",
    )
    comp_payload = {
        "published_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "repair": True,
    }

    return SceneCandidate(
        scene_id=scene_id,
        year=year,
        flag_uri=flag_uri,
        cog_uri=cog_path(root, str(cfg["source"]), year, scene_id),
        stac_uri=stac_path(root, str(cfg["source"]), year, scene_id),
        prov_uri=prov_uri,
        comp_uri=completion_path(root, str(cfg["source"]), year, scene_id),
        old_flag_sha=sha256_bytes(old_flag.tobytes()),
        candidate_flag_sha=sha256_file(local_flag),
        local_flag_path=str(local_flag),
        scl11_px=scl11_px,
        unflagged_px=unflagged_px,
        added_snow_px=added_snow_px,
        stac_payload=stac_payload,
        prov_payload=prov_payload,
        comp_payload=comp_payload,
        aoi=aoi,
        scl_mask_sha=sha256_bytes((scl == 11).tobytes()),
        item_json_sha=item_json_sha,
    )


# ── ledger staging ────────────────────────────────────────────────────


def stage_ledger(
    cfg: dict[str, Any],
    candidates: list[SceneCandidate],
    tmpdir: Path,
    run_id: str,
    contract,
) -> tuple[str, bytes]:
    """Stage the full repaired ledger; verify the allowlisted diff."""
    from berlin_lst_downscaling.data.io.storage import read_bytes

    raw = read_bytes(str(cfg["ard_ledger_uri"]))
    original = pq.read_table(io.BytesIO(raw))
    rows = original.to_pylist()
    key_index = {(r["scene_id"], r["source"]): i for i, r in enumerate(rows)}

    for cand in candidates:
        key = (cand.scene_id, str(cfg["source"]))
        if key not in key_index:
            raise RuntimeError(f"{cand.scene_id}: not in ledger")
        idx = key_index[key]
        old = rows[idx]
        new = dict(old)
        new["schema_hash"] = contract.schema_version_str()
        new["schema_version"] = contract.schema_version
        new["run_id"] = run_id
        new["updated_at"] = datetime.now(UTC)
        new["aoi_clear_px"] = int(cand.aoi["aoi_clear_px"])
        new["aoi_clear_frac"] = float(cand.aoi["aoi_clear_frac"])
        rows[idx] = new

        allowed = {
            "schema_hash",
            "schema_version",
            "run_id",
            "updated_at",
            "aoi_clear_px",
            "aoi_clear_frac",
        }
        for key_name, value in new.items():
            if key_name in allowed:
                continue
            if value != old[key_name]:
                raise RuntimeError(
                    f"{cand.scene_id}: unexpected ledger change in {key_name!r}"
                )

    table = pa.Table.from_pylist(rows, schema=original.schema)
    local = tmpdir / "ledger.parquet"
    pq.write_table(table, local)
    return str(local), raw


# ── snapshots ─────────────────────────────────────────────────────────


def collect_snapshots(
    cfg: dict[str, Any],
    candidates: list[SceneCandidate],
    client,
) -> dict[str, dict[str, Any]]:
    """Capture GCS metadata for every object the repair may touch."""
    uris: list[str] = []
    for cand in candidates:
        uris += [cand.flag_uri, cand.cog_uri, cand.stac_uri, cand.prov_uri, cand.comp_uri]
    uris += [
        str(cfg["ard_ledger_uri"]),
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.parquet",
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.csv",
        f"{str(cfg['audit_root']).rstrip('/')}/summary.json",
        str(cfg["manifest_uri"]),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "pairings.parquet"),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "manifest_report.json"),
    ]
    snapshots: dict[str, dict[str, Any]] = {}
    for uri in uris:
        snap = snapshot_object(client, uri)
        if snap is None:
            raise RuntimeError(f"GCS object missing for snapshot: {uri}")
        snapshots[uri] = snap
    return snapshots


# ── receipts ──────────────────────────────────────────────────────────


def build_receipt(
    cfg: dict[str, Any],
    candidates: list[SceneCandidate],
    snapshots: dict[str, dict[str, Any]],
    run_id: str,
    git_head: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "run_id": run_id,
        "git_head": git_head,
        "timestamp": datetime.now(UTC).isoformat(),
        "baseline": {
            "audit_run_id": cfg.get("audit_run_id"),
            "scenes_total": cfg.get("expected_scenes_total"),
            "scenes_with_scl11": cfg.get("expected_scenes_with_scl11"),
            "scenes_with_unflagged": cfg.get("expected_scenes_with_unflagged"),
            "scl11_px": cfg.get("expected_scl11_px"),
            "scl11_unflagged_px": cfg.get("expected_scl11_unflagged_px"),
        },
        "target_scenes": [
            {
                "scene_id": c.scene_id,
                "year": c.year,
                "scl11_px": c.scl11_px,
                "unflagged_px": c.unflagged_px,
                "added_snow_px": c.added_snow_px,
                "old_flag_sha": c.old_flag_sha,
                "candidate_flag_sha": c.candidate_flag_sha,
                "scl_mask_sha": c.scl_mask_sha,
                "item_json_sha": c.item_json_sha,
                "flag_uri": c.flag_uri,
                "cog_uri": c.cog_uri,
            }
            for c in candidates
        ],
        "reflectance_snapshots": {
            c.cog_uri: snapshots[c.cog_uri]
            for c in candidates
        },
        "flag_snapshots": {c.flag_uri: snapshots[c.flag_uri] for c in candidates},
        "sidecar_snapshots": {
            uri: snapshots[uri]
            for uri in snapshots
            if uri not in {c.cog_uri for c in candidates}
            and uri != str(cfg["ard_ledger_uri"])
        },
        "ledger_snapshot": snapshots[str(cfg["ard_ledger_uri"])],
        "allowed_write_prefixes": [
            str(cfg["ard_root"]),
            str(cfg["evidence_prefix"]),
        ],
    }


# ── dry-run ───────────────────────────────────────────────────────────


def run_dry_run(cfg: dict[str, Any]) -> int:
    """Full preflight + local staging. Never writes to GCS."""
    # git ancestor verification is a builder pre-apply step
    _, rows = _load_audit(cfg)
    targets = _select_targets(cfg, rows)

    from berlin_lst_downscaling.common.grid import canon_grid_10m
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.ledger import Ledger
    from berlin_lst_downscaling.data.selection.validate import load_bundle

    bundle, bundle_result = load_bundle(str(cfg["manifest_uri"]), require_item_href=True)
    if not bundle_result.ok:
        raise RuntimeError(f"manifest bundle invalid: {bundle_result.errors}")
    manifest_lookup = {
        row["scene_id"]: row
        for row in bundle.manifest_table.to_pylist()
        if row["source"] == str(cfg["source"])
    }
    ledger = Ledger.open(str(cfg["ard_ledger_uri"]))
    contract = contract_for_source(str(cfg["source"]))
    gbox = canon_grid_10m()
    aoi_mask = _load_aoi_mask(str(cfg["aoi_mask_uri"]), gbox)
    git_head = _git_head()
    run_id = uuid4().hex[:8]

    with tempfile.TemporaryDirectory(prefix="repair-s2-staging-") as tmp:
        tmpdir = Path(tmp)
        candidates: list[SceneCandidate] = []
        for i, audit_row in enumerate(targets, 1):
            scene_id = str(audit_row["scene_id"])
            manifest_row = manifest_lookup.get(scene_id)
            if manifest_row is None:
                raise RuntimeError(f"{scene_id}: not in manifest {cfg['source']} rows")
            led = ledger.get(scene_id, str(cfg["source"]))
            if led is None or led.status != "done":
                raise RuntimeError(f"{scene_id}: ledger row not done")
            print(f"  [{i}/{len(targets)}] staging {scene_id}")
            candidates.append(
                build_scene_candidate(
                    cfg, audit_row, manifest_row, led.__dict__, gbox, aoi_mask,
                    contract, tmpdir, run_id, git_head,
                )
            )

        _, ledger_raw = stage_ledger(cfg, candidates, tmpdir, run_id, contract)
        client = _gcs_client()
        snapshots = collect_snapshots(cfg, candidates, client)
        receipt = build_receipt(cfg, candidates, snapshots, run_id, git_head, "dry-run")

        receipt_dir = Path(tempfile.gettempdir()) / "opencode"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"repair-s2-snow-ice-{run_id}.receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2))
        ledger_path = receipt_dir / f"repair-s2-snow-ice-{run_id}.ledger.parquet"
        ledger_path.write_bytes(ledger_raw)

    total_unflagged = sum(c.unflagged_px for c in candidates)
    print(f"\nDry-run complete — repair {run_id}")
    print(f"  targets: {len(candidates)} (expected {cfg['expected_scenes_with_scl11']})")
    print(f"  unflagged SCL11 pixels to fix: {total_unflagged} "
          f"(expected {cfg['expected_scl11_unflagged_px']})")
    print(f"  reflectance COGs untouched: {len(candidates)}")
    print(f"  receipt: {receipt_path}")
    print(f"  ledger byte-copy: {ledger_path}")
    print("  zero GCS writes performed")

    if total_unflagged != int(cfg["expected_scl11_unflagged_px"]):
        raise RuntimeError(
            f"total unflagged {total_unflagged} != expected "
            f"{cfg['expected_scl11_unflagged_px']}"
        )
    return 0


# ── apply ─────────────────────────────────────────────────────────────


def _evidence_root(cfg: dict[str, Any], repair_id: str) -> str:
    return f"{str(cfg['evidence_prefix']).rstrip('/')}/{repair_id}"


def run_apply(cfg: dict[str, Any]) -> int:
    """Guarded production repair: lock → backup → publish → ledger → summary."""
    from berlin_lst_downscaling.common.grid import canon_grid_10m
    from berlin_lst_downscaling.data.ard.contract import contract_for_source
    from berlin_lst_downscaling.data.ard.ledger import Ledger
    from berlin_lst_downscaling.data.io.run_logging import RunLogSession, log_event
    from berlin_lst_downscaling.data.io.storage import atomic_write
    from berlin_lst_downscaling.data.selection.validate import load_bundle

    # git ancestor verification is a builder pre-apply step
    _, rows = _load_audit(cfg)
    targets = _select_targets(cfg, rows)

    bundle, bundle_result = load_bundle(str(cfg["manifest_uri"]), require_item_href=True)
    if not bundle_result.ok:
        raise RuntimeError(f"manifest bundle invalid: {bundle_result.errors}")
    manifest_lookup = {
        row["scene_id"]: row
        for row in bundle.manifest_table.to_pylist()
        if row["source"] == str(cfg["source"])
    }
    ledger = Ledger.open(str(cfg["ard_ledger_uri"]))
    contract = contract_for_source(str(cfg["source"]))
    gbox = canon_grid_10m()
    aoi_mask = _load_aoi_mask(str(cfg["aoi_mask_uri"]), gbox)
    git_head = _git_head()
    if not git_head:
        raise RuntimeError(
            "GIT_HEAD env is required for --apply — the builder sets it after "
            "verifying the pinned ancestor commits (configs/repair/s2_snow_ice.yaml)"
        )
    client = _gcs_client()
    repair_id = f"s2-snow-ice-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    run_id = uuid4().hex[:8]
    evidence = _evidence_root(cfg, repair_id)
    backup_root = f"{evidence}/backup"
    log_root = evidence

    with RunLogSession(log_root, pipeline="qa-repair-s2-snow-ice", run_id=run_id):
        log_event(_logger, logging.INFO, "repair_started", repair_id=repair_id, run_id=run_id)

        # ── lock ─────────────────────────────────────────────────────
        lock_uri = f"{evidence}/LOCK.json"
        lock_payload = json.dumps(
            {
                "repair_id": repair_id,
                "run_id": run_id,
                "started": datetime.now(UTC).isoformat(),
                "git_head": git_head,
                "pid": os.getpid(),
            },
            indent=2,
        )
        try:
            atomic_write(lock_uri, lock_payload, overwrite=True, if_generation_match=0)
        except Exception as exc:
            raise RuntimeError(f"cannot acquire repair lock {lock_uri}: {exc}") from exc
        log_event(_logger, logging.INFO, "repair_lock_acquired", lock_uri=lock_uri)

        try:
            with tempfile.TemporaryDirectory(prefix="repair-s2-apply-") as tmp:
                tmpdir = Path(tmp)
                candidates: list[SceneCandidate] = []
                for audit_row in targets:
                    scene_id = str(audit_row["scene_id"])
                    manifest_row = manifest_lookup.get(scene_id)
                    if manifest_row is None:
                        raise RuntimeError(f"{scene_id}: not in manifest")
                    led = ledger.get(scene_id, str(cfg["source"]))
                    if led is None or led.status != "done":
                        raise RuntimeError(f"{scene_id}: ledger row not done")
                    candidates.append(
                        build_scene_candidate(
                            cfg, audit_row, manifest_row, led.__dict__, gbox, aoi_mask,
                            contract, tmpdir, run_id, git_head,
                        )
                    )

                snapshots = collect_snapshots(cfg, candidates, client)

                # ── immutable backup ─────────────────────────────────
                backup_paths: dict[str, str] = {}
                for uri, snap in snapshots.items():
                    rel = uri.removeprefix("gs://berlin-lst-data/")
                    dst = f"{backup_root}/{rel}"
                    copy_backup(client, uri, snap, dst)
                    backup_paths[uri] = dst
                for uri, dst in backup_paths.items():
                    verify = snapshot_object(client, dst)
                    if (
                        verify is None
                        or verify["crc32c"] != snapshots[uri]["crc32c"]
                        or verify["size"] != snapshots[uri]["size"]
                    ):
                        raise RuntimeError(f"backup verification failed for {dst}")
                log_event(
                    _logger,
                    logging.INFO,
                    "backup_complete",
                    objects=len(backup_paths),
                    backup_root=backup_root,
                )

                # ── publish scenes sequentially ──────────────────────
                for cand in candidates:
                    _publish_scene(client, cand, snapshots, evidence)
                    log_event(
                        _logger,
                        logging.INFO,
                        "scene_repaired",
                        scene_id=cand.scene_id,
                        added_snow_px=cand.added_snow_px,
                    )

                # ── publish ledger (single CAS write) ────────────────
                _, ledger_raw = stage_ledger(cfg, candidates, tmpdir, run_id, contract)
                upload_bytes_generation(
                    client,
                    ledger_raw,
                    str(cfg["ard_ledger_uri"]),
                    if_generation_match=snapshots[str(cfg["ard_ledger_uri"])]["generation"],
                )
                reopened = Ledger.open(str(cfg["ard_ledger_uri"]))
                reopened_count = sum(
                    1
                    for c in candidates
                    if reopened.get(c.scene_id, str(cfg["source"])) is not None
                )
                if reopened_count != len(candidates):
                    raise RuntimeError("ledger reload missing target rows")
                log_event(_logger, logging.INFO, "ledger_published", rows=len(candidates))

                # ── receipts + summary ───────────────────────────────
                receipt = build_receipt(cfg, candidates, snapshots, run_id, git_head, "apply")
                receipt["repair_id"] = repair_id
                receipt["backup_root"] = backup_root
                receipt["lock_uri"] = lock_uri
                atomic_write(
                    f"{evidence}/receipt.json",
                    json.dumps(receipt, indent=2),
                    overwrite=True,
                    if_generation_match=0,
                )
                summary_payload = {
                    "repair_id": repair_id,
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "git_head": git_head,
                    "evidence_root": evidence,
                    "backup_root": backup_root,
                    "baseline": {
                        "audit_run_id": cfg.get("audit_run_id"),
                        "scenes": len(candidates),
                        "unflagged_px": sum(c.unflagged_px for c in candidates),
                    },
                    "per_scene": [
                        {
                            "scene_id": c.scene_id,
                            "old_flag_sha": c.old_flag_sha,
                            "candidate_flag_sha": c.candidate_flag_sha,
                            "added_snow_px": c.added_snow_px,
                            "flag_uri": c.flag_uri,
                        }
                        for c in candidates
                    ],
                    "reflectance_unchanged": {
                        c.cog_uri: snapshots[c.cog_uri] for c in candidates
                    },
                }
                atomic_write(
                    f"{evidence}/summary.json",
                    json.dumps(summary_payload, indent=2),
                    overwrite=True,
                    if_generation_match=0,
                )

                # ── release lock ─────────────────────────────────────
                lock_snap = snapshot_object(client, lock_uri)
                if lock_snap is not None:
                    delete_generation(
                        client,
                        lock_uri,
                        if_generation_match=lock_snap["generation"],
                    )

        except BaseException:
            log_event(
                _logger,
                logging.ERROR,
                "repair_failed",
                repair_id=repair_id,
                lock_uri=lock_uri,
                note="lock and backup retained; restore via --restore",
            )
            raise

    print(f"\nRepair applied — {repair_id}")
    print(f"  scenes: {len(candidates)}")
    print(f"  evidence: {evidence}")
    print("  next: post-audit to evidence post_audit/, validate_ard, reflectance recheck")
    return 0


def _publish_scene(client, cand: SceneCandidate, snapshots: dict, evidence: str) -> None:
    """Publish one scene's flag + sidecars with generation guards."""
    from berlin_lst_downscaling.data.ard.cog_layout import validate_strict_cog
    from berlin_lst_downscaling.data.io.storage import atomic_write
    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    snap_flag = snapshots[cand.flag_uri]
    snap_prov = snapshots[cand.prov_uri]
    snap_stac = snapshots[cand.stac_uri]
    snap_comp = snapshots[cand.comp_uri]

    # 1. remove completion marker first (exact generation)
    delete_generation(client, cand.comp_uri, if_generation_match=snap_comp["generation"])

    # 2. flag COG (exact generation precondition)
    upload_file_generation(
        client,
        cand.local_flag_path,
        cand.flag_uri,
        if_generation_match=snap_flag["generation"],
    )

    # 3. validate the published flag remotely. The staged candidate was
    #    already bit-asserted (assert_candidate) and strict-COG validated;
    #    byte-hash equality proves the remote flag is that candidate.
    remote_bytes = read_gcs_bytes(client, cand.flag_uri)
    if sha256_bytes(remote_bytes) != cand.candidate_flag_sha:
        raise RuntimeError(f"{cand.scene_id}: published flag hash mismatch")
    with rasterio.open(gdal_uri(cand.flag_uri)) as src:
        remote_flag = src.read(1)
        strict_ok = validate_strict_cog(gdal_uri(cand.flag_uri))
        if not strict_ok.valid:
            raise RuntimeError(
                f"{cand.scene_id}: remote strict COG failed: "
                f"{strict_ok.errors + strict_ok.warnings}"
            )
    # Byte-hash equality already proves remote == staged candidate (which
    # passed assert_candidate). Sanity-check the snow-bit count still.
    with rasterio.open(cand.local_flag_path) as src:
        staged_flag = src.read(1)
    if int(np.sum((remote_flag & SNOW_BIT) != 0)) != int(
        np.sum((staged_flag & SNOW_BIT) != 0)
    ):
        raise RuntimeError(f"{cand.scene_id}: remote snow bit count != staged")

    # 4. provenance + STAC (exact generation preconditions)
    atomic_write(
        cand.prov_uri,
        json.dumps(cand.prov_payload, indent=2),
        overwrite=True,
        if_generation_match=snap_prov["generation"],
    )
    atomic_write(
        cand.stac_uri,
        json.dumps(cand.stac_payload, indent=2),
        overwrite=True,
        if_generation_match=snap_stac["generation"],
    )

    # 5. completion last (absent precondition)
    atomic_write(
        cand.comp_uri,
        json.dumps(cand.comp_payload, indent=2),
        overwrite=True,
        if_generation_match=0,
    )


# ── restore ───────────────────────────────────────────────────────────


def run_restore(cfg: dict[str, Any], repair_id: str) -> int:
    """Restore every backed-up object byte-identically (explicit only)."""
    from berlin_lst_downscaling.data.io.storage import atomic_write

    evidence = _evidence_root(cfg, repair_id)
    summary_uri = f"{evidence}/summary.json"
    client = _gcs_client()
    summary = json.loads(read_gcs_bytes(client, summary_uri))
    backup_root = str(summary["backup_root"])
    receipt = json.loads(read_gcs_bytes(client, f"{evidence}/receipt.json"))

    restore_plan: list[tuple[str, str]] = []  # (original_uri, backup_uri)
    for uri in receipt["reflectance_snapshots"]:
        restore_plan.append((uri, f"{backup_root}/{uri.removeprefix('gs://berlin-lst-data/')}"))
    ledger_uri = str(cfg["ard_ledger_uri"])
    restore_plan.insert(
        0, (ledger_uri, f"{backup_root}/{ledger_uri.removeprefix('gs://berlin-lst-data/')}")
    )
    for cand in receipt["target_scenes"]:
        for suffix in (".flag.tif", ".stac.json", "provenance.json", "complete.json"):
            # reconstruct URIs from the receipt scene entries
            flag_uri = cand["flag_uri"]
            scene_dir = "/".join(flag_uri.split("/")[:-1])
            name = flag_uri.split("/")[-1].replace(".flag.tif", "")
            orig = {
                ".flag.tif": flag_uri,
                ".stac.json": f"{scene_dir}/{name}.stac.json",
                "provenance.json": f"{scene_dir}/provenance.json",
                "complete.json": f"{scene_dir}/complete.json",
            }[suffix]
            restore_plan.append(
                (orig, f"{backup_root}/{orig.removeprefix('gs://berlin-lst-data/')}")
            )

    # ledger first, then each scene's objects as a set
    print(f"Restoring repair {repair_id} from {backup_root}")
    for original, backup in restore_plan:
        data = read_gcs_bytes(client, backup)
        if original.endswith("complete.json"):
            snap = snapshot_object(client, original)
            if snap is not None:
                delete_generation(client, original, if_generation_match=snap["generation"])
            atomic_write(original, data, overwrite=True, if_generation_match=0)
        else:
            atomic_write(original, data, overwrite=True)
        restored = read_gcs_bytes(client, original)
        if restored != data:
            raise RuntimeError(f"restore verification failed for {original}")
        print(f"  restored {original}")

    # release lock if present
    lock_uri = f"{evidence}/LOCK.json"
    snap = snapshot_object(client, lock_uri)
    if snap is not None:
        delete_generation(client, lock_uri, if_generation_match=snap["generation"])
        print(f"  released {lock_uri}")

    print("Restore complete — verify with validate_ard and a fresh audit.")
    return 0


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.restore and args.apply:
        return run_restore(cfg, args.restore)
    if args.apply:
        return run_apply(cfg)
    return run_dry_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
