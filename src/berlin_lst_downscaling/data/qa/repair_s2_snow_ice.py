"""Mechanics for the one-off S2 snow/ice flag repair (WB2c-2, Stufe 1).

CLI + orchestration live in ``scripts/repair_s2_snow_ice_flags.py``; this
module provides the reusable, testable pieces: GCS helpers with retry,
the pure bit-OR candidate logic, per-scene candidate construction,
ledger staging, snapshot capture, receipts, and after-state verification.

The repair ORs ``FLAG_SNOW_ICE`` (bit 5) into the existing flag COG for
exactly the scenes the baseline audit flagged with SCL=11. Reflectance
COGs, the main ARD pipeline, and the ledger schema are untouched.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rioxarray  # noqa: F401 — registers the .rio accessor (repo-standard, masking.py:12)
import xarray as xr

_logger = logging.getLogger(__name__)

_CANONICAL_CRS = "EPSG:25833"


# ── GCS helpers (retry on transient errors) ───────────────────────────


def _retry(fn, *args, **kwargs):
    """Run *fn* with exponential backoff on transient GCS errors only.

    412 preconditions and 404s are not transient — they must fail fast
    (concurrent modification / missing object).
    """
    from google.api_core.exceptions import (  # type: ignore[import-untyped]
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    transient = (
        ConnectionError,
        TimeoutError,
        GatewayTimeout,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type(transient),
        reraise=True,
    )
    def _run():
        return fn(*args, **kwargs)

    return _run()


def snow_bit() -> int:
    """The snow/ice flag bit, checked against the canonical contract."""
    from berlin_lst_downscaling.data.ard.contract import Contract

    bit = Contract.FLAG_SNOW_ICE
    if bit != 1 << 5:
        raise RuntimeError(f"Contract.FLAG_SNOW_ICE unexpectedly changed to {bit}")
    return bit


def gcs_client():
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


def copy_backup(client, src_uri: str, src_snapshot: dict[str, Any], dst_uri: str) -> None:
    """Server-side copy of one exact source generation into the backup."""

    def _do():
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

    _retry(_do)


def upload_bytes_generation(
    client,
    data: bytes,
    dst_uri: str,
    if_generation_match: int | None,
) -> None:
    def _do():
        blob = _blob(client, dst_uri)
        blob.upload_from_string(
            data,
            if_generation_match=if_generation_match,
            checksum="auto",
        )

    _retry(_do)


def upload_file_generation(
    client,
    local_path: str | Path,
    dst_uri: str,
    if_generation_match: int | None,
) -> None:
    def _do():
        blob = _blob(client, dst_uri)
        blob.upload_from_filename(
            str(local_path),
            if_generation_match=if_generation_match,
            checksum="auto",
        )

    _retry(_do)


def delete_generation(client, uri: str, if_generation_match: int) -> None:
    def _do():
        blob = _blob(client, uri)
        blob.delete(if_generation_match=if_generation_match)

    _retry(_do)


def read_gcs_bytes(client, uri: str) -> bytes:
    return _blob(client, uri).download_as_bytes()


# ── candidate flag logic (pure) ───────────────────────────────────────


def build_candidate_flag(old_flag: np.ndarray, scl: np.ndarray, bit: int) -> np.ndarray:
    """OR the snow bit into existing flags exactly where source SCL==11."""
    candidate = old_flag.copy()
    candidate[scl == 11] |= bit
    return candidate


def assert_candidate(
    old_flag: np.ndarray, candidate: np.ndarray, scl: np.ndarray, bit: int
) -> list[str]:
    """Return invariant violations (empty == valid candidate)."""
    errors: list[str] = []
    if not np.array_equal(candidate & ~bit, old_flag & ~bit):
        errors.append("non-snow bits changed by candidate")
    outside = scl != 11
    if np.any((candidate[outside] & bit) != (old_flag[outside] & bit)):
        errors.append("snow bit changed on pixels without SCL=11")
    on_snow = candidate[scl == 11]
    if on_snow.size and not np.all((on_snow & bit) != 0):
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
    candidate_flag_sha: str
    stac_payload_sha: str
    prov_payload_sha: str
    comp_payload_sha: str
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


# ── loading published state ───────────────────────────────────────────


def git_head() -> str:
    """Return the git HEAD supplied by the builder, if any.

    The builder sets ``GIT_HEAD`` (verified against the pinned commits)
    before invoking ``--apply``; this keeps the repair tool free of
    subprocess and lets the caller control provenance.
    """
    return os.environ.get("GIT_HEAD", "")


def load_audit(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def select_targets(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = [r for r in rows if r.get("ok") and (r.get("scl11_px") or 0) > 0]
    if len(targets) != cfg.get("expected_scenes_with_scl11"):
        raise RuntimeError(
            f"target count {len(targets)} != expected "
            f"{cfg.get('expected_scenes_with_scl11')}"
        )
    if len({t["scene_id"] for t in targets}) != len(targets):
        raise RuntimeError("duplicate scene ids in audit rows")
    return targets


def grid_errors(src: Any, gbox) -> list[str]:
    errors: list[str] = []
    if str(src.crs).upper() != _CANONICAL_CRS:
        errors.append(f"CRS {src.crs} != {_CANONICAL_CRS}")
    if (src.height, src.width) != (gbox.shape.y, gbox.shape.x):
        errors.append(f"shape ({src.height},{src.width}) != canonical")
    for got, exp in zip(src.transform, gbox.transform, strict=True):
        if abs(got - exp) > 0.01:
            errors.append(f"transform coefficient mismatch: {got} != {exp}")
    return errors


def load_aoi_mask(aoi_mask_uri: str, gbox) -> np.ndarray:
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

    bit = snow_bit()
    scene_id = str(audit_row["scene_id"])
    year = int(ledger_row["year"])

    # ── exact source SCL from the manifest item ──────────────────────
    item_href = str(manifest_row["item_href"])
    if not item_href:
        raise RuntimeError(f"{scene_id}: manifest item_href missing")
    item = resolve_item_from_href(item_href, expected_id=scene_id)
    item_json_sha = sha256_bytes(json.dumps(item.to_dict(), sort_keys=True).encode("utf-8"))
    ds, _ = load_s2_scene(items=[item], bands=["SCL"], resolution=10)
    # SCL must land on the canonical grid — validate CRS and transform,
    # not just shape, so bit 5 is set on the right pixels.
    scl_crs_ok = str(ds.rio.crs).upper() == _CANONICAL_CRS
    scl_transform_ok = all(
        abs(g - e) < 0.01
        for g, e in zip(ds.rio.transform(), gbox.transform, strict=True)
    )
    if not (scl_crs_ok and scl_transform_ok):
        raise RuntimeError(
            f"{scene_id}: loaded SCL not on canonical grid "
            f"(crs_ok={scl_crs_ok}, transform_ok={scl_transform_ok})"
        )
    scl = np.round(ds["SCL"].values.squeeze()).astype(np.uint8)

    # ── published flag COG ───────────────────────────────────────────
    flag_uri = str(ledger_row["path_flag"])
    if not flag_uri:
        raise RuntimeError(f"{scene_id}: ledger has no flag path")
    with rasterio.open(gdal_uri(flag_uri)) as src:
        g_errors = grid_errors(src, gbox)
        if g_errors:
            raise RuntimeError(f"{scene_id}: flag grid mismatch: {g_errors}")
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
    candidate = build_candidate_flag(old_flag, scl, bit)
    violations = assert_candidate(old_flag, candidate, scl, bit)
    if violations:
        raise RuntimeError(f"{scene_id}: candidate invariants violated: {violations}")
    added_snow_px = int(np.sum((candidate & bit) != 0) - np.sum((old_flag & bit) != 0))
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
    ):
        if expect is not None and int(aoi[key]) != int(expect):
            raise RuntimeError(
                f"{scene_id}: AOI {key} changed {aoi[key]} != ledger {expect}"
            )
    # aoi_total_px is deliberately NOT compared to the ledger: the
    # multi-bit non-fill total fix (Package 1) recomputes it correctly
    # for scenes with overlapping flag bits; the repaired ledger row
    # must adopt the corrected value to stay consistent with the flag.
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
        candidate_flag_sha=sha256_file(local_flag),
        stac_payload_sha=sha256_bytes(json.dumps(stac_payload, indent=2).encode("utf-8")),
        prov_payload_sha=sha256_bytes(json.dumps(prov_payload, indent=2).encode("utf-8")),
        comp_payload_sha=sha256_bytes(json.dumps(comp_payload, indent=2).encode("utf-8")),
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
    """Stage the full repaired ledger; return path + MODIFIED bytes."""
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
        # aoi_total_px adopts the corrected multi-bit non-fill count so
        # the row stays consistent with the published flag (see
        # build_scene_candidate).
        new["aoi_total_px"] = int(cand.aoi["aoi_total_px"])
        rows[idx] = new

        allowed = {
            "schema_hash",
            "schema_version",
            "run_id",
            "updated_at",
            "aoi_clear_px",
            "aoi_clear_frac",
            "aoi_total_px",
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
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return str(local), buf.getvalue()


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
                "candidate_flag_sha": c.candidate_flag_sha,
                "stac_payload_sha": c.stac_payload_sha,
                "prov_payload_sha": c.prov_payload_sha,
                "comp_payload_sha": c.comp_payload_sha,
                "scl_mask_sha": c.scl_mask_sha,
                "item_json_sha": c.item_json_sha,
                "flag_uri": c.flag_uri,
                "cog_uri": c.cog_uri,
            }
            for c in candidates
        ],
        "reflectance_snapshots": {c.cog_uri: snapshots[c.cog_uri] for c in candidates},
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


# ── after-state verification ──────────────────────────────────────────


def verify_after_state(
    client,
    cfg: dict[str, Any],
    candidates: list[SceneCandidate],
    ledger_bytes: bytes,
    pre_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify every touched object post-publication and protected objects.

    Returns after-state snapshots. Raises on any mismatch: staged payload
    sha for flags/sidecars/completions/ledger; byte-identical
    generation/crc32c/size for reflectance, manifest, and baseline audit.
    """
    after: dict[str, Any] = {"touched": {}, "protected": {}}

    for cand in candidates:
        remote_flag = read_gcs_bytes(client, cand.flag_uri)
        if sha256_bytes(remote_flag) != cand.candidate_flag_sha:
            raise RuntimeError(f"{cand.scene_id}: after-state flag hash mismatch")
        for uri, staged_sha in (
            (cand.stac_uri, cand.stac_payload_sha),
            (cand.prov_uri, cand.prov_payload_sha),
            (cand.comp_uri, cand.comp_payload_sha),
        ):
            remote = read_gcs_bytes(client, uri)
            if sha256_bytes(remote) != staged_sha:
                raise RuntimeError(f"{cand.scene_id}: after-state hash mismatch for {uri}")
        after["touched"][cand.flag_uri] = snapshot_object(client, cand.flag_uri)
        after["touched"][cand.stac_uri] = snapshot_object(client, cand.stac_uri)
        after["touched"][cand.prov_uri] = snapshot_object(client, cand.prov_uri)
        after["touched"][cand.comp_uri] = snapshot_object(client, cand.comp_uri)

    ledger_remote = read_gcs_bytes(client, str(cfg["ard_ledger_uri"]))
    if sha256_bytes(ledger_remote) != sha256_bytes(ledger_bytes):
        raise RuntimeError("after-state ledger hash mismatch")
    after["touched"][str(cfg["ard_ledger_uri"])] = snapshot_object(
        client, str(cfg["ard_ledger_uri"])
    )

    protected_uris = [c.cog_uri for c in candidates] + [
        str(cfg["manifest_uri"]),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "pairings.parquet"),
        str(cfg["manifest_uri"]).replace("manifest.parquet", "manifest_report.json"),
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.parquet",
        f"{str(cfg['audit_root']).rstrip('/')}/scene_audit.csv",
        f"{str(cfg['audit_root']).rstrip('/')}/summary.json",
    ]
    for uri in protected_uris:
        now = snapshot_object(client, uri)
        before = pre_snapshots[uri]
        if (
            now is None
            or now["generation"] != before["generation"]
            or now["crc32c"] != before["crc32c"]
            or now["size"] != before["size"]
        ):
            raise RuntimeError(f"protected object changed during repair: {uri}")
        after["protected"][uri] = now

    return after


__all__ = [
    "SceneCandidate",
    "assert_candidate",
    "build_candidate_flag",
    "build_receipt",
    "build_scene_candidate",
    "collect_snapshots",
    "copy_backup",
    "delete_generation",
    "gcs_client",
    "git_head",
    "grid_errors",
    "load_aoi_mask",
    "load_audit",
    "read_gcs_bytes",
    "select_targets",
    "sha256_bytes",
    "snapshot_object",
    "snow_bit",
    "stage_ledger",
    "upload_bytes_generation",
    "upload_file_generation",
    "verify_after_state",
]
