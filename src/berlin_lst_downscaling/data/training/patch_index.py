"""WB3 patch index — 160x160 px windows over the published eligibility masks.

Enumerates non-overlapping ``16x16``-cell (``160x160`` px at 10 m) windows on
the **global canonical** EPSG:25833 100 m lattice and admits a window when at
least 95% of its 256 cells are ``training_eligible`` (244 of 256). The index is
derived only from the published ``training/v1`` scene manifest and eligibility
COGs: no feature pixels are recomputed and no patch rasters are written.

Anchors are multiples of the stride on the global lattice, so the same spatial
window carries the same anchor across every scene — the same convention as the
stable ``cell_id``. Windows that would extend past a mask edge are never
emitted (no truncated patches). Every patch inherits its scene's temporal
split, so the scene-before-patch invariant holds by construction.

Publication is separate from ``training/v1`` (which is immutable): the index is
written under its own versioned root with a create-only completion marker
written last, after a publisher-side readback.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
import rasterio

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.io import atomic_write, exists, publish_lock, read_bytes
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
    NO_ELIGIBLE_CELLS_REASON,
    cell_id,
    training_policy_hash,
)
from berlin_lst_downscaling.data.training.paths import (
    manifest_parquet,
    patch_index_completion,
    patch_index_parquet,
    patch_index_qa,
    release_completion,
)
from berlin_lst_downscaling.data.training.report import now_iso

# ── frozen patch geometry (WB3 decision, issue #14) ───────────────────

# Main patch: 160x160 px at 10 m = 16x16 cells at 100 m (1.6 km).
PATCH_CELLS: int = 16
PATCH_SIZE_10M: int = PATCH_CELLS * 10
# Non-overlapping v1 default (stride equals the patch size).
PATCH_STRIDE_CELLS: int = PATCH_CELLS
N_CELLS: int = PATCH_CELLS * PATCH_CELLS
# A window is valid when >= 95% of its 100 m cells are eligible: 243/256 =
# 0.94921 fails, so the integer minimum is 244 (ceil of 0.95 * 256).
MIN_ELIGIBLE_FRACTION: float = 0.95
MIN_ELIGIBLE_CELLS: int = math.ceil(MIN_ELIGIBLE_FRACTION * N_CELLS)

# An assessable scene with eligible cells but no window reaching the
# threshold is excluded from the index with this reason (distinct from the
# existing ``no_eligible_cells``).
NO_VALID_PATCH_REASON: str = "no_valid_patch"

PATCH_INDEX_SCHEMA_VERSION: int = 1

PATCH_INDEX_FIELDNAMES = [
    "scene_id",
    "year",
    "split",
    "s2_scene_id",
    "patch_id",
    "row",
    "col",
    "row_10m",
    "col_10m",
    "center_x",
    "center_y",
    "eligibility_mask",
    "n_eligible",
    "n_total",
    "eligible_frac",
]


def patch_index_policy_hash() -> str:
    """Return a stable SHA-256 fingerprint of the patch-index policy.

    Covers the frozen geometry (patch size, stride, anchor rule), the
    acceptance threshold, and the index schema version. It is deliberately
    independent of ``training_policy_hash``: the index is a derived WB3
    artifact, and a change here must not invalidate the immutable
    ``training/v1`` release.
    """
    payload = json.dumps(
        {
            "schema_version": PATCH_INDEX_SCHEMA_VERSION,
            "patch_cells": PATCH_CELLS,
            "patch_size_10m": PATCH_SIZE_10M,
            "stride_cells": PATCH_STRIDE_CELLS,
            "n_total": N_CELLS,
            "min_eligible_fraction": MIN_ELIGIBLE_FRACTION,
            "min_eligible_cells": MIN_ELIGIBLE_CELLS,
            "anchor_rule": (
                "complete windows anchored on the global canonical EPSG:25833 "
                "100 m lattice; anchor row/col are multiples of the stride"
            ),
            "cell_id_formula": (
                "E{origin_x + col*100}N{origin_y - row*100} on canonical EPSG:25833"
            ),
        },
        sort_keys=True,
    )
    return sha256_bytes(payload.encode())[:16]


# ── pure window enumeration ───────────────────────────────────────────


def enumerate_window_anchors(
    height: int,
    width: int,
    *,
    row0: int,
    col0: int,
) -> list[tuple[int, int, int, int]]:
    """Return the complete 16x16 windows of a mask as global anchors.

    ``row0``/``col0`` are the global canonical 100 m row/col of the mask's
    local ``(0, 0)`` cell. Returns ``(global_row, global_col, local_row,
    local_col)`` tuples in row-major order. Windows are anchored on global
    multiples of ``PATCH_STRIDE_CELLS`` and never truncated at an edge.
    """
    r_start = (-row0) % PATCH_STRIDE_CELLS
    c_start = (-col0) % PATCH_STRIDE_CELLS
    anchors: list[tuple[int, int, int, int]] = []
    for r in range(r_start, height - PATCH_CELLS + 1, PATCH_STRIDE_CELLS):
        for c in range(c_start, width - PATCH_CELLS + 1, PATCH_STRIDE_CELLS):
            anchors.append((row0 + r, col0 + c, r, c))
    return anchors


def _integral(eligible: np.ndarray) -> np.ndarray:
    """Return the summed-area table of a boolean mask (int64, padded by one)."""
    return np.pad(eligible.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _window_count(integral: np.ndarray, r: int, c: int) -> int:
    """Return the eligible-cell count of the window at local ``(r, c)``."""
    return int(
        integral[r + PATCH_CELLS, c + PATCH_CELLS]
        - integral[r, c + PATCH_CELLS]
        - integral[r + PATCH_CELLS, c]
        + integral[r, c]
    )


# ── input gates ───────────────────────────────────────────────────────


def _mask_grid_errors(src: rasterio.DatasetReader) -> list[str]:
    """Verify the published eligibility-mask COG contract (grid + values)."""
    errors: list[str] = []
    if str(src.crs) != "EPSG:25833":
        errors.append(f"CRS {src.crs}, expected EPSG:25833")
    if src.count != 1 or src.dtypes[0] != "uint8":
        errors.append(f"band count/dtype {src.count}/{src.dtypes[0]}, expected 1/uint8")
    if src.transform.a != CELL_SIZE_M or src.transform.e != -CELL_SIZE_M:
        errors.append(
            f"transform not north-up 100 m (a={src.transform.a}, e={src.transform.e})"
        )
    if src.transform.b != 0.0 or src.transform.d != 0.0:
        errors.append("rotated/skewed transform")
    if (
        abs(src.transform.xoff - CANON_GRID_ORIGIN_X) % CELL_SIZE_M > 1e-6
        or abs(src.transform.yoff - CANON_GRID_ORIGIN_Y) % CELL_SIZE_M > 1e-6
    ):
        errors.append("origin not on the canonical 100 m lattice")
    return errors


def _global_origin(src: rasterio.DatasetReader) -> tuple[int, int]:
    """Map a mask's local ``(0, 0)`` cell to the global canonical 100 m row/col."""
    gcol = round((src.transform.xoff - CANON_GRID_ORIGIN_X) / CELL_SIZE_M)
    grow = round((CANON_GRID_ORIGIN_Y - src.transform.yoff) / CELL_SIZE_M)
    return grow, gcol


def _verify_source_release(source_root: str) -> str:
    """Verify the completed source release and return its policy hash."""
    marker_uri = release_completion(source_root)
    if not exists(marker_uri):
        raise RuntimeError(
            f"source training release {source_root!r} has no completion marker "
            f"({marker_uri}) — the release is incomplete or unpublished"
        )
    try:
        marker = json.loads(read_bytes(marker_uri))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"source release marker unreadable ({marker_uri}): {exc}") from exc
    policy_hash = marker.get("policy_hash")
    if not policy_hash:
        raise RuntimeError(f"source release marker {marker_uri} carries no policy_hash")
    expected = training_policy_hash()
    if policy_hash != expected:
        raise RuntimeError(
            f"source release {source_root!r} policy_hash {policy_hash!r} != current "
            f"training policy {expected!r} — the published release is stale"
        )
    if not exists(manifest_parquet(source_root)):
        raise RuntimeError(f"source release manifest missing: {manifest_parquet(source_root)}")
    return str(policy_hash)


def _read_manifest(source_root: str) -> list[dict]:
    """Read the published scene manifest as a row list (deterministic order)."""
    table = pq.read_table(io.BytesIO(read_bytes(manifest_parquet(source_root))))
    rows = table.to_pylist()
    scene_ids = [str(r["scene_id"]) for r in rows]
    duplicates = sorted({s for s in scene_ids if scene_ids.count(s) > 1})
    if duplicates:
        raise RuntimeError(f"source manifest has duplicate scene_ids: {duplicates}")
    return sorted(rows, key=lambda r: str(r["scene_id"]))


# ── build ─────────────────────────────────────────────────────────────


@dataclass
class PatchIndexBuild:
    """A complete patch-index build (rows, QA summary, fingerprints)."""

    rows: list[dict]
    qa: dict
    source_policy_hash: str
    index_policy_hash: str
    readback: dict = field(default_factory=dict)


def build_patch_index(*, source_root: str, run_id: str) -> PatchIndexBuild:
    """Build the patch index from a completed ``training/v1`` release.

    Reads the published scene manifest and the per-scene eligibility COGs.
    Published scenes contribute their accepted windows; a published scene
    with no accepted window is recorded as ``no_valid_patch``; scenes
    already excluded in the release keep their existing reason and are
    skipped. Inference (2026) rows carry no mask and are skipped.
    """
    source_policy_hash = _verify_source_release(source_root)
    manifest = _read_manifest(source_root)

    rows: list[dict] = []
    exclusions: dict[str, int] = {}
    skipped: dict[str, int] = {}
    no_valid_patch_ids: list[str] = []
    scenes_with_patches = 0

    for scene in manifest:
        scene_id = str(scene["scene_id"])
        split = str(scene["split"])
        status = str(scene["status"])
        reason = str(scene.get("exclusion_reason") or "")
        mask_uri = str(scene.get("eligibility_mask") or "")

        if split == "inference":
            skipped[reason or "inference"] = skipped.get(reason or "inference", 0) + 1
            continue
        if status == "excluded":
            if reason == NO_ELIGIBLE_CELLS_REASON:
                exclusions[NO_ELIGIBLE_CELLS_REASON] = (
                    exclusions.get(NO_ELIGIBLE_CELLS_REASON, 0) + 1
                )
            else:
                key = reason or "excluded"
                skipped[key] = skipped.get(key, 0) + 1
            continue
        if not mask_uri:
            raise RuntimeError(
                f"{scene_id}: published scene has no eligibility mask reference "
                f"in the source manifest"
            )

        scene_rows = _scene_windows(scene=scene, scene_id=scene_id, mask_uri=mask_uri)
        if not scene_rows:
            exclusions[NO_VALID_PATCH_REASON] = exclusions.get(NO_VALID_PATCH_REASON, 0) + 1
            no_valid_patch_ids.append(scene_id)
            continue
        rows.extend(scene_rows)
        scenes_with_patches += 1

    rows.sort(key=lambda r: (r["scene_id"], r["row"], r["col"]))
    qa = _build_qa(
        rows=rows,
        exclusions=exclusions,
        skipped=skipped,
        no_valid_patch_ids=no_valid_patch_ids,
        scenes_with_patches=scenes_with_patches,
        source_root=source_root,
        source_policy_hash=source_policy_hash,
        index_policy_hash=patch_index_policy_hash(),
    )
    return PatchIndexBuild(
        rows=rows,
        qa=qa,
        source_policy_hash=source_policy_hash,
        index_policy_hash=patch_index_policy_hash(),
    )


def _scene_windows(
    *,
    scene: dict,
    scene_id: str,
    mask_uri: str,
) -> list[dict]:
    """Return the accepted patch rows for one published scene."""
    with rasterio.open(mask_uri) as src:
        errors = _mask_grid_errors(src)
        if errors:
            raise RuntimeError(f"{scene_id}: eligibility mask {mask_uri}: " + "; ".join(errors))
        mask = src.read(1)
        grow0, gcol0 = _global_origin(src)
        height, width = src.height, src.width

    if int(mask.max()) > 1:
        raise RuntimeError(f"{scene_id}: eligibility mask {mask_uri} has values outside {{0, 1}}")

    eligible = mask == 1
    integral = _integral(eligible)
    anchors = enumerate_window_anchors(height, width, row0=grow0, col0=gcol0)

    rows: list[dict] = []
    for grow, gcol, r, c in anchors:
        n_eligible = _window_count(integral, r, c)
        if n_eligible < MIN_ELIGIBLE_CELLS:
            continue
        rows.append(
            {
                "scene_id": scene_id,
                "year": int(scene["year"]),
                "split": str(scene["split"]),
                "s2_scene_id": str(scene.get("s2_scene_id") or ""),
                "patch_id": f"{scene_id}:{cell_id(grow, gcol)}",
                "row": grow,
                "col": gcol,
                "row_10m": grow * 10,
                "col_10m": gcol * 10,
                "center_x": CANON_GRID_ORIGIN_X + gcol * CELL_SIZE_M + PATCH_SIZE_10M / 2,
                "center_y": CANON_GRID_ORIGIN_Y - grow * CELL_SIZE_M - PATCH_SIZE_10M / 2,
                "eligibility_mask": mask_uri,
                "n_eligible": n_eligible,
                "n_total": N_CELLS,
                "eligible_frac": n_eligible / N_CELLS,
            }
        )
    return rows


def _build_qa(
    *,
    rows: list[dict],
    exclusions: dict[str, int],
    skipped: dict[str, int],
    no_valid_patch_ids: list[str],
    scenes_with_patches: int,
    source_root: str,
    source_policy_hash: str,
    index_policy_hash: str,
) -> dict:
    """Build the deterministic QA summary (counts, splits/years, coverage)."""
    by_split: dict[str, int] = {}
    by_year: dict[str, int] = {}
    distinct_scenes: dict[str, set[str]] = {}
    for row in rows:
        split = str(row["split"])
        by_split[split] = by_split.get(split, 0) + 1
        year = str(row["year"])
        by_year[year] = by_year.get(year, 0) + 1
        distinct_scenes.setdefault(split, set()).add(str(row["scene_id"]))
    scenes_by_split = {k: len(v) for k, v in sorted(distinct_scenes.items())}

    fracs = [float(row["eligible_frac"]) for row in rows]
    anchors = {(int(row["row"]), int(row["col"])) for row in rows}

    return {
        "schema_version": PATCH_INDEX_SCHEMA_VERSION,
        "index_policy_hash": index_policy_hash,
        "source": {
            "release_root": source_root,
            "policy_hash": source_policy_hash,
        },
        "geometry": {
            "crs": "EPSG:25833",
            "patch_cells": PATCH_CELLS,
            "patch_size_10m": PATCH_SIZE_10M,
            "stride_cells": PATCH_STRIDE_CELLS,
            "n_total": N_CELLS,
            "min_eligible_fraction": MIN_ELIGIBLE_FRACTION,
            "min_eligible_cells": MIN_ELIGIBLE_CELLS,
            "anchor_rule": "global canonical 100 m lattice, multiples of stride",
        },
        "patches_total": len(rows),
        "scenes_with_patches": scenes_with_patches,
        "scenes_no_valid_patch": exclusions.get(NO_VALID_PATCH_REASON, 0),
        "scenes_no_eligible_cells": exclusions.get(NO_ELIGIBLE_CELLS_REASON, 0),
        "scenes_skipped": sum(skipped.values()),
        "exclusions": dict(sorted(exclusions.items())),
        "skipped_reasons": dict(sorted(skipped.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_year": dict(sorted(by_year.items())),
        "scenes_by_split": scenes_by_split,
        "no_valid_patch_scene_ids": sorted(no_valid_patch_ids),
        "n_eligible_total": sum(int(row["n_eligible"]) for row in rows),
        "eligible_frac_min": min(fracs) if fracs else None,
        "eligible_frac_max": max(fracs) if fracs else None,
        "anchor_row_min": min((a[0] for a in anchors), default=None),
        "anchor_row_max": max((a[0] for a in anchors), default=None),
        "anchor_col_min": min((a[1] for a in anchors), default=None),
        "anchor_col_max": max((a[1] for a in anchors), default=None),
        "distinct_anchors": len(anchors),
    }


# ── publication ───────────────────────────────────────────────────────


def publish_patch_index(
    build: PatchIndexBuild,
    *,
    output_root: str,
    run_id: str,
) -> dict[str, str]:
    """Publish the index, QA report, and completion marker under ``output_root``.

    Refuses an occupied destination (artifacts without a completion marker
    are a partial publication and are never overwritten) and refuses to
    overwrite a completed index published under a different policy
    fingerprint. A completed index with the same fingerprints is an
    idempotent no-op. The completion marker is create-only and written last,
    after the publisher-side readback.
    """
    completion_uri = patch_index_completion(output_root)
    if exists(completion_uri):
        marker = json.loads(read_bytes(completion_uri))
        if _same_release(marker, build):
            build.readback = _readback_patch_index(
                build=build, output_root=output_root, marker_expected=True
            )
            return {"complete": completion_uri}
        raise RuntimeError(
            f"patch index already published under a different fingerprint "
            f"({marker.get('index_policy_hash')!r}/{marker.get('source_policy_hash')!r} != "
            f"{build.index_policy_hash!r}/{build.source_policy_hash!r}) — immutable"
        )

    for label, uri in (
        ("patch_index", patch_index_parquet(output_root)),
        ("qa", patch_index_qa(output_root)),
    ):
        if exists(uri):
            raise RuntimeError(
                f"patch-index destination is occupied without a completion marker "
                f"({label}: {uri}) — partial destination, refusing to overwrite"
            )

    lock_uri = f"{output_root.rstrip('/')}/.release.lock"
    lock_payload = {
        "run_id": run_id,
        "index_policy_hash": build.index_policy_hash,
        "source_policy_hash": build.source_policy_hash,
    }
    try:
        with publish_lock(lock_uri, lock_payload):
            # Recheck after acquiring the lock: another publisher may have
            # completed the index while we waited.
            if exists(completion_uri):
                marker = json.loads(read_bytes(completion_uri))
                if _same_release(marker, build):
                    build.readback = _readback_patch_index(
                        build=build, output_root=output_root, marker_expected=True
                    )
                    return {"complete": completion_uri}
                raise RuntimeError(
                    f"patch index already published under a different fingerprint "
                    f"({marker.get('index_policy_hash')!r} != {build.index_policy_hash!r})"
                )

            _write_parquet(build.rows, patch_index_parquet(output_root))
            atomic_write(
                patch_index_qa(output_root),
                json.dumps(build.qa, indent=2, sort_keys=True),
                overwrite=True,
            )

            build.readback = _readback_patch_index(
                build=build, output_root=output_root, marker_expected=False
            )
            if not build.readback.get("ok"):
                raise RuntimeError(
                    f"patch-index readback failed — completion marker NOT written: "
                    f"{build.readback.get('errors')}"
                )

            marker = {
                "published_at": now_iso(),
                "run_id": run_id,
                "index_policy_hash": build.index_policy_hash,
                "source_policy_hash": build.source_policy_hash,
                "patches_total": len(build.rows),
            }
            atomic_write(
                completion_uri,
                json.dumps(marker, indent=2),
                overwrite=False,
                if_generation_match=0,
            )
            build.readback = _readback_patch_index(
                build=build, output_root=output_root, marker_expected=True
            )
            if not build.readback.get("ok"):
                raise RuntimeError(
                    f"patch-index marker verification failed: {build.readback.get('errors')}"
                )
            return {
                "patch_index": patch_index_parquet(output_root),
                "qa": patch_index_qa(output_root),
                "complete": completion_uri,
            }
    except FileExistsError:
        raise RuntimeError(
            f"patch index is being published by another run (lock {lock_uri})"
        ) from None


def _same_release(marker: dict, build: PatchIndexBuild) -> bool:
    """Return True when an existing marker matches this build's fingerprints."""
    return (
        marker.get("index_policy_hash") == build.index_policy_hash
        and marker.get("source_policy_hash") == build.source_policy_hash
    )


def _readback_patch_index(
    *,
    build: PatchIndexBuild,
    output_root: str,
    marker_expected: bool,
) -> dict:
    """Verify the published index by re-reading its artifacts.

    Publisher-side readback (independent re-derivation is the validator's
    job): the Parquet and QA report must exist and parse, the row count and
    QA totals must agree with the build, and the completion marker must
    carry the build's fingerprints when expected.
    """
    errors: list[str] = []
    artifacts: dict[str, bool] = {}

    parquet_uri = patch_index_parquet(output_root)
    qa_uri = patch_index_qa(output_root)
    try:
        table = pq.read_table(io.BytesIO(read_bytes(parquet_uri)))
        artifacts[parquet_uri] = True
        if table.num_rows != len(build.rows):
            errors.append(
                f"patch_index row count {table.num_rows} != expected {len(build.rows)}"
            )
        missing = [name for name in PATCH_INDEX_FIELDNAMES if name not in table.column_names]
        if missing:
            errors.append(f"patch_index missing columns: {missing}")
    except Exception as exc:  # noqa: BLE001 — readback probe
        artifacts[parquet_uri] = False
        errors.append(f"patch_index parquet unreadable: {exc}")

    try:
        qa = json.loads(read_bytes(qa_uri))
        artifacts[qa_uri] = True
        if qa.get("patches_total") != len(build.rows):
            errors.append("qa patches_total does not match the build row count")
        if qa.get("index_policy_hash") != build.index_policy_hash:
            errors.append("qa index_policy_hash mismatch")
        if qa.get("source", {}).get("policy_hash") != build.source_policy_hash:
            errors.append("qa source policy_hash mismatch")
    except Exception as exc:  # noqa: BLE001
        artifacts[qa_uri] = False
        errors.append(f"patch_index QA unreadable: {exc}")

    if marker_expected:
        try:
            marker = json.loads(read_bytes(patch_index_completion(output_root)))
            if not _same_release(marker, build):
                errors.append("completion marker fingerprint mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"completion marker unreadable: {exc}")

    return {"ok": not errors, "artifacts": artifacts, "errors": errors}


def _write_parquet(rows: list[dict], uri: str) -> None:
    """Write a row list as a deterministic Parquet object via an atomic write."""
    import pyarrow as pa

    table = pa.Table.from_pylist(rows, schema=_parquet_schema())
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    atomic_write(uri, buf.getvalue().to_pybytes(), overwrite=True)


def _parquet_schema():
    """Return the explicit index schema (stable column order and types)."""
    import pyarrow as pa

    return pa.schema(
        [
            ("scene_id", pa.string()),
            ("year", pa.int64()),
            ("split", pa.string()),
            ("s2_scene_id", pa.string()),
            ("patch_id", pa.string()),
            ("row", pa.int64()),
            ("col", pa.int64()),
            ("row_10m", pa.int64()),
            ("col_10m", pa.int64()),
            ("center_x", pa.float64()),
            ("center_y", pa.float64()),
            ("eligibility_mask", pa.string()),
            ("n_eligible", pa.int64()),
            ("n_total", pa.int64()),
            ("eligible_frac", pa.float64()),
        ]
    )


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    from berlin_lst_downscaling.data.training.report import now_iso

    return now_iso()


__all__ = [
    "MIN_ELIGIBLE_CELLS",
    "MIN_ELIGIBLE_FRACTION",
    "NO_VALID_PATCH_REASON",
    "N_CELLS",
    "PATCH_CELLS",
    "PATCH_INDEX_FIELDNAMES",
    "PATCH_INDEX_SCHEMA_VERSION",
    "PATCH_SIZE_10M",
    "PATCH_STRIDE_CELLS",
    "PatchIndexBuild",
    "build_patch_index",
    "enumerate_window_anchors",
    "patch_index_policy_hash",
    "publish_patch_index",
]
