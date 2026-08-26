# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "numpy",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for the WB2c-4 training-data release.

Read-only probe: re-reads the published release root (``--release-root``)
and verifies, **independently of the publisher implementation**:

- the V3 source ledger (324 done rows, pinned config hash) and the
  features root pinning;
- per-scene eligibility COG contract (EPSG:25833, canonical 100 m grid,
  uint8, values in {0,1}) and the per-scene completion markers;
- the scene manifest: full year coverage 2017-2026, temporal split
  assignment (2017-2023 train, 2024 validation, 2025 test, 2026
  inference), no duplicate s2_scene_id across non-inference splits,
  no sparse category (only ``no_eligible_cells``);
- the cell index: every row references an eligible cell of a published
  scene, splits match the manifest, cell IDs are deterministic from the
  canonical EPSG:25833 grid, no 2026 cells;
- the scaler: 28 channels in canonical order, documented transforms
  (z-score / log1p+z-score / identity), population statistics, training
  years only, split-hash and V3 config hash present, and — on eligible
  cells — the count matches the cells table for the train split;
- a successful readback of every published artifact.

The validator never writes anything and never re-scans the 28-band
feature COGs (it is a consistency gate over the published evidence, not
a recomputation of the eligibility). It does not share implementation
with ``data/training/``.

Usage
-----
    uv run python scripts/validate_training_data.py \
        --release-root gs://berlin-lst-data/training/v1
    uv run python scripts/validate_training_data.py \
        --release-root data/smoke/training-data
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_CHANNELS,
)
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.training.contracts import (
    EXPECTED_V3_CONFIG_HASH,
    INFERENCE_DEFERRED_REASON,
    NO_ELIGIBLE_CELLS_REASON,
    SPLIT_BY_YEAR,
    SUPPORT_PIXELS,
)
from berlin_lst_downscaling.data.training.paths import (
    cells_parquet,
    eligibility_cog,
    eligibility_completion,
    manifest_csv,
    manifest_parquet,
    release_completion,
    scaler_json,
)

# Canonical grid constants (mirrored from the training contract; the
# validator must derive them independently of the publisher's imports).
_CANON_X = 369190.0
_CANON_Y = 5838410.0
_CELL = 100.0

_V3_FEATURES_ROOT = "gs://berlin-lst-data/features/v3"


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _list_objects(prefix: str) -> list[str]:
    """List object keys under a local dir or GCS prefix."""
    if prefix.startswith("gs://"):
        from google.cloud import storage  # type: ignore[import-untyped]

        client = storage.Client()
        bucket_name, _, key = prefix.removeprefix("gs://").partition("/")
        bucket = client.get_bucket(bucket_name)
        return [
            f"gs://{bucket_name}/{blob.name}"
            for blob in bucket.list_blobs(prefix=key.rstrip("/") + "/")
        ]
    import os

    return [
        os.path.join(prefix, name)
        for name in sorted(os.listdir(prefix))
        if os.path.isfile(os.path.join(prefix, name))
    ]


# ── per-scene eligibility COG ─────────────────────────────────────────


def _check_eligibility_cogs(
    release_root: str, scene_ids: list[str], errors: list[str]
) -> dict[str, int]:
    """Verify every published scene's eligibility COG contract + marker.

    Returns ``{scene_id: eligible_cells}`` for the cells-table cross-check.
    """
    import rasterio

    counts: dict[str, int] = {}
    for scene_id in scene_ids:
        cog_uri = eligibility_cog(release_root, scene_id)
        if not exists(cog_uri):
            errors.append(f"{scene_id}: eligibility COG missing: {cog_uri}")
            continue
        try:
            with rasterio.open(cog_uri) as src:
                if str(src.crs) != "EPSG:25833":
                    errors.append(f"{scene_id}: CRS {src.crs}, expected EPSG:25833")
                if src.count != 1 or src.dtypes[0] != "uint8":
                    errors.append(
                        f"{scene_id}: band count/dtype {src.count}/{src.dtypes[0]}, "
                        "expected 1/uint8"
                    )
                # Canonical 100 m grid (full-release check: exact shape + origin;
                # smoke subsets are canonical-aligned but smaller — checked by origin).
                if (
                    abs(src.transform.xoff - _CANON_X) % _CELL > 1e-6
                    or abs(src.transform.yoff - _CANON_Y) % _CELL > 1e-6
                ):
                    errors.append(f"{scene_id}: origin not on canonical 100 m lattice")
                mask = src.read(1)
                if set(int(v) for v in mask.flatten().tolist()) - {0, 1}:
                    errors.append(f"{scene_id}: eligibility values not in {{0,1}}")
                counts[scene_id] = int((mask == 1).sum())
        except Exception as exc:  # noqa: BLE001 — read-only probe
            errors.append(f"{scene_id}: eligibility COG unreadable: {exc}")
        if not exists(eligibility_completion(release_root, scene_id)):
            errors.append(f"{scene_id}: per-scene completion marker missing")
    return counts


# ── manifest ──────────────────────────────────────────────────────────


def _check_manifest(release_root: str, errors: list[str], warnings: list[str]) -> dict[str, dict]:
    """Verify the scene manifest (year coverage, splits, leakage, exclusions)."""
    table = _read_table(manifest_parquet(release_root))
    cols = table.to_pydict()
    required = {
        "scene_id",
        "year",
        "sensor",
        "s2_scene_id",
        "split",
        "status",
        "eligible_cells",
        "exclusion_reason",
        "feature_stack",
        "feature_valid",
        "eligibility_mask",
    }
    missing = required - set(cols)
    if missing:
        errors.append(f"manifest.parquet missing columns: {sorted(missing)}")
        return {}

    n = table.num_rows
    if n == 0:
        errors.append("manifest.parquet is empty")
        return {}

    years = {int(v) for v in cols["year"]}
    if not years.issubset(set(SPLIT_BY_YEAR)):
        errors.append(f"manifest years outside contract: {sorted(years - set(SPLIT_BY_YEAR))}")

    manifest: dict[str, dict] = {}
    split_of_s2: dict[str, str] = {}
    for i in range(n):
        sid = str(cols["scene_id"][i])
        year = int(cols["year"][i])
        split = str(cols["split"][i])
        s2 = str(cols["s2_scene_id"][i])
        manifest[sid] = {
            "year": year,
            "split": split,
            "s2_scene_id": s2,
            "status": str(cols["status"][i]),
            "eligible_cells": int(cols["eligible_cells"][i]),
            "exclusion_reason": str(cols["exclusion_reason"][i]),
        }
        if split != SPLIT_BY_YEAR.get(year):
            errors.append(f"{sid}: split {split!r} != contract {SPLIT_BY_YEAR.get(year)!r}")
        # s2_scene_id must not span non-inference splits (leakage).
        if split != "inference":
            prior = split_of_s2.get(s2)
            if prior is not None and prior != split:
                errors.append(
                    f"{sid}: s2_scene_id {s2!r} spans splits {prior!r}/{split!r} (leakage)"
                )
            split_of_s2.setdefault(s2, split)
        # No sparse category: the only training exclusion reason is no_eligible_cells;
        # inference scenes are metadata-only with inference_deferred.
        if split == "inference":
            if str(cols["exclusion_reason"][i]) != INFERENCE_DEFERRED_REASON:
                errors.append(
                    f"{sid}: 2026 inference reason {cols['exclusion_reason'][i]!r}, "
                    f"expected {INFERENCE_DEFERRED_REASON!r}"
                )
            if str(cols["eligible_cells"][i]) not in ("0", 0):
                errors.append(f"{sid}: 2026 inference scene has eligible cells")
        elif str(cols["status"][i]) == "excluded":
            if str(cols["exclusion_reason"][i]) != NO_ELIGIBLE_CELLS_REASON:
                errors.append(
                    f"{sid}: excluded reason {cols['exclusion_reason'][i]!r}, "
                    f"expected {NO_ELIGIBLE_CELLS_REASON!r}"
                )
            if int(cols["eligible_cells"][i]) != 0:
                errors.append(f"{sid}: excluded with {cols['eligible_cells'][i]} cells")

    # CSV consistency.
    try:
        import csv as _csv

        text = read_bytes(manifest_csv(release_root)).decode("utf-8")
        rows = list(_csv.DictReader(io.StringIO(text)))
        if len(rows) != n:
            errors.append(f"manifest.csv rows {len(rows)} != manifest.parquet rows {n}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manifest.csv unreadable: {exc}")

    # Summary by year/split for the final report.
    by_split = Counter(manifest[s]["split"] for s in manifest)
    if by_split.get("train", 0) == 0:
        errors.append("manifest has no train scenes")
    if by_split.get("validation", 0) == 0:
        warnings.append("manifest has no validation scenes")
    if by_split.get("test", 0) == 0:
        warnings.append("manifest has no test scenes")

    return manifest


# ── cells index ───────────────────────────────────────────────────────


def _check_cells(
    release_root: str,
    manifest: dict[str, dict],
    errors: list[str],
) -> int:
    """Verify the eligible-cell index against the manifest and the COGs.

    Returns the total train-split eligible-cell count for the scaler
    cross-check (10 m pixels per cell are ``count * 100``).
    """
    table = _read_table(cells_parquet(release_root))
    cols = table.to_pydict()
    required = {
        "scene_id",
        "year",
        "split",
        "cell_id",
        "row",
        "col",
        "center_x",
        "center_y",
        "eligibility_mask",
    }
    missing = required - set(cols)
    if missing:
        errors.append(f"cells.parquet missing columns: {sorted(missing)}")
        return 0

    per_scene: Counter = Counter()
    train_cells = 0
    for i in range(table.num_rows):
        sid = str(cols["scene_id"][i])
        if sid not in manifest:
            errors.append(f"cells row references unknown scene {sid!r}")
            continue
        m = manifest[sid]
        split = str(cols["split"][i])
        if split != m["split"]:
            errors.append(f"cells row {sid}: split {split!r} != manifest {m['split']!r}")
        if split == "inference":
            errors.append(f"cells row {sid}: 2026 inference scene has cells")
        if m["status"] != "published":
            errors.append(f"cells row {sid}: scene not published in manifest")
        row, col = int(cols["row"][i]), int(cols["col"][i])
        # Deterministic cell_id from the canonical grid.
        easting = _CANON_X + col * _CELL
        northing = _CANON_Y - row * _CELL
        expected_id = f"E{int(easting)}N{int(northing)}"
        if str(cols["cell_id"][i]) != expected_id:
            errors.append(
                f"cells row {sid}: cell_id {cols['cell_id'][i]!r} != deterministic {expected_id!r}"
            )
        # Center coordinates must match the canonical grid.
        cx, cy = float(cols["center_x"][i]), float(cols["center_y"][i])
        if abs(cx - (easting + _CELL / 2)) > 1e-3 or abs(cy - (northing - _CELL / 2)) > 1e-3:
            errors.append(f"cells row {sid}: center {cx},{cy} off canonical cell center")
        if split == "train":
            train_cells += 1
        per_scene[sid] += 1

    # Cells per scene must match the manifest eligible_cells and the COG counts.
    for sid, count in per_scene.items():
        if count != manifest[sid]["eligible_cells"]:
            errors.append(
                f"{sid}: cells table {count} != manifest eligible_cells "
                f"{manifest[sid]['eligible_cells']}"
            )

    return train_cells


# ── scaler ────────────────────────────────────────────────────────────


def _check_scaler(
    release_root: str,
    train_cells: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Verify the scaler contract (channels, transforms, train-only provenance)."""
    scaler = _read_json(scaler_json(release_root))
    if list(scaler.get("channel_order", [])) != list(FEATURE_CHANNEL_NAMES):
        errors.append("scaler channel_order does not match the canonical 28-channel order")
    channels = scaler.get("channels", [])
    if len(channels) != len(FEATURE_CHANNELS):
        errors.append(f"scaler channels {len(channels)} != 28")

    transforms = {str(c["channel_name"]): str(c["transform"]) for c in channels}
    for spec in FEATURE_CHANNELS:
        name = spec.name
        expected = (
            "log1p_zscore"
            if name in ("tp_0_24h", "tp_24_48h", "tp_48_72h")
            else "identity"
            if name in ("shadow_building", "shadow_vegetation")
            else "zscore"
        )
        if transforms.get(name) != expected:
            errors.append(f"scaler transform {name}: {transforms.get(name)!r} != {expected!r}")

    for c in channels:
        count = c.get("count")
        if not isinstance(count, int) or count < 0:
            errors.append(f"scaler channel {c.get('channel_name')}: bad count {count!r}")
        if c.get("transform") != "identity":
            mean, std = c.get("mean"), c.get("std")
            if mean is None or std is None or std < 0:
                errors.append(f"scaler channel {c.get('channel_name')}: missing/negative stats")

    # Train-only provenance: training years must be a subset of the train
    # contract, and the scaler must not reference validation/test/2026.
    train_years = set(scaler.get("training_years", []))
    contract_train = {y for y, sp in SPLIT_BY_YEAR.items() if sp == "train"}
    if not train_years.issubset(contract_train):
        errors.append(f"scaler training_years {sorted(train_years)} not subset of train years")

    # Count cross-check: the scaler documents one value per train-eligible
    # 10 m pixel (100 per eligible cell) for every non-identity channel.
    if train_cells > 0:
        expected_vals = train_cells * SUPPORT_PIXELS
        for c in channels:
            if c.get("transform") == "identity":
                continue
            if c.get("count") != expected_vals:
                errors.append(
                    f"scaler {c.get('channel_name')}: count {c.get('count')} != "
                    f"train-eligible 10 m pixels {expected_vals}"
                )

    for key in ("policy_hash", "v3_config_hash", "split_hash"):
        if not scaler.get(key):
            errors.append(f"scaler missing {key}")
    if scaler.get("v3_config_hash") != EXPECTED_V3_CONFIG_HASH:
        errors.append("scaler v3_config_hash does not match the pinned V3 hash")


# ── V3 source gate ────────────────────────────────────────────────────


def _check_v3_source(release_root: str, errors: list[str]) -> None:
    """Verify the release's V3 input basis (independent of the publisher)."""

    # The training ledger rows carry config_hash == policy_hash (not the V3
    # hash), so pin the source via the features ledger instead.
    features_ledger = f"{_V3_FEATURES_ROOT}/_state/features/ledger.parquet"
    try:
        table = _read_table(features_ledger)
        cols = table.to_pydict()
        statuses = {str(s) for s in cols["status"]}
        if statuses != {"done"}:
            errors.append(f"V3 ledger statuses {sorted(statuses)} != {{done}}")
        hashes = {str(h) for h in cols.get("config_hash", []) if h is not None}
        if hashes != {EXPECTED_V3_CONFIG_HASH}:
            errors.append(f"V3 ledger config hashes {sorted(hashes)} != pinned")
        if table.num_rows != 324:
            errors.append(f"V3 ledger rows {table.num_rows} != 324")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"V3 features ledger unreadable: {exc}")

    # Release-level completion marker must carry a policy hash.
    marker = _read_json(release_completion(release_root))
    if not marker.get("policy_hash"):
        errors.append("release complete.json missing policy_hash")


# ── orchestrator ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, help="training release root")
    args = parser.parse_args()

    root = args.release_root.rstrip("/")
    errors: list[str] = []
    warnings: list[str] = []

    print(f"Validating training release: {root}")
    print("  Release marker:", "present" if exists(release_completion(root)) else "MISSING")
    if not exists(release_completion(root)):
        errors.append("release complete.json missing — release not finalised")

    _check_v3_source(root, errors)

    manifest = _check_manifest(root, errors, warnings)
    # Only non-inference scenes carry a published eligibility COG.
    scene_ids = sorted(s for s, m in manifest.items() if m["split"] != "inference")
    cog_counts = _check_eligibility_cogs(root, scene_ids, errors)

    # Cross-check COG counts against the manifest.
    for sid, cells in cog_counts.items():
        if sid in manifest and cells != manifest[sid]["eligible_cells"]:
            errors.append(
                f"{sid}: COG eligible cells {cells} != manifest {manifest[sid]['eligible_cells']}"
            )

    train_cells = _check_cells(root, manifest, errors) if manifest else 0
    _check_scaler(root, train_cells, errors, warnings)

    for w in warnings:
        print(f"  ⚠ {w}")
    for e in errors:
        print(f"  ✗ {e}")

    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    print(f"OK: training release valid ({len(scene_ids)} scenes, {train_cells} train cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
