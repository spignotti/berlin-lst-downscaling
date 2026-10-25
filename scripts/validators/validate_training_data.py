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

- the source ledger (V3: 324 done rows, pinned config hash; v6: all done
  rows with one shared config hash) and the features root pinning;
- per-scene eligibility COG contract (EPSG:25833, canonical 100 m grid,
  uint8, values in {0,1}) and the per-scene completion markers;
- the scene manifest: full year coverage 2017-2026, temporal split
  assignment (2017-2023 train, 2024 validation, 2025 test, 2026
  inference), no duplicate s2_scene_id across non-inference splits,
  no sparse category (``no_eligible_cells``, plus the explicit
  implausible-LST denylist ``usgs_st_implausible_20220827`` and, with
  ``--ard-ledger``, the documented ARD v10 mosaic AOI coverage-gate
  exclusions ``ard sentinel2 not done``, which publish no mask);
- the cell index: every row references an eligible cell of a published
  scene, splits match the manifest, cell IDs are deterministic from the
  canonical EPSG:25833 grid, no 2026 cells;
- the scaler: the release's channel roster in canonical order, documented
  transforms (z-score / log1p+z-score / identity), population statistics,
  training years only, split-hash and config hash present, and — on
  eligible cells — the count matches the cells table for the train split;
- a successful readback of every published artifact.

``--features-release`` selects the contract: ``v5`` (default) is the
published training/v1+v2 contract (features/v5, 28-channel roster, three
precipitation windows, pinned ``EXPECTED_V3_CONFIG_HASH``); ``v6`` is the
training/v3 contract (features/v6 22-channel roster from the feature
schema in src — canopy cover instead of height maxima, one 0–72 h
precipitation sum with log1p then z-score — with the scaler hash compared
against the shared features/v6 ledger hash).
The implausible-LST denylist scenes are required in the release manifest
unless ``--expected-scenes-from-manifest`` proves they are absent from the
upstream selection bundle (reported as an explicit outcome, never skipped
silently).

The validator never writes anything and never re-scans the feature COGs
(it is a consistency gate over the published evidence, not
a recomputation of the eligibility). It does not share implementation
with ``data/training/``.

Usage
-----
    uv run python scripts/validators/validate_training_data.py \
        --release-root gs://berlin-lst-training-data/training/v1
    uv run python scripts/validators/validate_training_data.py \
        --release-root data/smoke/training-data
    uv run python scripts/validators/validate_training_data.py \
        --release-root gs://berlin-lst-training-data/training/v3 \
        --features-release v6 \
        --expected-scenes-from-manifest <v4 manifest> \
        --ard-ledger <v10 ledger>
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.training.contracts import (
    EXPECTED_V3_CONFIG_HASH,
    IMPLAUSIBLE_LST_REASON,
    IMPLAUSIBLE_LST_SCENE_IDS,
    INFERENCE_DEFERRED_REASON,
    NO_ELIGIBLE_CELLS_REASON,
    SPLIT_BY_YEAR,
    SUPPORT_PIXELS,
    scaler_transform_for,
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

_V3_FEATURES_ROOT = "gs://berlin-lst-training-data/features/v5"

# Feature release v6 (22-channel roster) feeding training/v3.
_V6_FEATURES_ROOT = "gs://berlin-lst-training-data/features/v6"

# Feature-release contracts this validator can check. ``v5`` is the
# published training/v1+v2 contract (default); ``v6`` is training/v3.
# The roster, order, and per-channel transforms always come from the
# feature schema in src (single source of truth, never duplicated here).
_FEATURE_RELEASES = ("v5", "v6")

_FEATURES_ROOT_BY_RELEASE = {
    "v5": _V3_FEATURES_ROOT,
    "v6": _V6_FEATURES_ROOT,
}


def _schema_for_features_release(features_release: str):
    """Return the feature roster for a validated release.

    ``v5`` is the 28-channel roster, ``v6`` the features/v6 roster. An
    unknown release is an error, never a silent pass.
    """
    if features_release not in _FEATURE_RELEASES:
        raise ValueError(
            f"unknown features release {features_release!r}: "
            f"expected one of {list(_FEATURE_RELEASES)}"
        )
    if features_release == "v6":
        return feature_schema(FEATURE_SCHEMA_VERSION_V6)
    return feature_schema(FEATURE_SCHEMA_VERSION)


# Exclusion reason the training pipeline records (via the Stage-1
# inventory) for a pairing whose S2 partner failed the ARD v10 mosaic AOI
# coverage gate. Mirrored from ``data/qa/inventory.py::_resolve_scene``
# (``"ard sentinel2 not done"``); the validator accepts it only with
# ``--ard-ledger`` proof that this scene's S2 partner failed the gate.
_S2_GATE_EXCLUSION_REASON = "ard sentinel2 not done"


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


# ── per-scene eligibility COG ─────────────────────────────────────────


def _check_eligibility_cogs(
    release_root: str,
    scene_ids: list[str],
    errors: list[str],
    expected_scenes: int | None,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Verify every published scene's eligibility COG contract + marker.

    Checks the full raster geometry: CRS EPSG:25833, single uint8 band,
    100 m pixel size with no rotation/shear, origin on the canonical
    100 m lattice, and for the full release (``expected_scenes`` set) the
    exact canonical 389x470 shape with north-up orientation. Also returns
    the deterministic cell-ID set per scene for the cells-table check.
    """
    import rasterio

    from berlin_lst_downscaling.data.training.contracts import cell_id as _cell_id

    # Canonical full-grid shape at 100 m (389 rows x 470 cols).
    _CANON_H, _CANON_W = 389, 470

    counts: dict[str, int] = {}
    cell_ids: dict[str, set[str]] = {}
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
                # Pixel size + orientation: exactly 100 m, no rotation/shear,
                # north-up (positive a, negative e).
                if src.transform.a != _CELL or src.transform.e != -_CELL:
                    errors.append(
                        f"{scene_id}: transform not north-up 100 m "
                        f"(a={src.transform.a}, e={src.transform.e})"
                    )
                if src.transform.b != 0.0 or src.transform.d != 0.0:
                    errors.append(f"{scene_id}: rotated/skewed transform")
                # Origin on the canonical 100 m lattice.
                if (
                    abs(src.transform.xoff - _CANON_X) % _CELL > 1e-6
                    or abs(src.transform.yoff - _CANON_Y) % _CELL > 1e-6
                ):
                    errors.append(f"{scene_id}: origin not on canonical 100 m lattice")
                if expected_scenes is not None and (
                    src.height != _CANON_H or src.width != _CANON_W
                ):
                    errors.append(
                        f"{scene_id}: full-release shape {src.height}x{src.width} != "
                        f"canonical {_CANON_H}x{_CANON_W}"
                    )
                mask = src.read(1)
                if set(int(v) for v in mask.flatten().tolist()) - {0, 1}:
                    errors.append(f"{scene_id}: eligibility values not in {{0,1}}")
                counts[scene_id] = int((mask == 1).sum())
                # Deterministic cell-ID set from global canonical row/col.
                rows, cols_idx = np.where(mask == 1)
                ids = set()
                for r, c in zip(rows, cols_idx, strict=False):
                    gcol = round((src.transform.xoff - _CANON_X) / _CELL) + int(c)
                    grow = round((_CANON_Y - src.transform.yoff) / _CELL) + int(r)
                    ids.add(_cell_id(grow, gcol))
                cell_ids[scene_id] = ids
        except Exception as exc:  # noqa: BLE001 — read-only probe
            errors.append(f"{scene_id}: eligibility COG unreadable: {exc}")
        if not exists(eligibility_completion(release_root, scene_id)):
            errors.append(f"{scene_id}: per-scene completion marker missing")
    return counts, cell_ids


# ── manifest ──────────────────────────────────────────────────────────


def _check_manifest(
    release_root: str,
    errors: list[str],
    warnings: list[str],
    expected_scenes: int | None,
    s2_gate_excluded: set[str] | None = None,
    bundle_anchors: set[str] | None = None,
    upstream_absent: list[str] | None = None,
) -> dict[str, dict]:
    """Verify the scene manifest (year coverage, splits, leakage, exclusions).

    ``s2_gate_excluded`` holds the Landsat scene ids proven (via the ARD
    v10 ledger) to pair with a coverage-gate-excluded S2 scene; only those
    may carry the gate exclusion reason.

    ``bundle_anchors`` holds the upstream selection-bundle anchor ids (from
    ``--expected-scenes-from-manifest``). A denylist scene absent from both
    the release manifest and the bundle is proven absent upstream: it is
    recorded in ``upstream_absent`` (explicit outcome, reported by the
    caller) instead of an error. A denylist scene present in the bundle but
    missing from the release manifest stays a hard error — the pipeline
    always emits an excluded row for an in-universe denylist scene.
    ``bundle_anchors`` without ``upstream_absent`` is a caller error (the
    outcome must go somewhere explicit, never silent).
    """
    if bundle_anchors is not None and upstream_absent is None:
        raise ValueError("bundle_anchors requires upstream_absent to record the outcome")
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
    if expected_scenes is not None and n != expected_scenes:
        errors.append(f"manifest rows {n} != expected {expected_scenes}")

    years = {int(v) for v in cols["year"]}
    if not years.issubset(set(SPLIT_BY_YEAR)):
        errors.append(f"manifest years outside contract: {sorted(years - set(SPLIT_BY_YEAR))}")
    if expected_scenes is not None:
        missing_years = set(SPLIT_BY_YEAR) - years
        if missing_years:
            errors.append(f"full release missing years: {sorted(missing_years)}")

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
        # No sparse category. Zero-cell scenes use no_eligible_cells.
        # The two cold 2022 scenes are an explicit denylist: excluded
        # before eligibility, with no mask and no cells.
        reason = str(cols["exclusion_reason"][i])
        # A gate-exclusion reason on a non-excluded row is inconsistent:
        # these pairings publish no mask and are recorded as excluded.
        if reason == _S2_GATE_EXCLUSION_REASON and str(cols["status"][i]) != "excluded":
            errors.append(
                f"{sid}: S2 gate exclusion must have status 'excluded', got {cols['status'][i]!r}"
            )
        if split == "inference":
            if reason != INFERENCE_DEFERRED_REASON:
                errors.append(
                    f"{sid}: 2026 inference reason {reason!r}, "
                    f"expected {INFERENCE_DEFERRED_REASON!r}"
                )
            if str(cols["eligible_cells"][i]) not in ("0", 0):
                errors.append(f"{sid}: 2026 inference scene has eligible cells")
        elif sid in IMPLAUSIBLE_LST_SCENE_IDS:
            if str(cols["status"][i]) != "excluded" or reason != IMPLAUSIBLE_LST_REASON:
                errors.append(
                    f"{sid}: implausible LST scene status {cols['status'][i]!r} "
                    f"reason {reason!r}, expected excluded / {IMPLAUSIBLE_LST_REASON!r}"
                )
            elif int(cols["eligible_cells"][i]) != 0:
                errors.append(f"{sid}: implausible LST scene has eligible cells")
            else:
                for field in ("feature_stack", "feature_valid", "eligibility_mask"):
                    if cols[field][i] not in ("", None):
                        errors.append(f"{sid}: implausible LST scene publishes {field}")
        elif str(cols["status"][i]) == "excluded":
            if reason == _S2_GATE_EXCLUSION_REASON:
                # ARD v10 coverage-gate exclusion: documented, never
                # silent — accepted only with --ard-ledger proof for
                # this scene, with status excluded and no asset refs
                # (no mask is published for these pairings).
                if s2_gate_excluded is None or sid not in s2_gate_excluded:
                    errors.append(f"{sid}: S2 gate exclusion without --ard-ledger proof")
                else:
                    for field in ("feature_stack", "feature_valid", "eligibility_mask"):
                        if cols[field][i] not in ("", None):
                            errors.append(f"{sid}: S2 gate exclusion publishes {field}")
            elif reason != NO_ELIGIBLE_CELLS_REASON:
                errors.append(
                    f"{sid}: excluded reason {reason!r}, expected {NO_ELIGIBLE_CELLS_REASON!r}"
                )
            if int(cols["eligible_cells"][i]) != 0:
                errors.append(f"{sid}: excluded with {cols['eligible_cells'][i]} cells")

    if expected_scenes is not None:
        missing_denied = IMPLAUSIBLE_LST_SCENE_IDS - set(manifest)
        if missing_denied:
            if bundle_anchors is not None:
                proven_upstream = sorted(s for s in missing_denied if s not in bundle_anchors)
                still_missing = sorted(s for s in missing_denied if s in bundle_anchors)
                if proven_upstream and upstream_absent is not None:
                    upstream_absent.extend(proven_upstream)
                if still_missing:
                    errors.append(
                        "full release missing implausible LST scenes "
                        f"present in the selection bundle: {still_missing}"
                    )
            else:
                errors.append(
                    f"full release missing implausible LST scenes: {sorted(missing_denied)}"
                )

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
    cog_cell_ids: dict[str, set[str]],
    errors: list[str],
) -> int:
    """Verify the eligible-cell index against the manifest and the COGs.

    Every (scene_id, cell_id) pair must be unique, and the per-scene cell
    set must equal the eligibility COG's cell set (no duplicates, no
    omitted cells). Returns the total train-split eligible-cell count for
    the scaler cross-check (10 m pixels per cell are ``count * 100``).
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
    per_scene_ids: dict[str, set[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    train_cells = 0
    for i in range(table.num_rows):
        sid = str(cols["scene_id"][i])
        cell = str(cols["cell_id"][i])
        pair = (sid, cell)
        if pair in seen_pairs:
            errors.append(f"cells table duplicate row: {sid} {cell}")
        seen_pairs.add(pair)
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
        if cell != expected_id:
            errors.append(f"cells row {sid}: cell_id {cell!r} != deterministic {expected_id!r}")
        # Center coordinates must match the canonical grid.
        cx, cy = float(cols["center_x"][i]), float(cols["center_y"][i])
        if abs(cx - (easting + _CELL / 2)) > 1e-3 or abs(cy - (northing - _CELL / 2)) > 1e-3:
            errors.append(f"cells row {sid}: center {cx},{cy} off canonical cell center")
        if split == "train":
            train_cells += 1
        per_scene[sid] += 1
        per_scene_ids.setdefault(sid, set()).add(cell)

    # Cells per scene must match the manifest eligible_cells and the COG
    # counts. Compare against EVERY scene with a COG cell set — a published
    # scene with eligible cells but no cells-table rows is a hard error
    # (missing rows treated as an empty set).
    compared = set(per_scene) | set(cog_cell_ids)
    for sid in sorted(compared):
        table_ids = per_scene_ids.get(sid, set())
        count = per_scene.get(sid, 0)
        if sid in manifest and count != manifest[sid]["eligible_cells"]:
            errors.append(
                f"{sid}: cells table {count} != manifest eligible_cells "
                f"{manifest[sid]['eligible_cells']}"
            )
        cog_ids = cog_cell_ids.get(sid)
        if cog_ids is None:
            if count:
                errors.append(f"{sid}: cells rows present but no eligibility COG cell set")
            continue
        if len(table_ids) != count:
            errors.append(f"{sid}: {count} cell rows but only {len(table_ids)} unique cell IDs")
        if table_ids != cog_ids:
            missing_ids = sorted(cog_ids - table_ids)
            extra_ids = sorted(table_ids - cog_ids)
            errors.append(
                f"{sid}: cells table cell set != COG cell set "
                f"(missing {len(missing_ids)}, extra {len(extra_ids)})"
            )

    return train_cells


# ── scaler ────────────────────────────────────────────────────────────


def _check_scaler(
    release_root: str,
    train_cells: int,
    errors: list[str],
    warnings: list[str],
    schema=None,
    expected_v3_hash: str | None = None,
) -> None:
    """Verify the scaler contract (channels, transforms, train-only provenance).

    The expected roster, order, and per-channel transforms come from
    ``schema`` (default: the published v5 28-channel roster, i.e. the
    training/v1+v2 behaviour, unchanged).     ``expected_v3_hash`` is the
    config hash the scaler must carry: the pinned ``EXPECTED_V3_CONFIG_HASH``
    for v5, the shared features/v6 ledger hash for v6 (resolved by
    ``_check_v3_source``). No check is dropped in either mode.
    """
    resolved = schema if schema is not None else _schema_for_features_release("v5")
    expected_hash = expected_v3_hash if expected_v3_hash is not None else EXPECTED_V3_CONFIG_HASH
    expected_names = list(resolved.names)
    scaler = _read_json(scaler_json(release_root))
    if list(scaler.get("channel_order", [])) != expected_names:
        errors.append(
            f"scaler channel_order does not match the canonical "
            f"{resolved.n_channels}-channel {resolved.release} order"
        )
    channels = scaler.get("channels", [])
    if len(channels) != resolved.n_channels:
        errors.append(f"scaler channels {len(channels)} != {resolved.n_channels}")

    transforms = {str(c["channel_name"]): str(c["transform"]) for c in channels}
    for name in expected_names:
        expected = scaler_transform_for(name, resolved)
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
    if scaler.get("v3_config_hash") != expected_hash:
        errors.append(
            f"scaler v3_config_hash {scaler.get('v3_config_hash')!r} does not match "
            f"the {resolved.release} config hash {expected_hash!r}"
        )


# ── V3 source gate ────────────────────────────────────────────────────


def _check_v3_source(
    release_root: str, errors: list[str], features_release: str = "v5"
) -> str | None:
    """Verify the release's feature input basis (independent of the publisher).

    v5 (training/v1+v2): every features/v5 ledger row is ``done``, carries
    the pinned ``EXPECTED_V3_CONFIG_HASH`` (pin source: the feature release
    v5 hash in ``data.training.contracts``), and the ledger holds exactly
    324 rows. Returns the pin.

    v6 (training/v3): the v6 feature hash is computed by the features/v6
    run (``config_hash_for_features``) and cannot be pinned before
    publication, so the gate checks self-consistency like the publisher's
    ``_verify_v6_release``: every features/v6 ledger row is ``done`` and
    carries one shared non-empty config hash (no hard-coded row count —
    ``configs/training/v3.yaml`` sets ``expected_scene_count: null`` and
    the v4 bundle is the universe). Returns that shared hash for the
    scaler cross-check. Returns ``None`` only when the ledger is
    unreadable (an error is recorded).
    """

    # The training ledger rows carry config_hash == policy_hash (not the V3
    # hash), so pin the source via the features ledger instead.
    if features_release not in _FEATURE_RELEASES:
        errors.append(
            f"unknown features release {features_release!r}: "
            f"expected one of {list(_FEATURE_RELEASES)}"
        )
        return None
    features_root = _FEATURES_ROOT_BY_RELEASE[features_release]
    features_ledger = f"{features_root}/_state/features/ledger.parquet"
    expected_hash: str | None = None
    try:
        table = _read_table(features_ledger)
        cols = table.to_pydict()
        statuses = {str(s) for s in cols["status"]}
        if statuses != {"done"}:
            errors.append(f"{features_release} ledger statuses {sorted(statuses)} != {{done}}")
        hashes = {str(h) for h in cols.get("config_hash", []) if h is not None}
        if features_release == "v6":
            non_empty = {h for h in hashes if h}
            if len(non_empty) != 1:
                errors.append(
                    f"v6 ledger must carry one shared config hash, found {sorted(hashes)}"
                )
            else:
                expected_hash = next(iter(non_empty))
        else:
            if hashes != {EXPECTED_V3_CONFIG_HASH}:
                errors.append(f"V3 ledger config hashes {sorted(hashes)} != pinned")
            if table.num_rows != 324:
                errors.append(f"V3 ledger rows {table.num_rows} != 324")
            expected_hash = EXPECTED_V3_CONFIG_HASH
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{features_release} features ledger unreadable: {exc}")
        return None

    # Release-level completion marker must carry a policy hash.
    marker = _read_json(release_completion(release_root))
    if not marker.get("policy_hash"):
        errors.append("release complete.json missing policy_hash")
    return expected_hash


# ── orchestrator ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, help="training release root")
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=None,
        help="exact manifest row count for the full release (e.g. 345); "
        "enforces full year coverage 2017-2026",
    )
    parser.add_argument(
        "--expected-scenes-from-manifest",
        default=None,
        metavar="MANIFEST_PARQUET",
        help="Derive the expected manifest row count from a selection "
        "manifest bundle (every anchor including 2026); mutually "
        "exclusive with --expected-scenes",
    )
    parser.add_argument(
        "--ard-ledger",
        default=None,
        metavar="ARD_LEDGER_PARQUET",
        help="ARD v10 ledger: pairings whose S2 partner failed the mosaic "
        "AOI coverage gate are accepted as documented exclusions "
        "(exclusion reason 'ard sentinel2 not done', no eligibility "
        "mask). Requires --expected-scenes-from-manifest. The release "
        "manifest row count is unchanged (one row per pairing).",
    )
    parser.add_argument(
        "--features-release",
        choices=list(_FEATURE_RELEASES),
        default="v5",
        help="Feature-release contract to validate: 'v5' is the published "
        "training/v1+v2 contract (features/v5, 28-channel roster, pinned "
        "V3 config hash; default); 'v6' is the training/v3 contract "
        "(features/v6 roster from the feature schema in src, scaler hash "
        "compared against the shared features/v6 ledger hash). A scaler or "
        "ledger that does not match the selected contract is an error.",
    )
    args = parser.parse_args()

    if args.expected_scenes is not None and args.expected_scenes_from_manifest is not None:
        parser.error("--expected-scenes and --expected-scenes-from-manifest are exclusive")
    if args.ard_ledger is not None and args.expected_scenes_from_manifest is None:
        parser.error("--ard-ledger requires --expected-scenes-from-manifest")
    schema = _schema_for_features_release(args.features_release)
    bundle_anchors: set[str] | None = None
    if args.expected_scenes_from_manifest is not None:
        from berlin_lst_downscaling.data.selection.bundle_counts import (
            anchor_years,
            expected_counts_for_bundle,
        )

        _, total = expected_counts_for_bundle(args.expected_scenes_from_manifest)
        print(f"  Expected scenes from manifest bundle: {total}")
        args.expected_scenes = total
        bundle_manifest_uri = args.expected_scenes_from_manifest.rstrip("/")
        bundle_root = bundle_manifest_uri.rsplit("/", 1)[0]
        pairings_uri = f"{bundle_root}/pairings.parquet"
        for uri in (args.expected_scenes_from_manifest, pairings_uri):
            if not exists(uri):
                parser.error(f"bundle artifact missing: {uri}")
        import pyarrow.parquet as _pq

        bundle_table = _pq.read_table(io.BytesIO(read_bytes(args.expected_scenes_from_manifest)))
        bundle_anchors = set(anchor_years(bundle_table))

    from berlin_lst_downscaling.data.selection.bundle_counts import (
        excluded_landsat_scene_ids,
    )

    s2_gate_excluded: set[str] = set()
    if args.ard_ledger is not None:
        s2_gate_excluded = set(
            excluded_landsat_scene_ids(args.expected_scenes_from_manifest, args.ard_ledger)
        )
        print(f"  S2 coverage-gate excluded pairings ({len(s2_gate_excluded)}):")
        for sid in sorted(s2_gate_excluded):
            print(f"    excluded pairing {sid} (S2 failed the AOI coverage gate)")

    root = args.release_root.rstrip("/")
    errors: list[str] = []
    warnings: list[str] = []
    upstream_absent: list[str] = []

    print(f"Validating training release: {root} (features-release {args.features_release})")
    print("  Release marker:", "present" if exists(release_completion(root)) else "MISSING")
    if not exists(release_completion(root)):
        errors.append("release complete.json missing — release not finalised")

    expected_v3_hash = _check_v3_source(root, errors, features_release=args.features_release)

    manifest = _check_manifest(
        root,
        errors,
        warnings,
        args.expected_scenes,
        s2_gate_excluded=s2_gate_excluded,
        bundle_anchors=bundle_anchors,
        upstream_absent=upstream_absent,
    )
    if upstream_absent:
        print(f"  Implausible-LST denylist scenes absent upstream ({len(upstream_absent)}):")
        for sid in sorted(upstream_absent):
            print(f"    {sid} not in the selection bundle — no release row expected")
    # Inference scenes, the implausible-LST denylist, and proven S2
    # gate exclusions publish no mask. Zero-cell exclusions still carry
    # an all-zero mask.
    scene_ids = sorted(
        s
        for s, m in manifest.items()
        if m["split"] != "inference"
        and s not in IMPLAUSIBLE_LST_SCENE_IDS
        and s not in s2_gate_excluded
    )
    cog_counts, cog_cell_ids = _check_eligibility_cogs(
        root, scene_ids, errors, args.expected_scenes
    )

    # Cross-check COG counts against the manifest.
    for sid, cells in cog_counts.items():
        if sid in manifest and cells != manifest[sid]["eligible_cells"]:
            errors.append(
                f"{sid}: COG eligible cells {cells} != manifest {manifest[sid]['eligible_cells']}"
            )

    train_cells = _check_cells(root, manifest, cog_cell_ids, errors) if manifest else 0
    _check_scaler(
        root,
        train_cells,
        errors,
        warnings,
        schema=schema,
        expected_v3_hash=expected_v3_hash,
    )

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
