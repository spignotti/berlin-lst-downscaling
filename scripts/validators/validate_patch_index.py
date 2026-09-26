# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "numpy",
#     "rasterio>=1.4.3",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for the WB3 patch index.

Read-only probe: re-reads the published index root (``--index-root``) and
verifies, **independently of the publisher implementation**:

- the completion marker (create-only, written last) and its policy
  fingerprints;
- the index schema, row alignment, and admission rule: every row is a
  complete 16x16-cell window anchored on a multiple of 16 of the global
  canonical EPSG:25833 100 m lattice, with ``n_eligible >= 244`` of 256
  (>= 95%);
- per-scene non-overlap: anchors within a scene are distinct and never
  closer than one patch;
- the QA report: every aggregate it publishes is re-derived from the index
  rows;
- the source release: the index scene set reconciles with the
  ``training/v1`` manifest (published scenes with patches, ``no_valid_patch``
  and ``no_eligible_cells`` accounting, split inheritance, ``s2_scene_id``);
- a **recomputation** of the accepted windows directly from the source
  eligibility COGs for a documented deterministic scene sample (all scenes
  by default): the accepted-window set and each window's ``n_eligible`` must
  match the index exactly.

The validator never writes anything and never re-scans the 28-band feature
COGs. It does not share implementation with ``data/training/patch_index.py``
— the geometry constants and the window enumeration are mirrored here so a
publisher bug cannot silently validate itself.

Usage
-----
    uv run python scripts/validators/validate_patch_index.py \
        --index-root gs://berlin-lst-training-data/training/patch-index/v1
    uv run python scripts/validators/validate_patch_index.py \
        --index-root data/smoke/patch-index
"""

from __future__ import annotations

import argparse
import io
import json

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import exists, read_bytes, resolve_canonical_uri

# Canonical grid + patch geometry (mirrored from the contract; the
# validator must derive them independently of the publisher's imports).
_CANON_X = 369190.0
_CANON_Y = 5838410.0
_CELL = 100.0
_PATCH_CELLS = 16
_N_CELLS = _PATCH_CELLS * _PATCH_CELLS
_MIN_ELIGIBLE_CELLS = 244  # ceil(0.95 * 256); 243/256 = 0.9492 fails
_NO_ELIGIBLE_CELLS_REASON = "no_eligible_cells"
_NO_VALID_PATCH_REASON = "no_valid_patch"
_NON_INFERENCE_SPLITS = {"train", "validation", "test"}

_REQUIRED_COLUMNS = [
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


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _cell_id(row: int, col: int) -> str:
    """Independent copy of the canonical cell-ID formula."""
    return f"E{int(_CANON_X + col * _CELL)}N{int(_CANON_Y - row * _CELL)}"


def _window_anchors(
    height: int, width: int, *, row0: int, col0: int
) -> list[tuple[int, int, int, int]]:
    """Independent enumeration of complete windows (slicing, not SAT).

    Returns ``(global_row, global_col, local_row, local_col)`` for every
    complete ``16x16`` window anchored on a global multiple of 16.
    """
    r_start = (-row0) % _PATCH_CELLS
    c_start = (-col0) % _PATCH_CELLS
    anchors: list[tuple[int, int, int, int]] = []
    r = r_start
    while r + _PATCH_CELLS <= height:
        c = c_start
        while c + _PATCH_CELLS <= width:
            anchors.append((row0 + r, col0 + c, r, c))
            c += _PATCH_CELLS
        r += _PATCH_CELLS
    return anchors


# ── marker + QA + row contract ────────────────────────────────────────


def _check_marker(root: str, errors: list[str]) -> dict:
    marker_uri = f"{root}/complete.json"
    if not exists(marker_uri):
        errors.append("completion marker missing — index not finalised")
        return {}
    try:
        marker = _read_json(marker_uri)
    except Exception as exc:
        errors.append(f"completion marker unreadable: {exc}")
        return {}
    for key in ("index_policy_hash", "source_policy_hash"):
        if not marker.get(key):
            errors.append(f"completion marker missing {key}")
    return marker


def _check_rows(rows: list[dict], errors: list[str]) -> dict:
    """Verify the per-row contract and return derived aggregates."""
    by_split: dict[str, int] = {}
    by_year: dict[str, int] = {}
    per_scene: dict[str, list[tuple[int, int]]] = {}
    anchors: set[tuple[int, int]] = set()
    fracs: list[float] = []
    n_eligible_total = 0
    seen: set[tuple[str, int, int]] = set()

    for row in rows:
        sid = str(row["scene_id"])
        grow, gcol = int(row["row"]), int(row["col"])
        key = (sid, grow, gcol)
        if key in seen:
            errors.append(f"{sid}: duplicate patch anchor ({grow}, {gcol})")
            continue
        seen.add(key)

        if grow % _PATCH_CELLS or gcol % _PATCH_CELLS:
            errors.append(f"{sid}: anchor ({grow}, {gcol}) not a multiple of {_PATCH_CELLS}")
        if int(row["n_total"]) != _N_CELLS:
            errors.append(f"{sid}: n_total {row['n_total']} != {_N_CELLS}")
        n_elig = int(row["n_eligible"])
        if not 0 <= n_elig <= _N_CELLS:
            errors.append(f"{sid}: n_eligible {n_elig} outside [0, {_N_CELLS}]")
        if n_elig < _MIN_ELIGIBLE_CELLS:
            errors.append(
                f"{sid}: ({grow}, {gcol}) n_eligible {n_elig} < {_MIN_ELIGIBLE_CELLS}"
            )
        frac = float(row["eligible_frac"])
        if frac != n_elig / _N_CELLS:
            errors.append(f"{sid}: ({grow}, {gcol}) eligible_frac {frac} != {n_elig}/{_N_CELLS}")
        if int(row["row_10m"]) != grow * 10 or int(row["col_10m"]) != gcol * 10:
            errors.append(f"{sid}: ({grow}, {gcol}) 10 m anchor is not 10x the 100 m anchor")
        if str(row["patch_id"]) != f"{sid}:{_cell_id(grow, gcol)}":
            errors.append(f"{sid}: ({grow}, {gcol}) patch_id mismatch")
        expected_cx = _CANON_X + gcol * _CELL + _PATCH_CELLS * _CELL / 2
        expected_cy = _CANON_Y - grow * _CELL - _PATCH_CELLS * _CELL / 2
        if float(row["center_x"]) != expected_cx or float(row["center_y"]) != expected_cy:
            errors.append(f"{sid}: ({grow}, {gcol}) center coordinates mismatch")
        split = str(row["split"])
        if split not in _NON_INFERENCE_SPLITS:
            errors.append(f"{sid}: split {split!r} is not a training split")
        if not str(row["eligibility_mask"] or ""):
            errors.append(f"{sid}: ({grow}, {gcol}) missing eligibility_mask reference")

        by_split[split] = by_split.get(split, 0) + 1
        year = str(row["year"])
        by_year[year] = by_year.get(year, 0) + 1
        per_scene.setdefault(sid, []).append((grow, gcol))
        anchors.add((grow, gcol))
        fracs.append(frac)
        n_eligible_total += n_elig

    # Per-scene non-overlap: every anchor is a multiple of the patch size
    # (verified above) and stride equals the patch size, so two patches
    # overlap iff they share an anchor. A repeated anchor within a scene is
    # therefore the only possible overlap.
    for sid, scene_anchors in per_scene.items():
        if len(set(scene_anchors)) != len(scene_anchors):
            errors.append(f"{sid}: repeated patch anchor within the scene (overlap)")

    distinct_scenes: dict[str, set[str]] = {}
    for row in rows:
        distinct_scenes.setdefault(str(row["split"]), set()).add(str(row["scene_id"]))

    return {
        "patches_total": len(rows),
        "scenes_with_patches": len(per_scene),
        "by_split": dict(sorted(by_split.items())),
        "by_year": dict(sorted(by_year.items())),
        "scenes_by_split": {k: len(v) for k, v in sorted(distinct_scenes.items())},
        "n_eligible_total": n_eligible_total,
        "eligible_frac_min": min(fracs) if fracs else None,
        "eligible_frac_max": max(fracs) if fracs else None,
        "distinct_anchors": len(anchors),
        "anchor_row_min": min((a[0] for a in anchors), default=None),
        "anchor_row_max": max((a[0] for a in anchors), default=None),
        "anchor_col_min": min((a[1] for a in anchors), default=None),
        "anchor_col_max": max((a[1] for a in anchors), default=None),
    }


def _check_qa(qa: dict, derived: dict, marker: dict, errors: list[str]) -> None:
    """Reconcile every published QA aggregate against the derived rows."""
    for key in ("patches_total", "scenes_with_patches", "n_eligible_total", "distinct_anchors"):
        if qa.get(key) != derived[key]:
            errors.append(f"qa {key} {qa.get(key)!r} != derived {derived[key]!r}")
    for key in (
        "by_split",
        "by_year",
        "scenes_by_split",
        "anchor_row_min",
        "anchor_row_max",
        "anchor_col_min",
        "anchor_col_max",
        "eligible_frac_min",
        "eligible_frac_max",
    ):
        if qa.get(key) != derived[key]:
            errors.append(f"qa {key} {qa.get(key)!r} != derived {derived[key]!r}")

    geometry = qa.get("geometry", {})
    if geometry.get("patch_cells") != _PATCH_CELLS:
        errors.append(f"qa geometry patch_cells {geometry.get('patch_cells')!r} != {_PATCH_CELLS}")
    if geometry.get("n_total") != _N_CELLS:
        errors.append(f"qa geometry n_total {geometry.get('n_total')!r} != {_N_CELLS}")
    if geometry.get("min_eligible_cells") != _MIN_ELIGIBLE_CELLS:
        errors.append("qa geometry min_eligible_cells does not match the 95% rule")
    if geometry.get("stride_cells") != _PATCH_CELLS:
        errors.append("qa geometry stride_cells is not the non-overlapping patch size")

    if marker:
        if qa.get("index_policy_hash") != marker.get("index_policy_hash"):
            errors.append("qa index_policy_hash != completion marker")
        if qa.get("source", {}).get("policy_hash") != marker.get("source_policy_hash"):
            errors.append("qa source policy_hash != completion marker")


# ── source reconciliation ─────────────────────────────────────────────


def _check_source_release(
    source_root: str,
    rows: list[dict],
    qa: dict,
    errors: list[str],
) -> None:
    """Reconcile the index against the source ``training/v1`` manifest."""
    marker_uri = f"{source_root.rstrip('/')}/complete.json"
    if not exists(marker_uri):
        errors.append(f"source release marker missing: {marker_uri}")
    else:
        try:
            source_marker = _read_json(marker_uri)
            if source_marker.get("policy_hash") != qa.get("source", {}).get("policy_hash"):
                errors.append("qa source policy_hash does not match the source release marker")
        except Exception as exc:
            errors.append(f"source release marker unreadable: {exc}")

    manifest_uri = f"{source_root.rstrip('/')}/manifest.parquet"
    try:
        table = _read_table(manifest_uri)
    except Exception as exc:
        errors.append(f"source manifest unreadable: {exc}")
        return
    manifest = {str(r["scene_id"]): r for r in table.to_pylist()}

    index_scenes = {str(r["scene_id"]) for r in rows}
    index_split = {str(r["scene_id"]): str(r["split"]) for r in rows}
    index_s2 = {str(r["scene_id"]): str(r["s2_scene_id"] or "") for r in rows}
    index_year = {str(r["scene_id"]): int(r["year"]) for r in rows}

    for sid in sorted(index_scenes):
        row = manifest.get(sid)
        if row is None:
            errors.append(f"{sid}: index row has no manifest entry")
            continue
        if str(row["status"]) != "published":
            errors.append(f"{sid}: index row for a non-published manifest scene")
        if str(row["split"]) != index_split[sid]:
            errors.append(f"{sid}: index split {index_split[sid]!r} != manifest {row['split']!r}")
        if int(row["year"]) != index_year[sid]:
            errors.append(f"{sid}: index year {index_year[sid]} != manifest {row['year']}")
        if str(row.get("s2_scene_id") or "") != index_s2[sid]:
            errors.append(f"{sid}: index s2_scene_id does not match the manifest")

    # Exclusion accounting: no_valid_patch scenes are published scenes with
    # eligible cells and no accepted window; no_eligible_cells scenes are the
    # manifest's own exclusion reason.
    qa_no_valid = set(qa.get("no_valid_patch_scene_ids", []))
    expected_no_valid = {
        sid
        for sid, row in manifest.items()
        if str(row["status"]) == "published"
        and int(row["eligible_cells"] or 0) > 0
        and sid not in index_scenes
    }
    if qa_no_valid != expected_no_valid:
        errors.append(
            f"qa no_valid_patch scene set {sorted(qa_no_valid)} != manifest-derived "
            f"{sorted(expected_no_valid)}"
        )
    if qa.get("scenes_no_valid_patch") != len(expected_no_valid):
        errors.append("qa scenes_no_valid_patch does not match the manifest-derived set")

    expected_no_eligible = {
        sid
        for sid, row in manifest.items()
        if str(row["exclusion_reason"] or "") == _NO_ELIGIBLE_CELLS_REASON
    }
    if qa.get("scenes_no_eligible_cells") != len(expected_no_eligible):
        errors.append("qa scenes_no_eligible_cells does not match the manifest")
    overlap = expected_no_valid & expected_no_eligible
    if overlap:
        errors.append(f"scenes counted as both no_valid_patch and no_eligible_cells: {overlap}")


# ── independent recomputation ─────────────────────────────────────────


def _recompute_scenes(
    rows_by_scene: dict[str, list[dict]],
    *,
    max_scenes: int,
    errors: list[str],
) -> int:
    """Recompute accepted windows from the source COGs for sampled scenes."""
    import rasterio

    scene_ids = sorted(rows_by_scene)
    sample = scene_ids if max_scenes <= 0 else scene_ids[:max_scenes]
    checked = 0
    for sid in sample:
        mask_uri = resolve_canonical_uri(str(rows_by_scene[sid][0]["eligibility_mask"]))
        if not exists(mask_uri):
            errors.append(f"{sid}: eligibility mask missing: {mask_uri}")
            continue
        try:
            with rasterio.open(mask_uri) as src:
                if str(src.crs) != "EPSG:25833":
                    errors.append(f"{sid}: mask CRS {src.crs} != EPSG:25833")
                if src.count != 1 or src.dtypes[0] != "uint8":
                    errors.append(f"{sid}: mask dtype/count {src.dtypes[0]}/{src.count}")
                if src.transform.a != _CELL or src.transform.e != -_CELL:
                    errors.append(f"{sid}: mask is not north-up 100 m")
                if (
                    abs(src.transform.xoff - _CANON_X) % _CELL > 1e-6
                    or abs(src.transform.yoff - _CANON_Y) % _CELL > 1e-6
                ):
                    errors.append(f"{sid}: mask origin is off the canonical lattice")
                mask = src.read(1)
                grow0 = round((_CANON_Y - src.transform.yoff) / _CELL)
                gcol0 = round((src.transform.xoff - _CANON_X) / _CELL)
                height, width = src.height, src.width
        except Exception as exc:
            errors.append(f"{sid}: eligibility mask unreadable: {exc}")
            continue

        if int(mask.max()) > 1:
            errors.append(f"{sid}: mask has values outside {{0, 1}}")
        eligible = mask == 1

        accepted: dict[tuple[int, int], int] = {}
        for grow, gcol, r, c in _window_anchors(height, width, row0=grow0, col0=gcol0):
            n_elig = int(eligible[r : r + _PATCH_CELLS, c : c + _PATCH_CELLS].sum())
            if n_elig >= _MIN_ELIGIBLE_CELLS:
                accepted[(grow, gcol)] = n_elig

        indexed = {
            (int(row["row"]), int(row["col"])): int(row["n_eligible"])
            for row in rows_by_scene[sid]
        }
        missing = sorted(set(accepted) - set(indexed))
        extra = sorted(set(indexed) - set(accepted))
        if missing:
            errors.append(f"{sid}: accepted windows missing from the index: {missing[:5]}")
        if extra:
            errors.append(f"{sid}: index windows not accepted on recomputation: {extra[:5]}")
        for anchor, n_elig in accepted.items():
            if anchor in indexed and indexed[anchor] != n_elig:
                errors.append(
                    f"{sid}: ({anchor[0]}, {anchor[1]}) n_eligible {indexed[anchor]} "
                    f"!= recomputed {n_elig}"
                )
        checked += 1
    return checked


# ── orchestrator ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True, help="patch-index root")
    parser.add_argument(
        "--source-root",
        default=None,
        help="source training/v1 root (defaults to the QA report's source.release_root)",
    )
    parser.add_argument(
        "--max-scene-checks",
        type=int,
        default=0,
        help="recompute at most N scenes (sorted; 0 = all scenes with patches)",
    )
    args = parser.parse_args()

    root = args.index_root.rstrip("/")
    errors: list[str] = []

    print(f"Validating patch index: {root}")
    marker = _check_marker(root, errors)

    try:
        table = _read_table(f"{root}/patch_index.parquet")
        missing_cols = [c for c in _REQUIRED_COLUMNS if c not in table.column_names]
        if missing_cols:
            errors.append(f"patch_index.parquet missing columns: {missing_cols}")
        rows = table.to_pylist()
    except Exception as exc:
        errors.append(f"patch_index.parquet unreadable: {exc}")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"FAIL: {len(errors)} finding(s)")
        return 1

    try:
        qa = _read_json(f"{root}/patch_index_qa.json")
    except Exception as exc:
        errors.append(f"patch_index_qa.json unreadable: {exc}")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"FAIL: {len(errors)} finding(s)")
        return 1

    derived = _check_rows(rows, errors)
    _check_qa(qa, derived, marker, errors)

    source_root = args.source_root or str(qa.get("source", {}).get("release_root") or "")
    if not source_root:
        errors.append("no source root: pass --source-root or provide QA source.release_root")
    else:
        _check_source_release(source_root, rows, qa, errors)

    rows_by_scene: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_scene.setdefault(str(row["scene_id"]), []).append(row)
    checked = _recompute_scenes(rows_by_scene, max_scenes=args.max_scene_checks, errors=errors)

    for e in errors:
        print(f"  ✗ {e}")

    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    print(
        f"OK: patch index valid ({derived['patches_total']} patches, "
        f"{derived['scenes_with_patches']} scenes, {checked} recomputed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
