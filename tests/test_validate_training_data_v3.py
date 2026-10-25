"""validate_training_data roster-aware contract: v5 unchanged, v6 for training/v3.

Covers task acceptance criteria 1-3 and 5 (offline, synthetic local files):
- v6 scaler (22-channel roster, single-precipitation log1p+zscore) passes;
  the pre-fix validator rejected it with the 12 reported findings.
- v5 validation behaves exactly as before (28-channel, pinned hash).
- cross-release shapes fail closed in both directions; no check is dropped
  (wrong hash is an error in both modes).
- the implausible-LST denylist is bundle-aware: absent upstream with
  --expected-scenes-from-manifest proof is an explicit outcome; present in
  the bundle but missing from the release is an error.
- main() passes end-to-end on a v3-shaped local release with mocks only
  for the COG/cells-table probes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
_VALIDATOR_PATH = _REPO / "scripts/validators/validate_training_data.py"

V6_HASH = "abc123abc123abc1"  # stand-in for the shared features/v6 ledger hash


def _load_validator(name: str = "validate_training_data_v3_test"):
    spec = importlib.util.spec_from_file_location(name, _VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v5_schema():
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION,
        feature_schema,
    )

    return feature_schema(FEATURE_SCHEMA_VERSION)


def _v6_schema():
    from berlin_lst_downscaling.data.features.contracts import (
        FEATURE_SCHEMA_VERSION_V6,
        feature_schema,
    )

    return feature_schema(FEATURE_SCHEMA_VERSION_V6)


def _scaler_payload(schema, v3_hash: str, count: int) -> dict:
    from berlin_lst_downscaling.data.training.contracts import scaler_transform_for

    return {
        "channel_order": list(schema.names),
        "channels": [
            {
                "channel_name": name,
                "transform": scaler_transform_for(name, schema),
                "count": count,
                "mean": 0.0,
                "std": 1.0,
            }
            for name in schema.names
        ],
        "policy_hash": "policy",
        "v3_config_hash": v3_hash,
        "split_hash": "split",
        "training_years": [2021],
    }


def _write_scaler_only(release_dir: Path, payload: dict) -> None:
    (release_dir / "scaler.json").write_text(json.dumps(payload, indent=2))


# ── roster comes from src, never duplicated ────────────────────────────


def test_v6_roster_comes_from_src_not_a_duplicate_list() -> None:
    """The validator resolves the v6 roster from the src feature schema."""
    module = _load_validator("validate_training_data_v3_src")
    schema = module._schema_for_features_release("v6")
    assert list(schema.names) == list(_v6_schema().names)
    assert schema.n_channels == _v6_schema().n_channels == 22
    v5 = module._schema_for_features_release("v5")
    assert v5.n_channels == 28
    # The validator module must not hard-code a second copy of the rosters.
    text = _VALIDATOR_PATH.read_text()
    for literal in (
        "canopy_cover_fraction",
        "tp_0_72h",
        "tp_0_24h",
        "building_height_std",
        "vegetation_height_max",
        "ssrd_antecedent",
        "vpd_scene",
    ):
        assert literal not in text, f"v6/v5 roster looks duplicated: {literal}"
    # Unknown releases fail closed, never silently pass.
    import pytest

    with pytest.raises(ValueError, match="unknown features release"):
        module._schema_for_features_release("v9")


# ── scaler contract ────────────────────────────────────────────────────


def test_scaler_accepts_v6_roster(tmp_path: Path) -> None:
    """A v6-correct scaler passes (the 10 scaler findings are fixed)."""
    from berlin_lst_downscaling.data.training.contracts import EXPECTED_V3_CONFIG_HASH

    module = _load_validator("validate_training_data_v3_scaler")
    assert EXPECTED_V3_CONFIG_HASH != V6_HASH  # the two pins really differ
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    schema = _v6_schema()
    _write_scaler_only(release_dir, _scaler_payload(schema, V6_HASH, 0))
    errors: list[str] = []
    module._check_scaler(str(release_dir), 0, errors, [], schema=schema, expected_v3_hash=V6_HASH)
    assert errors == []


def test_scaler_v2_behaviour_unchanged(tmp_path: Path) -> None:
    """training/v2 validation behaves as before (default args, pinned hash)."""
    from berlin_lst_downscaling.data.training.contracts import EXPECTED_V3_CONFIG_HASH

    module = _load_validator("validate_training_data_v3_v2")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    _write_scaler_only(release_dir, _scaler_payload(_v5_schema(), EXPECTED_V3_CONFIG_HASH, 0))
    errors: list[str] = []
    module._check_scaler(str(release_dir), 0, errors, [])
    assert errors == []

    # A reordered v5 scaler still fails, with the v5 release named.
    bad = _scaler_payload(_v5_schema(), EXPECTED_V3_CONFIG_HASH, 0)
    bad["channel_order"] = list(reversed(bad["channel_order"]))
    _write_scaler_only(release_dir, bad)
    errors2: list[str] = []
    module._check_scaler(str(release_dir), 0, errors2, [])
    assert any("28-channel v5 order" in e for e in errors2)


def test_scaler_rejects_cross_release_shapes(tmp_path: Path) -> None:
    """A v6 scaler validated as v5 (and vice versa) is an error, never a pass."""
    from berlin_lst_downscaling.data.training.contracts import EXPECTED_V3_CONFIG_HASH

    module = _load_validator("validate_training_data_v3_cross")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    _write_scaler_only(release_dir, _scaler_payload(_v6_schema(), V6_HASH, 0))
    errors: list[str] = []
    module._check_scaler(str(release_dir), 0, errors, [])
    assert any("22 != 28" in e for e in errors)
    assert any("pinned" in e or "2d2086f9d36e48fe" in e for e in errors)

    _write_scaler_only(release_dir, _scaler_payload(_v5_schema(), EXPECTED_V3_CONFIG_HASH, 0))
    errors2: list[str] = []
    module._check_scaler(
        str(release_dir), 0, errors2, [], schema=_v6_schema(), expected_v3_hash=V6_HASH
    )
    assert any("28 != 22" in e for e in errors2)


def test_scaler_hash_check_not_dropped_in_v6(tmp_path: Path) -> None:
    """A v6 scaler carrying the wrong hash is a hard error naming the hash."""
    module = _load_validator("validate_training_data_v3_hash")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    _write_scaler_only(release_dir, _scaler_payload(_v6_schema(), "wronghash", 0))
    errors: list[str] = []
    module._check_scaler(
        str(release_dir), 0, errors, [], schema=_v6_schema(), expected_v3_hash=V6_HASH
    )
    assert any("v3_config_hash" in e and V6_HASH in e for e in errors)


def test_v6_source_gate_returns_shared_hash_not_pin(tmp_path: Path) -> None:
    """_check_v3_source(v6) enforces one shared ledger hash and returns it."""
    module = _load_validator("validate_training_data_v3_gate")
    ledger = pa.table(
        {
            "status": ["done", "done"],
            "config_hash": [V6_HASH, V6_HASH],
            "period_or_vintage": ["A", "B"],
        }
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "complete.json").write_text('{"policy_hash": "p"}')

    real_read_table = module._read_table
    module._read_table = lambda uri: (  # noqa: E731
        ledger if uri.startswith("gs://") else real_read_table(uri)
    )
    real_read_json = module._read_json
    module._read_json = lambda uri: (  # noqa: E731
        json.loads((release_dir / "complete.json").read_text())
        if uri.startswith("gs://")
        else real_read_json(uri)
    )
    errors: list[str] = []
    assert module._check_v3_source(str(release_dir), errors, features_release="v6") == V6_HASH
    assert errors == []

    # Two hashes in the v6 ledger fail closed (no silent pick).
    module._read_table = lambda uri: pa.table(  # noqa: E731
        {
            "status": ["done", "done"],
            "config_hash": [V6_HASH, "otherhash1234567"],
            "period_or_vintage": ["A", "B"],
        }
    )
    errors2: list[str] = []
    assert module._check_v3_source(str(release_dir), errors2, features_release="v6") is None
    assert any("one shared config hash" in e for e in errors2)


# ── denylist × bundle proof ────────────────────────────────────────────


_MANIFEST_FIELDS = [
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
]


def _row(sid: str, year: int, split: str, s2: str) -> dict:
    return {
        "scene_id": sid,
        "year": year,
        "sensor": "LC08",
        "s2_scene_id": s2,
        "split": split,
        "status": "published",
        "eligible_cells": 5,
        "exclusion_reason": "",
        "feature_stack": "x",
        "feature_valid": "x",
        "eligibility_mask": "x",
    }


def _write_release_manifest(release_dir: Path, rows: list[dict]) -> None:
    pq.write_table(
        pa.table({k: [r[k] for r in rows] for k in _MANIFEST_FIELDS}),
        release_dir / "manifest.parquet",
    )
    with open(release_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _full_cover_rows() -> list[dict]:
    """One published row per year 2017-2026 (no denylist scenes)."""
    rows = []
    for year in range(2017, 2026):
        split = "train" if year <= 2023 else ("validation" if year == 2024 else "test")
        rows.append(_row(f"L{year}", year, split, f"S{year}"))
    rows.append(
        {
            **_row("L2026", 2026, "inference", "S2026"),
            "status": "excluded",
            "eligible_cells": 0,
            "exclusion_reason": "inference_deferred",
            "feature_stack": "",
            "feature_valid": "",
            "eligibility_mask": "",
        }
    )
    return rows


def test_denylist_absent_upstream_is_explicit_outcome(tmp_path: Path) -> None:
    """Denylist scenes missing from bundle + release pass with proof recorded."""
    from berlin_lst_downscaling.data.training.contracts import IMPLAUSIBLE_LST_SCENE_IDS

    module = _load_validator("validate_training_data_v3_deny1")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    rows = _full_cover_rows()
    _write_release_manifest(release_dir, rows)
    bundle_anchors = {r["scene_id"] for r in rows}
    assert not (IMPLAUSIBLE_LST_SCENE_IDS & bundle_anchors)

    errors: list[str] = []
    upstream_absent: list[str] = []
    manifest = module._check_manifest(
        str(release_dir),
        errors,
        [],
        len(rows),
        bundle_anchors=bundle_anchors,
        upstream_absent=upstream_absent,
    )
    assert errors == []
    assert sorted(upstream_absent) == sorted(IMPLAUSIBLE_LST_SCENE_IDS)
    assert len(manifest) == len(rows)


def test_denylist_in_bundle_but_missing_from_release_is_error(tmp_path: Path) -> None:
    """A denylist scene in the bundle but absent from the release stays an error."""
    from berlin_lst_downscaling.data.training.contracts import IMPLAUSIBLE_LST_SCENE_IDS

    module = _load_validator("validate_training_data_v3_deny2")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    rows = _full_cover_rows()
    _write_release_manifest(release_dir, rows)
    deny = sorted(IMPLAUSIBLE_LST_SCENE_IDS)[0]
    bundle_anchors = {r["scene_id"] for r in rows} | {deny}

    errors: list[str] = []
    module._check_manifest(
        str(release_dir),
        errors,
        [],
        len(rows),
        bundle_anchors=bundle_anchors,
        upstream_absent=[],
    )
    assert any("present in the selection bundle" in e and deny in e for e in errors)


def test_denylist_present_as_excluded_still_passes(tmp_path: Path) -> None:
    """Denylist scenes present as excluded rows keep passing (both modes)."""
    from berlin_lst_downscaling.data.training.contracts import (
        IMPLAUSIBLE_LST_REASON,
        IMPLAUSIBLE_LST_SCENE_IDS,
    )

    module = _load_validator("validate_training_data_v3_deny3")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    rows = _full_cover_rows()
    for deny in sorted(IMPLAUSIBLE_LST_SCENE_IDS):
        rows.append(
            {
                **_row(deny, 2022, "train", f"S-{deny[-6:]}"),
                "status": "excluded",
                "eligible_cells": 0,
                "exclusion_reason": IMPLAUSIBLE_LST_REASON,
                "feature_stack": "",
                "feature_valid": "",
                "eligibility_mask": "",
            }
        )
    _write_release_manifest(release_dir, rows)
    bundle_anchors = {r["scene_id"] for r in rows}
    errors: list[str] = []
    manifest = module._check_manifest(
        str(release_dir),
        errors,
        [],
        len(rows),
        bundle_anchors=bundle_anchors,
        upstream_absent=[],
    )
    assert errors == []
    assert len(manifest) == len(rows)


def test_bundle_anchors_without_outcome_list_is_caller_error(tmp_path: Path) -> None:
    """bundle_anchors without upstream_absent must not silently skip."""
    import pytest

    module = _load_validator("validate_training_data_v3_deny4")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    _write_release_manifest(release_dir, _full_cover_rows())
    with pytest.raises(ValueError, match="upstream_absent"):
        module._check_manifest(str(release_dir), [], [], 10, bundle_anchors={"L2017"})


# ── main() end-to-end on a v3-shaped release ───────────────────────────


def _write_bundle(bundle_dir: Path, anchors: list[tuple[str, int]]) -> None:
    pq.write_table(
        pa.table(
            {
                "scene_id": [a[0] for a in anchors],
                "source": ["landsat-c2-l2"] * len(anchors),
                "role": ["anchor"] * len(anchors),
                "year": [a[1] for a in anchors],
            }
        ),
        bundle_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "landsat_scene_id": [a[0] for a in anchors],
                "sentinel2_scene_id": [f"S-{a[0]}" for a in anchors],
            }
        ),
        bundle_dir / "pairings.parquet",
    )


def test_main_passes_v3_shaped_release(tmp_path, monkeypatch, capsys) -> None:
    """main() with --features-release v6 passes a v3-shaped local release.

    Bundle (11 anchors, 2017-2026, no denylist scenes) + release manifest
    (one row per pairing, 1 S2 gate exclusion, 2026 inference) + v6 scaler
    with the shared ledger hash. Only the COG/cells-table probes are
    mocked; manifest, source gate, and scaler checks run for real.
    """
    import berlin_lst_downscaling.data.io as io_module

    module = _load_validator("validate_training_data_v3_main")

    anchors = [(f"L{y}", y) for y in range(2017, 2026)] + [("L2026", 2026), ("LG", 2021)]
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle(bundle_dir, anchors)

    ard_path = tmp_path / "ard_ledger.parquet"
    gate_error = (
        "Sentinel-2 mosaic AOI coverage failed: missing 560178/8922403 pixels "
        "(0.06277345) > max_missing_aoi_fraction=0.005; item_ids=['S2A_X']"
    )
    pq.write_table(
        pa.table(
            {
                "scene_id": [f"S-{a[0]}" for a in anchors],
                "source": ["sentinel-2-l2a"] * len(anchors),
                "status": ["failed" if a[0] == "LG" else "done" for a in anchors],
                "last_error": [gate_error if a[0] == "LG" else None for a in anchors],
            }
        ),
        ard_path,
    )

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "complete.json").write_text('{"policy_hash": "p"}')
    rows: list[dict] = []
    for sid, year in anchors:
        s2 = f"S-{sid}"
        if sid == "LG":
            rows.append(
                {
                    **_row(sid, year, "train", s2),
                    "status": "excluded",
                    "eligible_cells": 0,
                    "exclusion_reason": "ard sentinel2 not done",
                    "feature_stack": "",
                    "feature_valid": "",
                    "eligibility_mask": "",
                }
            )
        elif year == 2026:
            rows.append(
                {
                    **_row(sid, year, "inference", s2),
                    "status": "excluded",
                    "eligible_cells": 0,
                    "exclusion_reason": "inference_deferred",
                    "feature_stack": "",
                    "feature_valid": "",
                    "eligibility_mask": "",
                }
            )
        else:
            split = "train" if year <= 2023 else ("validation" if year == 2024 else "test")
            rows.append(_row(sid, year, split, s2))
    _write_release_manifest(release_dir, rows)

    train_cells = 4
    schema = _v6_schema()
    payload = _scaler_payload(schema, V6_HASH, train_cells * 100)
    (release_dir / "scaler.json").write_text(json.dumps(payload, indent=2))

    monkeypatch.setattr(io_module, "exists", lambda uri: True)
    monkeypatch.setattr(module, "exists", lambda uri: True)
    v6_ledger = pa.table(
        {
            "status": ["done"],
            "config_hash": [V6_HASH],
            "period_or_vintage": ["L2017"],
        }
    )
    real_read_table = module._read_table
    module._read_table = lambda uri: (  # noqa: E731
        v6_ledger if "features/v6" in uri else real_read_table(uri)
    )
    monkeypatch.setattr(module, "_check_eligibility_cogs", lambda *args: ({}, {}))
    monkeypatch.setattr(module, "_check_cells", lambda *args: train_cells)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_training_data.py",
            "--release-root",
            str(release_dir),
            "--features-release",
            "v6",
            "--expected-scenes-from-manifest",
            str(bundle_dir / "manifest.parquet"),
            "--ard-ledger",
            str(ard_path),
        ],
    )
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "features-release v6" in out
    assert "S2 coverage-gate excluded pairings (1)" in out
    assert "absent upstream (2)" in out
    assert "OK: training release valid" in out
