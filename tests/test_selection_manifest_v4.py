"""Offline bundle round trips for additive exact S2 membership."""

import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from berlin_lst_downscaling.data.acquisition.ecostress import parse_granule_datetime
from berlin_lst_downscaling.data.selection.manifest import write_bundle
from berlin_lst_downscaling.data.selection.schema import ECOSTRESS_VALIDATION_IDS
from berlin_lst_downscaling.data.selection.validate import load_bundle, validate_manifest_table


def bundle_inputs():
    anchor = {
        "scene_id": "LC08_offline",
        "year": 2023,
        "datetime": datetime(2023, 6, 1, 10, tzinfo=UTC),
        "item_href": "https://example.test/landsat",
        "aoi_clear_px": 2,
        "aoi_total_px": 4,
        "aoi_clear_frac": 0.5,
    }
    s2 = dict(
        anchor,
        scene_id="s2-a",
        item_href="https://example.test/s2-a",
        s2_items=[
            {"item_id": "s2-a", "item_href": "https://example.test/s2-a"},
            {"item_id": "s2-b", "item_href": "https://example.test/s2-b"},
        ],
        s2_mosaic_group={"solar_day": "2023-06-01", "datatake_id": "GS2A_pass"},
    )
    pair = {
        "anchor": anchor,
        "s2": s2,
        "landsat_clear_px": 2,
        "joint_clear_px": 1,
        "joint_clear_frac": 0.5,
        "score": 0.5,
    }
    eco = [
        {"granule_id": gid, "year": 2018, "datetime": parse_granule_datetime(gid)}
        for gid in sorted(ECOSTRESS_VALIDATION_IDS)
    ]
    return [pair], eco


def write(tmp_path, *, mosaic=True, pairs=None):
    coupled, eco = bundle_inputs()
    return write_bundle(
        coupled if pairs is None else pairs,
        [],
        eco,
        manifest_path=str(tmp_path / "manifest.parquet"),
        pairings_path=str(tmp_path / "pairings.parquet"),
        report_path=str(tmp_path / "manifest_report.json"),
        cutoff_utc="2026-07-17T23:59:59Z",
        cfg={"sentinel2": {"mosaic": {"enabled": mosaic}}},
    )


def test_v4_roundtrip_keeps_exact_membership(tmp_path):
    result = write(tmp_path)
    bundle, validation = load_bundle(result.manifest_path)
    assert validation.ok, validation.errors
    assert bundle.report["schema_version"] == 4
    predictor = next(r for r in bundle.manifest_table.to_pylist() if r["role"] == "predictor")
    assert predictor["s2_items"] == bundle_inputs()[0][0]["s2"]["s2_items"]
    assert predictor["s2_mosaic_group"] == bundle_inputs()[0][0]["s2"]["s2_mosaic_group"]
    assert bundle.manifest_table.schema.metadata[b"schema_version"] == b"4"


def test_legacy_still_writes_v3(tmp_path):
    result = write(tmp_path, mosaic=False)
    table = pq.read_table(result.manifest_path)
    assert "s2_items" not in table.column_names
    assert json.loads((tmp_path / "manifest_report.json").read_text())["schema_version"] == 3
    assert load_bundle(result.manifest_path)[1].ok


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "duplicate",
        "unsorted",
        "href",
        "group",
        "missing-id",
        "representative",
        "group-day",
        "group-null-id",
        "wrong-href",
    ],
)
def test_v4_rejects_bad_membership(tmp_path, mutation):
    pairs, _ = bundle_inputs()
    s2 = pairs[0]["s2"]
    if mutation == "empty":
        s2["s2_items"] = []
    elif mutation == "duplicate":
        s2["s2_items"].append(s2["s2_items"][0])
    elif mutation == "unsorted":
        s2["s2_items"].reverse()
    elif mutation == "href":
        s2["s2_items"][1]["item_href"] = ""
    elif mutation == "group":
        s2["s2_mosaic_group"] = None
    elif mutation == "missing-id":
        s2["s2_items"][1]["item_id"] = None
    elif mutation == "representative":
        s2["scene_id"] = "not-a-member"
    elif mutation == "wrong-href":
        s2["item_href"] = "https://example.test/wrong"
    elif mutation == "group-day":
        s2["s2_mosaic_group"]["solar_day"] = "not-a-day"
    else:
        s2["s2_mosaic_group"]["datatake_id"] = None
    with pytest.raises(ValueError, match="[Mm]embership|s2_items|s2_mosaic_group|Schema mismatch"):
        write(tmp_path, pairs=pairs)


def test_v4_validator_does_not_silently_drop_membership_columns(tmp_path):
    result = write(tmp_path)
    table = pq.read_table(result.manifest_path)
    stripped = table.select([c for c in table.column_names if not c.startswith("s2_")])
    assert not validate_manifest_table(stripped).ok


def test_deduplicated_predictor_rejects_conflicting_memberships(tmp_path):
    from copy import deepcopy

    pairs, _ = bundle_inputs()
    second = deepcopy(pairs[0])
    second["anchor"]["scene_id"] = "LC09_offline"
    second["s2"]["s2_items"][1]["item_id"] = "s2-c"
    with pytest.raises(ValueError, match="Conflicting S2 membership"):
        write(tmp_path, pairs=pairs + [second])


def test_report_manifest_versions_must_agree(tmp_path):
    result = write(tmp_path)
    report_path = tmp_path / "manifest_report.json"
    report = json.loads(report_path.read_text())
    report["schema_version"] = 3
    report_path.write_text(json.dumps(report))
    validation = load_bundle(result.manifest_path)[1]
    assert "Report/manifest schema_version mismatch" in validation.errors


def test_selection_pipeline_reports_prefilter_and_coupling_drops(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from berlin_lst_downscaling.data.selection import pipeline

    pairs, eco = bundle_inputs()
    anchor, s2 = pairs[0]["anchor"], pairs[0]["s2"]
    s2.update(clear_frac=0.5, dt_days=0.0, landsat_clear_px=2, joint_clear_px=1)
    no_match = dict(anchor, scene_id="LC09_no_match")
    prefiltered = [
        {
            "anchor": dict(anchor, scene_id="LC08_sliver", aoi_valid_px=1, aoi_overlap_frac=0.01),
            "reason": "insufficient_aoi_overlap",
        },
        {"anchor": dict(anchor, scene_id="LC09_no_qa"), "reason": "aoi_overlap_unavailable"},
    ]
    monkeypatch.setattr(
        pipeline,
        "build_anchors",
        lambda cfg: (
            [anchor, no_match],
            {"n_total": 4, "n_kept": 2, "n_dropped": 2, "dropped": prefiltered},
        ),
    )
    monkeypatch.setattr(pipeline, "_resolve_landsat_items", lambda *args: [])
    monkeypatch.setattr(
        pipeline,
        "match_s2_candidates_with_clear_frac",
        lambda a, *args: [s2] if a["scene_id"] == anchor["scene_id"] else [],
    )
    monkeypatch.setattr(pipeline, "_resolve_ecostress_allowlist", lambda cfg: eco)
    cfg = OmegaConf.create(
        {
            "years": [2023],
            "months": [6],
            "bbox": [13, 52, 14, 53],
            "output_root": str(tmp_path),
            "cutoff_utc": "2026-07-17T23:59:59Z",
            "sentinel2": {
                "mosaic": {"enabled": True},
                "score": {"lambda": 0.1},
                "min_clear_frac": 0.05,
            },
            "landsat": {"anchor": {"min_aoi_overlap_frac": 0.05}},
        }
    )
    assert pipeline.run_couple(cfg) == 0
    report = json.loads((tmp_path / "manifest_report.json").read_text())
    assert report["counts"]["anchors_total"] == 4
    assert report["counts"]["coupled"] == 1
    assert report["counts"]["dropped"] == 3
    assert report["dropped_reasons"] == {
        "insufficient_aoi_overlap": 1,
        "aoi_overlap_unavailable": 1,
        "no_s2_in_window": 1,
    }
    sliver = next(r for r in report["dropped_anchors"] if r["scene_id"] == "LC08_sliver")
    assert sliver["aoi_overlap_frac"] == 0.01
    assert load_bundle(str(tmp_path / "manifest.parquet"))[1].ok


def test_write_bundle_to_non_local_uri(monkeypatch, tmp_path):
    """write_bundle works end to end for gs:// URIs (failed VM run).

    Regression for run selection-20261023T160311Z-36246C2F: both parquets
    reached GCS via pyarrow, then hashing crashed on local open() and the
    report was never written. The transport is faked (no GCS writes); the
    test asserts the exact uploaded bytes hash to the reported hashes and
    re-validate cleanly, and that no local ``gs:`` mirror is created.
    """
    import io

    from berlin_lst_downscaling.common.util import sha256_bytes
    from berlin_lst_downscaling.data.selection import manifest as manifest_module
    from berlin_lst_downscaling.data.selection.validate import validate_pairings_table

    store: dict[str, bytes] = {}

    def fake_atomic_write(uri, data, **kwargs):
        store[str(uri)] = bytes(data)

    monkeypatch.setattr(manifest_module, "atomic_write", fake_atomic_write)
    monkeypatch.chdir(tmp_path)
    coupled, eco = bundle_inputs()
    root = "gs://test-bucket/manifests/v4/test"
    result = write_bundle(
        coupled,
        [],
        eco,
        manifest_path=f"{root}/manifest.parquet",
        pairings_path=f"{root}/pairings.parquet",
        report_path=f"{root}/manifest_report.json",
        cutoff_utc="2026-07-17T23:59:59Z",
        cfg={"sentinel2": {"mosaic": {"enabled": True}}},
    )
    assert set(store) == {
        f"{root}/manifest.parquet",
        f"{root}/pairings.parquet",
        f"{root}/manifest_report.json",
    }
    manifest_bytes = store[f"{root}/manifest.parquet"]
    pairings_bytes = store[f"{root}/pairings.parquet"]
    assert sha256_bytes(manifest_bytes) == result.manifest_hash
    assert sha256_bytes(pairings_bytes) == result.pairings_hash

    manifest_table = pq.read_table(io.BytesIO(manifest_bytes))
    pairings_table = pq.read_table(io.BytesIO(pairings_bytes))
    assert validate_manifest_table(manifest_table, require_item_href=True).ok
    assert validate_pairings_table(pairings_table, manifest_table).ok

    report = json.loads(store[f"{root}/manifest_report.json"].decode("utf-8"))
    assert report["manifest_hash"] == result.manifest_hash
    assert report["pairings_hash"] == result.pairings_hash
    assert report["schema_version"] == 4

    # No local `gs:` mirror directory may be created for remote URIs.
    assert list(tmp_path.iterdir()) == []
