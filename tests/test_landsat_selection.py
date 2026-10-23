"""Small offline regressions for individual Landsat anchor footprints."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.selection import anchors


def _item(path_row: str):
    return SimpleNamespace(
        id=f"LC09_L2SP_{path_row}_20240629_02_T1",
        datetime=datetime(2024, 6, 29, 10, 20, tzinfo=UTC),
        properties={"datetime": "2024-06-29T10:20:00Z"},
        # Both bounding boxes contain the entire fake AOI. Only QA reveals
        # that the edge scene's actual data footprint barely intersects it.
        bbox=[13.0, 52.0, 14.0, 53.0],
    )


def _config(tmp_path, **thresholds):
    return OmegaConf.create(
        {
            "years": [2024],
            "months": [6],
            "bbox": [13.08, 52.34, 13.76, 52.68],
            "landsat": {
                "collection": "landsat-c2-l2",
                "anchor": {"min_clear_frac": 0.0, "min_aoi_overlap_frac": 0.05, **thresholds},
            },
            "aoi": {"mask_base": str(tmp_path)},
            "output_root": str(tmp_path),
        }
    )


def _fake_inputs(monkeypatch, valid_px=4, qa_override=None):
    edge, interior = _item("194023"), _item("193024")
    items = [edge, interior]
    calls = []

    class Catalog:
        def search(self, **kwargs):
            selected = [it for it in items if it.id in kwargs.get("ids", [it.id])]
            return SimpleNamespace(items=lambda: iter(selected))

    def load(**kwargs):
        calls.append(kwargs)
        # A fused load incorrectly makes the edge anchor look fully covered.
        qa = np.zeros((10, 10), dtype=np.uint16)
        if [it.id for it in kwargs["items"]] == [edge.id]:
            qa[:] = 1  # QA_PIXEL bit 0: fill
            qa.flat[:valid_px] = 0
            if qa_override is not None:
                qa = qa_override
        return xr.Dataset({"qa_pixel": (("time", "y", "x"), qa[None])})

    monkeypatch.setattr(anchors, "get_catalog", lambda: Catalog())
    monkeypatch.setattr(anchors, "stac_load", load)
    monkeypatch.setattr(anchors, "canon_grid_10m", lambda: object())
    monkeypatch.setattr(anchors, "load_aoi_mask", lambda *args: np.ones((10, 10), dtype=bool))
    return edge, interior, calls


def test_edge_194023_is_excluded_by_valid_footprint_before_coupling(monkeypatch, tmp_path):
    edge, interior, _ = _fake_inputs(monkeypatch)

    kept, stats = anchors.build_anchors(_config(tmp_path))

    assert [a["scene_id"] for a in kept] == [interior.id]
    assert stats["n_total"] == 2
    assert stats["n_dropped"] == 1
    dropped = stats["dropped"]
    assert dropped[0]["anchor"]["scene_id"] == edge.id
    assert dropped[0]["reason"] == "insufficient_aoi_overlap"
    assert dropped[0]["anchor"]["aoi_overlap_frac"] == pytest.approx(0.04)


@pytest.mark.parametrize("valid_px, accepted", [(0, False), (4, False), (5, True), (6, True)])
def test_overlap_threshold_is_inclusive(monkeypatch, tmp_path, valid_px, accepted):
    edge, _, calls = _fake_inputs(monkeypatch, valid_px)

    kept, stats = anchors.build_anchors(_config(tmp_path))

    assert (edge.id in [a["scene_id"] for a in kept]) is accepted
    # The footprint cannot be inflated by another path/row on the same day.
    assert len(calls) == 2
    assert all(len(call["items"]) == 1 for call in calls)
    assert all(call["nodata"] == 1 and call["resampling"] == "nearest" for call in calls)
    assert stats["n_kept"] + stats["n_dropped"] == stats["n_total"]


@pytest.mark.parametrize("threshold, accepted", [(0, True), (0.03, True), (0.06, False)])
def test_overlap_threshold_is_configurable(monkeypatch, tmp_path, threshold, accepted):
    edge, _, _ = _fake_inputs(monkeypatch)

    kept, _ = anchors.build_anchors(_config(tmp_path, min_aoi_overlap_frac=threshold))

    assert (edge.id in [a["scene_id"] for a in kept]) is accepted


def test_overlap_denominator_is_aoi_not_scene_or_valid_pixels(monkeypatch, tmp_path):
    edge, _, _ = _fake_inputs(monkeypatch, valid_px=1)
    aoi = np.zeros((10, 10), dtype=bool)
    aoi.flat[:20] = True
    monkeypatch.setattr(anchors, "load_aoi_mask", lambda *args: aoi)

    kept, _ = anchors.build_anchors(_config(tmp_path))

    measured = next(a for a in kept if a["scene_id"] == edge.id)
    assert measured["aoi_valid_px"] == 1
    assert measured["aoi_total_px"] == 20
    assert measured["aoi_overlap_frac"] == pytest.approx(0.05)


def test_clouds_count_as_footprint(monkeypatch, tmp_path):
    cloudy_qa = np.full((10, 10), 3 << 8, dtype=np.uint16)
    edge, _, _ = _fake_inputs(monkeypatch, qa_override=cloudy_qa)

    kept, _ = anchors.build_anchors(_config(tmp_path))

    measured = next(a for a in kept if a["scene_id"] == edge.id)
    assert measured["aoi_overlap_frac"] == 1.0


def test_overlap_gate_precedes_existing_clear_gate_without_changing_it(monkeypatch, tmp_path):
    edge, interior, _ = _fake_inputs(monkeypatch)
    clear_calls = []

    def clear_filter(candidates, cfg, threshold):
        clear_calls.append(([a["scene_id"] for a in candidates], threshold))
        return [], candidates

    monkeypatch.setattr(anchors, "_filter_by_pixel_clear_frac", clear_filter)

    kept, stats = anchors.build_anchors(_config(tmp_path, min_clear_frac=0.2))

    assert clear_calls == [([interior.id], 0.2)]
    assert kept == []
    assert stats["n_total"] == stats["n_dropped"] == 2
    assert [(d["anchor"]["scene_id"], d["reason"]) for d in stats["dropped"]] == [
        (edge.id, "insufficient_aoi_overlap"),
        (interior.id, "insufficient_clear_fraction"),
    ]


@pytest.mark.parametrize("missing", [np.nan, np.inf, 65535.0, 1.0])
def test_nonfinite_and_fill_pixels_never_inflate_footprint(monkeypatch, tmp_path, missing):
    qa = np.full((10, 10), missing)
    qa.flat[:4] = 0
    edge, _, _ = _fake_inputs(monkeypatch, qa_override=qa)

    kept, stats = anchors.build_anchors(_config(tmp_path))

    assert edge.id not in [a["scene_id"] for a in kept]
    assert stats["dropped"][0]["anchor"]["aoi_valid_px"] == 4


def test_explicit_raster_nodata_is_not_valid_coverage(monkeypatch, tmp_path):
    edge, _, _ = _fake_inputs(monkeypatch)
    qa = np.full((10, 10), 100, dtype=np.uint16)  # even value without fill bit
    qa.flat[:4] = 0
    ds = xr.Dataset({"qa_pixel": (("y", "x"), qa)})
    ds["qa_pixel"].rio.write_nodata(100, inplace=True)
    monkeypatch.setattr(anchors, "stac_load", lambda **kwargs: ds)

    kept, stats = anchors.build_anchors(_config(tmp_path))

    assert edge.id not in [a["scene_id"] for a in kept]
    assert stats["dropped"][0]["anchor"]["aoi_valid_px"] == 4


@pytest.mark.parametrize("failure", ["load", "empty_aoi"])
def test_unavailable_overlap_has_separate_reason_and_no_bbox_fallback(
    monkeypatch, tmp_path, failure
):
    _fake_inputs(monkeypatch)
    if failure == "load":

        def fail(**kwargs):
            raise OSError("Offline fake QA load failure")

        monkeypatch.setattr(anchors, "stac_load", fail)
    else:
        monkeypatch.setattr(anchors, "load_aoi_mask", lambda *args: np.zeros((10, 10), dtype=bool))

    kept, stats = anchors.build_anchors(_config(tmp_path))

    assert kept == []
    assert stats["n_dropped"] == 2
    assert {d["reason"] for d in stats["dropped"]} == {"aoi_overlap_unavailable"}


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_overlap_threshold_fails_before_catalog_access(monkeypatch, tmp_path, threshold):
    def forbidden():
        pytest.fail("Invalid overlap config must not access the catalog")

    monkeypatch.setattr(anchors, "get_catalog", forbidden)

    with pytest.raises(ValueError, match="min_aoi_overlap_frac"):
        anchors.build_anchors(_config(tmp_path, min_aoi_overlap_frac=threshold))


def test_legacy_config_does_not_enable_overlap_gate(monkeypatch, tmp_path):
    edge, interior, calls = _fake_inputs(monkeypatch)
    cfg = _config(tmp_path)
    del cfg.landsat.anchor.min_aoi_overlap_frac

    kept, stats = anchors.build_anchors(cfg)

    assert [a["scene_id"] for a in kept] == [edge.id, interior.id]
    assert calls == []
    assert stats == {"n_total": 2, "n_kept": 2, "n_dropped": 0}


def test_old_clear_checkpoint_cannot_supply_individual_footprint(monkeypatch, tmp_path):
    import pickle

    edge, _, calls = _fake_inputs(monkeypatch)
    checkpoint = tmp_path / "checkpoints" / "anchor_filter_checkpoint.pkl"
    checkpoint.parent.mkdir()
    legacy_bytes = pickle.dumps({edge.id: (1.0, 100, 100)})
    checkpoint.write_bytes(legacy_bytes)

    kept, _ = anchors.build_anchors(_config(tmp_path))

    assert edge.id not in [a["scene_id"] for a in kept]
    assert len(calls) == 2
    assert checkpoint.read_bytes() == legacy_bytes


def test_overlap_drop_records_match_manifest_report_input(monkeypatch, tmp_path):
    from berlin_lst_downscaling.data.selection.manifest import _summarize_dropped

    _fake_inputs(monkeypatch)
    _, stats = anchors.build_anchors(_config(tmp_path))

    assert _summarize_dropped(stats["dropped"]) == {"insufficient_aoi_overlap": 1}


def test_v4_config_adds_five_percent_policy_and_isolates_outputs():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "selection"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        legacy = compose(config_name="full_2017_2026")
        updated = compose(config_name="full_2017_2026_v4")

    assert "min_aoi_overlap_frac" not in legacy.landsat.anchor
    assert updated.landsat.anchor.min_aoi_overlap_frac == 0.05
    assert updated.landsat.anchor.min_clear_frac == legacy.landsat.anchor.min_clear_frac
    assert updated.output_root != legacy.output_root
    assert updated.checkpoint_dir == f"{updated.output_root}/checkpoints"
    # Every inherited policy field remains unchanged except the explicit v4 gates.
    updated = OmegaConf.to_container(updated, resolve=True)
    legacy = OmegaConf.to_container(legacy, resolve=True)
    updated["landsat"]["anchor"].pop("min_aoi_overlap_frac")
    assert updated["sentinel2"].pop("mosaic") == {"enabled": True}
    for key in ("output_root", "checkpoint_dir"):
        updated[key] = legacy[key]
    assert updated == legacy
