"""Offline regressions for recorded-member, per-item Sentinel-2 mosaics."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf
from rasterio.transform import from_origin

from berlin_lst_downscaling.data.acquisition import pc_client
from berlin_lst_downscaling.data.ard import pipeline, s2_mosaic
from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.publish_guard import (
    FrozenPrefixError,
    assert_sentinel2_output_root,
)
from berlin_lst_downscaling.data.ard.s2_mosaic import (
    coverage_metrics,
    load_aoi_on_grid,
    mosaic_s2_items,
    validate_missing_fraction,
)


def _item(item_id, baseline):
    return SimpleNamespace(
        id=item_id,
        properties={
            "s2:processing_baseline": baseline,
            "datetime": "2024-06-29T10:20:00Z",
            "s2:datatake_id": f"DATATAKE_N{baseline}",
        },
        assets={},
    )


def _cfg():
    return OmegaConf.create(
        {
            "scene_date": "2024-06-29",
            "bbox": [13.08, 52.34, 13.76, 52.68],
            "target_resolution_high": 10,
            "sentinel2": {"mosaic": {"enabled": True, "max_missing_aoi_fraction": 0.005}},
            "aoi": {"mask_base": "unused-offline"},
        }
    )


def _raw(resolution, dn, scl):
    scl = np.asarray(scl, dtype=np.float32)
    height, width = scl.shape
    bands = ["B02", "B03", "B04", "B08"] if resolution == 10 else ["B11", "B12"]
    ds = xr.Dataset(
        {b: (("y", "x"), np.full(scl.shape, dn, dtype=np.float32)) for b in bands},
        coords={
            "x": 400000 + (np.arange(width) + 0.5) * resolution,
            "y": 5800000 - (np.arange(height) + 0.5) * resolution,
        },
    )
    ds["SCL"] = (("y", "x"), scl)
    return ds.rio.write_crs("EPSG:25833").rio.write_transform(
        from_origin(400000, 5800000, resolution, resolution)
    )


def test_recorded_members_resolve_every_item_sorted(monkeypatch):
    resolver = Mock(side_effect=lambda href, expected_id: _item(expected_id, "04.00"))
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolver)
    fallback = Mock(side_effect=AssertionError("recorded members must never rediscover"))
    monkeypatch.setattr(pc_client, "resolve_exact_item", fallback)
    items = pipeline._resolve_manifest_items(
        "a",
        "sentinel-2-l2a",
        {
            "item_href": "href-a",
            "s2_mosaic_group": {"solar_day": "2024-06-29", "datatake_id": "DATATAKE"},
            "s2_items": [
                {"item_id": "b", "item_href": "href-b"},
                {"item_id": "a", "item_href": "href-a"},
            ],
        },
    )
    assert [item.id for item in items] == ["a", "b"]
    assert [call.kwargs["expected_id"] for call in resolver.call_args_list] == ["a", "b"]
    fallback.assert_not_called()


def test_mixed_baseline_complementary_pipeline(monkeypatch):
    # Existing ARD fails before loading: it tries one offset for both baselines.
    items = [_item("b", "04.00"), _item("a", "03.00")]

    def load(*, items, resolution, **kwargs):
        item = items[0]
        scl = [[4, 0], [4, 0]] if item.id == "a" else [[0, 4], [0, 4]]
        if resolution == 10:
            scl = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
        return _raw(resolution, 2000 if item.id == "a" else 4000, scl), [item.id]

    monkeypatch.setattr(pipeline, "load_s2_scene", load)
    monkeypatch.setattr(pipeline, "_solar_for_scene", lambda *_: (0, 0))
    # Patch the future AOI loader without importing a module that does not exist yet.
    monkeypatch.setattr(
        pipeline, "_load_s2_mosaic_aoi", lambda target, cfg: np.ones((4, 4), bool), raising=False
    )
    result = pipeline._run_sentinel2_scene(
        "a",
        "sentinel-2-l2a",
        2024,
        contract_for_source("sentinel-2-l2a", 10),
        _cfg(),
        None,
        "offline",
        items=items,
    )
    for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
        np.testing.assert_allclose(result[band].values[:, :2], 0.2)
        np.testing.assert_allclose(result[band].values[:, 2:], 0.3)
    assert not result.flag.values.any()
    assert result.attrs["s2_reflectance"]["item_ids"] == ["a", "b"]
    metadata = result.attrs["s2_reflectance"]
    assert [member["processing_baseline"] for member in metadata["items"]] == ["03.00", "04.00"]
    assert [member["boa_add_offset"] for member in metadata["items"]] == [0, -1000]
    assert metadata["coverage"]["missing_aoi_fraction"] == 0


def _recorded():
    return {
        "acquisition_datetime": "2024-06-29T10:20:00Z",
        "item_href": "href-a",
        "s2_mosaic_group": {"solar_day": "2024-06-29", "datatake_id": "DATATAKE"},
        "s2_items": [
            {"item_id": "b", "item_href": "href-b"},
            {"item_id": "a", "item_href": "href-a"},
        ],
    }


@pytest.mark.parametrize("failure", ["network", "mismatch", "group"])
def test_recorded_member_failure_never_falls_back(monkeypatch, failure):
    def resolve(href, expected_id):
        if expected_id == "b":
            if failure == "network":
                raise RuntimeError("member unavailable")
            if failure == "group":
                item = _item(expected_id, "04.00")
                item.properties["s2:datatake_id"] = "OTHER_DATATAKE"
                return item
            return _item("wrong-member", "04.00")
        return _item(expected_id, "03.00")

    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolve)
    fallback = Mock()
    monkeypatch.setattr(pc_client, "resolve_exact_item", fallback)
    with pytest.raises((RuntimeError, ValueError), match="unavailable|mismatch"):
        pipeline._resolve_manifest_items("a", "sentinel-2-l2a", _recorded())
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "members",
    [
        None,
        [],
        [{"item_id": "a"}],
        [{"item_id": "a", "item_href": " "}],
        [{"item_id": "b", "item_href": "href-b"}],
        [{"item_id": "a", "item_href": "x"}] * 2,
    ],
)
def test_mosaic_rejects_unrecorded_or_invalid_sets(monkeypatch, members):
    resolver = Mock()
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolver)
    monkeypatch.setattr(pc_client, "resolve_exact_item", resolver)
    meta = {**_recorded(), "s2_items": members}
    with pytest.raises(ValueError, match="s2_items|duplicate"):
        pipeline._resolve_manifest_items("a", "sentinel-2-l2a", meta, require_s2_members=True)
    resolver.assert_not_called()


def test_recorded_set_requires_group(monkeypatch):
    resolver = Mock()
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolver)
    with pytest.raises(ValueError, match="s2_mosaic_group"):
        pipeline._resolve_manifest_items(
            "a", "sentinel-2-l2a", {**_recorded(), "s2_mosaic_group": None}
        )
    resolver.assert_not_called()


def test_legacy_resolution_stays_single_item(monkeypatch):
    resolve = Mock(return_value=_item("a", "04.00"))
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolve)
    assert [
        it.id
        for it in pipeline._resolve_manifest_items("a", "sentinel-2-l2a", {"item_href": "href-a"})
    ] == ["a"]


def _install_overlap(monkeypatch, *, missing=False):
    """First donor is cloudy; later clear data must never replace it."""
    calls = []

    def load(*, items, resolution, **kwargs):
        assert len(items) == 1
        item = items[0]
        calls.append((item.id, resolution))
        if item.id == "a":
            scl = np.array([[9, 0], [9, 0]], dtype=np.float32)
        else:
            scl = np.full((2, 2), 4, dtype=np.float32)
            if missing:
                scl[:, 1] = 0
        if resolution == 10:
            scl = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
        ds = _raw(resolution, 2000 if item.id == "a" else 4000, scl)
        if resolution == 10 and item.id == "a":
            ds["B03"].values[:, :2] = np.nan
        return ds, [item.id]

    monkeypatch.setattr(pipeline, "load_s2_scene", load)
    monkeypatch.setattr(pipeline, "_solar_for_scene", lambda *_: (0, 0))
    monkeypatch.setattr(pipeline, "_load_s2_mosaic_aoi", lambda *_: np.ones((4, 4), bool))
    monkeypatch.setattr(s2_mosaic, "read_bytes", lambda *_: b"offline-aoi-mask")
    return calls


def _run_mosaic(items, cfg=None):
    return pipeline._run_sentinel2_scene(
        "a",
        "sentinel-2-l2a",
        2024,
        contract_for_source("sentinel-2-l2a", 10),
        cfg or _cfg(),
        None,
        "offline",
        items=items,
    )


def test_overlap_first_footprint_donor_preserves_clouds_and_band_nans(monkeypatch):
    calls = _install_overlap(monkeypatch)
    result = _run_mosaic([_item("b", "04.00"), _item("a", "03.00")])
    assert calls == [("a", 10), ("a", 20), ("b", 10), ("b", 20)]
    for band in ("B02", "B04", "B08"):
        np.testing.assert_allclose(result[band].values[:, :2], 0.2)
    for band in ("B03", "B11", "B12"):
        assert np.isnan(result[band].values[:, :2]).all()
    for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
        np.testing.assert_allclose(result[band].values[:, 2:], 0.3)
    assert np.all(result.flag.values[:, :2] == 2)  # CLOUDY from first donor
    assert np.all(result.flag.values[:, 2:] == 0)
    assert result.attrs["s2_reflectance"]["coverage"]["aoi_covered_px"] == 16
    forward = _run_mosaic([_item("a", "03.00"), _item("b", "04.00")])
    xr.testing.assert_identical(result, forward)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), None, "bad", True])
def test_missing_fraction_validation(value):
    with pytest.raises(ValueError, match="max_missing_aoi_fraction"):
        validate_missing_fraction(value)


def test_coverage_threshold_boundary_and_empty_aoi():
    footprint = np.ones((20, 10), bool)
    footprint[0, 0] = False
    aoi = np.ones_like(footprint)
    metrics = coverage_metrics(footprint, aoi)
    assert metrics["aoi_missing_px"] == 1
    assert metrics["missing_aoi_fraction"] == 0.005
    assert metrics["coverage_passed"]
    assert not coverage_metrics(footprint, aoi, 0.004)["coverage_passed"]
    assert coverage_metrics(footprint, aoi, 1)["coverage_passed"]
    aoi[0, 0] = False
    assert coverage_metrics(footprint, aoi, 0)["coverage_passed"]
    with pytest.raises(ValueError, match="empty"):
        coverage_metrics(footprint, np.zeros_like(aoi))
    with pytest.raises(ValueError, match="matching"):
        coverage_metrics(footprint, np.ones((2, 2), bool))


@pytest.mark.parametrize(
    "failure", ["coverage", "missing-aoi", "empty-aoi", "resolve", "legacy-row"]
)
def test_scene_failure_marks_ledger_before_any_product_write(monkeypatch, failure):
    _install_overlap(monkeypatch, missing=failure == "coverage")
    if failure == "missing-aoi":
        monkeypatch.setattr(
            pipeline, "_load_s2_mosaic_aoi", Mock(side_effect=FileNotFoundError("AOI"))
        )
    elif failure == "empty-aoi":
        monkeypatch.setattr(pipeline, "_load_s2_mosaic_aoi", lambda *_: np.zeros((4, 4), bool))
    events = Mock()
    monkeypatch.setattr(pipeline, "log_event", events)
    cog = Mock(side_effect=AssertionError("COG must not be written"))
    flag = Mock(side_effect=AssertionError("flag must not be written"))
    product = Mock(side_effect=AssertionError("STAC/provenance must not be written"))
    monkeypatch.setattr(pipeline, "write_cog_atomic", cog)
    monkeypatch.setattr(pipeline, "write_flag_cog_atomic", flag)
    monkeypatch.setattr(pipeline, "finalize_ard_product", product)
    resolver = Mock(
        side_effect=RuntimeError("member failed")
        if failure == "resolve"
        else lambda href, expected_id: _item(
            expected_id, "03.00" if expected_id == "a" else "04.00"
        )
    )
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolver)
    ledger = Mock()
    ledger.get.return_value = None
    pipeline._run_scene(
        "a",
        "sentinel-2-l2a",
        2024,
        contract_for_source("sentinel-2-l2a", 10),
        _cfg(),
        ledger,
        "offline",
        items=[_item("b", "04.00"), _item("a", "03.00")],
        manifest_meta=(
            _recorded()
            if failure == "resolve"
            else {"item_href": "href-a"}
            if failure == "legacy-row"
            else _recorded()
        ),
    )
    ledger.begin_attempt.assert_called_once()
    failed = ledger.upsert.call_args.args[0]
    assert failed.status == "failed"
    assert failed.last_error
    cog.assert_not_called()
    flag.assert_not_called()
    product.assert_not_called()
    if failure != "legacy-row":
        assert failed.schema_hash.startswith("s2-mosaic-v10:sha256:")
        assert failed.schema_hash == ledger.begin_attempt.call_args.args[0].schema_hash
        assert failed.schema_version == 10
    if failure not in ("resolve", "legacy-row"):
        evidence = [
            call.kwargs for call in events.call_args_list if call.args[2] == "s2_mosaic_prewrite"
        ]
        assert len(evidence) == 1
        assert evidence[0]["provenance"]["item_ids"] == ["a", "b"]
        if failure == "coverage":
            assert evidence[0]["provenance"]["coverage"]["missing_aoi_fraction"] == 0.5
            assert evidence[0]["coverage_passed"] is False


def test_aoi_loader_alignment_and_empty_fail_closed():
    from rasterio.io import MemoryFile

    target = _raw(10, 2000, np.ones((4, 4)))["B02"]
    for value in (0, 1):
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs="EPSG:25833",
                transform=from_origin(400000, 5800000, 20, 20),
            ) as dst:
                dst.write(np.full((2, 2), value, dtype=np.uint8), 1)
            if value:
                np.testing.assert_array_equal(
                    load_aoi_on_grid(memory.name, target), np.ones((4, 4), bool)
                )
            else:
                with pytest.raises(ValueError, match="empty"):
                    load_aoi_on_grid(memory.name, target)
    with pytest.raises(Exception, match="No such file|does not exist"):
        load_aoi_on_grid("/vsimem/s2-mosaic-nonexistent.tif", target)


def test_schema10_explicit_and_v9_root_guard_before_ledger(monkeypatch):
    assert contract_for_source("sentinel-2-l2a").schema_version == 9
    assert contract_for_source("sentinel-2-l2a", 10).schema_version == 10
    assert contract_for_source("landsat-c2-l2").schema_version == 7
    with pytest.raises(ValueError, match="Unsupported"):
        contract_for_source("landsat-c2-l2", 10)
    cfg = _cfg()
    cfg.sentinel2.schema_version = 10
    cfg.sources = ["sentinel-2-l2a"]
    cfg.output_root = "gs://berlin-lst-training-data/ard/full/2017-2026-s2-harmonized-v9/"
    ledger_open = Mock(side_effect=AssertionError("must guard before opening ledger"))
    monkeypatch.setattr(pipeline.Ledger, "open", ledger_open)
    with pytest.raises(FrozenPrefixError, match="schema-9"):
        pipeline.run(cfg)
    ledger_open.assert_not_called()
    # Legacy schema-9 config keeps its published output path.
    assert_sentinel2_output_root(cfg.output_root)
    assert_sentinel2_output_root(
        "gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10", schema_version=10
    )


@pytest.mark.parametrize("schema_version,enabled", [(9, True), (10, False)])
def test_mosaic_schema_switches_must_agree_before_ledger(monkeypatch, schema_version, enabled):
    cfg = _cfg()
    cfg.sentinel2.schema_version = schema_version
    cfg.sentinel2.mosaic.enabled = enabled
    cfg.sources = ["sentinel-2-l2a"]
    cfg.output_root = "unused"
    ledger_open = Mock()
    monkeypatch.setattr(pipeline.Ledger, "open", ledger_open)
    with pytest.raises(ValueError, match="must be set together"):
        pipeline.run(cfg)
    ledger_open.assert_not_called()


def test_mosaic_rejects_mismatched_grids_and_duplicate_ids(monkeypatch):
    _install_overlap(monkeypatch)
    masked = _run_mosaic([_item("a", "03.00"), _item("b", "04.00")])
    footprint = np.ones((4, 4), bool)
    with pytest.raises(ValueError, match="mismatched grid"):
        mosaic_s2_items(
            [("a", masked, footprint), ("b", masked.assign_coords(x=masked.x + 1), footprint)]
        )
    with pytest.raises(ValueError, match="duplicate"):
        mosaic_s2_items([("a", masked, footprint), ("a", masked, footprint)])


def test_uncovered_pixels_remain_fill_and_nan(monkeypatch):
    _install_overlap(monkeypatch, missing=True)
    cfg = _cfg()
    cfg.sentinel2.mosaic.max_missing_aoi_fraction = 1
    result = _run_mosaic([_item("a", "03.00"), _item("b", "04.00")], cfg)
    assert np.all(result.flag.values[:, 2:] == 1)
    for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
        assert np.isnan(result[band].values[:, 2:]).all()
    assert result.attrs["s2_reflectance"]["coverage"]["missing_aoi_fraction"] == 0.5


def test_singleton_item_times_do_not_change_spatial_donor_selection(monkeypatch):
    _install_overlap(monkeypatch)
    load = pipeline.load_s2_scene

    def load_with_time(**kwargs):
        ds, ids = load(**kwargs)
        time = "2024-06-29T10:20:00" if ids == ["a"] else "2024-06-29T10:20:01"
        return ds.expand_dims(time=[np.datetime64(time)]), ids

    monkeypatch.setattr(pipeline, "load_s2_scene", load_with_time)
    result = _run_mosaic([_item("b", "04.00"), _item("a", "03.00")])
    assert result.B02.dims == ("time", "y", "x")
    np.testing.assert_allclose(result.B02.values[0, :, :2], 0.2)
    np.testing.assert_allclose(result.B02.values[0, :, 2:], 0.3)


def test_success_logs_coverage_before_writes_and_passes_provenance(monkeypatch):
    _install_overlap(monkeypatch)
    cfg = _cfg()
    cfg.output_root = "unused-offline-output"
    cfg.aoi.min_overlap_px = 50
    events = []
    monkeypatch.setattr(pipeline, "log_event", lambda *args, **kwargs: events.append(args[2]))

    def write(*args, **kwargs):
        assert "s2_mosaic_prewrite" in events
        assert args[2].schema_version == 10

    data_writer = Mock(side_effect=write)
    monkeypatch.setattr(pipeline, "write_cog_atomic", data_writer)
    monkeypatch.setattr(pipeline, "write_flag_cog_atomic", Mock())
    monkeypatch.setattr(pipeline, "validate_cog", Mock(return_value=SimpleNamespace(ok=True)))
    monkeypatch.setattr(pipeline, "validate_flag_cog", Mock(return_value=SimpleNamespace(ok=True)))
    monkeypatch.setattr(pipeline, "compute_aoi_metrics", Mock(return_value={}))
    monkeypatch.setattr(
        pc_client,
        "resolve_item_from_href",
        Mock(
            side_effect=lambda href, expected_id: _item(
                expected_id, "03.00" if expected_id == "a" else "04.00"
            )
        ),
    )
    product = Mock()
    monkeypatch.setattr(pipeline, "finalize_ard_product", product)
    ledger = Mock()
    ledger.get.return_value = None
    pipeline._run_scene(
        "a",
        "sentinel-2-l2a",
        2024,
        contract_for_source("sentinel-2-l2a", 10),
        cfg,
        ledger,
        "offline",
        items=[_item("b", "04.00"), _item("a", "03.00")],
        manifest_meta=_recorded(),
    )
    data_writer.assert_called_once()
    product.assert_called_once()
    metadata = product.call_args.kwargs["source_metadata"]
    assert metadata["coverage"]["coverage_passed"] is True
    assert metadata["s2_mosaic_group"] == _recorded()["s2_mosaic_group"]
    done = ledger.upsert.call_args.args[0]
    assert done.status == "done"
    assert done.schema_version == 10
    assert done.schema_hash.startswith("s2-mosaic-v10:sha256:")
    assert done.schema_hash == metadata["input_contract_hash"]
    assert done.schema_hash == ledger.begin_attempt.call_args.args[0].schema_hash
    assert metadata["input_contract"]["s2_items"] == sorted(
        _recorded()["s2_items"], key=lambda m: m["item_id"]
    )


def test_manifest_carries_recorded_members_and_group_into_scene(monkeypatch):
    import pyarrow as pa

    from berlin_lst_downscaling.data.io import storage
    from berlin_lst_downscaling.data.selection import validate

    row = {
        "scene_id": "a",
        "source": "sentinel-2-l2a",
        "year": 2024,
        "acquisition_datetime": "2024-06-29T10:20:00Z",
        **_recorded(),
    }
    table = pa.Table.from_pylist([row])
    monkeypatch.setattr(s2_mosaic, "read_bytes", lambda *_: b"offline-aoi-mask")
    monkeypatch.setattr(storage, "exists", lambda *_: True)
    monkeypatch.setattr(
        validate,
        "load_bundle",
        Mock(
            return_value=(
                SimpleNamespace(manifest_table=table),
                SimpleNamespace(ok=True),
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "reconcile",
        lambda *_args, **_kwargs: [
            ("a", "sentinel-2-l2a", 2024, "new"),
        ],
    )
    run_scene = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    cfg = _cfg()
    cfg.manifest_uri = "unused-offline-manifest"
    cfg.max_scene_attempts = 3
    pipeline._process_manifest(
        "sentinel-2-l2a",
        contract_for_source("sentinel-2-l2a", 10),
        cfg,
        Mock(),
        "offline",
    )
    meta = run_scene.call_args.kwargs["manifest_meta"]
    assert meta["s2_items"] == row["s2_items"]
    assert meta["s2_mosaic_group"] == row["s2_mosaic_group"]


@pytest.mark.parametrize(
    "suffix",
    ["2017-2026-cutoff-20260717T235959Z", "2017-2026-qa-20261019", "2017-2026-qa-avg-20261019"],
)
@pytest.mark.parametrize("version", [9, 10])
def test_all_schema8_roots_remain_protected(suffix, version):
    with pytest.raises(FrozenPrefixError):
        assert_sentinel2_output_root(
            f"gs://berlin-lst-training-data/ard/full/{suffix}/", schema_version=version
        )


def _stub_manifest(monkeypatch, cfg, meta, *, version=4):
    import pyarrow as pa

    from berlin_lst_downscaling.data.io import storage
    from berlin_lst_downscaling.data.selection import validate

    row = {
        "scene_id": "a",
        "source": "sentinel-2-l2a",
        "year": 2024,
        "acquisition_datetime": "2024-06-29T10:20:00Z",
        **meta,
    }
    table = pa.Table.from_pylist([row]).replace_schema_metadata(
        {b"schema_version": str(version).encode()}
    )
    monkeypatch.setattr(storage, "exists", lambda *_: True)
    monkeypatch.setattr(s2_mosaic, "read_bytes", lambda *_: b"offline-aoi-mask")
    monkeypatch.setattr(
        validate,
        "load_bundle",
        Mock(
            return_value=(
                SimpleNamespace(manifest_table=table),
                SimpleNamespace(ok=True),
            )
        ),
    )
    cfg.manifest_uri = "offline-manifest"
    cfg.max_scene_attempts = 3


@pytest.mark.parametrize("enabled", [False, None])
def test_v4_rejected_by_schema9_before_reconcile_load_or_write(monkeypatch, enabled):
    cfg = _cfg()
    if enabled is None:
        del cfg.sentinel2
    else:
        cfg.sentinel2.mosaic.enabled = enabled
    _stub_manifest(monkeypatch, cfg, _recorded())
    reconcile = Mock(return_value=[])
    monkeypatch.setattr(pipeline, "reconcile", reconcile)
    run_scene = Mock()
    load = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    monkeypatch.setattr(pipeline, "load_s2_scene", load)
    ledger = Mock()
    with pytest.raises(ValueError, match="v4.*schema.10|schema.10.*v4"):
        pipeline._process_manifest(
            "sentinel-2-l2a",
            contract_for_source("sentinel-2-l2a"),
            cfg,
            ledger,
            "offline",
        )
    reconcile.assert_not_called()
    run_scene.assert_not_called()
    load.assert_not_called()
    ledger.begin_attempt.assert_not_called()
    ledger.upsert.assert_not_called()


def _complete_scene_for_resume(monkeypatch, cfg):
    _install_overlap(monkeypatch)
    cfg.output_root = "offline-products"
    cfg.aoi.min_overlap_px = 50
    for name in ("write_cog_atomic", "write_flag_cog_atomic", "finalize_ard_product"):
        monkeypatch.setattr(pipeline, name, Mock())
    for name in ("validate_cog", "validate_flag_cog"):
        monkeypatch.setattr(pipeline, name, Mock(return_value=SimpleNamespace(ok=True)))
    monkeypatch.setattr(pipeline, "compute_aoi_metrics", Mock(return_value={}))
    monkeypatch.setattr(
        pc_client,
        "resolve_item_from_href",
        Mock(
            side_effect=lambda href, expected_id: _item(expected_id, "04.00"),
        ),
    )
    ledger = Mock()
    ledger.get.return_value = None
    pipeline._run_scene(
        "a",
        "sentinel-2-l2a",
        2024,
        contract_for_source("sentinel-2-l2a", 10),
        cfg,
        ledger,
        "offline",
        manifest_meta=_recorded(),
    )
    done = ledger.upsert.call_args.args[0]
    assert done.status == "done"
    return done


@pytest.mark.parametrize(
    "change",
    [
        "membership",
        "threshold",
        "href",
        "group",
        "mask",
        "bbox",
        "crs",
        "grid",
        "s2-policy",
        "donor-rule",
        "aoi-uri",
        "aoi-bytes",
        "missing-hash",
        "unbound-hash",
    ],
)
def test_completed_mosaic_input_change_rejected_before_reconcile(monkeypatch, change):
    from berlin_lst_downscaling.data.ard import idempotency

    cfg = _cfg()
    done = _complete_scene_for_resume(monkeypatch, cfg)
    meta = _recorded()
    if change == "membership":
        meta["s2_items"].append({"item_id": "c", "item_href": "href-c"})
    elif change == "threshold":
        cfg.sentinel2.mosaic.max_missing_aoi_fraction = 0.001
    elif change == "href":
        meta["s2_items"][0]["item_href"] = "changed-href-b"
    elif change == "group":
        meta["s2_mosaic_group"]["datatake_id"] = "OTHER_DATATAKE"
    elif change == "mask":
        cfg.cloud_base_height_m = 2000
    elif change == "bbox":
        cfg.bbox[0] = 13.09
    elif change == "crs":
        cfg.target_crs = "EPSG:32633"
    elif change == "s2-policy":
        cfg.sentinel2.mosaic.additional_policy = "different"
    elif change == "donor-rule":
        monkeypatch.setattr(s2_mosaic, "DONOR_RULE", "different-donor-rule")
    elif change == "grid":
        grids = {res: s2_mosaic.canon_grid_for_resolution(res) for res in (10, 20)}

        def shifted_grid(res):
            transform = list(grids[res].transform)
            transform[2] += 10
            return SimpleNamespace(crs=grids[res].crs, shape=grids[res].shape, transform=transform)

        monkeypatch.setattr(s2_mosaic, "canon_grid_for_resolution", shifted_grid)
    elif change == "aoi-uri":
        cfg.aoi.mask_base = "different-aoi-root"
    elif change == "missing-hash":
        done.schema_hash = None
    elif change == "unbound-hash":
        done.schema_hash = "10"
    _stub_manifest(monkeypatch, cfg, meta)
    if change == "aoi-bytes":
        monkeypatch.setattr(s2_mosaic, "read_bytes", lambda *_: b"replacement-aoi-at-same-uri")
    ledger = Mock()
    ledger.get.return_value = done
    monkeypatch.setattr(idempotency, "_files_exist", lambda *_: True)
    reconcile = Mock(wraps=pipeline.reconcile)
    monkeypatch.setattr(pipeline, "reconcile", reconcile)
    run_scene = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    with pytest.raises(RuntimeError, match="input contract"):
        pipeline._process_manifest(
            "sentinel-2-l2a",
            contract_for_source("sentinel-2-l2a", 10),
            cfg,
            ledger,
            "offline",
        )
    reconcile.assert_not_called()
    run_scene.assert_not_called()
    ledger.begin_attempt.assert_not_called()
    ledger.upsert.assert_not_called()
    assert done.status == "done"  # Completed canonical product is not invalidated or overwritten.


def test_identical_v10_contract_resumes_without_resolution_or_writes(monkeypatch):
    from berlin_lst_downscaling.data.ard import idempotency

    cfg = _cfg()
    done = _complete_scene_for_resume(monkeypatch, cfg)
    meta = _recorded()
    meta["s2_items"].reverse()  # Input ordering does not change the sorted donor contract.
    _stub_manifest(monkeypatch, cfg, meta)
    ledger = Mock()
    ledger.get.return_value = done
    monkeypatch.setattr(idempotency, "_files_exist", lambda *_: True)
    run_scene = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    resolve = Mock(side_effect=AssertionError("unchanged inputs must not resolve again"))
    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolve)
    pipeline._process_manifest(
        "sentinel-2-l2a",
        contract_for_source("sentinel-2-l2a", 10),
        cfg,
        ledger,
        "resume",
    )
    run_scene.assert_not_called()
    resolve.assert_not_called()
    ledger.begin_attempt.assert_not_called()
    ledger.upsert.assert_not_called()


@pytest.mark.parametrize("version,members", [(3, None), (4, None), (4, [])])
def test_v10_requires_recorded_v4_before_reconcile(monkeypatch, version, members):
    cfg = _cfg()
    meta = {"item_href": "href-a"} if version == 3 else {**_recorded(), "s2_items": members}
    _stub_manifest(monkeypatch, cfg, meta, version=version)
    reconcile = Mock()
    read_aoi = Mock(side_effect=AssertionError("membership must be validated before AOI reads"))
    monkeypatch.setattr(pipeline, "reconcile", reconcile)
    monkeypatch.setattr(s2_mosaic, "read_bytes", read_aoi)
    with pytest.raises(ValueError, match="v4"):
        pipeline._process_manifest(
            "sentinel-2-l2a",
            contract_for_source("sentinel-2-l2a", 10),
            cfg,
            Mock(),
            "offline",
        )
    reconcile.assert_not_called()
    read_aoi.assert_not_called()


def test_legacy_v3_schema9_keeps_existing_reconciliation(monkeypatch):
    from berlin_lst_downscaling.data.ard import idempotency
    from berlin_lst_downscaling.data.ard.ledger import LedgerRow

    cfg = _cfg()
    cfg.sentinel2.mosaic.enabled = False
    _stub_manifest(monkeypatch, cfg, {"item_href": "href-a"}, version=3)
    read_aoi = Mock(side_effect=AssertionError("legacy resume must not hash AOI"))
    monkeypatch.setattr(s2_mosaic, "read_bytes", read_aoi)
    ledger = Mock()
    ledger.get.return_value = LedgerRow(
        scene_id="a",
        source="sentinel-2-l2a",
        year=2024,
        status="done",
        schema_version=9,
        schema_hash=None,
    )
    monkeypatch.setattr(idempotency, "_files_exist", lambda *_: True)
    reconcile = Mock(wraps=pipeline.reconcile)
    monkeypatch.setattr(pipeline, "reconcile", reconcile)
    run_scene = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    pipeline._process_manifest(
        "sentinel-2-l2a",
        contract_for_source("sentinel-2-l2a"),
        cfg,
        ledger,
        "offline",
    )
    reconcile.assert_called_once()
    run_scene.assert_not_called()
    read_aoi.assert_not_called()
    ledger.upsert.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "exporting", "pending", "exhausted"])
def test_unbound_existing_v10_cannot_resume_even_when_unfinished(monkeypatch, status):
    from berlin_lst_downscaling.data.ard.ledger import LedgerRow

    cfg = _cfg()
    _stub_manifest(monkeypatch, cfg, _recorded())
    ledger = Mock()
    ledger.get.return_value = LedgerRow(
        scene_id="a",
        source="sentinel-2-l2a",
        year=2024,
        status=status,
        schema_version=10,
        schema_hash=None,
    )
    reconcile = Mock()
    monkeypatch.setattr(pipeline, "reconcile", reconcile)
    with pytest.raises(RuntimeError, match="input contract"):
        pipeline._process_manifest(
            "sentinel-2-l2a",
            contract_for_source("sentinel-2-l2a", 10),
            cfg,
            ledger,
            "offline",
        )
    reconcile.assert_not_called()
    ledger.begin_attempt.assert_not_called()
    ledger.upsert.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "exporting", "done"])
def test_bound_v10_contract_can_retry_or_repair_missing_files(monkeypatch, status):
    from berlin_lst_downscaling.data.ard import idempotency

    cfg = _cfg()
    existing = _complete_scene_for_resume(monkeypatch, cfg)
    existing.status = status
    existing.attempts = 1
    _stub_manifest(monkeypatch, cfg, _recorded())
    ledger = Mock()
    ledger.get.return_value = existing
    monkeypatch.setattr(idempotency, "_files_exist", lambda *_: False)
    run_scene = Mock()
    monkeypatch.setattr(pipeline, "_run_scene", run_scene)
    pipeline._process_manifest(
        "sentinel-2-l2a",
        contract_for_source("sentinel-2-l2a", 10),
        cfg,
        ledger,
        "resume",
    )
    run_scene.assert_called_once()
    payload = run_scene.call_args.kwargs["s2_scene_contract"]
    assert s2_mosaic.s2_input_contract_hash(payload) == existing.schema_hash
    ledger.upsert.assert_not_called()


def test_direct_scene_call_preserves_completed_mosaic_on_contract_mismatch(monkeypatch):
    cfg = _cfg()
    done = _complete_scene_for_resume(monkeypatch, cfg)
    cfg.sentinel2.mosaic.max_missing_aoi_fraction = 0.001
    ledger = Mock()
    ledger.get.return_value = done
    writer = Mock()
    monkeypatch.setattr(pipeline, "write_cog_atomic", writer)
    with pytest.raises(RuntimeError, match="input contract"):
        pipeline._run_scene(
            "a",
            "sentinel-2-l2a",
            2024,
            contract_for_source("sentinel-2-l2a", 10),
            cfg,
            ledger,
            "direct",
            manifest_meta=_recorded(),
        )
    writer.assert_not_called()
    ledger.begin_attempt.assert_not_called()
    ledger.upsert.assert_not_called()
    assert done.status == "done"
