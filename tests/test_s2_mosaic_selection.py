"""Offline regressions for discovery and the selection/ARD membership boundary."""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.selection import clear_frac, s2_search
from berlin_lst_downscaling.data.selection.schema import policy_fingerprint

BBOX = (13.0, 52.0, 14.0, 53.0)
DT = datetime(2023, 6, 1, 10, tzinfo=UTC)


def item(item_id, *, day="2023-06-01", datatake="GS2A_20230601T100000_001_N05.00"):
    return SimpleNamespace(
        id=item_id,
        properties={"datetime": f"{day}T10:00:00Z", "s2:datatake_id": datatake},
        geometry={
            "type": "Polygon",
            "coordinates": [[[13, 52], [14, 52], [14, 53], [13, 53], [13, 52]]],
        },
        get_self_href=lambda: f"https://example.test/items/{item_id}",
    )


def config(mosaic=True):
    return OmegaConf.create(
        {
            "bbox": BBOX,
            "sentinel2": {
                "collection": "sentinel-2-l2a",
                "window_days": 3,
                "mosaic": {"enabled": mosaic},
            },
            "aoi": {"mask_base": "/offline"},
        }
    )


class Catalog:
    def __init__(self, items):
        self.members = items
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(items=lambda: iter(self.members))


@pytest.mark.parametrize("reverse", [False, True])
def test_scoring_records_exact_same_day_datatake_members(monkeypatch, reverse):
    """Adjacent days and same-day separate passes must never donate pixels."""
    a = item("a")
    b = item("b", datatake="GS2A_20230601T100000_001_N03.00")
    a.properties["s2:processing_baseline"] = "05.00"
    b.properties["s2:processing_baseline"] = "03.00"
    other_pass = item("c", datatake="GS2B_20230601T110000_002_N05.00")
    previous_day = item("d", day="2023-05-31")
    search_items = [b, previous_day, other_pass, a]
    catalog = Catalog(list(reversed(search_items)) if reverse else search_items)
    monkeypatch.setattr(s2_search, "get_catalog", lambda: catalog)
    scored = []

    def score(**kwargs):
        scored.append([i.id for i in kwargs["s2_items"]])
        return 0.5, {"aoi_px": 4, "l8_clear_px": 4, "s2_clear_px": 2, "intersect_px": 2}

    monkeypatch.setattr(clear_frac, "compute_clear_frac_with_counts", score)
    candidates = s2_search.match_s2_candidates_with_clear_frac(
        {"scene_id": "landsat", "datetime": DT, "date": "2023-06-01"}, [object()], config()
    )
    by_id = {c["scene_id"]: c for c in candidates}
    assert by_id["a"].get("s2_items") == [
        {"item_id": "a", "item_href": "https://example.test/items/a"},
        {"item_id": "b", "item_href": "https://example.test/items/b"},
    ]
    assert "b" not in by_id
    assert sorted(scored) == [["a", "b"], ["c"], ["d"]]


def test_overlap_policy_affects_hash_only_when_configured():
    legacy = config(False)
    assert policy_fingerprint(legacy) == "ea375050b1b4c607"  # captured before the fix
    changed = OmegaConf.create(OmegaConf.to_container(legacy))
    changed.landsat = {"anchor": {"min_aoi_overlap_frac": 0.05}}
    assert policy_fingerprint(changed) != policy_fingerprint(legacy)


def test_mosaic_policy_hash_is_additive():
    legacy = config(False)
    no_key = config(False)
    del no_key.sentinel2["mosaic"]
    assert policy_fingerprint(legacy) == policy_fingerprint(no_key)
    assert policy_fingerprint(config()) != policy_fingerprint(legacy)
    changed = config(False)
    changed.landsat = {"anchor": {"min_aoi_overlap_frac": 0.05}}
    previous = policy_fingerprint(changed)
    changed.landsat.anchor.min_aoi_overlap_frac = 0.1
    assert policy_fingerprint(changed) != previous


@pytest.mark.parametrize("first_gap,expected_clear", [(0, 2), (np.nan, 2), (1, 1)])
def test_selection_scl_first_footprint_valid_donor(monkeypatch, first_gap, expected_clear):
    """Cloud in the first footprint beats clear in a later overlapping tile."""
    a, b = item("a"), item("b")
    loads = []

    def load(**kwargs):
        loads.append(kwargs)
        band = kwargs["bands"][0]
        values = {"a": [9, first_gap, 4, 0], "b": [4, 4, 9, 0]}
        raw = [0, 0, 0, 1] if band == "qa_pixel" else values[kwargs["items"][0].id]
        return xr.Dataset(
            {band: (("time", "y", "x"), np.array(raw).reshape(1, 2, 2))},
            coords={"time": [np.datetime64("2023-06-01" if band == "qa_pixel" else "2023-05-31")]},
        )

    monkeypatch.setattr(clear_frac, "stac_load", load)
    monkeypatch.setattr(clear_frac, "canon_grid_10m", lambda: None)
    monkeypatch.setattr(clear_frac, "load_aoi_mask", lambda *args: np.ones((2, 2), bool))
    cf, counts = clear_frac.compute_clear_frac_with_counts(
        [object()], [b, a], BBOX, anchor_dt=DT, s2_mosaic=True
    )
    assert cf == pytest.approx(expected_clear / 3)
    assert counts["l8_clear_px"] == 3
    assert counts["s2_clear_px"] == expected_clear
    assert [[i.id for i in call["items"]] for call in loads[1:]] == [["a"], ["b"]]
    assert all(call["resampling"] == "nearest" and call["nodata"] == 0 for call in loads[1:])


def test_missing_datatake_fails_closed(monkeypatch):
    from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import discover_s2_mosaic_items

    seed = item("a")
    del seed.properties["s2:datatake_id"]
    with pytest.raises(ValueError, match="datatake"):
        discover_s2_mosaic_items(seed, bbox=BBOX, catalog=Catalog([seed]))


@pytest.mark.parametrize("reverse", [False, True])
def test_discovery_all_intersections_sorted_and_seed_retained(reverse):
    from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import (
        discover_s2_mosaic_items,
    )

    seed = item("middle")
    # Deliberately omit the seed from search and include more than two tiles.
    members = [item("z"), item("a"), item("b"), item("c")]
    outside = item("outside")
    outside.geometry = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
    }
    members.extend([outside, item("other", datatake="GS2B_other_N05.00")])
    if reverse:
        members.reverse()
    catalog = Catalog(members)
    found = discover_s2_mosaic_items(seed, bbox=BBOX, catalog=catalog)
    assert [i.id for i in found] == ["a", "b", "c", "middle", "z"]
    assert found[3] is seed
    assert "max_items" not in catalog.calls[0]


def test_group_uses_solar_day_and_baseline_independent_datatake():
    from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import s2_mosaic_group

    a, b = item("a"), item("b", datatake="GS2A_20230601T100000_001_N03.00")
    a.properties["datetime"] = "2023-06-01T23:30:00Z"
    b.properties["datetime"] = "2023-06-02T00:01:00Z"
    assert (
        s2_mosaic_group(a, bbox=BBOX)
        == s2_mosaic_group(b, bbox=BBOX)
        == {"solar_day": "2023-06-02", "datatake_id": "GS2A_20230601T100000_001"}
    )


def test_discovery_errors_propagate_without_single_tile_fallback():
    from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import discover_s2_mosaic_items

    def broken_search(**kwargs):
        raise RuntimeError("offline search failure")

    with pytest.raises(RuntimeError, match="offline search failure"):
        discover_s2_mosaic_items(
            item("a"), bbox=BBOX, catalog=SimpleNamespace(search=broken_search)
        )


def test_scoring_exact_refs_does_not_discover_again(monkeypatch):
    from berlin_lst_downscaling.data.acquisition import pc_client
    from berlin_lst_downscaling.data.acquisition.s2_mosaic_items import (
        s2_item_membership,
        s2_mosaic_group,
    )

    a, b = item("a"), item("b")
    candidate = {
        "scene_id": "a",
        "datetime": DT,
        "s2_items": s2_item_membership([a, b]),
        "s2_mosaic_group": s2_mosaic_group(a, bbox=BBOX),
    }
    monkeypatch.setattr(s2_search, "match_s2_candidates", lambda *args: [candidate])
    resolved = []

    def resolve(href, expected_id):
        resolved.append((href, expected_id))
        return {"a": a, "b": b}[expected_id]

    def no_search(*args, **kwargs):
        pytest.fail("Scoring must not rediscover items")

    def score(**kwargs):
        assert kwargs["s2_items"] == [a, b]
        return 0.5, {"aoi_px": 4, "intersect_px": 2, "l8_clear_px": 4}

    monkeypatch.setattr(pc_client, "resolve_item_from_href", resolve)
    monkeypatch.setattr(s2_search, "get_catalog", no_search)
    monkeypatch.setattr(s2_search, "discover_s2_mosaic_items", no_search)
    monkeypatch.setattr(clear_frac, "compute_clear_frac_with_counts", score)
    result = s2_search.match_s2_candidates_with_clear_frac({"datetime": DT}, [], config())
    assert result[0]["clear_frac"] == 0.5
    assert [sid for _, sid in resolved] == ["a", "b"]


def test_legacy_metadata_search_preserves_individual_tiles(monkeypatch):
    a, b = item("a"), item("b")
    monkeypatch.setattr(s2_search, "get_catalog", lambda: Catalog([b, a]))
    result = s2_search.match_s2_candidates({"datetime": DT}, config(False))
    assert [c["scene_id"] for c in result] == ["b", "a"]
    assert all("s2_items" not in c for c in result)


def test_cutoff_never_trims_group_membership(monkeypatch):
    a, b = item("a"), item("b")
    b.properties["datetime"] = "2023-06-01T11:00:00Z"
    monkeypatch.setattr(s2_search, "get_catalog", lambda: Catalog([b, a]))
    cfg = config()
    cfg.cutoff_utc = "2023-06-01T10:30:00Z"
    assert s2_search.match_s2_candidates({"datetime": DT}, cfg) == []


def test_envelope_only_seed_is_skipped_not_fatal(monkeypatch):
    """A bbox-only hit whose geometry misses the AOI must not abort the search."""
    envelope = item("envelope")
    envelope.geometry = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
    }
    a, b = item("a"), item("b")
    monkeypatch.setattr(s2_search, "get_catalog", lambda: Catalog([envelope, a, b]))
    candidates = s2_search.match_s2_candidates({"datetime": DT}, config())
    assert [c["scene_id"] for c in candidates] == ["a"]
    assert candidates[0]["s2_items"] == [
        {"item_id": "a", "item_href": "https://example.test/items/a"},
        {"item_id": "b", "item_href": "https://example.test/items/b"},
    ]


def test_only_envelope_seeds_yield_no_candidates(monkeypatch):
    envelope = item("envelope")
    envelope.geometry = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
    }
    monkeypatch.setattr(s2_search, "get_catalog", lambda: Catalog([envelope]))
    assert s2_search.match_s2_candidates({"datetime": DT}, config()) == []
