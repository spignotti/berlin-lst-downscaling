"""Offline tests for the ASTER GED v3 loader and the GED diagnostic variant.

Network access is always mocked; nothing here touches CMR or LP DAAC.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
import requests
from rasterio.warp import transform as warp_transform

from berlin_lst_downscaling.analysis import aster_ged
from berlin_lst_downscaling.analysis.aster_ged import (
    HDF5_SIGNATURE,
    MALAKAR2018_L8_B10,
    AsterGedError,
    MissingEarthdataTokenError,
    cmr_granule_urls,
    download_granule,
    ged_b10_from_bands,
    ged_coefficients_label,
    ged_dn_to_emissivity,
    geotransform_from_geolocation,
    load_ged_b10_for_bbox,
    mosaic_ged_tiles,
    read_ged_b10,
)
from berlin_lst_downscaling.analysis.emissivity import (
    forward_trad,
    ged_b10_on_scene_grid,
    lst_from_aux,
    thermal_constants_for_platform,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts/validators/diagnose_emissivity.py"
BBOX = (13.08, 52.34, 13.76, 52.68)
TOKEN = "SENTINEL-TOKEN-abc123"  # noqa: S105 — synthetic sentinel
MISSING_MSG = "EARTHDATA_TOKEN is not set; ASTER GED download needs a NASA Earthdata bearer token"


def _load_script():
    spec = importlib.util.spec_from_file_location("diagnose_emissivity_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: bytes = b"",
        content_type: str = "application/x-hdf5",
        json_data: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._body = body
        self._json = json_data

    def iter_content(self, chunk_size: int = 1024):
        for i in range(0, len(self._body), 4):
            yield self._body[i : i + 4]

    def json(self) -> dict:
        assert self._json is not None
        return self._json

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


def _chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{cur!s} {cur!r}")
        cur = cur.__cause__ or cur.__context__
    return " ".join(parts)


def _signature_valid(path) -> bool:
    p = Path(path)
    return p.is_file() and p.read_bytes()[: len(HDF5_SIGNATURE)] == HDF5_SIGNATURE


# ── physics ──────────────────────────────────────────────────────────────


def test_spectral_adjustment_known_values_nan_and_no_clamp() -> None:
    e13 = np.array([0.96, np.nan, 2.0])
    e14 = np.array([0.98, 0.97, 2.0])
    out = ged_b10_from_bands(e13, e14)
    assert out.dtype == np.float64
    assert out[0] == pytest.approx(0.6820 * 0.96 + 0.2578 * 0.98 + 0.0584)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(0.6820 * 2.0 + 0.2578 * 2.0 + 0.0584)  # not clamped
    assert MALAKAR2018_L8_B10 == (0.6820, 0.2578, 0.0584)


def test_dn_to_emissivity() -> None:
    out = ged_dn_to_emissivity(np.array([1000, 0, -9999, 975], dtype=np.int16))
    np.testing.assert_allclose(out, [1.0, np.nan, np.nan, 0.975])


def test_coefficients_label() -> None:
    assert ged_coefficients_label("landsat-8") == "L8 B10 (Malakar 2018)"
    assert (
        ged_coefficients_label("landsat-9") == "L8 B10 (Malakar 2018) applied to L9 — approximation"
    )
    with pytest.raises(ValueError):
        ged_coefficients_label("landsat-7")


# ── georeferencing ───────────────────────────────────────────────────────


def test_geotransform_2d_centres() -> None:
    lat, lon = np.meshgrid(np.linspace(53, 52, 11), np.linspace(13, 14, 11), indexing="ij")
    t = geotransform_from_geolocation(lat, lon)
    assert t.a == pytest.approx(0.1)
    assert t.e == pytest.approx(-0.1)
    assert t.c == pytest.approx(13 - 0.05)
    assert t.f == pytest.approx(53 + 0.05)


def test_geotransform_ged_1000_samples_and_1d_inputs() -> None:
    t = geotransform_from_geolocation(np.linspace(53, 52, 1000), np.linspace(13, 14, 1000))
    assert t.a == pytest.approx(1 / 999)
    assert t.c == pytest.approx(13 - 0.5 / 999)
    assert t.f == pytest.approx(53 + 0.5 / 999)


def test_geotransform_rejects_irregular_and_south_up() -> None:
    lat, lon = np.meshgrid(np.linspace(53, 52, 11), np.linspace(13, 14, 11), indexing="ij")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        geotransform_from_geolocation(lat, lon + 1e-3 * rng.standard_normal(lon.shape))
    with pytest.raises(ValueError):
        geotransform_from_geolocation(lat[::-1], lon)


# ── reading and mosaicking ───────────────────────────────────────────────


def _fake_subdatasets(monkeypatch, emis: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    data = {
        "Emissivity/Mean": emis,
        "Geolocation/Latitude": lat,
        "Geolocation/Longitude": lon,
    }
    monkeypatch.setattr(aster_ged, "_read_subdataset", lambda path, name: data[name])


def test_read_ged_b10(monkeypatch) -> None:
    rows, cols = 3, 4
    emis = np.full((5, rows, cols), 950, dtype=np.int16)
    emis[3] = 970
    emis[4] = 980
    emis[3, 0, 0] = -9999
    lat, lon = np.meshgrid(np.linspace(53, 52.9, rows), np.linspace(13, 13.3, cols), indexing="ij")
    _fake_subdatasets(monkeypatch, emis, lat[None], lon[None])
    out = read_ged_b10("dummy.h5")
    expected = 0.6820 * 0.970 + 0.2578 * 0.980 + 0.0584
    assert out["array"].shape == (rows, cols)
    assert out["array"][1, 1] == pytest.approx(expected)
    assert np.isnan(out["array"][0, 0])
    assert out["crs"] == "EPSG:4326"
    assert out["transform"].a == pytest.approx(0.1)
    assert out["transform"].c == pytest.approx(13 - 0.05)
    assert out["transform"].f == pytest.approx(53 + 0.025)


def test_read_ged_b10_wrong_shape(monkeypatch) -> None:
    lat, lon = np.meshgrid(np.linspace(53, 52.9, 3), np.linspace(13, 13.3, 4), indexing="ij")
    _fake_subdatasets(monkeypatch, np.full((3, 3, 4), 950, dtype=np.int16), lat, lon)
    with pytest.raises(AsterGedError):
        read_ged_b10("dummy.h5")


def _tile(c: float, value: float) -> dict:
    return {
        "array": np.full((5, 5), value),
        "transform": rasterio.Affine(0.1, 0.0, c, 0.0, -0.1, 53.05),
        "crs": "EPSG:4326",
    }


def test_mosaic_adjacent_tiles_share_edge() -> None:
    a, b = _tile(12.95, 0.95), _tile(13.35, 0.99)
    a["array"][2, 4] = np.nan  # filled from b in the shared column
    arr, t = mosaic_ged_tiles([a, b])
    assert arr.shape == (5, 9)
    assert t.c == pytest.approx(12.95)
    assert t.f == pytest.approx(53.05)
    assert arr[0, 4] == 0.95  # overlap keeps the first finite value
    assert arr[2, 4] == 0.99
    assert arr[0, 8] == 0.99


def test_mosaic_rejects_mismatched_resolution() -> None:
    a = _tile(12.95, 0.95)
    b = _tile(13.35, 0.99)
    b["transform"] = rasterio.Affine(0.05, 0.0, 13.35, 0.0, -0.05, 53.05)
    with pytest.raises(ValueError):
        mosaic_ged_tiles([a, b])


# ── reprojection ─────────────────────────────────────────────────────────

_UTM_TRANSFORM = rasterio.Affine(30.0, 0.0, 390000.0, 0.0, -30.0, 5820000.0)


def _field_around_scene(value: float, n: int = 60):
    lons, lats = warp_transform("EPSG:32633", "EPSG:4326", [390000.0], [5820000.0])
    half = n * 0.01 / 2
    t = rasterio.Affine(0.01, 0.0, lons[0] - half, 0.0, -0.01, lats[0] + half)
    return np.full((n, n), value), t


def test_reprojection_constant_and_nan_block() -> None:
    field, t = _field_around_scene(0.97)
    out = ged_b10_on_scene_grid(field, t, "EPSG:4326", (20, 20), _UTM_TRANSFORM, "EPSG:32633")
    np.testing.assert_allclose(out, 0.97, rtol=1e-12)
    field[20:40, 20:40] = np.nan
    holed = ged_b10_on_scene_grid(field, t, "EPSG:4326", (20, 20), _UTM_TRANSFORM, "EPSG:32633")
    assert bool(np.all(np.isnan(holed)))  # excluded, never filled


# ── auth / download ──────────────────────────────────────────────────────


def test_missing_token_fails_before_network(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    session = FakeSession(response=FakeResponse())
    with pytest.raises(MissingEarthdataTokenError) as e1:
        load_ged_b10_for_bbox(BBOX, tmp_path, session=session)
    assert str(e1.value) == MISSING_MSG
    with pytest.raises(MissingEarthdataTokenError) as e2:
        download_granule("https://example.org/x/AG100.h5", tmp_path, session=session)
    assert str(e2.value) == MISSING_MSG
    assert session.calls == []


def test_blank_token_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", "   ")
    with pytest.raises(MissingEarthdataTokenError):
        aster_ged.require_earthdata_token()


def test_http_error_does_not_leak_token(monkeypatch, tmp_path, caplog, capsys) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", TOKEN)
    url = "https://example.org/x/AG100.v003.52.013.0001.h5"
    session = FakeSession(response=FakeResponse(status_code=401))
    with caplog.at_level(logging.DEBUG), pytest.raises(AsterGedError) as exc:
        download_granule(url, tmp_path, session=session)
    assert "401" in str(exc.value)
    assert TOKEN not in str(exc.value)
    assert TOKEN not in repr(exc.value)
    assert TOKEN not in _chain_text(exc.value)
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err + caplog.text
    call = session.calls[0]
    assert call["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert TOKEN not in call["url"]
    assert list(tmp_path.iterdir()) == []


def test_connection_error_is_scrubbed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", TOKEN)
    session = FakeSession(exc=requests.ConnectionError(f"boom {TOKEN}"))
    with pytest.raises(AsterGedError) as exc:
        download_granule("https://example.org/x/a.h5", tmp_path, session=session)
    assert "boom" in str(exc.value)
    assert TOKEN not in _chain_text(exc.value)


def test_html_response_leaves_no_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", TOKEN)
    session = FakeSession(
        response=FakeResponse(body=b"<html>login</html>", content_type="text/html; charset=utf-8")
    )
    with pytest.raises(AsterGedError, match="HTML"):
        download_granule("https://example.org/x/a.h5", tmp_path, session=session)
    assert list(tmp_path.iterdir()) == []


def test_html_body_without_html_header_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", TOKEN)
    session = FakeSession(response=FakeResponse(body=b"<html>login</html>"))
    with pytest.raises(AsterGedError, match="HTML"):
        download_granule("https://example.org/x/a.h5", tmp_path, session=session)
    assert list(tmp_path.iterdir()) == []


def test_valid_download_is_atomic_and_cached(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", TOKEN)
    monkeypatch.setattr(aster_ged, "_is_valid_ged_file", _signature_valid)
    body = HDF5_SIGNATURE + b"payload"
    session = FakeSession(response=FakeResponse(body=body))
    path = download_granule("https://example.org/x/a.h5", tmp_path / "cache", session=session)
    assert path.read_bytes() == body
    assert [p.name for p in path.parent.iterdir()] == ["a.h5"]
    assert len(session.calls) == 1

    monkeypatch.delenv("EARTHDATA_TOKEN")
    failing = FakeSession(exc=AssertionError("network used"))
    again = download_granule("https://example.org/x/a.h5", tmp_path / "cache", session=failing)
    assert again == path
    assert failing.calls == []


# ── CMR ──────────────────────────────────────────────────────────────────


def _feed(*entries: list[str]) -> dict:
    return {"feed": {"entry": [{"links": [{"href": h} for h in links]} for links in entries]}}


def test_cmr_filters_links_and_passes_params() -> None:
    session = FakeSession(
        response=FakeResponse(
            json_data=_feed(
                ["s3://bucket/a.h5", "https://d.example/a.h5", "https://d.example/a.h5.xml"]
            )
        )
    )
    assert cmr_granule_urls(BBOX, session=session) == ["https://d.example/a.h5"]
    params = session.calls[0]["params"]
    assert params["bounding_box"] == "13.08,52.34,13.76,52.68"
    assert params["short_name"] == "AG100"
    assert params["version"] == "003"
    assert "Authorization" not in session.calls[0].get("headers", {})


def test_cmr_two_entries_empty_and_http_error() -> None:
    two = FakeSession(
        response=FakeResponse(
            json_data=_feed(["https://d.example/a.h5"], ["https://d.example/b.h5"])
        )
    )
    assert cmr_granule_urls(BBOX, session=two) == [
        "https://d.example/a.h5",
        "https://d.example/b.h5",
    ]
    with pytest.raises(AsterGedError):
        cmr_granule_urls(BBOX, session=FakeSession(response=FakeResponse(json_data=_feed())))
    with pytest.raises(AsterGedError, match="500"):
        cmr_granule_urls(BBOX, session=FakeSession(response=FakeResponse(status_code=500)))


# ── script level ─────────────────────────────────────────────────────────


def test_main_missing_token_exits_before_planetary_computer(monkeypatch, tmp_path, capsys) -> None:
    script = _load_script()
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("Planetary Computer must not be reached")

    monkeypatch.setattr(script, "_resolve_landsat_item", _fail)
    monkeypatch.setattr(script, "pc_search", _fail)
    code = script.main(
        [
            "--aster-ged",
            "--scenes",
            "LC09_L2SP_193023_20220624_02_T1",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert MISSING_MSG in capsys.readouterr().err


def _report(scene_id: str, season: str, platform: str, n: int, mean: float, slopes) -> dict:
    stat = {"mean": mean, "median": mean, "p05": mean - 1, "p95": mean + 1, "n": n}
    ge = {
        "overall": stat,
        "by_ndvi_bin": {label: stat for label in script_labels()},
        "by_surface": {"land": stat, "water": stat},
    }
    (a_slope, g_slope) = slopes
    return {
        "scene_id": scene_id,
        "season": season,
        "platform": platform,
        "delta_stats": {"static_ged": ge},
        "static_info": {"ged_coefficients": "x", "ged_frac_invalid_in_analysis_mask": 0.1},
        "relations": {
            "landsat_ndvi_vs_a": {"slope": a_slope, "r": -0.5, "n": n},
            "landsat_ndvi_vs_static_ged": {"slope": g_slope, "r": -0.4, "n": n},
        },
    }


def script_labels() -> tuple[str, ...]:
    from berlin_lst_downscaling.analysis.emissivity import NDVI_BIN_LABELS

    return NDVI_BIN_LABELS


def test_build_ged_summary_pools_scenes() -> None:
    script = _load_script()
    ok = [
        _report("s1", "summer", "landsat-8", 100, 1.0, (-7.0, -6.0)),
        _report("s2", "summer", "landsat-9", 300, 3.0, (-5.0, -4.0)),
    ]
    summary = script.build_ged_summary(ok, {"source": "aster-ged"})
    assert summary["overall"]["n"] == 400
    assert summary["overall"]["mean"] == pytest.approx(2.5)
    assert set(summary["by_group"]) == {"summer/landsat-8", "summer/landsat-9"}
    first = summary["per_scene"][0]
    assert first["landsat"]["d_slope"] == pytest.approx(-1.0)  # (a) minus static_ged
    assert first["s2"] is None


# ── end to end ───────────────────────────────────────────────────────────


def test_static_ged_equal_to_product_emis_gives_zero_delta() -> None:
    rng = np.random.default_rng(3)
    shape = (20, 20)
    lst_true = 290.0 + 10.0 * rng.random(shape)
    atran = 0.8 + 0.1 * rng.random(shape)
    urad = 1.2 + rng.random(shape)
    drad = 1.8 + rng.random(shape)
    k1, k2 = thermal_constants_for_platform("landsat-9")
    emis = np.full(shape, 0.972)
    trad = forward_trad(lst_true, urad, drad, atran, emis, k1, k2)
    field, t = _field_around_scene(0.972)
    ged = ged_b10_on_scene_grid(field, t, "EPSG:4326", shape, _UTM_TRANSFORM, "EPSG:32633")
    lst_a = lst_from_aux(trad, urad, drad, atran, emis, k1, k2)
    lst_b = lst_from_aux(trad, urad, drad, atran, ged, k1, k2)
    np.testing.assert_allclose(lst_a - lst_b, 0.0, atol=1e-6)


def test_synthetic_self_check_includes_ged_identity() -> None:
    report = _load_script().synthetic_self_check()
    assert report["roundtrip_ok"]
    assert report["static_ged_identity_ok"]
    assert report["static_ged_identity_max_abs_diff_K"] < 1e-6
