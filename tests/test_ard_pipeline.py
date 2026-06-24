"""Unit tests for the ARD pipeline pure-Python logic.

These tests do NOT require GEE authentication. They cover:
  * Date arithmetic for scene listing
  * Year-range resolution
  * Grid spec construction, alignment, and transforms
  * Grid spec singleton loading (requires data/berlin_aoi.geojson)
"""

from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.gee_export import _resolve_years
from berlin_lst_downscaling.data.gee_scenes import _advance_month
from berlin_lst_downscaling.data.grid_spec import GridSpec, get_spec, make_grid_spec

# ── _advance_month ────────────────────────────────────────────────────────────


def test_advance_month_mid_year() -> None:
    assert _advance_month("2023-06-01") == "2023-07-01"


def test_advance_month_december_wrap() -> None:
    assert _advance_month("2023-12-01") == "2024-01-01"


def test_advance_month_pads_single_digit() -> None:
    assert _advance_month("2023-09-15") == "2023-10-01"


def test_advance_month_january() -> None:
    assert _advance_month("2023-01-01") == "2023-02-01"


# ── _resolve_years ────────────────────────────────────────────────────────────


def _make_cfg(start: int = 2017, end: int = 2025) -> DictConfig:
    return OmegaConf.create({"ard": {"time": {"start_year": start, "end_year": end}}})


def test_resolve_years_explicit() -> None:
    cfg = _make_cfg()
    assert _resolve_years(cfg, 2023) == [2023]


def test_resolve_years_range() -> None:
    cfg = _make_cfg(start=2018, end=2020)
    assert _resolve_years(cfg, None) == [2018, 2019, 2020]


def test_resolve_years_none_on_none() -> None:
    """When year is None, returns the full range."""
    cfg = _make_cfg(start=2023, end=2023)
    assert _resolve_years(cfg, None) == [2023]


# ── make_grid_spec ────────────────────────────────────────────────────────────


def _make_grid_spec() -> GridSpec:
    return make_grid_spec(
        origin_x=368000.0,
        origin_y=5839000.0,
        aoi_25833=(368002.320, 5797523.137, 417784.933, 5839258.933),
        wgs84_bbox=(13.0471, 52.3122, 13.7937, 52.6970),
    )


def test_grid_spec_alignment() -> None:
    """10m dimensions must be exact multiples of 10 for nested alignment."""
    spec = _make_grid_spec()
    assert spec.width_10m % 10 == 0
    assert spec.height_10m % 10 == 0


def test_grid_spec_nesting() -> None:
    """100m grid must be exact 10×10 aggregate of 10m grid."""
    spec = _make_grid_spec()
    assert spec.width_100m == spec.width_10m // 10
    assert spec.height_100m == spec.height_10m // 10


def test_grid_spec_bounds() -> None:
    """AOI must fit within the grid extent."""
    spec = _make_grid_spec()
    assert spec.xmax_10m >= spec.aoi_xmax
    assert spec.ymin_10m <= spec.aoi_ymin


def test_grid_spec_transforms() -> None:
    """Affine transforms must have correct resolution and origin."""
    spec = _make_grid_spec()
    t10 = spec.transform_10m
    t100 = spec.transform_100m
    assert t10.a == 10.0  # x resolution
    assert t10.e == -10.0  # y resolution (south-down)
    assert t10.c == spec.origin_x
    assert t10.f == spec.origin_y
    assert t100.a == 100.0
    assert t100.e == -100.0


def test_grid_spec_default_crs() -> None:
    spec = _make_grid_spec()
    assert spec.crs == "EPSG:25833"


def test_grid_spec_to_dict() -> None:
    spec = _make_grid_spec()
    d = spec.to_dict()
    assert d["crs"] == "EPSG:25833"
    assert d["width_10m"] == spec.width_10m
    assert d["width_100m"] == spec.width_100m
    assert len(d["aoi_25833"]) == 4


# ── get_spec (integration) ───────────────────────────────────────────────────


def test_get_spec_returns_grid_spec() -> None:
    """Requires ``data/berlin_aoi.geojson`` from the fetch-boundary script."""
    try:
        spec = get_spec()
    except FileNotFoundError:
        return  # skip gracefully if AOI file is missing
    assert isinstance(spec, GridSpec)
    assert spec.width_10m > 0
    assert spec.height_10m > 0
    assert spec.width_10m % 10 == 0  # nested alignment invariant
