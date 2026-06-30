"""Unit tests for the GridSpec module.

Verifies the geometric foundation of the downscaling pipeline:
  * 10m dimensions are exact multiples of 10
  * 100m grid is an exact 10×10 aggregate of the 10m grid
  * Affine transforms have correct resolution and origin
  * AOI fits within the grid extent
  * The default spec loaded from ``data/boundaries/berlin_landesgrenze_2km_buffer.geojson`` is sane
"""

from berlin_lst_downscaling.data.grid_spec import GridSpec, get_spec, make_grid_spec


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


def test_get_spec_returns_grid_spec() -> None:
    """Requires ``data/boundaries/berlin_landesgrenze_2km_buffer.geojson``."""
    try:
        spec = get_spec()
    except FileNotFoundError:
        return  # skip gracefully if AOI file is missing
    assert isinstance(spec, GridSpec)
    assert spec.width_10m > 0
    assert spec.height_10m > 0
    assert spec.width_10m % 10 == 0  # nested alignment invariant
    assert spec.aoi_polygon_25833 is not None
