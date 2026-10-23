"""Synthetic checks for S2 harmonization, LoD2 heights, coverage, and diagnostics."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from affine import Affine
from omegaconf import OmegaConf
from rasterio.crs import CRS
from shapely.geometry import box

from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.masking import mask_s2
from berlin_lst_downscaling.data.ard.pipeline import run as run_ard
from berlin_lst_downscaling.data.ard.publish_guard import (
    FrozenPrefixError,
    assert_feature_output_root,
    assert_sentinel2_output_root,
)
from berlin_lst_downscaling.data.ard.s2_reflectance import (
    recover_harmonized_reflectance_from_clipped_ard,
    resolve_s2_reflectance_scale,
    scale_s2_digital_numbers,
)
from berlin_lst_downscaling.data.diagnostics import feature_correctness
from berlin_lst_downscaling.data.diagnostics.feature_correctness import (
    DiagnosticsRefused,
    analyze_lod2_tile,
    assert_output_dir,
    assert_upload_uri,
    catalog_from_frames,
    exclusion_reason_counts,
    main,
    missing_record_assets,
    scan_feature_stack,
    validate_plan,
)
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNELS,
    FEATURE_SCHEMA_VERSION,
)
from berlin_lst_downscaling.data.features.pipeline import run_features
from berlin_lst_downscaling.data.secondary.atom import AtomAsset
from berlin_lst_downscaling.data.secondary.building_raster import (
    accumulate_building,
    morphology_bands,
    new_accumulators,
)
from berlin_lst_downscaling.data.secondary.citygml import (
    Building,
    parse_lod2_buildings,
    parse_lod2_buildings_legacy,
    parse_lod2_buildings_with_stats,
)
from berlin_lst_downscaling.data.secondary.download import DownloadReceipt
from berlin_lst_downscaling.data.secondary.lod2 import (
    Lod2IngestStats,
    _process_lod2_tile,
    lod2_qa_stats,
)
from berlin_lst_downscaling.data.secondary.lod_vintages import _Accumulator

_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _ROOT / ".opencode/skills/google-access/scripts/run-feature-diagnostics-vm.sh"
_ALLOWLIST = _ROOT / ".opencode/skills/google-access/scripts/start-vm-run-tab.sh"
_OPERATOR = _ROOT / "scripts/operators/diagnose_feature_correctness.py"


def _ring(coords: list[tuple[float, float, float]]) -> str:
    body = " ".join(f"{x} {y} {z}" for x, y, z in coords)
    return f'<gml:posList srsDimension="3">{body}</gml:posList>'


def _square(x0: float, y0: float, x1: float, y1: float, z: float) -> str:
    return _ring([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)])


def _surface(tag: str, poly_id: str, ring: str) -> str:
    return (
        f"<bldg:{tag}><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        f'<gml:Polygon gml:id="{poly_id}"><gml:exterior><gml:LinearRing>{ring}'
        f"</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
        f"</gml:MultiSurface></bldg:lod2MultiSurface></bldg:{tag}>"
    )


def _citymodel(members: str) -> bytes:
    return (
        '<?xml version="1.0"?>'
        '<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
        'xmlns:gml="http://www.opengis.net/gml">'
        f"{members}</core:CityModel>"
    ).encode()


def _part_only_tile() -> bytes:
    ground = _surface("GroundSurface", "g-part", _square(0, 0, 4, 4, 0))
    return _citymodel(
        "<core:cityObjectMember><bldg:Building gml:id='b-part'>"
        "<bldg:consistsOfBuildingPart><bldg:BuildingPart gml:id='p1'>"
        f"<bldg:measuredHeight>12.5</bldg:measuredHeight>{ground}"
        "</bldg:BuildingPart></bldg:consistsOfBuildingPart>"
        "</bldg:Building></core:cityObjectMember>"
    )


def test_s2_scale_pre_and_post_baseline_and_nodata() -> None:
    """Pre-04.00 divides by 10000. Post-04.00 applies -1000 DN. NaN stays NaN."""
    pre = resolve_s2_reflectance_scale({"s2:processing_baseline": "03.01"})
    post = resolve_s2_reflectance_scale(
        {"s2:processing_baseline": "04.00", "s2:boa_add_offset": -1000}
    )
    assert pre.boa_add_offset == 0
    assert post.boa_add_offset == -1000
    pre_scaled = scale_s2_digital_numbers(np.array([5000.0, np.nan], dtype=np.float64), pre)
    post_scaled = scale_s2_digital_numbers(
        np.array([5000.0, 0.0, np.nan, 10500.0], dtype=np.float64),
        post,
    )
    assert pre_scaled[0] == pytest.approx(0.5)
    assert np.isnan(pre_scaled[1])
    assert post_scaled[0] == pytest.approx(0.4)
    assert post_scaled[1] == pytest.approx(0.0)
    assert np.isnan(post_scaled[2])
    assert post_scaled[3] == pytest.approx(0.95)


def test_clipped_ard_cannot_recover_dn_above_10000() -> None:
    """DN 10500 is reflectance 0.95 after the offset, but clipped ARD stores 1."""
    stored = np.array([1.0, np.nan], dtype=np.float32)
    recovered = recover_harmonized_reflectance_from_clipped_ard(stored, -1000)
    assert recovered[0] == pytest.approx(0.9)
    assert np.isnan(recovered[1])
    exact = scale_s2_digital_numbers(
        np.array([10500.0], dtype=np.float64),
        resolve_s2_reflectance_scale({"s2:processing_baseline": "05.10"}),
    )
    assert exact[0] == pytest.approx(0.95)
    assert recovered[0] != pytest.approx(float(exact[0]))


def test_s2_offset_refuses_a_date_without_metadata() -> None:
    """The acquisition date is not a fallback when baseline and offset are absent."""
    with pytest.raises(ValueError, match="acquisition date"):
        resolve_s2_reflectance_scale({"datetime": "2023-06-01T10:00:00Z"})


def test_s2_schema_and_feature_schema_are_versioned() -> None:
    """Schema 9 and feature schema 4 keep the published v5 contract untouched."""
    assert contract_for_source("sentinel-2-l2a").schema_version == 9
    assert FEATURE_SCHEMA_VERSION == 4


def _dataset(data: dict[str, np.ndarray], transform: Affine) -> xr.Dataset:
    ds = xr.Dataset({name: (("y", "x"), values) for name, values in data.items()})
    ds.rio.write_crs("EPSG:25833", inplace=True)
    ds.rio.write_transform(transform, inplace=True)
    return ds


def test_mask_s2_applies_offset_before_clip_and_keeps_fill_nodata() -> None:
    """A clear DN and a fill pixel are scaled with the scene offset inside mask_s2."""
    transform10 = Affine(10, 0, 0, 0, -10, 40)
    transform20 = Affine(20, 0, 0, 0, -20, 40)
    clear = np.full((4, 4), 5000.0, dtype=np.float32)
    scl = np.full((4, 4), 4.0, dtype=np.float32)
    scl[3, 3] = 0
    ds10 = _dataset(
        {"B02": clear, "B03": clear, "B04": clear, "B08": clear, "SCL": scl},
        transform10,
    )
    swir = np.full((2, 2), 5000.0, dtype=np.float32)
    ds20 = _dataset(
        {"B11": swir, "B12": swir, "SCL": np.full((2, 2), 4.0, dtype=np.float32)},
        transform20,
    )
    scale = resolve_s2_reflectance_scale(
        {"s2:processing_baseline": "04.00", "s2:boa_add_offset": -1000}
    )
    masked = mask_s2(ds10, ds20, OmegaConf.create({}), 180.0, 45.0, reflectance=scale)
    assert masked.attrs["s2_reflectance"]["boa_add_offset"] == -1000
    assert float(masked["B02"].values[0, 0]) == pytest.approx(0.4)
    assert np.isnan(masked["B02"].values[3, 3])


def test_lod2_building_part_height_is_kept_and_missing_height_is_not_zero() -> None:
    """A part-only measuredHeight is emitted. A building with no height is dropped."""
    buildings, stats = parse_lod2_buildings_with_stats(_part_only_tile())
    assert len(buildings) == 1
    assert buildings[0].measured_height == pytest.approx(12.5)
    assert buildings[0].height_source == "measured"
    assert parse_lod2_buildings_legacy(_part_only_tile()) == []
    flat = _citymodel(
        "<core:cityObjectMember><bldg:Building gml:id='b-flat'>"
        f"{_surface('GroundSurface', 'g-flat', _square(0, 0, 4, 4, 1))}"
        "</bldg:Building></core:cityObjectMember>"
    )
    dropped, dropped_stats = parse_lod2_buildings_with_stats(flat)
    assert dropped == []
    assert dropped_stats.dropped_missing_height == 1
    assert all(item.measured_height != 0 for item in parse_lod2_buildings(flat))


def test_lod2_z_extent_and_inherited_parent_height() -> None:
    """Z span is used when measuredHeight is absent. A part can inherit the parent."""
    z_tile = _citymodel(
        "<core:cityObjectMember><bldg:Building gml:id='b-z'>"
        f"{_surface('GroundSurface', 'g-z', _square(0, 0, 6, 6, 30))}"
        f"{_surface('RoofSurface', 'r-z', _square(0, 0, 6, 6, 42))}"
        "</bldg:Building></core:cityObjectMember>"
    )
    z_buildings, z_stats = parse_lod2_buildings_with_stats(z_tile)
    assert len(z_buildings) == 1
    assert z_buildings[0].measured_height == pytest.approx(12.0)
    assert z_buildings[0].height_source == "z_extent"
    assert z_stats.height_from_z_extent == 1

    inherited = _citymodel(
        "<core:cityObjectMember><bldg:Building gml:id='b-parent'>"
        "<bldg:measuredHeight>20</bldg:measuredHeight>"
        "<bldg:consistsOfBuildingPart><bldg:BuildingPart gml:id='p-inherit'>"
        f"{_surface('GroundSurface', 'g-inherit', _square(0, 0, 3, 3, 5))}"
        "</bldg:BuildingPart></bldg:consistsOfBuildingPart>"
        "</bldg:Building></core:cityObjectMember>"
    )
    parts, part_stats = parse_lod2_buildings_with_stats(inherited)
    assert len(parts) == 1
    assert parts[0].building_id == "p-inherit"
    assert parts[0].measured_height == pytest.approx(20.0)
    assert parts[0].height_source == "inherited"
    assert part_stats.parts_emitted == 1
    assert part_stats.emitted == 1


def test_small_building_missing_cell_center_is_counted() -> None:
    """A 2 m by 2 m footprint in a cell corner still contributes 4 m²."""
    transform = Affine(10, 0, 0, 0, -10, 20)
    shape = (2, 2)
    footprint = box(0.0, 18.0, 2.0, 20.0)
    assert not footprint.contains(box(4.9, 14.9, 5.1, 15.1).centroid)
    arrays = new_accumulators(shape)
    hit = accumulate_building(
        footprint,
        8.0,
        transform=transform,
        shape=shape,
        sum_h=arrays["sum_h"],
        sum_h2=arrays["sum_h2"],
        sum_area=arrays["sum_area"],
        count=arrays["count"],
        max_h=arrays["max_h"],
    )
    assert hit
    bands = morphology_bands(
        arrays["sum_h"], arrays["sum_h2"], arrays["sum_area"], arrays["count"], arrays["max_h"]
    )
    assert bands["building_coverage_ratio"][0, 0] == pytest.approx(0.04)
    assert bands["building_height_mean"][0, 0] == pytest.approx(8.0)
    assert np.isnan(bands["building_coverage_ratio"][1, 1])


def test_building_height_mean_is_area_weighted() -> None:
    """Two footprints in one cell average by intersection area, not by count."""
    transform = Affine(10, 0, 0, 0, -10, 20)
    shape = (1, 1)
    arrays = new_accumulators(shape)
    for footprint, height in ((box(0, 10, 5, 20), 10.0), (box(5, 10, 10, 20), 20.0)):
        accumulate_building(
            footprint,
            height,
            transform=transform,
            shape=shape,
            sum_h=arrays["sum_h"],
            sum_h2=arrays["sum_h2"],
            sum_area=arrays["sum_area"],
            count=arrays["count"],
            max_h=arrays["max_h"],
        )
    bands = morphology_bands(
        arrays["sum_h"], arrays["sum_h2"], arrays["sum_area"], arrays["count"], arrays["max_h"]
    )
    assert bands["building_height_mean"][0, 0] == pytest.approx(15.0)
    assert bands["building_height_std"][0, 0] == pytest.approx(5.0)
    assert bands["building_height_max"][0, 0] == pytest.approx(20.0)


def test_total_buildings_counts_emitted_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The QA counter is the number of buildings handed to the rasterizer."""
    tile = tmp_path / "tile.zip"
    with zipfile.ZipFile(tile, "w") as archive:
        archive.writestr("tile.gml", _part_only_tile())

    def _fake_download(**kwargs: object) -> DownloadReceipt:
        del kwargs
        return DownloadReceipt(
            uri="memory://tile.zip",
            byte_count=1,
            checksum="abc",
            local_cache_path=str(tile),
        )

    monkeypatch.setattr(
        "berlin_lst_downscaling.data.secondary.lod2.download_to_raw",
        _fake_download,
    )
    from odc.geo.geobox import GeoBox

    grid = GeoBox((4, 4), Affine(10, 0, 369190, 0, -10, 5838410), CRS.from_epsg(25833))
    arrays = new_accumulators((4, 4))
    ingest = Lod2IngestStats()
    asset = AtomAsset(
        url="https://example.invalid/tile.zip",
        filename="tile.zip",
        title="tile",
        easting=0,
        northing=0,
    )
    _process_lod2_tile(
        asset,
        grid,
        arrays["sum_h"],
        arrays["sum_h2"],
        arrays["count"],
        arrays["sum_area"],
        arrays["max_h"],
        str(tmp_path / "out"),
        ingest,
    )
    assert ingest.total_buildings == 1
    stats = lod2_qa_stats(mean_arr=np.full((4, 4), np.nan), ingest=ingest, tile_count=1)
    assert stats["total_buildings"] == 1

    vintage = _Accumulator(GeoBox((2, 2), Affine(10, 0, 0, 0, -10, 20), CRS.from_epsg(25833)))
    parsed = parse_lod2_buildings(_part_only_tile())
    vintage.add_tile(parsed)
    assert vintage.total_buildings == 1


def _valid_stack(height: int, width: int) -> np.ndarray:
    stack = np.zeros((len(FEATURE_CHANNELS), height, width), dtype=np.float32)
    for index, channel in enumerate(FEATURE_CHANNELS):
        assert channel.valid_range is not None
        low, high = channel.valid_range
        stack[index] = (low + high) / 2.0
    return stack


def test_exclusion_reason_counts_name_the_invalid_channel() -> None:
    """A target-valid cell with one bad subpixel is excluded and attributed."""
    stack = _valid_stack(20, 20)
    stack[0, 10, 0] = np.nan
    target = np.ones((2, 2), dtype=bool)
    target[0, 1] = False
    counted = exclusion_reason_counts(target, stack, aoi_fraction=np.ones(target.shape))
    assert counted["excluded_cells"] == 1
    assert counted["eligible_cells"] == 2
    assert counted["per_channel"]["B02"] == 1
    assert counted["per_channel"]["ndvi"] == 0


def _write_tif(
    path: Path, data: np.ndarray, transform: Affine, *, count_last: bool = False
) -> None:
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": "EPSG:25833",
        "transform": transform,
    }
    del count_last
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def test_scan_and_lod2_report_and_operator_flags(tmp_path: Path) -> None:
    """Scopes (a), (b), and (c) run on synthetic inputs and refuse raster metadata-only."""
    stack = _valid_stack(20, 23)
    stack[3, 0, 0] = np.nan
    feature = tmp_path / "stack.tif"
    landsat = tmp_path / "lst.tif"
    flag = tmp_path / "flag.tif"
    _write_tif(feature, stack, Affine(10, 0, 0, 0, -10, 200))
    _write_tif(landsat, np.full((2, 3), 300.0, dtype=np.float32), Affine(100, 0, 0, 0, -100, 200))
    flags = np.zeros((2, 3), dtype=np.uint8)
    flags[:, 2] = 1  # Ceiling-grid edge targets are invalid in this fixture.
    _write_tif(flag, flags, Affine(100, 0, 0, 0, -100, 200))
    aoi = tmp_path / "aoi_10m.tif"
    _write_tif(aoi, np.ones((20, 23), dtype=np.uint8), Affine(10, 0, 0, 0, -10, 200))
    tile = tmp_path / "one.gml"
    tile.write_bytes(_part_only_tile())
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "LC08_TEST",
                    "year": 2022,
                    "s2_scene_id": "S2A_TEST",
                    "properties": {
                        "s2:processing_baseline": "04.00",
                        "s2:boa_add_offset": -1000,
                    },
                    "feature_stack": str(feature),
                    "landsat_cog": str(landsat),
                    "landsat_flag": str(flag),
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert (
        main(
            [
                "--scope",
                "s2-baseline",
                "--scope",
                "eligibility-gaps",
                "--scope",
                "lod2-parser",
                "--aoi-mask",
                str(aoi),
                "--scenes-json",
                str(scenes),
                "--lod2-tile",
                str(tile),
                "--yes",
                "--output-dir",
                str(output),
                "--run-id",
                "synthetic",
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["s2_baseline"][0]["boa_add_offset"] == -1000
    assert summary["s2_baseline"][0]["processing_baseline"] == "04.00"
    assert summary["eligibility_gaps"]["excluded_cells"] == 1
    assert summary["eligibility_gaps"]["per_channel"]["B08"] == 1
    assert summary["eligibility_gaps"]["by_year"]["2022"]["excluded_cells"] == 1
    assert summary["eligibility_gaps"]["scenes"][0]["dropped_edge_cols"] == 0
    assert summary["eligibility_gaps"]["scenes"][0]["padded_edge_cols"] == 7
    lod2 = summary["lod2_parser"][0]
    assert lod2["height_locations"]["buildingpart_only_measured_height"] == 1
    assert lod2["height_locations"]["missing_measured_height"] == 1
    assert lod2["legacy"]["count"] == 0
    assert lod2["current"]["count"] == 1
    assert lod2["current"]["min"] == pytest.approx(12.5)
    assert "summary.md" in {path.name for path in output.iterdir()}

    report = analyze_lod2_tile(tile)
    assert report["current"]["histogram_1m"]["12"] == 1

    untouched = tmp_path / "estimate"
    assert main(["--scope", "s2-baseline", "--estimate-only", "--output-dir", str(untouched)]) == 0
    assert not untouched.exists()

    with pytest.raises(DiagnosticsRefused, match="eligibility-gaps"):
        validate_plan(["eligibility-gaps"], metadata_only=True, lod2_tiles=[])
    refused = main(
        [
            "--scope",
            "eligibility-gaps",
            "--metadata-only",
            "--estimate-only",
            "--scenes-json",
            str(scenes),
        ]
    )
    assert refused == 2
    with pytest.raises(DiagnosticsRefused, match="exactly one"):
        validate_plan(["lod2-parser"], metadata_only=True, lod2_tiles=[])
    local = tmp_path / "local"
    assert (
        main(
            [
                "--scope",
                "s2-baseline",
                "--scope",
                "lod2-parser",
                "--metadata-only",
                "--yes",
                "--scenes-json",
                str(scenes),
                "--lod2-tile",
                str(tile),
                "--output-dir",
                str(local),
            ]
        )
        == 0
    )
    assert (local / "summary.json").exists()


def test_diagnostics_paths_and_launcher_are_read_only() -> None:
    """Uploads stay under the diagnostics prefix. The operator does not start a VM."""
    with pytest.raises(DiagnosticsRefused):
        assert_upload_uri("gs://berlin-lst-training-data/features/v5/diagnostics/")
    allowed = assert_upload_uri(
        "gs://berlin-lst-training-data/evaluation/diagnostics/feature-diagnostics-test"
    )
    assert allowed.endswith("/")
    with pytest.raises(DiagnosticsRefused):
        assert_output_dir(Path("/workspace/docs/results/diagnostics"))
    operator = _OPERATOR.read_text(encoding="utf-8")
    library = (
        _ROOT / "src/berlin_lst_downscaling/data/diagnostics/feature_correctness.py"
    ).read_text(encoding="utf-8")
    for source in (operator, library):
        assert "start-vm" not in source
        assert "gcloud" not in source
        assert "compute instances start" not in source
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "vm_init_run",
        "vm_push_deploy",
        "vm_start_and_wait_ssh",
        "vm_launch_detached",
        "vm_poll",
        "vm_finish",
        "vm_stop",
        "evaluation/diagnostics/",
    ):
        assert token in launcher
    subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)  # noqa: S603, S607
    subprocess.run(["bash", "-n", str(_ALLOWLIST)], check=True)  # noqa: S603, S607
    common = _ROOT / ".opencode/skills/google-access/scripts/vm-runner-common.sh"
    subprocess.run(["bash", "-n", str(common)], check=True)  # noqa: S603, S607
    common_text = common.read_text(encoding="utf-8")
    assert "no output yet" in common_text
    assert "[ -s" in common_text
    assert "vm_preserve_logs" in common_text
    assert "gcloud storage cp" in common_text
    assert "/tmp/${WRAP_RUN_ID}" in common_text  # noqa: S108
    assert "tail -n 50" in common_text
    stop = common_text.split("vm_stop()", 1)[1]
    assert stop.index("vm_preserve_logs") < stop.index("stop-vm.sh")
    assert "PYTHONUNBUFFERED=1" in launcher
    assert "python -u" in launcher
    assert "--expected-scene-count ${EXPECTED_SCENE_COUNT}" in launcher
    assert "--output-dir /workspace/feature-diagnostics/${WRAP_RUN_ID}" in launcher
    assert 'mkdir -p "$dest" || true' in common_text
    assert 'EXPECTED_SCENE_COUNT="${EXPECTED_SCENE_COUNT:-324}"' in launcher
    allowlist = _ALLOWLIST.read_text(encoding="utf-8")
    assert allowlist.count("run-feature-diagnostics-vm.sh") >= 3


def test_catalog_join_uses_manifest_columns_without_opening_gcs() -> None:
    """Training rows pick up the Sentinel-2 item href from the selection frame."""
    training = pd.DataFrame(
        [
            {
                "scene_id": "LC08_1",
                "year": 2021,
                "s2_scene_id": "S2_1",
                "feature_stack": "",
                "split": "train",
                "status": "published",
            }
        ]
    )
    selection = pd.DataFrame(
        [
            {
                "scene_id": "S2_1",
                "source": "sentinel-2-l2a",
                "item_href": "https://example.invalid/item.json",
            }
        ]
    )
    rows = catalog_from_frames(
        training,
        selection,
        features_root="gs://berlin-lst-training-data/features/v5",
        ard_root="gs://berlin-lst-training-data/ard/full/example",
    )
    assert rows[0]["item_href"] == "https://example.invalid/item.json"
    assert rows[0]["feature_stack"].endswith("/LC08_1/LC08_1.tif")
    assert "landsat-c2-l2/2021/LC08_1" in rows[0]["landsat_cog"]


def test_catalog_drops_inference_deferred_and_keeps_no_eligible_cells() -> None:
    """2026 deferred rows have no stack; empty-eligibility rows still do."""
    training = pd.DataFrame(
        [
            {
                "scene_id": "LC08_KEEP",
                "year": 2020,
                "s2_scene_id": "S2_KEEP",
                "exclusion_reason": "no_eligible_cells",
                "feature_stack": "gs://bucket/features/v5/LC08_KEEP/LC08_KEEP.tif",
            },
            {
                "scene_id": "LC08_L2SP_192023_20260503_02_T1",
                "year": 2026,
                "s2_scene_id": "S2_DEFER",
                "exclusion_reason": "inference_deferred",
                "feature_stack": "gs://bucket/features/v5/LC08_L2SP_192023_20260503_02_T1.tif",
            },
        ]
    )
    selection = pd.DataFrame(
        [{"scene_id": "S2_KEEP", "source": "sentinel-2-l2a", "item_href": "https://example/a"}]
    )
    rows = catalog_from_frames(
        training,
        selection,
        features_root="gs://bucket/features/v5",
        ard_root="gs://bucket/ard",
    )
    assert [row["scene_id"] for row in rows] == ["LC08_KEEP"]
    assert rows[0]["feature_stack"].endswith("LC08_KEEP.tif")


def test_missing_assets_listed_before_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing feature stack exits 2 and names the path. No raster is opened."""
    present = tmp_path / "present"
    records = [
        {
            "scene_id": "LC08_KEEP",
            "feature_stack": str(present),
            "landsat_cog": str(tmp_path / "lst.tif"),
            "landsat_flag": str(tmp_path / "flag.tif"),
        }
    ]

    def fake_exists(path: str) -> bool:
        return path == str(present)

    missing = missing_record_assets(records, exists=fake_exists)
    assert str(tmp_path / "lst.tif") in missing
    assert str(tmp_path / "flag.tif") in missing
    assert str(present) not in missing

    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "LC08_MISSING",
                    "year": 2022,
                    "s2_scene_id": "S2",
                    "properties": {"s2:processing_baseline": "04.00"},
                    "feature_stack": str(tmp_path / "absent-stack.tif"),
                    "landsat_cog": str(tmp_path / "absent-lst.tif"),
                    "landsat_flag": str(tmp_path / "absent-flag.tif"),
                }
            ]
        ),
        encoding="utf-8",
    )

    def none_exist(_path: str) -> bool:
        return False

    monkey_holder = {"opened": False}

    def refuse_open(*_args: object, **_kwargs: object) -> None:
        monkey_holder["opened"] = True
        raise AssertionError("raster opened")

    original_exists = feature_correctness.storage_exists
    original_open = feature_correctness.rasterio.open
    feature_correctness.storage_exists = none_exist  # type: ignore[method-assign]
    feature_correctness.rasterio.open = refuse_open  # type: ignore[method-assign]
    try:
        code = main(
            [
                "--scope",
                "eligibility-gaps",
                "--scenes-json",
                str(scenes),
                "--yes",
                "--output-dir",
                str(tmp_path / "out"),
                "--expected-scene-count",
                "1",
            ]
        )
    finally:
        feature_correctness.storage_exists = original_exists  # type: ignore[method-assign]
        feature_correctness.rasterio.open = original_open
    captured = capsys.readouterr()
    assert code == 2
    assert "absent-stack.tif" in captured.out
    assert "absent-lst.tif" in captured.out
    assert "absent-flag.tif" in captured.out
    assert monkey_holder["opened"] is False
    assert "Scenes: 1" in captured.out


def test_expected_scene_count_uses_selected_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The estimate counts selected rows and refuses a mismatched pin."""
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "A",
                    "year": 2021,
                    "exclusion_reason": "inference_deferred",
                    "properties": {"s2:processing_baseline": "04.00"},
                },
                {
                    "scene_id": "B",
                    "year": 2021,
                    "exclusion_reason": "no_eligible_cells",
                    "properties": {"s2:processing_baseline": "04.00"},
                },
            ]
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "--scope",
            "s2-baseline",
            "--estimate-only",
            "--scenes-json",
            str(scenes),
            "--expected-scene-count",
            "324",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "Scenes: 1" in captured.out
    assert "expected-scene-count 324" in captured.out


def test_scan_feature_stack_rasterio_error_names_the_path() -> None:
    """A 404 from rasterio becomes DiagnosticsRefused and names the stack."""
    from rasterio.errors import RasterioIOError

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RasterioIOError("HTTP response code: 404")

    original = feature_correctness.rasterio.open
    feature_correctness.rasterio.open = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(DiagnosticsRefused, match="features/v5/missing.tif"):
            scan_feature_stack(
                "gs://bucket/features/v5/missing.tif",
                "gs://bucket/lst.tif",
                "gs://bucket/flag.tif",
            )
    finally:
        feature_correctness.rasterio.open = original


def test_mid_scan_exception_writes_summary_and_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unhandled scan error still writes the summary and uploads the tree."""
    feature = tmp_path / "stack.tif"
    landsat = tmp_path / "lst.tif"
    flag = tmp_path / "flag.tif"
    feature.write_bytes(b"not-a-real-open")
    landsat.write_bytes(b"x")
    flag.write_bytes(b"x")
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "LC08_BOOM",
                    "year": 2022,
                    "s2_scene_id": "S2",
                    "properties": {"s2:processing_baseline": "04.00"},
                    "feature_stack": str(feature),
                    "landsat_cog": str(landsat),
                    "landsat_flag": str(flag),
                }
            ]
        ),
        encoding="utf-8",
    )
    uploads: list[tuple[Path, str]] = []

    def fake_scan(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("boom mid scan")

    def fake_upload(source: Path, destination: str) -> None:
        uploads.append((source, destination))

    monkeypatch.setattr(feature_correctness, "scan_feature_stack", fake_scan)
    monkeypatch.setattr(feature_correctness, "_upload_tree", fake_upload)
    output = tmp_path / "out"
    uri = "gs://berlin-lst-training-data/evaluation/diagnostics/unit-crash/"
    code = main(
        [
            "--scope",
            "eligibility-gaps",
            "--scenes-json",
            str(scenes),
            "--yes",
            "--output-dir",
            str(output),
            "--upload-uri",
            uri,
            "--run-id",
            "unit-crash",
        ]
    )
    assert code == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "boom mid scan" in summary["error"]
    assert "Traceback" in summary["traceback"]
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "boom mid scan" in markdown
    assert uploads == [(output, uri)]


def test_consecutive_runs_do_not_mix_jsonl_rows(tmp_path: Path) -> None:
    """A second run into the same directory replaces results.jsonl."""
    output = tmp_path / "out"

    def scenes(scene_id: str) -> Path:
        path = tmp_path / f"{scene_id}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "scene_id": scene_id,
                        "year": 2022,
                        "s2_scene_id": scene_id,
                        "properties": {"s2:processing_baseline": "04.00"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        return path

    args = [
        "--scope",
        "s2-baseline",
        "--metadata-only",
        "--yes",
        "--output-dir",
        str(output),
    ]
    assert main([*args, "--scenes-json", str(scenes("SCENE_A")), "--run-id", "a"]) == 0
    assert main([*args, "--scenes-json", str(scenes("SCENE_B")), "--run-id", "b"]) == 0
    lines = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["scene_id"] for row in lines] == ["SCENE_B"]


def test_mid_scan_refusal_writes_summary_and_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DiagnosticsRefused during the scan is recorded and uploaded, then exits 2."""
    feature = tmp_path / "stack.tif"
    landsat = tmp_path / "lst.tif"
    flag = tmp_path / "flag.tif"
    for path in (feature, landsat, flag):
        path.write_bytes(b"x")
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {
                    "scene_id": "LC08_MISSING",
                    "year": 2022,
                    "s2_scene_id": "S2",
                    "properties": {"s2:processing_baseline": "04.00"},
                    "feature_stack": str(feature),
                    "landsat_cog": str(landsat),
                    "landsat_flag": str(flag),
                }
            ]
        ),
        encoding="utf-8",
    )
    uploads: list[tuple[Path, str]] = []

    def fake_scan(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise DiagnosticsRefused(f"cannot open raster {feature}")

    monkeypatch.setattr(feature_correctness, "scan_feature_stack", fake_scan)
    monkeypatch.setattr(
        feature_correctness,
        "_upload_tree",
        lambda source, destination: uploads.append((source, destination)),
    )
    output = tmp_path / "out"
    uri = "gs://berlin-lst-training-data/evaluation/diagnostics/unit-refused/"
    code = main(
        [
            "--scope",
            "eligibility-gaps",
            "--scenes-json",
            str(scenes),
            "--yes",
            "--output-dir",
            str(output),
            "--upload-uri",
            uri,
            "--run-id",
            "unit-refused",
        ]
    )
    assert code == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "cannot open raster" in summary["error"]
    assert "Traceback" in summary["traceback"]
    assert "cannot open raster" in (output / "summary.md").read_text(encoding="utf-8")
    assert uploads == [(output, uri)]


def test_published_prefixes_are_refused_before_any_ledger_write() -> None:
    """Schema 9 and schema 4 cannot reconcile into the published roots."""
    frozen_s2 = "gs://berlin-lst-training-data/ard/full/2017-2026-cutoff-20260717T235959Z"
    frozen_features = "gs://berlin-lst-training-data/features/v5"
    with pytest.raises(FrozenPrefixError):
        assert_sentinel2_output_root(frozen_s2 + "/")
    with pytest.raises(FrozenPrefixError):
        run_ard(OmegaConf.create({"output_root": frozen_s2, "sources": ["sentinel-2-l2a"]}))
    assert_sentinel2_output_root(
        "gs://berlin-lst-training-data/ard/full/2017-2026-s2-harmonized-v9"
    )
    with pytest.raises(FrozenPrefixError):
        assert_feature_output_root(frozen_features)
    with pytest.raises(FrozenPrefixError):
        run_features(OmegaConf.create({"output_root": "gs://berlin-lst-training-data/features/v3"}))
    assert_feature_output_root("gs://berlin-lst-training-data/features/v6")
    assert_feature_output_root("data/smoke/features")


def test_vintage_window_keeps_a_building_that_misses_every_cell_center() -> None:
    """A corner sliver is painted by the vintage accumulator, not marked outside."""
    from odc.geo.geobox import GeoBox

    grid = GeoBox((2, 2), Affine(10, 0, 0, 0, -10, 20), CRS.from_epsg(25833))
    accumulator = _Accumulator(grid)
    shed = Building(
        building_id="shed",
        footprint=box(0.1, 19.6, 0.4, 19.9),
        measured_height=6.0,
    )
    accumulator.add_tile([shed])
    assert accumulator.buildings_rasterized == 1
    assert accumulator.buildings_outside_grid == 0
    assert accumulator.area[0, 0] == pytest.approx(0.09)
    assert accumulator.count[0, 0] == 1
