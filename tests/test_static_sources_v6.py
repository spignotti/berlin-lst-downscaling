"""Static sources v6: LoD2 range contract, fail-closed ledger, idempotent rerun, validator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from omegaconf import OmegaConf
from rasterio.crs import CRS

from berlin_lst_downscaling.data.secondary.idempotency import reconcile
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.secondary.lod2 import contract_for_lod2_morphology
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    finalize_secondary_product,
    vintage_interval,
)
from berlin_lst_downscaling.data.secondary.source_pipeline import (
    expected_source_items,
    find_not_done_items,
)
from berlin_lst_downscaling.data.secondary.validate import validate_secondary_cog

_REPO = Path(__file__).resolve().parents[1]


def _load_script(relative: str, name: str):
    path = _REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── criterion 1: LoD2 range contract admits the Fernsehturm ────────────


def test_lod2_contract_admits_fernsehturm() -> None:
    contract = contract_for_lod2_morphology()
    by_name = {b.name: b for b in contract.output_bands}
    assert by_name["building_height_mean"].valid_range == (0.0, 400.0)
    assert by_name["building_height_max"].valid_range == (0.0, 400.0)
    assert by_name["building_height_std"].valid_range == (0.0, 200.0)
    # Reported production maxima from the failed v6 run (TV tower cells).
    assert by_name["building_height_mean"].valid_range[1] >= 250.5623
    assert by_name["building_height_std"].valid_range[1] >= 108.8154
    assert by_name["building_height_max"].valid_range[1] >= 253.4840
    # Negative/NaN checks stay: lower bounds unchanged.
    assert by_name["building_height_mean"].valid_range[0] == 0.0
    assert by_name["building_height_std"].valid_range[0] == 0.0
    assert by_name["building_height_max"].valid_range[0] == 0.0


def test_validate_secondary_cog_accepts_tower_heights(tmp_path: Path) -> None:
    """A COG with Fernsehturm-scale heights passes; negatives still fail."""
    contract = contract_for_lod2_morphology()
    grid = GeoBox((4, 4), Affine(10, 0, 0, 0, -10, 20), CRS.from_epsg(25833))
    cog = tmp_path / "tower.tif"
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:25833",
        "transform": grid.transform,
    }
    mean = np.full((4, 4), 10.0, dtype=np.float32)
    mean[0, 0] = 250.5623  # TV tower cell
    std = np.full((4, 4), 1.0, dtype=np.float32)
    std[0, 0] = 108.8154
    bcr = np.full((4, 4), 0.3, dtype=np.float32)
    mx = np.full((4, 4), 12.0, dtype=np.float32)
    mx[0, 0] = 253.4840
    with rasterio.open(cog, "w", **profile) as dst:
        dst.write(mean, 1)
        dst.write(std, 2)
        dst.write(bcr, 3)
        dst.write(mx, 4)
    result = validate_secondary_cog(str(cog), contract, grid)
    assert result.ok, result.errors

    # Negative heights stay invalid.
    mean[1, 1] = -5.0
    with rasterio.open(cog, "w", **profile) as dst:
        dst.write(mean, 1)
        dst.write(std, 2)
        dst.write(bcr, 3)
        dst.write(mx, 4)
    result = validate_secondary_cog(str(cog), contract, grid)
    assert not result.ok


# ── criterion 2: fail-closed ledger audit ─────────────────────────────


def _row(item_id: str, source: str, period: str, status: str, **kw) -> SecondaryLedgerRow:
    return SecondaryLedgerRow(
        item_id=item_id,
        source=source,
        period_or_vintage=period,
        status=status,
        run_id="test",
        config_hash="hash",
        output_uri=kw.get("output_uri"),
        completion_uri=kw.get("completion_uri"),
        last_error=kw.get("last_error"),
    )


def test_expected_source_items_v6_defaults() -> None:
    cfg = OmegaConf.create(
        {"sources": ["imperviousness", "vegetation_height", "terrain_height", "lod2_morphology"]}
    )
    items = expected_source_items(cfg)
    assert sorted(i[0] for i in items) == [
        "imperviousness_2016",
        "imperviousness_2021",
        "lod2_morphology_2024",
        "terrain_height_2021",
        "vegetation_height_2020",
    ]


def test_find_not_done_items_fails_closed(tmp_path: Path) -> None:
    led = SecondaryLedger.open(str(tmp_path / "ledger.parquet"))
    cog = tmp_path / "a.tif"
    comp = tmp_path / "complete.json"
    cog.write_bytes(b"x")
    comp.write_bytes(b"{}")
    led.upsert(_row("a_1", "a", "1", "done", output_uri=str(cog), completion_uri=str(comp)))
    led.upsert(_row("b_2", "b", "2", "failed", last_error="boom"))

    errors = find_not_done_items(led, [("a_1", "a", "1"), ("b_2", "b", "2")])
    assert len(errors) == 1 and "b_2" in errors[0] and "failed" in errors[0]

    led2 = SecondaryLedger.open(str(tmp_path / "ledger2.parquet"))
    led2.upsert(_row("a_1", "a", "1", "done", output_uri=str(cog), completion_uri=str(comp)))
    assert find_not_done_items(led2, [("a_1", "a", "1")]) == []

    # Missing row is also a failure.
    assert find_not_done_items(led2, [("a_1", "a", "1"), ("c_3", "c", "3")])


def test_check_static_sources_ledger_script(tmp_path: Path) -> None:
    check = _load_script(
        "scripts/operators/check_static_sources_ledger.py", "check_static_sources_ledger"
    )
    check_main = check.main

    root = tmp_path / "release"
    root.mkdir()
    cog = root / "a.tif"
    cog.write_bytes(b"x")

    # Missing ledger -> FAIL.
    assert check_main(["--source-root", str(root)]) == 1

    led = SecondaryLedger.open(str(root / "ledger.parquet"))
    led.upsert(_row("a_1", "a", "1", "failed", last_error="boom"))
    assert check_main(["--source-root", str(root)]) == 1

    led3_root = tmp_path / "release3"
    led3_root.mkdir()
    cog3 = led3_root / "a.tif"
    comp3 = led3_root / "complete.json"
    cog3.write_bytes(b"x")
    comp3.write_bytes(b"{}")
    led3 = SecondaryLedger.open(str(led3_root / "ledger.parquet"))
    led3.upsert(_row("a_1", "a", "1", "done", output_uri=str(cog3), completion_uri=str(comp3)))
    assert check_main(["--source-root", str(led3_root)]) == 0


# ── criterion 3: idempotent rerun (reconcile + atomic overwrite) ───────


def test_reconcile_skips_done_retries_failed(tmp_path: Path) -> None:
    led = SecondaryLedger.open(str(tmp_path / "ledger.parquet"))
    cog = tmp_path / "done.tif"
    comp = tmp_path / "complete.json"
    cog.write_bytes(b"x")
    comp.write_bytes(b"{}")
    led.upsert(
        SecondaryLedgerRow(
            item_id="done_1",
            source="s",
            period_or_vintage="1",
            status="done",
            run_id="r1",
            config_hash="h1",
            output_uri=str(cog),
            completion_uri=str(comp),
        )
    )
    led.upsert(
        SecondaryLedgerRow(
            item_id="failed_2",
            source="s",
            period_or_vintage="2",
            status="failed",
            run_id="r1",
            last_error="boom",
        )
    )
    # Done with matching hash and present artifacts -> skip.
    assert reconcile([("done_1", "s", "1")], led, "h1") == []
    # Done with changed hash -> reprocess.
    assert [t[3] for t in reconcile([("done_1", "s", "1")], led, "h2")] == ["config_changed"]
    # Failed -> retry, done rows untouched.
    assert [t[3] for t in reconcile([("failed_2", "s", "2")], led, "h1")] == ["retry"]


def test_finalize_overwrites_cog_safely(tmp_path: Path) -> None:
    """A rerun of a failed row replaces the already-written COG (overwrite)."""
    grid = GeoBox((4, 4), Affine(10, 0, 0, 0, -10, 20), CRS.from_epsg(25833))
    contract = contract_for_lod2_morphology()
    start, end = vintage_interval(2024)

    def _prepared(value: float) -> PreparedSecondaryProduct:
        ds = xr.Dataset(
            {
                "building_height_mean": (("y", "x"), np.full((4, 4), value, dtype=np.float32)),
                "building_height_std": (("y", "x"), np.full((4, 4), 1.0, dtype=np.float32)),
                "building_coverage_ratio": (
                    ("y", "x"),
                    np.full((4, 4), 0.5, dtype=np.float32),
                ),
                "building_height_max": (("y", "x"), np.full((4, 4), value, dtype=np.float32)),
            }
        )
        ds = ds.rio.write_crs(str(grid.crs))
        ds = ds.rio.write_transform(grid.transform)
        return PreparedSecondaryProduct(
            source="lod2_morphology",
            item_key="2024",
            category="morphology",
            dataset=ds,
            contract=contract,
            nominal_interval=(start, end),
            source_metadata={},
            qa_stats={},
            config_hash="h",
        )

    prod_dir = str(tmp_path / "prod")
    first = finalize_secondary_product(_prepared(10.0), grid, prod_dir, "r1")
    with rasterio.open(first.cog_uri) as src:
        assert float(src.read(1)[0, 0]) == pytest.approx(10.0)
    # Rerun overwrites the same final path (no stale COG).
    second = finalize_secondary_product(_prepared(250.0), grid, prod_dir, "r2")
    assert second.cog_uri == first.cog_uri
    with rasterio.open(second.cog_uri) as src:
        assert float(src.read(1)[0, 0]) == pytest.approx(250.0)


# ── criterion 4: validator works against the v6 layout ─────────────────


def _write_lod_cog(path: Path, grid, *, mean: float = 10.0) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid.shape.y,
        "width": grid.shape.x,
        "count": 4,
        "dtype": "float32",
        "crs": str(grid.crs),
        "transform": grid.transform,
    }
    h, w = grid.shape.y, grid.shape.x
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((h, w), mean, dtype=np.float32), 1)
        dst.write(np.full((h, w), 1.0, dtype=np.float32), 2)
        dst.write(np.full((h, w), 0.3, dtype=np.float32), 3)
        dst.write(np.full((h, w), 12.0, dtype=np.float32), 4)


def _fake_v6_root(tmp_path: Path, grid, *, n_tiles: int = 2) -> Path:
    root = tmp_path / "v6"
    prod = root / "ard" / "static" / "sources" / "lod2_morphology" / "2024"
    prod.mkdir(parents=True)
    _write_lod_cog(prod / "lod2_morphology_2024.tif", grid)
    prov = {
        "source_metadata": {
            "tiles": [
                {
                    "easting": 390000 + 1000 * i,
                    "northing": 5810000,
                    "checksum": "x",
                    "byte_count": 1,
                    "filename": f"t{i}.zip",
                }
                for i in range(n_tiles)
            ]
        }
    }
    (prod / "provenance.json").write_text(json.dumps(prov))
    return root


def test_validate_lod_coverage_v6_only_fake_layout(tmp_path: Path, monkeypatch) -> None:
    validator = _load_script(
        "scripts/validators/validate_lod_coverage.py", "validate_lod_coverage_v6"
    )

    from berlin_lst_downscaling.data.qa.stage1_raw import analysis_grid_10m

    bbox = (13.40, 52.52, 13.41, 52.53)
    grid = analysis_grid_10m(bbox)
    root = _fake_v6_root(tmp_path, grid, n_tiles=2)
    monkeypatch.setattr(validator, "_EXPECTED_TILES_2024", 2)

    argv = [
        "validate_lod_coverage.py",
        "--static-sources-root",
        str(root),
        "--vintages",
        "2024",
        "--aoi",
        str(tmp_path / "no-aoi.tif"),
        "--bbox",
        ",".join(str(v) for v in bbox),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert validator.main() == 0


def test_validate_lod_coverage_missing_manifest_fails_closed(tmp_path: Path, monkeypatch) -> None:
    validator = _load_script(
        "scripts/validators/validate_lod_coverage.py", "validate_lod_coverage_v6b"
    )

    from berlin_lst_downscaling.data.qa.stage1_raw import analysis_grid_10m

    bbox = (13.40, 52.52, 13.41, 52.53)
    grid = analysis_grid_10m(bbox)
    root = _fake_v6_root(tmp_path, grid, n_tiles=2)
    monkeypatch.setattr(validator, "_EXPECTED_TILES_2024", 2)
    monkeypatch.setattr(validator, "_EXPECTED_MEMBERS", {2017: 1, 2021: 1, 2022: 1})

    # Requesting a historical vintage whose manifest is absent must fail
    # closed with FINDINGS (return 1), not crash with a traceback.
    argv = [
        "validate_lod_coverage.py",
        "--static-sources-root",
        str(root),
        "--vintages",
        "2021,2024",
        "--aoi",
        str(tmp_path / "no-aoi.tif"),
        "--bbox",
        ",".join(str(v) for v in bbox),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert validator.main() == 1


def test_resolve_lod_coverage_vintages_2024_only(tmp_path: Path, monkeypatch) -> None:
    from berlin_lst_downscaling.data.features import lod_coverage as coverage
    from berlin_lst_downscaling.data.qa.stage1_raw import analysis_grid_10m

    bbox = (13.40, 52.52, 13.41, 52.53)
    grid = analysis_grid_10m(bbox)
    root = _fake_v6_root(tmp_path, grid, n_tiles=2)
    monkeypatch.setattr(coverage, "_EXPECTED_TILES_2024", 2)

    out = coverage.resolve_lod_coverage_artifacts(str(root), vintages=(2024,))
    assert sorted(out) == [2024]
    mask = coverage.rasterize_lod_coverage(out[2024], grid)
    assert mask.shape == (grid.shape.y, grid.shape.x)
