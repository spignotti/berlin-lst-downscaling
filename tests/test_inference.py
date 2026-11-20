"""Offline tests for 10 m scene inference. No network and no real scenes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
import torch
from affine import Affine
from odc.geo.geobox import GeoBox
from rasterio.crs import CRS
from torch import nn

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.features.paths import ledger_path
from berlin_lst_downscaling.data.secondary.ledger import SecondaryLedger, SecondaryLedgerRow
from berlin_lst_downscaling.data.training.contracts import CANON_GRID_ORIGIN_X, CANON_GRID_ORIGIN_Y
from berlin_lst_downscaling.modeling.channels import ActiveChannelSelection
from berlin_lst_downscaling.modeling.inference import (
    PRIOR_PLACEHOLDER_K,
    build_prior_10m,
    patch_bbox_10m,
    plan_tiles,
    predict_scene,
    relative_difference,
    reproduction_mismatch,
    require_validation_patches,
    run_scene,
    scenes_with_done_features,
    score_patch_like_export,
    seed_from_checkpoint_name,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    ScalerChannel,
    ScalerSpec,
    prior_model_channel,
)

_OX = CANON_GRID_ORIGIN_X
_OY = CANON_GRID_ORIGIN_Y
_REPO = Path(__file__).resolve().parents[1]


def _affine(res: float, col0: int, row0: int) -> Affine:
    return Affine(res, 0.0, _OX + col0 * res, 0.0, -res, _OY - row0 * res)


def _grid(height: int, width: int, col0: int, row0: int, crs: str = "EPSG:25833") -> GeoBox:
    return GeoBox((height, width), _affine(10.0, col0, row0), crs)


class _Task:
    def __init__(self, model: nn.Module, channels: int, *, residual: bool) -> None:
        self.model = model
        self.n_active_channels = channels
        self.residual_prior = residual

    def eval(self) -> _Task:
        self.model.eval()
        return self

    def to(self, device: torch.device) -> _Task:
        self.model.to(device)
        return self


class _Zero(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            inputs.shape[0],
            1,
            inputs.shape[2],
            inputs.shape[3],
            dtype=torch.float32,
            device=inputs.device,
        )


class _PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            inputs.shape[0],
            1,
            inputs.shape[2],
            inputs.shape[3],
            dtype=torch.float32,
            device=inputs.device,
        )


class _StampCentre(nn.Module):
    """Broadcast the feature value at the tile's centre pixel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        centre = inputs[:, 0, inputs.shape[-2] // 2, inputs.shape[-1] // 2]
        return (
            centre.reshape(-1, 1, 1, 1)
            .expand(-1, 1, inputs.shape[-2], inputs.shape[-1])
            .contiguous()
        )


def _load_script(name: str):
    path = _REPO / "scripts" / "operators" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_under_test"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiles_partition_the_scene_and_keep_the_earlier_centre() -> None:
    tiles = plan_tiles(160, 200, 160, 80)
    by_col = {tile.col0: (tile.out_col0, tile.out_col1) for tile in tiles}
    assert by_col == {0: (0, 120), 40: (120, 200)}
    assert {tile.row0 for tile in tiles} == {0}
    cover = np.zeros((160, 200), dtype=np.int16)
    for tile in tiles:
        cover[tile.out_row0 : tile.out_row1, tile.out_col0 : tile.out_col1] += 1
    assert int(cover.min()) == 1
    assert int(cover.max()) == 1


def test_tile_origins_stay_on_the_100m_lattice_including_a_ragged_edge() -> None:
    wide = plan_tiles(160, 4699, 160, 80)
    assert max(tile.col0 for tile in wide) == 4540
    assert all(tile.col0 % 10 == 0 and tile.row0 % 10 == 0 for tile in wide)
    last = plan_tiles(80, 250, 80, 40)
    assert max(tile.col0 for tile in last) == 170
    with pytest.raises(ValueError, match="16"):
        plan_tiles(80, 80, 150, 80)
    with pytest.raises(ValueError, match="lattice"):
        plan_tiles(80, 80, 160, 64)


def test_centre_crop_selects_the_owning_tile() -> None:
    height, width = 160, 200
    features = np.zeros((1, height, width), dtype=np.float32)
    features[0, :, :] = np.arange(width, dtype=np.float32)[None, :]
    prior = np.full((height, width), PRIOR_PLACEHOLDER_K, dtype=np.float32)
    valid = np.ones((height, width), dtype=np.uint8)
    out = predict_scene(
        _Task(_StampCentre(), 1, residual=False),
        features,
        prior,
        feature_valid=valid,
        tile_px=160,
        stride_px=80,
        device="cpu",
        batch_size=1,
    )
    assert np.all(out[:, :120] == 80)
    assert np.all(out[:, 120:] == 120)


def test_native_and_block_priors_follow_the_global_lattice() -> None:
    st = np.full((4, 6), np.nan, dtype=np.float32)
    flag = np.full((4, 6), 2, dtype=np.uint8)
    st[0, 0], flag[0, 0] = 300, 0  # global (8, 8)
    st[0, 1], flag[0, 1] = 310, 0  # global (8, 9) → block (0, 0) mean 305
    st[0, 2], flag[0, 2] = 320, 1  # global (8, 10) flagged
    st[1, 2], flag[1, 2] = 100, 0  # global (9, 10) outside 150–400 K
    st[2, 2], flag[2, 2] = 330, 0  # global (10, 10) → block (1, 1)
    flag[2, 0] = 0  # global (10, 8) stays NaN
    transform = _affine(100.0, 8, 8)
    grid = _grid(40, 60, 80, 80)
    native = build_prior_10m(st, flag, transform, grid, "native100")
    block = build_prior_10m(st, flag, transform, grid, "block1000")
    assert native[0, 0] == np.float32(300)
    assert block[0, 0] == np.float32(305)
    assert np.isnan(native[10, 0])
    assert block[10, 0] == np.float32(305)
    assert np.isnan(native[0, 20]) and np.isnan(block[0, 20])
    assert native[20, 20] == np.float32(330)
    assert block[20, 20] == np.float32(330)
    assert not np.any(native == 0) and not np.any(block == 0)
    # The 10 m window is only cell (9, 8). Its block mean still includes (8, 8) and (8, 9).
    excerpt = _grid(10, 10, 80, 90)
    assert np.isnan(build_prior_10m(st, flag, transform, excerpt, "native100")).all()
    assert np.all(build_prior_10m(st, flag, transform, excerpt, "block1000") == np.float32(305))


def test_prior_refuses_a_grid_that_is_not_the_canonical_lattice() -> None:
    st = np.full((2, 2), 300, dtype=np.float32)
    flag = np.zeros((2, 2), dtype=np.uint8)
    transform = _affine(100.0, 0, 0)
    with pytest.raises(ValueError, match="EPSG:25833"):
        build_prior_10m(st, flag, transform, _grid(10, 10, 0, 0, crs="EPSG:4326"))
    shifted = Affine(10.0, 0.0, _OX + 4.0, 0.0, -10.0, _OY)
    with pytest.raises(ValueError, match="integer"):
        build_prior_10m(st, flag, transform, GeoBox((10, 10), shifted, "EPSG:25833"))
    south = Affine(100.0, 0.0, _OX, 0.0, 100.0, _OY)
    with pytest.raises(ValueError, match="south-up"):
        build_prior_10m(st, flag, south, _grid(10, 10, 0, 0))


def test_zero_residual_equals_the_prior_and_masks_everything_else() -> None:
    features = np.zeros((1, 80, 80), dtype=np.float32)
    prior = np.full((80, 80), np.nan, dtype=np.float32)
    prior[:, :40] = PRIOR_PLACEHOLDER_K
    valid = np.zeros((80, 80), dtype=np.uint8)
    valid[:, :50] = 1
    out = predict_scene(
        _Task(_Zero(), 1, residual=True),
        features,
        prior,
        feature_valid=valid,
        tile_px=80,
        stride_px=80,
        device="cpu",
        batch_size=2,
    )
    assert np.all(out[:, :40] == np.float32(PRIOR_PLACEHOLDER_K))
    assert np.isnan(out[:, 40:]).all()
    leaked = predict_scene(
        _Task(_PlusOne(), 1, residual=True),
        features,
        prior,
        feature_valid=valid,
        tile_px=80,
        stride_px=80,
        device="cpu",
        batch_size=1,
    )
    assert np.all(leaked[:, :40] == np.float32(PRIOR_PLACEHOLDER_K + 1))
    assert np.isnan(leaked[:, 40:]).all()


def test_non_finite_model_output_on_a_valid_pixel_raises() -> None:
    class _Nan(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]),
                torch.nan,
                dtype=torch.float32,
            )

    features = np.zeros((1, 80, 80), dtype=np.float32)
    prior = np.full((80, 80), PRIOR_PLACEHOLDER_K, dtype=np.float32)
    valid = np.ones((80, 80), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="non-finite"):
        predict_scene(
            _Task(_Nan(), 1, residual=False),
            features,
            prior,
            feature_valid=valid,
            tile_px=80,
            stride_px=80,
            device="cpu",
            batch_size=1,
        )


def test_score_matches_the_export_reduction_and_reproduction_tolerance() -> None:
    prediction = np.full((160, 160), 500, dtype=np.float32)
    target = np.full((16, 16), 290, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True
    mask[1, 2] = True
    error_sum, cells = score_patch_like_export(prediction, target, mask)
    assert cells == 2
    assert error_sum == pytest.approx(2 * 110)
    assert relative_difference(1.00005, 1.0) == pytest.approx(5e-5)
    assert relative_difference(1.0, 0.0) == float("inf")
    assert (
        reproduction_mismatch(got_sum=1.0, got_cells=3, reference_sum=1.0, reference_cells=3)
        is None
    )
    assert reproduction_mismatch(got_sum=1.0002, got_cells=3, reference_sum=1.0, reference_cells=3)
    assert reproduction_mismatch(got_sum=1.0, got_cells=2, reference_sum=1.0, reference_cells=3)


def test_validation_gate_refuses_every_other_split(tmp_path: Path) -> None:
    table = pa.table({"patch_id": ["ok", "held"], "split": ["validation", "test"]})
    pq.write_table(table, tmp_path / "patch_index.parquet")
    assert require_validation_patches(str(tmp_path), ["ok"]) == {"ok": "validation"}
    with pytest.raises(ValueError, match="test"):
        require_validation_patches(str(tmp_path), ["ok", "held"])
    with pytest.raises(ValueError, match="unique"):
        require_validation_patches(str(tmp_path), ["ok", "ok"])


def test_seed_token_is_read_from_the_checkpoint_uri() -> None:
    uri = "gs://bucket/v6m-070b33c-l4-seed2-ep40-cos/best.ckpt"
    assert seed_from_checkpoint_name(uri) == 2
    assert seed_from_checkpoint_name("best.ckpt") == 0
    with pytest.raises(ValueError, match="more than one seed"):
        seed_from_checkpoint_name("seed1-and-seed2-best.ckpt")


def test_done_feature_scenes_come_from_the_ledger(tmp_path: Path) -> None:
    root = tmp_path / "features"
    uri = ledger_path(str(root))
    Path(uri).parent.mkdir(parents=True)
    missing = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        scenes_with_done_features(str(missing))
    ledger = SecondaryLedger.open(uri)
    done = "LC09_L2SP_193023_20240601"
    ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_done",
            source="feature_stack",
            period_or_vintage=done,
            status="done",
        )
    )
    ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_pending",
            source="feature_stack",
            period_or_vintage="LC09_L2SP_193024_20240601",
            status="pending",
        )
    )
    assert scenes_with_done_features(str(root)) == [done]
    ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_done_again",
            source="feature_stack",
            period_or_vintage=done,
            status="done",
        )
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        scenes_with_done_features(str(root))


def test_cli_scene_selection_and_device(tmp_path: Path) -> None:
    cli = _load_script("run_berlin_inference.py")
    with pytest.raises(SystemExit):
        cli.main(["--checkpoint", "x.ckpt", "--config-name", "v6_l4_locked", "--out", "out"])
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--checkpoint",
                "x.ckpt",
                "--config-name",
                "v6_l4_locked",
                "--scenes",
                "LC09_L2SP_193023_20240601",
                "--year",
                "2024",
                "--out",
                "out",
            ]
        )
    root = tmp_path / "features"
    uri = ledger_path(str(root))
    Path(uri).parent.mkdir(parents=True)
    ledger = SecondaryLedger.open(uri)
    ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_2024",
            source="feature_stack",
            period_or_vintage="LC09_L2SP_193023_20240601",
            status="done",
        )
    )
    ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_bad",
            source="feature_stack",
            period_or_vintage="not-a-scene",
            status="done",
        )
    )
    with pytest.raises(ValueError, match="scene id"):
        cli.select_scene_ids(str(root), scenes=None, year=2024)
    clean = tmp_path / "clean"
    clean_uri = ledger_path(str(clean))
    Path(clean_uri).parent.mkdir(parents=True)
    clean_ledger = SecondaryLedger.open(clean_uri)
    scene = "LC09_L2SP_193023_20240601"
    clean_ledger.upsert(
        SecondaryLedgerRow(
            item_id="feature_2024",
            source="feature_stack",
            period_or_vintage=scene,
            status="done",
        )
    )
    assert cli.select_scene_ids(str(clean), scenes=None, year=2024) == [scene]
    with pytest.raises(RuntimeError, match="2023"):
        cli.select_scene_ids(str(clean), scenes=None, year=2023)
    assert cli.select_scene_ids(str(clean), scenes=[scene], year=None) == [scene]
    assert cli.require_device("cpu").type == "cpu"


def test_cuda_request_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_script("run_berlin_inference.py")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        cli.require_device("cuda")


def test_reproduction_cli_imports_and_requires_its_arguments() -> None:
    script = _load_script("check_inference_reproduction.py")
    with pytest.raises(SystemExit):
        script.main([])


def test_locked_config_keeps_the_pinned_roots() -> None:
    cli = _load_script("run_berlin_inference.py")
    cfg = cli.compose_inference_config("v6_l4_locked", seed=2)
    assert str(cfg.features_root).startswith("gs://")
    assert str(cfg.ard_root).startswith("gs://")
    assert str(cfg.training_root).endswith("training/v3")
    assert int(cfg.trainer.max_epochs) == 40
    assert str(cfg.trainer.lr_schedule.name) == "cosine"
    assert cfg.evaluate_test is False
    assert "seed-2" in str(cfg.output_root).split("/")


def _write_tif(
    path: Path, array: np.ndarray, transform: Affine, *, crs: str = "EPSG:25833"
) -> None:
    data = np.asarray(array)
    bands = data[None] if data.ndim == 2 else data
    profile: dict[str, object] = {
        "driver": "GTiff",
        "height": bands.shape[1],
        "width": bands.shape[2],
        "count": bands.shape[0],
        "dtype": bands.dtype,
        "crs": CRS.from_epsg(int(crs.split(":")[1])),
        "transform": transform,
    }
    if np.issubdtype(bands.dtype, np.floating):
        profile["nodata"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)


def _provenance():
    from berlin_lst_downscaling.modeling.inference import InferenceProvenance

    return InferenceProvenance(
        checkpoint_uri="local/best.ckpt",
        checkpoint_sha256="abc",
        config_name="v6_l4_locked",
        arm="L4",
        seed=2,
        code_commit="abc123",
        scaler_sha256="def",
    )


def test_run_scene_writes_a_cog_and_refuses_a_bad_grid(tmp_path: Path) -> None:
    from berlin_lst_downscaling.modeling.inference import _exact_canonical_offset

    grid = canon_grid_10m()
    col0, row0 = _exact_canonical_offset(grid.transform, 10.0)
    size = 80
    cell_c = col0 // 10
    cell_r = row0 // 10
    n_col = (col0 + size + 9) // 10 - cell_c
    n_row = (row0 + size + 9) // 10 - cell_r
    features = tmp_path / "features.tif"
    mask = tmp_path / "feature_valid.tif"
    landsat = tmp_path / "st.tif"
    flag = tmp_path / "flag.tif"
    _write_tif(features, np.zeros((1, size, size), dtype=np.float32), grid.transform)
    _write_tif(mask, np.ones((size, size), dtype=np.uint8), grid.transform)
    hundred = _affine(100.0, cell_c, cell_r)
    _write_tif(landsat, np.full((n_row, n_col), PRIOR_PLACEHOLDER_K, dtype=np.float32), hundred)
    _write_tif(flag, np.zeros((n_row, n_col), dtype=np.uint8), hundred)
    scaler = ScalerSpec(
        channel_order=("b0",),
        channels=(ScalerChannel(1, "b0", "identity", None, None),),
        policy_hash="test",
    )
    selection = ActiveChannelSelection(names=("b0",), indices=(0,))
    payload = run_scene(
        _Task(_Zero(), 1, residual=True),
        scene_id="LC09_L2SP_193023_20240601",
        features_uri=str(features),
        feature_valid_uri=str(mask),
        landsat_cog=str(landsat),
        landsat_flag=str(flag),
        scaler=scaler,
        selection=selection,
        out_dir=str(tmp_path / "out"),
        provenance=_provenance(),
        prior_mode="native100",
        device="cpu",
        batch_size=1,
    )
    assert payload["output_valid_px"] == size * size
    assert payload["mean_abs_output_minus_prior_10m"] == pytest.approx(0.0, abs=1e-5)
    native = payload["native_100m"]
    assert isinstance(native, dict)
    assert native["count"] == (size // 10) * (size // 10)
    product = (
        tmp_path / "out" / "LC09_L2SP_193023_20240601" / "LC09_L2SP_193023_20240601.lst10m.tif"
    )
    with rasterio.open(product) as src:
        assert src.crs.to_epsg() == 25833
        assert src.dtypes == ("float32",)
        written = src.read(1)
    assert np.allclose(written, PRIOR_PLACEHOLDER_K)
    report = json.loads(product.with_name("LC09_L2SP_193023_20240601.inference.json").read_text())
    assert report["prior_mode"] == "native100"
    assert report["checkpoint_sha256"] == "abc"
    bad = tmp_path / "bad_crs.tif"
    _write_tif(bad, np.zeros((1, size, size), dtype=np.float32), grid.transform, crs="EPSG:4326")
    with pytest.raises(RuntimeError, match="EPSG:25833"):
        run_scene(
            _Task(_Zero(), 1, residual=True),
            scene_id="LC09_L2SP_193023_20240601",
            features_uri=str(bad),
            feature_valid_uri=str(mask),
            landsat_cog=str(landsat),
            landsat_flag=str(flag),
            scaler=scaler,
            selection=selection,
            out_dir=str(tmp_path / "out2"),
            provenance=_provenance(),
            device="cpu",
            batch_size=1,
        )


def test_patch_bbox_is_the_reader_window() -> None:
    ref = PatchRef(
        patch_id="p",
        scene_id="LC09_L2SP_193023_20240601",
        s2_scene_id="S2",
        split="validation",
        year=2024,
        row=12,
        col=34,
        n_eligible=1,
        eligibility_mask="mask.tif",
    )
    assert patch_bbox_10m(ref, col0_10=4, row0_10=2) == (34 * 10 - 4, 12 * 10 - 2, 160, 160)
    # The affine round-trip used by the zero model is exact at the placeholder.
    assert prior_model_channel(np.float32(PRIOR_PLACEHOLDER_K)) == np.float32(0)


def test_mask_native_invalid_matches_native_prior_support() -> None:
    """block1000 output masked to native-valid cells keeps exactly the native100 support."""
    from affine import Affine

    from berlin_lst_downscaling.common.grid import GeoBox
    from berlin_lst_downscaling.modeling.inference import build_prior_10m

    st = np.full((2, 2), 300.0, dtype=np.float32)
    flag = np.zeros((2, 2), dtype=np.uint8)
    flag[0, 1] = 1
    t100 = Affine(100.0, 0.0, 369190.0, 0.0, -100.0, 5838410.0)
    grid = GeoBox((20, 20), Affine(10.0, 0.0, 369190.0, 0.0, -10.0, 5838410.0), "EPSG:25833")
    block = build_prior_10m(st, flag, t100, grid, "block1000")
    native = build_prior_10m(st, flag, t100, grid, "native100")
    masked = np.where(np.isfinite(native), block, np.nan)
    assert np.isfinite(block).all()
    assert np.isnan(masked[:10, 10:]).all()
    assert np.isfinite(masked[:10, :10]).all() and np.isfinite(masked[10:, :]).all()
