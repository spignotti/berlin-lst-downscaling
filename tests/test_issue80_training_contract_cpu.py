"""Issue #80 section 5 CPU verification matrix (synthetic, offline, no fit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.training.patch_index import (
    MIN_ELIGIBLE_CELLS,
    N_CELLS,
    PatchIndexAdmissionPolicy,
)
from berlin_lst_downscaling.modeling.aggregation import (
    pool_10m_to_100m,
    pool_block_means_from_accumulators,
)
from berlin_lst_downscaling.modeling.contracts import RealBatch, RealSampleMeta
from berlin_lst_downscaling.modeling.metrics import (
    THERMAL_AWARE_GRAD_WEIGHT,
    THERMAL_AWARE_SSIM_WEIGHT,
    MaskedSSIM,
    SupportWeightedSum,
    _gradient_candidate_pair_count,
    _ssim_candidate_window_count,
    masked_abs_error_sums,
    masked_gradient_stats,
    masked_l1_loss,
    masked_ssim_stats,
    prediction_outside_lst_range_counts,
    thermal_objective_batch,
)
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, _json_digest
from berlin_lst_downscaling.modeling.patches import _build_scene_prior
from berlin_lst_downscaling.modeling.real_task import RealLSTTask

_ARITH_300 = 300.0
_T4_REF = float(((280.0**4 + 320.0**4) / 2.0) ** 0.25)
_T4_F32_TOL = 1e-5


def _mixed_10m_block(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    warm = torch.full((50,), 280.0, dtype=dtype)
    cold = torch.full((50,), 320.0, dtype=dtype)
    return torch.cat([warm, cold]).reshape(1, 1, 10, 10)


def test_analytic_10m_pooling_280_320_k() -> None:
    block = _mixed_10m_block()
    assert float(pool_10m_to_100m(block, "arithmetic")) == pytest.approx(_ARITH_300)
    assert float(pool_10m_to_100m(block, "t4")) == pytest.approx(_T4_REF, rel=_T4_F32_TOL)


def test_analytic_100m_prior_280_320_k(tmp_path: Path) -> None:
    st = np.concatenate(
        [np.full(50, 280.0, dtype=np.float32), np.full(50, 320.0, dtype=np.float32)]
    ).reshape(10, 10)
    flag = np.zeros((10, 10), dtype=np.uint8)
    lst = tmp_path / "lst.tif"
    flg = tmp_path / "flag.tif"
    for path, arr, dtype in ((lst, st, "float32"), (flg, flag, "uint8")):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=dtype,
            crs="EPSG:25833",
            transform=from_origin(369190, 5838410, 100, 100),
        ) as dst:
            dst.write(arr, 1)
    prior = _build_scene_prior("s", str(lst), str(flg), method="t4")
    assert float(prior.block_means[0, 0]) == pytest.approx(_T4_REF, rel=_T4_F32_TOL)
    prior_a = _build_scene_prior("s", str(lst), str(flg), method="arithmetic")
    assert float(prior_a.block_means[0, 0]) == pytest.approx(_ARITH_300)


def test_prior_accumulator_path_matches_closed_form() -> None:
    counts = np.array([[100]], dtype=np.int64)
    sum_t = np.array([[300.0 * 100]], dtype=np.float64)
    sum_t4 = np.array([[((280.0**4 + 320.0**4) / 2.0) * 100]], dtype=np.float64)
    out = pool_block_means_from_accumulators(sum_t, sum_t4, counts, method="t4")
    assert float(out[0, 0]) == pytest.approx(_T4_REF, rel=1e-6)


def test_t4_promotes_half_and_bfloat16() -> None:
    block = _mixed_10m_block(dtype=torch.float16)
    pooled = pool_10m_to_100m(block, "t4")
    assert pooled.dtype == torch.float32
    assert float(pooled) == pytest.approx(_T4_REF, rel=_T4_F32_TOL)
    block_bf = _mixed_10m_block(dtype=torch.bfloat16)
    assert float(pool_10m_to_100m(block_bf, "t4")) == pytest.approx(_T4_REF, rel=_T4_F32_TOL)


def test_t4_backward_is_finite_and_deterministic() -> None:
    block = _mixed_10m_block().clone().requires_grad_(True)
    out_a = pool_10m_to_100m(block, "t4")
    out_a.backward()
    grad_a = block.grad.detach().clone()
    block.grad = None
    out_b = pool_10m_to_100m(block, "t4")
    out_b.backward()
    assert bool(torch.isfinite(block.grad).all())
    assert torch.allclose(grad_a, block.grad)


def test_residual_pool_prior_plus_residual_not_residual_plus_prior() -> None:
    prior = torch.full((1, 1, 10, 10), 280.0)
    residual = torch.cat([torch.full((50,), 0.0), torch.full((50,), 40.0)]).reshape(1, 1, 10, 10)
    combined = prior + residual
    correct = float(pool_10m_to_100m(combined, "t4"))
    wrong = float(pool_10m_to_100m(prior, "t4")) + float(pool_10m_to_100m(residual, "arithmetic"))
    assert correct == pytest.approx(_T4_REF, rel=_T4_F32_TOL)
    assert wrong == pytest.approx(300.0)
    assert correct != pytest.approx(wrong, rel=1e-4)


def test_zero_residual_head_identity_per_method() -> None:
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    prior_10m = _mixed_10m_block()
    for method in ("arithmetic", "t4"):
        pooled_prior = pool_10m_to_100m(prior_10m, method, mask)
        pooled_pred = pool_10m_to_100m(prior_10m + torch.zeros_like(prior_10m), method, mask)
        assert float(pooled_pred) == pytest.approx(float(pooled_prior))


def test_masked_l1_hand_numerator_and_count() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[0.0, 5.0], [1.0, 2.0]]]])
    mask = torch.tensor([[[[True, True], [True, False]]]])
    total, count = masked_abs_error_sums(pred, target, mask)
    assert float(count) == 3.0
    assert float(total) == 6.0
    assert float(masked_l1_loss(pred, target, mask)) == pytest.approx(2.0)


def test_masked_l1_rejects_all_invalid_and_isolates_nan_in_backward() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[[[0.0, float("nan")], [1.0, 2.0]]]])
    mask = torch.tensor([[[[True, False], [True, True]]]])
    loss = masked_l1_loss(pred, target, mask)
    loss.backward()
    assert bool(torch.isfinite(pred.grad).all())
    with pytest.raises(ValueError, match="no valid"):
        masked_l1_loss(pred, target, torch.zeros_like(mask))


def test_ssim_all_valid_16x16_window_counts() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = torch.full((1, 1, 16, 16), 301.0)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    assert _ssim_candidate_window_count(pred) == 100
    _, supported = masked_ssim_stats(pred, target, mask)
    assert int(supported) == 100


def test_ssim_holes_exclude_touching_windows() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = pred.clone()
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    mask[0, 0, 7, 7] = False
    _, supported_center = masked_ssim_stats(pred, target, mask)
    assert int(supported_center) == 100 - 49
    mask[0, 0, 0, :] = False
    mask[0, 0, :, 0] = False
    _, supported_border = masked_ssim_stats(pred, target, mask)
    assert int(supported_border) < int(supported_center)


def test_ssim_zero_support_is_unavailable_not_free_penalty() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = torch.full((1, 1, 16, 16), 310.0)
    mask = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    mask[0, 0, 15, 15] = True
    batch = thermal_objective_batch(pred, target, mask)
    assert float(batch.ssim_supported) == 0.0
    assert float(batch.ssim_penalty_unweighted) == 0.0
    assert float(batch.loss) == pytest.approx(float(batch.l1_unweighted))


def _unsupported_mask() -> torch.Tensor:
    mask = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    mask[0, 0, 15, 15] = True
    return mask


def test_unsupported_patch_is_excluded_from_ssim_batch_mean() -> None:
    """A patch with no valid SSIM window does not contribute 0 to the batch mean.

    L1 still includes that patch. An all-unsupported batch keeps ssim_supported
    at 0 and leaves the SSIM mean unavailable.
    """
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = torch.full((1, 1, 16, 16), 301.0)
    mask_ok = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    alone = thermal_objective_batch(pred, target, mask_ok)

    mixed_pred = torch.cat([pred, pred + 40.0], dim=0)
    mixed_target = torch.cat([target, target + 5.0], dim=0)
    mixed_mask = torch.cat([mask_ok, _unsupported_mask()], dim=0)
    mixed = thermal_objective_batch(mixed_pred, mixed_target, mixed_mask)
    assert float(mixed.ssim_supported) == float(alone.ssim_supported)
    assert float(mixed.ssim_sum / mixed.ssim_supported) == pytest.approx(
        float(alone.ssim_sum / alone.ssim_supported)
    )
    assert float(mixed.l1_count) > float(alone.l1_count)
    assert float(mixed.l1_unweighted) != pytest.approx(float(alone.l1_unweighted))

    empty = thermal_objective_batch(pred, target, _unsupported_mask())
    assert float(empty.ssim_supported) == 0.0
    assert float(empty.loss) == pytest.approx(float(empty.l1_unweighted))
    metric = MaskedSSIM()
    if float(empty.ssim_supported) > 0.0:
        metric.update(empty.ssim_sum, empty.ssim_supported)
    assert float(metric.window_count) == 0.0
    metric.update(alone.ssim_sum, alone.ssim_supported)
    assert float(metric.compute()) == pytest.approx(float(alone.ssim_sum / alone.ssim_supported))

    task = RealLSTTask(loss_name="thermal_aware", n_active_channels=1, depth=1, base_width=4)
    logged: dict[str, object] = {}

    def _log(name: str, value: object, **_kwargs: object) -> None:
        logged[name] = value

    task.log = _log  # type: ignore[method-assign]
    task._accumulate_thermal_audit(empty, train=True)
    assert "train/ssim_supported" in logged
    assert float(logged["train/ssim_supported"]) == 0.0  # type: ignore[arg-type]
    assert float(task.train_thermal_ssim.denominator) == 0.0


def test_eval_ssim_is_unclamped_and_logs_outside_range_share() -> None:
    """Validation SSIM uses raw predictions. The training objective still clamps."""
    pred = torch.full((1, 1, 16, 16), 500.0)
    target = torch.full((1, 1, 16, 16), 300.0)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    raw, raw_count = masked_ssim_stats(pred, target, mask, clamp_predictions=False)
    clamped, clamped_count = masked_ssim_stats(pred, target, mask, clamp_predictions=True)
    assert int(raw_count) == int(clamped_count) > 0
    assert float(raw) != pytest.approx(float(clamped))

    capped = torch.full_like(pred, LST_RANGE_K[1])
    capped_sum, _ = masked_ssim_stats(capped, target, mask, clamp_predictions=False)
    trained = thermal_objective_batch(pred, target, mask)
    assert float(trained.ssim_sum) == pytest.approx(float(capped_sum))

    outside, total = prediction_outside_lst_range_counts(pred)
    assert float(outside / total) == pytest.approx(1.0)
    half = pred.clone()
    half[:, :, :, :8] = 300.0
    outside_half, total_half = prediction_outside_lst_range_counts(half)
    assert float(outside_half / total_half) == pytest.approx(0.5)

    task = RealLSTTask(loss_name="masked_l1", n_active_channels=1, depth=1, base_width=4)

    def _predict(_batch: RealBatch) -> torch.Tensor:
        return torch.full((1, 1, 160, 160), 500.0)

    task.forward = _predict  # type: ignore[method-assign]
    logged: set[str] = set()

    def _log(name: str, _value: object, **_kwargs: object) -> None:
        logged.add(name)

    task.log = _log  # type: ignore[method-assign]
    meta = RealSampleMeta(
        patch_id="s:E0N0",
        scene_id="s",
        split="validation",
        year=2024,
        row=0,
        col=0,
        filled_feature_pixels=0,
    )
    batch = RealBatch(
        features=torch.zeros(1, 1, 160, 160),
        lst_prior=torch.zeros(1, 1, 160, 160),
        target_100m=torch.full((1, 1, 16, 16), 300.0),
        mask_100m=torch.ones(1, 1, 16, 16, dtype=torch.bool),
        metadata=[meta],
    )
    task.validation_step(batch, 0)
    assert "validation/ssim_supported" in logged
    assert "validation/pred_outside_lst_range_frac" in logged
    assert float(task.val_pred_outside_lst_range.compute()) == pytest.approx(1.0)
    pooled = torch.full((1, 1, 16, 16), 500.0)
    expected, _ = masked_ssim_stats(
        pooled, batch.target_100m, batch.mask_100m, clamp_predictions=False
    )
    assert float(task.val_ssim.compute()) == pytest.approx(float(expected / raw_count))


def test_gradient_support_16x16_all_valid() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = pred.clone()
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    assert _gradient_candidate_pair_count(pred) == 480
    _, supported, candidate = masked_gradient_stats(pred, target, mask)
    assert int(candidate) == 480
    assert int(supported) == 480


def test_gradient_isolated_valid_cell_has_no_supported_edges() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = pred.clone()
    mask = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    mask[0, 0, 8, 8] = True
    _, supported, _ = masked_gradient_stats(pred, target, mask)
    assert int(supported) == 0


def test_gradient_holes_and_excluded_value_invariance() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = pred.clone()
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    mask[0, 0, 4:8, 4:8] = False
    _, supported_hole, _ = masked_gradient_stats(pred, target, mask)
    pred_dirty = pred.clone()
    pred_dirty[~mask] = -999.0
    _, supported_dirty, _ = masked_gradient_stats(pred_dirty, target, mask)
    assert int(supported_hole) < 480
    assert int(supported_dirty) == int(supported_hole)


def test_thermal_objective_audit_weighted_sum() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = torch.full((1, 1, 16, 16), 301.0)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    batch = thermal_objective_batch(pred, target, mask)
    expected = (
        batch.l1_unweighted
        + THERMAL_AWARE_SSIM_WEIGHT * batch.ssim_penalty_unweighted
        + THERMAL_AWARE_GRAD_WEIGHT * batch.gradient_unweighted
    )
    assert float(batch.loss) == pytest.approx(float(expected))


def test_support_weighted_epoch_mean_invariant_to_batch_partition() -> None:
    pred = torch.full((1, 1, 16, 16), 300.0)
    target = torch.full((1, 1, 16, 16), 305.0)
    mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
    whole = thermal_objective_batch(pred, target, mask)
    left = thermal_objective_batch(pred[:, :, :, :8], target[:, :, :, :8], mask[:, :, :, :8])
    right = thermal_objective_batch(pred[:, :, :, 8:], target[:, :, :, 8:], mask[:, :, :, 8:])
    metric = SupportWeightedSum()
    metric.update(whole.l1_sum, whole.l1_count)
    split = SupportWeightedSum()
    split.update(left.l1_sum, left.l1_count)
    split.update(right.l1_sum, right.l1_count)
    assert float(metric.compute()) == pytest.approx(float(split.compute()))


def _write_minimal_patch_cache(tmp_path: Path, *, aggregation: str) -> Path:
    root = tmp_path / "cache"
    root.mkdir()
    n = 1
    ch = 1
    shapes = {
        "features": [n, ch, 160, 160],
        "prior": [n, 1, 160, 160],
        "target": [n, 1, 16, 16],
        "mask": [n, 1, 16, 16],
    }
    provenance: dict[str, object] = {
        "patch_index_root": "local",
        "training_root": "local",
        "features_root": "local",
        "ard_root": "local",
        "aggregation": aggregation,
        "channel_order": ["c0"],
    }
    np.save(root / "features.npy", np.zeros(shapes["features"], dtype=np.float32))
    np.save(root / "prior.npy", np.zeros(shapes["prior"], dtype=np.float32))
    np.save(root / "target.npy", np.zeros(shapes["target"], dtype=np.float32))
    np.save(root / "mask.npy", np.ones(shapes["mask"], dtype=bool))
    names = ("features.npy", "prior.npy", "target.npy", "mask.npy")
    array_bytes = sum((root / name).stat().st_size for name in names)
    meta = {
        "patch_id": "p0",
        "scene_id": "s",
        "split": "train",
        "year": 2020,
        "row": 0,
        "col": 0,
        "filled_feature_pixels": 1,
    }
    manifest = {
        "version": 1,
        "provenance": provenance,
        "provenance_sha256": _json_digest(provenance),
        "patch_ids": ["p0"],
        "patch_ids_sha256": _json_digest(["p0"]),
        "metadata": [meta],
        "active_channels": ch,
        "shapes": shapes,
        "dtypes": {"features": "float32", "prior": "float32", "target": "float32", "mask": "bool"},
        "array_bytes": array_bytes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_patch_cache_rejects_cross_method_provenance(tmp_path: Path) -> None:
    root = _write_minimal_patch_cache(tmp_path, aggregation="arithmetic")
    provenance = json.loads((root / "manifest.json").read_text())["provenance"]
    PatchCache(root, provenance)
    other = dict(provenance)
    other["aggregation"] = "t4"
    with pytest.raises(RuntimeError, match="provenance"):
        PatchCache(root, other)


def test_legacy_checkpoint_without_aggregation_defaults_arithmetic(tmp_path: Path) -> None:
    import lightning

    task = RealLSTTask(loss_name="masked_l1")
    ckpt = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": lightning.__version__,
            "state_dict": task.state_dict(),
            "hyper_parameters": {
                "n_active_channels": task.n_active_channels,
                "base_width": 32,
                "depth": 4,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "record_probe_metrics": False,
                "residual_prior": False,
                "loss_name": "masked_l1",
            },
        },
        ckpt,
    )
    loaded = RealLSTTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    assert loaded.aggregation == "arithmetic"


def test_index_threshold_boundaries() -> None:
    strict = PatchIndexAdmissionPolicy()
    train_230 = PatchIndexAdmissionPolicy(train_min_eligible_fraction=230 / N_CELLS)
    train_200 = PatchIndexAdmissionPolicy(train_min_eligible_fraction=200 / N_CELLS)

    def admitted(n: int, split: str, policy: PatchIndexAdmissionPolicy) -> bool:
        return n >= policy.min_eligible_cells_for_split(split)

    assert not admitted(243, "validation", strict)
    assert admitted(244, "validation", strict)
    assert not admitted(229, "train", train_230)
    assert admitted(230, "train", train_230)
    assert not admitted(199, "train", train_200)
    assert admitted(200, "train", train_200)
    for policy in (strict, train_230, train_200):
        assert policy.min_eligible_cells_for_split("validation") == MIN_ELIGIBLE_CELLS
        assert policy.min_eligible_cells_for_split("test") == MIN_ELIGIBLE_CELLS
        assert not admitted(243, "validation", policy)
        assert admitted(244, "test", policy)


def test_stage5_logging_key_names_cover_train_and_validation() -> None:
    expected = {
        "train/thermal_l1_unweighted",
        "train/thermal_l1_weighted",
        "train/thermal_ssim_unweighted",
        "train/thermal_ssim_weighted",
        "train/thermal_grad_unweighted",
        "train/thermal_grad_weighted",
        "validation/thermal_l1_unweighted",
        "validation/thermal_l1_weighted",
        "validation/thermal_ssim_unweighted",
        "validation/thermal_ssim_weighted",
        "validation/thermal_grad_unweighted",
        "validation/thermal_grad_weighted",
    }
    task = RealLSTTask(loss_name="thermal_aware")
    logged: set[str] = set()

    class _Logger:
        def log(self, name: str, *_args, **_kwargs) -> None:
            logged.add(name)

    task.log = _Logger().log  # type: ignore[method-assign]
    task._log_thermal_batch_metrics(train=True)
    task._log_thermal_batch_metrics(train=False)
    assert expected.issubset(logged)
