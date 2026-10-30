#!/usr/bin/env python3
"""Smoke Tag-11 ablation configs (issues #57 and #63).

GCS-free. Verifies each locked cumulative stage and each isolation profile
resolves, passes the ablation lock, selects the expected V3 channel subset
(including non-prefix shadow sets), and that :class:`RealLSTTask` accepts a
batch under that width and loss. Also checks the v6 ablation arms and the v6 run
matrix (offline). Each v6 arm runs a CPU forward/loss check on synthetic train and
validation batches through the uncached collation path, the job-local patch cache
built with bounded ordered reads, or both (``--cache-mode``). Exit 0 on GO.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.channels import (
    ABLATION_ISOLATION_CHANNELS,
    ABLATION_LOCKED_CONFIG_NAMES,
    ABLATION_STAGE_CHANNELS,
    V6_ABLATION_ARMS,
    feature_order_for,
    schema_version_from_config,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealBatch,
    RealSampleMeta,
)
from berlin_lst_downscaling.modeling.guards import (
    assert_ablation_lock,
    cache_build_workers_from_config,
    contract_invariants,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.metrics import (
    masked_l1_loss,
    pool_10m_to_100m,
    thermal_aware_loss,
)
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    build_patch_cache,
    ordered_bounded_map,
)
from berlin_lst_downscaling.modeling.patches import RealSample, collate_real_batch
from berlin_lst_downscaling.modeling.real_task import RealLSTTask
from berlin_lst_downscaling.modeling.v6_ablation_matrix import (
    load_v6_matrix,
    plan_runs,
    plan_totals,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs" / "modeling")
_MATRIX_PATH = _REPO_ROOT / "configs" / "ablation" / "v6_matrix.yaml"
_V6_SOURCE_CHANNELS = 22
_CACHE_MODES = ("both", "off", "on")
# Wiring-only SSIM data range on fake tensors; the production L comes from the committed artifact.
_WIRING_SSIM_RANGE_K = 40.0


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def _fake_sample(patch_id: str = "smoke:0", n_channels: int = 28) -> RealSample:
    features = np.arange(n_channels * REAL_PATCH_PX * REAL_PATCH_PX, dtype=np.float32).reshape(
        n_channels, REAL_PATCH_PX, REAL_PATCH_PX
    )
    prior = np.zeros((1, REAL_PATCH_PX, REAL_PATCH_PX), dtype=np.float32)
    target = np.full((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), 290.0, dtype=np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    mask[:, 0, :] = False
    mask[:, :, 0] = False
    target[~mask] = np.nan
    meta = RealSampleMeta(
        patch_id=patch_id,
        scene_id="LC08_L2SP_000000_20180101",
        split="train",
        year=2018,
        row=5800,
        col=3690,
        filled_feature_pixels=0,
    )
    return RealSample(
        meta=meta,
        features=features,
        lst_prior_k=prior,
        target_100m=target,
        mask_100m=mask,
    )


def _check_stage(config_name: str) -> list[str]:
    failures: list[str] = []
    cfg = _compose(config_name)
    isolation = cfg.get("ablation_isolation")
    stage = None if isolation is not None else int(cfg.ablation_stage)
    if isolation is not None:
        expected = ABLATION_ISOLATION_CHANNELS[str(isolation)]
    else:
        expected = ABLATION_STAGE_CHANNELS[int(stage)]
    try:
        guard_modeling_config(cfg, config_name)
        contract_invariants(cfg)
        assert_ablation_lock(cfg)
    except Exception as exc:  # noqa: BLE001 — smoke reports every guard failure
        failures.append(f"{config_name}: guard failed: {exc}")
        return failures

    selection = selection_from_config(cfg)
    if selection.names != expected:
        failures.append(
            f"{config_name}: channels {list(selection.names)} != expected {list(expected)}"
        )
        return failures

    sample = _fake_sample()
    batch = collate_real_batch(
        [sample, _fake_sample("smoke:1")],
        n_active_channels=selection.n_active,
        channel_indices=selection.indices,
    )
    if tuple(batch.features.shape) != (2, selection.n_active, REAL_PATCH_PX, REAL_PATCH_PX):
        failures.append(
            f"{config_name}: collated feature shape {tuple(batch.features.shape)} "
            f"!= (2, {selection.n_active}, {REAL_PATCH_PX}, {REAL_PATCH_PX})"
        )

    # Non-contiguous shadow sets must pull V3 indices 26/27, not the next prefix.
    if stage == 3 or config_name == "isolate_shadows_locked":
        got = batch.features[0, -2:].detach().cpu().numpy()
        expect = sample.features[np.asarray(selection.indices[-2:], dtype=np.intp)]
        if not np.array_equal(got, expect):
            failures.append(f"{config_name}: shadow selection did not use the resolved V3 indices")
    if config_name == "isolate_shadows_locked" and selection.indices != (
        *range(10),
        26,
        27,
    ):
        failures.append(f"{config_name}: expected spectral prefix plus V3 indices 26/27")
    if config_name == "isolate_era5_locked" and selection.indices != (
        *range(10),
        *range(18, 26),
    ):
        failures.append(f"{config_name}: expected spectral prefix plus V3 indices 18–25")

    loss_name = str(cfg.contract.loss)
    task = RealLSTTask(
        n_active_channels=selection.n_active,
        base_width=32,
        depth=4,
        residual_prior=True,
        loss_name=loss_name,
        ssim_data_range_k=_WIRING_SSIM_RANGE_K,
    )
    task.eval()
    with torch.inference_mode():
        prediction = task(batch)
        pooled = pool_10m_to_100m(prediction)
        if loss_name == "thermal_aware":
            loss = thermal_aware_loss(
                pooled, batch.target_100m, batch.mask_100m, data_range_k=_WIRING_SSIM_RANGE_K
            )
        else:
            loss = masked_l1_loss(pooled, batch.target_100m, batch.mask_100m)
    if not torch.isfinite(loss):
        failures.append(f"{config_name}: non-finite {loss_name} loss")
    # Stage-5 regression: TorchMetrics SSIM used reflection_pad2d, whose CUDA
    # backward is banned under Trainer(deterministic=True). The pad-free map
    # must survive a deterministic backward on CPU (and on CUDA when present).
    if loss_name == "thermal_aware":
        pred = torch.randn(1, 1, REAL_PATCH_PX // 10, REAL_PATCH_PX // 10, requires_grad=True)
        target = pred.detach() + 0.1
        mask = torch.ones_like(pred)
        was_det = torch.are_deterministic_algorithms_enabled()
        was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            loss_bwd = thermal_aware_loss(pred, target, mask, data_range_k=_WIRING_SSIM_RANGE_K)
            loss_bwd.backward()
        except RuntimeError as exc:
            failures.append(
                f"{config_name}: thermal_aware backward under deterministic "
                f"algorithms failed: {exc}"
            )
        finally:
            torch.use_deterministic_algorithms(was_det, warn_only=was_warn)
        if pred.grad is None or not torch.isfinite(pred.grad).all():
            failures.append(
                f"{config_name}: thermal_aware deterministic backward produced bad grad"
            )
    if not isinstance(batch, RealBatch):
        failures.append(f"{config_name}: collate did not return RealBatch")
    return failures


def _guard_v6(cfg, config_name: str) -> None:
    guard_modeling_config(cfg, config_name)
    contract_invariants(cfg)
    assert_ablation_lock(cfg)


def _check_v6_arm(arm_id: str, cache_mode: str = "both") -> list[str]:
    arm = V6_ABLATION_ARMS[arm_id]
    name = arm.config_name
    cfg = _compose(name)
    try:
        _guard_v6(cfg, name)
    except Exception as exc:  # noqa: BLE001 — report every guard failure
        return [f"{name}: guard failed: {exc}"]
    failures: list[str] = []
    selection = selection_from_config(cfg)
    if selection.names != arm.channels:
        return [f"{name}: channels {list(selection.names)} != expected {list(arm.channels)}"]
    expected_order = "v6_registry" if arm_id in ("L4", "L5") else "v6_named_subset"
    if feature_order_for(selection, schema_version_from_config(cfg)) != expected_order:
        failures.append(f"{name}: feature order is not {expected_order}")
    if str(cfg.contract.feature_order) != expected_order:
        failures.append(f"{name}: contract.feature_order is not {expected_order}")
    task = RealLSTTask(
        n_active_channels=selection.n_active,
        base_width=32,
        depth=4,
        residual_prior=True,
        loss_name=arm.loss,
        ssim_data_range_k=_WIRING_SSIM_RANGE_K,
    )
    task.eval()
    splits = {split: _v6_split_samples(split) for split in ("train", "validation")}
    for split, samples in splits.items():
        uncached = collate_real_batch(
            samples, n_active_channels=selection.n_active, channel_indices=selection.indices
        )
        batches = {"uncached": uncached} if cache_mode in ("both", "off") else {}
        if cache_mode in ("both", "on"):
            cached = _cached_batch(cfg, samples, selection.n_active, selection.indices)
            for key in ("features", "lst_prior", "target_100m", "mask_100m"):
                if not _identical(getattr(cached, key), getattr(uncached, key)):
                    failures.append(f"{name} {split}: cached {key} differs from uncached")
            if [m.patch_id for m in cached.metadata] != [m.patch_id for m in uncached.metadata]:
                failures.append(f"{name} {split}: cached patch order differs from uncached")
            batches["cached"] = cached
        for path, batch in batches.items():
            failures.extend(
                _v6_forward_failures(
                    name, f"{split}/{path}", task, arm.loss, batch, selection.n_active
                )
            )
    return failures


def _identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if not left.is_floating_point():
        return torch.equal(left, right)
    return torch.equal(left.isnan(), right.isnan()) and torch.equal(
        left.nan_to_num(), right.nan_to_num()
    )


def _v6_split_samples(split: str) -> list[RealSample]:
    year = 2018 if split == "train" else 2024
    samples = []
    for i in range(3):
        sample = _fake_sample(f"smoke:{split}:{i}", n_channels=_V6_SOURCE_CHANNELS)
        sample.features[i % _V6_SOURCE_CHANNELS] += float(i + 1)
        samples.append(replace(sample, meta=replace(sample.meta, split=split, year=year)))
    return samples


def _cached_batch(
    cfg, samples: list[RealSample], n_active: int, indices: tuple[int, ...]
) -> RealBatch:
    provenance: dict[str, object] = {
        "channel_order": [f"c{i}" for i in indices],
        "channel_indices": list(indices),
        "source_channels": _V6_SOURCE_CHANNELS,
    }
    with tempfile.TemporaryDirectory(prefix="ablation-cache-") as tmp:
        root = Path(tmp) / "cache"
        build_patch_cache(
            root,
            ordered_bounded_map(
                lambda s: s, samples, max_workers=cache_build_workers_from_config(cfg)
            ),
            patch_ids=[s.meta.patch_id for s in samples],
            active_channels=n_active,
            source_channels=_V6_SOURCE_CHANNELS,
            provenance=provenance,
            max_bytes=int(cfg.data.cache_max_bytes),
            channel_indices=indices,
        )
        cache = PatchCache(root, provenance)
        batch = collate_real_batch(
            [cache[i] for i in range(len(cache))], n_active_channels=n_active
        )
        del cache
    return batch


def _v6_forward_failures(
    name: str, label: str, task: RealLSTTask, loss_name: str, batch: RealBatch, n_active: int
) -> list[str]:
    if tuple(batch.features.shape) != (len(batch.metadata), n_active, REAL_PATCH_PX, REAL_PATCH_PX):
        return [f"{name} {label}: collated feature shape {tuple(batch.features.shape)}"]
    with torch.inference_mode():
        pooled = pool_10m_to_100m(task(batch))
        if loss_name == "thermal_aware":
            loss = thermal_aware_loss(
                pooled, batch.target_100m, batch.mask_100m, data_range_k=_WIRING_SSIM_RANGE_K
            )
        else:
            loss = masked_l1_loss(pooled, batch.target_100m, batch.mask_100m)
    if not torch.isfinite(loss):
        return [f"{name} {label}: non-finite {loss_name} loss"]
    return []


def _check_v6_matrix() -> tuple[list[str], str]:
    failures: list[str] = []
    matrix = load_v6_matrix(_MATRIX_PATH)
    config_files = {p.stem for p in Path(_CONFIG_DIR).glob("v6_*_locked.yaml")}
    matrix_configs = {V6_ABLATION_ARMS[arm_id].config_name for arm_id in matrix.arms}
    if config_files != matrix_configs:
        failures.append(
            f"v6 matrix/config mismatch: only in configs={sorted(config_files - matrix_configs)} "
            f"only in matrix={sorted(matrix_configs - config_files)}"
        )
    roots: dict[str, set[str]] = {}
    for run in plan_runs(matrix):
        overrides = [f"seed={run.seed}", f"trainer.max_epochs={run.max_epochs}"]
        cfg = _compose(run.config_name, overrides)
        try:
            _guard_v6(cfg, run.config_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{run.config_name} {overrides}: {exc}")
            continue
        seen = roots.setdefault(run.arm_id, set())
        if str(cfg.output_root) in seen:
            failures.append(f"{run.config_name}: output_root repeats across seeds")
        seen.add(str(cfg.output_root))
    runs, hours, usd = plan_totals(matrix)
    return failures, f"v6 matrix runs={runs} est_gpu_h={hours:.1f} est_usd={usd:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-mode",
        choices=_CACHE_MODES,
        default="both",
        help="v6 arm data path to exercise: uncached collation, the patch cache, or both",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    for config_name in ABLATION_LOCKED_CONFIG_NAMES:
        failures.extend(_check_stage(config_name))

    for arm_id, arm in V6_ABLATION_ARMS.items():
        arm_failures = _check_v6_arm(arm_id, args.cache_mode)
        failures.extend(arm_failures)
        status = "FAIL" if arm_failures else "PASS"
        print(f"v6 arm {arm_id} ({arm.config_name}) cache_mode={args.cache_mode}: {status}")
    matrix_failures, matrix_summary = _check_v6_matrix()
    failures.extend(matrix_failures)

    # Stage-1 first-C path must stay untouched by named-subset wiring.
    stage1 = _compose("stage1_locked")
    try:
        guard_modeling_config(stage1, "stage1_locked")
        contract_invariants(stage1)
        selection = selection_from_config(stage1)
        if selection.names != ABLATION_STAGE_CHANNELS[1]:
            failures.append("stage1_locked channel selection drifted from first-10 V3")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"stage1_locked regression: {exc}")

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        locked_t4 = compose(config_name="stage5_locked", overrides=["aggregation=t4"])
    try:
        guard_modeling_config(locked_t4, "stage5_locked")
        failures.append("stage5_locked must reject aggregation=t4")
    except ValueError:
        pass

    if failures:
        for line in failures:
            print(f"FAIL: {line}", file=sys.stderr)
        print("ablation config smoke NO-GO", file=sys.stderr)
        return 1
    summary = ", ".join(
        f"stage{stage}=C{len(names)}"
        for stage, names in ABLATION_STAGE_CHANNELS.items()
        if stage >= 2
    )
    summary += ", " + ", ".join(
        f"isolate-{name}=C{len(names)}" for name, names in ABLATION_ISOLATION_CHANNELS.items()
    )
    summary += ", " + ", ".join(
        f"{arm_id}=C{len(arm.channels)}" for arm_id, arm in V6_ABLATION_ARMS.items()
    )
    print(f"ablation config smoke GO — {summary}; {matrix_summary}; cache_mode={args.cache_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
