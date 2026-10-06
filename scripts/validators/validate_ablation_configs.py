#!/usr/bin/env python3
"""Smoke Tag-11 ablation configs (issue #57): compose, guard, loader select, model.

GCS-free. Verifies each stage2–5 locked profile resolves, passes the ablation
lock, selects the expected V3 channel subset (including the non-prefix stage-3
shadow set), and that :class:`RealLSTTask` accepts a batch under that width
and loss. Exit 0 on GO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.channels import (
    ABLATION_LOCKED_CONFIG_NAMES,
    ABLATION_STAGE_CHANNELS,
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
    contract_invariants,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.metrics import (
    masked_l1_loss,
    pool_10m_to_100m,
    thermal_aware_loss,
)
from berlin_lst_downscaling.modeling.patches import RealSample, collate_real_batch
from berlin_lst_downscaling.modeling.real_task import RealLSTTask

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs" / "modeling")


def _compose(config_name: str):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name=config_name)


def _fake_sample(patch_id: str = "smoke:0") -> RealSample:
    features = np.arange(28 * REAL_PATCH_PX * REAL_PATCH_PX, dtype=np.float32).reshape(
        28, REAL_PATCH_PX, REAL_PATCH_PX
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
    stage = int(cfg.ablation_stage)
    try:
        guard_modeling_config(cfg, config_name)
        contract_invariants(cfg)
        assert_ablation_lock(cfg)
    except Exception as exc:  # noqa: BLE001 — smoke reports every guard failure
        failures.append(f"{config_name}: guard failed: {exc}")
        return failures

    selection = selection_from_config(cfg)
    expected = ABLATION_STAGE_CHANNELS[stage]
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

    # Non-contiguous stage-3 must pull V3 indices 26/27, not positions 18/19.
    if stage == 3:
        got = batch.features[0, 18:20].detach().cpu().numpy()
        expect = sample.features[np.asarray(selection.indices[18:20], dtype=np.intp)]
        if not np.array_equal(got, expect):
            failures.append(f"{config_name}: stage-3 shadow selection did not use V3 indices 26/27")

    loss_name = str(cfg.contract.loss)
    task = RealLSTTask(
        n_active_channels=selection.n_active,
        base_width=32,
        depth=4,
        residual_prior=True,
        loss_name=loss_name,
    )
    task.eval()
    with torch.inference_mode():
        prediction = task(batch)
        pooled = pool_10m_to_100m(prediction)
        if loss_name == "thermal_aware":
            loss = thermal_aware_loss(pooled, batch.target_100m, batch.mask_100m)
        else:
            loss = masked_l1_loss(pooled, batch.target_100m, batch.mask_100m)
    if not torch.isfinite(loss):
        failures.append(f"{config_name}: non-finite {loss_name} loss")
    if not isinstance(batch, RealBatch):
        failures.append(f"{config_name}: collate did not return RealBatch")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    failures: list[str] = []
    for config_name in ABLATION_LOCKED_CONFIG_NAMES:
        failures.extend(_check_stage(config_name))

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
    print(f"ablation config smoke GO — {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
