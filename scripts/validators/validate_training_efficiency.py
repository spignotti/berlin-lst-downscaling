"""Run local, synthetic guard and patch-cache checks for efficiency work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import RandomSampler

from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealSampleMeta,
)
from berlin_lst_downscaling.modeling.efficiency_protocol import (
    EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD,
    EFFICIENCY_MAX_HOURLY_RATE_USD,
    EFFICIENCY_MAX_JOB_EXPOSURE_USD,
    EFFICIENCY_MAX_NONCOMPUTE_USD,
    EFFICIENCY_MAX_TOTAL_COMPUTE_USD,
    EFFICIENCY_MAX_TOTAL_USD,
    EFFICIENCY_SESSION_ID,
    EFFICIENCY_SLOT_ROLES,
    REVALIDATION_EFFICIENCY_J1_IMAGE_DIGEST,
    REVALIDATION_EFFICIENCY_J1_RESOURCE_NAME,
    REVALIDATION_EFFICIENCY_J1_RUN_LABEL,
    REVALIDATION_EFFICIENCY_J1_SOURCE_SHA,
    REVALIDATION_EFFICIENCY_J3_RESOURCE_NAME,
    REVALIDATION_EFFICIENCY_J3_RUN_LABEL,
)
from berlin_lst_downscaling.modeling.guards import (
    assert_stage1_efficiency,
    assert_stage1_efficiency_scope,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    assert_samples_equal,
    build_patch_cache,
    estimate_cache_bytes,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealSample,
    RealSourceConfig,
    collate_real_batch,
    load_patch_refs,
)
from berlin_lst_downscaling.modeling.real_task import (
    CachedRealPatchDataset,
    ProbeScope,
    RealPatchDataModule,
    RealPatchDataset,
)

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")
_EFFICIENCY_LEDGER = (
    Path(__file__).resolve().parents[2]
    / "data/runs/.stage1-efficiency-control"
    / EFFICIENCY_SESSION_ID
    / "slots.json"
)
_SLOT_ROLES = EFFICIENCY_SLOT_ROLES
_DEFAULT_BASELINE = Path(
    "docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json"
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVALIDATION_SPECS = {
    1: {
        "run_label": REVALIDATION_EFFICIENCY_J1_RUN_LABEL,
        "source_sha": REVALIDATION_EFFICIENCY_J1_SOURCE_SHA,
        "image_digest": REVALIDATION_EFFICIENCY_J1_IMAGE_DIGEST,
        "resource_name": REVALIDATION_EFFICIENCY_J1_RESOURCE_NAME,
        "reason": "checkpoint metadata is top-level in producer evidence",
    },
    3: {
        "run_label": REVALIDATION_EFFICIENCY_J3_RUN_LABEL,
        "source_sha": REVALIDATION_EFFICIENCY_J1_SOURCE_SHA,
        "image_digest": REVALIDATION_EFFICIENCY_J1_IMAGE_DIGEST,
        "resource_name": REVALIDATION_EFFICIENCY_J3_RESOURCE_NAME,
        "reason": "J3 reuses J2 loader timing; it does not duplicate that timing block",
    },
}


def _sample(index: int, *, all_invalid: bool = False) -> RealSample:
    patch_id = f"synthetic-scene-{index}:anchor-{index}"
    mask = np.zeros((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    if not all_invalid:
        mask[:, 1:7, 2:9] = True
    target = np.full((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), np.nan, dtype=np.float32)
    target[mask] = 301.25
    features = np.zeros((28, REAL_PATCH_PX, REAL_PATCH_PX), dtype=np.float32)
    features[:10] = np.float32(index + 0.25)
    features[10:] = np.float32(99.0)
    return RealSample(
        meta=RealSampleMeta(
            patch_id=patch_id,
            scene_id=f"synthetic-scene-{index}",
            split="train" if index == 0 else "validation",
            year=2018 + index,
            row=index,
            col=index + 1,
            filled_feature_pixels=7,
        ),
        features=features,
        lst_prior_k=np.full((1, REAL_PATCH_PX, REAL_PATCH_PX), 300.5, dtype=np.float32),
        target_100m=target,
        mask_100m=mask,
    )


def _guard_checks() -> list[str]:
    failures: list[str] = []
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        valid = compose(config_name="stage1_efficiency")
        test = compose(
            config_name="stage1_efficiency",
            overrides=['data.splits=["train","validation","test"]'],
        )
        full = compose(config_name="stage1_locked")
        unbounded = compose(config_name="real_full")
        test_smoke = compose(config_name="real_smoke")
        learning_fit = compose(
            config_name="stage1_probe",
            overrides=[
                "+stage1_efficiency=true",
                "+stage1_efficiency_fit=true",
                "trainer.precision=16-mixed",
                "data.num_workers=0",
            ],
        )
    try:
        assert_stage1_efficiency(valid)
        try:
            guard_modeling_config(valid, "stage1_efficiency")
        except ValueError:
            print("  PASS guard: fit runner rejects measurement-only profile")
        else:
            failures.append("efficiency profile entered the fit runner")
            print("  FAIL guard: efficiency profile entered the fit runner")
    except Exception as exc:
        failures.append(f"valid efficiency profile rejected: {exc}")
        print(f"  FAIL guard: bounded efficiency profile ({exc})")
    try:
        assert_stage1_efficiency(test)
    except ValueError:
        print("  PASS guard: efficiency profile rejects test split")
    else:
        failures.append("efficiency profile accepted test split")
        print("  FAIL guard: efficiency profile accepted test split")
    try:
        guard_modeling_config(full, "stage1_locked")
    except ValueError:
        print("  PASS guard: full Stage-1 remains blocked")
    else:
        failures.append("full Stage-1 config was executable")
        print("  FAIL guard: full Stage-1 config was executable")
    try:
        guard_modeling_config(unbounded, "real_full")
    except ValueError:
        print("  PASS guard: generic unbounded real fit remains blocked")
    else:
        failures.append("generic real_full config was executable")
        print("  FAIL guard: generic real_full config was executable")
    try:
        guard_modeling_config(test_smoke, "real_smoke")
    except ValueError:
        print("  PASS guard: generic real smoke cannot admit test pixels")
    else:
        failures.append("generic real_smoke config admitted test pixels")
        print("  FAIL guard: generic real_smoke config admitted test pixels")
    try:
        with patch.dict(
            os.environ,
            {
                "VERTEX_PROFILE": "efficiency",
                "VERTEX_EFFICIENCY_SESSION": EFFICIENCY_SESSION_ID,
                "VERTEX_EFFICIENCY_ROLE": "baseline",
                "VERTEX_EFFICIENCY_SLOT": "1",
            },
        ):
            guard_modeling_config(learning_fit, "stage1_probe")
        print("  PASS guard: allowlisted efficiency learning fit admits measured operations")
    except ValueError as exc:
        failures.append(f"allowlisted efficiency fit rejected: {exc}")
        print(f"  FAIL guard: allowlisted efficiency fit rejected ({exc})")
    try:
        guard_modeling_config(learning_fit, "stage1_probe")
    except ValueError:
        print("  PASS guard: learning fit blocked outside a reserved worker context")
    else:
        failures.append("efficiency learning fit accepted outside its reserved worker")
        print("  FAIL guard: efficiency learning fit accepted outside its reserved worker")

    valid_scope = {
        "patches_per_split": {"train": 500, "validation": 190},
        "years_per_split": {"train": [2018, 2019, 2020]},
        "partial_mask_patches_per_split": {"train": 12, "validation": 4},
    }
    try:
        assert_stage1_efficiency_scope(valid, valid_scope)
        print("  PASS guard: realized performance cohort meets bounds")
    except ValueError as exc:
        failures.append(f"valid realized cohort rejected: {exc}")
        print(f"  FAIL guard: realized performance cohort ({exc})")
    no_partial = {**valid_scope, "partial_mask_patches_per_split": {"train": 0, "validation": 0}}
    try:
        assert_stage1_efficiency_scope(valid, no_partial)
    except ValueError:
        print("  PASS guard: cohort without partial masks rejected")
    else:
        failures.append("cohort without partial masks accepted")
        print("  FAIL guard: cohort without partial masks accepted")
    return failures


def _cache_checks(root: Path) -> list[str]:
    failures: list[str] = []
    provenance: dict[str, object] = {
        "source_fingerprint": "synthetic-v1",
        "index_policy_hash": "synthetic-index",
        "scaler_digest": "synthetic-scaler",
        "channel_order": [f"band-{i}" for i in range(10)],
        "geometry": [REAL_PATCH_PX, REAL_PATCH_CELLS],
    }
    source = [_sample(0), _sample(1)]
    ids = [sample.meta.patch_id for sample in source]
    cache_root = root / "valid"
    try:
        manifest = build_patch_cache(
            cache_root,
            iter(source),
            patch_ids=ids,
            active_channels=10,
            provenance=provenance,
            max_bytes=estimate_cache_bytes(len(ids), 10) + 1024,
        )
        cache = PatchCache(cache_root, provenance)
        if cache.patch_ids != tuple(ids):
            raise AssertionError("cache changed canonical patch order")
        if any(array.flags.writeable for array in cache.arrays.values()):
            raise AssertionError("cache mapping is writable")
        if manifest["array_bytes"] <= 0:
            raise AssertionError("cache did not report array bytes")
        for index, original in enumerate(source):
            assert_samples_equal(original, cache[index], 10)
        cached_dataset = CachedRealPatchDataset(cache_root, provenance, [0, 1])
        cached_samples = [cached_dataset[0], cached_dataset[1]]
        source_batch = collate_real_batch(source, n_active_channels=10)
        cache_batch = collate_real_batch(cached_samples, n_active_channels=10)
        if not torch.equal(source_batch.features, cache_batch.features):
            raise AssertionError("source/cache collation changed active feature tensors")
        if not torch.equal(source_batch.lst_prior, cache_batch.lst_prior):
            raise AssertionError("source/cache collation changed prior tensors")
        if not np.array_equal(
            source_batch.target_100m.numpy(), cache_batch.target_100m.numpy(), equal_nan=True
        ) or not torch.equal(source_batch.mask_100m, cache_batch.mask_100m):
            raise AssertionError("source/cache collation changed target or mask")
        source_order = list(RandomSampler(source, generator=torch.Generator().manual_seed(0)))
        cache_order = list(
            RandomSampler(cached_dataset, generator=torch.Generator().manual_seed(0))
        )
        if source_order != cache_order:
            raise AssertionError("source/cache seeded sampler order differs")
        moved = cache_batch.to("cpu", non_blocking=True)
        if not all(
            torch.equal(left, right)
            for left, right in (
                (cache_batch.features, moved.features),
                (cache_batch.lst_prior, moved.lst_prior),
                (cache_batch.mask_100m, moved.mask_100m),
            )
        ) or not np.array_equal(
            cache_batch.target_100m.numpy(), moved.target_100m.numpy(), equal_nan=True
        ):
            raise AssertionError("custom batch .to(device) changed tensor values")
        invalid_target = cache[0].target_100m[~cache[0].mask_100m]
        if cache[0].features.shape[0] != 10 or not np.isnan(invalid_target).all():
            raise AssertionError("active-channel or invalid-target preservation failed")
        if cache[0].features.max() >= 99.0:
            raise AssertionError("unused channels entered the active cache")
        print("  PASS cache: exact active tensors, metadata, mask, order, and read-only mapping")
    except Exception as exc:
        failures.append(f"valid cache failed: {exc}")
        print(f"  FAIL cache: valid source round-trip ({exc})")

    try:
        PatchCache(cache_root, {**provenance, "source_fingerprint": "stale"})
    except RuntimeError:
        print("  PASS cache: stale provenance rejected")
    else:
        failures.append("stale cache provenance accepted")
        print("  FAIL cache: stale provenance accepted")

    try:
        build_patch_cache(
            root / "over-budget",
            iter(source),
            patch_ids=ids,
            active_channels=10,
            provenance=provenance,
            max_bytes=1,
        )
    except ValueError:
        print("  PASS cache: over-budget build rejected before creating root")
    else:
        failures.append("over-budget cache accepted")
        print("  FAIL cache: over-budget cache accepted")

    try:
        build_patch_cache(
            root / "all-invalid",
            iter([_sample(0, all_invalid=True)]),
            patch_ids=[_sample(0, all_invalid=True).meta.patch_id],
            active_channels=10,
            provenance=provenance,
            max_bytes=estimate_cache_bytes(1, 10) + 1024,
        )
    except RuntimeError:
        if (root / "all-invalid" / "manifest.json").exists():
            failures.append("failed all-invalid build published readiness")
        print("  PASS cache: all-invalid sample rejected without readiness manifest")
    else:
        failures.append("all-invalid sample accepted")
        print("  FAIL cache: all-invalid sample accepted")

    malformed = {
        "prior": replace(
            source[0],
            lst_prior_k=np.full((1, 1, REAL_PATCH_PX), 300.5, dtype=np.float32),
        ),
        "target": replace(
            source[0],
            target_100m=np.full((1, 1, REAL_PATCH_CELLS), 301.25, dtype=np.float32),
        ),
        "mask": replace(
            source[0],
            mask_100m=np.ones((1, 1, REAL_PATCH_CELLS), dtype=bool),
        ),
    }
    for name, sample in malformed.items():
        bad_root = root / f"bad-shape-{name}"
        try:
            build_patch_cache(
                bad_root,
                iter([sample]),
                patch_ids=[sample.meta.patch_id],
                active_channels=10,
                provenance=provenance,
                max_bytes=estimate_cache_bytes(1, 10) + 1024,
            )
        except RuntimeError:
            if (bad_root / "manifest.json").exists():
                failures.append(f"malformed {name} published readiness")
            print(f"  PASS cache: broadcastable malformed {name} shape rejected")
        else:
            failures.append(f"malformed {name} shape was accepted")
            print(f"  FAIL cache: malformed {name} shape was accepted")
    return failures


def _selection_checks() -> list[str]:
    failures: list[str] = []
    source = RealSourceConfig("index", "training", "features", "ard")
    scope = ProbeScope(
        max_refs_per_split={"train": 4, "validation": 2},
        max_refs_per_scene=2,
        require_partial_masks=True,
    )
    refs: list[PatchRef] = []
    for split, years, count in (
        ("train", (2018, 2019, 2020), 4),
        ("validation", (2024,), 2),
    ):
        for i in range(count):
            year = years[i % len(years)]
            scene_id = f"synthetic-{split}-{year}-{i}"
            refs.append(
                PatchRef(
                    patch_id=f"{scene_id}:anchor-{i}",
                    scene_id=scene_id,
                    s2_scene_id=f"s2-{i}",
                    split=split,
                    year=year,
                    row=i,
                    col=i,
                    n_eligible=100 if i == 0 else REAL_PATCH_CELLS * REAL_PATCH_CELLS,
                    eligibility_mask=f"mask-{i}",
                )
            )
    module = RealPatchDataModule(
        source,
        mode="stream",
        splits=("train", "validation"),
        probe_scope=scope,
    )
    selected = module._select_refs(refs)
    repeated = module._select_refs(refs)
    if selected != repeated:
        failures.append("scene-spread partial-mask selection was not deterministic")
    for split, split_refs in selected.items():
        if not any(0 < ref.n_eligible < REAL_PATCH_CELLS * REAL_PATCH_CELLS for ref in split_refs):
            failures.append(f"{split} selection did not include a partial mask")
    no_partial = [replace(ref, n_eligible=REAL_PATCH_CELLS * REAL_PATCH_CELLS) for ref in refs]
    try:
        module._select_refs(no_partial)
    except ValueError:
        pass
    else:
        failures.append("index cohort without a partial-mask candidate was accepted")
    if failures:
        print(f"  FAIL selection: {failures}")
    else:
        print("  PASS selection: deterministic scene spread includes partial masks")

    fallback_module = RealPatchDataModule(
        source,
        mode="stream",
        splits=("train", "validation"),
        cache_root=Path("unbuilt-cache"),
        cache_max_bytes=1024,
    )
    fallback_module._datasets["train"] = RealPatchDataset(source, refs[:1])
    try:
        fallback_module.train_dataloader()
    except RuntimeError:
        print("  PASS guard: failed cache cannot fall back to streaming")
    else:
        failures.append("cache-enabled module silently loaded the source dataset")
        print("  FAIL guard: cache-enabled module silently loaded the source dataset")
    return failures


def _evidence_checks() -> list[str]:
    """Prove the evidence gate accepts a cache slot and rejects universe drift."""
    failures: list[str] = []
    train_ids = [f"train-{i}" for i in range(384)]
    val_ids = [f"val-{i}" for i in range(128)]
    scope = {
        "requested_patch_ids": {"train": train_ids, "validation": val_ids},
        "patch_ids": {"train": train_ids, "validation": val_ids},
        "requested_per_split": {"train": 384, "validation": 128},
        "patches_per_split": {"train": 384, "validation": 128},
        "skipped_refs": {"train": [], "validation": []},
        "partial_mask_patches_per_split": {"train": 1, "validation": 1},
        "years_per_split": {"train": [2018, 2019, 2020], "validation": [2024]},
        "train_shuffle_order": list(range(384)),
    }
    ids = [*train_ids, *val_ids]
    provenance = {
        "admitted_patch_ids": scope["patch_ids"],
        "skipped_refs": scope["skipped_refs"],
        "channel_order": [f"channel-{i}" for i in range(10)],
    }
    manifest = {
        "patch_ids": ids,
        "patch_ids_sha256": hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "provenance": provenance,
        "array_bytes": 1000,
    }
    timing = {}
    for workers, seconds in ((0, 5.0), (2, 4.0), (4, 6.0)):
        timing[f"cache_workers_{workers}_pin_false"] = {
            split: {
                "first_pass": {"wall_seconds": seconds * 1.2},
                "steady_repeats": [{"wall_seconds": seconds / 3}] * 3,
            }
            for split in ("train", "validation")
        }
    timing["cache_workers_2_pin_true"] = {
        split: {
            "first_pass": {"wall_seconds": 1.8},
            "steady_repeats": [{"wall_seconds": 5.0 / 3}] * 3,
        }
        for split in ("train", "validation")
    }
    for pinned, seconds in ((False, 4.0), (True, 5.0)):
        timing[f"cache_pin_pair_workers_2_pin_{str(pinned).lower()}"] = {
            split: {
                "first_pass": {"wall_seconds": seconds * 1.2},
                "steady_repeats": [{"wall_seconds": seconds / 3}] * 3,
            }
            for split in ("train", "validation")
        }
    source_timing = {
        split: {
            "first_pass": {"wall_seconds": 2.0},
            "steady_repeats": [{"wall_seconds": 1.0}] * 3,
        }
        for split in ("train", "validation")
    }
    timing["source_workers_2"] = source_timing
    source_sha = "a" * 40
    image_digest = "sha256:" + "b" * 64
    control = {
        "profile": "efficiency",
        "source_sha": source_sha,
        "image_digest": image_digest,
        "run_label": "stage1-efficiency-j1-control",
        "data_scope": scope,
        "efficiency": {
            "slot": 1,
            "role": "baseline",
            "results": {"run_label": "stage1-efficiency-j1-control"},
        },
    }
    evidence = {
        "profile": "efficiency",
        "source_sha": source_sha,
        "image_digest": image_digest,
        "run_label": "stage1-efficiency-j2-cache",
        "data_scope": scope,
        "efficiency": {
            "slot": 2,
            "role": "cache",
            "complete": True,
            "test_access": False,
            "cache_manifest": manifest,
            "results": {
                "status": "complete",
                "slot": 2,
                "role": "cache",
                "session": EFFICIENCY_SESSION_ID,
                "run_label": "stage1-efficiency-j2-cache",
                "source_sha": source_sha,
                "image_digest": image_digest,
                "scratch_optimizer_steps": 0,
                "test_access": False,
                "budget": {
                    "hourly_rate_usd": 0.9,
                    "hourly_rate_source": "synthetic local self-check source label",
                    "projected_exposure_usd": 0.825,
                    "projected_noncompute_total_usd": 2.0,
                },
                "config": {
                    "stage1_efficiency": True,
                    "stage1_efficiency_fit": False,
                    "stage1_full": False,
                    "stage1_lock": False,
                    "stage1_efficiency_role": "cache",
                    "stage1_residual_prior": True,
                    "seed": 0,
                    "data": {
                        "splits": ["train", "validation"],
                        "max_patches_per_split": 512,
                        "n_active_channels": 10,
                        "mode": "stream",
                        "batch_size": 4,
                        "num_workers": 2,
                        "pin_memory": False,
                        "probe": {
                            "max_refs_per_split": {"train": 384, "validation": 128},
                            "require_partial_masks": True,
                        },
                    },
                    "trainer": {"max_epochs": 1, "precision": "32-true"},
                },
                "environment": {
                    "cuda": {"available": True, "name": "Tesla T4"},
                    "disk": {"free_bytes": 10_000_000},
                },
                "cache": {
                    "scope_equal": True,
                    "train_sampler_order_equal": True,
                    "source_cache_exact_samples": {"train": 384, "validation": 128},
                },
                "selected_loader": {"workers": 2, "pin_memory": False},
                "timing": timing,
                "cpu_gpu_validation": {
                    "absolute_mae_difference": 0.0,
                    "cpu_wall_seconds": 1.0,
                    "gpu_wall_seconds": 0.5,
                    "valid_cells": 100,
                    "cpu_identity_mae_k": 0.0,
                    "gpu_identity_mae_k": 0.0,
                    "cpu_correction_mean_abs_k": 0.0,
                    "gpu_correction_mean_abs_k": 0.0,
                },
            },
        },
    }
    errors = validate_efficiency_evidence(evidence, {}, control=control)
    if errors:
        failures.extend(f"valid J2 evidence rejected: {error}" for error in errors)
        print(f"  FAIL evidence: cache slot ({errors})")
    else:
        print("  PASS evidence: cache slot reconciles source, manifests, and timings")
    changed = json.loads(json.dumps(evidence))
    changed["data_scope"]["patch_ids"]["validation"].pop()
    errors = validate_efficiency_evidence(changed, {}, control=control)
    if not any("fixed performance universe" in error for error in errors):
        failures.append("cache evidence accepted changed performance universe")
        print("  FAIL evidence: changed performance universe accepted")
    else:
        print("  PASS evidence: changed performance universe rejected")
    j2 = json.loads(json.dumps(evidence))
    j2["efficiency"]["results"]["selected_loader"] = {
        "workers": 0,
        "pin_memory": False,
    }
    diagnostic_results = {"config": {"data": {"num_workers": 0, "pin_memory": False}}}
    errors = []
    _validate_selected_cache_loader(diagnostic_results, j2, errors, slot=3)
    if errors:
        failures.extend(f"J3 accepted J2 loader but had errors: {error}" for error in errors)
        print(f"  FAIL evidence: J3 J2-loader reuse ({errors})")
    else:
        print("  PASS evidence: J3 reuses J2 loader without a duplicate timing block")
    diagnostic_results["config"]["data"]["num_workers"] = 2
    errors = []
    _validate_selected_cache_loader(diagnostic_results, j2, errors, slot=3)
    if not any("worker count differs" in error for error in errors):
        failures.append("J3 accepted a loader configuration different from J2")
        print("  FAIL evidence: J3 loader mismatch accepted")
    else:
        print("  PASS evidence: J3 loader mismatch rejected")
    budget = evidence["efficiency"]["results"]["budget"]
    ledger_row = {
        "session": EFFICIENCY_SESSION_ID,
        "slot": 2,
        "state": "submitted",
        "run_label": evidence["run_label"],
        "role": "cache",
        "source_sha": evidence["source_sha"],
        "image_digest": evidence["image_digest"],
        "hourly_rate_source": budget["hourly_rate_source"],
        "verified_hourly_rate_usd": budget["hourly_rate_usd"],
        "projected_exposure_usd": budget["projected_exposure_usd"],
        "projected_noncompute_total_usd": budget["projected_noncompute_total_usd"],
    }
    with tempfile.TemporaryDirectory(
        dir=Path(os.environ["TMPDIR"]) / "opencode",
        prefix="efficiency-ledger-self-check-",
    ) as temporary:
        ledger_path = Path(temporary) / "slots.json"
        prior_row = {
            "slot": 1,
            "projected_exposure_usd": 1.25,
            "projected_noncompute_total_usd": 2.0,
        }
        ledger_path.write_text(json.dumps([prior_row, ledger_row]), encoding="utf-8")
        with patch(__name__ + "._EFFICIENCY_LEDGER", ledger_path):
            errors = _reservation_errors(evidence, 2)
            if errors:
                failures.append(f"ledger rejected matching provenance and numeric budget: {errors}")
                print(f"  FAIL budget provenance: {errors}")
            else:
                print("  PASS budget provenance: text is compared exactly and numeric values parse")
            ledger_row["hourly_rate_source"] = "different source note"
            ledger_path.write_text(json.dumps([prior_row, ledger_row]), encoding="utf-8")
            errors = _reservation_errors(evidence, 2)
            if not any("hourly_rate_source differs" in error for error in errors):
                failures.append("ledger accepted different hourly-rate provenance")
                print("  FAIL budget provenance: source-note drift accepted")
            else:
                print("  PASS budget provenance: source-note drift rejected")
    started = datetime(2026, 1, 1, tzinfo=UTC)
    if _provisioned_seconds(started, started + timedelta(seconds=30)) != 30:
        failures.append("Vertex datetime elapsed-time calculation is incorrect")
        print("  FAIL budget timestamp: SDK datetime elapsed time")
    else:
        print("  PASS budget timestamp: SDK datetime elapsed time")
    return failures


def _checkpoint_checks() -> list[str]:
    failures: list[str] = []
    temporary_parent = Path(os.environ["TMPDIR"]) / "opencode"
    if not temporary_parent.is_dir():
        return [f"checkpoint self-check temporary parent is missing: {temporary_parent}"]
    with tempfile.TemporaryDirectory(
        dir=temporary_parent, prefix="efficiency-checkpoint-self-check-"
    ) as temporary:
        checkpoint = Path(temporary) / "best.ckpt"
        payload = b"synthetic selected checkpoint"
        checkpoint.write_bytes(payload)
        reference = {
            "uri": "gs://synthetic/efficiency/best.ckpt",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        evidence = {
            "profile": "efficiency",
            "source_sha": "a" * 40,
            "image_digest": "sha256:" + "b" * 64,
            "run_label": REVALIDATION_EFFICIENCY_J1_RUN_LABEL,
            "checkpoint": reference,
            "efficiency": {
                "slot": 1,
                "role": "baseline",
                "learning": {"complete": True},
            },
        }
        mismatch = _verify_checkpoint_hash(evidence, checkpoint)
        if mismatch:
            failures.append(f"top-level selected checkpoint was rejected: {mismatch}")
            print(f"  FAIL checkpoint: {mismatch}")
        else:
            print("  PASS checkpoint: producer's top-level selected-checkpoint accepted")
        corrupted = Path(temporary) / "corrupted.ckpt"
        corrupted.write_bytes(payload + b"x")
        if _verify_checkpoint_hash(evidence, corrupted) is None:
            failures.append("selected-checkpoint hash mismatch was accepted")
            print("  FAIL checkpoint: mismatched file hash accepted")
        else:
            print("  PASS checkpoint: mismatched file hash rejected")
        isolated = json.loads(json.dumps(evidence))
        del isolated["checkpoint"]
        isolated_errors: list[str] = []
        _validate_learning(isolated["efficiency"], isolated, {}, isolated_errors)
        if not any(
            "learning fit has no retained selected-checkpoint hash/size" in error
            for error in isolated_errors
        ):
            failures.append("learning validator accepted evidence without its checkpoint")
            print("  FAIL checkpoint: missing-checkpoint evidence accepted")
        else:
            print("  PASS checkpoint: missing-checkpoint evidence rejected")
    return failures


def _revalidation_checks() -> list[str]:
    from types import SimpleNamespace

    from google.cloud.aiplatform_v1.types import JobState

    failures: list[str] = []
    temporary_parent = Path(os.environ["TMPDIR"]) / "opencode"
    if not temporary_parent.is_dir():
        return [f"revalidation self-check temporary parent is missing: {temporary_parent}"]
    spec = _REVALIDATION_SPECS[3]
    evidence = {
        "profile": "efficiency",
        "run_label": spec["run_label"],
        "source_sha": spec["source_sha"],
        "image_digest": spec["image_digest"],
        "efficiency": {
            "slot": 3,
            "role": "diagnostic",
            "validated_settings": {"amp_projected_runtime_gain": 0.12},
        },
    }
    with tempfile.TemporaryDirectory(
        dir=temporary_parent, prefix="efficiency-revalidation-self-check-"
    ) as temporary:
        root = Path(temporary)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        row = {
            "session": EFFICIENCY_SESSION_ID,
            "slot": 3,
            "role": "diagnostic",
            "run_label": spec["run_label"],
            "source_sha": spec["source_sha"],
            "image_digest": spec["image_digest"],
            "resource_name": spec["resource_name"],
            "state": "validation_failed",
            "validation_verdict": "fail",
            "evidence_sha256": _sha256(evidence_path),
            "terminal_state": "JOB_STATE_SUCCEEDED",
        }
        ledger = root / "slots.json"
        previous = [
            {"slot": 1, "state": "validated"},
            {"slot": 2, "state": "validated"},
        ]
        ledger.write_text(json.dumps([*previous, row]), encoding="utf-8")
        with patch(__name__ + "._EFFICIENCY_LEDGER", ledger):
            with patch(
                "google.cloud.aiplatform_v1.JobServiceClient.get_custom_job",
                return_value=SimpleNamespace(state=JobState.JOB_STATE_SUCCEEDED),
            ):
                _record_revalidation_audit(3, evidence, evidence_path, passed=True)
            updated = json.loads(ledger.read_text(encoding="utf-8"))[2]
            attempts = updated.get("initial_validation_attempts") or []
            if (
                updated.get("state") != "validated"
                or len(attempts) != 2
                or attempts[0].get("verdict") != "fail"
                or attempts[1].get("verdict") != "pass"
                or attempts[0].get("evidence_sha256") != attempts[1].get("evidence_sha256")
                or updated.get("amp_projected_runtime_gain") != 0.12
            ):
                failures.append("J3 revalidation did not preserve its initial failed verdict")
                print("  FAIL revalidation: initial failure was not retained")
            else:
                print("  PASS revalidation: same J3 artifact records fail then pass")
            if not _failed_revalidation_errors(3, updated, evidence, evidence_path):
                failures.append("a validated J3 slot remained eligible for another revalidation")
                print("  FAIL revalidation: repeated revalidation was not rejected")
            else:
                print("  PASS revalidation: duplicate revalidation rejected")
            mismatched = {**evidence, "run_label": "stage1-efficiency-j3-other-artifact"}
            if not _failed_revalidation_errors(3, row, mismatched, evidence_path):
                failures.append("J3 revalidation accepted a different run label")
                print("  FAIL revalidation: different evidence identity accepted")
            else:
                print("  PASS revalidation: different evidence identity rejected")
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _provisioned_seconds(start_time: datetime, end_time: datetime) -> float:
    return (end_time - start_time).total_seconds()


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)


def _validate_scope(scope: dict, errors: list[str]) -> None:
    ids = scope.get("patch_ids")
    requested_ids = scope.get("requested_patch_ids")
    counts = scope.get("patches_per_split")
    requested = scope.get("requested_per_split")
    skipped = scope.get("skipped_refs")
    partial = scope.get("partial_mask_patches_per_split")
    years = scope.get("years_per_split")
    if not all(
        isinstance(v, dict)
        for v in (ids, requested_ids, counts, requested, skipped, partial, years)
    ):
        _error(
            "performance data_scope is missing IDs/counts/skips/partial-mask/year accounting",
            errors,
        )
        return
    if sorted(ids) != ["train", "validation"]:
        _error(f"performance scope splits are not train/validation: {sorted(ids)}", errors)
    expected_requested = {"train": 384, "validation": 128}
    for split, minimum in expected_requested.items():
        admitted_ids = [str(value) for value in ids.get(split, [])]
        frozen_ids = [str(value) for value in requested_ids.get(split, [])]
        skipped_rows = skipped.get(split, [])
        if len(set(frozen_ids)) != len(frozen_ids) or len(set(admitted_ids)) != len(admitted_ids):
            _error(f"{split}: duplicate requested/admitted patch IDs", errors)
        if int(requested.get(split, -1)) != len(frozen_ids):
            _error(f"{split}: requested count does not match frozen IDs", errors)
        if len(frozen_ids) != minimum:
            _error(f"{split}: frozen request count differs from {minimum}", errors)
        if int(counts.get(split, -1)) != len(admitted_ids):
            _error(f"{split}: admitted count does not match admitted IDs", errors)
        if set(admitted_ids) & {str(row.get("patch_id")) for row in skipped_rows}:
            _error(f"{split}: patch ID is both admitted and skipped", errors)
        if len(frozen_ids) != len(admitted_ids) + len(skipped_rows):
            _error(f"{split}: frozen IDs do not reconcile to admitted plus skipped", errors)
        if len(admitted_ids) < minimum:
            _error(f"{split}: admitted {len(admitted_ids)} < required {minimum}", errors)
        if int(partial.get(split, 0)) <= 0:
            _error(f"{split}: no admitted partial-mask patches", errors)
    if len(years.get("train", [])) < 3:
        _error("train performance cohort covers fewer than three years", errors)
    if any("test" in str(patch_id) for patch_id in ids.get("test", [])):
        _error("test split appears in performance scope", errors)


def _manifest_valid(manifest: dict, scope: dict, errors: list[str], *, name: str) -> None:
    ids_by_split = scope.get("patch_ids") or {}
    expected_ids = [*ids_by_split.get("train", []), *ids_by_split.get("validation", [])]
    actual_ids = manifest.get("patch_ids")
    if not isinstance(actual_ids, list) or actual_ids != expected_ids:
        _error(f"{name}: cache manifest patch IDs/order differ from the fixed scope", errors)
        return
    digest = hashlib.sha256(
        json.dumps(actual_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("patch_ids_sha256") != digest:
        _error(f"{name}: cache manifest patch-ID digest mismatch", errors)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        _error(f"{name}: cache provenance is missing", errors)
        return
    if provenance.get("admitted_patch_ids") != ids_by_split:
        _error(f"{name}: cache provenance admitted IDs differ from measured scope", errors)
    if provenance.get("skipped_refs") != scope.get("skipped_refs"):
        _error(f"{name}: cache provenance exclusions differ from measured scope", errors)
    channel_order = provenance.get("channel_order")
    if not isinstance(channel_order, list) or len(channel_order) != 10:
        _error(f"{name}: cache is not the locked first-C=10 view", errors)
    array_bytes = manifest.get("array_bytes")
    if not _finite_number(array_bytes) or not 0 < float(array_bytes) <= 40 * 1024**3:
        _error(f"{name}: cache array byte count is missing or exceeds the 40-GiB guard", errors)


def _steady_repeats(block: object, split: str, errors: list[str], label: str) -> list[dict]:
    if not isinstance(block, dict):
        _error(f"{label}: missing timing block", errors)
        return []
    split_block = block.get(split)
    if not isinstance(split_block, dict):
        _error(f"{label}: missing {split} timing", errors)
        return []
    first = split_block.get("first_pass")
    repeats = split_block.get("steady_repeats")
    if not isinstance(first, dict) or not isinstance(repeats, list) or len(repeats) != 3:
        _error(f"{label}/{split}: expected a first pass plus three steady repeats", errors)
        return []
    for row in [first, *repeats]:
        if not _finite_number(row.get("wall_seconds")) or float(row["wall_seconds"]) <= 0:
            _error(f"{label}/{split}: invalid wall duration", errors)
    return repeats


def _projected_amp_gain(diagnostic_results: dict, cache_results: dict) -> float | None:
    candidates = (diagnostic_results.get("compute") or {}).get("candidates", {})
    fp32 = candidates.get("fp32_batch4", [])
    amp = candidates.get("mixed_batch4", [])
    if len(fp32) != 3 or len(amp) != 3:
        return None
    selected = cache_results.get("selected_loader", {})
    loader = cache_results.get("timing", {}).get(
        f"cache_workers_{selected.get('workers')}_pin_{str(selected.get('pin_memory')).lower()}",
        {},
    )
    loader_seconds = (
        sum(
            float(repeat["wall_seconds"])
            for split in ("train", "validation")
            for repeat in loader.get(split, {}).get("steady_repeats", [])
        )
        / 3
    )
    fp32_step = float(np.median([row["wall_seconds_repeats"][0] for row in fp32]))
    amp_step = float(np.median([row["wall_seconds_repeats"][0] for row in amp]))
    total_batches = math.ceil(384 / 4) + math.ceil(128 / 4)
    return (
        max(0.0, fp32_step - amp_step)
        * total_batches
        / max(1e-12, loader_seconds + fp32_step * total_batches)
    )


def _validate_learning(
    efficiency: dict,
    evidence: dict,
    baseline: dict,
    errors: list[str],
) -> tuple[list[float], list[float], float, float] | None:
    learning = efficiency.get("learning")
    if not isinstance(learning, dict) or learning.get("complete") is not True:
        _error("six-epoch learning evidence is missing or incomplete", errors)
        return None
    checkpoint = _checkpoint_reference(evidence)
    if (
        not isinstance(checkpoint, dict)
        or not str(checkpoint.get("uri", "")).endswith(".ckpt")
        or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256", "")))
        or not isinstance(checkpoint.get("bytes"), int)
        or checkpoint.get("bytes", 0) <= 0
    ):
        _error("learning fit has no retained selected-checkpoint hash/size", errors)
        return None
    scope = learning.get("data_scope")
    epochs = learning.get("epoch_metrics")
    summary = learning.get("summary")
    if not isinstance(scope, dict) or not isinstance(epochs, list) or not isinstance(summary, dict):
        _error("learning evidence is missing its scope, curve, or summary", errors)
        return None
    ids = scope.get("patch_ids")
    if not isinstance(ids, dict) or sorted(ids) != ["train", "validation"]:
        _error("learning scope is not exactly train/validation", errors)
        return None
    requested_ids = scope.get("requested_patch_ids")
    requested_counts = scope.get("requested_per_split")
    admitted_counts = scope.get("patches_per_split")
    skipped_refs = scope.get("skipped_refs")
    scenes = scope.get("scenes_per_split")
    years = scope.get("years_per_split")
    if not all(
        isinstance(value, dict)
        for value in (requested_ids, requested_counts, admitted_counts, skipped_refs, scenes, years)
    ):
        _error("learning scope lacks requested/admitted/skipped cohort accounting", errors)
        return None
    requested_sizes = {"train": 128, "validation": 64}
    minimum_admitted = {"train": 120, "validation": 60}
    minimum_scenes = {"train": 16, "validation": 8}
    for split in ("train", "validation"):
        requested = [str(value) for value in requested_ids.get(split, [])]
        admitted = [str(value) for value in ids.get(split, [])]
        skipped = skipped_refs.get(split, [])
        skipped_ids = {str(row.get("patch_id")) for row in skipped}
        if (
            int(requested_counts.get(split, -1)) != requested_sizes[split]
            or len(requested) != requested_sizes[split]
        ):
            _error(f"learning {split} requested cohort differs from the preregistered size", errors)
        if int(admitted_counts.get(split, -1)) != len(admitted):
            _error(f"learning {split} admitted count differs from its IDs", errors)
        if len(admitted) < minimum_admitted[split]:
            _error(f"learning {split} admitted count is below the recovery minimum", errors)
        if len(requested) != len(set(requested)) or len(admitted) != len(set(admitted)):
            _error(f"learning {split} contains duplicate patch IDs", errors)
        if set(admitted) & skipped_ids or set(requested) != set(admitted) | skipped_ids:
            _error(f"learning {split} requested/admitted/skipped IDs do not reconcile", errors)
        if len(scenes.get(split, [])) < minimum_scenes[split]:
            _error(f"learning {split} cohort has too few scenes", errors)
    if len(years.get("train", [])) < 3:
        _error("learning train cohort covers fewer than three years", errors)
    if sorted(evidence.get("data_scope", {}).get("patch_ids", {})) != ["train", "validation"]:
        _error("performance scope admits a non-training split", errors)

    resolved = summary.get("resolved_config")
    if not isinstance(resolved, dict):
        _error("learning summary has no resolved config", errors)
        return None
    locked = (
        ("seed", 0),
        ("data.n_active_channels", 10),
        ("data.max_patches_per_split", None),
        ("data.shuffle_train", True),
        ("model.depth", 4),
        ("model.base_width", 32),
        ("trainer.learning_rate", 1.0e-3),
        ("trainer.weight_decay", 0.0),
        ("trainer.max_epochs", 6),
    )
    for dotted, expected_value in locked:
        current: object = resolved
        for part in dotted.split("."):
            current = current.get(part) if isinstance(current, dict) else None
        if current != expected_value:
            _error(
                f"learning method drift: {dotted}={current!r} (expected {expected_value!r})",
                errors,
            )
    source = RealSourceConfig(
        patch_index_root=str(resolved["patch_index_root"]),
        training_root=str(resolved["training_root"]),
        features_root=str(resolved["features_root"]),
        ard_root=str(resolved["ard_root"]),
    )
    published = {
        ref.patch_id: ref.split for ref in load_patch_refs(source, splits=("train", "validation"))
    }
    for split in ("train", "validation"):
        wrong = [
            patch_id for patch_id in ids.get(split, []) if published.get(str(patch_id)) != split
        ]
        if wrong:
            _error(f"learning {split} includes IDs not in that published split", errors)

    baseline_index = {
        str(row["patch_id"]): row
        for row in baseline.get("patches", [])
        if str(row.get("split")) == "validation"
    }
    val_sum = 0.0
    val_cells = 0.0
    for patch_id in ids.get("validation", []):
        row = baseline_index.get(str(patch_id))
        if row is None:
            _error("learning validation cohort is not matched to the naive baseline", errors)
            return None
        val_sum += float(row["abs_error_sum"])
        val_cells += float(row["valid_cells"])
    if val_cells <= 0:
        _error("learning validation cohort has no matched naive support", errors)
        return None
    naive = val_sum / val_cells

    train_values = [entry.get("train_mae_100m") for entry in epochs]
    val_values = [entry.get("validation_mae_100m") for entry in epochs]
    if (
        len(train_values) != 6
        or len(val_values) != 6
        or not all(_finite_number(value) for value in [*train_values, *val_values])
    ):
        _error("learning curve must have six finite train/validation epochs", errors)
        return None
    train = [float(value) for value in train_values]
    val = [float(value) for value in val_values]
    support = [entry.get("validation_valid_cells") for entry in epochs]
    if not all(_finite_number(value) and float(value) > 0 for value in support):
        _error("learning epoch curve is missing validation valid-cell support", errors)
    elif len({float(value) for value in support}) != 1:
        _error("learning validation valid-cell support changed across epochs", errors)
    best = min(val)
    best_epoch = val.index(best) + 1
    if best_epoch != summary.get("best_epoch"):
        _error("learning selected epoch differs from the validation argmin", errors)
    if (
        not _finite_number(summary.get("best_metric"))
        or abs(float(summary["best_metric"]) - best) > 1e-3
    ):
        _error("learning best metric differs from validation curve argmin", errors)
    if (
        not _finite_number(summary.get("reload_recomputed"))
        or abs(float(summary["reload_recomputed"]) - best) > 1e-3
    ):
        _error("learning reload score does not reproduce best validation MAE", errors)
    if summary.get("residual_prior") is not True:
        _error("learning run did not use the locked residual prior", errors)
    if train[-1] > 0.95 * train[0]:
        _error("learning train MAE improved by less than 5%", errors)
    if best > min(1.5 * naive, 8.0):
        _error("learning best validation MAE failed the recovery naive cap", errors)
    if val[-1] > 1.25 * best:
        _error("learning final validation MAE diverged beyond the recovery cap", errors)
    identity = summary.get("residual_identity_mae_k")
    correction = summary.get("residual_correction_mean_abs_k")
    if not _finite_number(identity) or float(identity) > 0.001:
        _error("learning residual identity exceeded 0.001 K", errors)
    if not _finite_number(correction) or float(correction) < 0.05 * naive:
        _error("learning selected residual correction is a prior passthrough", errors)
    return train, val, naive, best


def validate_efficiency_evidence(
    evidence: dict,
    baseline: dict,
    *,
    control: dict | None = None,
    cache_evidence: dict | None = None,
    diagnostic_evidence: dict | None = None,
) -> list[str]:
    """Independently check one bounded slot and its role-specific prerequisites."""
    errors: list[str] = []
    if evidence.get("profile") != "efficiency":
        return ["evidence profile is not efficiency"]
    item = evidence.get("efficiency")
    if not isinstance(item, dict) or item.get("complete") is not True:
        return ["efficiency evidence is incomplete"]
    slot = item.get("slot")
    role = item.get("role")
    if not isinstance(slot, int) or role != _SLOT_ROLES.get(slot):
        return ["efficiency slot/role binding is invalid"]
    if not _SHA40.fullmatch(str(evidence.get("source_sha", ""))):
        errors.append("source SHA is missing or malformed")
    if not _SHA256.fullmatch(str(evidence.get("image_digest", ""))):
        errors.append("image digest is missing or malformed")
    if item.get("test_access") is not False:
        errors.append("efficiency evidence does not prove test access was disabled")
    results = item.get("results")
    if not isinstance(results, dict) or results.get("status") != "complete":
        errors.append("bounded measurement results are incomplete")
        return errors
    if results.get("slot") != slot or results.get("role") != role:
        errors.append("result slot/role does not match evidence envelope")
    if results.get("scratch_optimizer_steps") != 0 or results.get("test_access") is not False:
        errors.append("scratch measurements stepped an optimizer or accessed test data")
    if results.get("session") != EFFICIENCY_SESSION_ID:
        errors.append("efficiency result session ID is not the approved four-slot session")
    if results.get("source_sha") != evidence.get("source_sha") or results.get(
        "image_digest"
    ) != evidence.get("image_digest"):
        errors.append("result source/image identity differs from evidence envelope")
    if evidence.get("run_label") != results.get("run_label"):
        errors.append("efficiency result run label differs from the evidence envelope")
    _validate_efficiency_config(results, item, errors)
    budget = results.get("budget")
    if not isinstance(budget, dict):
        errors.append("efficiency evidence has no budget calculation")
    else:
        rate = budget.get("hourly_rate_usd")
        rate_source = budget.get("hourly_rate_source")
        exposure = budget.get("projected_exposure_usd")
        other = budget.get("projected_noncompute_total_usd")
        if not _finite_number(rate) or float(rate) > EFFICIENCY_MAX_HOURLY_RATE_USD:
            errors.append("efficiency rate is missing or exceeds $1.00/hour")
        if not _finite_number(exposure) or float(exposure) > EFFICIENCY_MAX_JOB_EXPOSURE_USD:
            errors.append("efficiency projected job compute exceeds $1.25")
        if not _finite_number(other) or float(other) > EFFICIENCY_MAX_NONCOMPUTE_USD:
            errors.append("cumulative non-compute estimate is missing or exceeds $2.00")
        if not isinstance(rate_source, str) or len(rate_source.strip()) < 20:
            errors.append("hourly rate source/provenance note is missing")
        if (
            _finite_number(exposure)
            and _finite_number(other)
            and EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD
            + float(exposure)
            + float(other)
            > EFFICIENCY_MAX_TOTAL_USD
        ):
            errors.append("historical and slot compute plus non-compute estimate exceeds $10")

    environment = results.get("environment")
    if not isinstance(environment, dict):
        errors.append("efficiency evidence has no execution environment")
    else:
        cuda = environment.get("cuda") or {}
        disk = environment.get("disk") or {}
        if cuda.get("available") is not True or "T4" not in str(cuda.get("name", "")).upper():
            errors.append("efficiency job did not record the pinned Vertex T4 device")
        if not _finite_number(disk.get("free_bytes")) or float(disk["free_bytes"]) <= 0:
            errors.append("efficiency job did not retain positive local disk-capacity evidence")

    scope = evidence.get("data_scope")
    if not isinstance(scope, dict):
        errors.append("efficiency evidence has no performance data_scope")
        return errors
    _validate_scope(scope, errors)
    timing = results.get("timing")
    if not isinstance(timing, dict):
        errors.append("efficiency evidence has no timing object")
        return errors

    def require_timing(key: str) -> None:
        block = timing.get(key)
        for split in ("train", "validation"):
            _steady_repeats(block, split, errors, key)

    if slot == 1:
        require_timing("source_workers_2")
        learning = _validate_learning(item, evidence, baseline, errors)
        if learning is None:
            return errors
        results_cfg = results.get("config") or {}
        if results_cfg.get("trainer", {}).get("precision") != "32-true":
            errors.append("J1 control must be FP32")
        if results_cfg.get("data", {}).get("num_workers") != 2:
            errors.append("J1 control must use two loader workers")
        if results_cfg.get("data", {}).get("mode") != "stream":
            errors.append("J1 control must use the existing stream reader")
        if results_cfg.get("data", {}).get("cache_root") is not None:
            errors.append("J1 control must not use a cache")
    elif slot == 2:
        if control is None:
            errors.append("J2 validation requires the validated J1 control evidence")
        else:
            _validate_same_performance_universe(evidence, control, errors)
        require_timing("source_workers_2")
        for workers in (0, 2, 4):
            require_timing(f"cache_workers_{workers}_pin_false")
        selected = results.get("selected_loader")
        if not isinstance(selected, dict) or selected.get("workers") not in (0, 2, 4):
            errors.append("J2 did not record an allowed selected loader")
        else:
            timings = results.get("timing") or {}
            require_timing(f"cache_workers_{selected['workers']}_pin_false")
            candidate_sums = {}
            for workers in (0, 2, 4):
                block = timings.get(f"cache_workers_{workers}_pin_false", {})
                candidate_sums[workers] = sum(
                    float(repeat["wall_seconds"])
                    for split in ("train", "validation")
                    for repeat in block.get(split, {}).get("steady_repeats", [])
                )
            if candidate_sums and selected["workers"] != min(
                candidate_sums, key=candidate_sums.get
            ):
                errors.append("J2 selected worker count is not the measured no-pin winner")
            pinned_block = timings.get(f"cache_pin_pair_workers_{selected['workers']}_pin_true", {})
            unpinned_pair = timings.get(
                f"cache_pin_pair_workers_{selected['workers']}_pin_false", {}
            )
            require_timing(f"cache_pin_pair_workers_{selected['workers']}_pin_true")
            require_timing(f"cache_pin_pair_workers_{selected['workers']}_pin_false")
            pinned_sum = sum(
                float(repeat["wall_seconds"])
                for split in ("train", "validation")
                for repeat in pinned_block.get(split, {}).get("steady_repeats", [])
            )
            unpinned_pair_sum = sum(
                float(repeat["wall_seconds"])
                for split in ("train", "validation")
                for repeat in unpinned_pair.get(split, {}).get("steady_repeats", [])
            )
            if bool(selected.get("pin_memory")) != (pinned_sum < unpinned_pair_sum):
                errors.append("J2 pinned-memory decision does not match its paired timing")
        _validate_cache_pair(item, scope, errors)
        lifecycle = results.get("cpu_gpu_validation")
        if (
            not isinstance(lifecycle, dict)
            or not _finite_number(lifecycle.get("absolute_mae_difference"))
            or float(lifecycle["absolute_mae_difference"]) > 0.001
        ):
            errors.append("J2 CPU/GPU validation lifecycle differs by more than 0.001 K")
        elif any(
            not _finite_number(lifecycle.get(key)) or float(lifecycle[key]) <= 0
            for key in ("cpu_wall_seconds", "gpu_wall_seconds", "valid_cells")
        ):
            errors.append("J2 CPU/GPU lifecycle timing or support is invalid")
        elif any(
            not _finite_number(lifecycle.get(key))
            for key in (
                "cpu_identity_mae_k",
                "gpu_identity_mae_k",
                "cpu_correction_mean_abs_k",
                "gpu_correction_mean_abs_k",
            )
        ):
            errors.append("J2 CPU/GPU identity and correction measurements are incomplete")
        elif (
            float(lifecycle["cpu_identity_mae_k"]) > 0.001
            or float(lifecycle["gpu_identity_mae_k"]) > 0.001
            or abs(
                float(lifecycle["cpu_correction_mean_abs_k"])
                - float(lifecycle["gpu_correction_mean_abs_k"])
            )
            > 0.001
        ):
            errors.append("J2 CPU/GPU residual identity/correction differs beyond tolerance")
    elif slot == 3:
        if cache_evidence is None:
            errors.append("J3 validation requires the validated J2 cache evidence")
        else:
            _validate_same_performance_universe(evidence, cache_evidence, errors)
            _validate_selected_cache_loader(results, cache_evidence, errors, slot=3)
        _validate_cache_pair(item, scope, errors)
        cache_record = results.get("cache")
        if not isinstance(cache_record, dict):
            errors.append("J3 has no cache build evidence")
        compute = results.get("compute")
        candidates = compute.get("candidates") if isinstance(compute, dict) else None
        if not isinstance(candidates, dict):
            errors.append("J3 has no counterbalanced compute candidates")
        else:
            for name in ("fp32_batch4", "mixed_batch4", "fp32_batch8_compute_only"):
                repeats = candidates.get(name)
                if not isinstance(repeats, list) or len(repeats) != 3:
                    errors.append(f"J3 candidate {name} needs three counterbalanced repeats")
                    continue
                for repeat in repeats:
                    if not _finite_number(repeat.get("wall_seconds_repeats", [None])[0]):
                        errors.append(f"J3 candidate {name} has invalid measured duration")
        if control is not None and cache_evidence is not None and isinstance(candidates, dict):
            fp32 = candidates.get("fp32_batch4", [])
            amp = candidates.get("mixed_batch4", [])
            cache_results = (cache_evidence.get("efficiency") or {}).get("results", {})
            loader = cache_results.get("selected_loader", {})
            loader_timings = cache_results.get("timing", {})
            loader_block = loader_timings.get(
                f"cache_workers_{loader.get('workers')}_pin_{str(loader.get('pin_memory')).lower()}",
                {},
            )
            loader_seconds = (
                sum(
                    float(repeat["wall_seconds"])
                    for split in ("train", "validation")
                    for repeat in loader_block.get(split, {}).get("steady_repeats", [])
                )
                / 3
            )
            if len(fp32) == 3 and len(amp) == 3:
                fp32_seconds = float(np.median([row["wall_seconds_repeats"][0] for row in fp32]))
                amp_seconds = float(np.median([row["wall_seconds_repeats"][0] for row in amp]))
                batches = math.ceil(384 / 4) + math.ceil(128 / 4)
                projected_gain = (
                    max(0.0, fp32_seconds - amp_seconds)
                    * batches
                    / max(1e-12, loader_seconds + fp32_seconds * batches)
                )
                results["amp_projected_runtime_gain"] = projected_gain
                if (
                    results.get("config", {}).get("trainer", {}).get("precision") == "16-mixed"
                    and projected_gain < 0.10
                ):
                    errors.append(
                        "AMP selection lacks the preregistered 10% projected complete-runtime gain"
                    )
    elif slot == 4:
        require_timing("selected_cache_loader")
        if control is None or cache_evidence is None or diagnostic_evidence is None:
            errors.append("J4 requires validated J1, J2, and J3 evidence")
        else:
            _validate_same_performance_universe(evidence, control, errors)
            _validate_same_performance_universe(evidence, cache_evidence, errors)
            _validate_same_performance_universe(evidence, diagnostic_evidence, errors)
            selected = (
                (cache_evidence.get("efficiency") or {}).get("results", {}).get("selected_loader")
            )
            actual_config = results.get("config") or {}
            if isinstance(selected, dict):
                if actual_config.get("data", {}).get("num_workers") != selected.get("workers"):
                    errors.append("J4 worker count differs from the validated J2 loader winner")
                if actual_config.get("data", {}).get("pin_memory") != selected.get("pin_memory"):
                    errors.append("J4 pin-memory setting differs from the validated J2 winner")
        if cache_evidence is not None:
            learning_manifest = item.get("learning_cache_manifest")
            learning_scope = (item.get("learning") or {}).get("data_scope")
            if not isinstance(learning_manifest, dict) or not isinstance(learning_scope, dict):
                errors.append("J4 is missing the learning-cohort cache manifest/scope")
            else:
                _manifest_valid(learning_manifest, learning_scope, errors, name="learning cache")
        _validate_cache_pair(item, scope, errors)
        learning = _validate_learning(item, evidence, baseline, errors)
        if learning is not None and control is not None:
            _validate_same_learning_cohort(evidence, control, errors)
            control_item = control.get("efficiency") or {}
            control_learning = _validate_learning(control_item, control, baseline, errors)
            if control_learning is not None:
                train, val, _, _ = learning
                base_train, base_val, _, _ = control_learning
                for name, candidate, reference in (
                    ("best validation", min(val), min(base_val)),
                    ("final validation", val[-1], base_val[-1]),
                ):
                    allowed = max(0.05, 0.05 * reference)
                    if candidate - reference > allowed:
                        errors.append(
                            f"J4 {name} degraded by {candidate - reference:.4f} K > {allowed:.4f} K"
                        )
                if train[0] <= 0 or base_train[0] <= 0:
                    errors.append("J1/J4 train curves are invalid")
        precision = (
            (item.get("learning", {}).get("summary", {}).get("resolved_config") or {})
            .get("trainer", {})
            .get("precision")
        )
        if precision not in ("32-true", "16-mixed"):
            errors.append("J4 precision is not an allowed candidate")
        if cache_evidence is not None and diagnostic_evidence is not None:
            diagnostic_results = (diagnostic_evidence.get("efficiency") or {}).get("results", {})
            candidates = (diagnostic_results.get("compute") or {}).get("candidates", {})
            fp32 = candidates.get("fp32_batch4", [])
            amp = candidates.get("mixed_batch4", [])
            if len(fp32) == 3 and len(amp) == 3:
                cache_results = (cache_evidence.get("efficiency") or {}).get("results", {})
                selected = cache_results.get("selected_loader", {})
                loader = cache_results.get("timing", {}).get(
                    f"cache_workers_{selected.get('workers')}_pin_{str(selected.get('pin_memory')).lower()}",
                    {},
                )
                loader_seconds = (
                    sum(
                        float(repeat["wall_seconds"])
                        for split in ("train", "validation")
                        for repeat in loader.get(split, {}).get("steady_repeats", [])
                    )
                    / 3
                )
                fp32_step = float(np.median([row["wall_seconds_repeats"][0] for row in fp32]))
                amp_step = float(np.median([row["wall_seconds_repeats"][0] for row in amp]))
                total_batches = math.ceil(384 / 4) + math.ceil(128 / 4)
                amp_gain = (
                    max(0.0, fp32_step - amp_step)
                    * total_batches
                    / max(1e-12, loader_seconds + fp32_step * total_batches)
                )
                if precision == "16-mixed" and amp_gain < 0.10:
                    errors.append(f"J4 selected 16-mixed but conservative gain is {amp_gain:.1%}")
        elif precision == "16-mixed":
            errors.append("cannot validate AMP selection without J2 and J3 evidence")
    return errors


def _validate_efficiency_config(results: dict, item: dict, errors: list[str]) -> None:
    cfg = results.get("config")
    if not isinstance(cfg, dict):
        _error("efficiency result has no resolved configuration", errors)
        return
    data = cfg.get("data") or {}
    trainer = cfg.get("trainer") or {}
    probe = data.get("probe") or {}
    if cfg.get("stage1_efficiency") is not True or cfg.get("stage1_efficiency_fit") is not False:
        _error("efficiency job config is missing its measurement-only markers", errors)
    if cfg.get("stage1_full") is not False or cfg.get("stage1_lock") is not False:
        _error("efficiency job config carries a full-run marker", errors)
    if cfg.get("stage1_efficiency_role") != item.get("role"):
        _error("efficiency resolved role differs from the evidence role", errors)
    if data.get("splits") != ["train", "validation"]:
        _error("efficiency profile does not restrict access to train/validation", errors)
    if data.get("max_patches_per_split") != 512:
        _error("efficiency profile patch cap differs from its preregistered limit", errors)
    if probe.get("max_refs_per_split") != {"train": 384, "validation": 128}:
        _error("efficiency profile differs from the fixed 384/128 performance cohort", errors)
    if probe.get("require_partial_masks") is not True:
        _error("efficiency profile does not require partial-mask representation", errors)
    if trainer.get("max_epochs") != 1 or data.get("batch_size") != 4:
        _error(
            "efficiency profile is not the bounded batch-4, non-fitting measurement config", errors
        )
    if data.get("n_active_channels") != 10 or cfg.get("stage1_residual_prior") is not True:
        _error("efficiency config changed the frozen Stage-1 representation", errors)
    if data.get("mode") != "stream" or cfg.get("seed") != 0:
        _error("efficiency config changed the source mode or seed", errors)
    if trainer.get("precision") not in ("32-true", "16-mixed"):
        _error("efficiency config precision is outside the preregistered candidates", errors)
    if data.get("num_workers") not in (0, 2, 4) or not isinstance(data.get("pin_memory"), bool):
        _error("efficiency config loader settings are outside the preregistered candidates", errors)


def _read_efficiency_evidence(path: Path) -> dict:
    evidence = _load_object(path)
    if evidence.get("profile") != "efficiency":
        raise ValueError(f"not an efficiency evidence object: {path}")
    return evidence


def _verify_checkpoint_hash(evidence: dict, path: Path) -> str | None:
    checkpoint = _checkpoint_reference(evidence)
    recorded = str((checkpoint or {}).get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", recorded):
        return "evidence has no valid selected-checkpoint SHA-256"
    if not path.is_file():
        return f"downloaded checkpoint not found: {path}"
    if _sha256(path) != recorded:
        return f"downloaded checkpoint SHA-256 differs from evidence: {path}"
    return None


def _checkpoint_reference(evidence: dict) -> dict | None:
    checkpoint = evidence.get("checkpoint")
    return checkpoint if isinstance(checkpoint, dict) else None


def _write_efficiency_ledger(rows: list[dict]) -> None:
    lock = _EFFICIENCY_LEDGER.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        temporary = _EFFICIENCY_LEDGER.with_suffix(".partial")
        temporary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, _EFFICIENCY_LEDGER)
    finally:
        lock.unlink(missing_ok=True)


def _record_revalidation_audit(
    slot: int, evidence: dict, evidence_path: Path, passed: bool
) -> None:
    """Preserve a confirmed J1/J3 validator failure while rechecking its evidence."""
    from datetime import UTC, datetime

    if not _EFFICIENCY_LEDGER.is_file():
        raise RuntimeError("efficiency ledger is missing; cannot record revalidation")
    rows = json.loads(_EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 1 <= slot <= len(rows):
        raise RuntimeError("efficiency ledger does not contain the evidence slot")
    row = rows[slot - 1]
    errors = _failed_revalidation_errors(slot, row, evidence, evidence_path)
    if errors:
        raise RuntimeError("revalidation audit rejected: " + "; ".join(errors))
    spec = _REVALIDATION_SPECS[slot]
    from google.cloud import aiplatform_v1
    from google.cloud.aiplatform_v1.types import JobState

    job = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": "europe-west3-aiplatform.googleapis.com"}
    ).get_custom_job(name=spec["resource_name"])
    if JobState(job.state) != JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError("revalidation resource is no longer JOB_STATE_SUCCEEDED")
    attempts = [
        {
            "attempt": 1,
            "state": "validation_failed",
            "verdict": "fail",
            "evidence_sha256": row["evidence_sha256"],
            "reason": spec["reason"],
        }
    ]
    attempts.append(
        {
            "attempt": 2,
            "state": "validated" if passed else "validation_failed",
            "verdict": "pass" if passed else "fail",
            "evidence_sha256": _sha256(evidence_path),
            "reason": spec["reason"],
        }
    )
    if not passed:
        row["initial_validation_attempts"] = attempts
        _write_efficiency_ledger(rows)
        return
    row.update(
        {
            "state": "validated",
            "validation_verdict": "pass",
            "evidence_sha256": _sha256(evidence_path),
            "initial_validation_attempts": attempts,
            "revalidated_at": datetime.now(UTC).isoformat(),
            "revalidation_note": f"same immutable J{slot} evidence; {spec['reason']}",
        }
    )
    if slot == 3 and passed:
        row["amp_projected_runtime_gain"] = (
            (evidence.get("efficiency") or {}).get("validated_settings") or {}
        ).get("amp_projected_runtime_gain")
    _write_efficiency_ledger(rows)


def _failed_revalidation_errors(
    slot: int, row: dict, evidence: dict, evidence_path: Path
) -> list[str]:
    spec = _REVALIDATION_SPECS.get(slot)
    if spec is None:
        return ["revalidation is not authorized for this efficiency slot"]
    expected = {
        "session": EFFICIENCY_SESSION_ID,
        "slot": slot,
        "role": _SLOT_ROLES[slot],
        "run_label": spec["run_label"],
        "source_sha": spec["source_sha"],
        "image_digest": spec["image_digest"],
        "resource_name": spec["resource_name"],
        "state": "validation_failed",
        "validation_verdict": "fail",
        "terminal_state": "JOB_STATE_SUCCEEDED",
    }
    errors = [
        f"failed ledger {key} differs from the approved revalidation identity"
        for key, value in expected.items()
        if row.get(key) != value
    ]
    item = evidence.get("efficiency") or {}
    if any(
        (
            evidence.get("run_label") != spec["run_label"],
            evidence.get("source_sha") != spec["source_sha"],
            evidence.get("image_digest") != spec["image_digest"],
            item.get("slot") != slot,
            item.get("role") != _SLOT_ROLES[slot],
        )
    ):
        errors.append("evidence identity differs from the approved revalidation artifact")
    if row.get("initial_validation_attempts") is not None:
        errors.append("this failed slot already has a revalidation attempt")
    if not evidence_path.is_file() or row.get("evidence_sha256") != _sha256(evidence_path):
        errors.append("revalidation evidence differs from the initially failed immutable object")
    return errors


def _mark_ledger(slot: int, evidence: dict, evidence_path: Path, passed: bool) -> None:
    if not _EFFICIENCY_LEDGER.is_file():
        raise RuntimeError("efficiency ledger is missing; cannot mark slot validation")
    rows = json.loads(_EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 1 <= slot <= len(rows):
        raise RuntimeError("efficiency ledger does not contain the evidence slot")
    row = rows[slot - 1]
    result = evidence.get("efficiency") or {}
    if row.get("session") != EFFICIENCY_SESSION_ID or row.get("slot") != slot:
        raise RuntimeError("efficiency evidence does not match the active ledger session/slot")
    if row.get("run_label") != evidence.get("run_label") or row.get("run_label") != result.get(
        "results", {}
    ).get("run_label"):
        raise RuntimeError("efficiency evidence run label differs from the reserved slot")
    if row.get("role") != result.get("role"):
        raise RuntimeError("efficiency evidence role differs from the reserved slot")
    if row.get("source_sha") != result.get("results", {}).get("source_sha"):
        raise RuntimeError("efficiency evidence source SHA differs from the reserved slot")
    if row.get("image_digest") != result.get("results", {}).get("image_digest"):
        raise RuntimeError("efficiency evidence image digest differs from the reserved slot")
    if row.get("state") not in ("submitted", "awaiting_validation"):
        raise RuntimeError(
            f"efficiency slot state is {row.get('state')!r}, expected submitted or "
            "awaiting_validation"
        )
    if row.get("terminal_state") not in (None, "JOB_STATE_SUCCEEDED"):
        raise RuntimeError("efficiency slot did not terminate with JOB_STATE_SUCCEEDED")
    resource_name = row.get("resource_name")
    if not resource_name:
        raise RuntimeError(
            "slot has no recorded Vertex job resource; submission outcome is ambiguous"
        )
    from google.cloud import aiplatform_v1
    from google.cloud.aiplatform_v1.types import JobState

    job = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": "europe-west3-aiplatform.googleapis.com"}
    ).get_custom_job(name=str(resource_name))
    if JobState(job.state) != JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError("recorded Vertex job is not JOB_STATE_SUCCEEDED")
    created_at = str(job.create_time) if job.create_time is not None else None
    started_at = str(job.start_time) if job.start_time is not None else None
    ended_at = str(job.end_time) if job.end_time is not None else None
    provisioned_seconds = None
    estimated_compute_cost = None
    if job.start_time is not None and job.end_time is not None:
        provisioned_seconds = _provisioned_seconds(job.start_time, job.end_time)
        estimated_compute_cost = (
            provisioned_seconds * float(row["verified_hourly_rate_usd"]) / 3600.0
        )
    if slot > 1 and any(item.get("state") != "validated" for item in rows[: slot - 1]):
        raise RuntimeError("cannot validate this slot while a prior slot is unvalidated")
    total_compute = EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD + sum(
        float(item.get("projected_exposure_usd", 0.0)) for item in rows
    )
    total_other = max(
        (float(item.get("projected_noncompute_total_usd", 0.0)) for item in rows),
        default=0.0,
    )
    if (
        total_compute > EFFICIENCY_MAX_TOTAL_COMPUTE_USD
        or total_other > EFFICIENCY_MAX_NONCOMPUTE_USD
        or total_compute + total_other > EFFICIENCY_MAX_TOTAL_USD
    ):
        passed = False
    row.update(
        {
            "state": "validated" if passed else "validation_failed",
            "validation_verdict": "pass" if passed else "fail",
            "evidence_sha256": _sha256(evidence_path),
            "created_at": created_at,
            "started_at": started_at,
            "ended_at": ended_at,
            "provisioned_seconds": provisioned_seconds,
            "terminal_state": "JOB_STATE_SUCCEEDED",
            "estimated_compute_cost_usd": estimated_compute_cost,
        }
    )
    validated_settings = result.get("validated_settings") or {}
    if slot == 2 and passed:
        row["selected_loader"] = validated_settings.get("selected_loader")
    if slot == 3 and passed:
        row["amp_projected_runtime_gain"] = validated_settings.get("amp_projected_runtime_gain")
    _write_efficiency_ledger(rows)


def _reservation_errors(
    evidence: dict,
    slot: int,
    *,
    allow_failed_revalidation: bool = False,
    evidence_path: Path | None = None,
) -> list[str]:
    if not _EFFICIENCY_LEDGER.is_file():
        return ["efficiency ledger is missing"]
    rows = json.loads(_EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < slot:
        return ["efficiency ledger has no reservation for this slot"]
    row = rows[slot - 1]
    item = evidence.get("efficiency") or {}
    results = item.get("results") or {}
    budget = results.get("budget") or {}
    errors: list[str] = []
    if row.get("session") != EFFICIENCY_SESSION_ID or row.get("slot") != slot:
        errors.append("ledger session/slot differs from this evidence")
    if row.get("state") not in ("submitted", "awaiting_validation"):
        if not allow_failed_revalidation or evidence_path is None:
            errors.append(f"ledger state is {row.get('state')!r}, not awaiting validation")
        else:
            errors.extend(_failed_revalidation_errors(slot, row, evidence, evidence_path))
    if row.get("run_label") != evidence.get("run_label"):
        errors.append("ledger run label differs from the evidence envelope")
    for label, left, right in (
        ("role", row.get("role"), item.get("role")),
        ("source SHA", row.get("source_sha"), evidence.get("source_sha")),
        ("image digest", row.get("image_digest"), evidence.get("image_digest")),
    ):
        if left != right:
            errors.append(f"ledger {label} differs from the evidence")
    if row.get("hourly_rate_source") != budget.get("hourly_rate_source"):
        errors.append("ledger hourly_rate_source differs from evidence budget")
    for row_key, budget_key in (
        ("verified_hourly_rate_usd", "hourly_rate_usd"),
        ("projected_exposure_usd", "projected_exposure_usd"),
        ("projected_noncompute_total_usd", "projected_noncompute_total_usd"),
    ):
        actual = budget.get(budget_key)
        if not _finite_number(actual) or abs(float(row.get(row_key, -1)) - float(actual)) > 1e-7:
            errors.append(f"ledger {row_key} differs from evidence budget")
    total_compute = EFFICIENCY_HISTORICAL_COMPUTE_RESERVE_USD + sum(
        float(entry.get("projected_exposure_usd", 0.0)) for entry in rows
    )
    total_other = max(
        (float(entry.get("projected_noncompute_total_usd", 0.0)) for entry in rows),
        default=0.0,
    )
    if (
        total_compute > EFFICIENCY_MAX_TOTAL_COMPUTE_USD
        or total_other > EFFICIENCY_MAX_NONCOMPUTE_USD
        or total_compute + total_other > EFFICIENCY_MAX_TOTAL_USD
    ):
        errors.append("efficiency ledger aggregate estimate exceeds its approved caps")
    return errors


def _verified_prior_evidence(path: Path | None, expected_slot: int) -> dict | None:
    if path is None:
        return None
    evidence = _read_efficiency_evidence(path)
    item = evidence.get("efficiency") or {}
    if item.get("slot") != expected_slot or item.get("role") != _SLOT_ROLES[expected_slot]:
        raise ValueError(f"prior evidence is not J{expected_slot} ({_SLOT_ROLES[expected_slot]})")
    rows = json.loads(_EFFICIENCY_LEDGER.read_text(encoding="utf-8"))
    row = rows[expected_slot - 1]
    if row.get("state") != "validated" or row.get("evidence_sha256") != _sha256(path):
        raise ValueError(f"J{expected_slot} evidence is not the validated ledger artifact")
    return evidence


def _validate_cache_pair(item: dict, scope: dict, errors: list[str]) -> None:
    results = item.get("results") or {}
    cache_result = results.get("cache")
    if not isinstance(cache_result, dict):
        errors.append("cache candidate has no source/cache comparison")
        return
    if (
        cache_result.get("scope_equal") is not True
        or cache_result.get("train_sampler_order_equal") is not True
    ):
        errors.append("source/cache scope or seeded sampler order differs")
    exact = cache_result.get("source_cache_exact_samples")
    counts = scope.get("patches_per_split") or {}
    if not isinstance(exact, dict) or any(
        int(exact.get(split, -1)) != int(counts.get(split, 0)) for split in ("train", "validation")
    ):
        errors.append("source/cache exact comparison did not cover all admitted patches")
    manifest = item.get("cache_manifest")
    if not isinstance(manifest, dict):
        errors.append("create-only cache manifest is missing from evidence")
    else:
        _manifest_valid(manifest, scope, errors, name="performance cache")


def _validate_selected_cache_loader(
    candidate_results: dict, cache_evidence: dict, errors: list[str], *, slot: int
) -> None:
    selected = (
        (cache_evidence.get("efficiency") or {}).get("results", {}).get("selected_loader")
    )
    if not isinstance(selected, dict):
        errors.append("validated J2 evidence has no selected cache loader")
        return
    data = (candidate_results.get("config") or {}).get("data") or {}
    if data.get("num_workers") != selected.get("workers"):
        errors.append(f"J{slot} worker count differs from the validated J2 loader winner")
    if data.get("pin_memory") != selected.get("pin_memory"):
        errors.append(f"J{slot} pin-memory setting differs from the validated J2 loader winner")


def _validate_same_performance_universe(left: dict, right: dict, errors: list[str]) -> None:
    left_scope = left.get("data_scope") or {}
    right_scope = right.get("data_scope") or {}
    for key in (
        "requested_patch_ids",
        "patch_ids",
        "skipped_refs",
        "partial_mask_patches_per_split",
        "train_shuffle_order",
    ):
        if left_scope.get(key) != right_scope.get(key):
            errors.append(f"efficiency jobs changed the fixed performance universe ({key})")


def _validate_same_learning_cohort(candidate: dict, control: dict, errors: list[str]) -> None:
    candidate_scope = (candidate.get("efficiency") or {}).get("learning", {}).get(
        "data_scope"
    ) or {}
    control_scope = (control.get("efficiency") or {}).get("learning", {}).get("data_scope") or {}
    for key in (
        "requested_patch_ids",
        "patch_ids",
        "skipped_refs",
        "train_shuffle_order",
        "patches_per_split",
        "partial_mask_patches_per_split",
    ):
        if candidate_scope.get(key) != control_scope.get(key):
            errors.append(f"J1/J4 learning cohort differs in {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--baseline", type=Path, default=_DEFAULT_BASELINE)
    parser.add_argument("--control-evidence", type=Path)
    parser.add_argument("--cache-evidence", type=Path)
    parser.add_argument("--diagnostic-evidence", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--mark-ledger", action="store_true")
    parser.add_argument(
        "--record-revalidation",
        action="store_true",
        help="append a terminal-state-preserving revalidation audit entry to the ledger row",
    )
    args = parser.parse_args()
    if args.self_check:
        failures = _guard_checks()
        failures.extend(_selection_checks())
        failures.extend(_evidence_checks())
        failures.extend(_checkpoint_checks())
        failures.extend(_revalidation_checks())
        with tempfile.TemporaryDirectory(prefix="stage1-efficiency-") as temporary:
            failures.extend(_cache_checks(Path(temporary)))
        if failures:
            print(f"SELF-CHECK FAILED: {failures}")
            return 1
        print("SELF-CHECK OK: efficiency guard and synthetic cache checks passed")
        return 0
    if args.record_revalidation and not args.mark_ledger:
        parser.error("--record-revalidation requires --mark-ledger")

    if args.evidence is None:
        parser.error("--evidence is required unless --self-check is used")
    if not args.evidence.is_file() or not args.baseline.is_file():
        parser.error("evidence and baseline files must exist")

    evidence = _read_efficiency_evidence(args.evidence)
    baseline = _load_object(args.baseline)
    item = evidence.get("efficiency") or {}
    slot = item.get("slot")
    role = item.get("role")
    if not isinstance(slot, int) or slot not in _SLOT_ROLES:
        parser.error("efficiency evidence slot is invalid")
    if slot in (1, 4) and args.checkpoint is None:
        parser.error("learning slots J1/J4 require --checkpoint for SHA-256 verification")
    if slot == 4 and args.control_checkpoint is None:
        parser.error("J4 requires --control-checkpoint for the J1 checkpoint SHA-256 verification")
    if args.record_revalidation and slot not in _REVALIDATION_SPECS:
        parser.error("--record-revalidation is restricted to the confirmed J1/J3 artifacts")
    required_prior = {
        1: (),
        2: ("control",),
        3: ("control", "cache"),
        4: ("control", "cache", "diagnostic"),
    }[slot]
    if "control" in required_prior and args.control_evidence is None:
        parser.error(f"J{slot} requires --control-evidence J1")
    if "cache" in required_prior and args.cache_evidence is None:
        parser.error(f"J{slot} requires --cache-evidence J2")
    if "diagnostic" in required_prior and args.diagnostic_evidence is None:
        parser.error(f"J{slot} requires --diagnostic-evidence J3")
    control = _verified_prior_evidence(args.control_evidence, 1) if args.control_evidence else None
    cache = _verified_prior_evidence(args.cache_evidence, 2) if args.cache_evidence else None
    diagnostic = (
        _verified_prior_evidence(args.diagnostic_evidence, 3) if args.diagnostic_evidence else None
    )
    errors = validate_efficiency_evidence(
        evidence,
        baseline,
        control=control,
        cache_evidence=cache,
        diagnostic_evidence=diagnostic,
    )
    if args.checkpoint is not None:
        mismatch = _verify_checkpoint_hash(evidence, args.checkpoint)
        if mismatch:
            errors.append(mismatch)
    if args.control_checkpoint is not None and control is not None:
        mismatch = _verify_checkpoint_hash(control, args.control_checkpoint)
        if mismatch:
            errors.append(mismatch)
    if args.mark_ledger:
        errors.extend(
            _reservation_errors(
                evidence,
                slot,
                allow_failed_revalidation=args.record_revalidation,
                evidence_path=args.evidence,
            )
        )
    passed = not errors
    if passed and slot == 2:
        item["validated_settings"] = {
            "selected_loader": (item.get("results") or {}).get("selected_loader")
        }
    elif passed and slot == 3 and cache is not None:
        item["validated_settings"] = {
            "amp_projected_runtime_gain": _projected_amp_gain(
                item.get("results") or {},
                (cache.get("efficiency") or {}).get("results", {}),
            )
        }
    for error in errors:
        print(f"FAIL: {error}")
    if passed:
        print(f"PASS: efficiency slot {slot} ({role}) evidence validated")
        if slot == 3 and cache is not None:
            gain = _projected_amp_gain(
                item.get("results") or {},
                (cache.get("efficiency") or {}).get("results", {}),
            )
            if gain is not None:
                print(f"  conservative projected AMP complete-runtime gain: {gain:.1%}")
    if args.mark_ledger:
        if not isinstance(evidence.get("run_label"), str):
            parser.error("cannot mark ledger without a valid evidence run label")
        try:
            if args.record_revalidation:
                _record_revalidation_audit(slot, evidence, args.evidence, passed)
            else:
                _mark_ledger(slot, evidence, args.evidence, passed)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"LEDGER ERROR: {exc}")
            return 1
        action = "revalidated" if args.record_revalidation else "validated"
        print(f"ledger: slot {slot} {action} -> {'pass' if passed else 'fail'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
