"""Run local, synthetic guard and patch-cache checks for efficiency work."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.contracts import (
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealSampleMeta,
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
from berlin_lst_downscaling.modeling.patches import RealSample

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")


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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("--self-check is currently the only supported mode")

    failures = _guard_checks()
    with tempfile.TemporaryDirectory(prefix="stage1-efficiency-") as temporary:
        failures.extend(_cache_checks(Path(temporary)))
    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK OK: efficiency guard and synthetic cache checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
