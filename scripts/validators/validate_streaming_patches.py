# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "google-cloud-storage>=3.12.0",
#     "torch>=2.2,<2.3",
#     "lightning>=2.6.5",
#     "hydra-core>=1.3.3",
# ]
# ///
"""Independent validator for the real patch streaming path (WB3, issue #30).

Read-only QA over the published sources, run on the compute host rather than a
workstation. It builds the real data module in three shapes — eager, streamed
in-process, and streamed with loader workers plus a seeded shuffle — and checks
that the read path, not just the fit loop, is stream-invariant:

- **Admitted universe.** Every arm must admit exactly the same patch IDs per
  split, record the same skipped refs with the same reasons, and accumulate the
  same exclusion counters as the eager arm. The chosen reader/mask decide the
  universe, so any difference is a streaming defect.
- **Tensor contract.** Every batch of every arm passes
  :func:`validate_real_batch` (geometry, prior channel, finiteness).
- **First-C selection.** When the config selects fewer than the full 28
  channels, the eager batches must equal the first C channels of a bounded
  28-channel reference read, with identical prior/target/mask/metadata and the
  same admitted IDs. The reader, its scaler, and the naive baseline stay on the
  full stack; the selection happens at collation.
- **Identical values.** Eager and streamed-in-process batches are compared
  field by field in canonical order: metadata, mask exactly, and the float
  tensors exactly with masked target ``NaN`` treated as equal.
- **Seeded shuffle.** The shuffled train loader's observed batch order must be
  the recorded sampler order applied to the admitted IDs, must be a permutation
  of the admitted set, and must replay identically from a freshly seeded
  module. A shuffle is never required to differ from index order.

No model is trained and nothing is written: this is the non-training half of
``nox -s smoke-real-comparison``. Configuration (sources, batch size, channels,
seed) is composed from the same Hydra config the runner uses, so the validator
cannot drift from it. Reads are always bounded: an unbounded config (a full
run) is capped to the first four indexed rows per split, never expanded to the
whole split.

Usage
-----
    uv run python scripts/validators/validate_streaming_patches.py
    uv run python scripts/validators/validate_streaming_patches.py \
        --config-name real_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    RealBatch,
    validate_real_batch,
)
from berlin_lst_downscaling.modeling.patches import RealSourceConfig
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule

# The validator is a bounded QA gate: it never reads a full split. An
# unbounded config (a full run) is capped to the first four indexed rows.
_VALIDATOR_MAX_PATCHES = 4

_SPLITS = ("train", "validation", "test")
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"

# Arm name -> (mode, num_workers, shuffle_train). The eager arm is the
# reference; every other arm must match its admitted universe.
_ARMS: dict[str, tuple[str, int, int]] = {
    "eager": ("eager", 0, 0),
    "stream/0": ("stream", 0, 0),
    "stream/2+shuffle": ("stream", 2, 1),
}


def _resolve_config(config_name: str) -> DictConfig:
    """Compose the runner's config, so the validator cannot drift from it."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name=config_name)


def _source_from(cfg: DictConfig) -> RealSourceConfig:
    required = ("patch_index_root", "training_root", "features_root", "ard_root")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"real path requires {missing} in the config")
    return RealSourceConfig(
        patch_index_root=str(cfg.patch_index_root).rstrip("/"),
        training_root=str(cfg.training_root).rstrip("/"),
        features_root=str(cfg.features_root).rstrip("/"),
        ard_root=str(cfg.ard_root).rstrip("/"),
    )


def _arm(
    source: RealSourceConfig,
    *,
    mode: str,
    num_workers: int,
    shuffle_train: bool,
    batch_size: int,
    max_patches: int | None,
    seed: int,
    n_active_channels: int,
) -> RealPatchDataModule:
    return RealPatchDataModule(
        source,
        batch_size=batch_size,
        n_active_channels=n_active_channels,
        max_patches_per_split=max_patches,
        mode=mode,
        num_workers=num_workers,
        shuffle_train=shuffle_train,
        seed=seed,
    )


def _load_batches(module: RealPatchDataModule, split: str) -> list[RealBatch]:
    loader = {
        "train": module.train_dataloader,
        "validation": module.val_dataloader,
        "test": module.test_dataloader,
    }[split]()
    return list(loader)


def _batch_ids(batches: list[RealBatch]) -> list[str]:
    return [meta.patch_id for batch in batches for meta in batch.metadata]


def _float_values_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality of two float tensors, with equal ``NaN`` positions.

    ``torch.equal`` treats ``NaN != NaN``; the masked 100 m target legitimately
    carries ``NaN``, so an exact comparison must match the NaN positions.
    """
    return bool(torch.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))


def _batch_difference(a: RealBatch, b: RealBatch) -> str | None:
    """Return a description of the first field that differs, or None."""
    if a.metadata != b.metadata:
        return "metadata differs"
    if not torch.equal(a.mask_100m, b.mask_100m):
        return "mask_100m differs"
    for name, x, y in (
        ("features", a.features, b.features),
        ("lst_prior", a.lst_prior, b.lst_prior),
        ("target_100m", a.target_100m, b.target_100m),
    ):
        if not _float_values_equal(x, y):
            return f"{name} differs"
    return None


def _first_c_difference(
    a: RealBatch, b: RealBatch, n_active_channels: int
) -> str | None:
    """Return the first field where a first-C batch differs from a 28-channel one.

    ``a`` is the selected-channel batch and ``b`` the full reference: the prior,
    target, mask, and metadata must be identical, and ``a.features`` must equal
    the first ``n_active_channels`` of ``b.features``.
    """
    if a.metadata != b.metadata:
        return "metadata differs"
    if not torch.equal(a.mask_100m, b.mask_100m):
        return "mask_100m differs"
    if not _float_values_equal(a.lst_prior, b.lst_prior):
        return "lst_prior differs"
    if not _float_values_equal(a.target_100m, b.target_100m):
        return "target_100m differs"
    if a.features.shape[1] != n_active_channels:
        return (
            f"selected batch carries {a.features.shape[1]} channels, "
            f"expected {n_active_channels}"
        )
    if not _float_values_equal(a.features, b.features[:, :n_active_channels]):
        return f"features[:{n_active_channels}] differs from the full 28-channel read"
    return None


def _compare_batches(
    name_a: str, name_b: str, a: list[RealBatch], b: list[RealBatch], errors: list[str]
) -> None:
    if len(a) != len(b):
        errors.append(f"{name_a} vs {name_b}: {len(a)} vs {len(b)} batches")
        return
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        difference = _batch_difference(x, y)
        if difference is not None:
            errors.append(f"{name_a} vs {name_b} batch {i}: {difference}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the real patch streaming path without training."
    )
    parser.add_argument(
        "--config-name",
        default="real_smoke",
        help="modeling config to compose (default real_smoke)",
    )
    args = parser.parse_args()

    cfg = _resolve_config(args.config_name)
    source = _source_from(cfg)
    seed = int(cfg.seed)
    batch_size = int(cfg.data.batch_size)
    n_active = int(cfg.data.n_active_channels)
    raw_bound = cfg.data.get("max_patches_per_split")
    # Never expand a full-run config into a full read: cap every config (bounded
    # or unbounded) to the validator's own bound. The effective bound is
    # reported so the evidence states what was read.
    max_patches = (
        _VALIDATOR_MAX_PATCHES
        if raw_bound is None
        else min(int(raw_bound), _VALIDATOR_MAX_PATCHES)
    )

    errors: list[str] = []
    print(f"Validating real streaming patches against {source.patch_index_root}")
    print(
        f"  effective bound: max_patches_per_split={max_patches} (validator cap "
        f"{_VALIDATOR_MAX_PATCHES}), batch_size={batch_size}, "
        f"n_active_channels={n_active}"
    )

    modules: dict[str, RealPatchDataModule] = {}
    for name, (mode, workers, shuffle) in _ARMS.items():
        modules[name] = _arm(
            source,
            mode=mode,
            num_workers=workers,
            shuffle_train=bool(shuffle),
            batch_size=batch_size,
            max_patches=max_patches,
            seed=seed,
            n_active_channels=n_active,
        )
        modules[name].setup()

    scopes = {name: module.stats() for name, module in modules.items()}
    eager = scopes["eager"]

    # ── admitted universe, skipped refs, and exclusions must not depend on mode ──
    for name in _ARMS:
        if name == "eager":
            continue
        scope = scopes[name]
        if scope["mode"] != _ARMS[name][0]:
            errors.append(f"{name}: reported mode {scope['mode']!r} != {_ARMS[name][0]!r}")
        for split in _SPLITS:
            if scope["patch_ids"][split] != eager["patch_ids"][split]:
                errors.append(
                    f"{name}: {split} admitted patch IDs differ from eager "
                    f"({len(scope['patch_ids'][split])} vs {len(eager['patch_ids'][split])})"
                )
            if scope["skipped_per_split"][split] != eager["skipped_per_split"][split]:
                errors.append(f"{name}: {split} skipped-ref count differs from eager")
        if scope["skipped_refs"] != eager["skipped_refs"]:
            errors.append(f"{name}: skipped-ref reasons differ from eager")
        if scope["exclusions"] != eager["exclusions"]:
            errors.append(f"{name}: exclusion counters differ from eager")
    if eager["train_shuffle_order"] is not None:
        errors.append("eager: recorded a train shuffle order")

    # ── materialize every arm; every batch must satisfy the tensor contract ──
    batches: dict[str, dict[str, list[RealBatch]]] = {}
    failed: set[tuple[str, str]] = set()
    for name, module in modules.items():
        batches[name] = {}
        for split in _SPLITS:
            try:
                loaded = _load_batches(module, split)
            except RuntimeError as exc:
                # Either the split admitted nothing or an admitted patch became
                # unreadable; the message carries which, and either is loud.
                errors.append(f"{name}/{split}: loader unavailable: {exc}")
                failed.add((name, split))
                batches[name][split] = []
                continue
            try:
                for batch in loaded:
                    validate_real_batch(batch, n_active_channels=n_active)
            except ValueError as exc:
                errors.append(f"{name}/{split}: contract violation: {exc}")
            batches[name][split] = loaded

    def _usable(name: str, split: str) -> bool:
        """True when both sides of a comparison actually loaded."""
        return (name, split) not in failed

    # ── eager and streamed-in-process must be identical in canonical order ──
    for split in _SPLITS:
        if _usable("eager", split) and _usable("stream/0", split):
            _compare_batches(
                "eager", "stream/0", batches["eager"][split], batches["stream/0"][split], errors
            )

    # ── shuffled arm: validation/test identical; train is the recorded permutation ──
    shuffled_name = "stream/2+shuffle"
    for split in ("validation", "test"):
        if _usable("eager", split) and _usable(shuffled_name, split):
            _compare_batches(
                "eager",
                shuffled_name,
                batches["eager"][split],
                batches[shuffled_name][split],
                errors,
            )

    admitted = list(eager["patch_ids"]["train"])
    recorded = scopes[shuffled_name]["train_shuffle_order"]
    train_loaded = _usable(shuffled_name, "train")
    observed = _batch_ids(batches[shuffled_name]["train"])
    if train_loaded and sorted(observed) != sorted(admitted):
        errors.append(
            f"{shuffled_name}: train batch IDs are not a permutation of the admitted train set "
            f"({len(observed)} vs {len(admitted)})"
        )
    if not recorded:
        errors.append(f"{shuffled_name}: no train shuffle order recorded")
    elif len(recorded) != len(admitted):
        errors.append(
            f"{shuffled_name}: recorded order length {len(recorded)} != admitted {len(admitted)}"
        )
    elif train_loaded:
        expected_order = [admitted[i] for i in recorded]
        if observed != expected_order:
            errors.append(f"{shuffled_name}: loader batch order != recorded sampler order")

    # ── a freshly seeded module must replay the same shuffled order ──
    replay = _arm(
        source,
        mode="stream",
        num_workers=2,
        shuffle_train=True,
        batch_size=batch_size,
        max_patches=max_patches,
        seed=seed,
        n_active_channels=n_active,
    )
    replay.setup()
    if replay.stats()["train_shuffle_order"] != recorded:
        errors.append(f"{shuffled_name}: recorded order not reproduced by a fresh module")
    try:
        replay_order = _batch_ids(_load_batches(replay, "train"))
        if train_loaded and replay_order != observed:
            errors.append(f"{shuffled_name}: fresh loader did not replay the same train order")
    except RuntimeError as exc:
        errors.append(f"{shuffled_name}: replay loader unavailable: {exc}")

    # ── first-C selection must equal the first C of the full 28-channel read ──
    # Only meaningful when the config selects fewer than the full stack; the
    # reference is the same bounded, eager read at 28 channels.
    if n_active != N_FEATURE_CHANNELS:
        reference = _arm(
            source,
            mode="eager",
            num_workers=0,
            shuffle_train=False,
            batch_size=batch_size,
            max_patches=max_patches,
            seed=seed,
            n_active_channels=N_FEATURE_CHANNELS,
        )
        reference.setup()
        ref_scope = reference.stats()
        for split in _SPLITS:
            if ref_scope["patch_ids"][split] != eager["patch_ids"][split]:
                errors.append(
                    f"first-C reference: {split} admitted IDs differ from eager"
                )
        for split in _SPLITS:
            if not _usable("eager", split):
                continue
            try:
                ref_batches = _load_batches(reference, split)
            except RuntimeError as exc:
                errors.append(f"first-C reference/{split}: loader unavailable: {exc}")
                continue
            selected = batches["eager"][split]
            if len(selected) != len(ref_batches):
                errors.append(
                    f"first-C reference vs eager {split}: "
                    f"{len(selected)} vs {len(ref_batches)} batches"
                )
                continue
            for i, (x, y) in enumerate(zip(selected, ref_batches, strict=True)):
                difference = _first_c_difference(x, y, n_active)
                if difference is not None:
                    errors.append(
                        f"first-C vs full-28 {split} batch {i}: {difference}"
                    )

    # ── human-oriented QA summary ──
    for name in _ARMS:
        counts = scopes[name]["patches_per_split"]
        print(
            f"  {name:<17} admitted: " + " ".join(f"{split}={counts[split]}" for split in _SPLITS)
        )
    for e in errors:
        print(f"  ✗ {e}")
    if errors:
        print(f"FAIL: {len(errors)} finding(s)")
        return 1
    total = sum(len(ids) for ids in eager["patch_ids"].values())
    first_c_note = (
        f"; first {n_active} channels equal the full 28-channel read"
        if n_active != N_FEATURE_CHANNELS
        else ""
    )
    print(
        f"OK: streaming path is read-invariant ({total} admitted patches, "
        f"identical IDs/exclusions/tensors across eager, stream/0 and "
        f"stream/2+shuffle; seeded train order reproduced{first_c_note})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
