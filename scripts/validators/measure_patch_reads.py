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
"""Measure the real patch reader's data phases (issue #35).

Read-only timing probe over the published WB3 sources, run on the compute
host. It does not train, does not convert formats, and writes nothing to
GCS. It answers one question: is re-opening the native target and eligibility
mask for every admitted ref — once in admission and once in the read pass
(``modeling/patches.py:623,642``) — a material share of the data phase at
full-split scale?

What it measures
----------------
1. **In-process attribution.** A ``_TimedReader`` subclass times
   ``_read_target_and_mask`` (the target + mask windows, the duplicated
   work), ``_read_features``, and scene-prior builds for a bounded,
   scene-spread sample. Admission and read are measured separately, so the
   read-pass target/mask time is the duplicate cost.
2. **Streamed worker effect.** A ``RealPatchDataset`` + ``DataLoader`` pass
   over the same admitted sample for ``num_workers`` in {0, 2}, repeated, so
   the duplicated opens and the per-worker scene-prior rebuild show up in the
   end-to-end wall time of the real streaming path.

Sampling is deterministic: refs keep the published index's canonical
``(scene_id, row, col)`` order, and the probe takes up to
``ceil(cap / min-scenes)`` refs per scene until it reaches ``cap`` per split,
so the sample spans several scenes rather than one scene's contiguous block.

# decision: sample with a per-scene cap instead of a plain "first K per
# split", because the index is ordered by scene and a plain prefix can land
# entirely inside one scene, hiding the per-scene prior cost the issue asks
# about. Alternative: plain first-K (rejected: non-representative).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSourceConfig,
    collate_real_batch,
    load_patch_refs,
    patch_index_fingerprints,
)

_SPLITS = ("train", "validation", "test")
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"


# ── timing helpers ────────────────────────────────────────────────────


def _summary(values: list[float]) -> dict[str, Any]:
    """Aggregate a timing series; medians are the reported figure."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min_s": ordered[0],
        "median_s": ordered[len(ordered) // 2],
        "max_s": ordered[-1],
        "sum_s": float(sum(ordered)),
    }


class _Phases:
    """Mutable per-phase timing accumulators."""

    def __init__(self) -> None:
        self.target_mask_s = 0.0
        self.features_s = 0.0
        self.prior_s = 0.0
        self.target_mask_calls = 0
        self.features_calls = 0
        self.prior_builds = 0
        self.total_s = 0.0

    def delta(self, before: _Phases) -> dict[str, Any]:
        return {
            "total_s": self.total_s - before.total_s,
            "target_mask_s": self.target_mask_s - before.target_mask_s,
            "features_s": self.features_s - before.features_s,
            "prior_s": self.prior_s - before.prior_s,
            "target_mask_calls": self.target_mask_calls - before.target_mask_calls,
            "features_calls": self.features_calls - before.features_calls,
            "prior_builds": self.prior_builds - before.prior_builds,
        }

    def copy(self) -> _Phases:
        clone = _Phases()
        clone.__dict__.update(self.__dict__)
        return clone


class _TimedReader(RealPatchReader):
    """Reader that accumulates phase timings without changing behavior.

    Timing lives in a subclass, not in ``modeling/patches.py``: production
    code stays free of measurement instrumentation.
    """

    def __init__(self, cfg: RealSourceConfig) -> None:
        super().__init__(cfg)
        self.phases = _Phases()

    def _prior_for(self, scene_id: str):
        known = scene_id in self._priors
        started = perf_counter()
        try:
            return super()._prior_for(scene_id)
        finally:
            self.phases.prior_s += perf_counter() - started
            if not known:
                self.phases.prior_builds += 1

    def _read_target_and_mask(self, ref: PatchRef, prior):
        started = perf_counter()
        try:
            return super()._read_target_and_mask(ref, prior)
        finally:
            self.phases.target_mask_s += perf_counter() - started
            self.phases.target_mask_calls += 1

    def _read_features(self, ref: PatchRef):
        started = perf_counter()
        try:
            return super()._read_features(ref)
        finally:
            self.phases.features_s += perf_counter() - started
            self.phases.features_calls += 1


# ── config + sampling ─────────────────────────────────────────────────


def _source_from(cfg) -> RealSourceConfig:
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


def _select_sample(
    by_split: dict[str, list[PatchRef]], per_split_cap: int, min_scenes: int
) -> dict[str, list[PatchRef]]:
    """Take a deterministic, scene-spread sample per split, in index order."""
    per_scene = max(1, math.ceil(per_split_cap / min_scenes))
    selected: dict[str, list[PatchRef]] = {}
    for split, refs in by_split.items():
        chosen: list[PatchRef] = []
        per_scene_taken: dict[str, int] = {}
        for ref in refs:
            if len(chosen) >= per_split_cap and len(per_scene_taken) >= min_scenes:
                break
            taken = per_scene_taken.get(ref.scene_id, 0)
            if taken < per_scene:
                chosen.append(ref)
                per_scene_taken[ref.scene_id] = taken + 1
        selected[split] = chosen
    return selected


# ── measurement ───────────────────────────────────────────────────────


def _time_admission(
    reader: _TimedReader, refs: list[PatchRef], deadline: float
) -> tuple[list[PatchRef], dict[str, Any], int]:
    """Time the admission pass; return admitted refs, phase delta, refs seen."""
    before = reader.phases.copy()
    admitted: list[PatchRef] = []
    seen = 0
    started = perf_counter()
    for ref in refs:
        if perf_counter() > deadline:
            break
        seen += 1
        if reader.admission_reason(ref) is None:
            admitted.append(ref)
    reader.phases.total_s += perf_counter() - started
    return admitted, reader.phases.delta(before), seen


def _time_read_pass(
    reader: _TimedReader, refs: list[PatchRef], deadline: float
) -> tuple[dict[str, Any], int]:
    """Time one read pass over the admitted refs."""
    before = reader.phases.copy()
    read = 0
    started = perf_counter()
    for ref in refs:
        if perf_counter() > deadline:
            break
        sample = reader.read_patch(ref)
        if sample is None:
            raise RuntimeError(f"admitted patch {ref.patch_id} became unreadable")
        read += 1
    reader.phases.total_s += perf_counter() - started
    return reader.phases.delta(before), read


def _time_loader_passes(
    source: RealSourceConfig,
    admitted: list[PatchRef],
    *,
    batch_size: int,
    num_workers: int,
    repeats: int,
    deadline: float,
) -> dict[str, Any]:
    """Wall-time repeated DataLoader passes over the real streamed dataset."""
    from torch.utils.data import DataLoader

    from berlin_lst_downscaling.modeling.real_task import RealPatchDataset

    dataset = RealPatchDataset(source, admitted)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_real_batch,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    passes: list[float] = []
    batches = 0
    for _ in range(repeats):
        if perf_counter() > deadline:
            break
        started = perf_counter()
        batches = 0
        for _batch in loader:
            batches += 1
        passes.append(perf_counter() - started)
    return {
        "num_workers": num_workers,
        "worker_processes": num_workers,
        "batches_per_pass": batches,
        "passes": passes,
        **_summary(passes),
    }


def _share(part_s: float, whole_s: float) -> float | None:
    return None if whole_s <= 0 else part_s / whole_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the real patch reader's data phases.")
    parser.add_argument("--config-name", default="real_full")
    parser.add_argument("--patches-per-split", type=int, default=60)
    parser.add_argument("--min-scenes", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1500.0,
        help="internal wall budget; phases stop early and report reduced n",
    )
    parser.add_argument("--sha", default="", help="deployed commit SHA for the record")
    parser.add_argument("--output", default="data/smoke/patch-read-timing/report.json")
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name=args.config_name)
    source = _source_from(cfg)
    deadline = perf_counter() + args.max_seconds

    fingerprints = patch_index_fingerprints(source.patch_index_root)
    print(f"Measuring {source.patch_index_root} (config {args.config_name})")

    all_refs = load_patch_refs(source, splits=_SPLITS)
    by_split: dict[str, list[PatchRef]] = {split: [] for split in _SPLITS}
    for ref in all_refs:
        by_split[ref.split].append(ref)
    sample = _select_sample(by_split, args.patches_per_split, args.min_scenes)
    order = [ref for split in _SPLITS for ref in sample[split]]

    if not order:
        print("FAIL: no patch refs selected — the published index is empty for these splits")
        return 1

    reader = _TimedReader(source)
    # Resolve every scene's Landsat COG/flag once up front, as the data module
    # does, so the admission pass measures per-ref work rather than ledger reads.
    reader.preload({ref.scene_id for ref in order})

    admitted, admission, admitted_seen = _time_admission(reader, order, deadline)
    if not admitted:
        print("FAIL: no ref admitted — cannot measure the read path")
        return 1

    read_passes: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        delta, read = _time_read_pass(reader, admitted, deadline)
        read_passes.append({**delta, "read_refs": read, "truncated": read < len(admitted)})

    first_read = read_passes[0]
    first_read_n = int(first_read["read_refs"])
    warm_reads = read_passes[1:]
    warm_total = (
        {
            "n": len(warm_reads),
            "median_total_s": _median([p["total_s"] for p in warm_reads]),
            "median_target_mask_s": _median([p["target_mask_s"] for p in warm_reads]),
            "median_features_s": _median([p["features_s"] for p in warm_reads]),
            "median_prior_s": _median([p["prior_s"] for p in warm_reads]),
        }
        if warm_reads
        else None
    )
    # Per-ref figures divide by what the pass actually read, so a pass
    # truncated by --max-seconds is not under-reported.
    warm_target_mask_per_ref = [
        p["target_mask_s"] / max(int(p["read_refs"]), 1) for p in warm_reads
    ]
    warm_total_per_ref = [
        p["total_s"] / max(int(p["read_refs"]), 1) for p in warm_reads
    ]

    loader_runs = [
        _time_loader_passes(
            source,
            admitted,
            batch_size=args.batch_size,
            num_workers=workers,
            repeats=args.repeats,
            deadline=deadline,
        )
        for workers in (0, 2)
    ]

    distinct_scenes = sorted({ref.scene_id for ref in order})
    full_per_split = {split: len(refs) for split, refs in by_split.items()}
    full_total = sum(full_per_split.values())

    # Duplicate cost is the read-pass target/mask time: admission already paid
    # the same two windows for every admitted ref.
    duplicate_per_ref_cold = first_read["target_mask_s"] / max(first_read_n, 1)
    duplicate_per_ref_warm = _median(warm_target_mask_per_ref) if warm_reads else None
    data_phase_cold = admission["total_s"] + first_read["total_s"]
    data_phase_warm = (
        admission["total_s"] + warm_total["median_total_s"] if warm_total else None
    )

    report: dict[str, Any] = {
        "sha": args.sha,
        "config_name": args.config_name,
        "patch_index": fingerprints,
        "settings": {
            "patches_per_split": args.patches_per_split,
            "min_scenes": args.min_scenes,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "max_seconds": args.max_seconds,
        },
        "sample": {
            "requested_refs": len(order),
            "admitted_refs": len(admitted),
            "admitted_seen": admitted_seen,
            "distinct_scenes": len(distinct_scenes),
            "requested_per_split": {split: len(refs) for split, refs in sample.items()},
            "admitted_per_split": _count_per_split(admitted, _SPLITS),
        },
        "full_index": {
            "refs_per_split": full_per_split,
            "refs_total": full_total,
            "patches_total": fingerprints["patches_total"],
        },
        "admission": {
            **admission,
            "per_ref_s": admission["total_s"] / max(admitted_seen, 1),
            "target_mask_share": _share(admission["target_mask_s"], admission["total_s"]),
        },
        "read_cold": {
            **first_read,
            "per_ref_s": first_read["total_s"] / max(first_read_n, 1),
            "target_mask_share": _share(first_read["target_mask_s"], first_read["total_s"]),
        },
        "read_warm": (
            {
                **warm_total,
                "per_ref_s": _median(warm_total_per_ref),
                "target_mask_per_ref_s": duplicate_per_ref_warm,
                "target_mask_share": _share(
                    warm_total["median_target_mask_s"], warm_total["median_total_s"]
                ),
            }
            if warm_total
            else None
        ),
        "read_passes": read_passes,
        "duplicate_cost": {
            "per_ref_cold_s": duplicate_per_ref_cold,
            "per_ref_warm_s": duplicate_per_ref_warm,
            "share_of_admission_plus_read_cold": _share(
                first_read["target_mask_s"], data_phase_cold
            ),
            "share_of_admission_plus_read_warm": (
                _share(warm_total["median_target_mask_s"], data_phase_warm)
                if warm_total and data_phase_warm
                else None
            ),
        },
        "loader": loader_runs,
    }
    report["extrapolation"] = _extrapolate(report, full_per_split)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"  sample: {len(order)} requested, {len(admitted)} admitted, "
        f"{len(distinct_scenes)} scenes across {len(_SPLITS)} splits"
    )
    print(
        f"  admission: {admission['total_s']:.2f}s total, "
        f"target/mask {admission['target_mask_s']:.2f}s, "
        f"prior builds {admission['prior_builds']}"
    )
    if warm_total:
        print(
            f"  read cold: {first_read['total_s']:.2f}s "
            f"(target/mask {first_read['target_mask_s']:.2f}s) | "
            f"warm median: {warm_total['median_total_s']:.2f}s "
            f"(target/mask {warm_total['median_target_mask_s']:.2f}s)"
        )
    else:
        print(
            f"  read cold only (no warm pass): {first_read['total_s']:.2f}s "
            f"(target/mask {first_read['target_mask_s']:.2f}s)"
        )
    for run in loader_runs:
        print(
            f"  loader workers={run['num_workers']}: passes={run['passes']} "
            f"median={run.get('median_s', float('nan')):.2f}s"
        )
    print(f"REPORT_JSON: {json.dumps(report)}")
    print("OK: patch read timing measured")
    return 0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _count_per_split(refs: list[PatchRef], splits: tuple[str, ...]) -> dict[str, int]:
    counts = {split: 0 for split in splits}
    for ref in refs:
        counts[ref.split] = counts.get(ref.split, 0) + 1
    return counts


def _extrapolate(report: dict[str, Any], full_per_split: dict[str, int]) -> dict[str, Any]:
    """Approximate full-index duplicate cost from the sample's unit costs.

    Only the duplicate target/mask windows are extrapolated; the admission
    share is not, so the figure is a lower bound on any saving and an upper
    bound on the duplicate cost itself. Training reads the train and
    validation splits, not the test split, so both scopes are reported.
    """
    per_ref_warm = report["duplicate_cost"]["per_ref_warm_s"]
    per_ref_cold = report["duplicate_cost"]["per_ref_cold_s"]
    full_refs = sum(full_per_split.values())
    trained_refs = full_per_split.get("train", 0) + full_per_split.get("validation", 0)

    def _scope(refs: int) -> dict[str, Any]:
        warm = None if per_ref_warm is None else per_ref_warm * refs
        return {
            "refs": refs,
            "duplicate_seconds_warm_s": warm,
            "duplicate_seconds_cold_s": per_ref_cold * refs,
            "duplicate_minutes_warm": None if warm is None else warm / 60.0,
            "duplicate_minutes_cold": per_ref_cold * refs / 60.0,
        }

    return {
        "full_refs": full_refs,
        "train_plus_validation_refs": trained_refs,
        "all_splits": _scope(full_refs),
        "train_plus_validation": _scope(trained_refs),
        "caveat": (
            "sample unit cost applied to the full ref count; the full-index "
            "admitted share is unknown, so both figures are approximations"
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
