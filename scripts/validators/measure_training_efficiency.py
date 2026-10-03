"""Measure bounded train/validation loading and GPU work for Stage-1 efficiency."""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.modeling.contracts import RealBatch, validate_real_batch
from berlin_lst_downscaling.modeling.guards import (
    assert_stage1_efficiency,
    assert_stage1_efficiency_scope,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.metrics import (
    MaskedMAE,
    masked_l1_loss,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patch_cache import assert_samples_equal
from berlin_lst_downscaling.modeling.patches import RealSourceConfig
from berlin_lst_downscaling.modeling.real_task import (
    ProbeScope,
    RealLSTTask,
    RealPatchDataModule,
    reconstruct_prior_kelvin,
)
from berlin_lst_downscaling.modeling.run import real_source_config, run_modeling

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")
_EFFICIENCY_SESSION = "stage1-efficiency-20261002"
_SLOT_ROLES = {1: "baseline", 2: "cache", 3: "diagnostic", 4: "final"}
_REPEATS = 3
_WORKERS = (0, 2, 4)


def _compose(role: str, precision: str, workers: int, pin_memory: bool):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="stage1_efficiency",
            overrides=[
                f"stage1_efficiency_role={role}",
                f"trainer.precision={precision}",
                f"data.num_workers={workers}",
                f"data.pin_memory={str(pin_memory).lower()}",
            ],
        )
    assert_stage1_efficiency(cfg)
    return cfg


def _scope_module(cfg, source: RealSourceConfig, *, cache_root: Path | None) -> RealPatchDataModule:
    probe = cfg.data.probe
    module = RealPatchDataModule(
        source,
        batch_size=int(cfg.data.batch_size),
        max_patches_per_split=int(cfg.data.max_patches_per_split),
        n_active_channels=int(cfg.data.n_active_channels),
        mode=str(cfg.data.mode),
        num_workers=int(cfg.data.num_workers),
        shuffle_train=bool(cfg.data.shuffle_train),
        seed=int(cfg.seed),
        splits=tuple(str(s) for s in cfg.data.splits),
        probe_scope=ProbeScope(
            max_refs_per_split={str(k): int(v) for k, v in probe.max_refs_per_split.items()},
            max_refs_per_scene=int(probe.max_refs_per_scene),
            require_partial_masks=bool(probe.get("require_partial_masks", False)),
        ),
        pin_memory=bool(cfg.data.pin_memory),
        cache_root=cache_root,
        cache_max_bytes=int(cfg.stage1_efficiency_cache_max_bytes),
    )
    return module


def _same_scope(left: dict, right: dict) -> None:
    for key in (
        "requested_per_split",
        "requested_patch_ids",
        "patches_per_split",
        "skipped_refs",
        "patch_ids",
        "years_per_split",
        "partial_mask_patches_per_split",
    ):
        if left.get(key) != right.get(key):
            raise RuntimeError(f"source/cache scope mismatch in {key}")


def _compare_samples(source: RealPatchDataModule, cached: RealPatchDataModule) -> dict[str, int]:
    if source.reader is None:
        raise RuntimeError("source module was not set up")
    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        source_refs = source._admitted[split]
        cache_dataset = cached._datasets[split]
        for index, ref in enumerate(source_refs):
            original = source.reader.read_patch(ref)
            if original is None:
                raise RuntimeError(f"admitted source became unreadable: {ref.patch_id}")
            assert_samples_equal(original, cache_dataset[index], source.n_active_channels)
        counts[split] = len(source_refs)
    return counts


def _iterate_pass(
    module: RealPatchDataModule,
    split: str,
    loader,
    *,
    expect_pinned: bool,
) -> dict:
    waits_ms: list[float] = []
    observed: list[str] = []
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        before = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        waits_ms.append((time.perf_counter() - before) * 1000.0)
        ids = [item.patch_id for item in batch.metadata]
        if expect_pinned:
            tensors = (batch.features, batch.lst_prior, batch.target_100m, batch.mask_100m)
            if not all(tensor.is_pinned() for tensor in tensors):
                raise RuntimeError("pin_memory was enabled but a RealBatch tensor is not pinned")
        observed.extend(ids)
    wall_seconds = time.perf_counter() - started
    expected = list(module.stats()["patch_ids"][split])
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise RuntimeError(f"{split} loader pass changed the admitted patch universe")
    sorted_waits = sorted(waits_ms)
    p95 = sorted_waits[min(len(sorted_waits) - 1, int(0.95 * len(sorted_waits)))]
    return {
        "split": split,
        "patches": len(observed),
        "batches": len(waits_ms),
        "wall_seconds": wall_seconds,
        "batch_wait_median_ms": sorted_waits[len(sorted_waits) // 2],
        "batch_wait_p95_ms": p95,
        "pinned_verified": expect_pinned,
    }


def _measure_loading(module: RealPatchDataModule, repeats: int = _REPEATS) -> dict:
    output: dict[str, dict[str, object]] = {}
    expected_scope = module.stats()
    assert_stage1_efficiency_scope(
        _compose("baseline", "32-true", module.num_workers, module.pin_memory),
        expected_scope,
    )
    for split in ("train", "validation"):
        loader = module.train_dataloader() if split == "train" else module.val_dataloader()
        passes = []
        for _ in range(repeats + 1):
            passes.append(_iterate_pass(module, split, loader, expect_pinned=module.pin_memory))
        output[split] = {
            "first_pass": passes[0],
            "steady_repeats": passes[1:],
        }
    return output


def _measure_paired_loading(
    source: RealPatchDataModule, cached: RealPatchDataModule
) -> tuple[dict, dict]:
    """Counterbalance source/cache loader order at the current two-worker setting."""
    outputs = {"source": {}, "cache": {}}
    for split in ("train", "validation"):
        loaders = {
            "source": source.train_dataloader() if split == "train" else source.val_dataloader(),
            "cache": cached.train_dataloader() if split == "train" else cached.val_dataloader(),
        }
        first_order = ("source", "cache") if split == "train" else ("cache", "source")
        first = {}
        for label in first_order:
            module = source if label == "source" else cached
            first[label] = _iterate_pass(
                module, split, loaders[label], expect_pinned=module.pin_memory
            )
        steady = {"source": [], "cache": []}
        for repeat in range(_REPEATS):
            order = first_order if repeat % 2 == 0 else tuple(reversed(first_order))
            for label in order:
                module = source if label == "source" else cached
                steady[label].append(
                    _iterate_pass(module, split, loaders[label], expect_pinned=module.pin_memory)
                )
        for label in ("source", "cache"):
            outputs[label][split] = {
                "first_pass": first[label],
                "steady_repeats": steady[label],
                "order": [
                    first_order,
                    *[
                        first_order if i % 2 == 0 else tuple(reversed(first_order))
                        for i in range(_REPEATS)
                    ],
                ],
            }
    return outputs["source"], outputs["cache"]


def _measure_loader_candidates(
    module: RealPatchDataModule,
    settings: list[tuple[int, bool]],
) -> dict[str, dict]:
    """Counterbalance candidate DataLoaders using the same admitted dataset."""
    loaders: dict[str, dict[str, object]] = {}
    for workers, pinned in settings:
        module.num_workers = workers
        module.pin_memory = pinned
        loaders[f"workers_{workers}_pin_{str(pinned).lower()}"] = {
            split: module.train_dataloader() if split == "train" else module.val_dataloader()
            for split in ("train", "validation")
        }
    output: dict[str, dict] = {name: {} for name in loaders}
    for split in ("train", "validation"):
        candidate_names = list(loaders)
        first_pass: dict[str, dict] = {}
        for name in candidate_names:
            first_pass[name] = _iterate_pass(
                module, split, loaders[name][split], expect_pinned=name.endswith("pin_true")
            )
        steady = {name: [] for name in candidate_names}
        orders: list[list[str]] = []
        for repeat in range(_REPEATS):
            order = candidate_names if repeat % 2 == 0 else list(reversed(candidate_names))
            orders.append(order)
            for name in order:
                steady[name].append(
                    _iterate_pass(
                        module,
                        split,
                        loaders[name][split],
                        expect_pinned=name.endswith("pin_true"),
                    )
                )
        for name in candidate_names:
            output[name][split] = {
                "first_pass": first_pass[name],
                "steady_repeats": steady[name],
                "counterbalanced_order": orders,
            }
    return output


def _to_device(batch: RealBatch, *, pin_memory: bool) -> RealBatch:
    return batch.to("cuda", non_blocking=pin_memory)


def _compute_measurement(
    cpu_batches: list[RealBatch],
    *,
    precision: str,
    active_channels: int,
    trace_path: Path | None,
    repeats: int = _REPEATS,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("efficiency compute measurement requires a visible CUDA device")
    if len(cpu_batches) < 3:
        raise RuntimeError("at least three fixed batches are required for compute repeats")
    pin_memory = all(
        all(
            tensor.is_pinned()
            for tensor in (batch.features, batch.lst_prior, batch.target_100m, batch.mask_100m)
        )
        for batch in cpu_batches
    )
    torch.cuda.reset_peak_memory_stats()
    transfer_wall: list[float] = []
    transfer_events: list[float] = []
    device_batches: list[RealBatch] = []
    for _repeat in range(repeats):
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        transfer_start.record()
        current_batches = [_to_device(batch, pin_memory=pin_memory) for batch in cpu_batches]
        transfer_end.record()
        transfer_end.synchronize()
        transfer_wall.append(time.perf_counter() - wall_start)
        transfer_events.append(transfer_start.elapsed_time(transfer_end) / 1000.0)
        device_batches = current_batches
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    task = RealLSTTask(
        n_active_channels=active_channels,
        base_width=32,
        depth=4,
        learning_rate=1e-3,
        weight_decay=0,
        residual_prior=True,
    ).to("cuda")
    task.train()
    enabled = precision == "16-mixed"
    if precision not in ("32-true", "16-mixed"):
        raise ValueError(f"unsupported diagnostic precision: {precision}")

    def work(batch: RealBatch) -> None:
        task.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
            prediction = task(batch)
            loss = masked_l1_loss(
                pool_10m_to_100m(prediction.float()), batch.target_100m.float(), batch.mask_100m
            )
        loss.backward()

    for batch in device_batches[:2]:
        work(batch)
    torch.cuda.synchronize()
    wall_repeats: list[float] = []
    event_repeats: list[float] = []
    for repeat in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        work(device_batches[repeat % len(device_batches)])
        end.record()
        end.synchronize()
        wall_repeats.append(time.perf_counter() - wall_start)
        event_repeats.append(start.elapsed_time(end) / 1000.0)

    profiler_record: dict[str, object] | None = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        def save_trace(profile_result) -> None:
            profile_result.export_chrome_trace(str(trace_path))
            trace_path.with_suffix(".txt").write_text(
                profile_result.key_averages().table(sort_by="self_cuda_time_total", row_limit=30),
                encoding="utf-8",
            )

        profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=save_trace,
        )
        with profile:
            for batch in device_batches[:4]:
                work(batch)
                profile.step()
        summary_path = trace_path.with_suffix(".txt")
        profiler_record = {
            "summary": (
                summary_path.read_text(encoding="utf-8")[:6000] if summary_path.is_file() else ""
            ),
            "steps": min(4, len(device_batches)),
        }

    return {
        "precision": precision,
        "batch_size": int(cpu_batches[0].features.shape[0]),
        "loader_pin_memory": pin_memory,
        "transfer_wall_seconds_repeats": transfer_wall,
        "transfer_cuda_event_seconds_repeats": transfer_events,
        "wall_seconds_repeats": wall_repeats,
        "cuda_event_seconds_repeats": event_repeats,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "profiler": profiler_record,
    }


def _compute_batches(
    module: RealPatchDataModule, batch_size: int, count: int = 4
) -> list[RealBatch]:
    original = module.batch_size
    module.batch_size = batch_size
    try:
        loader = module.train_dataloader()
        batches: list[RealBatch] = []
        for batch in loader:
            batches.append(batch)
            if len(batches) >= count:
                break
        return batches
    finally:
        module.batch_size = original


def _compare_cpu_gpu_lifecycle(module: RealPatchDataModule) -> dict[str, float | int]:
    """Compare full bounded validation metrics on CPU and GPU with shared weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("CPU/GPU lifecycle comparison requires CUDA")
    torch.manual_seed(0)
    cpu_task = RealLSTTask(
        n_active_channels=module.n_active_channels, base_width=32, depth=4, residual_prior=True
    ).eval()
    gpu_task = (
        RealLSTTask(
            n_active_channels=module.n_active_channels, base_width=32, depth=4, residual_prior=True
        )
        .to("cuda")
        .eval()
    )
    gpu_task.load_state_dict(cpu_task.state_dict())
    cpu_metric = MaskedMAE()
    gpu_metric = MaskedMAE()
    cpu_identity = MaskedMAE()
    gpu_identity = MaskedMAE()
    cpu_correction_sum = 0.0
    gpu_correction_sum = 0.0
    cpu_cells = 0.0
    gpu_cells = 0.0
    cpu_start = time.perf_counter()
    with torch.inference_mode():
        for batch in module.val_dataloader():
            validate_real_batch(batch, n_active_channels=module.n_active_channels)
            cpu_prediction_10m = cpu_task(batch)
            cpu_prediction = pool_10m_to_100m(cpu_prediction_10m)
            cpu_metric.update(cpu_prediction, batch.target_100m, batch.mask_100m)
            cpu_prior = pool_10m_to_100m(reconstruct_prior_kelvin(batch.lst_prior))
            cpu_identity.update(cpu_prediction, cpu_prior, batch.mask_100m)
            valid_cpu = batch.mask_100m.bool()
            cpu_correction_sum += float((cpu_prediction - cpu_prior)[valid_cpu].abs().sum())
            cpu_cells += float(valid_cpu.sum())
    cpu_seconds = time.perf_counter() - cpu_start

    gpu_start = time.perf_counter()
    with torch.inference_mode():
        for batch in module.val_dataloader():
            gpu_batch = _to_device(
                batch,
                pin_memory=all(
                    tensor.is_pinned()
                    for tensor in (
                        batch.features,
                        batch.lst_prior,
                        batch.target_100m,
                        batch.mask_100m,
                    )
                ),
            )
            validate_real_batch(gpu_batch, n_active_channels=module.n_active_channels)
            gpu_prediction_10m = gpu_task(gpu_batch)
            gpu_prediction = pool_10m_to_100m(gpu_prediction_10m)
            gpu_metric.update(gpu_prediction, gpu_batch.target_100m, gpu_batch.mask_100m)
            gpu_prior = pool_10m_to_100m(reconstruct_prior_kelvin(gpu_batch.lst_prior))
            gpu_identity.update(gpu_prediction, gpu_prior, gpu_batch.mask_100m)
            valid_gpu = gpu_batch.mask_100m.bool()
            gpu_correction_sum += float((gpu_prediction - gpu_prior)[valid_gpu].abs().sum())
            gpu_cells += float(valid_gpu.sum())
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start
    cpu_mae = float(cpu_metric.compute())
    gpu_mae = float(gpu_metric.compute())
    cpu_cells = float(cpu_metric.valid_cells)
    gpu_cells = float(gpu_metric.valid_cells)
    difference = abs(cpu_mae - gpu_mae)
    if cpu_cells != gpu_cells or difference > 1e-3:
        raise RuntimeError(
            f"CPU/GPU validation lifecycle differs: MAE {cpu_mae:.6f}/{gpu_mae:.6f} K, "
            f"cells {cpu_cells}/{gpu_cells}"
        )
    cpu_identity_mae = float(cpu_identity.compute())
    gpu_identity_mae = float(gpu_identity.compute())
    cpu_correction = cpu_correction_sum / cpu_cells
    gpu_correction = gpu_correction_sum / gpu_cells
    if max(cpu_identity_mae, gpu_identity_mae) > 1e-3:
        raise RuntimeError("CPU/GPU residual identity exceeds the 0.001 K tolerance")
    if abs(cpu_correction - gpu_correction) > 1e-3:
        raise RuntimeError("CPU/GPU residual correction differs by more than 0.001 K")
    return {
        "cpu_validation_mae": cpu_mae,
        "gpu_validation_mae": gpu_mae,
        "absolute_mae_difference": difference,
        "cpu_identity_mae_k": cpu_identity_mae,
        "gpu_identity_mae_k": gpu_identity_mae,
        "cpu_correction_mean_abs_k": cpu_correction,
        "gpu_correction_mean_abs_k": gpu_correction,
        "valid_cells": int(cpu_cells),
        "cpu_wall_seconds": cpu_seconds,
        "gpu_wall_seconds": gpu_seconds,
    }


def _run_learning_fit(
    cfg,
    output_root: Path,
    *,
    precision: str,
    workers: int,
    pin_memory: bool,
    cache_root: Path | None,
) -> dict[str, object]:
    learning_root = output_root / "learning"
    overrides = [
        f"output_root={learning_root}",
        "+stage1_efficiency=true",
        "+stage1_efficiency_fit=true",
        f"trainer.precision={precision}",
        f"data.num_workers={workers}",
        f"data.pin_memory={str(pin_memory).lower()}",
        f"stage1_efficiency_cache_max_bytes={int(cfg.stage1_efficiency_cache_max_bytes)}",
    ]
    if cache_root is not None:
        overrides.append(f"data.cache_root={cache_root}")
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        learning_cfg = compose(config_name="stage1_probe", overrides=overrides)
    guard_modeling_config(learning_cfg, "stage1_probe")
    run_id = uuid4().hex[:8]
    with RunLogSession(str(learning_root), pipeline="modeling", run_id=run_id):
        log_event(
            logging.getLogger(__name__),
            logging.INFO,
            "efficiency_learning_fit",
            run_id=run_id,
            config_name="stage1_probe",
            precision=precision,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        result = run_modeling(learning_cfg, run_id=run_id, config_name="stage1_probe")
    if not result.run_ok:
        raise RuntimeError("matched recovery fit returned run_ok=false")
    if cache_root is not None:
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("matched recovery cache has no readiness manifest")
        (output_root / "learning_cache_manifest.json").write_bytes(manifest_path.read_bytes())
    summary_path = learning_root / "probe_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("matched recovery fit completed without probe_summary.json")
    return {
        "run_id": run_id,
        "output_root": str(learning_root),
        "best_checkpoint": result.best_checkpoint,
        "epoch_metrics": str(learning_root / "epoch_metrics.json"),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _environment(output_root: Path) -> dict[str, object]:
    memory: dict[str, object] = {
        "peak_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    cuda: dict[str, object] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    disk = shutil.disk_usage(output_root)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "GDAL_NUM_THREADS",
            )
        },
        "memory": memory,
        "disk": {
            "path": str(output_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "cuda": cuda,
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    expected_env = {
        "VERTEX_EFFICIENCY_SESSION": _EFFICIENCY_SESSION,
        "VERTEX_EFFICIENCY_ROLE": args.role,
        "VERTEX_EFFICIENCY_SLOT": str(args.slot),
        "VERTEX_EFFICIENCY_WORKERS": str(args.workers),
        "VERTEX_EFFICIENCY_PRECISION": args.precision,
        "VERTEX_EFFICIENCY_PIN_MEMORY": str(args.pin_memory).lower(),
        "VERTEX_RUN_LABEL": args.run_label,
        "VERTEX_SOURCE_SHA": args.source_sha,
        "VERTEX_IMAGE_DIGEST": args.image_digest,
        "VERTEX_EFFICIENCY_RATE_USD": str(args.hourly_rate_usd),
        "VERTEX_EFFICIENCY_RATE_SOURCE": args.hourly_rate_source,
        "VERTEX_EFFICIENCY_EXPOSURE_USD": f"{args.projected_exposure_usd:.8f}",
        "VERTEX_EFFICIENCY_NONCOMPUTE_TOTAL_USD": str(args.projected_noncompute_total_usd),
    }
    mismatches = {
        name: (os.environ.get(name), expected)
        for name, expected in expected_env.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"efficiency invocation does not match launcher authorization: {mismatches}"
        )
    if _SLOT_ROLES.get(args.slot) != args.role:
        raise RuntimeError("efficiency slot and role do not match the fixed schedule")
    if args.hourly_rate_usd > 1.03 or args.projected_exposure_usd > 1.0:
        raise RuntimeError("efficiency invocation exceeds its rate or per-job compute cap")
    if (
        args.projected_exposure_usd + args.projected_noncompute_total_usd > 10.0
        or args.projected_noncompute_total_usd > 6.22
    ):
        raise RuntimeError("efficiency invocation exceeds the all-in experiment budget")
    expected_root = Path("data/runs/efficiency") / args.run_label
    if args.output_root.resolve() != expected_root.resolve():
        raise RuntimeError("efficiency output root differs from the launcher-reserved run root")
    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"efficiency output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = _compose(args.role, args.precision, args.workers, args.pin_memory)
    source = real_source_config(cfg)
    started_at = datetime.now(UTC).isoformat()
    monotonic_start = time.monotonic()
    results: dict[str, object] = {
        "status": "running",
        "session": _EFFICIENCY_SESSION,
        "started_at": started_at,
        "role": args.role,
        "slot": args.slot,
        "scratch_optimizer_steps": 0,
        "test_access": False,
        "run_label": args.run_label,
        "source_sha": args.source_sha,
        "image_digest": args.image_digest,
        "budget": {
            "hourly_rate_usd": args.hourly_rate_usd,
            "hourly_rate_source": args.hourly_rate_source,
            "projected_exposure_usd": args.projected_exposure_usd,
            "projected_noncompute_total_usd": args.projected_noncompute_total_usd,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": _environment(output_root),
        "timing": {},
    }
    stage = "performance_setup"
    (output_root / "data").mkdir()
    try:
        source_cfg = _compose("baseline", "32-true", 2, False)
        source_dm = _scope_module(source_cfg, source, cache_root=None)
        setup_start = time.perf_counter()
        source_dm.setup()
        source_setup = time.perf_counter() - setup_start
        source_scope = source_dm.stats()
        assert_stage1_efficiency_scope(source_cfg, source_scope)
        (output_root / "data_scope.json").write_text(
            json.dumps(source_scope, indent=2, sort_keys=True), encoding="utf-8"
        )

        if args.role == "baseline":
            source_dm.num_workers = 2
            results["timing"]["source_workers_2"] = _measure_loading(source_dm)
            results["timing"]["source_setup_seconds"] = source_setup
            cpu_batches = _compute_batches(source_dm, 4)
            results["compute"] = {
                "fp32_batch4": _compute_measurement(
                    cpu_batches,
                    precision="32-true",
                    active_channels=int(cfg.data.n_active_channels),
                    trace_path=output_root / "profile" / "control-fp32.json",
                )
            }
        elif args.role in ("cache", "diagnostic", "final"):
            stage = "cache_build"
            cache_root = output_root / "cache" / "performance"
            cache_root.parent.mkdir(parents=True)
            cache_cfg = _compose("cache", "32-true", 2, False)
            cache_dm = _scope_module(cache_cfg, source, cache_root=cache_root)
            cache_setup_start = time.perf_counter()
            cache_dm.setup()
            cache_setup = time.perf_counter() - cache_setup_start
            cache_scope = cache_dm.stats()
            cache_manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
            (output_root / "cache_manifest.json").write_text(
                json.dumps(cache_manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            assert_stage1_efficiency_scope(cache_cfg, cache_scope)
            _same_scope(source_scope, cache_scope)
            compare_start = time.perf_counter()
            compared = _compare_samples(source_dm, cache_dm)
            compare_seconds = time.perf_counter() - compare_start
            results["cache"] = {
                "root": str(cache_root),
                "manifest": str(output_root / "cache_manifest.json"),
                "array_bytes": cache_manifest["array_bytes"],
                "build_seconds_including_admission": cache_setup,
                "cache_build_seconds": cache_dm.cache_build_seconds,
                "admission_setup_seconds": cache_setup - cache_dm.cache_build_seconds,
                "source_cache_exact_samples": compared,
                "source_cache_comparison_seconds": compare_seconds,
                "source_setup_seconds": source_setup,
                "scope_equal": True,
                "train_sampler_order_equal": (
                    source_scope["train_shuffle_order"] == cache_scope["train_shuffle_order"]
                ),
            }
            if results["cache"]["train_sampler_order_equal"] is not True:
                raise RuntimeError("source/cache seeded train order differs")

            if args.role == "cache":
                source_dm.num_workers = 2
                cache_dm.num_workers = 2
                paired_source, paired_cache = _measure_paired_loading(source_dm, cache_dm)
                results["timing"]["source_workers_2"] = paired_source
                results["timing"]["cache_workers_2_pin_false"] = paired_cache
                results["timing"]["source_setup_seconds"] = source_setup
                worker_candidates = _measure_loader_candidates(cache_dm, [(0, False), (4, False)])
                for key, timing_block in worker_candidates.items():
                    results["timing"][f"cache_{key}"] = timing_block
                estimates = {
                    workers: sum(
                        row["wall_seconds"]
                        for split in ("train", "validation")
                        for row in results["timing"][f"cache_workers_{workers}_pin_false"][split][
                            "steady_repeats"
                        ]
                    )
                    for workers in _WORKERS
                }
                winner = min(estimates, key=estimates.get)
                pin_candidates = _measure_loader_candidates(
                    cache_dm, [(winner, False), (winner, True)]
                )
                pin_estimates = {
                    pinned: sum(
                        row["wall_seconds"]
                        for split in ("train", "validation")
                        for row in pin_candidates[f"workers_{winner}_pin_{str(pinned).lower()}"][
                            split
                        ]["steady_repeats"]
                    )
                    for pinned in (False, True)
                }
                selected_pin = min(pin_estimates, key=pin_estimates.get)
                selected_seconds = pin_estimates[selected_pin]
                for pinned in (False, True):
                    key = f"workers_{winner}_pin_{str(pinned).lower()}"
                    results["timing"][f"cache_pin_pair_{key}"] = pin_candidates[key]
                results["selected_loader"] = {
                    "workers": winner,
                    "pin_memory": selected_pin,
                    "measured_seconds": selected_seconds,
                }
                cache_dm.pin_memory = selected_pin
                results["cpu_gpu_validation"] = _compare_cpu_gpu_lifecycle(cache_dm)
            elif args.role == "diagnostic":
                cache_dm.num_workers = args.workers
                cache_dm.pin_memory = args.pin_memory
                batches4 = _compute_batches(cache_dm, 4)
                batches8 = _compute_batches(cache_dm, 8)
                candidates = {
                    "fp32_batch4": (batches4, "32-true"),
                    "mixed_batch4": (batches4, "16-mixed"),
                    "fp32_batch8_compute_only": (batches8, "32-true"),
                }
                orders = [
                    ("fp32_batch4", "mixed_batch4", "fp32_batch8_compute_only"),
                    ("mixed_batch4", "fp32_batch8_compute_only", "fp32_batch4"),
                    ("fp32_batch8_compute_only", "fp32_batch4", "mixed_batch4"),
                ]
                measured: dict[str, list[dict]] = {key: [] for key in candidates}
                for repeat, order in enumerate(orders):
                    for name in order:
                        batches, precision = candidates[name]
                        measurement = _compute_measurement(
                            batches,
                            precision=precision,
                            active_channels=int(cfg.data.n_active_channels),
                            trace_path=(
                                output_root / "logs" / "modeling" / "profile" / "compute-fp32.json"
                                if name == "fp32_batch4" and repeat == 0
                                else None
                            ),
                            repeats=1,
                        )
                        measured[name].append(measurement)
                results["compute"] = {
                    "counterbalanced_order": orders,
                    "candidates": measured,
                }
            else:
                cache_dm.num_workers = args.workers
                cache_dm.pin_memory = args.pin_memory
                results["timing"]["selected_cache_loader"] = _measure_loading(cache_dm)
                results["compute"] = {
                    f"{args.precision}_batch4": _compute_measurement(
                        _compute_batches(cache_dm, 4),
                        precision=args.precision,
                        active_channels=int(cfg.data.n_active_channels),
                        trace_path=None,
                    )
                }
        else:
            raise ValueError(f"unsupported efficiency role {args.role}")

        results["environment"] = _environment(output_root)
        results["status"] = "complete"
        results["finished_at"] = datetime.now(UTC).isoformat()
        results["elapsed_seconds"] = time.monotonic() - monotonic_start
        result_path = output_root / "efficiency_results.json"
        result_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )

        if args.run_learning_fit:
            stage = "learning_fit"
            learning_cache = output_root / "cache" / "learning" if args.role == "final" else None
            if learning_cache is not None:
                learning_cache.parent.mkdir(parents=True, exist_ok=True)
            fit = _run_learning_fit(
                cfg,
                output_root,
                precision=args.precision,
                workers=args.workers,
                pin_memory=args.pin_memory,
                cache_root=learning_cache,
            )
            results["learning_fit"] = fit
            results["environment"] = _environment(output_root)
            result_path.write_text(
                json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
        return results
    except Exception as exc:
        results["status"] = "incomplete"
        results["finished_at"] = datetime.now(UTC).isoformat()
        results["elapsed_seconds"] = time.monotonic() - monotonic_start
        results["failure"] = {"stage": stage, "exception_type": type(exc).__name__}
        (output_root / "efficiency_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("baseline", "cache", "diagnostic", "final"), required=True
    )
    parser.add_argument("--slot", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--hourly-rate-source", required=True)
    parser.add_argument("--projected-exposure-usd", type=float, required=True)
    parser.add_argument("--projected-noncompute-total-usd", type=float, required=True)
    parser.add_argument("--precision", choices=("32-true", "16-mixed"), default="32-true")
    parser.add_argument("--workers", choices=(0, 2, 4), type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--run-learning-fit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("efficiency jobs require the single pinned Vertex T4 device")
    if args.run_learning_fit != (args.role in ("baseline", "final")):
        parser.error("only the reserved baseline/final efficiency slots run a learning fit")
    print(
        f"efficiency role={args.role} label={args.run_label} device={torch.cuda.get_device_name(0)}"
    )
    result = measure(args)
    print(
        f"efficiency result={args.output_root / 'efficiency_results.json'} "
        f"status={result['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
