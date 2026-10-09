# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Random-forest residual baseline runner (Hydra-driven, issue #59).

Trains one forest per Tag-11 stage feature set on a shared seeded reservoir of
eligible train cells, scores every published validation/test patch through the
shared reader/pooling/metrics path, and writes all artifacts under the
configured run output root. Nothing is written to GCS and no canonical artifact
is touched.

Usage
-----
    # Smoke: bounded subset, small trees, local ephemeral output
    uv run python scripts/runners/run_random_forest.py --config-name random_forest_smoke

    # Full: every published validation/test patch, all four stage feature sets
    uv run python scripts/runners/run_random_forest.py --config-name random_forest_full \\
        output_root=data/runs/random-forest/<run-id>

Requires ADC for the read-only GCS sources. Exits non-zero when the train pool
is empty, the per-split accounting does not close, a serialized-forest replay
differs, or no valid cell is evaluated (fail closed).
"""

from __future__ import annotations

import logging
import time
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import hydra
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSourceConfig,
    load_patch_refs,
)
from berlin_lst_downscaling.modeling.random_forest import (
    CONFIG_FILE,
    PROBE_FILE,
    TRAIN_ROWS_FILE,
    EvalSplit,
    ForestSpec,
    StageResult,
    StageSelection,
    build_report,
    collect_eval_split,
    collect_train_pool,
    git_revision_for,
    load_model,
    model_file_name,
    probe_residuals,
    resolve_stage_selection,
    score_stage,
    stage_matrix,
    summarize_records,
    verify_replay,
    write_model,
    write_report,
    write_train_rows,
)

_logger = logging.getLogger(__name__)

_EVAL_SPLITS = ("validation", "test")
_REPLAY_ROWS = 2048
_PROBE_SCALARS = ("patch_id", "split", "filled_feature_pixels")


def _source_config(cfg: DictConfig) -> RealSourceConfig:
    """Build the published-source config from the resolved Hydra config."""
    required = ("patch_index_root", "training_root", "features_root", "ard_root")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"random forest requires {missing} in the config")
    return RealSourceConfig(
        patch_index_root=str(cfg.patch_index_root),
        training_root=str(cfg.training_root),
        features_root=str(cfg.features_root),
        ard_root=str(cfg.ard_root),
    )


def _take_first(refs: list[PatchRef], limit: object) -> list[PatchRef]:
    """Bound a split's refs to the first ``limit`` indexed rows (None = all)."""
    if limit is None:
        return refs
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("patch bound must be an integer or null")
    count = limit
    if count < 0:
        raise ValueError(f"patch bound must be >= 0, got {count}")
    return refs[:count]


def _probe_indices(cache: EvalSplit, per_split: int) -> list[int]:
    """Return the deterministic probe positions of one cached split."""
    return list(range(min(per_split, len(cache.patch_ids))))


@hydra.main(
    config_path="../../configs/modeling", config_name="random_forest_full", version_base=None
)
def main(cfg: DictConfig) -> int:
    """Fit the stage forests, score the eval splits, and write artifacts."""
    started = time.monotonic()
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)
    splits = tuple(str(s) for s in (cfg.get("splits") or _EVAL_SPLITS))
    if not splits or any(split not in _EVAL_SPLITS for split in splits):
        raise ValueError(f"splits must be a non-empty subset of {list(_EVAL_SPLITS)}")
    stages: list[StageSelection] = [resolve_stage_selection(int(s)) for s in cfg.get("stages", ())]
    if not stages:
        raise ValueError("at least one stage is required")
    spec = ForestSpec.from_config(cfg)
    seed = int(cfg.get("seed", 0))
    max_cells = int(cfg.get("max_cells", 500_000))
    max_train_patches = cfg.get("max_train_patches")
    max_patches = cfg.get("max_patches_per_split")
    probe_count = int(cfg.get("probe_patches_per_split", 1))
    if probe_count < 1:
        raise ValueError(f"probe_patches_per_split must be >= 1, got {probe_count}")
    config_name = HydraConfig.get().job.config_name
    profile = str(cfg.get("profile", ""))
    if profile not in ("full", "smoke"):
        raise ValueError("profile must be full or smoke")
    if seed != 0 or max_cells < 1 or max_cells > 500_000:
        raise ValueError("seed must be 0 and the reservoir cap must be <= 500000")
    if [stage.stage for stage in stages] != [1, 2, 3, 4]:
        raise ValueError("RF requires exactly stages 1, 2, 3, 4")
    if splits != _EVAL_SPLITS:
        raise ValueError("RF requires validation and test splits in that order")
    if profile == "full" and (
        max_train_patches is not None or max_patches is not None or max_cells != 500_000
    ):
        raise ValueError("full RF requires all patches and max_cells=500000")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    if any(Path(output_root).glob("random_forest_*")):
        raise ValueError("RF output root already contains artifacts; choose a fresh root")

    with (
        RunLogSession(output_root, pipeline="random_forest", run_id=run_id, level=level),
        TemporaryDirectory(prefix="cache-", dir=output_root) as cache_root,
        wandb.init(
            project=str(cfg.wandb.project),
            mode="offline",
            dir=str(Path(output_root, "logs", "random_forest")),
            name=f"rf-{profile}-{run_id}",
            job_type="random_forest",
            config={
                "git_commit": git_revision_for(output_root, run_id),
                "profile": profile,
                "forest": spec.payload(),
                "seed": seed,
                "max_cells": max_cells,
                "stage_channels": {str(stage.stage): list(stage.names) for stage in stages},
            },
        ) as experiment,
    ):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            output_root=output_root,
            config_name=str(config_name),
            splits=list(splits),
            stages=[stage.stage for stage in stages],
            seed=seed,
            max_cells=max_cells,
            max_train_patches=max_train_patches,
            max_patches_per_split=max_patches,
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            n_jobs=spec.n_jobs,
        )
        source = _source_config(cfg)
        refs = load_patch_refs(source, splits=("train", *splits))
        for ref in refs:
            valid_year = (
                ref.split == "train"
                and 2017 <= ref.year <= 2023
                or ref.split == "validation"
                and ref.year == 2024
                or ref.split == "test"
                and ref.year == 2025
            )
            if not valid_year:
                raise ValueError(f"split/year drift for patch {ref.patch_id}")
        train_refs = _take_first([ref for ref in refs if ref.split == "train"], max_train_patches)
        if not train_refs:
            raise ValueError("no train patch is in scope")
        eval_refs = {
            split: _take_first([ref for ref in refs if ref.split == split], max_patches)
            for split in splits
        }
        reader = RealPatchReader(source)

        Path(output_root).mkdir(parents=True, exist_ok=True)
        Path(output_root, CONFIG_FILE).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

        pool = collect_train_pool(reader, train_refs, max_cells=max_cells, seed=seed)
        if pool.selected_cells == 0:
            raise RuntimeError("the train pool selected no cell")
        rows_path = f"{output_root}/{TRAIN_ROWS_FILE}"
        rows_sha256 = write_train_rows(pool.keys, rows_path)
        log_event(
            _logger,
            logging.INFO,
            "train_pool",
            requested=pool.requested_patches,
            read=pool.read_patches,
            exclusions=pool.exclusions,
            candidate_cells=pool.candidate_cells,
            duplicate_cells=pool.duplicate_cells,
            selected_cells=pool.selected_cells,
            overlapping_windows=pool.overlapping_windows,
            rows_sha256=rows_sha256,
            seconds=round(pool.seconds, 1),
        )

        # Fit one stage at a time, serialize it, and replay the artifact before
        # dropping the in-memory forest; scoring reloads one artifact at a time
        # so at most one forest is resident.
        stage_results: dict[int, StageResult] = {}
        for stage in stages:
            model = spec.build()
            fit_started = time.monotonic()
            matrix = stage_matrix(pool.features, pool.prior_100m, stage)
            model.fit(matrix, pool.residual_k)
            fit_seconds = time.monotonic() - fit_started
            model_path = f"{output_root}/{model_file_name(stage.stage)}"
            model_sha256 = write_model(model, model_path)
            replay = verify_replay(model, model_path, matrix, rows=_REPLAY_ROWS)
            if not replay["identical"]:
                raise RuntimeError(
                    f"stage {stage.stage}: serialized forest does not replay identically"
                )
            stage_results[stage.stage] = StageResult(
                stage=stage.stage,
                channels=stage.names,
                n_active_channels=stage.n_active_channels,
                feature_order=stage.feature_order,
                model_file=model_file_name(stage.stage),
                model_sha256=model_sha256,
                fit_seconds=fit_seconds,
                replay=replay,
                splits={},
                patches=[],
            )
            log_event(
                _logger,
                logging.INFO,
                "stage_fit",
                stage=stage.stage,
                n_active_channels=stage.n_active_channels,
                feature_order=stage.feature_order,
                fit_seconds=round(fit_seconds, 1),
                model_sha256=model_sha256,
                replay_rows=replay["rows"],
                replay_identical=replay["identical"],
            )
            experiment.log({f"stage{stage.stage}/fit_seconds": fit_seconds})
            del model, matrix

        caches: dict[str, EvalSplit] = {}
        for split in splits:
            cache = collect_eval_split(reader, eval_refs[split], split=split, cache_root=cache_root)
            caches[split] = cache
            log_event(
                _logger,
                logging.INFO,
                "eval_split",
                split=split,
                requested=cache.requested_patches,
                evaluated=len(cache.patch_ids),
                valid_cells=int(cache.mask_100m.sum()),
                exclusions=cache.exclusions,
                seconds=round(cache.seconds, 1),
            )

        stage_probes: dict[int, dict[str, dict[int, np.ndarray]]] = {}
        for stage in stages:
            model: RandomForestRegressor = load_model(
                f"{output_root}/{model_file_name(stage.stage)}"
            )
            summaries = {}
            for split in splits:
                cache = caches[split]
                records = score_stage(cache, model, stage)
                summaries[split] = summarize_records(
                    records,
                    requested=cache.requested_patches,
                    exclusions=cache.exclusions,
                )
                stage_results[stage.stage].patches.extend(records)
                stage_probes.setdefault(stage.stage, {})[split] = probe_residuals(
                    cache, model, stage, _probe_indices(cache, probe_count)
                )
                summary = summaries[split]
                log_event(
                    _logger,
                    logging.INFO,
                    "stage_scores",
                    stage=stage.stage,
                    split=split,
                    evaluated=summary.evaluated_patches,
                    valid_cells=summary.valid_cells,
                    mae=summary.mae,
                    ssim=summary.ssim,
                )
            experiment.log(
                {
                    f"stage{stage.stage}/{split}/{metric}": value
                    for split, summary in summaries.items()
                    for metric, value in {
                        "mae_100m_k": summary.mae,
                        "ssim_100m": summary.ssim,
                        "valid_cells": summary.valid_cells,
                    }.items()
                }
            )
            stage_results[stage.stage].splits = summaries
            del model

        probes = _write_probes(caches, stage_probes, probe_count, output_root, splits)
        report = build_report(
            cfg=source,
            spec=spec,
            scope={
                "wandb": {
                    "id": experiment.id,
                    "mode": "offline",
                    "project": str(cfg.wandb.project),
                },
                "eval_splits": {split: len(eval_refs[split]) for split in splits},
                "profile": profile,
                "max_cells": max_cells,
                "seed": seed,
                "train_patches": len(train_refs),
                "max_train_patches": max_train_patches,
                "max_patches_per_split": max_patches,
            },
            pool=pool,
            max_cells=max_cells,
            seed=seed,
            rows_file=TRAIN_ROWS_FILE,
            rows_sha256=rows_sha256,
            stage_results=stage_results,
            probes=probes,
            resolved_config=CONFIG_FILE,
            cache_bytes=sum(
                Path(file).stat().st_size
                for cache in caches.values()
                for file in cache.feature_files
            ),
            runtime_seconds=time.monotonic() - started,
            git_revision=git_revision_for(output_root, run_id),
        )
        uri = write_report(report, f"{output_root}/random_forest_report.json")
        payload = report.to_payload()
        log_event(
            _logger,
            logging.INFO,
            "report",
            uri=uri,
            runtime_seconds=round(float(payload["runtime_seconds"]), 1),
        )

        print(f"Random forest residual baseline - run {run_id} ({payload['method']})")
        print(
            f"  train pool {pool.selected_cells} rows from {pool.read_patches}/"
            f"{pool.requested_patches} patches ({pool.candidate_cells} candidate cells, "
            f"{pool.duplicate_cells} duplicate, {pool.seconds:.0f} s)"
        )
        for stage in stages:
            result = stage_results[stage.stage]
            print(
                f"  stage {stage.stage} ({result.n_active_channels} ch, "
                f"{result.feature_order}): fit {result.fit_seconds:.0f} s"
            )
            for split in splits:
                summary = result.splits[split]
                mae = "n/a" if summary.mae is None else f"{summary.mae:.4f}"
                ssim = "n/a" if summary.ssim is None else f"{summary.ssim:.4f}"
                print(
                    f"    {split:<11} patches {summary.evaluated_patches}/"
                    f"{summary.requested_patches} | cells {summary.valid_cells} | "
                    f"MAE {mae} K | SSIM {ssim} ({summary.ssim_windows} windows)"
                )
        for reason, count in sorted(pool.exclusions.items()):
            print(f"  train excluded [{reason}]: {count}")
        for split in splits:
            for reason, count in sorted(caches[split].exclusions.items()):
                print(f"  {split} excluded [{reason}]: {count}")
        print(f"  Report: {uri}")

        _check_accounting(stage_results, splits, caches)
        return 0


def _write_probes(
    caches: dict[str, EvalSplit],
    stage_probes: dict[int, dict[str, dict[int, np.ndarray]]],
    probe_count: int,
    output_root: str,
    splits: tuple[str, ...],
) -> dict:
    """Write native features, physical prior and per-pixel residual replay probes."""
    entries: list[dict[str, str]] = []
    scalar_arrays: dict[str, list] = {key: [] for key in _PROBE_SCALARS}
    matrix_arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "prior_100m": [],
        "target_100m": [],
        "mask_100m": [],
    }
    residual_arrays: dict[int, list[np.ndarray]] = {stage: [] for stage in stage_probes}
    for split in splits:
        cache = caches[split]
        for index in _probe_indices(cache, probe_count):
            entries.append({"patch_id": cache.patch_ids[index], "split": split})
            scalar_arrays["patch_id"].append(cache.patch_ids[index])
            scalar_arrays["split"].append(split)
            scalar_arrays["filled_feature_pixels"].append(cache.filled_feature_pixels[index])
            matrix_arrays["features"].append(
                np.load(cache.feature_files[index], allow_pickle=False)
            )
            matrix_arrays["prior_100m"].append(cache.prior_100m[index])
            matrix_arrays["target_100m"].append(cache.target_100m[index])
            matrix_arrays["mask_100m"].append(cache.mask_100m[index])
            for stage in stage_probes:
                residual_arrays[stage].append(stage_probes[stage][split][index])
    stacked: dict[str, np.ndarray] = {
        key: np.array(values) for key, values in scalar_arrays.items()
    }
    stacked.update({key: np.stack(values) for key, values in matrix_arrays.items()})
    stacked.update(
        {f"residual_stage{stage}": np.stack(values) for stage, values in residual_arrays.items()}
    )
    probe_path = f"{output_root}/{PROBE_FILE}"
    np.savez(probe_path, **stacked)
    return {
        "file": PROBE_FILE,
        "sha256": sha256(Path(probe_path).read_bytes()).hexdigest(),
        "patches": entries,
    }


def _check_accounting(
    stage_results: dict[int, StageResult],
    splits: tuple[str, ...],
    caches: dict[str, EvalSplit],
) -> None:
    """Raise when the run's own audit does not close."""
    failures: list[str] = []
    first_stage = min(stage_results)
    for stage, result in sorted(stage_results.items()):
        for split in splits:
            summary = result.splits[split]
            accounted = summary.evaluated_patches + sum(summary.exclusions.values())
            if accounted != summary.requested_patches:
                failures.append(
                    f"stage {stage} {split}: accounted {accounted} != "
                    f"requested {summary.requested_patches}"
                )
            if summary.evaluated_patches != len(caches[split].patch_ids):
                failures.append(f"stage {stage} {split}: evaluated patches != cached patches")
            if summary.valid_cells != int(caches[split].mask_100m.sum()):
                failures.append(f"stage {stage} {split}: valid cells differ from the cached mask")
            if summary.valid_cells <= 0:
                failures.append(f"stage {stage} {split}: no valid cell evaluated")
            reference = stage_results[first_stage].splits[split]
            if (summary.valid_cells, summary.evaluated_patches) != (
                reference.valid_cells,
                reference.evaluated_patches,
            ):
                failures.append(f"stage {stage} {split}: evaluated set differs across stages")
        for split in splits:
            ids = [record.patch_id for record in result.patches if record.split == split]
            reference_ids = [
                record.patch_id
                for record in stage_results[first_stage].patches
                if record.split == split
            ]
            if ids != reference_ids:
                failures.append(
                    f"stage {stage} {split}: scored patch IDs differ from stage {first_stage}"
                )
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
