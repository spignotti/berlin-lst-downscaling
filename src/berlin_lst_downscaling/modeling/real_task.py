"""Real contract-conforming Lightning path (WB3, issue #18).

The model predicts at 10 m from C selected channels of the release's full
stack (28 for V3, 22 for v6; C = the stack width reads all of it) plus the
separate ``lst_prior`` channel. Training compares the exact 10x10 pooled
prediction against the native 100 m Landsat target under
``training_eligible@100m`` and selects checkpoints on the cell-weighted
masked MAE. SSIM at 100 m is logged as a secondary diagnostic only.

This module is additive: ``modeling/task.py`` and ``modeling/synthetic.py``
keep their synthetic lifecycle and its ``validation/loss`` smoke untouched.

``RealPatchDataModule`` lives here rather than in ``modeling/patches.py`` so
the reader stays free of Lightning and the naive baseline can reuse it
without importing the training framework.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler

from berlin_lst_downscaling.data.io import log_event
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
)
from berlin_lst_downscaling.modeling.aggregation import (
    AGGREGATION_ARITHMETIC,
    AggregationMethod,
    resolve_aggregation_method,
)
from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    PRIOR_AFFINE_OFFSET_K,
    PRIOR_AFFINE_SCALE_K,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealBatch,
    validate_real_batch,
)
from berlin_lst_downscaling.modeling.lr_schedule import log_current_lr, optimizer_config
from berlin_lst_downscaling.modeling.metrics import (
    METRIC_ONLY_SSIM_KEYS,
    SSIM_LEGACY_DATA_RANGE_K,
    THERMAL_AWARE_GRAD_WEIGHT,
    THERMAL_AWARE_SSIM_WEIGHT,
    MaskedMAE,
    MaskedSSIM,
    ScaledSupportWeightedSum,
    SupportCounter,
    SupportedWindows,
    SupportRatio,
    SupportWeightedSum,
    ThermalObjectiveBatch,
    ValidCells,
    masked_l1_loss,
    masked_ssim_stats,
    masked_ssim_stats_patch_range,
    pool_10m_to_100m,
    prediction_outside_lst_range_counts,
    thermal_objective_batch,
)
from berlin_lst_downscaling.modeling.patch_cache import (
    PatchCache,
    build_patch_cache,
    ordered_bounded_map,
    read_with_retry,
    validate_read_workers,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSourceConfig,
    ScalerSpec,
    collate_real_batch,
    load_patch_refs,
    merge_scaler_moments,
    patch_index_fingerprints,
    scaler_from_moments,
)
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    PrebuiltCacheError,
    fetch_prebuilt_cache,
    prebuilt_key,
    prebuilt_key_components,
    prebuilt_prefix,
)
from berlin_lst_downscaling.modeling.published_cache_remap import (
    FeatureRemap,
    fit_feature_remap,
)
from berlin_lst_downscaling.modeling.ssim_range import resolve_task_ssim_data_range
from berlin_lst_downscaling.modeling.timing import timed_phase
from berlin_lst_downscaling.modeling.unet import UNet

logger = logging.getLogger(__name__)

_REAL_SPLITS = ("train", "validation", "test")
# Dataset key of the validation patches on held-out anchors (spatial holdout only).
_HOLDOUT_VALIDATION = "validation_holdout"


@dataclass(frozen=True)
class ProbeScope:
    """Bounded, scene-spread selection for the Stage-1 learning probe.

    Replaces the first-N index bound with a round-robin over scenes in
    canonical ``(scene_id, row, col)`` order, so a bounded cohort is not
    dominated by one or two contiguous scene blocks. The per-split target and
    the per-scene cap are declared here and asserted against the resolved
    config by ``run.py:assert_stage1_probe`` before any source read.
    """

    max_refs_per_split: dict[str, int]
    max_refs_per_scene: int
    require_partial_masks: bool = False


def _batch_count(n_patches: int, batch_size: int) -> int:
    """Number of batches a split of ``n_patches`` collates into (ceil)."""
    if n_patches == 0:
        return 0
    return (n_patches + batch_size - 1) // batch_size


def reconstruct_prior_kelvin(lst_prior: Tensor) -> Tensor:
    """Recover the physical Kelvin prior from the model's normalized prior channel.

    The inverse of the fixed affine in ``contracts.PRIOR_AFFINE_*`` (issue #47).
    Only the prior input and the fixed constants are used, never the target, so
    the reconstruction cannot leak supervision into the residual path.
    """
    return lst_prior * PRIOR_AFFINE_SCALE_K + PRIOR_AFFINE_OFFSET_K


class _BatchDataset(Dataset[RealBatch]):
    """Pre-collated batches for one split (deterministic order)."""

    def __init__(self, batches: list[RealBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> RealBatch:
        return self.batches[idx]


class RealPatchDataset(Dataset[RealSample]):
    """Lazy map-style dataset over admitted patch refs.

    Yields a :class:`RealSample` per indexed ref by reading its COG window on
    demand, so a split's feature tensors are never all resident at once. Refs
    are admitted before construction (see :meth:`RealPatchDataModule.setup`),
    so under stable published sources no unreadable window reaches
    ``__getitem__``; one that does raises rather than silently shrinking the
    comparison universe.

    The reader is created lazily per process, so a worker never inherits an
    open ``rasterio`` dataset from the parent and each worker opens its own on
    first access. This holds under both the ``fork`` and ``spawn`` start
    methods.
    """

    def __init__(
        self,
        source: RealSourceConfig,
        refs: Sequence[PatchRef],
        scaler: ScalerSpec | None = None,
    ) -> None:
        self.source = source
        self.refs = list(refs)
        # None reads with the published scaler; the spatial holdout passes its refit.
        self.scaler = scaler
        self._reader: RealPatchReader | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> RealSample:
        sample = self._reader_for_process().read_patch(self.refs[idx])
        if sample is None:
            ref = self.refs[idx]
            raise RuntimeError(
                f"admitted patch {ref.patch_id} is unreadable — published sources "
                f"changed between admission and read"
            )
        return sample

    def _reader_for_process(self) -> RealPatchReader:
        """Return this process's reader, opening one on first access.

        The scene ledger is preloaded once so worker access does not re-read it
        per scene; the reader caches it for the life of the process.
        """
        if self._reader is None:
            reader = RealPatchReader(self.source)
            if self.scaler is not None:
                reader.scaler = self.scaler
            reader.preload({ref.scene_id for ref in self.refs})
            self._reader = reader
        return self._reader


class CachedRealPatchDataset(Dataset[RealSample]):
    """Lazy read-only view into a job-local cache, reopened per worker process."""

    def __init__(
        self,
        root: Path,
        provenance: dict[str, object],
        indices: Sequence[int],
        remap: FeatureRemap | None = None,
    ) -> None:
        self.root = root
        self.provenance = provenance
        self.indices = list(indices)
        self.remap = remap
        self._cache: PatchCache | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> RealSample:
        if not 0 <= idx < len(self.indices):
            raise IndexError(idx)
        if self._cache is None:
            self._cache = PatchCache(self.root, self.provenance)
        sample = self._cache[self.indices[idx]]
        if self.remap is None:
            return sample
        return replace(sample, features=self.remap(sample.features))


class RealPatchDataModule(LightningDataModule):
    """Contract-conforming real patch batches.

    ``setup`` admits every selected ref once — the same prior, footprint, and
    eligibility-mask checks the reader applies at read time — and records which
    patches entered each split and which were dropped, with a reason. That
    pre-fit accounting is what keeps the model and the naive baseline
    comparable; it reads no feature window.

    ``mode="eager"`` then pre-collates every admitted patch (a bounded smoke
    convenience). ``mode="stream"`` builds a lazy map-style dataset and reads
    each patch on demand in the loader workers, so a full split is never held
    in memory at once. ``max_patches_per_split`` bounds every split to its
    first N indexed rows — the same ref-level bound the naive baseline uses,
    so the two arms cover an identical requested universe. It defaults to
    unbounded, which is a full-run choice and is expected to be invoked
    explicitly, not by CI. ``probe_scope`` replaces that first-N bound with a
    scene-spread selection (see :class:`ProbeScope`). ``splits`` restricts which
    published splits are even read; the Stage-1 probe passes train/validation
    only, so no test ref is ever loaded.

    ``holdout_anchors`` (spatial holdout, default off) is a set of global
    ``(row, col)`` anchors. Train refs on these anchors are dropped before
    admission, the scaler is refit on the admitted train patches only, and the
    validation loader yields the patches on the other anchors;
    ``holdout_val_dataloader`` yields the validation patches on the held-out
    anchors. With ``None`` nothing changes. For the TESSERA arm the published prebuilt
    cache is read as it is: every train ref is admitted, then held-out rows leave the train
    view and the kept features are re-scaled (``published_cache_remap``).
    """

    def __init__(
        self,
        source: RealSourceConfig,
        *,
        batch_size: int = 4,
        n_active_channels: int = N_FEATURE_CHANNELS,
        channel_indices: Sequence[int] | None = None,
        max_patches_per_split: int | None = None,
        scene_ids: Sequence[str] | None = None,
        mode: str = "eager",
        num_workers: int = 0,
        shuffle_train: bool = False,
        seed: int = 0,
        splits: Sequence[str] = _REAL_SPLITS,
        probe_scope: ProbeScope | None = None,
        pin_memory: bool = False,
        cache_root: Path | None = None,
        cache_max_bytes: int = 0,
        cache_build_workers: int = 1,
        reuse_ready_cache: bool = False,
        cache_source: str = "job_local",
        prebuilt_uri_root: str | None = None,
        allow_job_local_fallback: bool = False,
        prebuilt_download_workers: int = 8,
        prebuilt_download_retries: int = 3,
        tessera_embeddings_root: str | None = None,
        holdout_anchors: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        if holdout_anchors is not None:
            if not holdout_anchors:
                raise ValueError("holdout_anchors must be None or a non-empty set")
            if tuple(splits) != ("train", "validation"):
                raise ValueError("spatial holdout requires splits ('train', 'validation')")
            if tessera_embeddings_root:
                # TESSERA: read the published cache (read-only), then drop held-out rows,
                # refit the scaler and re-scale the cached features (published_cache_remap).
                if cache_source != "prebuilt" or allow_job_local_fallback:
                    raise ValueError(
                        "spatial holdout for the TESSERA arm needs the prebuilt patch cache "
                        "without a job-local fallback"
                    )
            elif cache_source != "job_local":
                raise ValueError("spatial holdout supports only the job-local patch cache")
        self.holdout_anchors = holdout_anchors
        self._remap_published = holdout_anchors is not None and bool(tessera_embeddings_root)
        self.reuse_ready_cache = reuse_ready_cache
        if cache_source not in ("job_local", "prebuilt"):
            raise ValueError(f"cache_source {cache_source!r} is not one of 'job_local', 'prebuilt'")
        if cache_source == "prebuilt" and (cache_root is None or not prebuilt_uri_root):
            raise ValueError("cache_source 'prebuilt' requires cache_root and prebuilt_uri_root")
        if prebuilt_download_workers < 1 or prebuilt_download_retries < 1:
            raise ValueError("prebuilt download workers and retries must be >= 1")
        self.cache_source = cache_source
        self.prebuilt_uri_root = prebuilt_uri_root
        self.allow_job_local_fallback = allow_job_local_fallback
        self.prebuilt_download_workers = prebuilt_download_workers
        self.prebuilt_download_retries = prebuilt_download_retries
        self.cache_build_workers = validate_read_workers(cache_build_workers)
        if mode not in ("eager", "stream"):
            raise ValueError(f"data mode {mode!r} is not one of 'eager', 'stream'")
        if shuffle_train and mode != "stream":
            raise ValueError("shuffle_train is only supported by the streaming mode")
        unknown_splits = sorted(set(splits) - set(_REAL_SPLITS))
        if unknown_splits:
            raise ValueError(f"unknown split(s) {unknown_splits}; expected {_REAL_SPLITS}")
        if channel_indices is None:
            resolved_indices = tuple(range(n_active_channels))
        else:
            resolved_indices = tuple(int(i) for i in channel_indices)
            if len(resolved_indices) != n_active_channels:
                raise ValueError(
                    f"channel_indices length {len(resolved_indices)} != "
                    f"n_active_channels {n_active_channels}"
                )
        self.tessera_embeddings_root = tessera_embeddings_root or None
        self.source = source
        self.batch_size = batch_size
        self.n_active_channels = n_active_channels
        self.channel_indices = resolved_indices
        self.max_patches_per_split = max_patches_per_split
        self.scene_ids = tuple(scene_ids) if scene_ids else None
        self.mode = mode
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.seed = seed
        self.splits = tuple(splits)
        self.probe_scope = probe_scope
        self.pin_memory = pin_memory
        self.cache_root = cache_root
        self.cache_max_bytes = cache_max_bytes
        self._datasets: dict[str, _BatchDataset | RealPatchDataset | CachedRealPatchDataset] = {}
        self._setup_done = False
        self._reset_setup_state()

    def _reset_setup_state(self) -> None:
        """Clear everything ``setup`` populates so a failed setup can be retried."""
        self.cache_build_seconds = 0.0
        self.source_setup_seconds = 0.0
        self.admission_seconds = 0.0
        self.cache_reused = False
        self.patch_cache_source: str | None = None
        self.patch_cache_prefix: str | None = None
        self.patch_cache_key: str | None = None
        self.prebuilt_download_seconds: float | None = None
        self.prebuilt_verify_seconds: float | None = None
        self.prebuilt_load_seconds: float | None = None
        self.prebuilt_bytes: int | None = None
        self.prebuilt_shards: int | None = None
        self.prebuilt_build_provenance: dict[str, object] | None = None
        self.prebuilt_commit_match: bool | None = None
        self.prebuilt_build_commit: str | None = None
        self.prebuilt_running_commit: str | None = None
        self.holdout_scaler: ScalerSpec | None = None
        self._feature_remap_evidence: dict[str, Any] | None = None
        self._published_provenance: dict[str, object] | None = None
        self.holdout_scaler_fit_seconds = 0.0
        self._holdout_train_removed = 0
        self.reader: RealPatchReader | None = None
        self._admitted: dict[str, list[PatchRef]] = {}
        self._requested: dict[str, list[PatchRef]] = {split: [] for split in self.splits}
        self._skipped: dict[str, list[dict[str, str]]] = {split: [] for split in self.splits}
        self._datasets.clear()

    def _select_refs(self, refs: list[PatchRef]) -> dict[str, list[PatchRef]]:
        """Bound the requested refs per split, before any exclusion.

        The naive baseline and the model must cover an identical requested
        universe, so selection happens on refs and admission never refills a
        dropped row. ``probe_scope`` selects a deterministic scene-spread
        cohort; otherwise the first-N bound matches the baseline exactly.
        """
        if self.probe_scope is None:
            taken: Counter[str] = Counter()
            requested: dict[str, list[PatchRef]] = {split: [] for split in self.splits}
            for ref in refs:
                if (
                    self.max_patches_per_split is not None
                    and taken[ref.split] >= self.max_patches_per_split
                ):
                    continue
                taken[ref.split] += 1
                requested[ref.split].append(ref)
            return requested

        by_split_scene: dict[str, dict[str, list[PatchRef]]] = {split: {} for split in self.splits}
        for ref in refs:
            by_split_scene.get(ref.split, {}).setdefault(ref.scene_id, []).append(ref)

        requested = {split: [] for split in self.splits}
        for split in self.splits:
            target = self.probe_scope.max_refs_per_split.get(split)
            if target is None or target <= 0:
                continue
            scenes = by_split_scene[split]
            per_scene = {
                scene_id: rows[: self.probe_scope.max_refs_per_scene]
                for scene_id, rows in scenes.items()
            }
            if self.probe_scope.require_partial_masks:
                total_cells = REAL_PATCH_CELLS * REAL_PATCH_CELLS
                for scene_id in scenes:
                    partial = next(
                        (ref for ref in per_scene[scene_id] if 0 < ref.n_eligible < total_cells),
                        None,
                    )
                    if partial is not None:
                        requested[split].append(partial)
                        break
                if not requested[split]:
                    raise ValueError(
                        f"{split} efficiency cohort has no partial-mask candidate in index metadata"
                    )
            index = 0
            while len(requested[split]) < target:
                progressed = False
                for scene_id in scenes:
                    rows = per_scene[scene_id]
                    if index < len(rows) and all(
                        chosen.patch_id != rows[index].patch_id for chosen in requested[split]
                    ):
                        requested[split].append(rows[index])
                        progressed = True
                        if len(requested[split]) >= target:
                            break
                if not progressed:
                    break
                index += 1
        return requested

    def setup(self, stage: str | None = None) -> None:
        # Idempotent once preparation completed: admission is expensive and the
        # lifecycle may set the module up explicitly (to record the read scope)
        # before ``fit``. A failed setup leaves no partial state and can retry.
        if self._setup_done:
            return
        try:
            self._prepare()
        except BaseException:
            self._reset_setup_state()
            raise
        self._setup_done = True

    def _prepare(self) -> None:
        with timed_phase("data_ref_loading", logger=logger):
            self.reader = RealPatchReader(self.source)
            refs = load_patch_refs(self.source, splits=self.splits, scene_ids=self.scene_ids)
        # Bound the *refs* per split, before any exclusion, so the requested
        # universe matches the baseline's selection exactly.
        requested = self._select_refs(refs)
        if self.holdout_anchors is not None and not self._remap_published:
            kept = [ref for ref in requested["train"] if not self._held_out(ref)]
            self._holdout_train_removed = len(requested["train"]) - len(kept)
            requested["train"] = kept
        self._requested = requested

        # Resolve every scene's Landsat COG/flag once, so the admission pass
        # reads the ARD ledger a single time rather than once per scene.
        # Timed separately from per-ref admission (which includes first-use
        # scene-prior construction) so later projections can scale each phase.
        setup_started = time.perf_counter()
        with timed_phase("source_preload", logger=logger, scenes=len({r.scene_id for r in refs})):
            self.reader.preload({ref.scene_id for ref in refs})
        self.source_setup_seconds = time.perf_counter() - setup_started
        admission_started = time.perf_counter()
        with timed_phase("admission", logger=logger):
            for split, split_refs in requested.items():
                admitted = self._admit(self.reader, split, split_refs)
                self._admitted[split] = admitted
        self.admission_seconds = time.perf_counter() - admission_started
        if self.holdout_anchors is not None:
            seen, unseen = self._validation_partition()
            if not seen or not unseen:
                raise RuntimeError(
                    f"spatial holdout needs validation patches on both sides, got "
                    f"{len(seen)} seen and {len(unseen)} held out"
                )
            if not self._remap_published:
                self._fit_holdout_scaler()
        if self.cache_root is not None:
            # The cache build reads every admitted patch once; an eager
            # pre-collation would read the features a second time.
            if self.cache_source == "prebuilt":
                self._materialize_prebuilt()
            else:
                self._build_cache()
                self.patch_cache_source = "job_local"
            return
        live_refs: dict[str, list[PatchRef]] = dict(self._admitted)
        if self.holdout_anchors is not None:
            seen, unseen = self._validation_partition()
            validation = self._admitted["validation"]
            live_refs["validation"] = [validation[i] for i in seen]
            live_refs[_HOLDOUT_VALIDATION] = [validation[i] for i in unseen]
        for split, admitted in live_refs.items():
            if self.mode == "eager":
                self._datasets[split] = _BatchDataset(self._read_batches(self.reader, admitted))
            else:
                self._datasets[split] = RealPatchDataset(
                    self.source, admitted, scaler=self.holdout_scaler
                )

    def _held_out(self, ref: PatchRef) -> bool:
        return self.holdout_anchors is not None and (ref.row, ref.col) in self.holdout_anchors

    def _validation_partition(self) -> tuple[list[int], list[int]]:
        """Indices into the admitted validation refs: seen anchors, held-out anchors."""
        seen: list[int] = []
        unseen: list[int] = []
        for index, ref in enumerate(self._admitted.get("validation", [])):
            (unseen if self._held_out(ref) else seen).append(index)
        return seen, unseen

    def _fit_holdout_scaler(self) -> None:
        """Refit the scaler on the admitted train patches and install it on the reader.

        One threaded pass reads every remaining train patch unscaled and merges
        the per-patch moments in canonical order, so the result does not depend
        on the worker count. The cache build and every later read then scale
        with this refit; the published scaler only supplies order and transforms.
        """
        if self.reader is None:
            raise RuntimeError("the holdout scaler fit requires an initialized reader")
        reader = self.reader
        refs = self._admitted.get("train", [])
        if not refs:
            raise RuntimeError("spatial holdout left no admitted train patch to fit the scaler")
        workers = self.cache_build_workers
        local = threading.local()

        def moments_of(ref: PatchRef):
            if workers == 1:
                thread_reader = reader
            else:
                thread_reader = getattr(local, "reader", None)
                if thread_reader is None:
                    thread_reader = reader.copy_for_reads()
                    local.reader = thread_reader
            return read_with_retry(
                lambda: thread_reader.read_scaler_moments(ref), patch_id=ref.patch_id
            )

        started = time.perf_counter()
        with timed_phase(
            "holdout_scaler_fit",
            logger=logger,
            cache_build_workers=workers,
            patch_count=len(refs),
        ):
            total = None
            parts = ordered_bounded_map(moments_of, refs, max_workers=workers)
            try:
                for part in parts:
                    total = merge_scaler_moments(total, part)
            finally:
                parts.close()
        if total is None:
            raise RuntimeError("the holdout scaler fit read no patch")
        self.holdout_scaler = scaler_from_moments(reader.scaler, total)
        reader.scaler = self.holdout_scaler
        self.holdout_scaler_fit_seconds = time.perf_counter() - started

    def _cache_patch_order(self) -> list[PatchRef]:
        """Admitted refs in cache row order: train rows, then validation rows."""
        if set(self.splits) != {"train", "validation"}:
            raise ValueError("job-local cache requires exactly train and validation splits")
        ordered_refs = [ref for split in ("train", "validation") for ref in self._admitted[split]]
        if not ordered_refs:
            raise RuntimeError("cannot build an empty job-local cache")
        return ordered_refs

    def _cache_provenance(self, ordered_refs: list[PatchRef]) -> tuple[dict[str, object], int]:
        """Return the cache provenance and the source channel width."""
        if self.reader is None:
            raise RuntimeError("cache provenance requires an initialized reader")
        reader = self.reader
        scope = self.stats()
        scaler_record = asdict(reader.scaler)
        scaler_digest = hashlib.sha256(
            json.dumps(scaler_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        source_channels = len(reader.scaler.channel_order)
        if len(reader.scaler.channels) != source_channels:
            raise RuntimeError(
                f"scaler channel entries ({len(reader.scaler.channels)}) disagree with "
                f"channel_order ({source_channels}); refusing to size the cache"
            )
        channel_order = [reader.scaler.channel_order[i] for i in self.channel_indices]
        provenance: dict[str, object] = {
            "source_roots": asdict(self.source),
            "patch_index_fingerprints": patch_index_fingerprints(self.source.patch_index_root),
            "scaler_digest": scaler_digest,
            "channel_order": channel_order,
            "channel_indices": list(self.channel_indices),
            "source_channels": source_channels,
            "geometry": {
                "crs": "EPSG:25833",
                "origin_x": CANON_GRID_ORIGIN_X,
                "origin_y": CANON_GRID_ORIGIN_Y,
                "cell_size_m": CELL_SIZE_M,
                "patch_px": REAL_PATCH_PX,
                "target_cells": REAL_PATCH_CELLS,
                "anchors": [[ref.row, ref.col] for ref in ordered_refs],
            },
            "preprocessing_version": 2,
            "aggregation": self.source.aggregation,
            "requested_patch_ids": scope["requested_patch_ids"],
            "admitted_patch_ids": scope["patch_ids"],
            "skipped_refs": scope["skipped_refs"],
        }
        if self.tessera_embeddings_root:
            provenance["tessera"] = {
                "root": self.tessera_embeddings_root,
                "encoding": "int8-scale-float32-v1",
            }
        return provenance, source_channels

    def cache_provenance(self) -> dict[str, object]:
        """The provenance the cache of this (set up) module is keyed on."""
        if self._published_provenance is not None:
            return self._published_provenance
        return self._cache_provenance(self._cache_patch_order())[0]

    def prebuilt_components(self) -> dict[str, object]:
        """Key components of the prebuilt cache this (set up) module resolves to."""
        if self.reader is None:
            raise RuntimeError("setup() must run before prebuilt_components()")
        return prebuilt_key_components(
            self.cache_provenance(),
            source_channel_order=list(self.reader.scaler.channel_order),
            splits=("train", "validation"),
        )

    def _install_remapped_datasets(self, cache_root: Path, provenance: dict[str, object]) -> None:
        """Install the TESSERA holdout datasets over the unchanged published cache.

        The cache holds every admitted train row scaled with the published scaler. The
        held-out train rows are left out of the train view, the scaler is refit on the
        remaining train rows and every dataset re-scales the kept feature channels with it.
        """
        # tessera_adapter imports this module, so the import cannot sit at the top.
        from berlin_lst_downscaling.modeling.tessera_adapter import tessera_kept_channel_indices

        if self.reader is None:
            raise RuntimeError("the holdout remap requires an initialized reader")
        train_refs = self._admitted["train"]
        n_train = len(train_refs)
        kept = [i for i, ref in enumerate(train_refs) if not self._held_out(ref)]
        if not kept:
            raise RuntimeError("spatial holdout left no admitted train patch to fit the scaler")
        seen, unseen = self._validation_partition()
        validation_rows = [n_train + i for i in sorted(seen + unseen)]
        started = time.perf_counter()
        with timed_phase("holdout_scaler_fit", logger=logger, patch_count=len(kept)):
            remap, refit, evidence = fit_feature_remap(
                PatchCache(cache_root, provenance),
                published=self.reader.scaler,
                channel_indices=self.channel_indices,
                columns=tessera_kept_channel_indices(),
                train_rows=kept,
                check_rows=[*kept, *validation_rows],
                workers=self.cache_build_workers,
            )
        self.holdout_scaler_fit_seconds = time.perf_counter() - started
        self.holdout_scaler = refit
        self._feature_remap_evidence = {**evidence, "patch_cache_key": self.patch_cache_key}
        self._published_provenance = provenance
        self._holdout_train_removed = n_train - len(kept)
        self._admitted["train"] = [train_refs[i] for i in kept]
        self._datasets["train"] = CachedRealPatchDataset(cache_root, provenance, kept, remap)
        self._datasets["validation"] = CachedRealPatchDataset(
            cache_root, provenance, [n_train + i for i in seen], remap
        )
        self._datasets[_HOLDOUT_VALIDATION] = CachedRealPatchDataset(
            cache_root, provenance, [n_train + i for i in unseen], remap
        )

    def _install_cached_datasets(self, cache_root: Path, provenance: dict[str, object]) -> None:
        if self._remap_published:
            self._install_remapped_datasets(cache_root, provenance)
            return
        offset = 0
        for split in ("train", "validation"):
            refs = self._admitted[split]
            if split == "validation" and self.holdout_anchors is not None:
                # The cache keeps every validation row; the two loaders index into it.
                seen, unseen = self._validation_partition()
                self._datasets[split] = CachedRealPatchDataset(
                    cache_root, provenance, [offset + i for i in seen]
                )
                self._datasets[_HOLDOUT_VALIDATION] = CachedRealPatchDataset(
                    cache_root, provenance, [offset + i for i in unseen]
                )
            else:
                self._datasets[split] = CachedRealPatchDataset(
                    cache_root, provenance, range(offset, offset + len(refs))
                )
            offset += len(refs)

    def _build_cache(self) -> None:
        """Build an ordered cache from admitted refs and replace split datasets."""
        if self.cache_root is None or self.reader is None:
            raise RuntimeError("cache build requires a configured path and initialized reader")
        reader = self.reader
        ordered_refs = self._cache_patch_order()
        patch_ids = [ref.patch_id for ref in ordered_refs]
        provenance, source_channels = self._cache_provenance(ordered_refs)

        cache_root = self.cache_root
        ready = (cache_root / "manifest.json").is_file()
        if cache_root.exists() and not ready:
            raise RuntimeError(
                f"cache root {cache_root} exists without a ready manifest; "
                "refusing to reuse or delete an incomplete cache"
            )
        if ready and not self.reuse_ready_cache:
            # The provenance does not pin the reader code revision, so a cache
            # left by an earlier run is refused unless reuse is requested.
            raise RuntimeError(
                f"cache root {cache_root} already holds a ready cache; "
                "refusing to reuse it across runs"
            )
        self.cache_reused = ready
        with timed_phase(
            "cache_build",
            logger=logger,
            cache_build_workers=self.cache_build_workers,
            patch_count=len(patch_ids),
            cache_reused=ready,
        ):
            cache_started = time.perf_counter()
            if ready:
                self._verify_ready_cache(cache_root, provenance, patch_ids)
            else:
                self._write_cache(cache_root, reader, ordered_refs, source_channels, provenance)
            self.cache_build_seconds = time.perf_counter() - cache_started
        PatchCache(cache_root, provenance)
        self._install_cached_datasets(cache_root, provenance)

    def build_validation_only_cache(self, cache_root: Path, *, max_bytes: int) -> dict[str, object]:
        """Write a validation-only, full-schema-width job-local cache; return its provenance.

        Validation-only: the module must be set up with ``splits == ("validation",)``
        and identity channel indices over the whole source stack, so each consumer
        selects its own channels at collation time. The test split is never cached.
        Used by the v6 validation rescore only; training paths keep ``_build_cache``.
        """
        if self.reader is None or not self._setup_done:
            raise RuntimeError("setup() must run before build_validation_only_cache()")
        if self.holdout_anchors is not None:
            raise RuntimeError("validation-only cache is not defined for the spatial holdout")
        if self.splits != ("validation",):
            raise RuntimeError(
                f"validation-only cache requires splits ('validation',), got {self.splits!r}"
            )
        if self.cache_root is not None:
            raise RuntimeError("validation-only cache requires a module set up without a cache")
        source_width = len(self.reader.scaler.channel_order)
        if self.channel_indices != tuple(range(source_width)):
            raise RuntimeError(
                "validation-only cache requires identity channel indices over the "
                f"{source_width}-channel source stack, got {self.channel_indices!r}"
            )
        refs = self._admitted.get("validation", [])
        if not refs:
            raise RuntimeError("cannot build an empty validation-only cache")
        bad = sorted({ref.split for ref in refs if ref.split != "validation"})
        if bad:
            raise RuntimeError(f"validation-only cache refuses non-validation refs from {bad}")
        if cache_root.exists():
            raise RuntimeError(f"validation-only cache root {cache_root} already exists")
        provenance, source_channels = self._cache_provenance(refs)
        self.cache_max_bytes = max_bytes
        with timed_phase(
            "cache_build",
            logger=logger,
            cache_build_workers=self.cache_build_workers,
            patch_count=len(refs),
            cache_reused=False,
        ):
            started = time.perf_counter()
            self._write_cache(cache_root, self.reader, refs, source_channels, provenance)
            self.cache_build_seconds = time.perf_counter() - started
        self.patch_cache_source = "job_local"
        PatchCache(cache_root, provenance)
        return provenance

    def _materialize_prebuilt(self) -> None:
        """Fetch the versioned prebuilt cache and install it as the split datasets."""
        if self.cache_root is None or self.reader is None or self.prebuilt_uri_root is None:
            raise RuntimeError("prebuilt cache requires cache_root, uri root and a reader")
        cache_root = self.cache_root
        ordered_refs = self._cache_patch_order()
        patch_ids = [ref.patch_id for ref in ordered_refs]
        provenance, _ = self._cache_provenance(ordered_refs)
        components = self.prebuilt_components()
        prefix = prebuilt_prefix(self.prebuilt_uri_root, components)
        self.patch_cache_prefix = prefix
        self.patch_cache_key = prebuilt_key(components)
        if cache_root.exists():
            raise RuntimeError(
                f"cache root {cache_root} already exists; "
                "refusing to fetch a prebuilt cache into it"
            )
        try:
            fetched = fetch_prebuilt_cache(
                prefix,
                cache_root,
                expected_components=components,
                max_workers=self.prebuilt_download_workers,
                retries=self.prebuilt_download_retries,
            )
            self.prebuilt_download_seconds = cast(float, fetched["prebuilt_download_seconds"])
            self.prebuilt_verify_seconds = cast(float, fetched["prebuilt_verify_seconds"])
            self.prebuilt_bytes = cast(int, fetched["prebuilt_bytes"])
            self.prebuilt_shards = cast(int, fetched["prebuilt_shards"])
            self.prebuilt_build_provenance = cast(
                dict[str, object], fetched["prebuilt_build_provenance"]
            )
            self.prebuilt_commit_match = cast("bool | None", fetched["prebuilt_commit_match"])
            self.prebuilt_build_commit = cast("str | None", fetched["prebuilt_build_commit"])
            self.prebuilt_running_commit = cast("str | None", fetched["prebuilt_running_commit"])
            with timed_phase("prebuilt_load", logger=logger, patch_count=len(patch_ids)):
                load_started = time.perf_counter()
                self._verify_ready_cache(cache_root, provenance, patch_ids)
                self._install_cached_datasets(cache_root, provenance)
                self.prebuilt_load_seconds = time.perf_counter() - load_started
        except (PrebuiltCacheError, RuntimeError) as exc:
            if not self.allow_job_local_fallback:
                raise
            log_event(
                logger,
                logging.WARNING,
                "patch_cache_prebuilt_fallback",
                prefix=prefix,
                reason=str(exc),
                field=getattr(exc, "field", None),
            )
            if cache_root.exists():
                shutil.rmtree(cache_root)
            self._datasets.clear()
            self._build_cache()
            self.patch_cache_source = "job_local_fallback"
            return
        self.patch_cache_source = "prebuilt"

    def _verify_ready_cache(
        self, cache_root: Path, provenance: dict[str, object], patch_ids: list[str]
    ) -> None:
        cache = PatchCache(cache_root, provenance)
        if list(cache.patch_ids) != patch_ids:
            raise RuntimeError(f"ready cache {cache_root} holds a different ordered patch list")
        if cache.manifest.get("active_channels") != self.n_active_channels:
            raise RuntimeError(f"ready cache {cache_root} has a different active channel count")
        if cache.manifest.get("channel_indices") != list(self.channel_indices):
            raise RuntimeError(f"ready cache {cache_root} has different channel indices")

    def _write_cache(
        self,
        cache_root: Path,
        reader: RealPatchReader,
        ordered_refs: list[PatchRef],
        source_channels: int,
        provenance: dict[str, object],
    ) -> None:
        workers = self.cache_build_workers
        local = threading.local()

        def read_one(ref: PatchRef) -> RealSample:
            if workers == 1:
                thread_reader = reader
            else:
                thread_reader = getattr(local, "reader", None)
                if thread_reader is None:
                    thread_reader = reader.copy_for_reads()
                    local.reader = thread_reader
            return read_with_retry(lambda: thread_reader.read_patch(ref), patch_id=ref.patch_id)

        samples = ordered_bounded_map(read_one, ordered_refs, max_workers=workers)
        try:
            build_patch_cache(
                cache_root,
                samples,
                patch_ids=[ref.patch_id for ref in ordered_refs],
                active_channels=self.n_active_channels,
                source_channels=source_channels,
                provenance=provenance,
                max_bytes=self.cache_max_bytes,
                channel_indices=self.channel_indices,
            )
        finally:
            samples.close()

    def _admit(self, reader: RealPatchReader, split: str, refs: list[PatchRef]) -> list[PatchRef]:
        """Keep the readable refs in canonical order; record the skipped ones."""
        admitted: list[PatchRef] = []
        for ref in refs:
            reason = reader.admission_reason(ref)
            if reason is None and self.tessera_embeddings_root:
                from berlin_lst_downscaling.data.embeddings.loader import (
                    tessera_window_fully_valid,
                )

                if not tessera_window_fully_valid(
                    self.tessera_embeddings_root,
                    ref.year,
                    row_100m=ref.row,
                    col_100m=ref.col,
                ):
                    reason = "tessera_invalid"
            if reason is None:
                admitted.append(ref)
            else:
                self._skipped[split].append({"patch_id": ref.patch_id, "reason": reason})
        return admitted

    def stats(self) -> dict[str, object]:
        """Describe the real read: per-split patches, IDs, skips, and exclusions.

        Counts and IDs come from the admitted refs in canonical index order,
        never from loader iteration order, so a shuffled stream reports the
        same universe as the eager path. Retained as run evidence so a
        comparison can be audited: which patches entered each split, and which
        were dropped with which reason.
        """
        scope = self._base_stats()
        if self.holdout_anchors is not None:
            seen, unseen = self._validation_partition()
            scope["spatial_holdout"] = {
                "holdout_anchors": len(self.holdout_anchors),
                "train_removed": self._holdout_train_removed,
                "validation_seen": len(seen),
                "validation_held_out": len(unseen),
                "scaler_fit_seconds": self.holdout_scaler_fit_seconds,
                "scaler": None if self.holdout_scaler is None else asdict(self.holdout_scaler),
            }
            if self._feature_remap_evidence is not None:
                scope["spatial_holdout"]["feature_remap"] = self._feature_remap_evidence
        return scope

    def _base_stats(self) -> dict[str, object]:
        if self.reader is None:
            raise RuntimeError("setup() must run before stats()")
        return {
            "mode": self.mode,
            "source_setup_seconds": self.source_setup_seconds,
            "admission_seconds": self.admission_seconds,
            "cache_build_seconds": self.cache_build_seconds,
            "cache_build_workers": self.cache_build_workers,
            "cache_reused": self.cache_reused,
            "patch_cache_source": self.patch_cache_source,
            "patch_cache_prefix": self.patch_cache_prefix,
            "patch_cache_key": self.patch_cache_key,
            "tessera_embedding_source": (
                "quantized_patch_cache"
                if self.tessera_embeddings_root and self.patch_cache_source is not None
                else "window_read"
                if self.tessera_embeddings_root
                else None
            ),
            "prebuilt_download_seconds": self.prebuilt_download_seconds,
            "prebuilt_verify_seconds": self.prebuilt_verify_seconds,
            "prebuilt_load_seconds": self.prebuilt_load_seconds,
            "prebuilt_bytes": self.prebuilt_bytes,
            "prebuilt_shards": self.prebuilt_shards,
            "prebuilt_build_provenance": self.prebuilt_build_provenance,
            "prebuilt_commit_match": self.prebuilt_commit_match,
            "prebuilt_build_commit": self.prebuilt_build_commit,
            "prebuilt_running_commit": self.prebuilt_running_commit,
            "batches_per_split": {
                split: _batch_count(len(refs), self.batch_size)
                for split, refs in self._admitted.items()
            },
            "patches_per_split": {split: len(refs) for split, refs in self._admitted.items()},
            "requested_per_split": {split: len(refs) for split, refs in self._requested.items()},
            # Only the bounded probe retains the requested IDs; a full run would
            # duplicate the admitted list and bloat its data_scope.json.
            "requested_patch_ids": (
                {split: [ref.patch_id for ref in refs] for split, refs in self._requested.items()}
                if self.probe_scope is not None
                else None
            ),
            "scenes_per_split": {
                split: sorted({ref.scene_id for ref in refs})
                for split, refs in self._admitted.items()
            },
            "years_per_split": {
                split: sorted({ref.year for ref in refs}) for split, refs in self._admitted.items()
            },
            "partial_mask_patches_per_split": {
                split: sum(0 < ref.n_eligible < REAL_PATCH_CELLS * REAL_PATCH_CELLS for ref in refs)
                for split, refs in self._admitted.items()
            },
            "patch_ids": {
                split: [ref.patch_id for ref in refs] for split, refs in self._admitted.items()
            },
            "skipped_refs": self._skipped,
            "skipped_per_split": {split: len(refs) for split, refs in self._skipped.items()},
            "exclusions": dict(self.reader.exclusions),
            "train_shuffle_order": self._train_shuffle_order(),
        }

    def _train_sampler(self, dataset: RealPatchDataset | CachedRealPatchDataset) -> RandomSampler:
        """A dedicated seeded sampler, so the train order is reproducible.

        A dedicated ``RandomSampler`` is used instead of ``shuffle=True`` so the
        order `train_shuffle_order` records is exactly the order the train
        loader yields: a ``shuffle=True`` loader draws its worker base seed from
        the generator *before* the sampler permutes, so a plain ``RandomSampler``
        reconstruction would not match. This assumes a single-device run; a
        multi-device lifecycle would substitute a ``DistributedSampler``.
        """
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(self.seed))

    def _train_shuffle_order(self) -> list[int] | None:
        """The seeded epoch-1 train index order, or ``None`` when unshuffled.

        Produced by the same sampler the train loader uses, so the recorded
        evidence is the order training actually sees.

        # decision: record the shuffled order as run evidence, because the plan
        # requires a verifiable same-seed replay and a shuffled stream is
        # otherwise unobservable from canonical-order stats. Alternative:
        # assert it only inside the smoke step (rejected — not retained).
        """
        if self.mode != "stream" or not self.shuffle_train:
            return None
        dataset = self._datasets.get("train")
        if not isinstance(dataset, (RealPatchDataset, CachedRealPatchDataset)) or len(dataset) == 0:
            return None
        return list(self._train_sampler(dataset))

    def _read_batches(self, reader: RealPatchReader, refs: list[PatchRef]) -> list[RealBatch]:
        """Read admitted patches and collate them in canonical order."""
        batches: list[RealBatch] = []
        chunk: list[RealSample] = []
        for ref in refs:
            sample = reader.read_patch(ref)
            if sample is None:
                raise RuntimeError(
                    f"admitted patch {ref.patch_id} became unreadable during the eager read"
                )
            chunk.append(sample)
            if len(chunk) == self.batch_size:
                batches.append(
                    collate_real_batch(
                        chunk,
                        n_active_channels=self.n_active_channels,
                        channel_indices=self.channel_indices,
                        embeddings_root=self.tessera_embeddings_root,
                    )
                )
                chunk = []
        if chunk:
            batches.append(
                collate_real_batch(
                    chunk,
                    n_active_channels=self.n_active_channels,
                    channel_indices=self.channel_indices,
                    embeddings_root=self.tessera_embeddings_root,
                )
            )
        return batches

    def _loader(self, split: str) -> DataLoader[RealBatch]:
        dataset = self._datasets.get(split)
        if dataset is None or len(dataset) == 0:
            raise RuntimeError(f"real path has no {split} batch to train on")
        if self.cache_root is not None and not isinstance(dataset, CachedRealPatchDataset):
            raise RuntimeError("job-local cache setup failed; refusing a silent streaming fallback")
        if isinstance(dataset, _BatchDataset):
            return DataLoader(dataset, batch_size=None, shuffle=False)
        shuffle = self.shuffle_train and split == "train"
        # Cached tensors are already channel-selected; re-applying V3 indices
        # would be wrong for non-contiguous ablation subsets. Live readers still
        # yield the release's full stack (28 for V3, 22 for v6) and need the
        # configured indices.
        if isinstance(dataset, CachedRealPatchDataset):
            collate_fn = partial(
                collate_real_batch,
                n_active_channels=self.n_active_channels,
                embeddings_root=self.tessera_embeddings_root,
                require_cached_tessera=bool(self.tessera_embeddings_root),
            )
        else:
            collate_fn = partial(
                collate_real_batch,
                n_active_channels=self.n_active_channels,
                channel_indices=self.channel_indices,
                embeddings_root=self.tessera_embeddings_root,
            )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler(dataset) if shuffle else None,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed) if shuffle else None,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )
        # The dataset yields RealSample; ``collate_real_batch`` stacks each
        # batch into RealBatch, so the loader's yielded type is RealBatch.
        return cast(DataLoader[RealBatch], loader)

    def train_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("validation")

    def holdout_val_dataloader(self) -> DataLoader[RealBatch]:
        """Validation patches on the held-out anchors (spatial holdout only)."""
        if self.holdout_anchors is None:
            raise RuntimeError("holdout_val_dataloader requires a spatial holdout")
        return self._loader(_HOLDOUT_VALIDATION)

    def test_dataloader(self) -> DataLoader[RealBatch]:
        return self._loader("test")


class RealLSTTask(LightningModule):
    """10 m LST downscaling task over the contract-conforming batch.

    Parameters
    ----------
    n_active_channels:
        Number of active feature channels (first-C of the V3 order). The U-Net
        receives one extra input channel for ``lst_prior``.
    base_width / depth:
        U-Net capacity. ``depth`` must satisfy ``160 % 2 ** depth == 0``; the
        contract's patch size requires ``depth <= 4``.
    learning_rate / weight_decay:
        AdamW optimizer settings.
    residual_prior:
        Recovery representation (issue #47). When True the U-Net predicts a
        Kelvin *correction* added to the reconstructed prior, and the final head
        is zero-initialized so the model starts exactly at the prior-expand arm.
        The output, pooling, loss, and target are unchanged.
    loss_name:
        Training objective. ``masked_l1`` is Stages 1–4; ``thermal_aware`` is
        Stage 5 (same inputs, structural + gradient penalties). Checkpoint
        selection stays masked MAE in both cases.
    ssim_data_range_k:
        Fixed SSIM data range L. ``None`` resolves the committed train-derived
        artifact (raises while it is a placeholder); the resolved value is
        saved in the hyperparameters.
    ssim_data_range:
        Provenance dict of L (``SsimDataRange.payload()``), saved in the
        checkpoint. On reload from an artifact-sourced checkpoint the current
        artifact sha256 must match, otherwise ``RuntimeError``.
    """

    def __init__(
        self,
        n_active_channels: int = N_FEATURE_CHANNELS,
        base_width: int = 32,
        depth: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        record_probe_metrics: bool = False,
        residual_prior: bool = False,
        loss_name: str = "masked_l1",
        aggregation: AggregationMethod | str = AGGREGATION_ARITHMETIC,
        ssim_data_range_k: float | None = None,
        ssim_data_range: dict[str, Any] | None = None,
        lr_schedule: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.lr_schedule = lr_schedule
        # Reassigned before save_hyperparameters so the resolved L and its
        # provenance are saved; a reload re-checks the artifact hash.
        ssim_data_range_k, ssim_data_range = resolve_task_ssim_data_range(
            ssim_data_range_k, ssim_data_range
        )
        self.ssim_data_range_k = ssim_data_range_k
        self.ssim_data_range = ssim_data_range
        if loss_name not in ("masked_l1", "thermal_aware"):
            raise ValueError(f"loss_name must be 'masked_l1' or 'thermal_aware', got {loss_name!r}")
        self.model = UNet(in_channels=n_active_channels + 1, base_width=base_width, depth=depth)
        self.n_active_channels = n_active_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.record_probe_metrics = record_probe_metrics
        self.residual_prior = residual_prior
        self.loss_name = loss_name
        self.aggregation: AggregationMethod = resolve_aggregation_method(aggregation)
        if residual_prior:
            # Zero head only in the recovery mode: the initial output is then
            # exactly the reconstructed prior, which the probe guard verifies.
            for param in self.model.head.parameters():
                param.detach().zero_()
        self.train_mae = MaskedMAE()
        self.val_mae = MaskedMAE()
        self.val_ssim = MaskedSSIM()
        self.val_ssim_windows = SupportedWindows()
        # Validation-only appendix variants; never in the loss or checkpoint selection.
        self.val_ssim_L250 = MaskedSSIM()
        self.val_ssim_Lpatch = MaskedSSIM()
        self.val_ssim_Lpatch_zero_range = SupportCounter()
        self.val_valid_cells = ValidCells()
        self.train_thermal_l1 = SupportWeightedSum()
        self.train_thermal_ssim = SupportWeightedSum()
        self.train_thermal_grad = SupportWeightedSum()
        self.train_thermal_l1_weighted = ScaledSupportWeightedSum(1.0)
        self.train_thermal_ssim_weighted = ScaledSupportWeightedSum(THERMAL_AWARE_SSIM_WEIGHT)
        self.train_thermal_grad_weighted = ScaledSupportWeightedSum(THERMAL_AWARE_GRAD_WEIGHT)
        self.val_thermal_l1 = SupportWeightedSum()
        self.val_thermal_ssim = SupportWeightedSum()
        self.val_thermal_grad = SupportWeightedSum()
        self.val_thermal_l1_weighted = ScaledSupportWeightedSum(1.0)
        self.val_thermal_ssim_weighted = ScaledSupportWeightedSum(THERMAL_AWARE_SSIM_WEIGHT)
        self.val_thermal_grad_weighted = ScaledSupportWeightedSum(THERMAL_AWARE_GRAD_WEIGHT)
        self.val_ssim_candidate = SupportCounter()
        self.val_grad_candidate = SupportCounter()
        self.val_ssim_coverage = SupportRatio()
        self.val_grad_coverage = SupportRatio()
        self.val_pred_outside_lst_range = SupportRatio()
        # Reconstructible model/loss configuration for load_from_checkpoint.
        self.save_hyperparameters()

    def _training_loss(
        self, prediction_100m: Tensor, target_100m: Tensor, mask_100m: Tensor
    ) -> Tensor:
        if self.loss_name == "thermal_aware":
            return thermal_objective_batch(
                prediction_100m, target_100m, mask_100m, data_range_k=self.ssim_data_range_k
            ).loss
        return masked_l1_loss(prediction_100m, target_100m, mask_100m)

    def _accumulate_thermal_audit(self, batch_stats: ThermalObjectiveBatch, *, train: bool) -> None:
        if self.loss_name != "thermal_aware":
            return
        l1_metric = self.train_thermal_l1 if train else self.val_thermal_l1
        ssim_metric = self.train_thermal_ssim if train else self.val_thermal_ssim
        grad_metric = self.train_thermal_grad if train else self.val_thermal_grad
        l1_w = self.train_thermal_l1_weighted if train else self.val_thermal_l1_weighted
        ssim_w = self.train_thermal_ssim_weighted if train else self.val_thermal_ssim_weighted
        grad_w = self.train_thermal_grad_weighted if train else self.val_thermal_grad_weighted
        l1_metric.update(batch_stats.l1_sum, batch_stats.l1_count)
        l1_w.update(batch_stats.l1_sum, batch_stats.l1_count)
        # A batch whose every patch lacks a valid SSIM window is omitted from
        # the support-weighted mean. Its zero penalty is not inserted as a score.
        if float(batch_stats.ssim_supported) > 0.0:
            penalty_sum = batch_stats.ssim_supported - batch_stats.ssim_sum
            ssim_metric.update(penalty_sum, batch_stats.ssim_supported)
            ssim_w.update(penalty_sum, batch_stats.ssim_supported)
        if train:
            self.log(
                "train/ssim_supported",
                batch_stats.ssim_supported,
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
            )
        if float(batch_stats.grad_supported) > 0.0:
            grad_metric.update(batch_stats.grad_sum, batch_stats.grad_supported)
            grad_w.update(batch_stats.grad_sum, batch_stats.grad_supported)
        if not train:
            self.val_ssim_candidate.update(batch_stats.ssim_candidate)
            self.val_grad_candidate.update(batch_stats.grad_candidate)
            self.val_ssim_coverage.update(batch_stats.ssim_supported, batch_stats.ssim_candidate)
            self.val_grad_coverage.update(batch_stats.grad_supported, batch_stats.grad_candidate)

    def _log_thermal_batch_metrics(self, *, train: bool) -> None:
        """Register thermal audit metrics for Lightning epoch reduction (auto-reset)."""
        if self.loss_name != "thermal_aware":
            return
        prefix = "train" if train else "validation"
        if train:
            l1_u, l1_w = self.train_thermal_l1, self.train_thermal_l1_weighted
            ssim_u, ssim_w = self.train_thermal_ssim, self.train_thermal_ssim_weighted
            grad_u, grad_w = self.train_thermal_grad, self.train_thermal_grad_weighted
        else:
            l1_u, l1_w = self.val_thermal_l1, self.val_thermal_l1_weighted
            ssim_u, ssim_w = self.val_thermal_ssim, self.val_thermal_ssim_weighted
            grad_u, grad_w = self.val_thermal_grad, self.val_thermal_grad_weighted
        pairs = (
            (f"{prefix}/thermal_l1_unweighted", l1_u),
            (f"{prefix}/thermal_l1_weighted", l1_w),
            (f"{prefix}/thermal_ssim_unweighted", ssim_u),
            (f"{prefix}/thermal_ssim_weighted", ssim_w),
            (f"{prefix}/thermal_grad_unweighted", grad_u),
            (f"{prefix}/thermal_grad_weighted", grad_w),
        )
        for key, metric in pairs:
            self.log(key, metric, on_step=False, on_epoch=True)
        if not train:
            self.log("validation/ssim_coverage", self.val_ssim_coverage, on_epoch=True)
            self.log("validation/grad_coverage", self.val_grad_coverage, on_epoch=True)

    # ── lifecycle ─────────────────────────────────────────────────────

    def forward(self, batch: RealBatch) -> Tensor:
        """Predict the 10 m LST map from features concatenated with the prior."""
        validate_real_batch(batch, n_active_channels=self.n_active_channels)
        inputs = torch.cat([batch.features, batch.lst_prior], dim=1)
        delta = self.model(inputs)
        if not self.residual_prior:
            return delta
        return reconstruct_prior_kelvin(batch.lst_prior).float() + delta.float()

    def training_step(self, batch: RealBatch, batch_idx: int) -> Tensor:
        prediction_100m = pool_10m_to_100m(self(batch), self.aggregation, batch.mask_100m)
        if self.loss_name == "thermal_aware":
            objective = thermal_objective_batch(
                prediction_100m,
                batch.target_100m,
                batch.mask_100m,
                data_range_k=self.ssim_data_range_k,
            )
            loss = objective.loss
            self._accumulate_thermal_audit(objective, train=True)
            self._log_thermal_batch_metrics(train=True)
        else:
            loss = self._training_loss(prediction_100m, batch.target_100m, batch.mask_100m)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_mae.update(prediction_100m, batch.target_100m, batch.mask_100m)
        self.log("train/mae_100m", self.train_mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: RealBatch, batch_idx: int) -> None:
        # The whole validation computation stays in FP32: Lightning's
        # mixed-precision context also covers validation, so only disabling
        # autocast around the forward left SSIM's convolution in float16,
        # where Kelvin-scale moments overflow (J4 NaN SSIM).
        with torch.autocast(device_type=batch.features.device.type, enabled=False):
            prediction = self(batch)
            prediction_100m = pool_10m_to_100m(
                prediction.float(), self.aggregation, batch.mask_100m
            )
            self.val_mae.update(prediction_100m, batch.target_100m, batch.mask_100m)
            if self.loss_name == "thermal_aware":
                objective = thermal_objective_batch(
                    prediction_100m,
                    batch.target_100m,
                    batch.mask_100m,
                    data_range_k=self.ssim_data_range_k,
                )
                self._accumulate_thermal_audit(objective, train=False)
                self._log_thermal_batch_metrics(train=False)
            # Diagnostic SSIM sees the unclamped pooled prediction. The training
            # objective above still clamps inside thermal_objective_batch.
            ssim_sum, ssim_count = masked_ssim_stats(
                prediction_100m,
                batch.target_100m,
                batch.mask_100m,
                clamp_predictions=False,
                data_range_k=self.ssim_data_range_k,
            )
            self.val_ssim.update(ssim_sum, ssim_count)
            self.val_ssim_windows.update(ssim_count)
            l250_sum, l250_count = masked_ssim_stats(
                prediction_100m,
                batch.target_100m,
                batch.mask_100m,
                clamp_predictions=False,
                data_range_k=SSIM_LEGACY_DATA_RANGE_K,
            )
            self.val_ssim_L250.update(l250_sum, l250_count)
            lp_sum, lp_count, lp_zero = masked_ssim_stats_patch_range(
                prediction_100m, batch.target_100m, batch.mask_100m
            )
            self.val_ssim_Lpatch.update(lp_sum, lp_count)
            self.val_ssim_Lpatch_zero_range.update(lp_zero)
            outside_count, pixel_count = prediction_outside_lst_range_counts(prediction_100m)
            self.val_pred_outside_lst_range.update(outside_count, pixel_count)
            self.val_valid_cells.update(batch.mask_100m)
        # Masked MAE is the selection metric; SSIM is logged only.
        self.log("validation/mae_100m", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/ssim_100m", self.val_ssim, on_step=False, on_epoch=True)
        self.log("validation/ssim_windows", self.val_ssim_windows, on_step=False, on_epoch=True)
        l250_key, lpatch_key, lpatch_zero_key = METRIC_ONLY_SSIM_KEYS
        self.log(l250_key, self.val_ssim_L250, on_step=False, on_epoch=True)
        self.log(lpatch_key, self.val_ssim_Lpatch, on_step=False, on_epoch=True)
        self.log(lpatch_zero_key, self.val_ssim_Lpatch_zero_range, on_step=False, on_epoch=True)
        self.log(
            "validation/ssim_supported",
            ssim_count,
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
        )
        self.log(
            "validation/pred_outside_lst_range_frac",
            self.val_pred_outside_lst_range,
            on_step=False,
            on_epoch=True,
        )
        if self.record_probe_metrics:
            self.log("validation/valid_cells", self.val_valid_cells, on_step=False, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        log_current_lr(self)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer_config(optimizer, self.lr_schedule)


__all__ = ["RealLSTTask", "RealPatchDataModule", "reconstruct_prior_kelvin"]
