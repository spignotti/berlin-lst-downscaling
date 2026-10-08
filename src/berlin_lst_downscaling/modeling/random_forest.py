"""Random-forest residual baseline for the Stage-1 comparison (issue #59).

Trains one forest per Tag-11 feature set (stages 1-4) on the published WB3
patch universe and scores every validation/test patch through the same reader,
mask, pooling, and metrics the U-Net and the naive prior-expand baseline use.

Method
------
- One training row is one eligible 100 m cell: the exact 10x10 block means of
  the scaled, already-neutralized 10 m features, the physical coarse LST prior
  at 100 m, and the target residual ``Landsat_100m - prior_100m`` in Kelvin.
  No independent 10 m target is assumed, so the forest is supervised only at
  the native 100 m cell grid.
- Inference reconstructs the 10 m field as ``prior_10m + residual`` (the
  residual is predicted per 10 m pixel from native features plus prior),
  followed by shared exact 10x10 pooling and masked MAE/SSIM at 100 m.
  Transfer from pooled training features to native inference features is
  an explicit RF assumption.
- Training rows are a seeded reservoir (at most ``max_cells``) over the
  index-ordered stream of eligible train cells, one row per
  ``(scene_id, year, global_row, global_col)``. All stages fit on the *same*
  rows and differ only in their channel subset.
- Eval features are read once per patch and cached on run-local disk, so the four
  stages never repeat the GCS reads.

The module never writes to a canonical release; every artifact goes under the
run output root.
"""

from __future__ import annotations

import json
import logging
import platform
import resource
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import skops.io as sio
import torch
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor

from berlin_lst_downscaling.data.io import log_event, run_context_path
from berlin_lst_downscaling.data.training.report import now_iso
from berlin_lst_downscaling.modeling.baseline import PRIOR_RULE, PatchRecord, SplitSummary
from berlin_lst_downscaling.modeling.channels import (
    ABLATION_STAGE_CHANNELS,
    feature_order_for,
    resolve_active_channels,
)
from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    POOL_FACTOR,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
)
from berlin_lst_downscaling.modeling.metrics import (
    masked_abs_error_sums,
    masked_ssim_stats,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSourceConfig,
    patch_index_fingerprints,
)

_logger = logging.getLogger(__name__)

METHOD = "random_forest_residual"
TARGET_RULE = (
    "Landsat_100m - prior_100m (K): one row per eligible 100 m cell, target and "
    "prior on the native 100 m cell grid"
)
INFERENCE_RULE = (
    "physical 10 m prior + per-pixel residual from native 10 m features and prior, scored "
    "by the shared exact nested 10x10 pooling at 100 m"
)
RESERVOIR_RULE = (
    "Algorithm R over the index-ordered train-cell stream, seeded; at most "
    "max_cells rows, one row per distinct (scene_id, year, global_row, global_col)"
)
WEIGHTING_RULE = (
    "one row per distinct (scene_id, year, global_row, global_col); a cell "
    "repeated by an overlapping window is skipped, not upweighted"
)
PRIMARY_METRIC = "masked MAE @ 100 m (cell-weighted over valid cells)"
SECONDARY_METRIC = (
    "SSIM @ 100 m, 7x7 uniform window, fully-valid windows only, "
    "logged only (never a selection gate)"
)
PIPELINE = "random_forest"
TRAIN_ROWS_FILE = "random_forest_train_rows.parquet"
PROBE_FILE = "random_forest_probe.npz"
CONFIG_FILE = "random_forest_config.yaml"


def model_file_name(stage: int) -> str:
    """Return the artifact file name of one stage's fitted forest."""
    return f"random_forest_stage{stage}.skops"


# ── configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ForestSpec:
    """The frozen forest hyperparameters, identical across all feature sets.

    ``n_estimators``, ``max_depth``, and ``n_jobs`` are the only resource
    knobs a smoke profile may shrink; the criterion, bootstrap, and
    ``max_features`` stay the frozen protocol values.
    """

    n_estimators: int = 200
    max_depth: int = 20
    min_samples_leaf: int = 5
    max_features: float = 1.0
    bootstrap: bool = True
    criterion: str = "squared_error"
    random_state: int = 0
    n_jobs: int = 2

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ForestSpec:
        """Build and validate the spec from the resolved Hydra config."""
        raw = cfg.get("forest")
        forest = {} if raw is None else {str(k): v for k, v in dict(raw).items()}
        spec = cls(
            n_estimators=int(forest.get("n_estimators", 200)),
            max_depth=int(forest.get("max_depth", 20)),
            min_samples_leaf=int(forest.get("min_samples_leaf", 5)),
            max_features=float(forest.get("max_features", 1.0)),
            bootstrap=bool(forest.get("bootstrap", True)),
            criterion=str(forest.get("criterion", "squared_error")),
            random_state=int(forest.get("random_state", cfg.get("seed", 0))),
            n_jobs=int(forest.get("n_jobs", 2)),
        )
        if spec.n_estimators < 1:
            raise ValueError(f"forest.n_estimators must be >= 1, got {spec.n_estimators}")
        if spec.max_depth < 1:
            raise ValueError(f"forest.max_depth must be >= 1, got {spec.max_depth}")
        if spec.min_samples_leaf < 1:
            raise ValueError(f"forest.min_samples_leaf must be >= 1, got {spec.min_samples_leaf}")
        if spec.criterion != "squared_error":
            raise ValueError(f"the frozen criterion is 'squared_error', got {spec.criterion!r}")
        if spec.bootstrap is not True:
            raise ValueError("the frozen protocol requires bootstrap=True")
        if spec.max_features != 1.0:
            raise ValueError(
                f"the frozen protocol requires max_features=1.0, got {spec.max_features}"
            )
        if spec.n_jobs not in (1, 2):
            raise ValueError("forest.n_jobs must be 1 or 2")
        if spec.min_samples_leaf != 5 or spec.random_state != 0:
            raise ValueError("the RF method requires min_samples_leaf=5 and random_state=0")
        if cfg.get("profile") == "full" and (spec.n_estimators, spec.max_depth) != (200, 20):
            raise ValueError("full RF requires 200 trees and max_depth=20")
        return spec

    def build(self) -> RandomForestRegressor:
        """Return an unfitted forest with exactly these hyperparameters."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "criterion": self.criterion,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class StageSelection:
    """One Tag-11 feature set resolved to V3 channel indices."""

    stage: int
    names: tuple[str, ...]
    indices: tuple[int, ...]
    feature_order: str

    @property
    def n_active_channels(self) -> int:
        return len(self.names)


def resolve_stage_selection(stage: int) -> StageSelection:
    """Resolve a stage number through the published ablation channel ladder."""
    channels = ABLATION_STAGE_CHANNELS.get(stage)
    if channels is None:
        raise ValueError(f"stage must be one of {sorted(ABLATION_STAGE_CHANNELS)}, got {stage!r}")
    selection = resolve_active_channels(
        n_active_channels=len(channels), active_channel_names=channels
    )
    return StageSelection(
        stage=stage,
        names=selection.names,
        indices=selection.indices,
        feature_order=feature_order_for(selection),
    )


def library_versions() -> dict[str, str]:
    """Return the interpreter and numerical-stack versions recorded per run."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "scikit-learn": package_version("scikit-learn"),
        "skops": package_version("skops"),
        "torch": torch.__version__,
    }


# ── exact 10 m -> 100 m pooling and inference reconstruction ─────────


def pool_10m_cells_to_100m(values_10m: np.ndarray) -> np.ndarray:
    """Return the exact nested 10x10 block mean of ``(..., 160, 160)`` values.

    Accumulates in float64 and casts once to float32, so the training feature
    matrix and an independent recomputation of the same 10 m pixels land on
    the same float32 values (the shared torch pooling remains the scoring
    path; this is the training-side marshal).
    """
    if values_10m.shape[-2] % POOL_FACTOR or values_10m.shape[-1] % POOL_FACTOR:
        raise ValueError(f"extent {values_10m.shape[-2:]} is not divisible by {POOL_FACTOR}")
    lead = values_10m.shape[:-2]
    h, w = values_10m.shape[-2:]
    reshaped = values_10m.astype(np.float64).reshape(
        *lead, h // POOL_FACTOR, POOL_FACTOR, w // POOL_FACTOR, POOL_FACTOR
    )
    return reshaped.mean(axis=(-3, -1)).astype(np.float32)


def reconstruct_10m_prediction(prior_100m: np.ndarray, residual_10m: np.ndarray) -> np.ndarray:
    """Add a native 10 m residual field to the physical coarse prior."""
    if prior_100m.shape != (REAL_PATCH_CELLS, REAL_PATCH_CELLS):
        raise ValueError("invalid 100 m prior shape")
    if residual_10m.shape != (REAL_PATCH_PX, REAL_PATCH_PX):
        raise ValueError("invalid 10 m residual shape")
    prior_10m = np.repeat(np.repeat(prior_100m, POOL_FACTOR, axis=0), POOL_FACTOR, axis=1)
    return (prior_10m + residual_10m)[None, None]


def stage_matrix(features: np.ndarray, prior: np.ndarray, stage: StageSelection) -> np.ndarray:
    """Select active channel columns and append the physical Kelvin prior."""
    if features.ndim != 2 or features.shape[1] != N_FEATURE_CHANNELS:
        raise ValueError("invalid training feature matrix")
    if prior.shape != (len(features),):
        raise ValueError("prior row count differs from features")
    return np.ascontiguousarray(
        np.column_stack((features[:, np.asarray(stage.indices)], prior)), dtype=np.float32
    )


def stage_inputs(features: np.ndarray, stage: StageSelection) -> np.ndarray:
    """Return the stage channel subset of an ``(..., 28, H, W)`` feature array.

    Channel selection happens after scaling, on model input only, exactly as
    the ablation configs select their channels.
    """
    if features.ndim < 3 or features.shape[-3] != N_FEATURE_CHANNELS:
        raise ValueError(f"expected {N_FEATURE_CHANNELS} feature channels, got {features.shape}")
    indices = np.asarray(stage.indices, dtype=np.intp)
    return np.ascontiguousarray(features[..., indices, :, :], dtype=np.float32)


def predict_10m(
    model: RandomForestRegressor,
    features: np.ndarray,
    prior_100m: np.ndarray,
    stage: StageSelection,
) -> np.ndarray:
    """Predict each 10 m pixel from its active channels and physical prior."""
    if features.shape != (N_FEATURE_CHANNELS, REAL_PATCH_PX, REAL_PATCH_PX):
        raise ValueError("invalid native feature shape")
    prior = np.repeat(np.repeat(prior_100m, POOL_FACTOR, axis=0), POOL_FACTOR, axis=1)
    matrix = stage_matrix(features.reshape(N_FEATURE_CHANNELS, -1).T, prior.ravel(), stage)
    residual = np.asarray(model.predict(matrix), dtype=np.float32)
    if not np.isfinite(residual).all():
        raise ValueError("forest returned non-finite residuals")
    return residual.reshape(REAL_PATCH_PX, REAL_PATCH_PX)


def _patch_cells(sample: RealSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pooled 100 m features, the physical 100 m prior, and the residual."""
    pooled = pool_10m_cells_to_100m(sample.features)
    prior_cells = pool_10m_cells_to_100m(sample.lst_prior_k)[0]
    residual_cells = sample.target_100m[0] - prior_cells
    return pooled, prior_cells, residual_cells


# ── training pool ─────────────────────────────────────────────────────


@dataclass
class TrainPool:
    """The selected training rows, one per eligible train cell observation."""

    keys: list[tuple[str, int, int, int]]
    features: np.ndarray  # (n, 28) float32, pooled 100 m channels
    prior_100m: np.ndarray  # (n,) float32
    residual_k: np.ndarray  # (n,) float32
    candidate_cells: int
    duplicate_cells: int
    requested_patches: int
    read_patches: int
    exclusions: dict[str, int]
    anchor_stride_cells: int
    duplicate_anchors: int
    overlapping_windows: int
    seconds: float

    @property
    def selected_cells(self) -> int:
        return len(self.keys)


def _audit_train_anchors(refs: Sequence[PatchRef]) -> dict[str, int]:
    """Verify the published anchor lattice and count overlapping train windows.

    The index anchors every complete 16x16-cell window on a multiple of the
    16-cell stride, so two *distinct* anchors of one scene can never overlap;
    a duplicate anchor is the only overlap signal. A non-aligned anchor means
    the published lattice has drifted and raises.
    """
    unaligned = 0
    seen: set[tuple[str, int, int]] = set()
    duplicate = 0
    for ref in refs:
        if ref.row % REAL_PATCH_CELLS or ref.col % REAL_PATCH_CELLS:
            unaligned += 1
            continue
        key = (ref.scene_id, ref.row, ref.col)
        if key in seen:
            duplicate += 1
        seen.add(key)
    if unaligned:
        raise RuntimeError(
            f"{unaligned} train anchors are not multiples of the {REAL_PATCH_CELLS}-cell "
            "stride: the published patch-index lattice has drifted"
        )
    return {
        "anchor_stride_cells": REAL_PATCH_CELLS,
        "duplicate_anchors": duplicate,
        "overlapping_windows": duplicate,
    }


def collect_train_pool(
    reader: RealPatchReader,
    refs: Sequence[PatchRef],
    *,
    max_cells: int,
    seed: int,
) -> TrainPool:
    """Select the seeded reservoir of training rows over the train patches.

    One pass over the refs: each admitted patch is pooled to 100 m cells and
    its eligible cells enter the reservoir in row-major order. In-place
    replacement keeps the selected feature rows without a second pass.
    """
    if max_cells < 1:
        raise ValueError(f"max_cells must be >= 1, got {max_cells}")
    audit = _audit_train_anchors(refs)
    # Distinct 16-aligned anchors are disjoint windows; the explicit key set is
    # the fallback for a drifted index that admits overlapping windows.
    dedupe: set[tuple[str, int, int]] | None = set() if audit["overlapping_windows"] else None
    rng = np.random.default_rng(seed)
    features = np.empty((max_cells, N_FEATURE_CHANNELS), dtype=np.float32)
    prior_100m = np.empty(max_cells, dtype=np.float32)
    residual_k = np.empty(max_cells, dtype=np.float32)
    slots: list[tuple[str, int, int, int] | None] = [None] * max_cells
    candidates = 0
    duplicates = 0
    distinct = 0
    read_patches = 0
    before = Counter(reader.exclusions)
    started = time.monotonic()
    for position, ref in enumerate(refs):
        if position % 100 == 0:
            log_event(
                _logger,
                logging.INFO,
                "read_progress",
                split=ref.split,
                completed=position,
                total=len(refs),
            )
        sample = reader.read_patch(ref)
        if sample is None:
            continue
        read_patches += 1
        pooled, prior_cells, residual_cells = _patch_cells(sample)
        drs, dcs = np.nonzero(sample.mask_100m[0])
        for dr, dc in zip(drs.tolist(), dcs.tolist(), strict=True):
            candidates += 1
            global_row = ref.row + dr
            global_col = ref.col + dc
            if dedupe is not None:
                cell_key = (ref.scene_id, global_row, global_col)
                if cell_key in dedupe:
                    duplicates += 1
                    continue
                dedupe.add(cell_key)
            distinct += 1
            if distinct <= max_cells:
                slot = distinct - 1
            else:
                draw = int(rng.integers(1, distinct + 1))
                if draw > max_cells:
                    continue
                slot = draw - 1
            slots[slot] = (ref.scene_id, ref.year, global_row, global_col)
            features[slot] = pooled[:, dr, dc]
            prior_100m[slot] = prior_cells[dr, dc]
            residual_k[slot] = residual_cells[dr, dc]
    selected = min(distinct, max_cells)
    keys: list[tuple[str, int, int, int]] = []
    for slot in slots[:selected]:
        if slot is None:
            raise RuntimeError("reservoir slot accounting drifted")
        keys.append(slot)
    return TrainPool(
        keys=keys,
        features=features[:selected],
        prior_100m=prior_100m[:selected],
        residual_k=residual_k[:selected],
        candidate_cells=candidates,
        duplicate_cells=duplicates,
        requested_patches=len(refs),
        read_patches=read_patches,
        exclusions={k: v for k, v in (Counter(reader.exclusions) - before).items()},
        anchor_stride_cells=audit["anchor_stride_cells"],
        duplicate_anchors=audit["duplicate_anchors"],
        overlapping_windows=audit["overlapping_windows"],
        seconds=time.monotonic() - started,
    )


def train_rows_digest(keys: Sequence[tuple[str, int, int, int]]) -> str:
    """Return the selection fingerprint over the sorted canonical key lines."""
    ordered = sorted((str(scene), int(year), int(row), int(col)) for scene, year, row, col in keys)
    payload = "".join(f"{scene}\t{year}\t{row}\t{col}\n" for scene, year, row, col in ordered)
    return sha256(payload.encode("utf-8")).hexdigest()


def write_train_rows(keys: Sequence[tuple[str, int, int, int]], uri: str) -> str:
    """Write the selected row identifiers (sorted) and return the file SHA-256."""
    ordered = sorted((str(scene), int(year), int(row), int(col)) for scene, year, row, col in keys)
    table = pa.table(
        {
            "scene_id": pa.array([row[0] for row in ordered], type=pa.string()),
            "year": pa.array([row[1] for row in ordered], type=pa.int16()),
            "row": pa.array([row[2] for row in ordered], type=pa.int32()),
            "col": pa.array([row[3] for row in ordered], type=pa.int32()),
        }
    )
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return sha256(path.read_bytes()).hexdigest()


# ── evaluation cache and scoring ──────────────────────────────────────


@dataclass
class EvalSplit:
    """One evaluated split with its features read exactly once."""

    split: str
    requested_patches: int
    patch_ids: list[str]
    feature_files: list[str]  # one run-local native feature .npy per patch
    prior_100m: np.ndarray  # (P, 16, 16) float32
    target_100m: np.ndarray  # (P, 16, 16) float32
    mask_100m: np.ndarray  # (P, 16, 16) bool
    filled_feature_pixels: list[int]
    exclusions: dict[str, int]
    seconds: float


def collect_eval_split(
    reader: RealPatchReader, refs: Sequence[PatchRef], *, split: str, cache_root: str
) -> EvalSplit:
    """Cache native features on disk; retain compact target/prior metadata."""
    directory = Path(cache_root, split)
    directory.mkdir(parents=True, exist_ok=False)
    feature_files: list[str] = []
    prior: list[np.ndarray] = []
    target: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    patch_ids: list[str] = []
    filled: list[int] = []
    before = Counter(reader.exclusions)
    started = time.monotonic()
    for position, ref in enumerate(refs):
        if position % 100 == 0:
            log_event(
                _logger,
                logging.INFO,
                "read_progress",
                split=ref.split,
                completed=position,
                total=len(refs),
            )
        sample = reader.read_patch(ref)
        if sample is None:
            continue
        prior_cells = pool_10m_cells_to_100m(sample.lst_prior_k)[0]
        feature_file = directory / f"{len(feature_files):06d}.npy"
        np.save(feature_file, sample.features, allow_pickle=False)
        feature_files.append(str(feature_file))
        prior.append(prior_cells)
        target.append(sample.target_100m[0])
        mask.append(sample.mask_100m[0])
        patch_ids.append(sample.meta.patch_id)
        filled.append(sample.meta.filled_feature_pixels)
    if not patch_ids:
        raise ValueError(f"split {split!r} evaluated no patch")
    return EvalSplit(
        split=split,
        requested_patches=len(refs),
        patch_ids=patch_ids,
        feature_files=feature_files,
        prior_100m=np.stack(prior),
        target_100m=np.stack(target),
        mask_100m=np.stack(mask),
        filled_feature_pixels=filled,
        exclusions={k: v for k, v in (Counter(reader.exclusions) - before).items()},
        seconds=time.monotonic() - started,
    )


def score_stage(
    cache: EvalSplit, model: RandomForestRegressor, stage: StageSelection
) -> list[PatchRecord]:
    """Score one stage over a cached split through the shared 10 m pooling path."""
    records: list[PatchRecord] = []
    for i, patch_id in enumerate(cache.patch_ids):
        if i % 100 == 0:
            log_event(
                _logger,
                logging.INFO,
                "score_progress",
                stage=stage.stage,
                split=cache.split,
                completed=i,
                total=len(cache.patch_ids),
            )
        features = np.load(cache.feature_files[i], allow_pickle=False)
        residual = predict_10m(model, features, cache.prior_100m[i], stage)
        prediction = torch.from_numpy(reconstruct_10m_prediction(cache.prior_100m[i], residual))
        pooled = pool_10m_to_100m(prediction)
        target = torch.from_numpy(cache.target_100m[i][None, None])
        mask = torch.from_numpy(cache.mask_100m[i][None, None])
        error_sum, count = masked_abs_error_sums(pooled, target, mask)
        ssim_sum, ssim_windows = masked_ssim_stats(pooled, target, mask)
        records.append(
            PatchRecord(
                patch_id=patch_id,
                split=cache.split,
                valid_cells=int(count),
                abs_error_sum=float(error_sum),
                ssim_sum=float(ssim_sum),
                ssim_windows=int(ssim_windows),
                filled_feature_pixels=cache.filled_feature_pixels[i],
            )
        )
    return records


def summarize_records(
    records: Sequence[PatchRecord], *, requested: int, exclusions: dict[str, int]
) -> SplitSummary:
    """Aggregate per-patch contributions into the split summary."""
    return SplitSummary(
        requested_patches=requested,
        evaluated_patches=len(records),
        valid_cells=sum(record.valid_cells for record in records),
        abs_error_sum=sum(record.abs_error_sum for record in records),
        ssim_sum=sum(record.ssim_sum for record in records),
        ssim_windows=sum(record.ssim_windows for record in records),
        filled_feature_pixels=sum(record.filled_feature_pixels for record in records),
        exclusions=dict(exclusions),
    )


def probe_residuals(
    cache: EvalSplit,
    model: RandomForestRegressor,
    stage: StageSelection,
    indices: Sequence[int],
) -> dict[int, np.ndarray]:
    """Return the predicted residual of each probe patch of a cached split."""
    out: dict[int, np.ndarray] = {}
    for i in indices:
        features = np.load(cache.feature_files[i], allow_pickle=False)
        out[i] = predict_10m(model, features, cache.prior_100m[i], stage)
    return out


# ── model artifacts and replay ────────────────────────────────────────


def write_model(model: RandomForestRegressor, uri: str) -> str:
    """Persist a forest using the type-audited skops format."""
    blob = sio.dumps(model)
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return sha256(blob).hexdigest()


def load_model(uri: str) -> RandomForestRegressor:
    """Load only the forest's known tree type and validate node indices."""
    blob = Path(uri).read_bytes()
    trusted = ["sklearn.tree._tree.Tree"]
    unknown = sio.get_untrusted_types(data=blob)
    if set(unknown) - set(trusted):
        raise ValueError(f"unexpected model types: {unknown}")
    model = sio.loads(blob, trusted=trusted)
    if not isinstance(model, RandomForestRegressor):
        raise ValueError("artifact must be a RandomForestRegressor")
    n_features = int(getattr(model, "n_features_in_", 0))
    if n_features < 1:
        raise ValueError("forest has no fitted input features")
    for estimator in model.estimators_:
        tree = estimator.tree_
        left, right, feature = tree.children_left, tree.children_right, tree.feature
        leaf = (left == -1) & (right == -1)
        positions = np.arange(tree.node_count)
        valid = leaf | (
            (left > positions)
            & (right > positions)
            & (left < tree.node_count)
            & (right < tree.node_count)
            & (feature >= 0)
            & (feature < n_features)
        )
        if not valid.all():
            raise ValueError("invalid or cyclic forest node indices")
    return model


def verify_replay(
    model: RandomForestRegressor,
    artifact_uri: str,
    matrix: np.ndarray,
    *,
    rows: int,
) -> dict[str, Any]:
    """Replay the serialized artifact on a bounded row sample.

    The reloaded forest must return bitwise-identical predictions to the
    in-memory one, so a later independent replay of the artifact is evidence
    about the same model that produced the report.
    """
    loaded = load_model(artifact_uri)
    n = min(rows, matrix.shape[0])
    rows_matrix = np.ascontiguousarray(matrix[:n], dtype=np.float32)
    if n == 0:
        return {"rows": 0, "identical": False, "max_abs_diff_k": None}
    expected = np.asarray(model.predict(rows_matrix), dtype=np.float32)
    replayed = np.asarray(loaded.predict(rows_matrix), dtype=np.float32)
    return {
        "rows": n,
        "identical": bool(np.array_equal(expected, replayed)),
        "max_abs_diff_k": float(np.max(np.abs(expected - replayed))),
    }


# ── report ────────────────────────────────────────────────────────────


@dataclass
class StageResult:
    """One fitted stage forest, its artifact, and its scored splits."""

    stage: int
    channels: tuple[str, ...]
    n_active_channels: int
    feature_order: str
    model_file: str
    model_sha256: str
    fit_seconds: float
    replay: dict[str, Any]
    splits: dict[str, SplitSummary]
    patches: list[PatchRecord]

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "channels": list(self.channels),
            "n_active_channels": self.n_active_channels,
            "feature_order": self.feature_order,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "fit_seconds": self.fit_seconds,
            "replay": self.replay,
            "splits": {
                name: {**asdict(summary), "mae": summary.mae, "ssim": summary.ssim}
                for name, summary in self.splits.items()
            },
            "patches": [asdict(record) for record in self.patches],
        }


@dataclass
class RandomForestReport:
    """The writable random-forest artifact for one invocation."""

    forest: dict[str, Any]
    library_versions: dict[str, str]
    scope: dict[str, Any]
    train_pool: dict[str, Any]
    stages: dict[int, StageResult]
    probes: dict[str, Any]
    source: dict[str, str]
    fingerprints: dict[str, str]
    resolved_config: str
    runtime_seconds: float
    resource_usage: dict[str, int]
    git_revision: str
    written_at: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "target_rule": TARGET_RULE,
            "inference_rule": INFERENCE_RULE,
            "prior_rule": PRIOR_RULE,
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "library_versions": self.library_versions,
            "forest": self.forest,
            "scope": self.scope,
            "train_pool": self.train_pool,
            "stages": {str(stage): result.to_payload() for stage, result in self.stages.items()},
            "probes": self.probes,
            "source": self.source,
            "fingerprints": self.fingerprints,
            "resolved_config": self.resolved_config,
            "runtime_seconds": self.runtime_seconds,
            "resource_usage": self.resource_usage,
            "git_revision": self.git_revision,
            "written_at": self.written_at,
        }


def build_report(
    *,
    cfg: RealSourceConfig,
    spec: ForestSpec,
    scope: dict[str, Any],
    pool: TrainPool,
    max_cells: int,
    seed: int,
    rows_file: str,
    rows_sha256: str,
    stage_results: dict[int, StageResult],
    probes: dict[str, Any],
    resolved_config: str,
    runtime_seconds: float,
    cache_bytes: int,
    git_revision: str,
) -> RandomForestReport:
    """Assemble the writable artifact from the run's evidence."""
    return RandomForestReport(
        forest=spec.payload(),
        library_versions=library_versions(),
        scope=scope,
        train_pool={
            "rule": RESERVOIR_RULE,
            "weighting": WEIGHTING_RULE,
            "seed": seed,
            "max_cells": max_cells,
            "requested_patches": pool.requested_patches,
            "read_patches": pool.read_patches,
            "exclusions": pool.exclusions,
            "candidate_cells": pool.candidate_cells,
            "duplicate_cells": pool.duplicate_cells,
            "selected_cells": pool.selected_cells,
            "anchor_stride_cells": pool.anchor_stride_cells,
            "duplicate_anchors": pool.duplicate_anchors,
            "overlapping_windows": pool.overlapping_windows,
            "rows_file": rows_file,
            "rows_sha256": rows_sha256,
            "selection_sha256": train_rows_digest(pool.keys),
            "seconds": pool.seconds,
        },
        stages=stage_results,
        probes=probes,
        source={
            "patch_index_root": cfg.patch_index_root,
            "training_root": cfg.training_root,
            "features_root": cfg.features_root,
            "ard_root": cfg.ard_root,
        },
        fingerprints=patch_index_fingerprints(cfg.patch_index_root),
        resolved_config=resolved_config,
        runtime_seconds=runtime_seconds,
        resource_usage={
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if platform.system() == "Darwin" else 1024),
            "eval_cache_bytes": cache_bytes,
        },
        git_revision=git_revision,
        written_at=now_iso(),
    )


def write_report(report: RandomForestReport, uri: str) -> str:
    """Write the report to a run-output path (never a release root)."""
    payload = json.dumps(report.to_payload(), indent=2, sort_keys=True)
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return str(path)


def git_revision_for(output_root: str, run_id: str) -> str:
    """Read the Git revision the runner recorded for this run.

    Mirrors ``modeling/baseline.py``'s helper with this pipeline's log path.
    """
    uri = run_context_path(output_root, PIPELINE, run_id)
    if not Path(uri).is_file():
        return "unknown"
    with open(uri, encoding="utf-8") as fh:
        return str(json.load(fh).get("git_commit", "unknown"))


__all__ = [
    "CONFIG_FILE",
    "EvalSplit",
    "ForestSpec",
    "INFERENCE_RULE",
    "METHOD",
    "PIPELINE",
    "PRIMARY_METRIC",
    "PROBE_FILE",
    "RESERVOIR_RULE",
    "RandomForestReport",
    "SECONDARY_METRIC",
    "StageResult",
    "StageSelection",
    "TARGET_RULE",
    "TRAIN_ROWS_FILE",
    "TrainPool",
    "WEIGHTING_RULE",
    "build_report",
    "collect_eval_split",
    "collect_train_pool",
    "git_revision_for",
    "library_versions",
    "load_model",
    "model_file_name",
    "pool_10m_cells_to_100m",
    "predict_10m",
    "probe_residuals",
    "reconstruct_10m_prediction",
    "resolve_stage_selection",
    "score_stage",
    "stage_inputs",
    "stage_matrix",
    "summarize_records",
    "train_rows_digest",
    "verify_replay",
    "write_model",
    "write_report",
    "write_train_rows",
]
