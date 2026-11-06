"""Validation-only rescore of the frozen v6 ablation matrix (27 ``best.ckpt`` files).

Scope
-----
One job scores the 27 retained best checkpoints (9 arms x seeds 0, 1, 2 of the
matrix at source SHA ``070b33c``) on the 2024 validation split only. No training
step runs, no checkpoint is written, and no test (2025) ref, metric, or export
is ever loaded: the data module is built with ``splits == ("validation",)`` and
every run is composed with ``evaluate_test=false``.

Cache reuse
-----------
The validation patches are read from the published sources once and written to a
single job-local cache at the full v6 schema width (22 channels, identity channel
indices). Each arm reads that cache read-only and collates with its own channel
indices, which is the live-reader collation path of ``RealPatchDataModule`` and is
byte-identical to what the arm's own training cache stored. No prebuilt cache is
read or written.

Identity checks
---------------
The checkpoint is downloaded generation-pinned and hashed, loaded strictly, and
its hyperparameters are compared with the locked arm config. The stored
``ModelCheckpoint`` best score is re-derived from the unclamped validation MAE
within the tolerance the training lifecycle uses.

Outputs
-------
Per run one schema-version-2 per-patch export (``eval_validation_patches.parquet``)
plus one manifest, uploaded create-only (exports first, manifest last). The export
is the canonical input for ``analysis.scene_bootstrap`` and for the subsequent
spatial analysis: it carries pooled per-patch/scene sums only, no cell-level
predictions and no raw 10 m tensors.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import cast

import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.io import log_event, storage
from berlin_lst_downscaling.modeling.aggregation import (
    AggregationMethod,
    resolve_aggregation_method,
)
from berlin_lst_downscaling.modeling.channels import (
    V6_ABLATION_ARMS,
    schema_version_from_config,
    selection_from_config,
)
from berlin_lst_downscaling.modeling.contracts import RealBatch, validate_real_batch
from berlin_lst_downscaling.modeling.evaluation_exports import (
    PATCH_EXPORT_SCHEMA_VERSION,
    EvaluationIdentity,
    patch_export_filename,
    report_metrics,
    score_batch_patches,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.guards import (
    contract_invariants,
    early_stopping_from_config,
    evaluate_test_from_config,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.lr_schedule import lr_schedule_from_config
from berlin_lst_downscaling.modeling.metrics import (
    EVAL_PREDICTION_CLAMP_K,
    MaskedMAE,
    pool_10m_to_100m,
)
from berlin_lst_downscaling.modeling.patch_cache import PatchCache, _json_digest
from berlin_lst_downscaling.modeling.patches import (
    collate_real_batch,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.real_task import CachedRealPatchDataset, RealLSTTask
from berlin_lst_downscaling.modeling.run import (
    build_real_datamodule,
    real_source_config,
    resolve_run_ssim_data_range,
)
from berlin_lst_downscaling.modeling.ssim_range import SsimDataRange
from berlin_lst_downscaling.modeling.v6_ablation_matrix import load_v6_matrix

_logger = logging.getLogger(__name__)

MATRIX_SOURCE_SHA = "070b33ce4323834b7813db4609b5052702c4505f"
MATRIX_LABEL_PREFIX = "v6m-070b33c"
MATRIX_EVIDENCE_PREFIX = "gs://berlin-lst-training-data/qa/modeling/vertex-smoke"
RESCORE_LABEL = "v6m-070b33c-val-rescore-notest-r2"
RESCORE_PREFIX = f"{MATRIX_EVIDENCE_PREFIX}/{RESCORE_LABEL}"
RESCORE_MANIFEST_URI = f"{RESCORE_PREFIX}/manifest.json"
RESCORE_SPLIT = "validation"
RESCORE_SPLITS: tuple[str, ...] = ("validation",)
EXPECTED_ARMS: tuple[str, ...] = (
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "LOO-M",
    "LOO-S",
    "NO-NDVI-NDWI",
    "NO-RED-NIR-IDX",
)
EXPECTED_SEEDS: tuple[int, ...] = (0, 1, 2)
EXPECTED_MAX_EPOCHS = 40
EXPECTED_LR_SCHEDULE = "cosine"
EXPECTED_RUNS = 27
MANIFEST_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_FILE = REPO_ROOT / "configs" / "ablation" / "v6_matrix.yaml"
CONFIG_DIR = REPO_ROOT / "configs" / "modeling"

_MONITOR = "validation/mae_100m"
_SELECTION_TOLERANCE = 1e-3
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class RescoreRun:
    """One checkpoint to rescore and where its validation export goes."""

    arm_id: str
    seed: int
    config_name: str
    run_label: str
    checkpoint_uri: str
    export_uri: str

    @property
    def hydra_overrides(self) -> tuple[str, ...]:
        return (
            f"seed={self.seed}",
            f"trainer.max_epochs={EXPECTED_MAX_EPOCHS}",
            f"trainer.lr_schedule.name={EXPECTED_LR_SCHEDULE}",
            "evaluate_test=false",
        )


def validate_inventory(runs: Sequence[RescoreRun]) -> None:
    """Fail closed unless ``runs`` is exactly the 27-run (arm, seed) matrix."""
    if len(runs) != EXPECTED_RUNS:
        raise ValueError(f"rescore inventory must hold {EXPECTED_RUNS} runs, got {len(runs)}")
    pairs = [(run.arm_id, run.seed) for run in runs]
    if len(set(pairs)) != len(pairs):
        raise ValueError("rescore inventory has a duplicate (arm, seed)")
    for field in ("run_label", "checkpoint_uri", "export_uri"):
        values = [getattr(run, field) for run in runs]
        if len(set(values)) != len(values):
            raise ValueError(f"rescore inventory has a duplicate {field}")
    for run in runs:
        if run.arm_id not in EXPECTED_ARMS:
            raise ValueError(f"unknown arm {run.arm_id!r} in the rescore inventory")
        if run.seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected seed {run.seed!r} for arm {run.arm_id}")
    expected_pairs = {(arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_SEEDS}
    if set(pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(pairs))
        raise ValueError(f"rescore inventory is missing (arm, seed) pairs {missing}")
    checkpoints = {run.checkpoint_uri for run in runs}
    for run in runs:
        if not run.export_uri.startswith(RESCORE_PREFIX + "/"):
            raise ValueError(f"export {run.export_uri} is outside {RESCORE_PREFIX}/")
        if run.export_uri in checkpoints:
            raise ValueError(f"export {run.export_uri} would overwrite a checkpoint")


def rescore_inventory(matrix_path: Path = MATRIX_FILE) -> tuple[RescoreRun, ...]:
    """Expand the frozen matrix into the 27 checkpoint/export pairs."""
    matrix = load_v6_matrix(matrix_path)
    if matrix.max_epochs != EXPECTED_MAX_EPOCHS:
        raise ValueError(f"matrix max_epochs must be {EXPECTED_MAX_EPOCHS}")
    if matrix.lr_schedule != EXPECTED_LR_SCHEDULE:
        raise ValueError(f"matrix lr_schedule must be {EXPECTED_LR_SCHEDULE!r}")
    if matrix.early_stopping is not False:
        raise ValueError("matrix early_stopping must be false")
    if set(matrix.arms) != set(EXPECTED_ARMS) or set(V6_ABLATION_ARMS) != set(EXPECTED_ARMS):
        raise ValueError("matrix arms or the arm registry differ from the expected nine arms")
    runs: list[RescoreRun] = []
    for arm_id in EXPECTED_ARMS:
        if tuple(matrix.arms[arm_id].seeds) != EXPECTED_SEEDS:
            raise ValueError(
                f"arm {arm_id}: seeds must be {list(EXPECTED_SEEDS)}, "
                f"got {list(matrix.arms[arm_id].seeds)}"
            )
        for seed in EXPECTED_SEEDS:
            label = f"{MATRIX_LABEL_PREFIX}-{arm_id.lower()}-seed{seed}-ep{EXPECTED_MAX_EPOCHS}-cos"
            runs.append(
                RescoreRun(
                    arm_id=arm_id,
                    seed=seed,
                    config_name=V6_ABLATION_ARMS[arm_id].config_name,
                    run_label=label,
                    checkpoint_uri=f"{MATRIX_EVIDENCE_PREFIX}/{label}/best.ckpt",
                    export_uri=f"{RESCORE_PREFIX}/{label}/{patch_export_filename(RESCORE_SPLIT)}",
                )
            )
    validate_inventory(runs)
    return tuple(runs)


def compose_run_config(
    run: RescoreRun, *, output_root: str, config_dir: Path = CONFIG_DIR
) -> DictConfig:
    """Compose the arm's locked config exactly as the matrix run did, and verify it.

    The v6 lock requires an ``output_root`` under ``runs/`` with a ``seed-<n>``
    segment; the segment is appended here, so callers pass the per-run directory.
    The composed path is only validated and recorded: nothing is written to it.
    """
    seed_segment = f"seed-{run.seed}"
    if seed_segment not in output_root.split("/"):
        output_root = f"{output_root.rstrip('/')}/{seed_segment}"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name=run.config_name,
            overrides=[*run.hydra_overrides, f"output_root={output_root}"],
        )
    guard_modeling_config(cfg, run.config_name)
    contract_invariants(cfg)
    checks = {
        "ablation_release == v6": cfg.get("ablation_release") == "v6",
        "ablation_arm": cfg.get("ablation_arm") == run.arm_id,
        "seed": int(cfg.seed) == run.seed,
        "trainer.max_epochs": int(cfg.trainer.max_epochs) == EXPECTED_MAX_EPOCHS,
        "lr_schedule": lr_schedule_from_config(cfg).name == EXPECTED_LR_SCHEDULE,
        "early_stopping disabled": early_stopping_from_config(cfg).enabled is False,
        "evaluate_test false": evaluate_test_from_config(cfg) is False,
        "data.kind real": str(cfg.data.kind) == "real",
        "tessera disabled": not bool(OmegaConf.select(cfg, "data.tessera_enabled") or False),
        "data.splits": tuple(str(s) for s in cfg.data.splits) == ("train", "validation"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{run.run_label}: composed config is off-contract: {failed}")
    return cfg


def assert_shared_data_identity(cfgs: Sequence[DictConfig]) -> None:
    """Fail unless every run reads the same sources, batch size, release, schema, aggregation."""
    if not cfgs:
        raise ValueError("no configs to compare")

    def identity(cfg: DictConfig) -> dict[str, object]:
        return {
            "source": real_source_config(cfg),
            "batch_size": int(cfg.data.batch_size),
            "data_release_id": str(cfg.data_release_id),
            "schema_version": schema_version_from_config(cfg),
            "aggregation": resolve_aggregation_method(cfg.get("aggregation")),
            "cohort": OmegaConf.to_container(
                OmegaConf.masked_copy(
                    cfg.data,
                    ["scene_ids", "max_patches_per_split", "probe", "mode", "splits"],
                ),
                resolve=True,
            ),
        }

    first = identity(cfgs[0])
    for index, cfg in enumerate(cfgs[1:], start=1):
        other = identity(cfg)
        differing = sorted(key for key in first if first[key] != other[key])
        if differing:
            raise ValueError(f"config {index} differs from config 0 in {differing}")


@dataclass(frozen=True)
class CheckpointRef:
    """A generation-pinned, hashed local copy of a remote checkpoint."""

    uri: str
    generation: int
    bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(uri: str, dest: Path) -> CheckpointRef:
    """Download ``uri`` pinned to the generation just observed; verify size; hash it."""
    from google.cloud import storage as gcs

    if not uri.startswith("gs://"):
        raise ValueError(f"checkpoint URI must be gs://, got {uri!r}")
    bucket_name, _, key = uri[len("gs://") :].partition("/")
    if not bucket_name or not key:
        raise ValueError(f"checkpoint URI has no bucket/key: {uri!r}")
    blob = gcs.Client().bucket(bucket_name).get_blob(key)
    if blob is None:
        raise FileNotFoundError(uri)
    generation, expected_size = blob.generation, blob.size
    if generation is None or expected_size is None:
        raise RuntimeError(f"{uri}: object metadata lacks generation or size")
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest), if_generation_match=generation)
    size = dest.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"{uri}: downloaded {size} bytes, object has {expected_size}")
    return CheckpointRef(uri=uri, generation=int(generation), bytes=size, sha256=_sha256_file(dest))


def checkpoint_selection_state(path: Path) -> dict[str, object]:
    """Read the checkpoint's own ModelCheckpoint state and verify it is the best-epoch file."""
    # Own Lightning checkpoints carry hyperparameter objects, so weights_only stays off.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    callbacks = ckpt.get("callbacks") if isinstance(ckpt, dict) else None
    if not isinstance(callbacks, dict):
        raise RuntimeError(f"{path.name}: checkpoint has no callbacks state")
    states = [
        state
        for key, state in callbacks.items()
        if str(key).startswith("ModelCheckpoint")
        and isinstance(state, dict)
        and state.get("monitor") == _MONITOR
    ]
    if len(states) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one ModelCheckpoint monitoring {_MONITOR}, "
            f"found {len(states)}"
        )
    state = states[0]
    best = state.get("best_model_score")
    if best is None:
        raise RuntimeError(f"{path.name}: ModelCheckpoint has no best_model_score")
    best_score = float(best)
    if not math.isfinite(best_score):
        raise RuntimeError(f"{path.name}: best_model_score is not finite ({best_score})")
    current = state.get("current_score")
    if current is not None and abs(float(current) - best_score) > 1e-9:
        raise RuntimeError(
            f"{path.name}: current_score {float(current)} != best_model_score {best_score}; "
            "this is not the best-epoch checkpoint"
        )
    best_path_name = Path(str(state.get("best_model_path", ""))).name
    if not best_path_name.startswith("best-"):
        raise RuntimeError(f"{path.name}: best_model_path {best_path_name!r} is not a best-* file")
    return {
        "best_model_score": best_score,
        "best_model_path_name": best_path_name,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }


def load_rescore_task(path: Path, cfg: DictConfig, ssim_range: SsimDataRange) -> RealLSTTask:
    """Load the checkpoint strictly and compare its hyperparameters with the locked config."""
    task = RealLSTTask.load_from_checkpoint(str(path), map_location="cpu", strict=True)
    hparams = task.hparams
    expected: dict[str, object] = {
        "n_active_channels": selection_from_config(cfg).n_active,
        "base_width": int(cfg.model.base_width),
        "depth": int(cfg.model.depth),
        "learning_rate": float(cfg.trainer.learning_rate),
        "weight_decay": float(cfg.trainer.weight_decay),
        "residual_prior": bool(cfg.get("stage1_residual_prior", False)),
        "loss_name": str(cfg.contract.loss),
        "aggregation": resolve_aggregation_method(cfg.get("aggregation")),
        "lr_schedule": lr_schedule_from_config(cfg).payload(),
        "ssim_data_range": ssim_range.payload(),
    }
    for field, want in expected.items():
        if field not in hparams:
            raise RuntimeError(f"{path.name}: checkpoint hparams lack {field!r}")
        got = hparams[field]
        if field == "aggregation":
            got = resolve_aggregation_method(got)
        elif field in ("lr_schedule", "ssim_data_range"):
            got = dict(got) if got is not None else None
        if got != want:
            raise RuntimeError(f"{path.name}: hparam {field!r} is {got!r}, config says {want!r}")
    task.eval()
    return task


@dataclass(frozen=True)
class ValidationCache:
    """The single job-local, validation-only, full-width cache all arms read."""

    root: Path
    provenance: dict[str, object]
    patch_ids: tuple[str, ...]
    scope: dict[str, object]
    build_seconds: float
    source_channels: int

    @property
    def provenance_sha256(self) -> str:
        return _json_digest(self.provenance)


def build_validation_cache(cfg: DictConfig, cache_root: Path) -> ValidationCache:
    """Read the validation split once and write the full-schema-width cache."""
    width = feature_schema(FEATURE_SCHEMA_VERSION_V6).n_channels
    dm = build_real_datamodule(
        cfg,
        cache_root=None,
        cache_max_bytes=0,
        splits=RESCORE_SPLITS,
        mode="stream",
        shuffle_train=False,
        n_active_channels=width,
        channel_indices=tuple(range(width)),
        cache_source="job_local",
        allow_job_local_fallback=False,
    )
    dm.setup()
    provenance = dm.build_validation_only_cache(cache_root, max_bytes=int(cfg.data.cache_max_bytes))
    scope = dm.stats()
    per_split = scope["patches_per_split"]
    if not isinstance(per_split, dict) or set(per_split) != {RESCORE_SPLIT}:
        raise RuntimeError(f"validation cache read unexpected splits: {per_split!r}")
    cache = PatchCache(cache_root, provenance)
    for file in cache_root.iterdir():
        if file.is_file():
            file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return ValidationCache(
        root=cache_root,
        provenance=provenance,
        patch_ids=cache.patch_ids,
        scope=scope,
        build_seconds=float(dm.cache_build_seconds),
        source_channels=width,
    )


def validation_loader(cache: ValidationCache, cfg: DictConfig) -> DataLoader[RealBatch]:
    """Read the shared cache with this arm's channel selection (live-reader collation)."""
    sel = selection_from_config(cfg)
    dataset = CachedRealPatchDataset(cache.root, cache.provenance, range(len(cache.patch_ids)))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate_real_batch, n_active_channels=sel.n_active, channel_indices=sel.indices
        ),
    )
    # ``collate_real_batch`` stacks each batch of RealSample into a RealBatch.
    return cast(DataLoader[RealBatch], loader)


def score_validation(
    task: RealLSTTask,
    loader: DataLoader[RealBatch],
    *,
    aggregation: AggregationMethod,
    ssim_range: SsimDataRange,
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    """Score every validation batch; return per-patch rows and the unclamped MAE recheck."""
    recheck = MaskedMAE()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            validate_real_batch(batch, n_active_channels=task.n_active_channels)
            dev_batch = dataclasses.replace(
                batch,
                features=batch.features.to(device),
                lst_prior=batch.lst_prior.to(device),
                target_100m=batch.target_100m.to(device),
                mask_100m=batch.mask_100m.to(device),
            )
            prediction = task(dev_batch).float().cpu()
            expected = (batch.features.shape[0], 1) + tuple(batch.features.shape[2:])
            if tuple(prediction.shape) != expected:
                raise RuntimeError(
                    f"prediction shape mismatch: {tuple(prediction.shape)} != {expected}"
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("checkpoint produced non-finite predictions")
            pooled = pool_10m_to_100m(prediction, aggregation, batch.mask_100m)
            recheck.update(pooled, batch.target_100m, batch.mask_100m)
            batch_rows = score_batch_patches(
                pooled,
                batch.target_100m,
                batch.mask_100m,
                batch.metadata,
                data_range_k=ssim_range.data_range_k,
            )
            for row in batch_rows:
                if row["split"] != RESCORE_SPLIT:
                    raise RuntimeError(f"row {row['patch_id']} has split {row['split']!r}")
            rows.extend(batch_rows)
    if not rows:
        raise RuntimeError("validation scoring produced no rows")
    return rows, float(recheck.compute())


def upload_create_only(local: Path, uri: str) -> dict[str, object]:
    """Upload ``local`` to ``uri`` under the rescore prefix.

    Create-only, never overwrite, never delete.
    """
    if not uri.startswith(RESCORE_PREFIX + "/"):
        raise ValueError(f"{uri} is outside the rescore prefix {RESCORE_PREFIX}/")
    storage.create_only_upload(local, uri)
    return {"uri": uri, "sha256": _sha256_file(local), "bytes": local.stat().st_size}


def _check_run_args(source_sha: str, image_digest: str, manifest_uri: str) -> None:
    if manifest_uri != RESCORE_MANIFEST_URI:
        raise ValueError(f"manifest URI must be {RESCORE_MANIFEST_URI}, got {manifest_uri!r}")
    if not _SHA_RE.match(source_sha):
        raise ValueError("source_sha must be 40 lowercase hex characters")
    if not _DIGEST_RE.match(image_digest):
        raise ValueError("image_digest must be sha256:<64 hex>")


def run_rescore(
    *,
    output_root: Path,
    source_sha: str,
    image_digest: str,
    manifest_uri: str,
    device: str,
    upload: bool = True,
) -> dict[str, object]:
    """Score all 27 checkpoints on validation 2024 and publish exports, manifest last."""
    _check_run_args(source_sha, image_digest, manifest_uri)
    runs = rescore_inventory()
    cfgs = {
        run.run_label: compose_run_config(
            run, output_root=str(output_root / "runs" / run.run_label)
        )
        for run in runs
    }
    assert_shared_data_identity(list(cfgs.values()))
    if upload:
        for uri in (manifest_uri, *(run.export_uri for run in runs)):
            if storage.exists(uri):
                raise FileExistsError(f"create-only destination already exists: {uri}")

    # The cache holds the 22-channel validation tensors, so size it by the largest
    # arm budget: every arm shares one cache and the budget only bounds disk use.
    ref_cfg = max(cfgs.values(), key=lambda c: int(c.data.cache_max_bytes))
    source = real_source_config(ref_cfg)
    fingerprints = patch_index_fingerprints(source.patch_index_root)
    ssim_range = resolve_run_ssim_data_range(ref_cfg, expected_fingerprints=fingerprints)
    output_root.mkdir(parents=True, exist_ok=True)

    log_event(_logger, logging.INFO, "rescore_cache_build_start", runs=len(runs))
    cache = build_validation_cache(ref_cfg, output_root / "cache")
    (output_root / "data_scope.json").write_text(
        json.dumps(cache.scope, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_event(
        _logger,
        logging.INFO,
        "rescore_cache_built",
        patches=len(cache.patch_ids),
        build_seconds=cache.build_seconds,
        provenance_sha256=cache.provenance_sha256,
    )

    entries: list[dict[str, object]] = []
    export_infos: list[dict[str, object]] = []
    export_paths: dict[str, Path] = {}
    for run in runs:
        cfg = cfgs[run.run_label]
        seed_everything(run.seed)
        dest = output_root / "checkpoints" / f"{run.run_label}.ckpt"
        run_dir = output_root / "runs" / run.run_label
        aggregation = resolve_aggregation_method(cfg.get("aggregation"))
        try:
            ref_ckpt = fetch_checkpoint(run.checkpoint_uri, dest)
            selection = checkpoint_selection_state(dest)
            task = load_rescore_task(dest, cfg, ssim_range).to(device)
            rows, recomputed = score_validation(
                task,
                validation_loader(cache, cfg),
                aggregation=aggregation,
                ssim_range=ssim_range,
                device=torch.device(device),
            )
        finally:
            dest.unlink(missing_ok=True)
        best_score = float(selection["best_model_score"])  # type: ignore[arg-type]
        tolerance = _SELECTION_TOLERANCE * max(1.0, abs(best_score))
        if abs(recomputed - best_score) > tolerance:
            raise RuntimeError(
                f"{run.run_label}: recomputed validation MAE {recomputed:.6f} != stored "
                f"best_model_score {best_score:.6f} (tolerance {tolerance:.6f})"
            )
        if len(rows) != len(cache.patch_ids):
            raise RuntimeError(
                f"{run.run_label}: scored {len(rows)} patches, cache holds {len(cache.patch_ids)}"
            )
        report = report_metrics(rows)
        export_path = write_patch_export(
            rows,
            run_dir / patch_export_filename(RESCORE_SPLIT),
            identity=EvaluationIdentity(
                arm=run.arm_id,
                seed=run.seed,
                run_id=run.run_label,
                checkpoint=run.checkpoint_uri,
            ),
            split=RESCORE_SPLIT,
            ssim_range=ssim_range,
        )
        export_paths[run.run_label] = export_path
        export_info: dict[str, object] = {
            "path": export_path.relative_to(output_root).as_posix(),
            "uri": None,
            "sha256": _sha256_file(export_path),
            "bytes": export_path.stat().st_size,
            "rows": len(rows),
            "schema_version": PATCH_EXPORT_SCHEMA_VERSION,
        }
        export_infos.append(export_info)
        entry: dict[str, object] = {
            "arm": run.arm_id,
            "seed": run.seed,
            "config_name": run.config_name,
            "original_run_label": run.run_label,
            "hydra_overrides": list(run.hydra_overrides),
            "checkpoint": {
                "uri": ref_ckpt.uri,
                "generation": ref_ckpt.generation,
                "bytes": ref_ckpt.bytes,
                "sha256": ref_ckpt.sha256,
                "best_model_score": best_score,
                "best_model_path_name": selection["best_model_path_name"],
                "epoch": selection["epoch"],
                "global_step": selection["global_step"],
            },
            "recomputed_validation_mae_100m": recomputed,
            "validation_report": report,
            "patches": len(rows),
            "scenes": len({str(row["scene_id"]) for row in rows}),
            "export": export_info,
            "split": RESCORE_SPLIT,
            "evaluate_test": False,
        }
        (run_dir / "rescore_run.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8"
        )
        entries.append(entry)
        log_event(
            _logger,
            logging.INFO,
            "rescore_run_done",
            run_label=run.run_label,
            recomputed_validation_mae_100m=recomputed,
            best_model_score=best_score,
        )

    manifest = _build_manifest(
        runs=runs,
        entries=entries,
        cache=cache,
        ssim_range=ssim_range,
        fingerprints=fingerprints,
        source_sha=source_sha,
        image_digest=image_digest,
        manifest_uri=manifest_uri,
        device=device,
        uploaded=upload,
        data_release_id=str(ref_cfg.data_release_id),
    )
    if upload:
        # The manifest holds these same dicts, so the recorded values land in it.
        for run, info in zip(runs, export_infos, strict=True):
            info.update(upload_create_only(export_paths[run.run_label], run.export_uri))
            log_event(_logger, logging.INFO, "rescore_export_uploaded", uri=run.export_uri)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if upload:
        upload_create_only(manifest_path, manifest_uri)
        log_event(_logger, logging.INFO, "rescore_manifest_uploaded", uri=manifest_uri)
    return manifest


def _build_manifest(
    *,
    runs: Sequence[RescoreRun],
    entries: list[dict[str, object]],
    cache: ValidationCache,
    ssim_range: SsimDataRange,
    fingerprints: dict[str, str],
    source_sha: str,
    image_digest: str,
    manifest_uri: str,
    device: str,
    uploaded: bool,
    data_release_id: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "v6_validation_rescore",
        "rescore_label": RESCORE_LABEL,
        "evaluate_test": False,
        "split": RESCORE_SPLIT,
        "splits_read": list(RESCORE_SPLITS),
        "test_split": "not_read",
        "matrix": {
            "source_sha": MATRIX_SOURCE_SHA,
            "label_prefix": MATRIX_LABEL_PREFIX,
            "matrix_file": "configs/ablation/v6_matrix.yaml",
            "max_epochs": EXPECTED_MAX_EPOCHS,
            "lr_schedule": EXPECTED_LR_SCHEDULE,
            "early_stopping": False,
            "arms": list(EXPECTED_ARMS),
            "seeds": list(EXPECTED_SEEDS),
            "runs": len(runs),
        },
        "rescore_source_sha": source_sha,
        "image_digest": image_digest,
        "device": device,
        "destination_prefix": RESCORE_PREFIX,
        "manifest_uri": manifest_uri,
        "uploaded": uploaded,
        "patch_cache": {
            "source": "job_local_validation_only",
            "builds": 1,
            "patches": len(cache.patch_ids),
            "provenance_sha256": cache.provenance_sha256,
            "build_seconds": cache.build_seconds,
            "source_channels": cache.source_channels,
            "prebuilt_read": False,
        },
        "data_scope": {
            "patches_per_split": cache.scope["patches_per_split"],
            "skipped_per_split": cache.scope.get("skipped_per_split", {}),
            "exclusions": cache.scope["exclusions"],
        },
        "data_release_id": data_release_id,
        "source_roots": cache.provenance.get("source_roots"),
        "patch_index_fingerprints": fingerprints,
        "ssim_data_range": ssim_range.payload(),
        "prediction_clamp_k": [float(v) for v in EVAL_PREDICTION_CLAMP_K],
        "export_schema_version": PATCH_EXPORT_SCHEMA_VERSION,
        "export_consumers": ["analysis.scene_bootstrap", "spatial analysis"],
        "runs": entries,
        "created_at": datetime.now(UTC).isoformat(),
    }
