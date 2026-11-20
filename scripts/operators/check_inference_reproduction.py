"""Compare block1000 scene inference on validation-2024 patches with an export.

The new path runs on each patch's exact 160 px window. The score is the export
score: clamp the prediction, pool 10×10 with an arithmetic mean, then sum the
absolute error on the patch's 100 m eligibility mask and native target. SSIM
is not computed. Any split other than validation is refused before the
checkpoint is opened and before a feature window is read.

A patch fails when ``valid_cells`` differ or the relative difference of
``abs_error_sum`` exceeds ``1e-4``. The process then exits non-zero.

Usage
-----
    uv run python scripts/operators/check_inference_reproduction.py \\
        --checkpoint gs://…/best.ckpt \\
        --config-name v6_l4_locked \\
        --patch-ids <id> \\
        --export /path/to/per_patch.parquet \\
        --out data/runs/berlin-inference/reproduction
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
import rasterio

from berlin_lst_downscaling.data.io import (
    RunLogSession,
    ensure_gdal_gcs_auth,
    log_event,
    read_bytes,
)
from berlin_lst_downscaling.modeling.channels import selection_from_config
from berlin_lst_downscaling.modeling.inference import (
    PRIOR_BLOCK1000,
    InferenceProvenance,
    feature_uris,
    patch_bbox_10m,
    relative_difference,
    reproduction_mismatch,
    require_validation_patches,
    resolve_scene_landsat,
    run_scene,
    scaler_sha256,
    scene_product_paths,
    score_patch_like_export,
    seed_from_checkpoint_name,
)
from berlin_lst_downscaling.modeling.patches import (
    RealPatchReader,
    assert_scaler_matches_config,
    load_patch_refs,
    load_scaler,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.prebuilt_cache import running_code_commit
from berlin_lst_downscaling.modeling.run import real_source_config, resolve_run_ssim_data_range
from berlin_lst_downscaling.modeling.v6_val_cell_export import ssim_range_for_checkpoint
from berlin_lst_downscaling.modeling.v6_val_rescore import load_rescore_task

_logger = logging.getLogger(__name__)


def _inference_cli():
    """Load the sibling operator. ``scripts`` is not a package."""
    path = Path(__file__).resolve().with_name("run_berlin_inference.py")
    spec = importlib.util.spec_from_file_location("run_berlin_inference_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--patch-ids", nargs="+", required=True)
    parser.add_argument("--export", required=True, help="per-patch parquet")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out", required=True)
    return parser


def _export_rows(uri: str) -> dict[str, tuple[float, int]]:
    table = pq.read_table(
        io.BytesIO(read_bytes(uri)),
        columns=["patch_id", "abs_error_sum", "valid_cells"],
    )
    rows: dict[str, tuple[float, int]] = {}
    for patch_id, error_sum, cells in zip(
        table.column("patch_id").to_pylist(),
        table.column("abs_error_sum").to_pylist(),
        table.column("valid_cells").to_pylist(),
        strict=True,
    ):
        key = str(patch_id)
        if key in rows:
            raise RuntimeError(f"export has a duplicate patch_id {key!r}")
        rows[key] = (float(error_sum), int(cells))
    return rows


def _read_prediction(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cli = _inference_cli()
    device = cli.require_device(args.device)
    seed = seed_from_checkpoint_name(args.checkpoint)
    cfg = cli.compose_inference_config(args.config_name, seed=seed)
    # Refuse a non-validation id before the checkpoint and before any feature read.
    require_validation_patches(str(cfg.patch_index_root), args.patch_ids)
    ensure_gdal_gcs_auth()
    source = real_source_config(cfg)
    scaler = load_scaler(source.training_root)
    assert_scaler_matches_config(cfg, scaler)
    selection = selection_from_config(cfg)
    checkpoint_path, checkpoint_sha = cli.resolve_checkpoint(
        args.checkpoint, Path(args.out) / "_checkpoints"
    )
    fingerprints = patch_index_fingerprints(source.patch_index_root)
    ssim_range = resolve_run_ssim_data_range(cfg, expected_fingerprints=fingerprints)
    ssim_range = ssim_range_for_checkpoint(checkpoint_path, ssim_range)
    task = load_rescore_task(checkpoint_path, cfg, ssim_range)
    provenance = InferenceProvenance(
        checkpoint_uri=args.checkpoint,
        checkpoint_sha256=checkpoint_sha,
        config_name=args.config_name,
        arm=str(cfg.ablation_arm),
        seed=seed,
        code_commit=running_code_commit() or "unknown",
        scaler_sha256=scaler_sha256(source.training_root),
    )
    refs = {ref.patch_id: ref for ref in load_patch_refs(source, splits=("validation",))}
    missing = [patch_id for patch_id in args.patch_ids if patch_id not in refs]
    if missing:
        raise RuntimeError(f"validation patch ids missing after the index filter: {missing}")
    export = _export_rows(args.export)
    absent = [patch_id for patch_id in args.patch_ids if patch_id not in export]
    if absent:
        raise RuntimeError(f"export is missing patch ids: {absent}")
    reader = RealPatchReader(source)
    col0_10, row0_10, _col100, _row100 = reader.analysis_offsets
    scored: list[tuple[str, float, int, float, int]] = []
    with RunLogSession(args.out, pipeline="inference-reproduction", run_id=uuid4().hex[:8]):
        log_event(
            _logger,
            logging.INFO,
            "inference_reproduction_start",
            patches=list(args.patch_ids),
            prior_mode=PRIOR_BLOCK1000,
            config_name=args.config_name,
        )
        for patch_id in args.patch_ids:
            ref = refs[patch_id]
            features_uri, feature_valid_uri = feature_uris(source.features_root, ref.scene_id)
            landsat_cog, landsat_flag = resolve_scene_landsat(source.ard_root, ref.scene_id)
            run_scene(
                task,
                scene_id=ref.scene_id,
                features_uri=features_uri,
                feature_valid_uri=feature_valid_uri,
                landsat_cog=landsat_cog,
                landsat_flag=landsat_flag,
                scaler=scaler,
                selection=selection,
                out_dir=args.out,
                provenance=provenance,
                prior_mode=PRIOR_BLOCK1000,
                bbox=patch_bbox_10m(ref, col0_10=col0_10, row0_10=row0_10),
                device=device,
                batch_size=int(cfg.data.batch_size),
            )
            tif_path, _json_path = scene_product_paths(args.out, ref.scene_id)
            prediction = _read_prediction(tif_path)
            target_mask = reader.read_target_and_mask(ref)
            if target_mask is None:
                raise RuntimeError(
                    f"{ref.patch_id}: target and eligibility mask are unreadable "
                    f"({dict(reader.exclusions)})"
                )
            got_sum, got_cells = score_patch_like_export(prediction, target_mask[0], target_mask[1])
            reference_sum, reference_cells = export[patch_id]
            scored.append((patch_id, got_sum, got_cells, reference_sum, reference_cells))
        log_event(_logger, logging.INFO, "inference_reproduction_scored", patches=len(scored))
    print(
        "patch_id\tabs_error_sum\treference_abs_error_sum\trel_diff\t"
        "valid_cells\treference_valid_cells\tstatus"
    )
    failed = False
    for patch_id, got_sum, got_cells, reference_sum, reference_cells in scored:
        reason = reproduction_mismatch(
            got_sum=got_sum,
            got_cells=got_cells,
            reference_sum=reference_sum,
            reference_cells=reference_cells,
        )
        if reason is not None:
            failed = True
        difference = relative_difference(got_sum, reference_sum)
        rel = "inf" if difference == float("inf") else f"{difference:.6g}"
        status = "ok" if reason is None else reason
        print(
            f"{patch_id}\t{got_sum:.8g}\t{reference_sum:.8g}\t{rel}\t"
            f"{got_cells}\t{reference_cells}\t{status}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
