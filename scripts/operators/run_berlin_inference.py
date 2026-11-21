"""Run frozen v6 U-Net inference on features/v6 scenes and optionally mosaic an overpass.

The locked config is composed with the matrix schedule (40 epochs, cosine) and
checked before any root override. ``--features-root`` and ``--ard-root`` change
only the files that are read. The scaler stays on the pinned training root.
``native100`` is the inference prior. ``block1000`` is the training prior and
is selected explicitly. Nothing here scores the 2025 test split.

Usage
-----
    uv run python scripts/operators/run_berlin_inference.py \\
        --checkpoint gs://…/v6m-070b33c-l4-seed2-ep40-cos/best.ckpt \\
        --config-name v6_l4_locked \\
        --scenes LC09_L2SP_193023_20250608 LC09_L2SP_193024_20250608 \\
        --out data/runs/berlin-inference/l4-seed2 \\
        --mosaic
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import torch
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, ensure_gdal_gcs_auth, log_event
from berlin_lst_downscaling.modeling.channels import V6_ABLATION_ARMS, selection_from_config
from berlin_lst_downscaling.modeling.guards import contract_invariants, guard_modeling_config
from berlin_lst_downscaling.modeling.inference import (
    INFERENCE_CONFIGS,
    PRIOR_MODES,
    InferenceProvenance,
    feature_uris,
    resolve_scene_landsat,
    run_scene,
    scaler_sha256,
    scene_product_paths,
    scenes_with_done_features,
    seed_from_checkpoint_name,
    sha256_file,
)
from berlin_lst_downscaling.modeling.mosaic import (
    acquisition_year,
    group_scenes_by_overpass,
    write_overpass_mosaic,
)
from berlin_lst_downscaling.modeling.patches import (
    assert_scaler_matches_config,
    load_scaler,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.prebuilt_cache import running_code_commit
from berlin_lst_downscaling.modeling.run import real_source_config, resolve_run_ssim_data_range
from berlin_lst_downscaling.modeling.v6_val_cell_export import ssim_range_for_checkpoint
from berlin_lst_downscaling.modeling.v6_val_rescore import (
    CONFIG_DIR,
    EXPECTED_LR_SCHEDULE,
    EXPECTED_MAX_EPOCHS,
    fetch_checkpoint,
    load_rescore_task,
)

_logger = logging.getLogger(__name__)


def select_scene_ids(
    features_root: str,
    *,
    scenes: Sequence[str] | None,
    year: int | None,
) -> list[str]:
    """Return the scene ids to run. Exactly one of ``scenes`` or ``year`` is set.

    ``year`` keeps ledger rows whose ``feature_stack`` status is ``done`` and
    whose acquisition year matches. An unparseable id raises. An empty year
    raises.
    """
    if (scenes is None) == (year is None):
        raise ValueError("pass exactly one of scenes or year")
    if scenes is not None:
        if not scenes:
            raise ValueError("scenes is empty")
        if len(set(scenes)) != len(tuple(scenes)):
            raise ValueError("duplicate scene id in scenes")
        return [str(scene_id) for scene_id in scenes]
    matched = [
        scene_id
        for scene_id in scenes_with_done_features(features_root)
        if acquisition_year(scene_id) == year
    ]
    if not matched:
        raise RuntimeError(f"no done feature_stack scene for year {year} under {features_root}")
    return matched


def require_device(name: str) -> torch.device:
    """Return the requested device. An unavailable device raises; nothing falls back."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda was requested but CUDA is not available")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("device mps was requested but MPS is not available")
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"device must be cpu, mps, or cuda, got {name!r}")
    return torch.device(name)


def compose_inference_config(config_name: str, *, seed: int) -> DictConfig:
    """Compose a locked v6 arm the way the matrix run did, then guard it.

    Roots stay at the lock. Callers apply a features or ARD override only after
    this returns, and only for the files they open.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if config_name not in INFERENCE_CONFIGS:
        raise ValueError(f"config must be one of {INFERENCE_CONFIGS}, got {config_name!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an int >= 0, got {seed!r}")
    output_root = f"data/runs/berlin-inference/{config_name}/seed-{seed}"
    overrides = [
        f"seed={seed}",
        f"trainer.max_epochs={EXPECTED_MAX_EPOCHS}",
        f"trainer.lr_schedule.name={EXPECTED_LR_SCHEDULE}",
        "evaluate_test=false",
        f"output_root={output_root}",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    guard_modeling_config(cfg, config_name)
    contract_invariants(cfg)
    if str(cfg.ablation_arm) not in {"L1", "L4"}:
        raise RuntimeError(f"{config_name}: arm {cfg.ablation_arm!r} is not L1 or L4")
    if V6_ABLATION_ARMS[str(cfg.ablation_arm)].config_name != config_name:
        raise RuntimeError(f"{config_name} does not match arm {cfg.ablation_arm}")
    return cfg


def resolve_checkpoint(uri: str, dest_dir: Path) -> tuple[Path, str]:
    """Return a local checkpoint path and its sha256. ``gs://`` uses ``fetch_checkpoint``."""
    if uri.startswith("gs://"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(uri).name
        ref = fetch_checkpoint(uri, dest)
        return dest, ref.sha256
    path = Path(uri)
    if not path.is_file():
        raise FileNotFoundError(uri)
    return path, sha256_file(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local path or gs:// best.ckpt")
    parser.add_argument("--config-name", required=True, choices=INFERENCE_CONFIGS)
    parser.add_argument("--scenes", nargs="+", help="Landsat scene ids")
    parser.add_argument("--year", type=int, help="acquisition year of done feature scenes")
    parser.add_argument("--features-root", help="default: the locked config's features root")
    parser.add_argument("--ard-root", help="default: the locked config's ARD root")
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default="native100")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("COL0", "ROW0", "WIDTH", "HEIGHT"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mosaic", action="store_true")
    parser.add_argument("--tile-px", type=int, default=160)
    parser.add_argument(
        "--stride-px",
        type=int,
        default=160,
        help="160 (product): tiles on the patch lattice; 80: overlapping, centre crop",
    )
    parser.add_argument(
        "--mask-native-invalid",
        action="store_true",
        help="NaN wherever the 100 m Landsat cell is not native-valid (for block1000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.scenes is None) == (args.year is None):
        parser.error("pass exactly one of --scenes or --year")
    device = require_device(args.device)
    seed = seed_from_checkpoint_name(args.checkpoint)
    cfg = compose_inference_config(args.config_name, seed=seed)
    features_root = args.features_root or str(cfg.features_root)
    ard_root = args.ard_root or str(cfg.ard_root)
    scene_ids = select_scene_ids(features_root, scenes=args.scenes, year=args.year)
    ensure_gdal_gcs_auth()
    source = real_source_config(cfg)
    scaler = load_scaler(source.training_root)
    assert_scaler_matches_config(cfg, scaler)
    selection = selection_from_config(cfg)
    checkpoint_path, checkpoint_sha = resolve_checkpoint(
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
    bbox = None if args.bbox is None else (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
    written: list[str] = []
    with RunLogSession(args.out, pipeline="berlin-inference", run_id=uuid4().hex[:8]):
        log_event(
            _logger,
            logging.INFO,
            "berlin_inference_start",
            config_name=args.config_name,
            arm=provenance.arm,
            seed=seed,
            prior_mode=args.prior_mode,
            scenes=scene_ids,
            device=str(device),
        )
        for scene_id in scene_ids:
            features_uri, feature_valid_uri = feature_uris(features_root, scene_id)
            landsat_cog, landsat_flag = resolve_scene_landsat(ard_root, scene_id)
            run_scene(
                task,
                scene_id=scene_id,
                features_uri=features_uri,
                feature_valid_uri=feature_valid_uri,
                landsat_cog=landsat_cog,
                landsat_flag=landsat_flag,
                scaler=scaler,
                selection=selection,
                out_dir=args.out,
                provenance=provenance,
                prior_mode=args.prior_mode,
                bbox=bbox,
                device=device,
                batch_size=int(cfg.data.batch_size),
                mask_native_invalid=args.mask_native_invalid,
                tile_px=args.tile_px,
                stride_px=args.stride_px,
            )
            tif_path, _json_path = scene_product_paths(args.out, scene_id)
            written.append(tif_path)
            print(tif_path)
        if args.mosaic:
            for members in group_scenes_by_overpass(scene_ids).values():
                member_tifs = [scene_product_paths(args.out, scene_id)[0] for scene_id in members]
                tif_path, json_path = write_overpass_mosaic(member_tifs, args.out)
                print(tif_path)
                print(json_path)
        log_event(
            _logger,
            logging.INFO,
            "berlin_inference_done",
            scenes=len(scene_ids),
            products=len(written),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
