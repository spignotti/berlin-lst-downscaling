"""Build, pack and upload the versioned prebuilt patch cache for one config.

Runs the exact data path a training run uses (same reader, scaler, channel
selection and threaded cache build) on the operator machine, cuts the resulting
job-local cache into checksummed shards, and uploads them under
``<uri-root>/v<format>/<key>/``. The prebuilt manifest is uploaded last, so an
interrupted upload leaves no usable prefix. Only train and validation are
cached; the 2025 test split is closed and no flag opens it. Reads the published
patch index and rasters, so a ``gs://`` root needs the operator's Application
Default Credentials; not run in CI.

Usage
-----
    uv run python scripts/operators/build_prebuilt_patch_cache.py \\
        --config-name v6_l5_locked --local-dir "$HOME/berlin-lst-cache-build/l5" --local-only
    uv run python scripts/operators/build_prebuilt_patch_cache.py \\
        --config-name v6_l5_locked --local-dir "$HOME/berlin-lst-cache-build/l5"

``--dry-run`` still builds the cache locally (the key needs the admitted
patches) but neither packs nor uploads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io.gdal_gcs import ensure_gdal_gcs_auth
from berlin_lst_downscaling.modeling.guards import guard_modeling_config
from berlin_lst_downscaling.modeling.prebuilt_cache import (
    ALLOWED_SPLITS,
    DEFAULT_SHARD_MAX_BYTES,
    pack_prebuilt_cache,
    prebuilt_key,
    prebuilt_manifest_exists,
    prebuilt_prefix,
    upload_prebuilt_cache,
)
from berlin_lst_downscaling.modeling.run import (
    _resolve_cache_budget,
    _resolve_patch_cache,
    build_real_datamodule,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"


def _assert_test_split_open() -> None:
    """Always raises: the 2025 test split is closed and no guard can open it."""
    raise RuntimeError(
        "test split guard: closed (2025 test split stays closed; no guard can open it)"
    )


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _parse_splits(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="modeling config")
    parser.add_argument("--override", action="append", default=[], help="Hydra override")
    parser.add_argument("--splits", default="train,validation", help="must be train,validation")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="always refused: the test split cannot be cached",
    )
    parser.add_argument("--local-dir", required=True, help="local build directory (never gs://)")
    parser.add_argument("--local-only", action="store_true", help="build and pack, no upload")
    parser.add_argument("--dry-run", action="store_true", help="build locally, print the key only")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing prefix")
    parser.add_argument("--shard-max-bytes", type=int, default=DEFAULT_SHARD_MAX_BYTES)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--uri-root", default=None, help="default: data.patch_cache.uri_root")
    args = parser.parse_args(argv)
    if "://" in args.local_dir:
        parser.error("--local-dir must be a local path (never gs://)")

    splits = _parse_splits(args.splits)
    if "test" in splits and not args.allow_test_split:
        print("ERROR: test split (2025) is closed; refusing to build")
        return 2
    if "test" in splits:
        try:
            _assert_test_split_open()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 2
    if set(splits) != set(ALLOWED_SPLITS) or len(splits) != len(ALLOWED_SPLITS):
        print(f"ERROR: splits must be exactly {','.join(ALLOWED_SPLITS)}, got {args.splits!r}")
        return 2

    cfg = _compose(args.config_name, list(args.override))
    guard_modeling_config(cfg, args.config_name)
    # The data module reads cfg.data.splits, not --splits; refuse before any
    # source read so the 2025 test refs are never loaded or admitted.
    declared = tuple(str(s) for s in (cfg.data.get("splits") or ()))
    if declared != ALLOWED_SPLITS:
        print(f"ERROR: data.splits must be exactly {list(ALLOWED_SPLITS)}, got {list(declared)}")
        return 2
    uri_root = args.uri_root or str(_resolve_patch_cache(cfg)["uri_root"])
    if ensure_gdal_gcs_auth() is not None:
        print("GDAL GCS: using Application Default Credentials")

    local_dir = Path(args.local_dir).expanduser()
    cache_dir = local_dir / "cache"
    out_dir = local_dir / "prebuilt"
    local_dir.mkdir(parents=True, exist_ok=True)
    data_module = build_real_datamodule(
        cfg,
        cache_root=cache_dir,
        cache_max_bytes=_resolve_cache_budget(cfg),
        cache_source="job_local",
    )
    data_module.setup()
    components = data_module.prebuilt_components()
    prefix = prebuilt_prefix(uri_root, components)
    summary: dict[str, Any] = {
        "config_name": args.config_name,
        "prefix": prefix,
        "key": prebuilt_key(components),
        "components": components,
    }
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, sort_keys=True))
        return 0
    if not args.local_only and prebuilt_manifest_exists(prefix) and not args.overwrite:
        print(f"ERROR: prebuilt patch cache already exists at {prefix}; pass --overwrite")
        return 2

    manifest = pack_prebuilt_cache(
        cache_dir,
        out_dir,
        components=components,
        split_counts=data_module.stats()["patches_per_split"],  # type: ignore[arg-type]
        shard_max_bytes=args.shard_max_bytes,
        provenance_extra={"config_name": args.config_name, "overrides": list(args.override)},
    )
    summary.update(
        {
            "local_artifact": str(out_dir),
            "shards": len(manifest["shards"]),  # type: ignore[arg-type]
            "total_count": manifest["total_count"],
            "splits": manifest["splits"],
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.local_only:
        return 0
    upload_prebuilt_cache(
        out_dir, prefix, overwrite=args.overwrite, max_workers=args.upload_workers
    )
    print(f"uploaded {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
