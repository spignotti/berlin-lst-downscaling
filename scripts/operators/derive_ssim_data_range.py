"""Derive the fixed SSIM data range L from the train split only.

L = p99.5 - p0.5 of the valid 100 m targets of the train split (2017-2023).
Only train refs are requested from the patch index and every ref is checked to
be train; validation and test are never read. Reads the published patch index
and the Landsat rasters, so a ``gs://`` root needs the operator's Application
Default Credentials; not run in CI. Writes a local YAML artifact only (never
GCS).

Usage
-----
    uv run python scripts/operators/derive_ssim_data_range.py --config-name v6_l5_locked

The script writes the next free version file (``configs/modeling/ssim_data_range/
v<N>.yaml``) and never overwrites an existing one. To activate it, point
``SSIM_DATA_RANGE_ARTIFACT`` at the new file in a reviewed PR; until a filled,
matching artifact is active every production path raises.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.data.io.gdal_gcs import ensure_gdal_gcs_auth
from berlin_lst_downscaling.data.training.report import now_iso
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSourceConfig,
    load_patch_refs,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.run import real_source_config
from berlin_lst_downscaling.modeling.ssim_range import (
    SSIM_DATA_RANGE_DIR,
    SsimRangeSample,
    SsimRangeStats,
    artifact_payload,
    check_artifact_writable,
    derive_ssim_data_range,
    is_artifact_version,
    next_artifact_version,
    write_artifact,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"
DERIVATION_SPLITS: tuple[str, ...] = ("train",)
assert DERIVATION_SPLITS == ("train",)  # noqa: S101


def derive_from_source(
    source: RealSourceConfig,
    *,
    reader_factory: Callable[[RealSourceConfig], Any] = RealPatchReader,
    refs_loader: Callable[..., list[PatchRef]] = load_patch_refs,
    fingerprints_loader: Callable[[str], Mapping[str, str]] = patch_index_fingerprints,
) -> tuple[SsimRangeStats, dict[str, int], int, dict[str, str]]:
    """Return ``(stats, exclusions, n_refs, fingerprints)`` from train refs only."""
    refs = list(refs_loader(source, splits=DERIVATION_SPLITS))
    for ref in refs:
        if ref.split != "train":
            raise RuntimeError(
                f"non-train ref {ref.patch_id!r} (split={ref.split!r}) in derivation"
            )
    reader = reader_factory(source)
    reader.preload({ref.scene_id for ref in refs})

    def samples() -> Iterator[SsimRangeSample]:
        for ref in refs:
            pair = reader.read_target_and_mask(ref)
            if pair is None:
                continue
            target, mask = pair
            yield SsimRangeSample(
                patch_id=ref.patch_id,
                scene_id=ref.scene_id,
                split=ref.split,
                year=ref.year,
                target_100m=target,
                mask_100m=mask,
            )

    stats = derive_ssim_data_range(samples())
    fingerprints = dict(fingerprints_loader(source.patch_index_root))
    return stats, dict(reader.exclusions), len(refs), fingerprints


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="v6_l5_locked", help="modeling config")
    parser.add_argument(
        "--out",
        default=None,
        help="new local version file v<N>.yaml (default: next free version in the config dir)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        out_path = SSIM_DATA_RANGE_DIR / f"{next_artifact_version(SSIM_DATA_RANGE_DIR)}.yaml"
    else:
        if "://" in args.out:
            parser.error("--out must be a local path (never gs://)")
        out_path = Path(args.out)
        if not is_artifact_version(out_path.stem):
            parser.error("--out file name must be v<N>.yaml")
    check_artifact_writable(out_path)
    version = out_path.stem

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name)
    source = real_source_config(cfg)
    if ensure_gdal_gcs_auth() is not None:
        print("GDAL GCS: using Application Default Credentials")
    stats, exclusions, n_refs, fingerprints = derive_from_source(source)
    payload = artifact_payload(
        stats,
        patch_index_root=source.patch_index_root,
        training_root=source.training_root,
        fingerprints=fingerprints,
        config_name=args.config_name,
        git_commit=_git_commit(),
        derived_at=now_iso(),
        exclusions=exclusions,
        n_refs_requested=n_refs,
        version=version,
    )
    out = write_artifact(payload, out_path)
    print(f"wrote {out}")
    print(
        f"L = {stats.data_range_k:.4f} K (p0.5 {stats.low_k:.4f}, p99.5 {stats.high_k:.4f}); "
        f"{stats.n_cells} cells, {stats.n_patches} patches, {stats.n_scenes} scenes, "
        f"{n_refs} refs requested, exclusions {exclusions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
