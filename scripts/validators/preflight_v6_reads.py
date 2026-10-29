"""Read-only preflight for a v6 modeling config against the published v3 chain.

Composes ``configs/modeling/<config-name>`` and runs the reads a v6 run starts
with: ``load_scaler``, the scaler/config check (``assert_scaler_matches_config``
for a v6 arm, one check per stage for the random forest, the full v6 roster for
the naive baseline), and ``load_patch_refs`` for the train and validation splits
only. The test split is never requested and no patch raster is read. Nothing is
written. Needs the operator's Application Default Credentials for
``gs://`` roots; not run in CI.

Usage
-----
    uv run python scripts/validators/preflight_v6_reads.py \
        --config-name random_forest_v6_smoke --limit 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.modeling.channels import (
    ActiveChannelSelection,
    schema_version_from_config,
)
from berlin_lst_downscaling.modeling.patches import (
    ScalerSpec,
    assert_scaler_matches_config,
    assert_scaler_spec_matches,
    load_patch_refs,
    load_scaler,
    patch_index_fingerprints,
)
from berlin_lst_downscaling.modeling.random_forest import resolve_stage_selection
from berlin_lst_downscaling.modeling.run import real_source_config

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "modeling"
PREFLIGHT_SPLITS: tuple[str, ...] = ("train", "validation")


def check_scaler_against_config(cfg: DictConfig, scaler: ScalerSpec) -> None:
    """Check the scaler against the config's channel selection.

    The naive baseline feeds no feature channels; it gets the full v6 roster
    check, which is stricter than its runner.
    """
    if cfg.get("ablation_release") == "v6":
        assert_scaler_matches_config(cfg, scaler)
        return
    version = schema_version_from_config(cfg)
    if cfg.get("stages"):
        selections = []
        for raw in cfg.stages:
            stage = resolve_stage_selection(int(raw), version)
            selections.append(ActiveChannelSelection(names=stage.names, indices=stage.indices))
    else:
        names = feature_schema(version).names
        selections = [ActiveChannelSelection(names=names, indices=tuple(range(len(names))))]
    for selection in selections:
        assert_scaler_spec_matches(scaler, schema_version=version, selection=selection)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="modeling config, e.g. v6_l4_locked")
    parser.add_argument("--limit", type=int, default=8, help="patch refs to read per split")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name)
    source = real_source_config(cfg)

    scaler = load_scaler(source.training_root)
    if scaler.channel_order != feature_schema(FEATURE_SCHEMA_VERSION_V6).names:
        raise RuntimeError(
            f"{args.config_name}: training root {source.training_root!r} is not a v6 release "
            f"({len(scaler.channel_order)} channels)"
        )
    check_scaler_against_config(cfg, scaler)
    print(f"scaler      : {len(scaler.channels)} channels, policy_hash={scaler.policy_hash}")

    fingerprints = patch_index_fingerprints(source.patch_index_root)
    print(f"patch index : source_policy_hash={fingerprints['source_policy_hash']}")
    for split in PREFLIGHT_SPLITS:
        refs = load_patch_refs(source, splits=(split,), limit=args.limit)
        if not refs:
            raise RuntimeError(f"{args.config_name}: no {split} patch refs in the index")
        print(f"{split:<11} : {len(refs)} refs read (limit {args.limit})")
    print(f"PASS {args.config_name}: v6 reads agree with the v3 chain (test split not read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
