# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""WB3 naive prior-expand baseline runner (Hydra-driven, issue #19).

Scores the 1000 m block-expanded prior against the native 100 m target on the
same admitted patch universe, mask, pooling, and metrics as the training
loop, and writes one ``baseline_report.json`` under the configured run output
root. Nothing is written to GCS and no canonical artifact is touched.

Usage
-----
    # Smoke: a bounded subset, local ephemeral output
    uv run python scripts/runners/run_baseline.py --config-name baseline_smoke

    # Full: every indexed patch of the requested splits; explicit invocation
    uv run python scripts/runners/run_baseline.py --config-name baseline_full

Requires ADC for the read-only GCS sources. Exits non-zero when a split's
accounting does not close (evaluated + exclusions != requested) or the
artifact cannot be written.
"""

from __future__ import annotations

import logging
from collections import Counter
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.modeling.baseline import (
    build_report,
    evaluate_baseline,
    git_revision_for,
    load_refs,
    write_report,
)
from berlin_lst_downscaling.modeling.patches import RealPatchReader, RealSourceConfig

_logger = logging.getLogger(__name__)


def _source_config(cfg: DictConfig) -> RealSourceConfig:
    """Build the published-source config from the resolved Hydra config."""
    required = ("patch_index_root", "training_root", "features_root", "ard_root")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"baseline requires {missing} in the config")
    return RealSourceConfig(
        patch_index_root=str(cfg.patch_index_root),
        training_root=str(cfg.training_root),
        features_root=str(cfg.features_root),
        ard_root=str(cfg.ard_root),
    )


@hydra.main(config_path="../../configs/modeling", config_name="baseline_full", version_base=None)
def main(cfg: DictConfig) -> int:
    """Evaluate the naive baseline and persist its artifact and run log."""
    run_id = uuid4().hex[:8]
    output_root = str(cfg.output_root)
    level = getattr(logging, str(cfg.get("logging_level", "INFO")).upper(), logging.INFO)
    splits = tuple(str(s) for s in (cfg.get("splits") or ["train", "validation", "test"]))
    scene_ids = tuple(str(s) for s in (cfg.get("scene_ids") or [])) or None
    max_patches = cfg.get("max_patches_per_split")

    with RunLogSession(output_root, pipeline="baseline", run_id=run_id, level=level):
        log_event(
            _logger,
            logging.INFO,
            "config",
            run_id=run_id,
            output_root=output_root,
            config_name=cfg.get("_hydra", {}).get("job", {}).get("config_name", "baseline_full"),
            splits=list(splits),
            scene_ids=list(scene_ids or []),
            max_patches_per_split=max_patches,
        )
        source = _source_config(cfg)
        refs = load_refs(
            source,
            splits=splits,
            scene_ids=scene_ids,
            max_patches_per_split=None if max_patches is None else int(max_patches),
        )
        reader = RealPatchReader(source)
        requested = Counter(ref.split for ref in refs)
        records, summaries = evaluate_baseline(
            reader, refs, requested_per_split=dict(requested)
        )

        report = build_report(
            cfg=source,
            records=records,
            summaries=summaries,
            exclusions=dict(reader.exclusions),
            git_revision=git_revision_for(output_root, run_id),
        )
        uri = write_report(
            report, f"{output_root.rstrip('/')}/baseline_report.json"
        )

        print(f"Baseline — run {run_id} ({report.method})")
        for split in sorted(summaries):
            summary = summaries[split]
            mae = "n/a" if summary.mae is None else f"{summary.mae:.4f}"
            ssim = "n/a" if summary.ssim is None else f"{summary.ssim:.4f}"
            print(
                f"  {split:<11} patches {summary.evaluated_patches}/"
                f"{summary.requested_patches} | cells {summary.valid_cells} | "
                f"MAE {mae} K | SSIM {ssim} ({summary.ssim_windows} windows)"
            )
        for reason, count in sorted(reader.exclusions.items()):
            print(f"  Excluded [{reason}]: {count}")
        print(f"  Report: {uri}")

        # Every indexed row of a requested split must be evaluated or excluded.
        for split, summary in summaries.items():
            accounted = summary.evaluated_patches + sum(summary.exclusions.values())
            if accounted != summary.requested_patches:
                print(
                    f"  FAIL: {split} accounting {accounted} != requested "
                    f"{summary.requested_patches}"
                )
                raise SystemExit(1)
        if not any(summary.valid_cells > 0 for summary in summaries.values()):
            print("  FAIL: no valid cell evaluated in any split")
            raise SystemExit(1)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
