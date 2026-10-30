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
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import hydra
from omegaconf import DictConfig

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6
from berlin_lst_downscaling.data.io import RunLogSession, log_event
from berlin_lst_downscaling.modeling.aggregation import resolve_aggregation_method
from berlin_lst_downscaling.modeling.baseline import (
    METHOD,
    build_report,
    evaluate_baseline,
    git_revision_for,
    load_refs,
    resolve_report_ssim_data_range,
    write_report,
)
from berlin_lst_downscaling.modeling.channels import (
    resolve_schema_selection,
    schema_version_from_config,
)
from berlin_lst_downscaling.modeling.evaluation_exports import (
    EvaluationIdentity,
    patch_export_filename,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.patches import (
    RealPatchReader,
    RealSourceConfig,
    assert_scaler_spec_matches,
)
from berlin_lst_downscaling.modeling.ssim_range import SsimDataRange

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
        aggregation=resolve_aggregation_method(cfg.get("aggregation")),
    )


def _resolve_source_and_ssim(cfg: DictConfig) -> tuple[RealSourceConfig, SsimDataRange]:
    """Build the source and load the SSIM data range once, before any data read."""
    source = _source_config(cfg)
    return source, resolve_report_ssim_data_range(source)


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
        source, ssim_range = _resolve_source_and_ssim(cfg)
        log_event(
            _logger,
            logging.INFO,
            "ssim_data_range",
            data_range_k=ssim_range.data_range_k,
            version=ssim_range.version,
            artifact_sha256=ssim_range.artifact_sha256,
        )
        refs = load_refs(
            source,
            splits=splits,
            scene_ids=scene_ids,
            max_patches_per_split=None if max_patches is None else int(max_patches),
        )
        reader = RealPatchReader(source)
        schema_version = schema_version_from_config(cfg)
        if schema_version == FEATURE_SCHEMA_VERSION_V6:
            assert_scaler_spec_matches(
                reader.scaler,
                schema_version=schema_version,
                selection=resolve_schema_selection(schema_version, n_active_channels=None),
            )
        requested = Counter(ref.split for ref in refs)
        records, summaries = evaluate_baseline(
            reader, refs, ssim_range=ssim_range, requested_per_split=dict(requested)
        )

        report = build_report(
            cfg=source,
            records=records,
            summaries=summaries,
            exclusions=dict(reader.exclusions),
            git_revision=git_revision_for(output_root, run_id),
            ssim_range=ssim_range,
        )
        uri = write_report(report, f"{output_root.rstrip('/')}/baseline_report.json")

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

        # Per-patch sidecar for the splits this run already evaluated; no new reads.
        seed_raw = cfg.get("seed")
        identity = EvaluationIdentity(
            arm=METHOD,
            seed=None if seed_raw is None else int(seed_raw),
            run_id=run_id,
            checkpoint=None,
        )
        for split in ("validation", "test"):
            if split not in summaries or summaries[split].evaluated_patches == 0:
                continue
            export_path = write_patch_export(
                [asdict(r) for r in records if r.split == split],
                Path(output_root) / patch_export_filename(split),
                identity=identity,
                split=split,
            )
            print(f"  Patch export [{split}]: {export_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
