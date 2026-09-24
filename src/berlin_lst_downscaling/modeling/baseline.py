"""Naive prior-expand baseline (WB3, issue #19).

A no-neural-network comparison anchor for Stage 1. The prediction is the
1000 m block-expanded native LST prior — the same tensor the model receives
as ``lst_prior`` — pooled back to 100 m with the *same* exact 10x10 pooling
the model metrics use. Because each 100 m cell lies inside one 1000 m block,
the pooled baseline equals that block's mean, which makes the reported score
auditable rather than merely plausible.

Both arms therefore share the patch reader, the admitted patch universe, the
eligibility mask, the pooling, and the metrics, so the scores are directly
comparable. Random Forest and the optional bilinear variant stay out of scope.

``# decision:`` the baseline reads through the full :class:`RealPatchReader`,
including the 28 feature bands it does not predict from. A feature-skipping
mode would avoid that read but add a second code path to the boundary module
the model also depends on; the same-reader guarantee is worth more here than
the bandwidth saved on a path this build does not execute.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from berlin_lst_downscaling.data.io import log_event, run_context_path
from berlin_lst_downscaling.data.training.report import now_iso
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
    load_patch_refs,
    patch_index_fingerprints,
)

_logger = logging.getLogger(__name__)

METHOD = "naive_prior_expand"
PRIOR_RULE = (
    "native 100 m Landsat LST mean-pooled 10x10 to 1000 m over native-valid "
    "cells, block-expanded to the 10 m grid on the global canonical lattice, "
    "then the exact nested 10x10 pool back to 100 m (= the 1000 m block mean)"
)


@dataclass
class PatchRecord:
    """One evaluated patch's contribution to the aggregate."""

    patch_id: str
    split: str
    valid_cells: int
    abs_error_sum: float
    ssim_sum: float
    ssim_windows: int
    filled_feature_pixels: int


@dataclass
class SplitSummary:
    """Aggregate metrics for one split."""

    requested_patches: int
    evaluated_patches: int
    valid_cells: int
    abs_error_sum: float
    ssim_sum: float
    ssim_windows: int
    filled_feature_pixels: int
    exclusions: dict[str, int] = field(default_factory=dict)

    @property
    def mae(self) -> float | None:
        return None if self.valid_cells == 0 else self.abs_error_sum / self.valid_cells

    @property
    def ssim(self) -> float | None:
        return None if self.ssim_windows == 0 else self.ssim_sum / self.ssim_windows


@dataclass
class BaselineReport:
    """The writable baseline artifact for one invocation."""

    method: str
    prior_rule: str
    primary_metric: str
    secondary_metric: str
    splits: dict[str, SplitSummary]
    patches: list[PatchRecord]
    exclusions: dict[str, int]
    source: dict[str, str]
    fingerprints: dict[str, str]
    git_revision: str
    written_at: str

    def to_payload(self) -> dict:
        return {
            "method": self.method,
            "prior_rule": self.prior_rule,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "splits": {
                name: {**asdict(summary), "mae": summary.mae, "ssim": summary.ssim}
                for name, summary in self.splits.items()
            },
            "patches": [asdict(record) for record in self.patches],
            "exclusions": self.exclusions,
            "source": self.source,
            "fingerprints": self.fingerprints,
            "git_revision": self.git_revision,
            "written_at": self.written_at,
        }


def _counter_delta(before: Counter[str], after: Counter[str]) -> dict[str, int]:
    """Return the exclusions added between two counter snapshots."""
    return {key: after[key] - before.get(key, 0) for key in sorted(after)}


def git_revision_for(output_root: str, run_id: str) -> str:
    """Read the Git revision the runner recorded for this baseline run.

    Mirrors ``modeling/run.py``'s helper locally so the baseline does not
    import the Lightning lifecycle module just for one field.
    """
    uri = run_context_path(output_root, "baseline", run_id)
    if not Path(uri).is_file():
        return "unknown"
    with open(uri, encoding="utf-8") as fh:
        return str(json.load(fh).get("git_commit", "unknown"))


def evaluate_baseline(
    reader: RealPatchReader,
    refs: list[PatchRef],
    *,
    requested_per_split: dict[str, int] | None = None,
) -> tuple[list[PatchRecord], dict[str, SplitSummary]]:
    """Score the prior-expand baseline over ``refs``, split by split.

    ``requested_per_split`` records how many indexed rows each split asked
    for, so a full run can show ``evaluated + exclusions == requested``.
    """
    records: list[PatchRecord] = []
    summaries: dict[str, SplitSummary] = {}
    by_split: dict[str, list[PatchRef]] = {}
    for ref in refs:
        by_split.setdefault(ref.split, []).append(ref)

    for split in sorted(by_split):
        split_refs = by_split[split]
        before = Counter(reader.exclusions)
        valid_cells = 0
        abs_error_sum = 0.0
        ssim_sum = 0.0
        ssim_windows = 0
        filled_total = 0
        evaluated = 0
        for ref in split_refs:
            sample = reader.read_patch(ref)
            if sample is None:
                continue
            record = _score_patch(sample)
            records.append(record)
            valid_cells += record.valid_cells
            abs_error_sum += record.abs_error_sum
            ssim_sum += record.ssim_sum
            ssim_windows += record.ssim_windows
            filled_total += record.filled_feature_pixels
            evaluated += 1
        requested = (
            requested_per_split.get(split, len(split_refs))
            if requested_per_split
            else len(split_refs)
        )
        summaries[split] = SplitSummary(
            requested_patches=requested,
            evaluated_patches=evaluated,
            valid_cells=valid_cells,
            abs_error_sum=abs_error_sum,
            ssim_sum=ssim_sum,
            ssim_windows=ssim_windows,
            filled_feature_pixels=filled_total,
            exclusions=_counter_delta(before, Counter(reader.exclusions)),
        )
        log_event(
            _logger,
            logging.INFO,
            "baseline_split",
            split=split,
            requested=requested,
            evaluated=evaluated,
            valid_cells=valid_cells,
            mae=summaries[split].mae,
        )
    return records, summaries


def _score_patch(sample: RealSample) -> PatchRecord:
    """Score one patch with the shared pooling and metric path."""
    prior_10m = torch.from_numpy(sample.lst_prior_k[None])  # (1, 1, H, W)
    target = torch.from_numpy(sample.target_100m[None])
    mask = torch.from_numpy(sample.mask_100m[None])
    prediction_100m = pool_10m_to_100m(prior_10m)
    error_sum, count = masked_abs_error_sums(prediction_100m, target, mask)
    ssim_sum, ssim_count = masked_ssim_stats(prediction_100m, target, mask)
    return PatchRecord(
        patch_id=sample.meta.patch_id,
        split=sample.meta.split,
        valid_cells=int(count),
        abs_error_sum=float(error_sum),
        ssim_sum=float(ssim_sum),
        ssim_windows=int(ssim_count),
        filled_feature_pixels=sample.meta.filled_feature_pixels,
    )


def build_report(
    *,
    cfg: RealSourceConfig,
    records: list[PatchRecord],
    summaries: dict[str, SplitSummary],
    exclusions: dict[str, int],
    git_revision: str,
) -> BaselineReport:
    """Assemble the writable artifact from the evaluated records."""
    return BaselineReport(
        method=METHOD,
        prior_rule=PRIOR_RULE,
        primary_metric="masked MAE @ 100 m (cell-weighted over valid cells)",
        secondary_metric=(
            "SSIM @ 100 m, 7x7 uniform window, fully-valid windows only, "
            "logged only (never a selection gate)"
        ),
        splits=summaries,
        patches=records,
        exclusions=exclusions,
        source={
            "patch_index_root": cfg.patch_index_root,
            "training_root": cfg.training_root,
            "features_root": cfg.features_root,
            "ard_root": cfg.ard_root,
        },
        fingerprints=patch_index_fingerprints(cfg.patch_index_root),
        git_revision=git_revision,
        written_at=now_iso(),
    )


def write_report(report: BaselineReport, uri: str) -> str:
    """Write the baseline artifact to a run-output path (never a release root)."""
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_payload(), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_refs(
    cfg: RealSourceConfig,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    scene_ids: tuple[str, ...] | None = None,
    max_patches_per_split: int | None = None,
) -> list[PatchRef]:
    """Return the patch refs the baseline should evaluate.

    ``max_patches_per_split`` bounds each split to its first N refs in the
    index's deterministic order, matching how the model's data module selects
    a bounded smoke subset — so both arms cover identical patch IDs.
    """
    refs = load_patch_refs(cfg, splits=splits, scene_ids=scene_ids)
    if max_patches_per_split is None:
        return refs
    counts: Counter[str] = Counter()
    bounded: list[PatchRef] = []
    for ref in refs:
        if counts[ref.split] >= max_patches_per_split:
            continue
        counts[ref.split] += 1
        bounded.append(ref)
    return bounded


__all__ = [
    "METHOD",
    "PRIOR_RULE",
    "BaselineReport",
    "PatchRecord",
    "SplitSummary",
    "build_report",
    "evaluate_baseline",
    "git_revision_for",
    "load_refs",
    "write_report",
]
