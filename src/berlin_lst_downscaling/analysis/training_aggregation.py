"""Arithmetic vs T4 prior aggregation analysis (issue #80, plan step 7)."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

TrainScope = Literal["full", "sample30"]
DomainLabel = Literal["primary_native_valid", "secondary_strict_patch"]
SplitLabel = Literal["train", "validation", "test"]
SummaryWeighting = Literal["block_weighted", "scene_weighted"]

STRICT_MIN_ELIGIBLE_CELLS = 244


@dataclass(frozen=True)
class SceneMeta:
    scene_id: str
    split: SplitLabel
    year: int
    scene_date: date


@dataclass
class BlockRecord:
    scene_id: str
    split: SplitLabel
    domain: DomainLabel
    block_id: str
    support_count: int
    arithmetic_k: float
    t4_k: float
    signed_diff_k: float
    sigma_k: float
    reference_t_k: float
    approx_diff_k: float
    approx_ratio: float | None
    strict_patch_block: bool


@dataclass
class SplitSummary:
    weighting: SummaryWeighting
    split: SplitLabel
    domain: DomainLabel
    signed_mean_diff_k: float
    mae_k: float
    p95_abs_diff_k: float
    block_count: int
    scene_count: int


@dataclass
class AnalysisResult:
    train_scope: TrainScope
    selection_policy: dict[str, Any]
    cost_estimate: dict[str, Any]
    incomplete: bool
    incomplete_reason: str | None
    blocks: list[BlockRecord] = field(default_factory=list)
    per_scene: dict[str, dict[str, Any]] = field(default_factory=dict)
    summaries: list[SplitSummary] = field(default_factory=list)
    prediction_10m_comparison: dict[str, Any] | None = None


def largest_remainder_allocation(weights: Sequence[float], total: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return []
    if total == 0:
        return [0] * len(weights)
    wsum = float(sum(weights))
    if wsum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    exact = [total * float(w) / wsum for w in weights]
    floors = [int(math.floor(x)) for x in exact]
    remainder = total - sum(floors)
    fractional = sorted(
        ((exact[i] - floors[i], i) for i in range(len(weights))),
        key=lambda t: (-t[0], t[1]),
    )
    for k in range(remainder):
        floors[fractional[k][1]] += 1
    return floors


def select_train_scenes(
    train_scenes: Sequence[SceneMeta],
    *,
    scope: TrainScope,
    seed: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    """Return train scene IDs and a persisted selection policy description."""
    if scope == "full":
        ids = sorted({s.scene_id for s in train_scenes})
        return ids, {"rule": "all_train_scenes", "seed": seed, "scene_ids": ids}
    strata: dict[tuple[int, int], list[SceneMeta]] = defaultdict(list)
    for scene in train_scenes:
        strata[(scene.year, scene.scene_date.month)].append(scene)
    keys = sorted(strata)
    if len(keys) > 30:
        rng = random.Random(seed)  # noqa: S311
        chosen_keys = sorted(rng.sample(keys, 30))
        picked: list[SceneMeta] = []
        for key in chosen_keys:
            candidates = sorted(strata[key], key=lambda s: (s.scene_date, s.scene_id))
            picked.append(rng.choice(candidates))
        ids = [s.scene_id for s in picked]
        return ids, {
            "rule": "seeded_sample_of_30_strata_one_scene_each_when_gt_30_strata",
            "seed": seed,
            "strata_count": len(keys),
            "selected_strata": [{"year": y, "month": m} for y, m in chosen_keys],
            "scene_ids": ids,
        }
    weights = [float(len(strata[k])) for k in keys]
    counts = largest_remainder_allocation(weights, 30)
    rng = random.Random(seed)  # noqa: S311
    selected: list[SceneMeta] = []
    for key, count in zip(keys, counts, strict=True):
        candidates = sorted(strata[key], key=lambda s: (s.scene_date, s.scene_id))
        if count >= len(candidates):
            selected.extend(candidates)
            continue
        pool = list(candidates)
        rng.shuffle(pool)
        selected.extend(sorted(pool[:count], key=lambda s: (s.scene_date, s.scene_id)))
    selected = sorted(selected, key=lambda s: (s.year, s.scene_date, s.scene_id))[:30]
    ids = [s.scene_id for s in selected]
    return ids, {
        "rule": "largest_remainder_year_month_with_date_tiebreak",
        "seed": seed,
        "strata": [{"year": k[0], "month": k[1], "weight": len(strata[k])} for k in keys],
        "allocation": dict(zip([f"{y}-{m:02d}" for y, m in keys], counts, strict=True)),
        "scene_ids": ids,
    }


def block_aggregation_pair(temps_k: np.ndarray) -> dict[str, Any]:
    """Compute arithmetic/T4 pair and sigma diagnostics for native-valid Kelvin cells."""
    values = np.asarray(temps_k, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("block has no native-valid temperatures")
    if np.any(values <= 0.0):
        raise ValueError("contributing Kelvin values must be > 0")
    arithmetic = float(values.mean())
    t4 = float(np.power(np.mean(np.power(values, 4)), 0.25))
    signed_diff = t4 - arithmetic
    sigma = float(values.std(ddof=0))
    approx = 1.5 * sigma**2 / arithmetic if arithmetic > 0 else float("nan")
    ratio: float | None
    if values.size <= 1 or sigma == 0.0:
        ratio = None
    else:
        ratio = signed_diff / approx if math.isfinite(approx) and approx != 0.0 else None
    return {
        "support_count": int(values.size),
        "arithmetic_k": arithmetic,
        "t4_k": t4,
        "signed_diff_k": signed_diff,
        "sigma_k": sigma,
        "reference_t_k": arithmetic,
        "approx_diff_k": approx,
        "approx_ratio": ratio,
    }


def summarize_records(
    records: Sequence[BlockRecord],
    *,
    split: SplitLabel,
    domain: DomainLabel,
    weighting: SummaryWeighting,
) -> SplitSummary:
    subset = [r for r in records if r.split == split and r.domain == domain]
    if not subset:
        return SplitSummary(
            weighting=weighting,
            split=split,
            domain=domain,
            signed_mean_diff_k=float("nan"),
            mae_k=float("nan"),
            p95_abs_diff_k=float("nan"),
            block_count=0,
            scene_count=0,
        )
    diffs = np.array([r.signed_diff_k for r in subset], dtype=np.float64)
    abs_diffs = np.abs(diffs)
    if weighting == "block_weighted":
        signed_mean = float(diffs.mean())
        mae = float(abs_diffs.mean())
        p95 = float(np.percentile(abs_diffs, 95))
    else:
        by_scene: dict[str, list[float]] = defaultdict(list)
        by_scene_abs: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            by_scene[row.scene_id].append(row.signed_diff_k)
            by_scene_abs[row.scene_id].append(abs(row.signed_diff_k))
        scene_signed = [float(np.mean(v)) for v in by_scene.values()]
        scene_mae = [float(np.mean(v)) for v in by_scene_abs.values()]
        signed_mean = float(np.mean(scene_signed))
        mae = float(np.mean(scene_mae))
        p95 = float(np.percentile(scene_mae, 95))
    return SplitSummary(
        weighting=weighting,
        split=split,
        domain=domain,
        signed_mean_diff_k=signed_mean,
        mae_k=mae,
        p95_abs_diff_k=p95,
        block_count=len(subset),
        scene_count=len({r.scene_id for r in subset}),
    )


def per_scene_breakdown(records: Sequence[BlockRecord]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[BlockRecord]] = defaultdict(list)
    for row in records:
        grouped[row.scene_id].append(row)
    for scene_id, rows in grouped.items():
        diffs = np.array([r.signed_diff_k for r in rows], dtype=np.float64)
        abs_diffs = np.abs(diffs)
        out[scene_id] = {
            "split": rows[0].split,
            "block_count": len(rows),
            "signed_mean_diff_k": float(diffs.mean()),
            "mae_k": float(abs_diffs.mean()),
            "p95_abs_diff_k": float(np.percentile(abs_diffs, 95)),
        }
    return out


def estimate_cost(
    *,
    train_scene_count: int,
    val_scene_count: int,
    test_scene_count: int,
    blocks_per_scene: float = 120.0,
    seconds_per_block: float = 0.35,
) -> dict[str, Any]:
    scenes = train_scene_count + val_scene_count + test_scene_count
    blocks = scenes * blocks_per_scene
    seconds = blocks * seconds_per_block
    return {
        "scenes": scenes,
        "train_scenes": train_scene_count,
        "validation_scenes": val_scene_count,
        "test_scenes": test_scene_count,
        "blocks_per_scene_assumed": blocks_per_scene,
        "seconds_per_block_assumed": seconds_per_block,
        "estimated_seconds": seconds,
        "estimated_vm_hours": seconds / 3600.0,
        "guideline_hours": 2.0,
        "basis": "synthetic defaults; real IO may differ",
    }


def run_synthetic_analysis(
    *,
    train_scope: TrainScope,
    runtime_seconds: float | None = None,
    strict_patch_blocks: Mapping[str, set[str]] | None = None,
) -> AnalysisResult:
    """Build a bounded synthetic analysis without reading GCS."""
    scenes = [
        SceneMeta("S2017", "train", 2017, date(2017, 6, 1)),
        SceneMeta("S2018", "train", 2018, date(2018, 7, 2)),
        SceneMeta("S2019", "train", 2019, date(2019, 8, 3)),
        SceneMeta("V2024", "validation", 2024, date(2024, 6, 4)),
        SceneMeta("T2025", "test", 2025, date(2025, 6, 5)),
    ]
    train_pool = [s for s in scenes if s.split == "train"]
    train_ids, policy = select_train_scenes(train_pool, scope=train_scope)
    selected = [s for s in scenes if s.split != "train" or s.scene_id in train_ids]
    cost = estimate_cost(
        train_scene_count=len(train_ids),
        val_scene_count=sum(1 for s in scenes if s.split == "validation"),
        test_scene_count=sum(1 for s in scenes if s.split == "test"),
    )
    records: list[BlockRecord] = []
    incomplete = False
    reason: str | None = None
    started = 0.0
    for scene in selected:
        for block_idx in range(3):
            if runtime_seconds is not None and (block_idx + started) * 0.01 > runtime_seconds:
                incomplete = True
                reason = "runtime bound reached"
                break
            temps = np.array([280.0 + block_idx, 320.0 - block_idx], dtype=np.float64)
            stats = block_aggregation_pair(temps)
            block_id = f"{scene.scene_id}:B{block_idx}"
            strict_ids = (
                strict_patch_blocks.get(scene.scene_id, set()) if strict_patch_blocks else set()
            )
            strict = block_id in strict_ids
            for domain in ("primary_native_valid", "secondary_strict_patch"):
                if domain == "secondary_strict_patch" and not strict:
                    continue
                records.append(
                    BlockRecord(
                        scene_id=scene.scene_id,
                        split=scene.split,
                        domain=domain,  # type: ignore[arg-type]
                        block_id=block_id,
                        strict_patch_block=strict,
                        **stats,
                    )
                )
        started += 3
    summaries: list[SplitSummary] = []
    for split in ("train", "validation", "test"):
        for domain in ("primary_native_valid", "secondary_strict_patch"):
            for weighting in ("block_weighted", "scene_weighted"):
                summaries.append(
                    summarize_records(records, split=split, domain=domain, weighting=weighting)  # type: ignore[arg-type]
                )
    return AnalysisResult(
        train_scope=train_scope,
        selection_policy=policy,
        cost_estimate=cost,
        incomplete=incomplete,
        incomplete_reason=reason,
        blocks=records,
        per_scene=per_scene_breakdown(records),
        summaries=summaries,
        prediction_10m_comparison=None,
    )


def write_outputs(result: AnalysisResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_scope": result.train_scope,
        "selection_policy": result.selection_policy,
        "cost_estimate": result.cost_estimate,
        "incomplete": result.incomplete,
        "incomplete_reason": result.incomplete_reason,
        "summaries": [asdict(s) for s in result.summaries],
        "per_scene": result.per_scene,
        "prediction_10m_comparison": result.prediction_10m_comparison,
        "conventions": {
            "sigma": "population ddof=0 over native-valid contributing cells",
            "reference_t": "arithmetic block mean (K)",
            "approximation": "1.5 * sigma**2 / T",
        },
    }
    summary_path = output_dir / "aggregation_analysis.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    import csv

    blocks_path = output_dir / "blocks.csv"
    with blocks_path.open("w", encoding="utf-8", newline="") as handle:
        if result.blocks:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(result.blocks[0]).keys()))
            writer.writeheader()
            for row in result.blocks:
                writer.writerow(asdict(row))
    return {"summary": str(summary_path), "blocks_csv": str(blocks_path), "logs": str(logs)}


def parse_scene_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    return date.fromisoformat(text[:10])


__all__ = [
    "AnalysisResult",
    "BlockRecord",
    "SceneMeta",
    "block_aggregation_pair",
    "estimate_cost",
    "largest_remainder_allocation",
    "per_scene_breakdown",
    "run_synthetic_analysis",
    "select_train_scenes",
    "summarize_records",
    "write_outputs",
]
