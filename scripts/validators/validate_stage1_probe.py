"""Independently validate a retained Stage-1 probe evidence record (issue #45).

Reconciles the probe's admitted cohort and per-epoch metrics against the
published comparison universe and a **same-cohort** naive baseline, then applies
the predeclared go/no-go screen. It re-derives the decision from the retained
numbers; it never trusts the run's own success flag.

Inputs are local files:

- the probe evidence JSON uploaded by ``scripts/operators/vertex_evidence.py``
  (``profile="probe"``);
- the committed naive-baseline report
  ``docs/results/baseline-full-<run>/baseline_report.json``.

Exit codes: 0 = go, 2 = valid evidence but no-go, 1 = evidence missing,
malformed, or incomplete (cannot decide). ``print()`` is the human summary.

Usage:
    uv run python scripts/validators/validate_stage1_probe.py \
        --evidence path/to/evidence.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.patches import (
    RealSourceConfig,
    load_patch_refs,
)
from berlin_lst_downscaling.modeling.run import assert_probe_minima, assert_stage1_probe

# Predeclared go/no-go screen. These are screening thresholds for whether the
# frozen method learns enough to justify the full run — not model-quality
# claims. The full-validation 1.61010 K anchor is not comparable to a subset, so
# the naive comparator below is computed over the probe's exact admitted cohort.
EXPECTED_EPOCHS = 6
TRAIN_IMPROVEMENT = 0.05  # final train MAE at least 5% below epoch one
VAL_IMPROVEMENT = 0.05  # best val MAE at least 5% below epoch one
VAL_REGRESSION_CAP = 1.25  # final val MAE at most 1.25x the best val MAE
NAIVE_FACTOR_CAP = 10.0  # best val MAE at most 10x the same-cohort naive MAE
RECHECK_TOLERANCE = 1e-3

DEFAULT_BASELINE = (
    "docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json"
)

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")


def _guard_self_check() -> int:
    """Exercise the fail-closed probe guards on the shipped config and on drift.

    Positive and negative cases, composed locally with no source read and no
    submission. This is the "positive and deliberately invalid config probe"
    the plan requires before paid work; it is separate from evidence validation.
    """
    def compose_probe(overrides: list[str]):
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            return compose(config_name="stage1_probe", overrides=overrides)

    guard_cases = [
        ("shipped stage1_probe", [], True),
        ("changed LR", ["trainer.learning_rate=0.01"], False),
        ("removed marker", ["stage1_probe=false"], False),
        ("unbounded cohort", ["data.probe.max_refs_per_split.train=1000000"], False),
        ("test split included", ['data.splits=["train","validation","test"]'], False),
        ("gcs output", ["output_root=gs://bucket/runs/x"], False),
        ("epochs drift", ["trainer.max_epochs=20"], False),
        ("depth drift", ["model.depth=3"], False),
    ]
    failures: list[str] = []
    for label, overrides, should_pass in guard_cases:
        try:
            assert_stage1_probe(compose_probe(overrides))
            accepted = True
        except Exception:
            accepted = False
        status = "PASS" if accepted == should_pass else "FAIL"
        if accepted != should_pass:
            failures.append(label)
        print(f"  {status} guard: {label} (accepted={accepted}, expected={should_pass})")

    cfg = compose_probe([])
    good_scope = {
        "patches_per_split": {"train": 128, "validation": 64},
        "scenes_per_split": {"train": list(range(16)), "validation": list(range(8))},
        "years_per_split": {"train": [2017, 2018, 2019], "validation": [2024]},
    }
    minima_cases = [
        ("cohort at minima", good_scope, True),
        (
            "cohort under-admitted",
            {**good_scope, "patches_per_split": {"train": 100, "validation": 64}},
            False,
        ),
        (
            "cohort too few scenes",
            {
                **good_scope,
                "scenes_per_split": {
                    "train": list(range(15)),
                    "validation": list(range(8)),
                },
            },
            False,
        ),
        (
            "cohort too few years",
            {**good_scope, "years_per_split": {"train": [2017, 2018], "validation": [2024]}},
            False,
        ),
        (
            "cohort includes test",
            {**good_scope, "patches_per_split": {"train": 128, "validation": 64, "test": 1}},
            False,
        ),
    ]
    for label, scope, should_pass in minima_cases:
        try:
            assert_probe_minima(cfg, scope)
            accepted = True
        except Exception:
            accepted = False
        status = "PASS" if accepted == should_pass else "FAIL"
        if accepted != should_pass:
            failures.append(label)
        print(f"  {status} minima: {label} (accepted={accepted}, expected={should_pass})")

    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK OK: probe guards accept the shipped config and reject drift")
    return 0


def _load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _scene_id(patch_id: str) -> str:
    return patch_id.split(":", 1)[0]


def _scene_year(scene_id: str) -> int | None:
    parts = scene_id.split("_")
    if len(parts) < 4 or len(parts[3]) < 4 or not parts[3][:4].isdigit():
        return None
    return int(parts[3][:4])


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _baseline_index(report: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in report.get("patches", []):
        index[str(entry["patch_id"])] = entry
    return index


def _published_index(resolved: dict) -> dict[str, str]:
    """Patch id -> split, read independently from the published patch index.

    Reconciles train *and* held-out IDs; the naive-baseline report only scores
    the validation and test splits.
    """
    source = RealSourceConfig(
        patch_index_root=str(resolved["patch_index_root"]),
        training_root=str(resolved["training_root"]),
        features_root=str(resolved["features_root"]),
        ard_root=str(resolved["ard_root"]),
    )
    refs = load_patch_refs(source)
    return {ref.patch_id: ref.split for ref in refs}


def _cohort_naive_mae(
    index: dict[str, dict], patch_ids: list[str], split: str
) -> tuple[float, int]:
    error_sum = 0.0
    cells = 0.0
    matched = 0
    for patch_id in patch_ids:
        entry = index.get(patch_id)
        if entry is None or str(entry.get("split")) != split:
            continue
        matched += 1
        error_sum += float(entry["abs_error_sum"])
        cells += float(entry["valid_cells"])
    if cells <= 0.0:
        raise ValueError(f"no valid cells for the {split} cohort in the baseline report")
    return error_sum / cells, matched


def validate(
    evidence: dict, baseline: dict, published: dict[str, str]
) -> tuple[bool, list[str], list[str]]:
    """Return ``(decidable, failures, notes)``.

    ``decidable`` is False when the evidence is structurally incomplete: the
    result is inconclusive, never a pass. ``published`` maps every published
    patch id to its split and is the authoritative reconciliation universe.
    """
    failures: list[str] = []
    notes: list[str] = []

    if evidence.get("profile") != "probe":
        return (
            False,
            [
                "evidence profile is not 'probe' (this validates the historical #45 "
                "probe; use validate_stage1_recovery.py for 'probe-residual')"
            ],
            notes,
        )
    scope = evidence.get("data_scope")
    probe = evidence.get("probe")
    if not isinstance(scope, dict) or not isinstance(probe, dict):
        return False, ["evidence is missing data_scope/probe blocks"], notes

    epochs = probe.get("epoch_metrics")
    summary = probe.get("summary")
    if not isinstance(epochs, list) or not epochs:
        return False, ["no epoch_metrics retained (run incomplete or evidence lost)"], notes
    if not isinstance(summary, dict) or not summary:
        return False, ["no probe summary retained"], notes

    if len(epochs) != EXPECTED_EPOCHS:
        return False, [f"retained {len(epochs)} epochs, expected {EXPECTED_EPOCHS}"], notes
    if probe.get("epochs_complete") is not True:
        failures.append("evidence reports epochs_complete=false")

    # ── cohort structure ──────────────────────────────────────────────
    patch_ids = scope.get("patch_ids")
    admitted = scope.get("patches_per_split")
    scenes = scope.get("scenes_per_split")
    if not isinstance(patch_ids, dict) or not isinstance(admitted, dict):
        return False, ["data_scope is missing patch_ids/patches_per_split"], notes
    if sorted(patch_ids) != ["train", "validation"]:
        failures.append(f"admitted splits {sorted(patch_ids)} are not train/validation only")

    resolved = summary.get("resolved_config")
    probe_cfg = (
        resolved.get("data", {}).get("probe") if isinstance(resolved, dict) else None
    )
    if not isinstance(probe_cfg, dict):
        return False, ["summary.resolved_config has no data.probe block"], notes
    expected_admitted = probe_cfg.get("min_admitted_per_split", {})
    expected_scenes = probe_cfg.get("min_scenes_per_split", {})
    expected_years = int(probe_cfg.get("min_train_years", 0))
    per_scene_cap = int(probe_cfg.get("max_refs_per_scene", 0))

    for split in ("train", "validation"):
        ids = [str(p) for p in patch_ids.get(split, [])]
        n_admitted = int(admitted.get(split, 0))
        if n_admitted != len(ids):
            failures.append(
                f"{split}: admitted count {n_admitted} != patch_ids length {len(ids)}"
            )
        need = int(expected_admitted.get(split, 0))
        if n_admitted < need:
            failures.append(f"{split}: admitted {n_admitted} < required {need}")
        if isinstance(scenes, dict):
            got_scenes = len(scenes.get(split, []))
            need_scenes = int(expected_scenes.get(split, 0))
            if got_scenes < need_scenes:
                failures.append(f"{split}: {got_scenes} scenes < required {need_scenes}")
        derived_scenes: dict[str, int] = {}
        for patch_id in ids:
            derived_scenes[_scene_id(patch_id)] = derived_scenes.get(_scene_id(patch_id), 0) + 1
        over_cap = {s: c for s, c in derived_scenes.items() if c > per_scene_cap}
        if over_cap:
            failures.append(
                f"{split}: scene(s) exceed the per-scene cap {per_scene_cap}: {over_cap}"
            )

    train_years = {
        y for y in (_scene_year(_scene_id(str(p))) for p in patch_ids.get("train", []))
        if y is not None
    }
    if len(train_years) < expected_years:
        failures.append(
            f"train covers {len(train_years)} years < required {expected_years}"
        )

    # ── independent reconciliation against the published index ────────
    baseline_index = _baseline_index(baseline)
    for split in ("train", "validation"):
        ids = [str(p) for p in patch_ids.get(split, [])]
        misattributed = [p for p in ids if published.get(p) != split]
        if misattributed:
            failures.append(
                f"{split}: {len(misattributed)} admitted patch(es) not in the published "
                f"{split} split, e.g. {misattributed[:3]}"
            )
        test_ids = [p for p in ids if published.get(p) == "test"]
        if test_ids:
            failures.append(f"{split}: test-split patch(es) present in the cohort: {test_ids[:3]}")

    # ── per-epoch metrics ─────────────────────────────────────────────
    train_maes: list[float] = []
    val_maes: list[float] = []
    for entry in epochs:
        train = entry.get("train_mae_100m")
        val = entry.get("validation_mae_100m")
        if not _finite(train) or not _finite(val):
            failures.append(f"epoch {entry.get('epoch')}: non-finite train/val MAE")
            continue
        train_maes.append(float(train))
        val_maes.append(float(val))

    if len(train_maes) != EXPECTED_EPOCHS:
        failures.append("epoch curve has non-finite or missing values; cannot screen")
        return False, failures, notes

    first_train, last_train = train_maes[0], train_maes[-1]
    first_val, last_val = val_maes[0], val_maes[-1]
    best_val = min(val_maes)
    best_epoch = val_maes.index(best_val) + 1

    notes.append(f"train MAE: {first_train:.4f} -> {last_train:.4f}")
    notes.append(
        f"val MAE:   {first_val:.4f} -> {last_val:.4f} "
        f"(best {best_val:.4f} @ epoch {best_epoch})"
    )

    if last_train > (1.0 - TRAIN_IMPROVEMENT) * first_train:
        failures.append(
            f"train MAE did not improve by {TRAIN_IMPROVEMENT:.0%} "
            f"({first_train:.4f} -> {last_train:.4f}): flat or diverging"
        )
    if best_val > (1.0 - VAL_IMPROVEMENT) * first_val:
        failures.append(
            f"best val MAE did not improve by {VAL_IMPROVEMENT:.0%} "
            f"({first_val:.4f} -> {best_val:.4f})"
        )
    if last_val > VAL_REGRESSION_CAP * best_val:
        failures.append(
            f"final val MAE {last_val:.4f} > {VAL_REGRESSION_CAP}x best {best_val:.4f}"
        )

    cohort_naive, matched = _cohort_naive_mae(
        baseline_index, [str(p) for p in patch_ids["validation"]], "validation"
    )
    notes.append(
        f"same-cohort naive validation MAE: {cohort_naive:.4f} "
        f"({matched} matched patches)"
    )
    if matched != int(admitted.get("validation", 0)):
        failures.append(
            f"cohort naive matched {matched} patches != admitted "
            f"{admitted.get('validation')}"
        )
    if best_val > NAIVE_FACTOR_CAP * cohort_naive:
        failures.append(
            f"best val MAE {best_val:.4f} > {NAIVE_FACTOR_CAP}x cohort naive {cohort_naive:.4f}"
        )

    # ── selected checkpoint consistency ───────────────────────────────
    selected_epoch = summary.get("best_epoch")
    best_metric = summary.get("best_metric")
    recomputed = summary.get("reload_recomputed")
    if selected_epoch != best_epoch:
        failures.append(f"summary best_epoch {selected_epoch} != argmin epoch {best_epoch}")
    if not _finite(best_metric) or abs(float(best_metric) - best_val) > RECHECK_TOLERANCE:
        failures.append(f"summary best_metric {best_metric!r} != curve best {best_val:.6f}")
    if not _finite(recomputed):
        failures.append("summary reload_recomputed is missing or non-finite")
    elif _finite(best_metric) and abs(float(recomputed) - float(best_metric)) > RECHECK_TOLERANCE:
        failures.append(
            f"reload recheck {recomputed!r} != selected {best_metric!r}"
        )

    return True, failures, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="exercise the probe guards (positive/negative) and exit; no evidence needed",
    )
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, type=Path)
    args = parser.parse_args()

    if args.self_check:
        return _guard_self_check()
    if args.evidence is None:
        print("FAIL: --evidence is required (or pass --self-check)")
        return 1
    if not args.evidence.is_file():
        print(f"FAIL: evidence not found: {args.evidence}")
        return 1
    if not args.baseline.is_file():
        print(f"FAIL: baseline report not found: {args.baseline}")
        return 1

    evidence = _load_json(args.evidence)
    baseline = _load_json(args.baseline)
    if not isinstance(evidence, dict) or not isinstance(baseline, dict):
        print("FAIL: evidence and baseline must be JSON objects")
        return 1

    summary = evidence.get("probe", {}).get("summary", {})
    resolved = summary.get("resolved_config") if isinstance(summary, dict) else None
    if not isinstance(resolved, dict):
        print("FAIL: evidence has no resolved_config to locate the published index")
        return 1
    try:
        published = _published_index(resolved)
    except Exception as exc:
        # A read failure (ADC, network, stale index) must surface, not crash.
        print(f"FAIL: could not read the published patch index: {exc}")
        return 1

    decidable, failures, notes = validate(evidence, baseline, published)
    for note in notes:
        print(f"  {note}")
    if not decidable:
        for failure in failures:
            print(f"INCONCLUSIVE: {failure}")
        return 1
    if failures:
        for failure in failures:
            print(f"NO-GO: {failure}")
        return 2
    print("GO: probe evidence is complete and the frozen method clears the screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
