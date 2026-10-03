"""Independently validate a retained Stage-1 recovery-probe evidence record (#47).

Screens the **residual** probe (evidence ``profile == "probe-residual"``) against
the pre-registered go/no-go criteria in ``docs/stage1-debug-results.md``: a
learning-off-init check, a same-cohort naive factor cap, a no-divergence cap, a
non-passthrough correction-amplitude floor, and the zero-init identity check. It
re-derives the decision from the retained numbers and the committed baseline
report; it never trusts the run's own success flag.

Inputs are local files:

- the recovery evidence JSON uploaded by ``scripts/operators/vertex_evidence.py``
  (``profile="probe-residual"``);
- the committed naive-baseline report
  ``docs/results/baseline-full-<run>/baseline_report.json``.

Exit codes: 0 = go, 2 = valid evidence but no-go, 1 = evidence missing,
malformed, incomplete, or the wrong profile (cannot decide). ``print()`` is the
human summary.

Usage:
    uv run python scripts/validators/validate_stage1_recovery.py --self-check
    uv run python scripts/validators/validate_stage1_recovery.py --evidence <evidence.json>
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
from berlin_lst_downscaling.modeling.run import (
    assert_probe_minima,
    assert_stage1_lock,
    assert_stage1_probe,
    assert_stage1_probe_lr3,
    guard_modeling_config,
)

# Pre-registered recovery screen (docs/stage1-debug-results.md). Screening
# thresholds only — not model-quality claims.
EXPECTED_EPOCHS = 6
TRAIN_IMPROVEMENT = 0.05  # final train MAE at least 5% below epoch one
VAL_NAIVE_FACTOR_CAP = 1.5  # best val MAE at most 1.5x the same-cohort naive
VAL_ABS_CAP_K = 8.0  # ... and at most this absolute value
VAL_REGRESSION_CAP = 1.25  # final val MAE at most 1.25x the best val MAE
CORRECTION_MIN_FRACTION = 0.05  # pooled |correction| at least 5% of naive
IDENTITY_TOLERANCE_K = 1e-3  # zero-init output vs prior arm
RECHECK_TOLERANCE = 1e-3
EVIDENCE_PROFILE = "probe-residual"
# The only learning rates the two recovery trials may have been run at.
ALLOWED_TRIAL_LR = (1.0e-3, 3.0e-3)

DEFAULT_BASELINE = "docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json"

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")


def _guard_self_check() -> int:
    """Exercise the recovery-probe guards on the shipped configs and on drift.

    Positive and negative cases, composed locally with no source read and no
    submission, plus deliberately invalid/valid screen evidence for the
    threshold logic.
    """

    def compose_cfg(config_name: str, overrides: list[str]):
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            return compose(config_name=config_name, overrides=overrides)

    guard_cases = [
        ("shipped stage1_probe", "stage1_probe", [], assert_stage1_probe, True),
        (
            "residual dropped",
            "stage1_probe",
            ["stage1_residual_prior=false"],
            assert_stage1_probe,
            False,
        ),
        (
            "probe LR drift",
            "stage1_probe",
            ["trainer.learning_rate=0.01"],
            assert_stage1_probe,
            False,
        ),
        (
            "lr3 cannot carry probe LR",
            "stage1_probe_lr3",
            ["trainer.learning_rate=1.0e-3"],
            assert_stage1_probe_lr3,
            False,
        ),
        ("shipped stage1_probe_lr3", "stage1_probe_lr3", [], assert_stage1_probe_lr3, True),
        (
            "lr3 residual dropped",
            "stage1_probe_lr3",
            ["stage1_residual_prior=false"],
            assert_stage1_probe_lr3,
            False,
        ),
        (
            "test split included",
            "stage1_probe",
            ['data.splits=["train","validation","test"]'],
            assert_stage1_probe,
            False,
        ),
    ]
    failures: list[str] = []
    for label, config_name, overrides, guard, should_pass in guard_cases:
        try:
            guard(compose_cfg(config_name, overrides))
            accepted = True
        except Exception:
            accepted = False
        status = "PASS" if accepted == should_pass else "FAIL"
        if accepted != should_pass:
            failures.append(label)
        print(f"  {status} guard: {label} (accepted={accepted}, expected={should_pass})")

    cfg = compose_cfg("stage1_probe", [])
    good_scope = {
        "patches_per_split": {"train": 128, "validation": 64},
        "scenes_per_split": {"train": list(range(16)), "validation": list(range(8))},
        "years_per_split": {"train": [2017, 2018, 2019], "validation": [2024]},
    }
    try:
        assert_probe_minima(cfg, good_scope)
        print("  PASS minima: cohort at minima")
    except Exception as exc:  # pragma: no cover - self-check reporting
        failures.append(f"minima at cohort: {exc}")
        print(f"  FAIL minima: cohort at minima ({exc})")

    # The frozen method lock remains inspectable, but full execution is blocked
    # until the Stage-1 efficiency gate and a separate approval are complete.
    try:
        locked = compose_cfg("stage1_locked", [])
        assert_stage1_lock(locked)
        try:
            guard_modeling_config(locked, "stage1_locked")
        except ValueError:
            print("  PASS guard: method lock valid; full execution blocked")
        else:
            failures.append("full execution guard accepted stage1_locked")
            print("  FAIL guard: full execution is not blocked")
    except ValueError as exc:  # pragma: no cover - self-check reporting
        failures.append(f"method lock rejected stage1_locked: {exc}")
        print(f"  FAIL guard: method lock remains valid ({exc})")
    try:
        guard_modeling_config(
            compose_cfg("contract_smoke", ["+stage1_residual_prior=true"]), "contract_smoke"
        )
        failures.append("residual marker accepted on contract_smoke")
        print("  FAIL guard: residual marker rejected off-lock/probe")
    except ValueError:
        print("  PASS guard: residual marker rejected off-lock/probe")
    try:
        assert_stage1_probe(compose_cfg("stage1_probe", []))
        try:
            guard_modeling_config(compose_cfg("stage1_probe", []), "stage1_probe")
        except ValueError:
            print("  PASS guard: standalone recovery probe is paused")
        else:
            failures.append("standalone recovery probe remained executable")
            print("  FAIL guard: standalone recovery probe is paused")
    except Exception as exc:  # pragma: no cover - self-check reporting
        failures.append(f"recovery probe contract could not be inspected: {exc}")
        print(f"  FAIL guard: recovery probe contract remains inspectable ({exc})")

    screen_failures = _screen_self_check()
    failures.extend(screen_failures)

    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK OK: recovery guards and screen accept the shipped config and reject drift")
    return 0


def _screen_evidence(
    *,
    train_maes: list[float],
    val_maes: list[float],
    residual_prior: bool,
    identity_k: float | None,
    correction_k: float | None,
    cohort_naive: float,
) -> list[str]:
    """Apply the recovery screen to already-extracted numbers (no I/O).

    Shared by the evidence path and the self-check, so the thresholds are proven
    to accept a GO-shaped record and reject each failure mode.
    """
    failures: list[str] = []
    if len(train_maes) != EXPECTED_EPOCHS or len(val_maes) != EXPECTED_EPOCHS:
        return ["epoch curve has the wrong length"]
    if not all(_finite(v) for v in train_maes + val_maes):
        return ["non-finite train/val MAE"]

    first_train, last_train = train_maes[0], train_maes[-1]
    last_val = val_maes[-1]
    best_val = min(val_maes)

    if last_train > (1.0 - TRAIN_IMPROVEMENT) * first_train:
        failures.append(
            f"train MAE did not improve by {TRAIN_IMPROVEMENT:.0%} "
            f"({first_train:.4f} -> {last_train:.4f}): flat"
        )
    cap = min(VAL_NAIVE_FACTOR_CAP * cohort_naive, VAL_ABS_CAP_K)
    if best_val > cap:
        failures.append(
            f"best val MAE {best_val:.4f} > min({VAL_NAIVE_FACTOR_CAP}x naive "
            f"{VAL_NAIVE_FACTOR_CAP * cohort_naive:.4f}, {VAL_ABS_CAP_K})"
        )
    if last_val > VAL_REGRESSION_CAP * best_val:
        failures.append(f"final val MAE {last_val:.4f} > {VAL_REGRESSION_CAP}x best {best_val:.4f}")
    if residual_prior is not True:
        failures.append("summary does not report the residual representation")
    if not _finite(identity_k) or float(identity_k) > IDENTITY_TOLERANCE_K:
        failures.append(f"residual identity MAE {identity_k!r} exceeds {IDENTITY_TOLERANCE_K} K")
    floor = CORRECTION_MIN_FRACTION * cohort_naive
    if not _finite(correction_k) or float(correction_k) < floor:
        failures.append(
            f"selected pooled correction {correction_k!r} < {CORRECTION_MIN_FRACTION:.0%} "
            f"of naive ({floor:.4f} K): prior passthrough"
        )
    return failures


def _screen_self_check() -> list[str]:
    """Prove the screen accepts a GO record and rejects each failure mode."""
    failures: list[str] = []
    naive = 1.5261
    go = {
        "train_maes": [2.8, 2.5, 2.2, 2.0, 1.9, 1.8],
        "val_maes": [2.0, 1.9, 1.85, 1.8, 1.78, 1.79],
        "residual_prior": True,
        "identity_k": 0.0,
        "correction_k": 0.5,
        "cohort_naive": naive,
    }
    cases = [("go-shaped record", go, False)]
    cases.append(
        (
            "flat train",
            {**go, "train_maes": [2.8, 2.8, 2.8, 2.79, 2.79, 2.78]},
            True,
        )
    )
    cases.append(
        (
            "far-from-naive val",
            {**go, "val_maes": [320.0, 318.0, 316.0, 315.0, 314.0, 313.0]},
            True,
        )
    )
    cases.append(("not residual", {**go, "residual_prior": False}, True))
    cases.append(("bad identity", {**go, "identity_k": 5.0}, True))
    cases.append(("passthrough correction", {**go, "correction_k": 0.0}, True))
    cases.append(("diverging val", {**go, "val_maes": [1.9, 1.8, 1.7, 1.6, 1.7, 2.6]}, True))

    for label, kwargs, should_fail in cases:
        failed = bool(_screen_evidence(**kwargs))
        status = "PASS" if failed == should_fail else "FAIL"
        if failed != should_fail:
            failures.append(label)
        print(f"  {status} screen: {label} (rejected={failed}, expected={should_fail})")
    return failures


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
    return {str(entry["patch_id"]): entry for entry in report.get("patches", [])}


def _published_index(resolved: dict) -> dict[str, str]:
    """Patch id -> split, read independently from the published patch index."""
    source = RealSourceConfig(
        patch_index_root=str(resolved["patch_index_root"]),
        training_root=str(resolved["training_root"]),
        features_root=str(resolved["features_root"]),
        ard_root=str(resolved["ard_root"]),
    )
    return {ref.patch_id: ref.split for ref in load_patch_refs(source)}


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
    """Return ``(decidable, failures, notes)`` for a recovery evidence record."""
    failures: list[str] = []
    notes: list[str] = []

    if evidence.get("profile") != EVIDENCE_PROFILE:
        return (
            False,
            [
                f"evidence profile is not {EVIDENCE_PROFILE!r} "
                "(the historical #45 validator handles profile='probe')"
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

    # A missing or non-finite recovery-state field makes the record undecidable,
    # not a NO-GO (docs/stage1-debug-results.md §3): the run fails closed before
    # fit if the identity wiring is wrong, so a missing field means evidence drift.
    if not isinstance(summary.get("residual_prior"), bool):
        return False, ["recovery evidence is missing the residual representation flag"], notes
    identity_k = summary.get("residual_identity_mae_k")
    correction_k = summary.get("residual_correction_mean_abs_k")
    if not _finite(identity_k) or not _finite(correction_k):
        return (
            False,
            ["recovery evidence is missing or carries non-finite residual identity/correction"],
            notes,
        )

    # ── cohort structure ──────────────────────────────────────────────
    patch_ids = scope.get("patch_ids")
    admitted = scope.get("patches_per_split")
    scenes = scope.get("scenes_per_split")
    years = scope.get("years_per_split")
    if not isinstance(patch_ids, dict) or not isinstance(admitted, dict):
        return False, ["data_scope is missing patch_ids/patches_per_split"], notes
    if sorted(patch_ids) != ["train", "validation"]:
        failures.append(f"admitted splits {sorted(patch_ids)} are not train/validation only")

    resolved = summary.get("resolved_config")
    data_cfg = resolved.get("data") if isinstance(resolved, dict) else None
    probe_cfg = data_cfg.get("probe") if isinstance(data_cfg, dict) else None
    if not isinstance(probe_cfg, dict):
        return False, ["summary.resolved_config has no data.probe block"], notes
    expected_admitted = probe_cfg.get("min_admitted_per_split", {})
    expected_scenes = probe_cfg.get("min_scenes_per_split", {})
    expected_years = int(probe_cfg.get("min_train_years", 0))
    per_scene_cap = int(probe_cfg.get("max_refs_per_scene", 0))

    trainer_cfg = resolved.get("trainer") if isinstance(resolved, dict) else None
    trial_lr = trainer_cfg.get("learning_rate") if isinstance(trainer_cfg, dict) else None
    if trial_lr not in ALLOWED_TRIAL_LR:
        failures.append(
            f"resolved_config trainer.learning_rate={trial_lr!r} is not one of the "
            f"recovery-trial rates {ALLOWED_TRIAL_LR}"
        )

    for split in ("train", "validation"):
        ids = [str(p) for p in patch_ids.get(split, [])]
        n_admitted = int(admitted.get(split, 0))
        if n_admitted != len(ids):
            failures.append(f"{split}: admitted count {n_admitted} != patch_ids length {len(ids)}")
        need = int(expected_admitted.get(split, 0))
        if n_admitted < need:
            failures.append(f"{split}: admitted {n_admitted} < required {need}")
        if isinstance(scenes, dict):
            got_scenes = len(scenes.get(split, []))
            need_scenes = int(expected_scenes.get(split, 0))
            if got_scenes < need_scenes:
                failures.append(f"{split}: {got_scenes} scenes < required {need_scenes}")
        derived: dict[str, int] = {}
        for patch_id in ids:
            derived[_scene_id(patch_id)] = derived.get(_scene_id(patch_id), 0) + 1
        over_cap = {s: c for s, c in derived.items() if c > per_scene_cap}
        if over_cap:
            failures.append(
                f"{split}: scene(s) exceed the per-scene cap {per_scene_cap}: {over_cap}"
            )

    train_years = {
        y
        for y in (_scene_year(_scene_id(str(p))) for p in patch_ids.get("train", []))
        if y is not None
    }
    if len(train_years) < expected_years:
        failures.append(f"train covers {len(train_years)} years < required {expected_years}")
    if isinstance(years, dict):
        for split in ("train", "validation"):
            if not years.get(split):
                failures.append(f"{split}: no year recorded")

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

    try:
        cohort_naive, matched = _cohort_naive_mae(
            baseline_index, [str(p) for p in patch_ids["validation"]], "validation"
        )
    except (ValueError, KeyError, TypeError) as exc:
        return (
            False,
            [f"cannot recompute the same-cohort naive baseline: {exc}"],
            notes,
        )
    notes.append(
        f"same-cohort naive validation MAE: {cohort_naive:.4f} ({matched} matched patches)"
    )
    if matched != int(admitted.get("validation", 0)):
        failures.append(
            f"cohort naive matched {matched} patches != admitted {admitted.get('validation')}"
        )
    best_val = min(val_maes)
    best_epoch = val_maes.index(best_val) + 1
    notes.append(
        f"train MAE: {train_maes[0]:.4f} -> {train_maes[-1]:.4f}; "
        f"val MAE: {val_maes[0]:.4f} -> {val_maes[-1]:.4f} (best {best_val:.4f} @ {best_epoch})"
    )

    failures.extend(
        _screen_evidence(
            train_maes=train_maes,
            val_maes=val_maes,
            residual_prior=summary.get("residual_prior"),
            identity_k=identity_k,
            correction_k=correction_k,
            cohort_naive=cohort_naive,
        )
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
        failures.append(f"reload recheck {recomputed!r} != selected {best_metric!r}")

    notes.append(
        f"identity {summary.get('residual_identity_mae_k')} K; "
        f"pooled correction {summary.get('residual_correction_mean_abs_k')} K "
        f"(floor {CORRECTION_MIN_FRACTION * cohort_naive:.4f} K)"
    )
    return True, failures, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
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
    print("GO: residual probe evidence clears the pre-registered recovery screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
