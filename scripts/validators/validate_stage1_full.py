"""Independently validate a retained full Stage-1 run evidence record (issue #53).

Screens the unbounded 20-epoch Stage-1 run (evidence ``profile == "full"``)
against the pre-registered decision table in ``docs/stage1-full-results.md``:
a learning-off-init check, a no-divergence cap, a non-passthrough correction
floor, the zero-init identity check, an exact same-universe reconciliation
against the committed naive baseline, and a one-shot 2025 test score. It
re-derives the GO / usable-anchor / NO-GO decision from the retained numbers and
the committed baseline report; it never trusts the run's own success flag.

Inputs are local files:

- the full-run evidence JSON uploaded by ``scripts/operators/vertex_evidence.py``
  (``profile="full"``);
- the committed naive-baseline report
  ``docs/results/baseline-full-<run>/baseline_report.json``.

Exit codes: 0 = go, 2 = usable anchor, 3 = no-go, 1 = evidence missing,
malformed, incomplete, or the wrong profile (cannot decide). ``print()`` is the
human summary.

Usage:
    uv run python scripts/validators/validate_stage1_full.py --self-check
    uv run python scripts/validators/validate_stage1_full.py --evidence <evidence.json>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.modeling.run import (
    assert_stage1_full_bounds,
    assert_stage1_lock,
    guard_modeling_config,
)

# Pre-registered full-run screen (docs/stage1-full-results.md). Screening
# thresholds only — not model-quality claims.
EXPECTED_EPOCHS = 20
TRAIN_IMPROVEMENT = 0.05  # final train MAE at least 5% below epoch one
VAL_REGRESSION_CAP = 1.25  # final val MAE at most 1.25x the best val MAE
CORRECTION_MIN_FRACTION = 0.05  # pooled |correction| at least 5% of full-val naive
IDENTITY_TOLERANCE_K = 1e-3  # zero-init output vs prior arm
RECHECK_TOLERANCE = 1e-3
USABLE_FACTOR = 1.5  # usable-anchor cap = 1.5x the split's own full naive
EVIDENCE_PROFILE = "full"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_BASELINE = "docs/results/baseline-full-20260929T084243Z-29A5F946/baseline_report.json"

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "modeling")


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _classify(
    *,
    train_maes: list[float],
    val_maes: list[float],
    residual_prior: object,
    identity_k: object,
    correction_k: object,
    val_naive: float,
    test_mae: float | None,
    test_naive: float,
) -> tuple[str, list[str]]:
    """Return ``(tier, failures)`` for a full-run curve and one-shot test score.

    Pure (no I/O) so the evidence path and the self-check prove the same
    thresholds accept each tier and reject each failure mode.
    """
    if len(train_maes) != EXPECTED_EPOCHS or len(val_maes) != EXPECTED_EPOCHS:
        return (
            "incomplete",
            [f"retained {len(val_maes)} validation epochs, expected {EXPECTED_EPOCHS}"],
        )
    if not all(_finite(v) for v in train_maes + val_maes):
        return "no-go", ["non-finite train/validation MAE (catastrophic fit)"]

    first_train, last_train = train_maes[0], train_maes[-1]
    last_val = val_maes[-1]
    best_val = min(val_maes)

    failures: list[str] = []
    if last_train > (1.0 - TRAIN_IMPROVEMENT) * first_train:
        failures.append(
            f"train MAE did not improve by {TRAIN_IMPROVEMENT:.0%} "
            f"({first_train:.4f} -> {last_train:.4f}): flat"
        )
    if last_val > VAL_REGRESSION_CAP * best_val:
        failures.append(f"final val MAE {last_val:.4f} > {VAL_REGRESSION_CAP}x best {best_val:.4f}")
    if residual_prior is not True:
        failures.append("summary does not report the residual representation")
    if not _finite(identity_k) or float(identity_k) > IDENTITY_TOLERANCE_K:
        failures.append(f"residual identity MAE {identity_k!r} exceeds {IDENTITY_TOLERANCE_K} K")
    floor = CORRECTION_MIN_FRACTION * val_naive
    if not _finite(correction_k) or float(correction_k) < floor:
        failures.append(
            f"selected pooled correction {correction_k!r} < {CORRECTION_MIN_FRACTION:.0%} "
            f"of full-val naive ({floor:.4f} K): prior passthrough"
        )
    if failures:
        return "no-go", failures

    if test_mae is None or not _finite(test_mae):
        return "incomplete", ["one-shot 2025 test score is missing or non-finite"]

    if best_val <= val_naive and float(test_mae) <= test_naive:
        return "go", []
    if best_val <= USABLE_FACTOR * val_naive and float(test_mae) <= USABLE_FACTOR * test_naive:
        return "usable", []
    return (
        "no-go",
        [
            f"best val {best_val:.4f} / test {float(test_mae):.4f} outside the usable-anchor "
            f"bounds (<= {USABLE_FACTOR}x each split's full naive)"
        ],
    )


def _screen_self_check() -> list[str]:
    """Prove the screen accepts each tier and rejects each failure mode."""
    failures: list[str] = []
    val_naive, test_naive = 1.61010, 1.39523
    train = [round(2.8 - i * 0.09, 4) for i in range(EXPECTED_EPOCHS)]
    base = {
        "train_maes": train,
        "val_maes": [round(1.30 - i * 0.01, 4) for i in range(EXPECTED_EPOCHS)],
        "residual_prior": True,
        "identity_k": 0.0,
        "correction_k": 0.5,
        "val_naive": val_naive,
        "test_mae": 1.1,
        "test_naive": test_naive,
    }
    cases: list[tuple[str, dict, str]] = [("go-shaped record", base, "go")]
    cases.append(
        ("usable anchor", {**base, "val_maes": [2.0] * EXPECTED_EPOCHS, "test_mae": 1.9}, "usable")
    )
    cases.append(
        ("no-go scores", {**base, "val_maes": [3.0] * EXPECTED_EPOCHS, "test_mae": 2.5}, "no-go")
    )
    cases.append(("flat train", {**base, "train_maes": [2.8] * EXPECTED_EPOCHS}, "no-go"))
    cases.append(
        ("non-finite fit", {**base, "val_maes": [float("nan")] + base["val_maes"][1:]}, "no-go")
    )
    cases.append(("not residual", {**base, "residual_prior": False}, "no-go"))
    cases.append(("passthrough correction", {**base, "correction_k": 0.0}, "no-go"))
    cases.append(("diverging val", {**base, "val_maes": base["val_maes"][:-1] + [5.0]}, "no-go"))
    cases.append(("short run", {**base, "val_maes": base["val_maes"][:6]}, "incomplete"))
    cases.append(("missing test", {**base, "test_mae": None}, "incomplete"))

    for label, kwargs, want in cases:
        got, _ = _classify(**kwargs)
        status = "PASS" if got == want else "FAIL"
        if got != want:
            failures.append(f"{label}: got {got}, expected {want}")
        print(f"  {status} screen: {label} (tier={got}, expected={want})")
    return failures


def _guard_self_check() -> int:
    """Exercise the Stage-1 full guards on the shipped config and on drift."""

    def compose_cfg(config_name: str, overrides: list[str]):
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            return compose(config_name=config_name, overrides=overrides)

    failures: list[str] = []
    guard_cases = [
        ("shipped stage1_locked", "stage1_locked", [], True),
        (
            "test split admitted",
            "stage1_locked",
            ['data.splits=["train","validation","test"]'],
            False,
        ),
        ("cpu accelerator", "stage1_locked", ["trainer.accelerator=cpu"], False),
        ("two devices", "stage1_locked", ["trainer.devices=2"], False),
        ("wandb offline", "stage1_locked", ["wandb.mode=offline"], False),
        ("gs output root", "stage1_locked", ["output_root=gs://bucket/run"], False),
        ("full marker dropped", "stage1_locked", ["stage1_full=false"], False),
        ("residual dropped", "stage1_locked", ["stage1_residual_prior=false"], False),
        ("seed drift", "stage1_locked", ["seed=1"], False),
    ]
    for label, config_name, overrides, should_pass in guard_cases:
        cfg = compose_cfg(config_name, overrides)
        try:
            assert_stage1_lock(cfg)
            assert_stage1_full_bounds(cfg)
            accepted = True
        except Exception:
            accepted = False
        status = "PASS" if accepted == should_pass else "FAIL"
        if accepted != should_pass:
            failures.append(label)
        print(f"  {status} guard: {label} (accepted={accepted}, expected={should_pass})")

    try:
        guard_modeling_config(compose_cfg("stage1_probe", []), "stage1_probe")
        print("  PASS guard: full guard accepts stage1_probe")
    except Exception as exc:  # pragma: no cover - self-check reporting
        failures.append(f"guard rejected stage1_probe: {exc}")
        print(f"  FAIL guard: full guard accepts stage1_probe ({exc})")

    failures.extend(_screen_self_check())

    if failures:
        print(f"SELF-CHECK FAILED: {failures}")
        return 1
    print("SELF-CHECK OK: full-run guards accept the shipped config and the screen tiers hold")
    return 0


def _load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _baseline_ids(report: dict, split: str) -> set[str]:
    return {
        str(entry["patch_id"])
        for entry in report.get("patches", [])
        if str(entry.get("split")) == split
    }


def _baseline_exclusions(report: dict, split: str) -> dict[str, int]:
    splits = report.get("splits") or {}
    entry = splits.get(split) or {}
    return {str(k): int(v) for k, v in (entry.get("exclusions") or {}).items()}


def _split_mae(report: dict, split: str) -> float:
    splits = report.get("splits") or {}
    return float((splits.get(split) or {})["mae"])


def _split_valid_cells(report: dict, split: str) -> float:
    splits = report.get("splits") or {}
    return float((splits.get(split) or {})["valid_cells"])


def validate(evidence: dict, baseline: dict) -> tuple[str, list[str], list[str]]:
    """Return ``(tier, failures, notes)`` for a full-run evidence record."""
    notes: list[str] = []
    if evidence.get("profile") != EVIDENCE_PROFILE:
        return "incomplete", [f"evidence profile is not {EVIDENCE_PROFILE!r}"], notes

    scope = evidence.get("data_scope")
    full = evidence.get("full")
    if not isinstance(scope, dict) or not isinstance(full, dict):
        return "incomplete", ["evidence is missing data_scope/full blocks"], notes

    epochs = full.get("epoch_metrics")
    summary = full.get("summary")
    test_scope = full.get("test_scope")
    if not isinstance(epochs, list) or not epochs:
        return "incomplete", ["no epoch_metrics retained (run incomplete or evidence lost)"], notes
    if not isinstance(summary, dict) or not summary:
        return "incomplete", ["no full summary retained"], notes
    if full.get("epochs_complete") is not True:
        return "incomplete", ["evidence reports epochs_complete=false"], notes

    # ── checkpoint reference ──────────────────────────────────────────
    checkpoint = evidence.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return "incomplete", ["evidence has no checkpoint reference"], notes
    uri = str(checkpoint.get("uri", ""))
    if (
        not uri.endswith(".ckpt")
        or not checkpoint.get("sha256")
        or not _SHA256_RE.match(str(checkpoint.get("sha256")))
    ):
        return "incomplete", ["checkpoint reference is missing a valid sha256/uri"], notes
    if not isinstance(checkpoint.get("bytes"), int) or int(checkpoint["bytes"]) <= 0:
        return "incomplete", ["checkpoint reference has no positive byte size"], notes
    notes.append(
        f"checkpoint {uri} sha256 {str(checkpoint['sha256'])[:12]}… ({checkpoint['bytes']} bytes)"
    )

    # ── comparison universe (exact IDs, not counts) ───────────────────
    patch_ids = scope.get("patch_ids")
    admitted = scope.get("patches_per_split")
    requested = scope.get("requested_per_split")
    skipped = scope.get("skipped_refs")
    if (
        not isinstance(patch_ids, dict)
        or not isinstance(admitted, dict)
        or not isinstance(requested, dict)
        or not isinstance(skipped, dict)
    ):
        return (
            "incomplete",
            ["data_scope is missing patch_ids/patches_per_split/requested/skipped"],
            notes,
        )

    if sorted(patch_ids) != ["train", "validation"]:
        return (
            "incomplete",
            [f"fit splits {sorted(patch_ids)} are not train/validation only"],
            notes,
        )

    for split in ("validation", "test"):
        expected = _baseline_ids(baseline, split)
        got = {str(p) for p in patch_ids.get(split, [])}
        if got != expected:
            return (
                "incomplete",
                [
                    f"{split}: admitted patch IDs differ from the baseline universe "
                    f"(model {len(got)}, baseline {len(expected)}, "
                    f"missing {len(expected - got)}, extra {len(got - expected)})"
                ],
                notes,
            )
        model_skip = Counter(str(r.get("reason")) for r in skipped.get(split, []))
        base_excl = _baseline_exclusions(baseline, split)
        if dict(model_skip) != base_excl:
            return (
                "incomplete",
                [f"{split}: skipped reasons {dict(model_skip)} != baseline exclusions {base_excl}"],
                notes,
            )
        if int(requested.get(split, -1)) != int(
            (baseline.get("splits") or {}).get(split, {}).get("requested_patches", -1)
        ):
            return (
                "incomplete",
                [f"{split}: requested {requested.get(split)} != baseline requested universe"],
                notes,
            )

    fingerprints = baseline.get("fingerprints") or {}
    total = int(fingerprints.get("patches_total", 0))
    val_req = int((baseline.get("splits") or {}).get("validation", {}).get("requested_patches", 0))
    test_req = int((baseline.get("splits") or {}).get("test", {}).get("requested_patches", 0))
    expected_train = total - val_req - test_req
    if total and int(requested.get("train", -1)) != expected_train:
        return (
            "incomplete",
            [f"train: requested {requested.get('train')} != derived {expected_train}"],
            notes,
        )

    if not isinstance(test_scope, dict) or sorted(test_scope.get("patch_ids") or {}) != ["test"]:
        return "incomplete", ["test_scope is missing or is not a test-only read"], notes

    # ── epoch curve and one-shot test numbers ─────────────────────────
    train_maes: list[float] = []
    val_maes: list[float] = []
    for entry in epochs:
        train = entry.get("train_mae_100m")
        val = entry.get("validation_mae_100m")
        if _finite(train):
            train_maes.append(float(train))
        if _finite(val):
            val_maes.append(float(val))
    if len(train_maes) != len(epochs) or len(val_maes) != len(epochs):
        return "incomplete", ["epoch curve has missing/non-finite values"], notes

    test = summary.get("test")
    test_mae = (
        float(test["mae_100m"])
        if isinstance(test, dict) and _finite(test.get("mae_100m"))
        else None
    )
    if test_mae is None:
        return "incomplete", ["summary has no finite one-shot test score"], notes

    # Model valid-cell totals must equal the baseline's over the same universe.
    model_test_cells = test.get("valid_cells") if isinstance(test, dict) else None
    if not _finite(model_test_cells) or float(model_test_cells) != _split_valid_cells(
        baseline, "test"
    ):
        return (
            "incomplete",
            [
                f"test valid cells {model_test_cells!r} != baseline "
                f"{_split_valid_cells(baseline, 'test')}"
            ],
            notes,
        )
    model_val_cells = summary.get("validation_valid_cells")
    if not _finite(model_val_cells) or float(model_val_cells) != _split_valid_cells(
        baseline, "validation"
    ):
        return (
            "incomplete",
            [
                f"validation valid cells {model_val_cells!r} != baseline "
                f"{_split_valid_cells(baseline, 'validation')}"
            ],
            notes,
        )
    notes.append(
        f"validation {int(float(model_val_cells))} cells, "
        f"test {int(float(model_test_cells))} cells match the baseline universe"
    )

    # Selected-checkpoint consistency with the retained curve.
    best_val = min(val_maes)
    best_epoch = val_maes.index(best_val) + 1
    if summary.get("best_epoch") != best_epoch:
        return (
            "incomplete",
            [f"summary best_epoch {summary.get('best_epoch')} != curve argmin {best_epoch}"],
            notes,
        )
    best_metric = summary.get("best_metric")
    if not _finite(best_metric) or abs(float(best_metric) - best_val) > RECHECK_TOLERANCE:
        return (
            "incomplete",
            [f"summary best_metric {best_metric!r} != curve best {best_val:.6f}"],
            notes,
        )
    recomputed = summary.get("reload_recomputed")
    if not _finite(recomputed) or abs(float(recomputed) - float(best_metric)) > RECHECK_TOLERANCE:
        return "incomplete", [f"reload recheck {recomputed!r} != selected {best_metric!r}"], notes

    val_naive = _split_mae(baseline, "validation")
    test_naive = _split_mae(baseline, "test")
    tier, failures = _classify(
        train_maes=train_maes,
        val_maes=val_maes,
        residual_prior=summary.get("residual_prior"),
        identity_k=summary.get("residual_identity_mae_k"),
        correction_k=summary.get("residual_correction_mean_abs_k"),
        val_naive=val_naive,
        test_mae=test_mae,
        test_naive=test_naive,
    )
    notes.append(
        f"train MAE {train_maes[0]:.4f} -> {train_maes[-1]:.4f}; "
        f"val MAE best {best_val:.4f} @ epoch {best_epoch}; test {test_mae:.4f}"
    )
    notes.append(
        f"full naive anchors: validation {val_naive:.5f} K, test {test_naive:.5f} K; "
        f"identity {summary.get('residual_identity_mae_k')} K, "
        f"pooled correction {summary.get('residual_correction_mean_abs_k')} K"
    )
    return tier, failures, notes


_TIER_EXIT = {"go": 0, "usable": 2, "no-go": 3, "incomplete": 1}


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

    tier, failures, notes = validate(evidence, baseline)
    for note in notes:
        print(f"  {note}")
    for failure in failures:
        print(f"  failure: {failure}")
    print(f"VERDICT: {tier.upper()}")
    return _TIER_EXIT[tier]


if __name__ == "__main__":
    raise SystemExit(main())
