"""Print the train/validation epoch curve from retained run evidence.

Read-only. Accepts a local path or a ``gs://`` URI to an ``epoch_metrics.json``
list, an evidence record with ``full.epoch_metrics`` or ``probe.epoch_metrics``,
or a mapping with a top-level ``epoch_metrics``. Only ``epoch``,
``train_mae_100m`` and ``validation_mae_100m`` are extracted; test metrics are
never read or printed.

Checkpoint-selection optimism (exploratory, next to the frozen rule): ``--arm LABEL FILE...``
(and ``--holdout-arm LABEL FILE...`` for spatial-holdout runs) takes the local
``epoch_metrics.json`` files of the seeds of one arm and reports, per arm, the best-epoch
validation MAE (what checkpoint selection picks), the last-epoch MAE, the mean of the last five
epochs (36-40 of a 40-epoch run) and best minus last, as mean and SD over the seeds. The
selected checkpoint is the minimum of a noisy curve, so best minus last is a negative number
that bounds how optimistic the selected figure is. A holdout run selects its checkpoint on the
seen anchors only, so its curve is the seen-anchor metric; the report says so. This mode reads
local files only (fetch them with ``gsutil cp``, see docs/spatial-holdout.md).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from berlin_lst_downscaling.data.io import read_bytes

Curve = list[tuple[int, float | None, float | None]]


def _epoch_list(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("full", "probe"):
            section = payload.get(key)
            if isinstance(section, dict) and isinstance(section.get("epoch_metrics"), list):
                return section["epoch_metrics"]
        if isinstance(payload.get("epoch_metrics"), list):
            return payload["epoch_metrics"]
    raise ValueError("no epoch_metrics list found in the input")


def _metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def extract_curve(payload: object) -> Curve:
    """Return ``(epoch, train_mae_100m, validation_mae_100m)`` per epoch entry."""
    curve: Curve = []
    for entry in _epoch_list(payload):
        if not isinstance(entry, dict):
            continue
        epoch = entry.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            continue
        curve.append(
            (epoch, _metric(entry.get("train_mae_100m")), _metric(entry.get("validation_mae_100m")))
        )
    return curve


def best_epoch(curve: Curve) -> tuple[int, float] | None:
    """Epoch with the lowest finite validation MAE (first on ties), or None."""
    best: tuple[int, float] | None = None
    for epoch, _train, validation in curve:
        if validation is None or not math.isfinite(validation):
            continue
        if best is None or validation < best[1]:
            best = (epoch, validation)
    return best


LAST_EPOCH = 40
LAST_WINDOW = range(36, LAST_EPOCH + 1)
HOLDOUT_NOTE = (
    "holdout run: validation_mae_100m is the seen-anchor metric the checkpoint was selected on"
)
OPTIMISM_NOTE = (
    "Exploratory, not part of the frozen rule. Best = the epoch checkpoint selection picks; "
    "best minus last (epoch 40) and best minus the mean of epochs 36-40 show how much of the "
    "selected figure is the minimum of a noisy curve. SD over seeds with ddof 1."
)


def run_optimism(curve: Curve) -> dict[str, float]:
    """Best-epoch, last-epoch and epochs 36-40 validation MAE of one 40-epoch run.

    Epochs are 1-based, as the run writes them. A curve that does not end at epoch 40 or lacks
    a finite value in epochs 36-40 raises ``ValueError`` instead of being summarised.
    """
    best = best_epoch(curve)
    if best is None:
        raise ValueError("no finite validation_mae_100m value")
    values = {epoch: validation for epoch, _train, validation in curve}
    if max(values) != LAST_EPOCH:
        raise ValueError(f"the curve ends at epoch {max(values)}, expected {LAST_EPOCH}")
    window = [values.get(epoch) for epoch in LAST_WINDOW]
    if any(v is None or not math.isfinite(v) for v in window):
        raise ValueError(f"epochs {LAST_WINDOW[0]}-{LAST_EPOCH} need a finite validation MAE")
    finite = [v for v in window if v is not None]
    last = finite[-1]
    mean_last = sum(finite) / len(finite)
    return {
        "best_epoch": float(best[0]),
        "best": best[1],
        "last": last,
        "mean_last": mean_last,
        "best_minus_last": best[1] - last,
        "best_minus_mean_last": best[1] - mean_last,
    }


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def arm_optimism(runs: list[dict[str, float]]) -> dict[str, tuple[float, float | None]]:
    """Mean and SD (ddof 1; None for one seed) over the seeds of each ``run_optimism`` quantity."""
    return {key: _mean_sd([run[key] for run in runs]) for key in runs[0] if key != "best_epoch"}


def _stat(stat: tuple[float, float | None]) -> str:
    mean, sd = stat
    return f"{mean:.4f} (SD {'-' if sd is None else f'{sd:.4f}'})"


def _read_local_curve(path: str) -> Curve:
    if "://" in path:
        raise ValueError(f"{path}: local files only; fetch it with gsutil cp first")
    return extract_curve(json.loads(Path(path).read_text(encoding="utf-8")))


def optimism_report(arms: dict[str, list[str]], holdout: set[str]) -> str:
    """Render the per-arm optimism table; ``arms`` maps a label to the files of its seeds."""
    lines = [OPTIMISM_NOTE]
    rows = (
        ("best epoch (selected)", "best"),
        ("last epoch (40)", "last"),
        ("mean of epochs 36-40", "mean_last"),
        ("best minus last", "best_minus_last"),
        ("best minus mean(36-40)", "best_minus_mean_last"),
    )
    for label, files in arms.items():
        runs = []
        for path in files:
            try:
                runs.append(run_optimism(_read_local_curve(path)))
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
        stats = arm_optimism(runs)
        epochs = ",".join(str(int(run["best_epoch"])) for run in runs)
        lines += ["", f"{label}: {len(runs)} seed(s), best epochs {epochs}"]
        if label in holdout:
            lines.append(f"  ({HOLDOUT_NOTE})")
        lines += [f"  {name:<24}{_stat(stats[key])}" for name, key in rows]
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        help="local path or gs:// URI of the evidence or epoch metrics",
    )
    for flag, what in (("--arm", "run"), ("--holdout-arm", "spatial-holdout run")):
        parser.add_argument(
            flag,
            nargs="+",
            action="append",
            default=[],
            metavar=("LABEL", "FILE"),
            help=f"checkpoint-selection optimism: an arm label and the local epoch_metrics.json "
            f"of each seed's {what}",
        )
    args = parser.parse_args(argv)
    if args.arm or args.holdout_arm:
        if args.source:
            parser.error("a source and --arm/--holdout-arm are different modes")
        labelled = [(*spec, False) for spec in args.arm] + [
            (*spec, True) for spec in args.holdout_arm
        ]
        arms: dict[str, list[str]] = {}
        for label, *files, _is_holdout in labelled:
            if not files or label in arms:
                parser.error(f"arm {label!r} needs files and may be given once")
            arms[label] = files
        holdout = {label for label, *_files, is_holdout in labelled if is_holdout}
        try:
            print(optimism_report(arms, holdout))
        except (OSError, ValueError) as exc:
            print(f"error: cannot summarise the epoch curves: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.source:
        parser.error("give a source, or --arm / --holdout-arm")
    try:
        payload = json.loads(read_bytes(args.source))
        curve = extract_curve(payload)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read an epoch curve: {exc}", file=sys.stderr)
        return 1
    if not curve:
        print("error: no epochs in the input", file=sys.stderr)
        return 1
    best = best_epoch(curve)
    if best is None:
        print("error: no finite validation_mae_100m value", file=sys.stderr)
        return 1
    print("epoch  train_mae_100m  validation_mae_100m")
    for epoch, train, validation in curve:
        mark = " *best" if epoch == best[0] else ""
        print(f"{epoch:5d}  {_cell(train):>14}  {_cell(validation):>19}{mark}")
    print(f"best_epoch={best[0]} validation_mae_100m={best[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
