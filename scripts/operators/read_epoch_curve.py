"""Print the train/validation epoch curve from retained run evidence.

Read-only. Accepts a local path or a ``gs://`` URI to an ``epoch_metrics.json``
list, an evidence record with ``full.epoch_metrics`` or ``probe.epoch_metrics``,
or a mapping with a top-level ``epoch_metrics``. Only ``epoch``,
``train_mae_100m`` and ``validation_mae_100m`` are extracted; test metrics are
never read or printed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

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


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local path or gs:// URI of the evidence or epoch metrics")
    args = parser.parse_args(argv)
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
