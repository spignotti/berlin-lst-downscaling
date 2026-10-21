# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare arithmetic vs T4 pooling on a stratified scene sample (issue #80).

Offline helpers and a small CLI for estimating read cost and allocating a
fixed scene budget across domains. Not invoked by CI; run manually once
``training/v2`` (or another release) is available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def largest_remainder_allocation(weights: Sequence[float], total: int) -> list[int]:
    """Allocate ``total`` integer slots proportional to ``weights`` (Hamilton)."""
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
        key=lambda t: t[0],
        reverse=True,
    )
    for k in range(remainder):
        floors[fractional[k][1]] += 1
    return floors


@dataclass(frozen=True)
class CostEstimate:
    scenes: int
    patches_per_scene: float
    reads_per_patch: int

    @property
    def total_feature_reads(self) -> int:
        return int(math.ceil(self.scenes * self.patches_per_scene * self.reads_per_patch))


def stratified_scene_sample(
    scene_ids: Sequence[str],
    *,
    domains: dict[str, Sequence[str]],
    budget: int,
) -> dict[str, list[str]]:
    """Pick up to ``budget`` scenes with largest-remainder counts per domain."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    weights = [float(len(domains[d])) for d in sorted(domains)]
    labels = sorted(domains)
    counts = largest_remainder_allocation(weights, budget)
    out: dict[str, list[str]] = {}
    for label, count in zip(labels, counts, strict=True):
        out[label] = list(domains[label])[:count]
    return out


def summarize_domains(scene_ids: Iterable[str], domain_of: dict[str, str]) -> dict[str, int]:
    tallies: dict[str, int] = defaultdict(int)
    for sid in scene_ids:
        tallies[domain_of.get(sid, "unknown")] += 1
    return dict(sorted(tallies.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30, help="stratified scene budget")
    parser.add_argument("--patches-per-scene", type=float, default=12.0)
    parser.add_argument("--reads-per-patch", type=int, default=3)
    parser.add_argument(
        "--domains-json",
        default=None,
        help="JSON map domain -> scene_id list (for dry-run allocation)",
    )
    args = parser.parse_args(argv)

    estimate = CostEstimate(
        scenes=args.budget,
        patches_per_scene=args.patches_per_scene,
        reads_per_patch=args.reads_per_patch,
    )
    print(
        json.dumps(
            {
                "cost_estimate": {
                    "scenes": estimate.scenes,
                    "patches_per_scene": estimate.patches_per_scene,
                    "reads_per_patch": estimate.reads_per_patch,
                    "total_feature_reads": estimate.total_feature_reads,
                },
                "note": (
                    "Arithmetic vs T4 comparison runs two pooling passes per patch; "
                    "double the pooling CPU, not necessarily the IO."
                ),
            },
            indent=2,
        )
    )

    if args.domains_json:
        domains = json.loads(args.domains_json)
        if not isinstance(domains, dict):
            raise SystemExit("domains-json must be an object")
        sample = stratified_scene_sample(
            [sid for ids in domains.values() for sid in ids],
            domains={str(k): list(v) for k, v in domains.items()},
            budget=args.budget,
        )
        print(json.dumps({"stratified_sample": sample}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
