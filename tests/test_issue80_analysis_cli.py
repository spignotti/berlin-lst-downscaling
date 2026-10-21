"""Issue #80: compare_training_aggregation CLI helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "operators" / "compare_training_aggregation.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_training_aggregation", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_largest_remainder_allocation_sums_to_total() -> None:
    mod = _load_compare_module()
    weights = [1.0, 1.0, 1.0]
    alloc = mod.largest_remainder_allocation(weights, 10)
    assert sum(alloc) == 10
    assert sorted(alloc) == [3, 3, 4]


def test_cli_cost_estimate_json() -> None:
    mod = _load_compare_module()
    assert mod.main(["--budget", "30"]) == 0
