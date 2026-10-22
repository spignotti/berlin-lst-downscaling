"""Kelvin temperature aggregation for priors and 10 m → 100 m pooling (issue #80).

Arithmetic block means stay the default. T4 applies only to physical Kelvin
fields: ``(mean(T**4))**0.25`` over contributing cells. Feature channels and
scaled predictors always use arithmetic pooling elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from berlin_lst_downscaling.modeling.contracts import POOL_FACTOR

AggregationMethod = Literal["arithmetic", "t4"]
AGGREGATION_ARITHMETIC: AggregationMethod = "arithmetic"
AGGREGATION_T4: AggregationMethod = "t4"
VALID_AGGREGATION_METHODS: frozenset[str] = frozenset({AGGREGATION_ARITHMETIC, AGGREGATION_T4})


class AggregationContractError(ValueError):
    """Raised when temperature aggregation inputs violate the Kelvin contract."""


def validate_aggregation_method(value: object, *, label: str = "aggregation") -> AggregationMethod:
    """Reject nulls, booleans, unknown strings, and wrong types."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    if value is None:
        raise ValueError(f"{label} must not be null")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    method = value.strip().lower()
    if method not in VALID_AGGREGATION_METHODS:
        raise ValueError(
            f"{label} must be one of {sorted(VALID_AGGREGATION_METHODS)}, got {value!r}"
        )
    return method  # type: ignore[return-value]


def resolve_aggregation_method(
    value: object | None, *, label: str = "aggregation"
) -> AggregationMethod:
    """Return arithmetic when ``value`` is omitted (legacy checkpoints and reports)."""
    if value is None:
        return AGGREGATION_ARITHMETIC
    return validate_aggregation_method(value, label=label)


def _optional_aggregation_field(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def resolve_report_aggregation(report: Mapping[str, Any]) -> AggregationMethod:
    """Read aggregation from a baseline or RF report payload.

    Current writers store the method under ``source.aggregation``. A legacy
    top-level ``aggregation`` field is accepted when it agrees with ``source``;
    when both are set and differ, this raises. When neither is set, historical
    artifacts default to arithmetic.
    """
    top = _optional_aggregation_field(report.get("aggregation"))
    source = report.get("source")
    nested: str | None = None
    if isinstance(source, Mapping):
        nested = _optional_aggregation_field(source.get("aggregation"))
    if top is not None and nested is not None and top != nested:
        raise ValueError(f"report aggregation mismatch: top-level {top!r} != source {nested!r}")
    return resolve_aggregation_method(top if top is not None else nested)


def min_eligible_cells_from_fraction(fraction: float) -> int:
    """Convert a fractional admission threshold to an inclusive integer minimum."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("min_eligible_fraction must be a finite real number")
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"min_eligible_fraction must be in (0, 1], got {fraction!r}")
    return math.ceil(float(fraction) * 256)


def _require_positive_kelvin(values: Tensor, *, context: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise AggregationContractError(f"{context}: contributing values must be finite Kelvin")
    if bool((values <= 0).any()):
        raise AggregationContractError(f"{context}: contributing values must be > 0 K for T4")


def pool_10m_to_100m(
    prediction_10m: Tensor,
    method: AggregationMethod = AGGREGATION_ARITHMETIC,
    mask_100m: Tensor | None = None,
) -> Tensor:
    """Exact nested 10×10 reduction on the canonical 100 m grid.

    ``method='arithmetic'`` is the block mean. A block that contains any NaN
    pools to NaN: the NaN is not dropped and is not replaced by zero, and it
    does not change a neighboring finite block. ``method='t4'`` applies the
    fourth-root-of-mean-fourth-powers reduction on physical Kelvin inputs and
    rejects a non-finite contributing pixel. When ``mask_100m`` is supplied, the positive-finite
    Kelvin contract applies only to 10 m pixels inside blocks whose 100 m
    supervision cell is valid; excluded blocks are sanitized before ``pow`` so
    they cannot disturb gradients.

    Extents that are not divisible by ``POOL_FACTOR`` raise. An incomplete edge
    (the leftover rows or columns past the last full 10×10 block) is never
    reduced as a smaller block and never enters the pooled map.
    """
    method = validate_aggregation_method(method)
    b, c, h, w = prediction_10m.shape
    if h % POOL_FACTOR or w % POOL_FACTOR:
        raise ValueError(f"prediction extent ({h}, {w}) is not divisible by {POOL_FACTOR}")
    blocks = prediction_10m.reshape(
        b, c, h // POOL_FACTOR, POOL_FACTOR, w // POOL_FACTOR, POOL_FACTOR
    )
    if method == AGGREGATION_ARITHMETIC:
        return blocks.mean(dim=(3, 5))
    hb, wb = h // POOL_FACTOR, w // POOL_FACTOR
    with torch.autocast(device_type=prediction_10m.device.type, enabled=False):
        tiles = blocks.permute(0, 1, 2, 4, 3, 5).float()
        if mask_100m is not None:
            if mask_100m.shape != (b, c, hb, wb):
                raise ValueError(
                    f"mask_100m shape {tuple(mask_100m.shape)} does not match pooled grid "
                    f"{(b, c, hb, wb)}"
                )
            block_valid = mask_100m.bool().reshape(b, c, hb, wb)
            if bool(block_valid.any()):
                contributing = tiles[block_valid]
                _require_positive_kelvin(
                    contributing,
                    context="10 m → 100 m T4 pooling (contributing block)",
                )
            safe = torch.where(
                block_valid.unsqueeze(-1).unsqueeze(-1),
                tiles,
                torch.ones_like(tiles),
            )
            powered = safe.pow(4).mean(dim=(-2, -1))
        else:
            _require_positive_kelvin(tiles, context="10 m → 100 m T4 pooling")
            powered = tiles.pow(4).mean(dim=(-2, -1))
        return powered.pow(0.25)


def pool_block_means_from_accumulators(
    sum_t: np.ndarray,
    sum_t4: np.ndarray | None,
    counts: np.ndarray,
    *,
    method: AggregationMethod = AGGREGATION_ARITHMETIC,
) -> np.ndarray:
    """Build float32 block means from float64 accumulators and per-block counts."""
    method = validate_aggregation_method(method)
    means = np.full(sum_t.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    if method == AGGREGATION_ARITHMETIC:
        means[ok] = sum_t[ok] / counts[ok]
        return means.astype(np.float32)
    if sum_t4 is None:
        raise ValueError("T4 prior aggregation requires sum_t4 accumulators")
    powered = sum_t4[ok] / counts[ok]
    if np.any(powered <= 0) or np.any(~np.isfinite(powered)):
        raise AggregationContractError("prior T4 block: non-positive or non-finite mean(T**4)")
    means[ok] = np.power(powered, 0.25)
    return means.astype(np.float32)


__all__ = [
    "AGGREGATION_ARITHMETIC",
    "AGGREGATION_T4",
    "AggregationContractError",
    "AggregationMethod",
    "VALID_AGGREGATION_METHODS",
    "min_eligible_cells_from_fraction",
    "pool_10m_to_100m",
    "pool_block_means_from_accumulators",
    "resolve_aggregation_method",
    "resolve_report_aggregation",
    "validate_aggregation_method",
]
