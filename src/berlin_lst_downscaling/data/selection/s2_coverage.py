"""Sentinel-2 mosaic AOI footprint coverage estimate for selection.

Selection scores S2 candidates by ``clear_frac`` — clear S2 pixels among
Landsat-clear AOI pixels — while ARD v10 gates on pre-cloud SCL footprint
coverage of the full AOI (``max_missing_aoi_fraction``, default 0.005). A
candidate with a large footprint hole can therefore pass selection yet fail
the ARD gate. This module mirrors the ARD footprint rule on synthetic,
offline-testable arrays so future selections can annotate (always) and gate
(opt-in) candidates before coupling.

The default keeps legacy behavior: annotation only. When
``sentinel2.mosaic.max_missing_aoi_fraction`` is explicitly configured,
candidates above the gate are dropped with reason
``s2_footprint_coverage_gate`` instead of being coupled. A selection rerun
is not required for the current release.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DROP_REASON = "s2_footprint_coverage_gate"


def footprint_missing_fraction(footprint: np.ndarray, aoi: np.ndarray) -> dict[str, Any]:
    """Mirror ARD's pre-cloud footprint-vs-AOI coverage on boolean masks.

    ``footprint`` is True where any recorded member has finite, nonzero SCL;
    ``aoi`` is True inside the Berlin AOI. Both must share shape.
    """
    foot = np.asarray(footprint, dtype=bool)
    inside = np.asarray(aoi, dtype=bool)
    if foot.shape != inside.shape:
        raise ValueError("footprint and AOI masks must share shape")
    total = int(np.count_nonzero(inside))
    if total == 0:
        raise ValueError("AOI mask is empty")
    missing = int(np.count_nonzero(inside & ~foot))
    fraction = missing / total
    return {
        "aoi_total_px": total,
        "aoi_missing_px": missing,
        "aoi_covered_px": total - missing,
        "missing_aoi_fraction": fraction,
    }


def mosaic_threshold(cfg: Any) -> float | None:
    """Return the opt-in selection footprint gate, or None (annotate only).

    Reads ``sentinel2.mosaic.max_missing_aoi_fraction`` when explicitly set.
    Absent keys preserve legacy behavior and legacy policy hashes.
    """
    try:
        sentinel2 = cfg.get("sentinel2", {})
        mosaic = sentinel2.get("mosaic", {}) if hasattr(sentinel2, "get") else {}
    except AttributeError:
        return None
    try:
        keys = set(dict(mosaic).keys()) if not isinstance(mosaic, dict) else set(mosaic.keys())
    except (TypeError, ValueError):
        return None
    if "max_missing_aoi_fraction" not in keys:
        return None
    try:
        value = mosaic.get("max_missing_aoi_fraction")
    except AttributeError:
        return None
    from berlin_lst_downscaling.data.ard.s2_mosaic import validate_missing_fraction

    return validate_missing_fraction(value)


def partition_by_footprint_gate(
    candidates: list[dict], threshold: float | None
) -> tuple[list[dict], list[dict]]:
    """Split annotated candidates into ``(passing, gated)``.

    Candidates without a numeric ``s2_aoi_missing_frac`` always pass (the
    estimate is best-effort and must never drop a candidate silently on
    missing diagnostics).
    """
    if threshold is None:
        return list(candidates), []
    passing: list[dict] = []
    gated: list[dict] = []
    for candidate in candidates:
        missing_frac = candidate.get("s2_aoi_missing_frac")
        if missing_frac is None or not np.isfinite(float(missing_frac)):
            passing.append(candidate)
        elif float(missing_frac) <= float(threshold):
            passing.append(candidate)
        else:
            gated.append(candidate)
    return passing, gated


__all__ = [
    "DROP_REASON",
    "footprint_missing_fraction",
    "mosaic_threshold",
    "partition_by_footprint_gate",
]
