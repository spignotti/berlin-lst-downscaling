"""Score + Tie-Break + Verwurf logic for Landsat-S2 coupling."""

from __future__ import annotations

import math


def couple_all(
    anchors: list[dict],
    s2_candidates_by_anchor: dict[str, list[dict]],
    cfg,
) -> tuple[list[dict], list[dict]]:
    """Couple all anchors with their S2 candidates.

    Returns (coupled_pairs, dropped_pairs) as lists of dicts.
    """
    coupled: list[dict] = []
    dropped: list[dict] = []

    for anchor in anchors:
        candidates = s2_candidates_by_anchor.get(anchor["scene_id"], [])
        result = couple_one_anchor(anchor, candidates, cfg)
        if result["status"] == "coupled":
            coupled.append(result)
        else:
            dropped.append(result)

    return coupled, dropped

def couple_one_anchor(
    anchor: dict,
    s2_candidates: list[dict],
    cfg,
) -> dict:
    """Apply score + tie-break to select best S2 or drop the anchor.

    Score formula (per Notion spec):
        score = clear_frac − λ · (Δt / 3)

    Tie-Break: highest score wins; if tied, smaller Δt wins.
    Drop rule: if no S2 candidate has clear_frac ≥ ``min_clear_frac``,
    the anchor is dropped.
    """
    if not s2_candidates:
        return {
            "status": "dropped",
            "anchor": anchor,
            "reason": "no_s2_in_window",
            "max_clear_frac": 0.0,
            "s2": None,
            "clear_frac": None,
            "score": None,
            "ecostress": [],
        }

    lam = getattr(cfg.sentinel2.score, "lambda", 0.1)
    min_cf = cfg.sentinel2.min_clear_frac

    # Score each candidate (clear_frac pre-computed and stored on the candidate)
    scored: list[tuple[dict, float]] = []
    for c in s2_candidates:
        cf = c.get("clear_frac")
        if cf is None or math.isnan(cf):
            continue
        score = cf - lam * (c["dt_days"] / 3.0)
        scored.append((c, score))

    # Sort: descending score, then ascending dt_days, then scene_id (final tie-break)
    scored.sort(key=lambda x: (-x[1], x[0]["dt_days"], x[0]["scene_id"]))

    max_clear_frac = max((c["clear_frac"] for c, _ in scored), default=0.0)

    if not scored or max_clear_frac < min_cf:
        return {
            "status": "dropped",
            "anchor": anchor,
            "reason": "no_s2_above_threshold",
            "max_clear_frac": max_clear_frac,
            "s2": None,
            "clear_frac": None,
            "score": None,
            "ecostress": [],
        }

    best_candidate, best_score = scored[0]
    l_clear = best_candidate.get("landsat_clear_px") or 0
    j_clear = best_candidate.get("joint_clear_px") or 0
    j_frac = best_candidate["clear_frac"]
    return {
        "status": "coupled",
        "anchor": anchor,
        "s2": best_candidate,
        "clear_frac": j_frac,
        "landsat_clear_px": l_clear,
        "joint_clear_px": j_clear,
        "joint_clear_frac": j_frac,
        "score": best_score,
        "ecostress": [],  # filled in by ecostress_subset step
    }