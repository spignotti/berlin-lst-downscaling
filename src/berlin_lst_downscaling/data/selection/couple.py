"""Score + Tie-Break + Verwurf logic for Landsat-S2 coupling."""

from __future__ import annotations

from berlin_lst_downscaling.data.selection import (
    Anchor,
    CoupledPair,
    DroppedPair,
    S2Candidate,
)


def couple_all(
    anchors: list[Anchor],
    s2_candidates_by_anchor: dict[str, list[S2Candidate]],
    cfg,
) -> tuple[list[CoupledPair], list[DroppedPair]]:
    """Couple all anchors with their S2 candidates.

    Parameters
    ----------
    anchors :
        All Landsat anchors (from ``build_anchors``).
    s2_candidates_by_anchor :
        Dict mapping ``anchor.scene_id`` → list of S2 candidates.
    cfg :
        Hydra config with ``sentinel2.score.lambda`` and
        ``sentinel2.min_clear_frac``.

    Returns
    -------
    tuple[list[CoupledPair], list[DroppedPair]]
        (coupled pairs, dropped pairs).
    """
    coupled: list[CoupledPair] = []
    dropped: list[DroppedPair] = []

    for anchor in anchors:
        candidates = s2_candidates_by_anchor.get(anchor.scene_id, [])
        result = couple_one_anchor(anchor, candidates, cfg)
        if isinstance(result, CoupledPair):
            coupled.append(result)
        else:
            dropped.append(result)

    return coupled, dropped


def couple_one_anchor(
    anchor: Anchor,
    s2_candidates: list[S2Candidate],
    cfg,
) -> CoupledPair | DroppedPair:
    """Apply score + tie-break to select best S2 or drop the anchor.

    Score formula (per Notion spec):
        score = clear_frac − λ · (Δt / 3)

    Tie-Break: highest score wins; if tied, smaller Δt wins.
    Drop rule: if no S2 candidate has clear_frac ≥ ``min_clear_frac``,
    the anchor is dropped.
    """
    if not s2_candidates:
        return DroppedPair(
            anchor=anchor,
            reason="no_s2_in_window",
            max_clear_frac=0.0,
        )

    lam = getattr(cfg.sentinel2.score, "lambda", 0.1)  # lambda is a reserved keyword; use getattr
    min_cf = cfg.sentinel2.min_clear_frac

    # Score each candidate (clear_frac pre-computed and stored on the candidate)
    scored: list[tuple[S2Candidate, float]] = []
    for c in s2_candidates:
        # clear_frac may be NaN if computation failed
        cf = getattr(c, "clear_frac", None)
        if cf is None or cf != cf:  # NaN guard
            cf = 0.0
        score = cf - lam * (c.dt_days / 3.0)
        scored.append((c, score))

    # Sort: descending score, then ascending dt_days (tie-break)
    scored.sort(key=lambda x: (-x[1], x[0].dt_days))
    best_candidate, best_score = scored[0]

    max_clear_frac = max(
        (getattr(c, "clear_frac", 0.0) or 0.0) for c, _ in scored
    )

    if max_clear_frac < min_cf:
        return DroppedPair(
            anchor=anchor,
            reason="no_s2_above_threshold",
            max_clear_frac=max_clear_frac,
        )

    return CoupledPair(
        anchor=anchor,
        s2=best_candidate,
        clear_frac=getattr(best_candidate, "clear_frac", 0.0) or 0.0,
        score=best_score,
        ecostress=[],  # filled in by ecostress_subset step
    )
