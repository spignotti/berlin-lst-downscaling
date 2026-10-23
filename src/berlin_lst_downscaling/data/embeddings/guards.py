"""TESSERA-arm evaluation guards — reuse the existing eval path.

The TESSERA arm trains 2017–2023, validates on 2024, and tests on 2025
(the same split as the main model). 2026 is excluded for this arm with an
explicit guard: TESSERA has no 2026 calendar-year embedding and the arm is
never used for 2026 inference.
"""

from __future__ import annotations

from berlin_lst_downscaling.data.embeddings.tessera import (
    TesseraYearError,
    validate_tessera_year,
)
from berlin_lst_downscaling.data.training.contracts import split_for_year


def assert_tessera_eval_year(year: int) -> str:
    """Validate an evaluation year for the TESSERA arm.

    Returns the temporal split (``validation`` for 2024, ``test`` for
    2025). Training years pass through with their ``train`` label so the
    same helper guards data loading; 2026 and any non-contract year raise.
    """
    if int(year) == 2026:
        raise TesseraYearError(
            "TESSERA arm excludes 2026: no 2026 embedding exists and the arm "
            "is never used for 2026 inference"
        )
    validate_tessera_year(int(year))
    split = split_for_year(int(year))
    if split not in ("train", "validation", "test"):
        raise TesseraYearError(f"TESSERA arm cannot evaluate split {split!r} for {year}")
    if split == "train":
        return split
    if int(year) == 2024 and split != "validation":
        raise TesseraYearError(f"2024 must be validation for the TESSERA arm, got {split!r}")
    if int(year) == 2025 and split != "test":
        raise TesseraYearError(f"2025 must be test for the TESSERA arm, got {split!r}")
    return split


def assert_tessera_not_2026(year: int) -> None:
    """Refuse any TESSERA-arm use for 2026 (inference or eval)."""
    if int(year) == 2026:
        raise TesseraYearError(
            "TESSERA arm excludes 2026: no 2026 embedding exists and the arm "
            "is never used for 2026 inference"
        )


__all__ = [
    "assert_tessera_eval_year",
    "assert_tessera_not_2026",
]
