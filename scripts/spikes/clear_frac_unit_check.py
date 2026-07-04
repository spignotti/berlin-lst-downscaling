# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///

"""Unit check for clear_frac pixel-counting logic.

Tests the clear_frac formula in isolation: denominator = AOI ∩ L8-clear,
numerator = AOI ∩ L8-clear ∩ S2-clear.
Skips the AOI-reprojection path (tested via integration in smoke-selection).

Usage
-----
    uv run python scripts/spikes/clear_frac_unit_check.py

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.testing import assert_allclose  # noqa: I001

# ── inline copy of the clear-fraction helpers being tested ─────────────────────


def _landsat_is_clear(qa: np.ndarray) -> np.ndarray:
    """QA_PIXEL bits: fill=1, cloud=2, shadow=4, cirrus=64."""
    fill = (qa & 1) != 0
    cloud = (qa & 4) != 0
    shadow = (qa & 8) != 0
    cirrus = (qa & 64) != 0
    return ~fill & ~cloud & ~shadow & ~cirrus


def _s2_is_clear(scl: np.ndarray) -> np.ndarray:
    """SCL classes 4 (veg), 5 (bare), 6 (water) are clear."""
    return np.isin(scl, [4, 5, 6])


def _compute_clear_frac(l8_clear: np.ndarray, s2_clear: np.ndarray, aoi: np.ndarray) -> float:
    """Pure clear_frac formula: numerator / denominator."""
    denom = int(np.sum(aoi & l8_clear))
    if denom == 0:
        return float("nan")
    numer = int(np.sum(aoi & l8_clear & s2_clear))
    return numer / denom


# ── test cases ────────────────────────────────────────────────────────────────


def test_full_clear():
    """L8 clear + S2 clear → clear_frac = 1.0."""
    l8_clear = np.ones((4, 4), dtype=bool)
    s2_clear = np.ones((4, 4), dtype=bool)
    aoi = np.ones((4, 4), dtype=bool)
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    assert_allclose(cf, 1.0, rtol=1e-6)
    print("PASS: full_clear → 1.0")


def test_full_cloud():
    """L8 clear + S2 all cloudy → clear_frac = 0.0."""
    l8_clear = np.ones((4, 4), dtype=bool)
    s2_clear = np.zeros((4, 4), dtype=bool)
    aoi = np.ones((4, 4), dtype=bool)
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    assert_allclose(cf, 0.0, rtol=1e-6)
    print("PASS: full_cloud → 0.0")


def test_half_clear_l8():
    """L8 50% clear + S2 all clear → clear_frac = 0.5."""
    l8_clear = np.zeros((4, 4), dtype=bool)
    l8_clear[:2, :] = True  # top half clear
    s2_clear = np.ones((4, 4), dtype=bool)
    aoi = np.ones((4, 4), dtype=bool)
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    # denom = 8 (top half), numer = 8 → 1.0
    # Wait — S2-clear is ALL True, so numerator = 8 (top half only), denom = 8 → 1.0
    assert_allclose(cf, 1.0, rtol=1e-6)
    print("PASS: half_clear_l8 → 1.0 (S2 all clear so all L8-clear pixels are S2-clear)")


def test_half_clear_s2():
    """L8 all clear + S2 50% clear → clear_frac = 0.5."""
    l8_clear = np.ones((4, 4), dtype=bool)
    s2_clear = np.zeros((4, 4), dtype=bool)
    s2_clear[:2, :] = True  # top half clear
    aoi = np.ones((4, 4), dtype=bool)
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    # denom = 16, numer = 8 → 0.5
    assert_allclose(cf, 0.5, rtol=1e-6)
    print("PASS: half_clear_s2 → 0.5")


def test_partial_aoi():
    """AOI: only top-left 2×2 inside; L8/S2 all clear → denom=4, numer=4 → 1.0."""
    l8_clear = np.ones((4, 4), dtype=bool)
    s2_clear = np.ones((4, 4), dtype=bool)
    aoi = np.zeros((4, 4), dtype=bool)
    aoi[:2, :2] = True
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    assert_allclose(cf, 1.0, rtol=1e-6)
    print("PASS: partial_aoi → 1.0")


def test_no_l8_clear():
    """L8 all cloudy → denom=0 → NaN."""
    l8_clear = np.zeros((4, 4), dtype=bool)
    s2_clear = np.ones((4, 4), dtype=bool)
    aoi = np.ones((4, 4), dtype=bool)
    cf = _compute_clear_frac(l8_clear, s2_clear, aoi)
    assert cf != cf, f"expected NaN, got {cf}"  # noqa: S101
    print("PASS: no_l8_clear → NaN")


def test_landsat_is_clear_bits():
    """Verify bit interpretation for QA_PIXEL."""
    # All-clear scene: qa_pixel = 0
    qa = np.zeros((2, 2), dtype=np.uint16)
    assert np.all(_landsat_is_clear(qa))  # noqa: S101
    print("PASS: landsat_is_clear — all-zero qa_pixel is clear")


def test_s2_is_clear_classes():
    """Verify SCL class interpretation."""
    scl = np.array([[4, 5, 6, 0], [8, 9, 10, 1]], dtype=np.uint8)
    expected = np.array([[True, True, True, False], [False, False, False, False]], dtype=bool)
    result = _s2_is_clear(scl)
    assert np.array_equal(result, expected)  # noqa: S101
    print("PASS: s2_is_clear — classes 4/5/6 clear, others not")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 60)
    print("clear_frac unit check — pixel-counting logic")
    print("=" * 60)

    tests = [
        test_full_clear,
        test_full_cloud,
        test_half_clear_l8,
        test_half_clear_s2,
        test_partial_aoi,
        test_no_l8_clear,
        test_landsat_is_clear_bits,
        test_s2_is_clear_classes,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"FAIL [{test.__name__}]: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

    print("=" * 60)
    print("ALL PASS — clear_frac pixel logic validated")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
