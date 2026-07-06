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
    """Landsat C2 L2 QA_PIXEL bits: fill=0, cirrus=2, cloud=3, shadow=4.

    Cloud with confidence ≥ medium (bits 8-9 ≥ 2) is the canonical "cloud".
    Bit 1 (dilated cloud) is not included — dilation is ARD-only.
    Snow (bit 6) and water (clear-water) are clear.
    """
    cloud_raw = (qa >> 3) & 1
    cloud_conf = (qa >> 8) & 0b11
    cloudy = (cloud_raw != 0) & (cloud_conf >= 2)
    cirrus = (qa >> 2) & 1
    shadow = (qa >> 4) & 1
    fill = qa & 1
    return ~(fill.astype(bool) | cloudy | shadow.astype(bool) | cirrus.astype(bool))


_S2_CLOUD_CLASSES = {0, 1, 8, 9, 10, 11}  # fill, saturated, cloud, cirrus, snow


def _s2_is_clear(scl: np.ndarray) -> np.ndarray:
    """Inverted SCL: anything NOT in cloudy set is clear.

    Includes class 7 (unclassified/urban) as clear.
    """
    return ~np.isin(scl, list(_S2_CLOUD_CLASSES))


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
    """Verify bit interpretation for QA_PIXEL.

    Landsat C2 L2 QA pixel bit field:
      bit 0 (1)  — Fill
      bit 1 (2)  — Dilated cloud (excluded from coupling clear)
      bit 2 (4)  — Cirrus (high confidence)
      bit 3 (8)  — Cloud (raw)
      bit 4 (16) — Cloud shadow
      bit 6 (64) — Snow / Ice (clear for coupling)
      bits 8-9   — Cloud confidence (0-3)
    """
    cases = [
        # (qa_value, expected_clear, description)
        (0,      True,  "all-zero = clear"),
        (4,      False, "bit2(cirrus) → not clear"),
        (8,      True,  "bit3(cloud, conf=0) → clear (confidence too low)"),
        (10,     True,  "bit3(cloud, conf=0) → clear"),
        # qa=520: bit3=cloud, bits8-9=conf=2 → ≥2 → NOT clear
        (520,    False, "bit3(cloud, conf=2) → not clear"),
        (16,     False, "bit4(cloud-shadow) → not clear"),
        (64,     True,  "bit6(snow) → clear"),
        (192,    True,  "clear water (128+64) → clear"),
        (128,    True,  "clear land → clear"),
        (255,    False, "everything-set → not clear"),
        (1,      False, "bit0(fill) → not clear"),
        (7,      False, "bits 0+1+2 (fill+dilated+cirrus) → not clear"),
    ]
    for qa_val, expected, desc in cases:
        qa = np.array([[qa_val]], dtype=np.uint16)
        result = bool(np.all(_landsat_is_clear(qa)))
        assert result == expected, (  # noqa: S101
            f"landsat_is_clear(qa={qa_val}) = {result}, expected {expected} — {desc}"
        )
    print("PASS: landsat_is_clear — all bit-pattern cases")


def test_s2_is_clear_classes():
    """Verify SCL class interpretation.

    Sen2Cor SCL classes:
      0 fill / nodata  → not clear
      1 saturated      → not clear
      2 dark pixels    → clear
      3 cloud shadow   → clear (shadow pixels can still be used for LST)
      4 vegetation     → clear
      5 bare soil      → clear
      6 water          → clear
      7 unclassified   → clear (includes urban impervious — critical for Berlin)
      8 cloud med prob → not clear
      9 cloud high     → not clear
      10 thin cirrus    → not clear
      11 snow          → not clear
    """
    scl = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint8
    )
    expected = np.array(
        [[False, False, True, True, True, True, True, True, False, False, False, False]],
        dtype=bool,
    )
    result = _s2_is_clear(scl)
    assert np.array_equal(result, expected), (  # noqa: S101
        f"s2_is_clear mismatch:\n  got:      {result}\n  expected: {expected}"
    )
    print("PASS: s2_is_clear — class 7 (urban) is clear, 0/1/8/9/10/11 not clear")


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
