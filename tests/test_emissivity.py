"""Unit tests for the emissivity-diagnostic physics (offline, synthetic)."""

from __future__ import annotations

import unittest

import numpy as np

from berlin_lst_downscaling.analysis.emissivity import (
    ATRAN_SCALE,
    EMIS_SCALE,
    K1_L8_B10,
    K1_L9_B10,
    K2_L8_B10,
    K2_L9_B10,
    NDVI_BIN_LABELS,
    QA_PIXEL_WATER_BIT,
    ST_SATURATED_DN,
    THERMAL_RADIANCE_SCALE,
    assign_imperv_bins,
    assign_ndvi_bins,
    aster_ged_b10_proxy,
    aux_dn_to_units,
    binned_stats,
    forward_trad,
    ged_b10_on_scene_grid,
    lst_from_aux,
    ndvi,
    ndvi_lst_relation,
    olisr_dn_to_reflectance,
    planck_forward_radiance,
    planck_invert_lst,
    pool_overall_stats,
    qa_water,
    st_dn_to_kelvin,
    surface_leaving_radiance,
    thermal_constants_for_platform,
    thermal_saturated,
    uniform_static_emissivity,
)


def _synthetic_scene(seed: int = 0) -> dict[str, np.ndarray]:
    """Build a physically consistent synthetic scene on a 24x24 grid."""
    rng = np.random.default_rng(seed)
    shape = (24, 24)
    lst_true = 290.0 + 12.0 * rng.random(shape)  # 290-302 K
    emis = 0.96 + 0.03 * rng.random(shape)  # 0.96-0.99
    atran = 0.75 + 0.15 * rng.random(shape)
    urad = 1.0 + 1.5 * rng.random(shape)
    drad = 1.5 + 1.5 * rng.random(shape)
    k1, k2 = K1_L9_B10, K2_L9_B10
    trad = forward_trad(lst_true, urad, drad, atran, emis, k1, k2)
    return {
        "lst_true": lst_true,
        "emis": emis,
        "atran": atran,
        "urad": urad,
        "drad": drad,
        "trad": trad,
        "k1": np.array(k1),
        "k2": np.array(k2),
    }


class PlanckInversionTest(unittest.TestCase):
    def test_forward_inverse_round_trip(self) -> None:
        temps = np.array([270.0, 285.0, 300.0, 315.0, 330.0])
        for k1, k2 in ((K1_L8_B10, K2_L8_B10), (K1_L9_B10, K2_L9_B10)):
            rad = planck_forward_radiance(temps, k1, k2)
            back = planck_invert_lst(rad, k1, k2)
            np.testing.assert_allclose(back, temps, rtol=1e-9)

    def test_known_landsat8_value(self) -> None:
        # BT reference: B10 radiance 9.0 W/(m^2.sr.um) inverts to ~295.7 K
        # (K2 / ln(K1 / 9 + 1) with the L8 Band 10 constants).
        out = planck_invert_lst(np.array([9.0]), K1_L8_B10, K2_L8_B10)
        self.assertAlmostEqual(float(out[0]), 295.7, delta=0.2)

    def test_nonpositive_radiance_is_nan_not_clamped(self) -> None:
        out = planck_invert_lst(np.array([0.0, -1.0, np.nan]), K1_L9_B10, K2_L9_B10)
        self.assertTrue(bool(np.all(~np.isfinite(out))))

    def test_degenerate_emissivity_is_nan(self) -> None:
        shape = (4, 4)
        ones = np.ones(shape)
        bad_eps = np.array([[0.0, -0.5, 1.5, np.nan]] * 4, dtype=np.float64).reshape(shape)
        out = surface_leaving_radiance(ones * 8, ones, ones * 2, ones * 0.8, bad_eps)
        self.assertTrue(bool(np.all(~np.isfinite(out))))


class SyntheticRoundTripTest(unittest.TestCase):
    def test_sanity_identity_recovers_true_lst(self) -> None:
        """Forward-model TRAD from known LST, then invert with true emissivity."""
        s = _synthetic_scene()
        k1, k2 = float(s["k1"]), float(s["k2"])
        recovered = lst_from_aux(s["trad"], s["urad"], s["drad"], s["atran"], s["emis"], k1, k2)
        diff = np.abs(recovered - s["lst_true"])
        self.assertLess(float(np.nanmax(diff)), 1e-6)
        self.assertLess(float(np.nanmean(diff)), 1e-9)

    def test_wrong_emissivity_shifts_lst_by_order_of_literature_sensitivity(self) -> None:
        """A 0.01 emissivity error must move LST by ~0.5-1 K (literature range)."""
        s = _synthetic_scene()
        k1, k2 = float(s["k1"]), float(s["k2"])
        base = lst_from_aux(s["trad"], s["urad"], s["drad"], s["atran"], s["emis"], k1, k2)
        shifted = lst_from_aux(
            s["trad"], s["urad"], s["drad"], s["atran"], s["emis"] - 0.01, k1, k2
        )
        delta = shifted - base
        mean_shift = float(np.nanmean(delta))
        self.assertGreater(mean_shift, 0.2)
        self.assertLess(mean_shift, 2.0)


class ScalingTest(unittest.TestCase):
    def test_st_scale_matches_usgs_equation(self) -> None:
        dn = np.array([[293, 61440, 0]], dtype=np.uint16)
        kelvin = st_dn_to_kelvin(dn)
        self.assertAlmostEqual(float(kelvin[0, 0]), 293 * 0.00341802 + 149.0, places=9)
        self.assertTrue(bool(np.isnan(kelvin[0, 2])))

    def test_aux_nodata_maps_to_nan(self) -> None:
        dn = np.array([[-9999, 8000]], dtype=np.int16)
        out = aux_dn_to_units(dn, THERMAL_RADIANCE_SCALE)
        self.assertTrue(bool(np.isnan(out[0, 0])))
        self.assertAlmostEqual(float(out[0, 1]), 8.0, places=9)
        emis = aux_dn_to_units(np.array([[9500]], dtype=np.int16), EMIS_SCALE)
        self.assertAlmostEqual(float(emis[0, 0]), 0.95, places=9)
        atran = aux_dn_to_units(np.array([[8500]], dtype=np.int16), ATRAN_SCALE)
        self.assertAlmostEqual(float(atran[0, 0]), 0.85, places=9)

    def test_oli_reflectance_scale(self) -> None:
        dn = np.array([[10000]], dtype=np.uint16)
        self.assertAlmostEqual(float(olisr_dn_to_reflectance(dn)[0, 0]), 0.075, places=9)

    def test_platform_constants(self) -> None:
        self.assertEqual(thermal_constants_for_platform("landsat-8"), (K1_L8_B10, K2_L8_B10))
        self.assertEqual(thermal_constants_for_platform("landsat-9"), (K1_L9_B10, K2_L9_B10))
        with self.assertRaises(ValueError):
            thermal_constants_for_platform("landsat-7")


class MasksAndStatsTest(unittest.TestCase):
    def test_ndvi_and_bins(self) -> None:
        nir = np.array([0.4, 0.05, 0.3, 0.0])
        red = np.array([0.1, 0.05, 0.3, 0.0])
        v = ndvi(nir, red)
        self.assertAlmostEqual(float(v[0]), 0.6, places=9)
        self.assertAlmostEqual(float(v[1]), 0.0, places=9)
        self.assertTrue(bool(np.isnan(v[3])))
        bins = assign_ndvi_bins(np.array([-0.5, 0.1, 0.3, 0.5, 0.9]))
        self.assertEqual(list(bins), [0, 1, 2, 3, 4])
        self.assertEqual(len(NDVI_BIN_LABELS), 5)

    def test_tirs_saturation_is_max_dn_not_a_radsat_bit(self) -> None:
        # LSDS-1619 v6 Table 6-4: QA_RADSAT bits 9-10 are unused for L8/9
        # (bit 8 is OLI band 9, bit 11 terrain occlusion), so TIRS
        # saturation is read from the ST DN itself (§6.1.2).
        dn = np.array([[0, 1000, 65534, ST_SATURATED_DN]], dtype=np.uint16)
        mask = thermal_saturated(dn)
        self.assertEqual(list(mask[0]), [False, False, False, True])
        self.assertEqual(ST_SATURATED_DN, 65535)

    def test_qa_water_bit(self) -> None:
        qa = np.array([[0, 1 << QA_PIXEL_WATER_BIT, (1 << QA_PIXEL_WATER_BIT) | (1 << 3)]])
        mask = qa_water(qa.astype(np.uint16))
        self.assertEqual(list(mask[0]), [False, True, True])

    def test_binned_stats_and_relation(self) -> None:
        values = np.array([1.0, 3.0, 2.0, 100.0])
        bins = np.array([0, 0, 1, 1])
        stats = binned_stats(values, bins, 3)
        self.assertEqual(stats[0]["n"], 2)
        self.assertAlmostEqual(stats[0]["mean"], 2.0)
        self.assertEqual(stats[2]["n"], 0)
        x = np.linspace(0.1, 0.8, 50)
        y = 300.0 - 15.0 * x
        rel = ndvi_lst_relation(x, y)
        self.assertAlmostEqual(rel["slope"], -15.0, places=9)
        self.assertAlmostEqual(rel["r"], -1.0, places=9)
        self.assertEqual(rel["n"], 50)

    def test_imperv_bins_edges_and_vintage(self) -> None:
        from berlin_lst_downscaling.data.secondary.imperviousness import vintage_for_scene_year

        vals = np.array([-1.0, 0.0, 9.99, 10.0, 29.9, 30.0, 59.9, 60.0, 89.9, 90.0, 100.0])
        bins = assign_imperv_bins(vals)
        self.assertEqual(list(bins), [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(int(assign_imperv_bins(np.array([np.nan]))[0]), -1)
        self.assertEqual(int(assign_imperv_bins(np.array([100.5]))[0]), -1)
        # Canonical project mapping (2017–2019 → 2016, 2020+ → 2021).
        self.assertEqual(vintage_for_scene_year(2018), 2016)
        self.assertEqual(vintage_for_scene_year(2019), 2016)
        self.assertEqual(vintage_for_scene_year(2020), 2021)
        self.assertEqual(vintage_for_scene_year(2024), 2021)

    def test_uniform_static_emissivity_is_median_and_rejects_empty(self) -> None:
        emis = np.array([[0.95, 0.97], [0.99, np.nan]])
        mask = np.array([[True, True], [True, False]])
        self.assertAlmostEqual(uniform_static_emissivity(emis, mask), 0.97)
        with self.assertRaises(ValueError):
            uniform_static_emissivity(emis, np.zeros_like(mask, dtype=bool))

    def test_ged_proxy_averages_bands(self) -> None:
        b13 = np.array([[0.96]])
        b14 = np.array([[0.98]])
        np.testing.assert_allclose(aster_ged_b10_proxy(b13, b14), [[0.97]])

    def test_ged_resampling_reaches_scene_grid(self) -> None:
        import rasterio

        # Synthetic coarse 100 m source (2x2) onto a 30 m scene grid (7x7).
        src = np.array([[0.95, 0.96], [0.97, 0.98]])
        src_transform = rasterio.Affine(100.0, 0.0, 0.0, 0.0, -100.0, 200.0)
        dst_transform = rasterio.Affine(30.0, 0.0, 0.0, 0.0, -30.0, 200.0)
        out = ged_b10_on_scene_grid(
            src, src_transform, "EPSG:32633", (7, 7), dst_transform, "EPSG:32633"
        )
        self.assertEqual(out.shape, (7, 7))
        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertGreaterEqual(float(out.min()), 0.95 - 1e-9)
        self.assertLessEqual(float(out.max()), 0.98 + 1e-9)

    def test_ged_resampling_preserves_constant_field(self) -> None:
        import rasterio

        src = np.full((3, 3), 0.972)
        src_transform = rasterio.Affine(100.0, 0.0, 0.0, 0.0, -100.0, 300.0)
        dst_transform = rasterio.Affine(30.0, 0.0, 0.0, 0.0, -30.0, 300.0)
        out = ged_b10_on_scene_grid(
            src, src_transform, "EPSG:32633", (10, 10), dst_transform, "EPSG:32633"
        )
        np.testing.assert_allclose(out, 0.972, rtol=1e-9)

    def test_pool_overall_stats_weights_by_pixel_count(self) -> None:
        pooled = pool_overall_stats(
            [
                {"mean": 1.0, "p05": -1.0, "p95": 2.0, "n": 100},
                {"mean": 3.0, "p05": 0.0, "p95": 5.0, "n": 300},
            ]
        )
        self.assertAlmostEqual(pooled["mean"], 2.5)
        self.assertAlmostEqual(pooled["p05"], -1.0)
        self.assertAlmostEqual(pooled["p95"], 5.0)
        self.assertEqual(pooled["n"], 400)
        empty = pool_overall_stats([{"mean": float("nan"), "p05": 0.0, "p95": 0.0, "n": 0}])
        self.assertTrue(bool(np.isnan(empty["mean"])))
        self.assertEqual(empty["n"], 0)


if __name__ == "__main__":
    unittest.main()
