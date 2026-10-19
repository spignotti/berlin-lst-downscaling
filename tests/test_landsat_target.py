"""Area-mean Landsat target: QA bit-OR and USGS 30 m grid phase."""

from __future__ import annotations

import unittest

import numpy as np
import rasterio

from berlin_lst_downscaling.common.grid import canon_grid_100m
from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.landsat_target import (
    USGS_30M_CORNER_PHASE_M,
    USGS_30M_M,
    aggregate_condition_planes,
    landsat_schema_hash,
    pack_landsat_flags,
    qa_condition_planes,
    usgs_30m_geobox_covering,
)


class LandsatTargetTest(unittest.TestCase):
    def test_schema_hash_records_the_policy_without_bumping_the_version(self) -> None:
        token = landsat_schema_hash(7, "average", "native_bit_or")
        self.assertEqual(token, "7|st=average|qa=native_bit_or")
        self.assertTrue(token.startswith("7|"))

    def test_usgs_grid_covers_the_canonical_100m_grid_on_the_30m_phase(self) -> None:
        target = canon_grid_100m()
        native = usgs_30m_geobox_covering(target)
        self.assertEqual(native.transform.c % USGS_30M_M, USGS_30M_CORNER_PHASE_M)
        self.assertEqual(native.transform.f % USGS_30M_M, USGS_30M_CORNER_PHASE_M)
        self.assertLessEqual(native.transform.c, target.transform.c)
        self.assertGreaterEqual(native.transform.f, target.transform.f)
        native_x1 = native.transform.c + USGS_30M_M * native.shape.x
        native_y0 = native.transform.f - USGS_30M_M * native.shape.y
        target_x1 = target.transform.c + float(target.transform.a) * target.shape.x
        target_y0 = target.transform.f + float(target.transform.e) * target.shape.y
        self.assertGreaterEqual(native_x1, target_x1)
        self.assertLessEqual(native_y0, target_y0)

    def test_overlapping_bits_are_ored_and_water_stays_clear(self) -> None:
        # Two 30 m pixels. The 100 m cell covers both. The second cell does not.
        src_transform = rasterio.Affine(30, 0, 15, 0, -30, 45)
        dst_transform = rasterio.Affine(100, 0, 0, 0, -100, 100)
        qa = np.array([[2 << 8, 1 << 7]], dtype=np.uint16)  # cloud conf 2, water
        planes = aggregate_condition_planes(
            qa_condition_planes(qa),
            src_transform,
            dst_transform,
            (1, 2),
        )
        contract = contract_for_source("landsat-c2-l2")
        flag = pack_landsat_flags(planes, contract, dilate_px=0)
        self.assertEqual(int(flag[0, 0] & contract.FLAG_CLOUDY), contract.FLAG_CLOUDY)
        self.assertEqual(int(flag[0, 0] & contract.FLAG_FILL), 0)
        self.assertEqual(int(flag[0, 1]), 0)

    def test_distinct_bits_in_one_cell_both_survive(self) -> None:
        src_transform = rasterio.Affine(30, 0, 15, 0, -30, 45)
        dst_transform = rasterio.Affine(100, 0, 0, 0, -100, 100)
        qa = np.array([[1, 1 << 2]], dtype=np.uint16)  # fill, cirrus
        planes = aggregate_condition_planes(
            qa_condition_planes(qa),
            src_transform,
            dst_transform,
            (1, 1),
        )
        contract = contract_for_source("landsat-c2-l2")
        flag = pack_landsat_flags(planes, contract, dilate_px=0)
        self.assertEqual(
            int(flag[0, 0] & (contract.FLAG_FILL | contract.FLAG_CIRRUS)),
            contract.FLAG_FILL | contract.FLAG_CIRRUS,
        )

    def test_cloud_dilation_stays_on_the_destination_grid(self) -> None:
        contract = contract_for_source("landsat-c2-l2")
        cloudy = np.zeros((3, 3), dtype=bool)
        cloudy[1, 1] = True
        empty = np.zeros((3, 3), dtype=bool)
        flag = pack_landsat_flags(
            {"fill": empty, "cloudy": cloudy, "shadow": empty, "cirrus": empty, "snow": empty},
            contract,
            dilate_px=1,
        )
        self.assertEqual(int(flag[1, 1] & contract.FLAG_CLOUDY), contract.FLAG_CLOUDY)
        self.assertEqual(int(flag[0, 1] & contract.FLAG_FILL), contract.FLAG_FILL)
        self.assertEqual(int(flag[1, 1] & contract.FLAG_FILL), 0)


if __name__ == "__main__":
    unittest.main()
