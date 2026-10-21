"""QA flags are bit unions. Distinct conditions are OR-ed, never averaged."""

from __future__ import annotations

import numpy as np
import xarray as xr
from affine import Affine
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard.contract import contract_for_source
from berlin_lst_downscaling.data.ard.masking import (
    _LS_ST_OFFSET,
    _LS_ST_SCALE,
    landsat_qa_to_clear_bits,
    mask_ecostress,
    mask_landsat,
)

_TRANSFORM = Affine(100, 0, 369190, 0, -100, 5838410)


def _dataset(data: dict[str, np.ndarray]) -> xr.Dataset:
    """Build a small canonical raster dataset the maskers can stamp a CRS onto."""
    ds = xr.Dataset({name: (("y", "x"), values) for name, values in data.items()})
    ds.rio.write_crs("EPSG:25833", inplace=True)
    ds.rio.write_transform(_TRANSFORM, inplace=True)
    return ds


def test_landsat_qa_clear_is_the_or_of_fill_cloud_shadow_and_cirrus() -> None:
    """A pixel is clear only when none of the blocking QA conditions is set."""
    qa = np.array([0, 1, 524, 16, 64, 192], dtype=np.uint16)
    clear = landsat_qa_to_clear_bits(qa)
    # 0 clear; bit 0 fill; cloud+cirrus (524); shadow (16); snow (64) and the
    # water-like pattern 192 stay clear, matching the QA helper.
    assert clear.tolist() == [True, False, False, False, True, True]


def test_landsat_flag_ors_qa_conditions_and_writes_kelvin() -> None:
    """Cloud and cirrus set two flag bits. Fill becomes NaN, not zero."""
    contract = contract_for_source("landsat-c2-l2")
    # 524 = cloud bit 3, confidence >= medium in bits 8-9, and cirrus bit 2.
    qa = np.array([[0, 1], [524, 16]], dtype=np.uint16)
    raw = np.full((2, 2), 10000.0, dtype=np.float32)
    out = mask_landsat(
        _dataset({"qa_pixel": qa, "lwir11": raw}),
        OmegaConf.create({"cloud_dilation_px": 0}),
    )
    flag = out.flag.values
    assert flag.dtype == np.uint8
    assert int(flag[0, 0]) == 0
    assert int(flag[0, 1]) == contract.FLAG_FILL
    assert int(flag[1, 0]) == contract.FLAG_CLOUDY | contract.FLAG_CIRRUS
    assert int(flag[1, 1]) == contract.FLAG_SHADOW
    assert (contract.FLAG_CLOUDY | contract.FLAG_CIRRUS) != (
        contract.FLAG_CLOUDY + contract.FLAG_CIRRUS
    ) / 2

    expected_k = 10000.0 * _LS_ST_SCALE + _LS_ST_OFFSET
    assert expected_k > 150.0
    np.testing.assert_allclose(out.st.values[0, 0], expected_k, rtol=0, atol=1e-4)
    assert np.isnan(out.st.values[0, 1])


def test_ecostress_flag_ors_cloud_water_and_qc() -> None:
    """Cloud, water, and QC each contribute bits. Fill LST is NaN, not zero."""
    contract = contract_for_source("ecostress")
    out = mask_ecostress(
        _dataset(
            {
                "lst": np.full((2, 2), 300.0, dtype=np.float32),
                "cloud": np.array([[1, 1], [0, 255]], dtype=np.uint8),
                "water": np.array([[0, 1], [0, 0]], dtype=np.uint8),
                "qc": np.array([[0, 1], [3, 0]], dtype=np.uint8),
            }
        ),
        OmegaConf.create({}),
    )
    flag = out.flag.values
    assert flag.dtype == np.uint8
    assert int(flag[0, 0]) == contract.FLAG_CLOUDY
    assert int(flag[0, 1]) == contract.FLAG_CLOUDY | contract.FLAG_FILL
    assert int(flag[1, 0]) == contract.FLAG_FILL
    assert int(flag[1, 1]) == contract.FLAG_FILL
    assert float(out.lst.values[0, 0]) == 300.0
    assert np.isnan(out.lst.values[0, 1])
    assert np.isnan(out.lst.values[1, 0])
    assert np.isnan(out.lst.values[1, 1])
