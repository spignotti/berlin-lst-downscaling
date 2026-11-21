"""ERA5 field derivation fails closed when the acquisition hour is missing."""

import numpy as np
import pytest
import xarray as xr

from berlin_lst_downscaling.data.dynamic.era5 import _derive_native_fields


def _month(hours: int) -> xr.Dataset:
    times = np.arange(
        np.datetime64("2026-07-01T00"),
        np.datetime64("2026-07-01T00") + hours,
        np.timedelta64(1, "h"),
    )
    shape = (len(times), 2, 3)
    rng = np.random.default_rng(0)
    data = {
        "t2m": 290 + rng.random(shape),
        "d2m": 280 + rng.random(shape),
        "u10": rng.random(shape),
        "v10": rng.random(shape),
        "ssrd": np.cumsum(np.full(shape, 3600.0 * 100), axis=0),
        "tp": np.cumsum(np.full(shape, 1e-4), axis=0),
    }
    return xr.Dataset(
        {
            k: (("valid_time", "latitude", "longitude"), v.astype("float32"))
            for k, v in data.items()
        },
        coords={"valid_time": times, "latitude": [52.6, 52.5], "longitude": [13.2, 13.3, 13.4]},
    )


def test_exact_hour_present_derives_fields() -> None:
    fields = _derive_native_fields(_month(24 * 10), np.datetime64("2026-07-05T10"))
    assert fields["t2m_scene"].shape == (2, 3)


def test_cache_ending_before_the_scene_raises() -> None:
    # cache ends 2026-07-03T23, scene at 2026-07-05T10: nearest hour would be filled in
    with pytest.raises(ValueError, match="no timestep at the acquisition hour"):
        _derive_native_fields(_month(24 * 3), np.datetime64("2026-07-05T10"))
