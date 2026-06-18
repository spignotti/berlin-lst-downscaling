"""Smoke tests."""

import berlin_lst_downscaling


def test_import() -> None:
    assert berlin_lst_downscaling.__version__ == "0.1.0"
