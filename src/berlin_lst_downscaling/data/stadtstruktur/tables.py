"""Small converters so pandas results stay explicit at the type boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frame_of(value: object) -> pd.DataFrame:
    """Return *value* when it is a DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError(type(value))


def series_of(value: object) -> pd.Series:
    """Return *value* when it is a Series."""
    if isinstance(value, pd.Series):
        return value
    raise TypeError(type(value))


def as_int(value: object) -> int | None:
    """Parse an optional integer from WFS or pandas scalars."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value == "":
            return None
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(float(value)):
            return None
        return int(value)
    return None


def as_int_required(value: object) -> int:
    """Parse an integer that must be present."""
    parsed = as_int(value)
    if parsed is None:
        raise TypeError(type(value))
    return parsed


def true_count(values: pd.Series) -> int:
    """Count true values in a boolean Series."""
    return int(np.count_nonzero(np.asarray(values, dtype=bool)))


def float_sum(values: pd.Series) -> float:
    """Sum a numeric Series as float."""
    return float(np.asarray(values, dtype=float).sum())
