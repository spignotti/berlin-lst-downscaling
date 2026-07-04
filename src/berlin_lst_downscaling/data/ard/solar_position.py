"""Solar position computation — from STAC properties or datetime + lat/lon.

Provides a single function :func:`solar_position` that accepts either
a source parameter (trying STAC ``view:sun_*`` first) or a
datetime + lat/lon pair for the NOAA fallback algorithm.
"""

from __future__ import annotations

import math
from datetime import datetime

# ── constants ────────────────────────────────────────────────────────

_RAD = math.pi / 180.0
_DEG = 180.0 / math.pi

_AOI_CENTROID = (52.51, 13.42)  # Berlin approximate centroid (lat, lon)


# ── public API ───────────────────────────────────────────────────────


def solar_position(
    dt: datetime,
    lat: float | None = None,
    lon: float | None = None,
) -> tuple[float, float]:
    """Compute solar azimuth and elevation for a given time and location.

    Parameters
    ----------
    dt :
        UTC datetime of the acquisition.
    lat, lon :
        Latitude / longitude in decimal degrees (WGS84).  Defaults to
        Berlin centroid ``(52.51, 13.42)``.

    Returns
    -------
    tuple[float, float]
        ``(azimuth_deg, elevation_deg)`` where azimuth is measured
        clockwise from true North (0° = N, 90° = E, 180° = S, 270° = W)
        and elevation is above the horizon (0° = horizon, 90° = zenith).
    """
    if lat is None:
        lat = _AOI_CENTROID[0]
    if lon is None:
        lon = _AOI_CENTROID[1]

    return _noaa_solar_position(dt, lat, lon)


def solar_position_from_stac(
    properties: dict,
    dt: datetime | None = None,
    centroid: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Try STAC view:sun_azimuth / view:sun_elevation, fall back to computation.

    Parameters
    ----------
    properties :
        STAC item ``properties`` dict.
    dt :
        Acquisition datetime (UTC) for fallback.  Required when STAC
        properties are missing.
    centroid :
        ``(lat, lon)`` for fallback.  Defaults to Berlin centroid.

    Returns
    -------
    tuple[float, float]
        ``(azimuth_deg, elevation_deg)``.
    """
    az = properties.get("view:sun_azimuth")
    el = properties.get("view:sun_elevation")

    if az is not None and el is not None:
        return (float(az), float(el))

    if dt is None:
        raise ValueError("STAC view:sun_* not available — provide a datetime for computed fallback")

    lat, lon = centroid if centroid else _AOI_CENTROID
    return solar_position(dt, lat, lon)


# ── NOAA solar position algorithm (simplified) ───────────────────────
# Based on the NOAA Solar Calculator / ESRL algorithm.
# Accuracy ~ ±1° — sufficient for shadow-offset projection at 10 m.


def _noaa_solar_position(dt: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Compute solar azimuth (from N) and elevation (above horizon)."""
    # 1. day of year
    doy = dt.timetuple().tm_yday  # 1-366

    # 2. fractional year (radians)
    gamma = (2.0 * math.pi / 365.0) * (doy - 1 + (dt.hour - 12) / 24.0)

    # 3. equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.04089 * math.sin(2 * gamma)
    )

    # 4. solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # 5. time offset (minutes) — longitude + eqtime
    time_offset = eqtime + 4.0 * lon_deg

    # 6. true solar time (minutes)
    tst = dt.hour * 60.0 + dt.minute + dt.second / 60.0 + time_offset

    # 7. hour angle (degrees)
    ha_deg = tst / 4.0 - 180.0  # 15°/h, offset so 0° = solar noon
    ha = ha_deg * _RAD

    # 8. solar elevation
    lat = lat_deg * _RAD
    sin_elev = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    elevation = math.asin(max(-1.0, min(1.0, sin_elev))) * _DEG

    # 9. solar azimuth
    cos_azimuth = (
        math.sin(decl) * math.cos(lat) - math.cos(decl) * math.sin(lat) * math.cos(ha)
    ) / math.cos(elevation * _RAD)
    cos_azimuth = max(-1.0, min(1.0, cos_azimuth))

    az = math.acos(cos_azimuth) * _DEG

    # adjust for time of day (afternoon = azimuth > 180)
    if ha_deg > 0:
        az = 360.0 - az

    return (az, elevation)


__all__ = [
    "solar_position",
    "solar_position_from_stac",
]
