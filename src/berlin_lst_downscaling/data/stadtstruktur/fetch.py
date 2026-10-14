"""Page the Berlin Stadtstruktur WFS into a source GeoDataFrame.

The differentiated layer is the raw product. It carries the ISU5 key
(``schluessel``), the 06.08 area type (``typ``), and the 06.07 structure
type (``ststrnr``). The sibling 06.07-only layer repeats the same block
parts, so it is not downloaded again.
"""

from __future__ import annotations

import json
from hashlib import sha256
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape
from tenacity import retry, stop_after_attempt, wait_exponential

from berlin_lst_downscaling.data.stadtstruktur.sources import TARGET_CRS, VintageSource
from berlin_lst_downscaling.data.stadtstruktur.tables import as_int, frame_of, series_of

_PAGE = 1000
_TIMEOUT_S = 180
_EASTING = (300_000.0, 500_000.0)
_NORTHING = (5_700_000.0, 5_900_000.0)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_json(url: str) -> dict:
    # requests uses certifi. The uv CPython build on macOS does not.
    response = requests.get(
        url,
        timeout=_TIMEOUT_S,
        headers={"User-Agent": "berlin-lst-downscaling/stadtstruktur"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(type(payload))
    return payload


def _feature_url(source: VintageSource, start: int) -> str:
    query = urlencode(
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": source.type_name,
            "outputFormat": "application/json",
            "srsName": TARGET_CRS,
            "count": str(_PAGE),
            "startIndex": str(start),
        }
    )
    return f"{source.wfs_url}?{query}"


def _typklar(properties: dict) -> str | None:
    value = properties.get("typklar", properties.get("typ_klar"))
    if value is None:
        return None
    return str(value)


def _as_int(value: object) -> int | None:
    return as_int(value)


def fetch_vintage(source: VintageSource) -> tuple[gpd.GeoDataFrame, dict]:
    """Download one vintage and return source-aligned polygons plus a receipt.

    Coordinates are requested in EPSG:25833. A response whose first vertex
    falls outside the Berlin easting/northing range is rejected.
    """
    features: list[dict] = []
    digest = sha256()
    matched: int | None = None
    start = 0
    while True:
        payload = _get_json(_feature_url(source, start))
        page = payload.get("features", [])
        matched = int(payload.get("numberMatched", matched or 0))
        raw = json.dumps(page, sort_keys=True, separators=(",", ":")).encode()
        digest.update(raw)
        features.extend(page)
        print(
            f"stadtstruktur {source.vintage}: page {start} "
            f"returned {len(page)} (accumulated {len(features)}/{matched})",
            flush=True,
        )
        if not page or len(features) >= matched:
            break
        start += len(page)

    if matched is not None and len(features) != matched:
        raise RuntimeError(
            f"{source.type_name}: downloaded {len(features)} features, service matched {matched}"
        )

    rows: list[dict] = []
    for feature in features:
        properties = feature["properties"]
        geometry = shape(feature["geometry"])
        representative = geometry.representative_point()
        if not (_EASTING[0] <= representative.x <= _EASTING[1]):
            raise RuntimeError(
                f"{source.vintage}: easting {representative.x} is outside EPSG:25833"
            )
        if not (_NORTHING[0] <= representative.y <= _NORTHING[1]):
            raise RuntimeError(
                f"{source.vintage}: northing {representative.y} is outside EPSG:25833"
            )
        key = properties.get("schluessel")
        rows.append(
            {
                "schluessel": None if key is None else str(key),
                "vintage": source.vintage,
                "reference_date": source.reference_date,
                "bez": None if properties.get("bez") is None else str(properties.get("bez")),
                "bezirk": properties.get("bezirk"),
                "typ": _as_int(properties.get("typ")),
                "typklar": _typklar(properties),
                "ststrnr": _as_int(properties.get("ststrnr")),
                "ststrname": properties.get("ststrname"),
                "nutz": _as_int(properties.get("nutz")),
                "nutzung": properties.get("nutzung"),
                "flalle_m2": None
                if properties.get("flalle") is None
                else float(properties["flalle"]),
                "geometry": geometry,
            }
        )

    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)
    receipt = {
        "vintage": source.vintage,
        "reference_date": source.reference_date,
        "wfs_url": source.wfs_url,
        "type_name": source.type_name,
        "license_id": "dl-de/zero-2.0",
        "feature_count": len(frame),
        "service_number_matched": matched,
        "page_sha256": digest.hexdigest(),
        "crs": TARGET_CRS,
    }
    return frame, receipt


def crosswalk(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per observed typ with its 06.07 code and names."""
    sizes = series_of(
        frame.groupby(["vintage", "typ", "typklar", "ststrnr", "ststrname"], dropna=False).size()
    )
    grouped = frame_of(sizes.reset_index(name="n_polygons"))
    return grouped.sort_values(["vintage", "typ", "ststrnr"], kind="mergesort")
