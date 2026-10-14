#!/usr/bin/env python3
"""Read-only ECOSTRESS C2 v002 feasibility inventory for the published anchors.

Source, purpose, grain
----------------------
Source: the published v3 scene manifest on GCS, and NASA CMR metadata for
``ECO_L2T_LSTE`` version 002. No granules, scenes, or checkpoints are read.
Purpose: count which Landsat anchors have an ECOSTRESS overpass close in
time whose 100 km tiles meet the Berlin boundary.
Grain: one anchor row is one published Landsat scene. One granule row is
one CMR L2T tile. One overpass is every granule that shares ``time_start``.

``±2 hours`` is a sensitivity bin, not a simultaneity claim. ECOSTRESS
cloud and QC inside the AOI are not read. Overlap is the planar area of
the MGRS 100 km tile, reprojected to EPSG:25833, divided by the Berlin
Landesgrenze. A CMR bounding box only checks that tile.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union

from berlin_lst_downscaling.data.io.storage import atomic_upload, read_bytes

MANIFEST = (
    "gs://berlin-lst-training-data/manifests/v3/"
    "2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet"
)
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
AOI_PATH = Path("data/boundaries/berlin_landesgrenze.geojson")
WINDOWS_MINUTES = (15, 30, 60, 120, 180, 360)
TEMPORAL = "2017-05-01T00:00:00Z,2026-07-17T23:59:59Z"
BBOX = (13.08, 52.34, 13.76, 52.68)
BERLIN = ZoneInfo("Europe/Berlin")
TILE_RE = re.compile(r"_(\d{2}[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V])_")
_COLUMN_ORIGIN = "AJSAJS"
_ROW_ORIGIN = "AFAFAF"
_MIN_NORTHING = {
    "C": 1_100_000,
    "D": 2_000_000,
    "E": 2_800_000,
    "F": 3_700_000,
    "G": 4_600_000,
    "H": 5_500_000,
    "J": 6_400_000,
    "K": 7_300_000,
    "L": 8_200_000,
    "M": 9_100_000,
    "N": 0,
    "P": 800_000,
    "Q": 1_700_000,
    "R": 2_600_000,
    "S": 3_500_000,
    "T": 4_400_000,
    "U": 5_300_000,
    "V": 6_200_000,
    "W": 7_000_000,
    "X": 7_900_000,
}


def _set_for_zone(zone: int) -> int:
    value = zone % 6
    return 6 if value == 0 else value


def _skip_io(code: int) -> int:
    if code in (73, 79):
        return code + 1
    return code


def _easting_from_char(letter: str, set_id: int) -> int:
    current = ord(_COLUMN_ORIGIN[set_id - 1])
    easting = 100_000
    rewound = False
    target = ord(letter)
    while current != target:
        current = _skip_io(current + 1)
        if current > 90:
            if rewound:
                raise ValueError(f"bad MGRS column {letter}")
            current = 65
            rewound = True
        easting += 100_000
    return easting


def _northing_from_char(letter: str, set_id: int, band: str) -> int:
    current = ord(_ROW_ORIGIN[set_id - 1])
    northing = 0
    rewound = False
    target = ord(letter)
    while current != target:
        current = _skip_io(current + 1)
        if current > 86:
            if rewound:
                raise ValueError(f"bad MGRS row {letter}")
            current = 65
            rewound = True
        northing += 100_000
    while northing < _MIN_NORTHING[band]:
        northing += 2_000_000
    return northing


def tile_origin(tile: str) -> tuple[int, int, int]:
    """Return the south-west corner and UTM zone of a 100 km MGRS square."""
    zone = int(tile[:2])
    band = tile[2]
    set_id = _set_for_zone(zone)
    easting = _easting_from_char(tile[3], set_id)
    northing = _northing_from_char(tile[4], set_id, band)
    return easting, northing, zone


def _check_mgrs() -> None:
    """Fail if the tile decoder misses a known Berlin point in UTM."""
    to_33 = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    to_32 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    checks = (
        ("33UUU", to_33.transform(13.413, 52.521)),
        ("33UVU", to_33.transform(13.72, 52.50)),
        ("32UQD", to_32.transform(13.12, 52.45)),
    )
    for tile, (x_coord, y_coord) in checks:
        easting, northing, _zone = tile_origin(tile)
        inside = easting <= x_coord < easting + 100_000 and northing <= y_coord < northing + 100_000
        if not inside:
            raise SystemExit(
                f"MGRS check failed for {tile}: point {(x_coord, y_coord)} "
                f"is outside {(easting, northing)}"
            )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _cmr_box(entry: dict) -> tuple[float, float, float, float] | None:
    boxes = entry.get("boxes") or []
    if not boxes:
        return None
    south, west, north, east = (float(part) for part in boxes[0].split())
    return west, south, east, north


def fetch_ecostress() -> tuple[pd.DataFrame, int, int]:
    """Page CMR. Returns the frame, the server hit count, and rows received."""
    rows: list[dict] = []
    hits = -1
    page = 1
    while True:
        response = requests.get(
            CMR,
            params={
                "short_name": "ECO_L2T_LSTE",
                "version": "002",
                "bounding_box": ",".join(str(value) for value in BBOX),
                "temporal": TEMPORAL,
                "page_size": 2000,
                "page_num": page,
            },
            timeout=120,
            headers={"User-Agent": "berlin-lst-downscaling/ecostress-inventory"},
        )
        response.raise_for_status()
        if page == 1:
            hits = int(response.headers["CMR-Hits"])
        entries = response.json()["feed"]["entry"]
        print(f"cmr page {page}: {len(entries)}", flush=True)
        if not entries:
            break
        for entry in entries:
            granule_id = entry.get("producer_granule_id") or entry.get("title")
            match = TILE_RE.search(granule_id or "")
            start = _parse_time(entry["time_start"])
            end = _parse_time(entry["time_end"])
            box = _cmr_box(entry)
            rows.append(
                {
                    "granule_id": granule_id,
                    "cmr_id": entry.get("id"),
                    "acquired_at": start,
                    "time_end": end,
                    "duration_s": (end - start).total_seconds(),
                    "day_night": entry.get("day_night_flag"),
                    "mgrs": None if match is None else match.group(1),
                    "box_west": None if box is None else box[0],
                    "box_south": None if box is None else box[1],
                    "box_east": None if box is None else box[2],
                    "box_north": None if box is None else box[3],
                }
            )
        if len(entries) < 2000:
            break
        page += 1
    frame = pd.DataFrame(rows).drop_duplicates("granule_id").reset_index(drop=True)
    return frame, hits, len(rows)


def _load_anchors() -> pd.DataFrame:
    raw = read_bytes(MANIFEST)
    frame = pd.read_parquet(BytesIO(raw))
    anchors = frame.loc[(frame["source"] == "landsat-c2-l2") & (frame["role"] == "anchor")].copy()
    anchors["acquired_at"] = pd.to_datetime(anchors["acquisition_datetime"], utc=True)
    return anchors.reset_index(drop=True)


def _load_aoi() -> Polygon:
    boundary = gpd.read_file(AOI_PATH)
    if boundary.crs is None or boundary.crs.to_epsg() != 25833:
        raise SystemExit(f"AOI CRS is {boundary.crs}, expected EPSG:25833")
    geometry = boundary.geometry.union_all()
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def _tile_polygon(tile: str, cache: dict[int, Transformer]) -> Polygon:
    easting, northing, zone = tile_origin(tile)
    transformer = cache.get(zone)
    if transformer is None:
        transformer = Transformer.from_crs(f"EPSG:326{zone:02d}", "EPSG:25833", always_xy=True)
        cache[zone] = transformer
    corners = (
        (easting, northing),
        (easting + 100_000, northing),
        (easting + 100_000, northing + 100_000),
        (easting, northing + 100_000),
    )
    return Polygon([transformer.transform(x_coord, y_coord) for x_coord, y_coord in corners])


def _attach_overlap(granules: pd.DataFrame, aoi: Polygon) -> pd.DataFrame:
    cache: dict[int, Transformer] = {}
    to_wgs84: dict[int, Transformer] = {}
    area = float(aoi.area)
    fractions: list[float] = []
    outside_box = 0
    missing_tile = 0
    worst_overrun = 0.0
    worst_id = None
    for row in granules.itertuples(index=False):
        if row.mgrs is None:
            missing_tile += 1
            fractions.append(0.0)
            continue
        polygon = _tile_polygon(row.mgrs, cache)
        fractions.append(float(polygon.intersection(aoi).area) / area)
        easting, northing, zone = tile_origin(row.mgrs)
        geographic = to_wgs84.get(zone)
        if geographic is None:
            geographic = Transformer.from_crs(f"EPSG:326{zone:02d}", "EPSG:4326", always_xy=True)
            to_wgs84[zone] = geographic
        if row.box_west is None:
            outside_box += 1
            continue
        west, south, east, north = row.box_west, row.box_south, row.box_east, row.box_north
        overrun = 0.0
        for x_coord, y_coord in (
            (easting, northing),
            (easting + 100_000, northing),
            (easting + 100_000, northing + 100_000),
            (easting, northing + 100_000),
        ):
            lon, lat = geographic.transform(x_coord, y_coord)
            overrun = max(
                overrun,
                west - lon,
                lon - east,
                south - lat,
                lat - north,
                0.0,
            )
        if overrun > 0.05:
            outside_box += 1
        if overrun > worst_overrun:
            worst_overrun = overrun
            worst_id = row.granule_id
    out = granules.copy()
    out["overlap_fraction"] = fractions
    out.attrs["tiles_outside_cmr_box"] = outside_box
    out.attrs["max_corner_overrun_deg"] = worst_overrun
    out.attrs["max_corner_overrun_granule"] = worst_id
    out.attrs["missing_mgrs"] = missing_tile
    return out


def _overpasses(granules: pd.DataFrame, aoi: Polygon) -> pd.DataFrame:
    cache: dict[int, Transformer] = {}
    area = float(aoi.area)
    grouped: dict[pd.Timestamp, list] = defaultdict(list)
    for row in granules.itertuples(index=False):
        if row.mgrs is None:
            continue
        grouped[pd.Timestamp(row.acquired_at)].append(row)
    records = []
    for stamp, members in grouped.items():
        polygons = [_tile_polygon(member.mgrs, cache) for member in members]
        covered = float(unary_union(polygons).intersection(aoi).area) / area
        flags = {member.day_night for member in members}
        records.append(
            {
                "acquired_at": stamp,
                "n_tiles": len(members),
                "overlap_fraction": covered,
                "day_night": next(iter(flags)) if len(flags) == 1 else "MIXED",
                "granule_ids": sorted(member.granule_id for member in members),
            }
        )
    frame = pd.DataFrame(records).sort_values("acquired_at").reset_index(drop=True)
    return frame


def _closest(anchor_ns: int, times_ns: pd.Series, values: pd.DataFrame, predicate) -> dict | None:
    """Scan every record inside the widest window and keep the nearest hit."""
    times = times_ns.to_numpy()
    position = int(times.searchsorted(anchor_ns))
    best: dict | None = None
    for direction, start in ((1, position), (-1, position - 1)):
        index = start
        while 0 <= index < len(times):
            delta_min = abs(int(times[index]) - anchor_ns) / 60_000_000_000
            if delta_min > WINDOWS_MINUTES[-1]:
                break
            row = values.iloc[index]
            if predicate(row) and (best is None or delta_min < best["minutes"]):
                best = {"minutes": float(delta_min), "row": row}
            index += direction
    return best


def _epoch_ns(series: pd.Series) -> pd.Series:
    """Integer nanoseconds since the epoch. Pandas may store microseconds."""
    stamps = pd.to_datetime(series, utc=True).dt.as_unit("ns")
    return stamps.astype("int64")


def pair(anchors: pd.DataFrame, granules: pd.DataFrame, overpasses: pd.DataFrame) -> pd.DataFrame:
    granules = granules.sort_values("acquired_at").reset_index(drop=True)
    overpasses = overpasses.sort_values("acquired_at").reset_index(drop=True)
    granule_ns = _epoch_ns(granules["acquired_at"])
    overpass_ns = _epoch_ns(overpasses["acquired_at"])
    records = []
    for scene in anchors.itertuples(index=False):
        stamp = pd.Timestamp(scene.acquired_at).as_unit("ns").value
        any_hit = _closest(
            stamp,
            granule_ns,
            granules,
            lambda row: row.box_east - row.box_west <= 180,
        )
        overlap = _closest(
            stamp, overpass_ns, overpasses, lambda row: float(row.overlap_fraction) > 0
        )
        overlap_10 = _closest(
            stamp, overpass_ns, overpasses, lambda row: float(row.overlap_fraction) >= 0.10
        )
        local_hour = None
        if overlap is not None:
            local_hour = overlap["row"].acquired_at.tz_convert(BERLIN).hour
        records.append(
            {
                "scene_id": scene.scene_id,
                "acquired_at": scene.acquired_at,
                "year": int(scene.year),
                "platform": scene.platform,
                "aoi_clear_frac": scene.aoi_clear_frac,
                "cloud_cover": scene.cloud_cover,
                "solar_elevation": scene.solar_elevation,
                "closest_any_id": None if any_hit is None else any_hit["row"].granule_id,
                "closest_any_minutes": None if any_hit is None else any_hit["minutes"],
                "closest_overlap_minutes": None if overlap is None else overlap["minutes"],
                "closest_overlap_fraction": None
                if overlap is None
                else float(overlap["row"].overlap_fraction),
                "closest_overlap_day_night": None if overlap is None else overlap["row"].day_night,
                "closest_overlap_n_tiles": None if overlap is None else int(overlap["row"].n_tiles),
                "closest_overlap_hour_berlin": local_hour,
                "closest_overlap10_minutes": None if overlap_10 is None else overlap_10["minutes"],
                "closest_overlap10_fraction": None
                if overlap_10 is None
                else float(overlap_10["row"].overlap_fraction),
            }
        )
    return pd.DataFrame(records)


def _counts(deltas: pd.Series) -> dict[str, int]:
    valid = deltas.dropna()
    return {f"within_{minutes}min": int((valid <= minutes).sum()) for minutes in WINDOWS_MINUTES}


def _median(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.median())


def _min(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.min())


def _merge_asof_disagreements(paired: pd.DataFrame, granules: pd.DataFrame) -> int:
    left = paired[["scene_id", "acquired_at"]].sort_values("acquired_at")
    kept = granules.loc[granules["box_east"] - granules["box_west"] <= 180]
    right = kept[["granule_id", "acquired_at"]].rename(columns={"acquired_at": "granule_at"})
    right = right.sort_values("granule_at")
    merged = pd.merge_asof(
        left,
        right,
        left_on="acquired_at",
        right_on="granule_at",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=WINDOWS_MINUTES[-1]),
    )
    comparison = paired.merge(merged, on="scene_id", suffixes=("", "_asof"))
    walked = comparison["closest_any_minutes"]
    delta = (comparison["granule_at"] - comparison["acquired_at"]).abs()
    asof_minutes = delta.dt.total_seconds() / 60
    both_missing = walked.isna() & comparison["granule_id"].isna()
    minutes_agree = (walked - asof_minutes).abs() <= 1e-6
    mismatched = comparison.loc[~both_missing & ~minutes_agree, ["scene_id"]].copy()
    mismatched["walk_min"] = walked.loc[mismatched.index]
    mismatched["asof_min"] = asof_minutes.loc[mismatched.index]
    if not mismatched.empty:
        print(mismatched.head(5).to_dict(orient="records"), flush=True)
    return int(len(mismatched))


def _overpass_disagreements(paired: pd.DataFrame, overpasses: pd.DataFrame) -> int:
    """Second path for the nearest overpass that actually meets Berlin."""
    left = paired[["scene_id", "acquired_at"]].sort_values("acquired_at")
    kept = overpasses.loc[overpasses["overlap_fraction"] > 0, ["acquired_at"]].rename(
        columns={"acquired_at": "overpass_at"}
    )
    kept = kept.sort_values("overpass_at")
    merged = pd.merge_asof(
        left,
        kept,
        left_on="acquired_at",
        right_on="overpass_at",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=WINDOWS_MINUTES[-1]),
    )
    comparison = paired.merge(merged, on="scene_id", suffixes=("", "_asof"))
    walked = comparison["closest_overlap_minutes"]
    delta = (comparison["overpass_at"] - comparison["acquired_at"]).abs()
    asof_minutes = delta.dt.total_seconds() / 60
    both_missing = walked.isna() & comparison["overpass_at"].isna()
    minutes_agree = (walked - asof_minutes).abs() <= 1e-6
    return int((~both_missing & ~minutes_agree).sum())


def _code_audit() -> dict[str, list[str]]:
    pipeline = Path("src/berlin_lst_downscaling/data/selection/pipeline.py").read_text(
        encoding="utf-8"
    )
    couple = Path("src/berlin_lst_downscaling/data/selection/couple.py").read_text(encoding="utf-8")
    config = Path("configs/selection/_base.yaml").read_text(encoding="utf-8")
    subset = Path("src/berlin_lst_downscaling/data/selection/ecostress_subset.py")
    if "window_hours" in pipeline:
        raise SystemExit("code audit stale: pipeline.py now references window_hours")
    if "ecostress_subset" not in couple or subset.exists():
        raise SystemExit("code audit stale: ecostress_subset wiring changed")
    if config.count("ECOv002_L2T_LSTE_") != 6:
        raise SystemExit("code audit stale: validation allowlist length changed")
    return {
        "working": [
            "data/acquisition/ecostress.py downloads one named ECO_L2T_LSTE.002 granule.",
            "ARD masking and scripts/validators/validate_ecostress_scenes.py exist.",
        ],
        "not_a_search": [
            "configs/selection/_base.yaml sets window_hours: 2. pipeline.py never reads it.",
            "Selection resolves six 2018-08-25 granules and writes dt_hours 0 and overlap_frac 1.",
            "data/selection/ecostress_subset.py is absent. couple.py still leaves that list empty.",
            "The published manifest contains those six granules and no other ECOSTRESS rows.",
        ],
    }


def _allowlist_status(granules: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    config = Path("configs/selection/_base.yaml").read_text(encoding="utf-8")
    ids = re.findall(r"ECOv002_L2T_LSTE_[0-9A-Z_]+", config)
    known = set(granules["granule_id"])
    present = [granule_id for granule_id in ids if granule_id in known]
    listed = granules.loc[granules["granule_id"].isin(present), "acquired_at"]
    nearest_hours = None
    if not listed.empty:
        target = pd.Timestamp(listed.min())
        delta = (anchors["acquired_at"] - target).abs().min()
        nearest_hours = float(delta.total_seconds() / 3600)
    return {
        "configured": len(ids),
        "found_in_cmr_bbox": len(present),
        "missing": sorted(set(ids) - set(present)),
        "nearest_published_anchor_hours": nearest_hours,
    }


def _real_granules(granules: pd.DataFrame) -> pd.DataFrame:
    return granules.loc[granules["box_east"] - granules["box_west"] <= 180]


def _first_granule(granules: pd.DataFrame) -> str:
    stamp = _real_granules(granules)["acquired_at"].min()
    return pd.Timestamp(stamp).isoformat()


def _anchors_before(anchors: pd.DataFrame, granules: pd.DataFrame) -> int:
    first = pd.Timestamp(_first_granule(granules))
    return int((anchors["acquired_at"] < first).sum())


def _graze_tile(granules: pd.DataFrame) -> dict[str, object]:
    """Tile whose overlap is a repeated corner sliver, if one dominates."""
    real = _real_granules(granules)
    sliver = real.loc[real["overlap_fraction"].between(1e-6, 0.001)]
    if sliver.empty:
        return {"mgrs": None, "n_granules": 0, "overlap_fraction": None}
    tile = sliver["mgrs"].mode().iloc[0]
    fraction = float(sliver.loc[sliver["mgrs"] == tile, "overlap_fraction"].median())
    return {
        "mgrs": tile,
        "n_granules": int((sliver["mgrs"] == tile).sum()),
        "overlap_fraction": fraction,
    }


def _summary(
    anchors: pd.DataFrame,
    granules: pd.DataFrame,
    overpasses: pd.DataFrame,
    paired: pd.DataFrame,
    hits: int,
    fetched: int,
    disagreements: int,
    overpass_disagreements: int,
) -> dict:
    within_2h = paired["closest_overlap_minutes"].le(120)
    hours = paired.loc[within_2h, "closest_overlap_hour_berlin"].dropna().astype(int)
    hour_counts = {str(hour): int((hours == hour).sum()) for hour in sorted(hours.unique())}
    return {
        "accessed_at": datetime.now(UTC).isoformat(),
        "product": "ECO_L2T_LSTE.002",
        "manifest": MANIFEST,
        "cmr": CMR,
        "temporal": TEMPORAL,
        "bbox": list(BBOX),
        "overlap_definition": (
            "Planar intersection of the MGRS 100 km tile with "
            "data/boundaries/berlin_landesgrenze.geojson in EPSG:25833, "
            "divided by the boundary area. An overpass unions every tile "
            "that shares time_start. This is tile coverage, not valid LST pixels."
        ),
        "n_landsat_anchors": int(len(anchors)),
        "anchor_years": {
            str(year): int(count)
            for year, count in anchors["year"].value_counts().sort_index().items()
        },
        "n_cmr_hits": hits,
        "n_ecostress_granules": int(len(granules)),
        "n_granules_missing_mgrs": int(granules.attrs.get("missing_mgrs", 0)),
        "n_granules_with_tile_overlap": int((granules["overlap_fraction"] > 0).sum()),
        "n_overpasses": int(len(overpasses)),
        "n_overpasses_with_tile_overlap": int((overpasses["overlap_fraction"] > 0).sum()),
        "first_berlin_granule_at": _first_granule(granules),
        "anchors_before_first_berlin_granule": _anchors_before(anchors, granules),
        "graze_tile": _graze_tile(granules),
        "anchors_with_any_granule_inside_6h": _counts(paired["closest_any_minutes"]),
        "anchors_with_overlapping_overpass": _counts(paired["closest_overlap_minutes"]),
        "anchors_with_overlap_at_least_0_10": _counts(paired["closest_overlap10_minutes"]),
        "closest_overlap_minutes": {
            "min": _min(paired["closest_overlap_minutes"]),
            "median": _median(paired["closest_overlap_minutes"]),
        },
        "closest_overlap10_minutes": {
            "min": _min(paired["closest_overlap10_minutes"]),
            "median": _median(paired["closest_overlap10_minutes"]),
        },
        "within_2h_overlap_hour_berlin": hour_counts,
        "within_2h_day_night": paired.loc[within_2h, "closest_overlap_day_night"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict(),
        "within_2h_landsat_clear_frac_at_least_0_05": int(
            (within_2h & paired["aoi_clear_frac"].ge(0.05)).sum()
        ),
        "within_2h_landsat_clear_frac_at_least_0_50": int(
            (within_2h & paired["aoi_clear_frac"].ge(0.50)).sum()
        ),
        "within_2h_rows_overlap_at_least_0_10": int(
            (within_2h & paired["closest_overlap_fraction"].ge(0.10)).sum()
        ),
        "within_2h_rows_overlap_at_least_0_50": int(
            (within_2h & paired["closest_overlap_fraction"].ge(0.50)).sum()
        ),
        "within_2h_dates_overlap_at_least_0_10": sorted(
            {
                stamp.date().isoformat()
                for stamp in paired.loc[
                    within_2h & paired["closest_overlap_fraction"].ge(0.10),
                    "acquired_at",
                ]
            }
        ),
        "checks": {
            "cmr_hits_match_fetched_rows": hits == fetched,
            "duplicate_granule_ids": fetched - len(granules),
            "merge_asof_disagreements": disagreements,
            "overlap_asof_disagreements": overpass_disagreements,
            "n_antimeridian_false_positives": int(
                (granules["box_east"] - granules["box_west"] > 180).sum()
            ),
            "tiles_outside_cmr_box_0_05deg": int(granules.attrs.get("tiles_outside_cmr_box", -1)),
            "max_corner_overrun_deg": granules.attrs.get("max_corner_overrun_deg"),
            "max_corner_overrun_granule": granules.attrs.get("max_corner_overrun_granule"),
        },
        "allowlist": _allowlist_status(granules, anchors),
        "not_assessed": [
            "ECOSTRESS cloud, QC, and valid thermal-pixel support require granule reads.",
            "Landsat cloud_cover is not Berlin AOI clear fraction. aoi_clear_frac is published.",
            "A two-hour window is only a sensitivity bin. It is not thermal simultaneity.",
            "Cross-sensor temperature bias was not measured. No checkpoint was inferred.",
            "Tile overlap upper-bounds imaged area: the 100 km square can extend beyond the swath.",
        ],
        "code_audit": _code_audit(),
        "notion": "https://app.notion.com/p/3f8356451f66816b8b43f69f9db5386e",
    }


def _require(summary: dict) -> None:
    checks = summary["checks"]
    if not checks["cmr_hits_match_fetched_rows"]:
        raise SystemExit("CMR-Hits does not match fetched granule rows")
    if checks["merge_asof_disagreements"] != 0:
        raise SystemExit("merge_asof disagreed with the window walk")
    if checks["overlap_asof_disagreements"] != 0:
        raise SystemExit("overpass merge_asof disagreed with the window walk")
    overrun = checks["max_corner_overrun_deg"]
    if overrun is None or overrun > 0.25:
        granule = checks["max_corner_overrun_granule"]
        raise SystemExit(f"MGRS tile corner is {overrun} degrees outside the CMR box of {granule}")
    if summary["n_granules_missing_mgrs"] != 0:
        raise SystemExit("one or more granules have no MGRS tile in the id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/results/ecostress-feasibility/summary.json"),
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    _check_mgrs()
    args.workdir.mkdir(parents=True, exist_ok=True)
    anchors = _load_anchors()
    granules, hits, fetched = fetch_ecostress()
    aoi = _load_aoi()
    granules = _attach_overlap(granules, aoi)
    overpasses = _overpasses(granules, aoi)
    paired = pair(anchors, granules, overpasses)
    disagreements = _merge_asof_disagreements(paired, granules)
    overpass_disagreements = _overpass_disagreements(paired, overpasses)
    summary = _summary(
        anchors,
        granules,
        overpasses,
        paired,
        hits,
        fetched,
        disagreements,
        overpass_disagreements,
    )
    args.workdir.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(args.workdir / "anchor_matches.parquet", index=False)
    granules.to_parquet(args.workdir / "granules.parquet", index=False)
    overpasses.to_parquet(args.workdir / "overpasses.parquet", index=False)
    _require(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_root:
        root = args.output_root.rstrip("/")
        for name in (
            "anchor_matches.parquet",
            "granules.parquet",
            "overpasses.parquet",
            "summary.json",
        ):
            atomic_upload(args.workdir / name, f"{root}/{name}")
        complete = {
            "release_id": "ecostress-feasibility-v1",
            "qa_pass": True,
            "n_landsat_anchors": summary["n_landsat_anchors"],
            "n_ecostress_granules": summary["n_ecostress_granules"],
        }
        complete_path = args.workdir / "complete.json"
        complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
        atomic_upload(complete_path, f"{root}/complete.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
