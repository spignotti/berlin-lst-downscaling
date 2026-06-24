"""
ECOSTRESS availability analysis for Berlin LST downscaling validation.
Uses NASA CMR + earthaccess to query ECO_L2T_LSTE v002 (70m LST COGs).

Usage:
    export EARTHDATA_TOKEN="<your-token>"
    uv run python notebooks/ecostress_availability.py
"""

import os
import socket
from collections import Counter
from datetime import datetime

# ── IPv4 workaround ────────────────────────────────────────────────────
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return [r for r in results if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only

import earthaccess

# ── Auth ───────────────────────────────────────────────────────────────
# Strategy: environment (EARTHDATA_TOKEN) → netrc → interactive
strategies = ['environment', 'netrc', 'interactive']
auth = None
for s in strategies:
    try:
        auth = earthaccess.login(strategy=s)
        if auth.authenticated:
            break
    except Exception:
        continue

if not auth or not auth.authenticated:
    print("ERROR: No valid Earthdata auth found. Set EARTHDATA_TOKEN env var.")
    exit(1)

# ── Parameters ─────────────────────────────────────────────────────────
BERLIN_BBOX = (13.08, 52.34, 13.76, 52.68)  # WGS84
START = '2018-01-01'
END = '2025-12-31'
# Landsat overpass Berlin is ~10:00 local time (UTC+1 in winter, UTC+2 in summer)
# Landsat-adjacent window: 8:00-12:00 local → 6:00-11:00 UTC depending on DST
LANDSAT_WINDOW = (6, 11)  # UTC hours — conservative window covering both winter/summer

# ── Query CMR ──────────────────────────────────────────────────────────
print("Searching CMR for ECO_L2T_LSTE v002 over Berlin...")
print(f"  Bbox: {BERLIN_BBOX}")
print(f"  Date: {START} – {END}")
print()

results = earthaccess.search_data(
    short_name='ECO_L2T_LSTE',
    version='002',
    bounding_box=BERLIN_BBOX,
    temporal=(START, END),
    count=5000,  # max results to fetch
)

total_hits = getattr(results, 'hit_count', len(results))
print(f"  Total granules in CMR: {total_hits}")
print(f"  Fetched:              {len(results)}")
if len(results) < total_hits:
    print("  ⚠️  Results limited by CMR page size — total_hits indicates more exist.")
if len(results) == 0:
    print("\n⚠️  No ECOSTRESS granules found for this bbox. Check coordinates.")
    exit(0)

# ── Parse metadata ─────────────────────────────────────────────────────
granules_by_year = Counter()
granules_by_month = Counter()
day_night = Counter()
landsat_adjacent_by_year = Counter()  # granules near Landsat overpass
tiles = Counter()
first_dates = []
overpass_times = []  # hours of day (UTC)
LANDSAT_WINDOW = (6, 11)  # UTC hours — covers ~8-12 local time in both CET/CEST

for g in results:
    meta = g.get('meta', {})
    umm = g.get('umm', {})
    
    # Date from temporal extent
    extent = umm.get('TemporalExtent', {})
    date_str = extent.get('RangeDateTime', {}).get('BeginningDateTime', '')
    
    if not date_str:
        # Try the granule UR or other metadata
        date_str = meta.get('native-id', '')
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', ''))
    except (ValueError, TypeError):
        continue

    hour_utc = dt.hour
    overpass_times.append(hour_utc)
    granules_by_year[dt.year] += 1
    granules_by_month[dt.month] += 1

    # Landsat-adjacent: ECOSTRESS near Landsat overpass (~10:00 local)
    # Landsat window UTC: 6-11 covers ~8:00-12:00 local in both CET (UTC+1) and CEST (UTC+2)
    near_landsat = LANDSAT_WINDOW[0] <= hour_utc <= LANDSAT_WINDOW[1]
    if near_landsat:
        landsat_adjacent_by_year[dt.year] += 1

    # Day/night: day = 6-18 UTC, night = 18-6 UTC
    if 6 <= hour_utc < 18:
        day_night['Day'] += 1
    else:
        day_night['Night'] += 1

    # Track first/last
    first_dates.append(dt)

    # MGRS tile from granule UR
    granule_ur = meta.get('native-id', '')
    # Extract tile code (e.g., ..._T33UUT_...)
    parts = granule_ur.split('_')
    for p in parts:
        if p.startswith('T') and len(p) in [5, 6] and p[1:].isalnum():
            tiles[p] += 1

first_dates.sort()

# ── Print results ──────────────────────────────────────────────────────
months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

print(f"\n{'='*65}")
print(f"  ECOSTRESS Availability — Berlin")
print(f"  {START} – {END}")
print(f"{'='*65}")

# Yearly
print(f"\n  Per Year:")
print(f"  {'Year':>6s} | {'Granules':>9s} | {'Landsat-adj.':>12s}")
print(f"  {'-'*6} | {'-'*9} | {'-'*12}")
for yr in sorted(granules_by_year):
    la = landsat_adjacent_by_year.get(yr, 0)
    pct = la / granules_by_year[yr] * 100 if granules_by_year[yr] > 0 else 0
    print(f"  {yr:>6d} | {granules_by_year[yr]:>9d} | {la:>5d} ({pct:>4.0f}%)")

# Monthly
print(f"\n  Per Month (all years):")
print(f"  {'Month':>6s} | {'Granules':>9s}")
print(f"  {'-'*6} | {'-'*9}")
for mi in sorted(granules_by_month):
    print(f"  {months[mi]:>6s} | {granules_by_month[mi]:>9d}")

# Overpass times
print(f"\n  Overpass Times (hour of day, UTC):")
time_bins = Counter()
for h in overpass_times:
    if h < 4:
        time_bins['  0–3 (late night)'] += 1
    elif h < 8:
        time_bins['  4–7 (early morning)'] += 1
    elif h < 12:
        time_bins[' 8–11 (morning)'] += 1
    elif h < 16:
        time_bins['12–15 (afternoon)'] += 1
    elif h < 20:
        time_bins['16–19 (evening)'] += 1
    else:
        time_bins['20–23 (night)'] += 1
for label, count in sorted(time_bins.items(), key=lambda x: int(x[0].strip().split('–')[0])):
    print(f"    {label}: {count}")
print(f"    ─────────────────────────────────────")
print(f"    Landsat window (6–11 UTC): {landsat_adjacent_by_year.total()} total")

# Day/night
print(f"\n  Day/Night split:")
for label in ['Day', 'Night']:
    print(f"    {label}: {day_night.get(label, 0)}")

# MGRS tiles
print(f"\n  MGRS Tiles (top 5):")
for tile, count in tiles.most_common(5):
    print(f"    {tile}: {count}")

# Dates
if first_dates:
    print(f"\n  First granule: {first_dates[0].date()}")
    print(f"  Last granule:  {first_dates[-1].date()}")

# Summer focus
summer_months = [5, 6, 7, 8, 9]  # May–Sep
total_count = sum(granules_by_month.values())
summer_count = sum(granules_by_month[m] for m in summer_months)
print(f"\n  Summer (May–Sep) granules: {summer_count} ({summer_count/total_count*100:.0f}% of total)")
print(f"  Other months: {total_count - summer_count}")

# 2024/2025 specific
print(f"\n  ── Focus: 2024 & 2025 ──")
for yr in [2024, 2025]:
    cnt = granules_by_year.get(yr, 0)
    la = landsat_adjacent_by_year.get(yr, 0)
    pct = la / cnt * 100 if cnt > 0 else 0
    print(f"  {yr}: {cnt} total, {la} Landsat-adjacent ({pct:.0f}%)")

print(f"\nDone.\n")
