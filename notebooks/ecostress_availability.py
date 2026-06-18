"""
ECOSTRESS availability analysis for Berlin LST downscaling validation.
Uses NASA CMR + earthaccess to query ECO_L2T_LSTE v002 (70m LST COGs).

Usage:
    export EARTHDATA_TOKEN="<your-token>"
    uv run python notebooks/ecostress_availability.py
"""

import os
import earthaccess
from collections import Counter
from datetime import datetime

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
END = '2026-06-30'

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
tiles = Counter()
first_dates = []
overpass_times = []  # hours of day

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

    granules_by_year[dt.year] += 1
    granules_by_month[dt.month] += 1
    overpass_times.append(dt.hour)

    # Day/night (ECOSTRESS has a Day/Night flag in metadata)
    data_quality = umm.get('DataQuality', {})
    day_night_flag = data_quality.get('Description', '') if data_quality else ''
    if 'NIGHT' in str(g).upper():
        day_night['Night'] += 1
    elif 'DAY' in str(g).upper():
        day_night['Day'] += 1
    else:
        day_night['Unknown'] += 1

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
print(f"\n{'='*65}")
print(f"  ECOSTRESS Availability — Berlin")
print(f"  {START} – {END}")
print(f"{'='*65}")

# Yearly
print(f"\n  Per Year:")
print(f"  {'Year':>6s} | {'Granules':>9s}")
print(f"  {'-'*6} | {'-'*9}")
for yr in sorted(granules_by_year):
    print(f"  {yr:>6d} | {granules_by_year[yr]:>9d}")

# Monthly
months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
print(f"\n  Per Month (all years):")
print(f"  {'Month':>6s} | {'Granules':>9s}")
print(f"  {'-'*6} | {'-'*9}")
for mi in sorted(granules_by_month):
    print(f"  {months[mi]:>6s} | {granules_by_month[mi]:>9d}")

# Overpass times
print(f"\n  Overpass Times (hour of day):")
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
summer_count = sum(granules_by_month[m] for m in summer_months)
total_count = sum(granules_by_month.values())
print(f"\n  Summer (May–Sep) granules: {summer_count} ({summer_count/total_count*100:.0f}% of total)")
print(f"  Other months: {total_count - summer_count}")

print(f"\nDone.\n")
