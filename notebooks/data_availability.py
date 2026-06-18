"""
Data availability analysis for Berlin LST downscaling.
Checks Landsat 8/9 TIRS and Sentinel-2 L2A scene counts per year/month.

Usage: uv run python notebooks/data_availability.py
"""

import ee
import pandas as pd

# ── Auth ───────────────────────────────────────────────────────────────
ee.Initialize(project='masterarbeit-berlin-lst')

# ── Parameters ─────────────────────────────────────────────────────────
BERLIN_BBOX = ee.Geometry.Rectangle([13.08, 52.34, 13.76, 52.68])
START = '2013-01-01'
END = '2026-06-30'
CLOUD_THRESHOLD = 20  # percent


# ── Helpers ────────────────────────────────────────────────────────────
def pull_scene_metadata(collection_id, cloud_prop, sat_prop, extra_props=None):
    """Extract per-scene metadata from a GEE collection.

    Returns DataFrame with columns: date, cloud_cover, satellite,
    year, month, plus any extra_props.
    """
    col = ee.ImageCollection(collection_id).filterBounds(BERLIN_BBOX).filterDate(START, END)

    prop_map = {
        'date_str': ('date().format("YYYY-MM-dd")', True),
        'cloud_cover': (cloud_prop, False),  # direct property
        'satellite': (sat_prop, False),
    }
    if extra_props:
        for ek, ep in extra_props.items():
            prop_map[ek] = (ep, False)

    def extract(img):
        d = {}
        for key, (expr, is_method) in prop_map.items():
            if key == 'date_str':
                d[key] = img.date().format('YYYY-MM-dd')
            else:
                val = img.get(expr)
                d[key] = val
        return ee.Feature(None, d)

    fc = col.map(extract)
    info = fc.getInfo()

    rows = []
    for f in info['features']:
        rows.append(f['properties'])

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date_str'])
    df['year'] = df['date'].dt.year.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    return df.drop(columns=['date_str'])


def summarize(df, label, cloud_threshold=CLOUD_THRESHOLD):
    """Print per-year and per-month summaries with/without cloud filter."""
    cloud_free = df[df['cloud_cover'] < cloud_threshold]

    # Per-year summary
    yearly = df.groupby('year').size().to_frame('total')
    yearly['cloud_free'] = cloud_free.groupby('year').size()
    yearly['filtered_out'] = yearly['total'] - yearly['cloud_free']
    yearly = yearly.fillna(0).astype(int)

    # Per-month summary (all years combined)
    monthly = df.groupby('month').size().to_frame('total')
    monthly['cloud_free'] = cloud_free.groupby('month').size()
    monthly['filtered_out'] = monthly['total'] - monthly['cloud_free']
    monthly = monthly.fillna(0).astype(int)

    # Count by satellite (if multiple)
    if 'satellite' in df.columns:
        sat_counts = df['satellite'].value_counts()
    else:
        sat_counts = pd.Series(dtype=int)

    return {
        'label': label,
        'total_scenes': len(df),
        'cloud_free_scenes': len(cloud_free),
        'yearly': yearly,
        'monthly': monthly,
        'satellite_counts': sat_counts,
    }


def print_summary(result):
    """Pretty-print a summary dict."""
    print(f"\n{'='*70}")
    print(f"  {result['label']}")
    print(f"  Total scenes: {result['total_scenes']}  "
          f"|  <{CLOUD_THRESHOLD}% cloud: {result['cloud_free_scenes']}  "
          f"|  Filtered out: {result['total_scenes'] - result['cloud_free_scenes']}")
    if len(result['satellite_counts']) > 1:
        print(f"  Satellites: {result['satellite_counts'].to_dict()}")

    print(f"\n  Per Year (total | <{CLOUD_THRESHOLD}% cloud | filtered):")
    y = result['yearly']
    for yr in sorted(y.index):
        row = y.loc[yr]
        print(f"    {yr}: {row['total']:4d} | {row['cloud_free']:4d} | {row['filtered_out']:4d}")

    months = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
              7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    print(f"\n  Per Month (all years, total | <{CLOUD_THRESHOLD}% cloud):")
    m = result['monthly']
    for mi in sorted(m.index):
        row = m.loc[mi]
        pct = (row['cloud_free'] / row['total'] * 100) if row['total'] > 0 else 0
        print(f"    {months[mi]:>3s}: {row['total']:4d} | {row['cloud_free']:4d} "
              f"({pct:.0f}% usable)")


# ── LANDSAT 8/9 TIRS ───────────────────────────────────────────────────
print("\n\n[1/3] Querying Landsat 8/9 ...", flush=True)

landsat_props = {'WRS_PATH': 'WRS_PATH', 'WRS_ROW': 'WRS_ROW'}
df_l8 = pull_scene_metadata(
    'LANDSAT/LC08/C02/T1_L2',
    cloud_prop='CLOUD_COVER', sat_prop='SPACECRAFT_ID',
    extra_props=landsat_props,
)
df_l9 = pull_scene_metadata(
    'LANDSAT/LC09/C02/T1_L2',
    cloud_prop='CLOUD_COVER', sat_prop='SPACECRAFT_ID',
    extra_props=landsat_props,
)

r8 = summarize(df_l8, 'Landsat 8 (LC08)')
r9 = summarize(df_l9, 'Landsat 9 (LC09)')
print_summary(r8)
print_summary(r9)

# Combined Landsat summary
df_landsat = pd.concat([df_l8, df_l9], ignore_index=True)
r_landsat = summarize(df_landsat, 'Landsat 8+9 Combined')
print_summary(r_landsat)

# First/last dates per satellite
print(f"\n  Landsat 8 first scene: {df_l8['date'].min().date()}  last: {df_l8['date'].max().date()}")
print(f"  Landsat 9 first scene: {df_l9['date'].min().date()}  last: {df_l9['date'].max().date()}")

# Unique WRS path/row combos
for name, df_s in [('Landsat 8', df_l8), ('Landsat 9', df_l9)]:
    if 'WRS_PATH' in df_s.columns:
        paths = df_s[['WRS_PATH', 'WRS_ROW']].drop_duplicates()
        print(f"  {name} WRS paths: {sorted(paths['WRS_PATH'].unique())}")


# ── SENTINEL-2 L2A ─────────────────────────────────────────────────────
print("\n\n[2/3] Querying Sentinel-2 L2A ...", flush=True)

def pull_s2_via_aggregate(collection_id, start, end):
    """Use aggregate_array to avoid 5000-element GEE limit."""
    col = ee.ImageCollection(collection_id).filterBounds(BERLIN_BBOX).filterDate(start, end)
    n = col.size().getInfo()
    if n == 0:
        return pd.DataFrame(columns=['date', 'cloud_cover', 'satellite'])
    dates_ms = col.aggregate_array('system:time_start').getInfo()
    clouds = col.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    sats = col.aggregate_array('SPACECRAFT_NAME').getInfo()
    df = pd.DataFrame({
        'date': pd.to_datetime(dates_ms, unit='ms'),
        'cloud_cover': clouds,
        'satellite': sats,
    })
    df['year'] = df['date'].dt.year.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    return df

try:
    df_s2 = pull_s2_via_aggregate('COPERNICUS/S2_SR_HARMONIZED', START, END)
    r_s2 = summarize(df_s2, 'Sentinel-2 L2A (S2_SR_HARMONIZED)')
    print_summary(r_s2)

    print(f"\n  Sentinel-2 first scene: {df_s2['date'].min().date()}  last: {df_s2['date'].max().date()}")
    if 'satellite' in df_s2.columns:
        counts = df_s2['satellite'].value_counts().to_dict()
        print(f"  Satellites: {counts}")
except Exception as e:
    print(f"  ERROR querying Sentinel-2: {e}")


# ── ECOSTRESS ──────────────────────────────────────────────────────────
print("\n\n[3/3] Checking ECOSTRESS ...", flush=True)

ecos = {
    'ECO2LSTE (LST)': 'ECOSTRESS/ECO2LSTE.001',
    'ECO3ETPTJPL (ET)': 'ECOSTRESS/ECO3ETPTJPL.001',
}

for name, col_id in ecos.items():
    try:
        col = ee.ImageCollection(col_id).filterBounds(BERLIN_BBOX).filterDate(START, END)
        n = col.size().getInfo()
        if n > 0:
            first = ee.Image(col.sort('system:time_start').first())
            first_date = first.date().format('YYYY-MM-dd').getInfo()
            last = ee.Image(col.sort('system:time_start', False).first())
            last_date = last.date().format('YYYY-MM-dd').getInfo()
            print(f"  {name}: {n} scenes ({first_date} – {last_date})")
        else:
            print(f"  {name}: 0 scenes — Berlin not covered")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")


print("\n\nDone.\n")
