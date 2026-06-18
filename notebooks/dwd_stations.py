"""
DWD station availability for Berlin LST downscaling.
Checks which climate parameters are available at which Berlin stations,
with time coverage for the 2018–2024 training period.

Usage: uv run python notebooks/dwd_stations.py
"""

import io
import re
import urllib.request
from collections import defaultdict

# ── Remote base URL ─────────────────────────────────────────────────────
BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"

# Parameter definitions: (folder, prefix, description, cols to parse)
PARAMETERS = [
    ("air_temperature",   "TU", "Air Temperature (2m, °C)",     (0, 1, 2, 3, 4, 5)),
    ("precipitation",     "RR", "Precipitation (mm)",           (0, 1, 2, 3, 4, 5)),
    ("wind",              "FF", "Wind Speed (m/s)",             (0, 1, 2, 3, 4, 5)),
    ("solar",             "ST", "Sunshine Duration (min)",      (0, 1, 2, 3, 4, 5)),
    ("cloudiness",        "N",  "Cloud Cover (1/8)",            (0, 1, 2, 3, 4, 5)),
    ("pressure",          "P0", "Surface Pressure (hPa)",        (0, 1, 2, 3, 4, 5)),
    ("sun",               "SD", "Sunshine Duration (SD, min)",   (0, 1, 2, 3, 4, 5)),
    ("moisture",          "TF", "Temperature + Humidity",       (0, 1, 2, 3, 4, 5)),
    ("dew_point",         "TD", "Dew Point Temperature (°C)",   (0, 1, 2, 3, 4, 5)),
]

# Berlin stations from TU listing (confirmed)
BERLIN_STATIONS = {
    399:  "Berlin-Alexanderplatz",
    400:  "Berlin-Buch",
    403:  "Berlin-Dahlem (FU)",
    410:  "Berlin-Kaniswall",
    420:  "Berlin-Marzahn",
    424:  "Berlin-Ostkreuz",
    427:  "Berlin Brandenburg (BER)",
    430:  "Berlin-Tegel",
    433:  "Berlin-Tempelhof",
}


def fetch_station_file(param_dir, prefix, subfolder="historical"):
    """Fetch station description text file from DWD CDC FTP.
    
    Tries several locations: subfolder/, then root level.
    """
    locations = [
        f"{BASE}/{param_dir}/{subfolder}/{prefix}_Stundenwerte_Beschreibung_Stationen.txt",
        f"{BASE}/{param_dir}/{prefix}_Stundenwerte_Beschreibung_Stationen.txt",
        f"{BASE}/{param_dir}/historical/{prefix}_Stundenwerte_Beschreibung_Stationen.txt",
        f"{BASE}/{param_dir}/recent/{prefix}_Stundenwerte_Beschreibung_Stationen.txt",
    ]
    for url in locations:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
                return text
        except Exception:
            continue
    return None


def parse_station_file(text):
    """Parse a DWD station description file into dict {station_id: {cols}}."""
    stations = {}
    in_data = False
    for line in text.splitlines():
        # Skip header lines before the separator
        if line.strip().startswith("----"):
            in_data = True
            continue
        if not in_data or not line.strip():
            continue

        # Split by whitespace. Format: STATIONS_ID VON_DATUM BIS_DATUM HOEHE LAT LON NAME BUNDESLAND STATUS
        parts = line.split()
        if len(parts) < 8:
            continue

        try:
            sid = int(parts[0])
            von = parts[1]
            bis = parts[2]
            hoehe = parts[3]
            lat = parts[4]
            lon = parts[5]

            # The station name can contain spaces, find BUNDESLAND marker
            # Known federal state names to locate the split
            states = ["Baden-Württemberg", "Bayern", "Berlin", "Brandenburg",
                      "Bremen", "Hamburg", "Hessen", "Mecklenburg-Vorpommern",
                      "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz",
                      "Saarland", "Sachsen", "Sachsen-Anhalt",
                      "Schleswig-Holstein", "Thüringen"]

            # Find the state in the remaining parts
            remaining = " ".join(parts[6:])
            state = "Unknown"
            name = remaining
            for s in sorted(states, key=len, reverse=True):
                if s in remaining:
                    state = s
                    idx = remaining.index(s)
                    name = remaining[:idx].strip()
                    break

            # Remove trailing non-bundesland tokens (like "Frei" or similar)
            status = ""
            if state == "Unknown" and len(parts) > 7:
                # Try harder: last column might be status
                name_parts = parts[6:-1] if len(parts) > 7 else parts[6:]
                name = " ".join(name_parts)
                status = parts[-1] if len(parts) > 7 else ""

            stations[sid] = {
                "von": von,
                "bis": bis,
                "hoehe": hoehe,
                "lat": lat,
                "lon": lon,
                "name": name.strip().rstrip(","),
                "bundesland": state,
                "status": status if status else ("Frei" if len(parts) > 8 else ""),
            }
        except (ValueError, IndexError):
            continue

    return stations


def print_coverage():
    """Check which Berlin stations have which parameters and time coverage."""
    print("=" * 78)
    print("  DWD Station Availability — Berlin Area")
    print(f"  Analysis date: 2026-06-18")
    print(f"  Source: opendata.dwd.de — CDC hourly observations")
    print("=" * 78)

    # Collect: for each station, which params + (von, bis)
    station_params = defaultdict(list)
    param_labels = {}

    for param_dir, prefix, label, _ in PARAMETERS:
        param_labels[param_dir] = label
        text = fetch_station_file(param_dir, prefix)
        if text is None:
            print(f"\n  ⚠️  {label}: station file not found — skipped")
            continue

        stations = parse_station_file(text)
        found = 0
        for sid in BERLIN_STATIONS:
            if sid in stations:
                s = stations[sid]
                station_params[sid].append({
                    "param": param_dir,
                    "label": label,
                    "von": s["von"],
                    "bis": s["bis"],
                    "hoehe": s["hoehe"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "name": s["name"],
                })
                found += 1

        print(f"\n  [{prefix}] {label}")
        if found == 0:
            print(f"    0 Berlin stations found.")
        else:
            for sid in sorted(BERLIN_STATIONS):
                if sid in stations:
                    s = stations[sid]
                    print(f"    • {BERLIN_STATIONS[sid]:30s} ({s['von']} – {s['bis']}) "
                          f"[s. {s['hoehe']}m]")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Summary — Parameter Coverage per Station")
    print("  (V = available, — = not available)")
    print("=" * 78)

    header_params = [(p[0], p[2]) for p in PARAMETERS]
    # Print header
    print(f"  {'Station':35s}", end="")
    for _, lbl in header_params:
        short = lbl.split("(")[0].strip()[:10]
        print(f" {short:>10s}", end="")
    print()

    cols = len(header_params)
    print(f"  {'-'*35}", end="")
    for _ in header_params:
        print(f" {'-'*10}", end="")
    print()

    for sid in sorted(BERLIN_STATIONS):
        name = BERLIN_STATIONS[sid]
        print(f"  {name:35s}", end="")
        for pdir, _ in header_params:
            # Check if this station has this param
            has = any(pp["param"] == pdir for pp in station_params.get(sid, []))
            print(f" {'✓':>10s}" if has else f" {'—':>10s}", end="")
        print()

    # ── Recommended stations for 2018–2024 ──────────────────────────────
    print("\n" + "=" * 78)
    print("  Recommended Stations — 2018–2024 Coverage")
    print("=" * 78)
    print()

    # Check which stations cover the full 2018-2024 period for most params
    for sid in sorted(BERLIN_STATIONS):
        name = BERLIN_STATIONS[sid]
        has_params = station_params.get(sid, [])
        if not has_params:
            continue

        n_params = len(has_params)
        active_after_2018 = sum(
            1 for pp in has_params
            if pp["bis"] >= "20201231" and pp["von"] <= "20180101"
        )

        full_record = sum(1 for pp in has_params
            if pp["von"] <= "20180101" and pp["bis"] >= "20241231")

        print(f"  {name} ({n_params} params, {active_after_2018} active 2018+)")

        if full_record == n_params:
            print(f"    ✅ Station covers 2018–2024 for ALL available parameters")
        else:
            early = [pp for pp in has_params if pp["von"] > "20180101"]
            late = [pp for pp in has_params if pp["bis"] < "20241231"]
            if early:
                print(f"    ⚠️  Starts after 2018-01: {', '.join(pp['label'] for pp in early)}")
            if late:
                print(f"    ⚠️  Ends before 2024-12: {', '.join(pp['label'] for pp in late)}")

    # ── Most useful stations ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Verdict")
    print("=" * 78)
    print()
    print("  Best stations for 2018–2024 training period (many params, long record):")
    print()
    
    # Rank by: number of params + both start/end coverage
    ranking = []
    for sid in sorted(BERLIN_STATIONS):
        has_params = station_params.get(sid, [])
        if not has_params:
            continue
        n = len(has_params)
        full = sum(1 for pp in has_params
                   if pp["von"] <= "20180101" and pp["bis"] >= "20241231")
        ranking.append((sid, BERLIN_STATIONS[sid], n, full))
    
    ranking.sort(key=lambda x: (-x[2], -x[3]))
    for sid, name, n, full in ranking:
        if n >= 3:
            print(f"    ★ {name:35s} — {n} params, {full} full 2018-2024")

    print()
    print("  Key insight: DWD provides free hourly climate data at multiple")
    print("  Berlin stations. Berlin-Dahlem (FU) and Berlin-Tempelhof have")
    print("  the longest continuous records. Berlin Brandenburg (BER) is")
    print("  the primary synoptic station with widest parameter range.")
    print()

if __name__ == "__main__":
    print_coverage()
