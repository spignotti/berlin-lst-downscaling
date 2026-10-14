"""Live source catalog for Umweltatlas Stadtstruktur.

Verified against the Berlin Open Data records and WFS GetCapabilities on
2026-10-14. Counts below are service ``numberMatched`` values from that
access, not a guarantee that a later service revision stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass

RELEASE_ID = "stadtstruktur-v1"
TARGET_CRS = "EPSG:25833"
CELL_SIZE_M = 100
BLOCK_CELLS = 10
BLOCK_SIZE_M = CELL_SIZE_M * BLOCK_CELLS

LICENSE_ID = "dl-de/zero-2.0"
LICENSE_NAME = "Datenlizenz Deutschland – Zero – Version 2.0"
LICENSE_URL = "https://www.govdata.de/dl-de/zero-2-0"
PUBLISHER = "Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin"

# Candidate B is a proposal for stratified evaluation. It is not an
# approved scientific classification and is not written as the cell label.
CANDIDATE_B_STATUS = "proposal_awaiting_scientific_approval"
CANDIDATE_B_JUSTIFICATION = (
    "Groups share building era and density, or a dominant surface type "
    "(vegetation, water, transport, construction). Public and special uses "
    "stay together because the official types mix schools, culture, and "
    "utilities at block scale. Original typ codes stay recoverable."
)

# Official 06.08 typ code -> proposal id. Water (100) has no 06.07 code.
CANDIDATE_B: dict[int, str] = {
    1: "dense_closed_block",
    2: "dense_closed_block",
    3: "dense_closed_block",
    6: "dense_closed_block",
    7: "dense_closed_block",
    10: "interwar_block_and_row",
    72: "interwar_block_and_row",
    8: "postwar_open_and_highrise",
    9: "postwar_open_and_highrise",
    11: "postwar_open_and_highrise",
    73: "postwar_open_and_highrise",
    21: "low_rise_residential",
    22: "low_rise_residential",
    23: "low_rise_residential",
    24: "low_rise_residential",
    25: "low_rise_residential",
    59: "low_rise_residential",
    29: "commercial_industrial",
    30: "commercial_industrial",
    31: "commercial_industrial",
    32: "commercial_industrial",
    33: "commercial_industrial",
    38: "commercial_industrial",
    12: "public_and_special",
    13: "public_and_special",
    17: "public_and_special",
    41: "public_and_special",
    43: "public_and_special",
    44: "public_and_special",
    45: "public_and_special",
    46: "public_and_special",
    47: "public_and_special",
    49: "public_and_special",
    51: "public_and_special",
    60: "public_and_special",
    91: "transport",
    92: "transport",
    93: "transport",
    94: "transport",
    99: "transport",
    16: "vegetated_open",
    27: "vegetated_open",
    36: "vegetated_open",
    37: "vegetated_open",
    53: "vegetated_open",
    54: "vegetated_open",
    55: "vegetated_open",
    56: "vegetated_open",
    57: "vegetated_open",
    58: "vegetated_open",
    98: "construction",
    100: "water",
}

CANDIDATE_B_LABELS: dict[str, str] = {
    "dense_closed_block": "Closed or heavily modified pre-war blocks",
    "interwar_block_and_row": "1920s–1930s block and row development",
    "postwar_open_and_highrise": "Post-war row, estate, and high-rise housing",
    "low_rise_residential": "Low-rise, village, and garden housing",
    "commercial_industrial": "Commercial and industrial areas",
    "public_and_special": "Public facilities and special uses",
    "transport": "Transport surfaces outside the street-land layer",
    "vegetated_open": "Vegetated and other open land",
    "construction": "Construction sites",
    "water": "Water",
}


@dataclass(frozen=True)
class VintageSource:
    """One published ISU5 Stadtstruktur vintage."""

    vintage: int
    reference_date: str
    wfs_url: str
    type_name: str
    service_feature_count: int
    notes: str


VINTAGES: tuple[VintageSource, ...] = (
    VintageSource(
        vintage=2020,
        reference_date="2020-12-31",
        wfs_url="https://gdi.berlin.de/services/wfs/ua_stadtstruktur_2020",
        type_name="ua_stadtstruktur_2020:ua_stadtstruktur_diff_2020",
        service_feature_count=26378,
        notes=(
            "Open-data record 'Stadtstruktur 2020 (Umweltatlas)'. "
            "Map imprint completed October 2021. Sachdatenstand 31.12.2020. "
            "The differentiated layer carries both 06.08 typ and 06.07 ststrnr."
        ),
    ),
    VintageSource(
        vintage=2021,
        reference_date="2021-12-31",
        wfs_url="https://gdi.berlin.de/services/wfs/ua_stadtstruktur",
        type_name="ua_stadtstruktur:b_stadtstruktur_differenziert_2021",
        service_feature_count=26397,
        notes=(
            "Annual ISU5 update. Feature count matches the Senate ISU5 note "
            "of 26,397 block parts for 31.12.2021."
        ),
    ),
    VintageSource(
        vintage=2022,
        reference_date="2022-12-31",
        wfs_url="https://gdi.berlin.de/services/wfs/ua_stadtstruktur",
        type_name="ua_stadtstruktur:b_stadtstruktur_differenziert_2022",
        service_feature_count=26422,
        notes="Annual ISU5 update, Sachdatenstand 31.12.2022.",
    ),
    VintageSource(
        vintage=2023,
        reference_date="2023-12-31",
        wfs_url="https://gdi.berlin.de/services/wfs/ua_stadtstruktur",
        type_name="ua_stadtstruktur:b_stadtstruktur_differenziert_2023",
        service_feature_count=26431,
        notes="Annual ISU5 update, Sachdatenstand 31.12.2023.",
    ),
    VintageSource(
        vintage=2024,
        reference_date="2024-12-31",
        wfs_url="https://gdi.berlin.de/services/wfs/ua_stadtstruktur",
        type_name="ua_stadtstruktur:b_stadtstruktur_differenziert_2024",
        service_feature_count=26613,
        notes="Annual ISU5 update, Sachdatenstand 31.12.2024. Latest open vintage.",
    ),
)

OPEN_DATA_RECORDS = {
    2020: "https://daten.berlin.de/datensaetze/stadtstruktur-2020-umweltatlas-wfs-cca244ae",
    2021: "https://daten.berlin.de/datensaetze/stadtstruktur-umweltatlas-ab-2021-wfs-8065af3b",
}

# Scene years in the LST window and whether a same-year vintage exists.
# No vintage is assigned here. Carry-back and carry-forward stay unapproved.
SCENE_YEAR_COVERAGE: dict[int, str] = {
    2017: "no_open_vintage",
    2018: "no_open_vintage",
    2019: "no_open_vintage",
    2020: "vintage_2020",
    2021: "vintage_2021",
    2022: "vintage_2022",
    2023: "vintage_2023",
    2024: "vintage_2024",
    2025: "no_open_vintage",
}


def vintage_by_year(year: int) -> VintageSource:
    """Return the catalog entry for one reference year."""
    for source in VINTAGES:
        if source.vintage == year:
            return source
    raise KeyError(year)
