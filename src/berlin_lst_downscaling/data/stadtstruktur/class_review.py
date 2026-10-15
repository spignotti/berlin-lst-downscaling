"""Class-support review for the published Stadtstruktur release.

Source, purpose, grain
----------------------
Source: ``evaluation/stadtstruktur/v1`` share, cell, and block tables.
Purpose: describe area, cell support, mixture, and vintage change for the
official codes and for three pre-registered groupings. This is not a model
input and it does not select classes from thermal error.
Grain: one support row is one class in one vintage. Area is the planar
share inside cells whose centre lies in Berlin. A class is counted once.
Scene anchors are a separate table and are not multiplied into area.

Group membership was fixed before these tables were computed. Sparse labels
do not move a type into another group.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry.base import BaseGeometry

from berlin_lst_downscaling.data.stadtstruktur.sources import CANDIDATE_B, TARGET_CRS
from berlin_lst_downscaling.data.stadtstruktur.tables import (
    as_int,
    as_int_required,
    float_sum,
    series_of,
)

RELEASE_ID = "stadtstruktur-class-review-v1"
SOURCE_RELEASE_ID = "stadtstruktur-v1"
APPROVAL = "awaiting_scientific_review"
CELL_M2 = 10_000.0
SPARSE_AREA_KM2 = 1.0
RARE_AREA_KM2 = 0.1
SPARSE_EFFECTIVE_CELLS = 50.0
PANEL_MATCH_MIN = 0.9
COMPARISON_HOLD_FRACTION = 0.5
UNASSIGNED_06_07 = "unassigned_06_07"
OPEN_LAND_TYPS = (55, 53, 56, 57, 54)
TRANSITION_PAIRS = ((2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2020, 2024))
EVAL_SCENE_VINTAGE: dict[int, int] = {
    2017: 2020,
    2018: 2020,
    2019: 2020,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2024,
    2025: 2024,
}
CARRY_BACK_YEARS = frozenset({2017, 2018, 2019})
CARRY_FORWARD_YEARS = frozenset({2025})

N_TYP_HISTOGRAM_NOTE = (
    "The v1 report field n_typ_histogram counts cells by how many typ values "
    "they contain. Its keys are mixture depth, not 06.08 typ codes, and this "
    "review does not use it as a class distribution."
)


@dataclass(frozen=True)
class GroupSpec:
    """One pre-registered group."""

    group_id: str
    label: str
    role: str
    note: str


PROPOSAL_B_GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        "dense_closed_block",
        "Closed pre-war blocks",
        "comparison",
        "Closed or heavily modified pre-war blocks of about four to six storeys.",
    ),
    GroupSpec(
        "mixed_shed_block",
        "Mixed shed-yard blocks",
        "boundary",
        "Lower-rise mixed shed yards. Not pooled with the closed five-to-six-storey blocks.",
    ),
    GroupSpec(
        "interwar_block_and_row",
        "Interwar blocks and rows",
        "comparison",
        "1920s–1940s block-edge and row housing.",
    ),
    GroupSpec(
        "postwar_open_and_highrise",
        "Post-war open housing and high-rises",
        "comparison",
        "Post-war estates, rows, and high-rises. Heterogeneous infill is separate.",
    ),
    GroupSpec(
        "heterogeneous_infill",
        "Heterogeneous inner-city infill",
        "boundary",
        "Post-1945 inner-city infill. Not pooled with post-war estates.",
    ),
    GroupSpec(
        "low_rise_residential",
        "Low-rise and garden housing",
        "comparison",
        "Village, row, detached, and villa housing, including weekend-house areas.",
    ),
    GroupSpec(
        "core_area",
        "Core area",
        "boundary",
        "Kerngebiet. Dense core, not an industrial estate.",
    ),
    GroupSpec(
        "utilities",
        "Utilities",
        "boundary",
        "Ver- und Entsorgung. Kept out of the commercial estate group.",
    ),
    GroupSpec(
        "commercial_industrial",
        "Commercial and industrial estates",
        "comparison",
        "Commercial and industrial estates and non-residential mixed areas.",
    ),
    GroupSpec(
        "public_and_special",
        "Public and special uses",
        "heterogeneous",
        "Schools, culture, health, and other public uses. Not a thermal stratum.",
    ),
    GroupSpec("forest", "Forest", "comparison", "Wald."),
    GroupSpec(
        "park_cemetery",
        "Parks, green space, and cemeteries",
        "comparison",
        "Parks and cemeteries. Cemetery is a boundary inside this group, not forest.",
    ),
    GroupSpec(
        "agriculture",
        "Agriculture and horticulture",
        "comparison",
        "Agriculture and nurseries. Horticulture is a boundary inside this group.",
    ),
    GroupSpec("allotment", "Allotments", "comparison", "Kleingartenanlagen."),
    GroupSpec("fallow", "Fallow land", "comparison", "Brachfläche."),
    GroupSpec(
        "camping",
        "Camping",
        "boundary",
        "Camping. Kept separate so a small class is not absorbed by a neighbour.",
    ),
    GroupSpec(
        "sealed_plaza",
        "Sealed plaza",
        "comparison",
        "Stadtplatz and promenade. Sealed open land, not vegetation.",
    ),
    GroupSpec(
        "sport_open",
        "Uncovered sports grounds",
        "boundary",
        "Uncovered sports grounds. Partly sealed, so not pooled with parks.",
    ),
    GroupSpec(
        "transport_mapped",
        "Mapped transport",
        "comparison",
        "Parking, rail, airport, and other mapped transport. ISU5 street land is absent.",
    ),
    GroupSpec(
        "construction",
        "Construction",
        "exclusion",
        "Construction sites. Exclusion or separate check, not a comparison group.",
    ),
    GroupSpec(
        "water",
        "Water",
        "exclusion",
        "Water. Exclusion or separate check. The official 06.07 code is null.",
    ),
)

PROPOSAL_C_GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        "residential",
        "All residential fabric",
        "comparison",
        "Coarse pool of every residential proposal-B group, including boundary groups.",
    ),
    GroupSpec(
        "commercial_industrial",
        "Commercial, core, and utilities",
        "comparison",
        "Coarse pool of estates, Kerngebiet, and utilities.",
    ),
    GroupSpec(
        "public_and_special",
        "Public and special uses",
        "heterogeneous",
        "Same public bin as proposal B.",
    ),
    GroupSpec(
        "vegetated_open",
        "Re-merged open land",
        "remerged_open_land",
        "Forest, park, agriculture, allotment, fallow, and camping pooled again.",
    ),
    GroupSpec(
        "transport_and_sealed",
        "Transport, plazas, and open sports",
        "incomplete",
        "Mapped transport plus sealed plazas and uncovered sports. Street land is absent.",
    ),
    GroupSpec(
        "construction",
        "Construction",
        "exclusion",
        "Same exclusion bin as proposal B.",
    ),
    GroupSpec("water", "Water", "exclusion", "Same exclusion bin as proposal B."),
)

# Official 06.08 typ -> proposal B. Membership follows the type description.
PROPOSAL_B: dict[int, str] = {
    1: "dense_closed_block",
    2: "dense_closed_block",
    3: "dense_closed_block",
    7: "dense_closed_block",
    6: "mixed_shed_block",
    10: "interwar_block_and_row",
    72: "interwar_block_and_row",
    9: "postwar_open_and_highrise",
    11: "postwar_open_and_highrise",
    73: "postwar_open_and_highrise",
    8: "heterogeneous_infill",
    21: "low_rise_residential",
    22: "low_rise_residential",
    23: "low_rise_residential",
    24: "low_rise_residential",
    25: "low_rise_residential",
    59: "low_rise_residential",
    29: "core_area",
    32: "utilities",
    30: "commercial_industrial",
    31: "commercial_industrial",
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
    55: "forest",
    53: "park_cemetery",
    27: "park_cemetery",
    56: "agriculture",
    36: "agriculture",
    37: "allotment",
    57: "fallow",
    58: "camping",
    54: "sealed_plaza",
    16: "sport_open",
    91: "transport_mapped",
    92: "transport_mapped",
    93: "transport_mapped",
    94: "transport_mapped",
    99: "transport_mapped",
    98: "construction",
    100: "water",
}

PROPOSAL_C_FROM_B: dict[str, str] = {
    "dense_closed_block": "residential",
    "mixed_shed_block": "residential",
    "interwar_block_and_row": "residential",
    "postwar_open_and_highrise": "residential",
    "heterogeneous_infill": "residential",
    "low_rise_residential": "residential",
    "core_area": "commercial_industrial",
    "utilities": "commercial_industrial",
    "commercial_industrial": "commercial_industrial",
    "public_and_special": "public_and_special",
    "forest": "vegetated_open",
    "park_cemetery": "vegetated_open",
    "agriculture": "vegetated_open",
    "allotment": "vegetated_open",
    "fallow": "vegetated_open",
    "camping": "vegetated_open",
    "sealed_plaza": "transport_and_sealed",
    "sport_open": "transport_and_sealed",
    "transport_mapped": "transport_and_sealed",
    "construction": "construction",
    "water": "water",
}

TYP_NOTES: dict[int, str] = {
    6: "Schuppenhof, two to four storeys. Boundary against the closed pre-war blocks.",
    8: "Heterogeneous inner-city infill after 1945. Boundary against post-war estates.",
    16: "Uncovered sports ground. Boundary against parks because the surface is partly sealed.",
    27: "Cemetery. Boundary inside park and green, not forest.",
    36: "Nursery or horticulture. Boundary inside agriculture.",
    59: "Weekend-house and allotment-like housing. Boundary against Kleingartenanlagen.",
    93: "Airport. Present in the 2020 vintage only.",
}

_B_BY_ID = {spec.group_id: spec for spec in PROPOSAL_B_GROUPS}
_C_BY_ID = {spec.group_id: spec for spec in PROPOSAL_C_GROUPS}


def assert_crosswalk_closed() -> None:
    """Raise when a pre-registered map drops or invents a published typ."""
    if set(PROPOSAL_B) != set(CANDIDATE_B):
        missing = sorted(set(CANDIDATE_B) - set(PROPOSAL_B))
        extra = sorted(set(PROPOSAL_B) - set(CANDIDATE_B))
        raise RuntimeError(f"proposal B typ map differs from published B: {missing} {extra}")
    group_ids = {spec.group_id for spec in PROPOSAL_B_GROUPS}
    if set(PROPOSAL_B.values()) != group_ids:
        raise RuntimeError("proposal B groups and typ map do not match")
    if set(PROPOSAL_C_FROM_B) != group_ids:
        raise RuntimeError("proposal C does not cover every proposal B group")
    if set(PROPOSAL_C_FROM_B.values()) != {spec.group_id for spec in PROPOSAL_C_GROUPS}:
        raise RuntimeError("proposal C labels do not cover the collapsed groups")
    for spec in PROPOSAL_B_GROUPS:
        if spec.role not in {"comparison", "boundary", "heterogeneous", "exclusion"}:
            raise RuntimeError(f"unknown proposal B role {spec.role}")


def scene_vintage(year: int) -> int | None:
    """Return the evaluation vintage for one scene year, if one is assigned."""
    return EVAL_SCENE_VINTAGE.get(year)


def carry_rule(year: int) -> str:
    """Name the pre-registered temporal rule for one scene year."""
    if year in CARRY_BACK_YEARS:
        return "carry_back"
    if year in CARRY_FORWARD_YEARS:
        return "carry_forward"
    if year in EVAL_SCENE_VINTAGE:
        return "same_year"
    return "unassigned"


def load_aoi(path: str) -> BaseGeometry:
    """Read the Berlin boundary as one valid EPSG:25833 geometry."""
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise RuntimeError(f"{path} has no CRS")
    projected = frame.to_crs(TARGET_CRS)
    geometry = make_valid(projected.geometry.union_all())
    if geometry.is_empty:
        raise RuntimeError("AOI is empty")
    return geometry


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(type(value))
    return float(value)


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def slim_polygons(blocks: gpd.GeoDataFrame, aoi: BaseGeometry) -> pd.DataFrame:
    """Keep block parts that intersect Berlin and drop their geometry."""
    mask = np.asarray(blocks.geometry.intersects(aoi), dtype=bool)
    kept = blocks.loc[mask]
    keys = [_text(value) for value in kept["schluessel"]]
    if any(key is None for key in keys):
        raise RuntimeError("null schluessel on an AOI polygon")
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate schluessel inside the AOI")
    return pd.DataFrame(
        {
            "schluessel": keys,
            "typ": [as_int_required(value) for value in kept["typ"]],
            "typklar": [_text(value) for value in kept["typklar"]],
            "ststrnr": [as_int(value) for value in kept["ststrnr"]],
            "ststrname": [_text(value) for value in kept["ststrname"]],
            "area_m2": np.asarray(kept.geometry.area, dtype=float),
        }
    )


def _ststr_id(value: object) -> str:
    parsed = as_int(value)
    if parsed is None:
        return UNASSIGNED_06_07
    return str(parsed)


@dataclass
class _Bucket:
    area_m2: float = 0.0
    n_cells: int = 0
    n_mixed: int = 0
    n_dominant_covered_0_5: int = 0
    n_dominant_covered_0_8: int = 0
    n_dominant_cell_0_5: int = 0
    n_dominant_cell_0_8: int = 0


def _accumulate(cell_ids: list[str], class_ids: list[str], areas: list[float]) -> tuple[
    dict[str, _Bucket],
    int,
    int,
]:
    covered: dict[str, float] = {}
    members: dict[str, set[str]] = {}
    pair_area: dict[tuple[str, str], float] = {}
    for cell_id, class_id, area in zip(cell_ids, class_ids, areas, strict=True):
        covered[cell_id] = covered.get(cell_id, 0.0) + area
        members.setdefault(cell_id, set()).add(class_id)
        pair = (cell_id, class_id)
        pair_area[pair] = pair_area.get(pair, 0.0) + area
    mixed = {cell_id for cell_id, classes in members.items() if len(classes) > 1}
    buckets: dict[str, _Bucket] = {}
    for (cell_id, class_id), area in pair_area.items():
        bucket = buckets.setdefault(class_id, _Bucket())
        bucket.area_m2 += area
        bucket.n_cells += 1
        if cell_id in mixed:
            bucket.n_mixed += 1
        share_covered = area / covered[cell_id]
        share_cell = area / CELL_M2
        if share_covered >= 0.5:
            bucket.n_dominant_covered_0_5 += 1
        if share_covered >= 0.8:
            bucket.n_dominant_covered_0_8 += 1
        if share_cell >= 0.5:
            bucket.n_dominant_cell_0_5 += 1
        if share_cell >= 0.8:
            bucket.n_dominant_cell_0_8 += 1
    return buckets, len(members), len(mixed)


def _support_row(
    *,
    vintage: int,
    level: str,
    class_id: str,
    label: str,
    role: str,
    bucket: _Bucket,
    n_polygons: int,
    classified_area_m2: float,
) -> dict[str, object]:
    area_km2 = bucket.area_m2 / 1_000_000.0
    effective = bucket.area_m2 / CELL_M2
    share = 0.0 if classified_area_m2 == 0 else bucket.area_m2 / classified_area_m2
    mixed_fraction = None if bucket.n_cells == 0 else bucket.n_mixed / bucket.n_cells
    return {
        "vintage": vintage,
        "level": level,
        "class_id": class_id,
        "class_label": label,
        "role": role,
        "area_m2": bucket.area_m2,
        "area_km2": area_km2,
        "share_of_classified": share,
        "n_polygons": n_polygons,
        "n_cells_touched": bucket.n_cells,
        "effective_cells": effective,
        "n_mixed_cells": bucket.n_mixed,
        "mixed_cell_fraction": mixed_fraction,
        "n_dominant_covered_0_5": bucket.n_dominant_covered_0_5,
        "n_dominant_covered_0_8": bucket.n_dominant_covered_0_8,
        "n_dominant_cell_0_5": bucket.n_dominant_cell_0_5,
        "n_dominant_cell_0_8": bucket.n_dominant_cell_0_8,
        "sparse": area_km2 < SPARSE_AREA_KM2 or effective < SPARSE_EFFECTIVE_CELLS,
        "rare": area_km2 < RARE_AREA_KM2,
    }


def support_for_level(
    *,
    vintage: int,
    level: str,
    cell_ids: list[str],
    class_ids: list[str],
    areas: list[float],
    catalog: list[tuple[str, str, str]],
    polygon_counts: dict[str, int],
    classified_area_m2: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Area, cell, mixture, and dominance rows for one grouping."""
    buckets, n_covered, n_mixed = _accumulate(cell_ids, class_ids, areas)
    rows: list[dict[str, object]] = []
    for class_id, label, role in catalog:
        rows.append(
            _support_row(
                vintage=vintage,
                level=level,
                class_id=class_id,
                label=label,
                role=role,
                bucket=buckets.get(class_id, _Bucket()),
                n_polygons=polygon_counts.get(class_id, 0),
                classified_area_m2=classified_area_m2,
            )
        )
    mixture = {
        "vintage": vintage,
        "level": level,
        "n_covered_cells": n_covered,
        "n_mixed_cells": n_mixed,
        "mixed_cell_fraction": None if n_covered == 0 else n_mixed / n_covered,
    }
    return rows, mixture


def _modal(pairs: list[tuple[str, str | None]]) -> dict[str, str | None]:
    counts: dict[str, dict[str, int]] = {}
    for key, label in pairs:
        if label is None:
            continue
        bucket = counts.setdefault(key, {})
        bucket[label] = bucket.get(label, 0) + 1
    chosen: dict[str, str | None] = {}
    for key, bucket in counts.items():
        chosen[key] = max(bucket.items(), key=lambda item: item[1])[0]
    return chosen


def build_crosswalk(slim_by_year: dict[int, pd.DataFrame]) -> list[dict[str, object]]:
    """One row per official typ, with every grouping and the boundary note."""
    frames = [slim_by_year[year] for year in sorted(slim_by_year)]
    combined = pd.concat(frames, ignore_index=True)
    typ_codes = [as_int_required(value) for value in combined["typ"]]
    observed = set(typ_codes)
    missing = sorted(observed - set(PROPOSAL_B))
    if missing:
        raise RuntimeError(f"typ codes missing from the crosswalk: {missing}")
    labels = _modal(
        [
            (str(typ), _text(label))
            for typ, label in zip(typ_codes, combined["typklar"], strict=True)
        ]
    )
    ststr_pairs: dict[int, list[int | None]] = {}
    ststr_names: list[tuple[str, str | None]] = []
    for typ, ststr, name in zip(
        typ_codes, combined["ststrnr"], combined["ststrname"], strict=True
    ):
        parsed = as_int(ststr)
        ststr_pairs.setdefault(typ, []).append(parsed)
        ststr_names.append((_ststr_id(parsed), _text(name)))
    name_for_ststr = _modal(ststr_names)
    rows: list[dict[str, object]] = []
    for typ in sorted(set(PROPOSAL_B)):
        variants = {value for value in ststr_pairs.get(typ, [])}
        modal_code = None
        if variants:
            counts: dict[int | None, int] = {}
            for value in ststr_pairs[typ]:
                counts[value] = counts.get(value, 0) + 1
            modal_code = max(counts.items(), key=lambda item: item[1])[0]
        group_b = PROPOSAL_B[typ]
        note = TYP_NOTES.get(typ, _B_BY_ID[group_b].note)
        if len(variants) > 1:
            note = f"{note} More than one official 06.07 code occurs on this typ."
        rows.append(
            {
                "typ": typ,
                "typklar": labels.get(str(typ)),
                "ststrnr": modal_code,
                "ststrname": name_for_ststr.get(_ststr_id(modal_code)),
                "n_ststrnr_variants": len(variants),
                "published_b": CANDIDATE_B[typ],
                "proposal_b": group_b,
                "proposal_b_role": _B_BY_ID[group_b].role,
                "proposal_c": PROPOSAL_C_FROM_B[group_b],
                "boundary_note": note,
            }
        )
    return rows


def _catalog_from_crosswalk(
    crosswalk: list[dict[str, object]],
) -> dict[str, list[tuple[str, str, str]]]:
    typ_rows = [
        (
            str(row["typ"]),
            "" if row["typklar"] is None else str(row["typklar"]),
            "official_typ",
        )
        for row in crosswalk
    ]
    ststr: dict[str, str] = {}
    for row in crosswalk:
        class_id = _ststr_id(row["ststrnr"])
        label = row["ststrname"]
        if class_id not in ststr and label is not None:
            ststr[class_id] = str(label)
    if UNASSIGNED_06_07 not in ststr:
        ststr[UNASSIGNED_06_07] = "No 06.07 code (water)"
    ststr_rows = [
        (class_id, ststr[class_id], "official_06_07") for class_id in sorted(ststr)
    ]
    published: dict[str, str] = {}
    for row in crosswalk:
        published[str(row["published_b"])] = str(row["published_b"])
    published_rows = [
        (class_id, class_id, "published_proposal") for class_id in sorted(published)
    ]
    proposal_b_rows = [
        (spec.group_id, spec.label, spec.role) for spec in PROPOSAL_B_GROUPS
    ]
    proposal_c_rows = [
        (spec.group_id, spec.label, spec.role) for spec in PROPOSAL_C_GROUPS
    ]
    return {
        "typ": typ_rows,
        "ststrnr": ststr_rows,
        "published_b": published_rows,
        "proposal_b": proposal_b_rows,
        "proposal_c": proposal_c_rows,
    }


def _class_ids(slim: pd.DataFrame, level: str) -> list[str]:
    typ_codes = [as_int_required(value) for value in slim["typ"]]
    if level == "typ":
        return [str(code) for code in typ_codes]
    if level == "ststrnr":
        return [_ststr_id(value) for value in slim["ststrnr"]]
    if level == "published_b":
        return [CANDIDATE_B[code] for code in typ_codes]
    if level == "proposal_b":
        return [PROPOSAL_B[code] for code in typ_codes]
    if level == "proposal_c":
        return [PROPOSAL_C_FROM_B[PROPOSAL_B[code]] for code in typ_codes]
    raise KeyError(level)


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def annotate_inside_shares(
    shares: pd.DataFrame, inside_cell_ids: set[str]
) -> tuple[pd.DataFrame, float, float]:
    """Split share rows on the cell-centre AOI rule and attach group ids."""
    cell_ids = [str(value) for value in shares["cell_id"]]
    areas = [float(value) for value in shares["area_m2"]]
    typ_codes = [as_int_required(value) for value in shares["typ"]]
    unknown = sorted({code for code in typ_codes if code not in PROPOSAL_B})
    if unknown:
        raise RuntimeError(f"observed typ codes are missing from the crosswalk: {unknown}")
    inside_rows: dict[str, list[object]] = {
        "cell_id": [],
        "block_id": [],
        "area_m2": [],
        "typ_id": [],
        "ststr_id": [],
        "published_b": [],
        "proposal_b": [],
        "proposal_c": [],
    }
    inside_area = 0.0
    outside_area = 0.0
    block_ids = [str(value) for value in shares["block_id"]]
    ststr_values = list(shares["ststrnr"])
    for index, cell_id in enumerate(cell_ids):
        area = areas[index]
        if cell_id not in inside_cell_ids:
            outside_area += area
            continue
        inside_area += area
        code = typ_codes[index]
        group_b = PROPOSAL_B[code]
        inside_rows["cell_id"].append(cell_id)
        inside_rows["block_id"].append(block_ids[index])
        inside_rows["area_m2"].append(area)
        inside_rows["typ_id"].append(str(code))
        inside_rows["ststr_id"].append(_ststr_id(ststr_values[index]))
        inside_rows["published_b"].append(CANDIDATE_B[code])
        inside_rows["proposal_b"].append(group_b)
        inside_rows["proposal_c"].append(PROPOSAL_C_FROM_B[group_b])
    return pd.DataFrame(inside_rows), inside_area, outside_area


def _string_column(frame: pd.DataFrame, name: str) -> list[str]:
    return [str(value) for value in frame[name]]


def _float_column(frame: pd.DataFrame, name: str) -> list[float]:
    return [float(value) for value in frame[name]]


def support_tables(
    *,
    vintage: int,
    inside: pd.DataFrame,
    slim: pd.DataFrame,
    catalogs: dict[str, list[tuple[str, str, str]]],
    classified_area_m2: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Per-class support and one mixture row per level."""
    rows: list[dict[str, object]] = []
    mixture: list[dict[str, object]] = []
    cell_ids = _string_column(inside, "cell_id")
    areas = _float_column(inside, "area_m2")
    level_column = {
        "typ": "typ_id",
        "ststrnr": "ststr_id",
        "published_b": "published_b",
        "proposal_b": "proposal_b",
        "proposal_c": "proposal_c",
    }
    for level, column in level_column.items():
        class_ids = _string_column(inside, column)
        level_rows, level_mixture = support_for_level(
            vintage=vintage,
            level=level,
            cell_ids=cell_ids,
            class_ids=class_ids,
            areas=areas,
            catalog=catalogs[level],
            polygon_counts=_count(_class_ids(slim, level)),
            classified_area_m2=classified_area_m2,
        )
        rows.extend(level_rows)
        mixture.append(level_mixture)
    return rows, mixture


def _polygon_index(slim: pd.DataFrame) -> dict[str, tuple[int, int | None, float]]:
    index: dict[str, tuple[int, int | None, float]] = {}
    for key, typ, ststr, area in zip(
        slim["schluessel"], slim["typ"], slim["ststrnr"], slim["area_m2"], strict=True
    ):
        index[str(key)] = (as_int_required(typ), as_int(ststr), float(area))
    return index


def transition_summary(
    slim_by_year: dict[int, pd.DataFrame],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Key match rates and typ flows. A low match rate is not treated as a panel."""
    indexes = {year: _polygon_index(slim) for year, slim in slim_by_year.items()}
    summaries: list[dict[str, object]] = []
    flows: list[dict[str, object]] = []
    for year_from, year_to in TRANSITION_PAIRS:
        left = indexes[year_from]
        right = indexes[year_to]
        keys_from = set(left)
        keys_to = set(right)
        both = keys_from & keys_to
        match_from = len(both) / len(keys_from)
        match_to = len(both) / len(keys_to)
        usable = match_from >= PANEL_MATCH_MIN and match_to >= PANEL_MATCH_MIN
        stable_typ = 0
        changed_typ = 0
        stable_ststr = 0
        changed_ststr = 0
        stable_area = 0.0
        changed_area_to = 0.0
        matched_area_to = 0.0
        flow_area: dict[tuple[int, int], float] = {}
        for key in both:
            typ_from, ststr_from, _area_from = left[key]
            typ_to, ststr_to, area_to = right[key]
            matched_area_to += area_to
            if typ_from == typ_to:
                stable_typ += 1
                stable_area += area_to
            else:
                changed_typ += 1
                changed_area_to += area_to
                pair = (typ_from, typ_to)
                flow_area[pair] = flow_area.get(pair, 0.0) + area_to
            if ststr_from == ststr_to:
                stable_ststr += 1
            else:
                changed_ststr += 1
        disappeared = keys_from - keys_to
        appeared = keys_to - keys_from
        summaries.append(
            {
                "vintage_from": year_from,
                "vintage_to": year_to,
                "n_from": len(keys_from),
                "n_to": len(keys_to),
                "n_matched": len(both),
                "match_fraction_of_from": match_from,
                "match_fraction_of_to": match_to,
                "panel_key_usable": usable,
                "n_stable_typ": stable_typ,
                "n_changed_typ": changed_typ,
                "n_stable_ststr": stable_ststr,
                "n_changed_ststr": changed_ststr,
                "n_disappeared": len(disappeared),
                "n_appeared": len(appeared),
                "stable_typ_area_m2": stable_area,
                "changed_typ_area_to_m2": changed_area_to,
                "matched_area_to_m2": matched_area_to,
                "stable_typ_area_fraction": (
                    None if matched_area_to == 0 else stable_area / matched_area_to
                ),
                "disappeared_area_m2": sum(left[key][2] for key in disappeared),
                "appeared_area_m2": sum(right[key][2] for key in appeared),
            }
        )
        if (year_from, year_to) == (2020, 2024):
            for (typ_from, typ_to), area in sorted(flow_area.items()):
                flows.append(
                    {
                        "vintage_from": year_from,
                        "vintage_to": year_to,
                        "typ_from": typ_from,
                        "typ_to": typ_to,
                        "area_to_m2": area,
                    }
                )
    return summaries, flows


def block_sensitivity(
    *,
    vintage: int,
    block_shares: pd.DataFrame,
    inside_block_ids: set[str],
) -> list[dict[str, object]]:
    """Mixture and smallest group area on 1 km blocks that meet the AOI cells."""
    block_ids = [str(value) for value in block_shares["block_id"]]
    areas = [float(value) for value in block_shares["area_m2"]]
    typ_codes = [as_int_required(value) for value in block_shares["typ"]]
    ststr_ids = [_ststr_id(value) for value in block_shares["ststrnr"]]
    columns = {
        "typ": [str(code) for code in typ_codes],
        "ststrnr": ststr_ids,
        "published_b": [CANDIDATE_B[code] for code in typ_codes],
        "proposal_b": [PROPOSAL_B[code] for code in typ_codes],
        "proposal_c": [PROPOSAL_C_FROM_B[PROPOSAL_B[code]] for code in typ_codes],
    }
    rows: list[dict[str, object]] = []
    for level, class_ids in columns.items():
        members: dict[str, set[str]] = {}
        group_area: dict[str, float] = {}
        for block_id, class_id, area in zip(block_ids, class_ids, areas, strict=True):
            if block_id not in inside_block_ids:
                continue
            members.setdefault(block_id, set()).add(class_id)
            group_area[class_id] = group_area.get(class_id, 0.0) + area
        n_blocks = len(members)
        n_mixed = sum(1 for classes in members.values() if len(classes) > 1)
        positive = [area for area in group_area.values() if area > 0]
        rows.append(
            {
                "vintage": vintage,
                "level": level,
                "n_blocks": n_blocks,
                "n_mixed_blocks": n_mixed,
                "mixed_block_fraction": None if n_blocks == 0 else n_mixed / n_blocks,
                "n_groups_with_area": len(positive),
                "min_group_area_km2": None if not positive else min(positive) / 1_000_000.0,
            }
        )
    return rows


def scene_assignment(anchor_counts: dict[int, int]) -> list[dict[str, object]]:
    """Assign scene years to vintages. Counts stay counts, not area weights."""
    years = sorted(set(EVAL_SCENE_VINTAGE) | set(anchor_counts))
    rows: list[dict[str, object]] = []
    for year in years:
        vintage = scene_vintage(year)
        rows.append(
            {
                "scene_year": year,
                "vintage": vintage,
                "carry": carry_rule(year),
                "n_anchors": int(anchor_counts.get(year, 0)),
            }
        )
    return rows


def _rows_for(
    support: list[dict[str, object]], *, vintage: int, level: str
) -> list[dict[str, object]]:
    return [
        row
        for row in support
        if row["vintage"] == vintage and row["level"] == level
    ]


def recommend(support: list[dict[str, object]]) -> dict[str, object]:
    """Fill the pre-registered recommendation. Do not mark it approved."""
    comparison = [
        row
        for row in _rows_for(support, vintage=2024, level="proposal_b")
        if row["role"] == "comparison"
    ]
    if not comparison:
        raise RuntimeError("proposal B has no comparison groups")
    sparse = [str(row["class_id"]) for row in comparison if bool(row["sparse"])]
    held = 1.0 - (len(sparse) / len(comparison))
    typ_area = {
        int(str(row["class_id"])): _as_float(row["area_km2"])
        for row in _rows_for(support, vintage=2024, level="typ")
    }
    open_areas = {typ: typ_area.get(typ, 0.0) for typ in OPEN_LAND_TYPS}
    open_confirmed = all(area > 0 for area in open_areas.values())
    support_status = (
        "adequate" if held >= COMPARISON_HOLD_FRACTION and open_confirmed else "mostly_sparse"
    )
    boundary_sparse = [
        str(row["class_id"])
        for row in _rows_for(support, vintage=2024, level="proposal_b")
        if row["role"] == "boundary" and bool(row["sparse"])
    ]
    if open_confirmed:
        published_status = "rejected_on_semantic_rule"
        published_sentence = (
            "Published candidate B is not the working grouping: Wald (55), "
            "Park (53), Landwirtschaft (56), Brache (57), and Stadtplatz (54) "
            "each have area in 2024 and were pooled in vegetated_open."
        )
    else:
        published_status = "semantic_objection_not_confirmed_by_area"
        published_sentence = (
            "Published candidate B remains semantically coarse, but at least one of "
            "Wald, Park, Landwirtschaft, Brache, and Stadtplatz has no 2024 area, "
            "so the area check does not confirm the split."
        )
    sparse_text = ", ".join(sparse) if sparse else "none"
    verb = "is" if len(sparse) == 1 else "are"
    text = (
        "Proposal B is the working grouping for a later stratified description. "
        "It is not approved. Official 06.07 ststrnr remains the external reference. "
        f"{published_sentence} "
        f"{len(sparse)} of {len(comparison)} comparison groups {verb} under the "
        f"pre-set floor of {SPARSE_AREA_KM2:.1f} km² or "
        f"{SPARSE_EFFECTIVE_CELLS:.0f} effective cells: {sparse_text}. "
        "Groups under that floor stay in their pre-registered class and are not "
        "merged into a neighbour. Proposal C re-pools open land and is only a "
        "sensitivity. Water and construction stay exclusion categories. Public and "
        "special uses stay one heterogeneous bin. No thermal error entered this choice."
    )
    return {
        "recommended_grouping": "proposal_b",
        "reference_grouping": "ststrnr",
        "published_b_status": published_status,
        "support_status": support_status,
        "comparison_groups": len(comparison),
        "comparison_groups_held_fraction": held,
        "sparse_comparison_groups": sparse,
        "sparse_boundary_groups": boundary_sparse,
        "open_land_area_km2_2024": {str(typ): area for typ, area in open_areas.items()},
        "open_land_split_confirmed": open_confirmed,
        "approval": APPROVAL,
        "text": text,
    }


def _check_area(name: str, left: float, right: float) -> float:
    difference = abs(left - right)
    if difference > 1.0:
        raise RuntimeError(f"{name} differs by {difference} m²")
    return difference


def reconcile_year(
    *,
    vintage: int,
    share_area_m2: float,
    report_share_area_m2: float,
    inside_area_m2: float,
    outside_area_m2: float,
    report_source_inside_m2: float,
    report_gap_m2: float,
    report_aoi_area_m2: float,
) -> dict[str, object]:
    """Check conservation. The centre rule is reported, not forced onto the gap."""
    share_diff = _check_area(
        f"{vintage} share area", share_area_m2, report_share_area_m2
    )
    split_diff = _check_area(
        f"{vintage} centre split", inside_area_m2 + outside_area_m2, share_area_m2
    )
    gap_diff = _check_area(
        f"{vintage} gap identity", report_source_inside_m2 + report_gap_m2, report_aoi_area_m2
    )
    return {
        "vintage": vintage,
        "share_area_m2": share_area_m2,
        "report_share_area_m2": report_share_area_m2,
        "share_abs_diff_m2": share_diff,
        "inside_centre_area_m2": inside_area_m2,
        "outside_centre_area_m2": outside_area_m2,
        "centre_split_abs_diff_m2": split_diff,
        "report_source_inside_aoi_m2": report_source_inside_m2,
        "street_gap_m2": report_gap_m2,
        "street_gap_km2": report_gap_m2 / 1_000_000.0,
        "report_aoi_area_m2": report_aoi_area_m2,
        "gap_plus_source_inside_abs_diff_m2": gap_diff,
        "inside_centre_minus_source_inside_m2": inside_area_m2 - report_source_inside_m2,
    }


def _svg(title: str, labels: list[str], values: list[float], unit: str) -> str:
    width = 860
    bar_height = 18
    gap = 6
    left = 280
    top = 36
    height = top + len(labels) * (bar_height + gap) + 24
    peak = max(values) if values else 1.0
    if peak <= 0:
        peak = 1.0
    plot_width = width - left - 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:sans-serif;font-size:12px;fill:#222}</style>",
        f'<text x="16" y="22">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + index * (bar_height + gap)
        length = plot_width * value / peak
        parts.append(
            f'<text x="16" y="{y + 13}">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{length:.1f}" height="{bar_height}" fill="#3d6b99"/>'
            f'<text x="{left + length + 6:.1f}" y="{y + 13}">{value:.2f} {html.escape(unit)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def figures(support: list[dict[str, object]], mixture: list[dict[str, object]]) -> dict[str, str]:
    """Two bar charts. Values are the 2024 review tables, not a styling layer."""
    proposal = sorted(
        _rows_for(support, vintage=2024, level="proposal_b"),
        key=lambda row: _as_float(row["area_km2"]),
        reverse=True,
    )
    area_svg = _svg(
        "Proposal B classified area, 2024",
        [str(row["class_id"]) for row in proposal],
        [_as_float(row["area_km2"]) for row in proposal],
        "km²",
    )
    mix_rows = [row for row in mixture if row["vintage"] == 2024]
    mix_values: list[float] = []
    for row in mix_rows:
        fraction = row["mixed_cell_fraction"]
        mix_values.append(0.0 if fraction is None else _as_float(fraction))
    mix_svg = _svg(
        "Share of covered 100 m cells that contain more than one class, 2024",
        [str(row["level"]) for row in mix_rows],
        mix_values,
        "",
    )
    return {
        "area_proposal_b_2024.svg": area_svg,
        "mixture_by_level_2024.svg": mix_svg,
    }


@dataclass
class ClassReview:
    """Tables and the compact report for one local run."""

    report: dict[str, object]
    support: pd.DataFrame
    crosswalk: pd.DataFrame
    transitions: pd.DataFrame
    transition_flows: pd.DataFrame
    mixture: pd.DataFrame
    block_sensitivity: pd.DataFrame
    figures: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(name)
    return value


def _vintage_reports(source_report: dict[str, object]) -> dict[int, dict[str, object]]:
    vintages = source_report["vintages"]
    if not isinstance(vintages, list):
        raise TypeError("v1 report vintages is not a list")
    parsed: dict[int, dict[str, object]] = {}
    for item in vintages:
        vintage = _mapping(item, "vintage")
        parsed[as_int_required(vintage["vintage"])] = vintage
    return parsed


def build_class_review(
    *,
    source_report: dict[str, object],
    inside_by_year: dict[int, pd.DataFrame],
    outside_area_by_year: dict[int, float],
    share_area_by_year: dict[int, float],
    slim_by_year: dict[int, pd.DataFrame],
    block_shares_by_year: dict[int, pd.DataFrame],
    anchor_counts: dict[int, int],
    file_sha256: dict[str, str],
    source_release_uri: str,
    manifest_uri: str,
) -> ClassReview:
    """Assemble the review from in-memory tables of the published release."""
    assert_crosswalk_closed()
    crosswalk = build_crosswalk(slim_by_year)
    catalogs = _catalog_from_crosswalk(crosswalk)
    by_year = _vintage_reports(source_report)
    support: list[dict[str, object]] = []
    mixture: list[dict[str, object]] = []
    reconciliation: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    for year in sorted(inside_by_year):
        inside = inside_by_year[year]
        classified = 0.0 if len(inside) == 0 else float_sum(series_of(inside["area_m2"]))
        level_rows, level_mixture = support_tables(
            vintage=year,
            inside=inside,
            slim=slim_by_year[year],
            catalogs=catalogs,
            classified_area_m2=classified,
        )
        typ_area = sum(
            _as_float(row["area_m2"]) for row in level_rows if row["level"] == "typ"
        )
        _check_area(f"{year} typ support", typ_area, classified)
        for level in ("ststrnr", "published_b", "proposal_b", "proposal_c"):
            level_area = sum(
                _as_float(row["area_m2"]) for row in level_rows if row["level"] == level
            )
            _check_area(f"{year} {level} support", level_area, classified)
        support.extend(level_rows)
        mixture.extend(level_mixture)
        vintage_report = by_year[year]
        coverage = _mapping(vintage_report["coverage"], f"{year} coverage")
        grid = _mapping(vintage_report["grid"], f"{year} grid")
        reconciliation.append(
            reconcile_year(
                vintage=year,
                share_area_m2=share_area_by_year[year],
                report_share_area_m2=_as_float(grid["share_area_m2"]),
                inside_area_m2=classified,
                outside_area_m2=outside_area_by_year[year],
                report_source_inside_m2=_as_float(coverage["source_inside_aoi_m2"]),
                report_gap_m2=_as_float(coverage["aoi_uncovered_m2"]),
                report_aoi_area_m2=_as_float(coverage["aoi_area_m2"]),
            )
        )
        sensitivity.extend(
            block_sensitivity(
                vintage=year,
                block_shares=block_shares_by_year[year],
                inside_block_ids=set(_string_column(inside, "block_id")),
            )
        )
    transitions, flows = transition_summary(slim_by_year)
    recommendation = recommend(support)
    page_sha = {
        str(year): str(by_year[year]["page_sha256"]) for year in sorted(by_year)
    }
    report: dict[str, object] = {
        "release_id": RELEASE_ID,
        "source_release_id": SOURCE_RELEASE_ID,
        "source_release_uri": source_release_uri,
        "manifest_uri": manifest_uri,
        "approval": APPROVAL,
        "n_typ_histogram_note": N_TYP_HISTOGRAM_NOTE,
        "thresholds": {
            "sparse_area_km2": SPARSE_AREA_KM2,
            "rare_area_km2": RARE_AREA_KM2,
            "sparse_effective_cells": SPARSE_EFFECTIVE_CELLS,
            "dominance": [0.5, 0.8],
            "panel_match_min": PANEL_MATCH_MIN,
            "comparison_hold_fraction": COMPARISON_HOLD_FRACTION,
        },
        "area_rule": (
            "Class area sums shares.parquet area_m2 on cells with inside_aoi. "
            "The denominator is that classified area, not the Berlin boundary. "
            "Street and other unmapped land stays out of every class share. "
            "Effective cells equal area_m2 / 10000. Polygon counts come from "
            "block parts that intersect the boundary, once per schluessel."
        ),
        "inputs": {
            "page_sha256": page_sha,
            "file_sha256": file_sha256,
        },
        "reconciliation": reconciliation,
        "mixture": mixture,
        "scene_year_assignment": scene_assignment(anchor_counts),
        "transitions": transitions,
        "recommendation": recommendation,
        "crosswalk": crosswalk,
        "support": support,
    }
    return ClassReview(
        report=report,
        support=pd.DataFrame(support),
        crosswalk=pd.DataFrame(crosswalk),
        transitions=pd.DataFrame(transitions),
        transition_flows=pd.DataFrame(flows),
        mixture=pd.DataFrame(mixture),
        block_sensitivity=pd.DataFrame(sensitivity),
        figures=figures(support, mixture),
    )
