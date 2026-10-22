"""Table builders. They consume dataframes and do not open rasters."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from berlin_lst_downscaling.data.eda.rules import (
    EVAL_SPLITS,
    PROPOSAL_B,
    SPARSE_AREA_KM2,
    SPARSE_EFFECTIVE_CELLS,
    group_label,
    group_role,
    scene_vintage,
)
from berlin_lst_downscaling.data.eda.stats import (
    distribution,
    hot_cold_counts,
    neighbor_abs_diffs,
    pearson,
    spearman,
    weighted_mean,
    weighted_quantile,
    weighted_sum,
    within_between,
    within_scene_correlations,
)
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.stadtstruktur.sources import BLOCK_CELLS
from berlin_lst_downscaling.data.training.contracts import NO_ELIGIBLE_CELLS_REASON

CHANNEL_FAMILY: dict[str, str] = {}
CHANNEL_UNIT: dict[str, str] = {}
for _schema_version in (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V6):
    for _channel in feature_schema(_schema_version).channels:
        CHANNEL_FAMILY[_channel.name] = _channel.family
        CHANNEL_UNIT[_channel.name] = _channel.unit


def feature_names_in(observations: pd.DataFrame) -> list[str]:
    """Roster order of the feature columns actually present.

    A complete v5 or v6 table keeps that roster. A one-channel slice,
    as the EDA operator reads, keeps the names that belong to a roster.
    """
    columns = {str(name) for name in observations.columns}
    best: list[str] = []
    for version in (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V6):
        names = list(feature_schema(version).names)
        if set(names) <= columns:
            return names
        present = [name for name in names if name in columns]
        if len(present) > len(best):
            best = present
    return best


_INVENTORY_GRAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("split_year_season", ("split", "year", "season")),
    ("split_year", ("split", "year")),
    ("year", ("year",)),
    ("split", ("split",)),
    ("all", ()),
)
_LST_GRAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scene", ("scene_id", "year", "season", "split")),
    ("split_year_season", ("split", "year", "season")),
    ("split_year", ("split", "year")),
    ("year", ("year",)),
    ("split", ("split",)),
    ("all", ()),
)


def _blank(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    return False


def _scalar_int(value: object) -> int:
    """Return one Python int. A one-element series or array is unwrapped."""
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise TypeError(f"expected one value, got {len(value)}")
        value = value.iloc[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"expected one value, got {value.size}")
        return _scalar_int(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or _missing(value):
        raise TypeError(f"expected an int, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected an int, got {type(value).__name__}")


def _scalar_float(value: object) -> float:
    """Return one Python float. A one-element series or array is unwrapped."""
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise TypeError(f"expected one value, got {len(value)}")
        value = value.iloc[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"expected one value, got {value.size}")
        return _scalar_float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or _missing(value):
        raise TypeError(f"expected a float, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected a float, got {type(value).__name__}")


def _as_int(value: object) -> int | None:
    if _missing(value):
        return None
    return _scalar_int(value)


def _require_series(value: object, label: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{label} is {type(value).__name__}, expected a Series")
    return value


def _require_frame(value: object, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} is {type(value).__name__}, expected a DataFrame")
    return value


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if len(frame) == 0:
        return 0
    values = frame[column]
    if isinstance(values, pd.Series):
        return int(values.nunique())
    if isinstance(values, np.ndarray):
        return int(np.unique(values).size)
    raise TypeError(f"{column} is {type(values).__name__}")


def _column_sum_int(frame: pd.DataFrame, column: str) -> int:
    values = frame[column]
    if isinstance(values, pd.Series):
        return _scalar_int(values.sum())
    if isinstance(values, np.ndarray):
        return _scalar_int(np.sum(values))
    raise TypeError(f"{column} is {type(values).__name__}")


def _na_identity(
    columns: tuple[str, ...],
    keys: tuple[str, ...],
    values: tuple[object, ...],
) -> dict[str, object]:
    identity: dict[str, object] = {column: pd.NA for column in columns}
    for name, item in zip(keys, values, strict=True):
        identity[name] = item
    return identity


def _group_key(key: object) -> tuple[object, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def _triple(key: object) -> tuple[object, object, object]:
    parts = _group_key(key)
    if len(parts) != 3:
        raise TypeError(f"expected a 3-part group key, got {key!r}")
    return parts[0], parts[1], parts[2]


def _ststr_id(value: object) -> str:
    parsed = _as_int(value)
    if parsed is None:
        return "na"
    return str(parsed)


def _sparse(area_km2: float, effective_cells: float) -> bool:
    return area_km2 < SPARSE_AREA_KM2 or effective_cells < SPARSE_EFFECTIVE_CELLS


def prepare_manifest(manifest: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach month and season. Scenes without a date stay ``season=unknown``."""
    frame = manifest.copy()
    dates = calendar.loc[:, ["scene_id", "month", "season"]].drop_duplicates("scene_id")
    frame = frame.merge(dates, on="scene_id", how="left")
    frame["season"] = frame["season"].fillna("unknown")
    frame["eligible_cells"] = frame["eligible_cells"].fillna(0).astype(int)
    frame["exclusion_reason"] = frame["exclusion_reason"].fillna("")
    return frame


def missing_scene_dates(manifest: pd.DataFrame, calendar: pd.DataFrame) -> list[str]:
    """Return manifest scene ids that have no acquisition month."""
    known = set(calendar["scene_id"].astype(str))
    return sorted(scene for scene in manifest["scene_id"].astype(str) if scene not in known)


def inventory_table(manifest: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Count scenes, eligible cells, and unique cells at each pre-set grain."""
    rows: list[dict[str, object]] = []
    published = manifest["status"].astype(str) == "published"
    for grain, keys in _INVENTORY_GRAINS:
        if keys:
            groups = manifest.groupby(list(keys), dropna=False, sort=False)
        else:
            groups = [((), manifest)]
        for key, part in groups:
            key_tuple = _group_key(key)
            identity = _na_identity(("split", "year", "season"), keys, key_tuple)
            scene_ids = list(set(part["scene_id"].astype(str)))
            matched = cells[cells["scene_id"].astype(str).isin(scene_ids)]
            cell_part = _require_frame(matched, "cell index")
            part_published = part[part["status"].astype(str) == "published"]
            status = part["status"].astype(str)
            rows.append(
                {
                    "grain": grain,
                    **identity,
                    "n_scenes": int(len(part)),
                    "n_published": int(len(part_published)),
                    "n_excluded": _scalar_int((status != "published").sum()),
                    "n_eligible_cells_manifest": _column_sum_int(part, "eligible_cells"),
                    "n_cell_index_rows": int(len(cell_part)),
                    "n_unique_cells": _nunique(cell_part, "cell_id"),
                }
            )
    if not rows:
        return _blank(
            [
                "grain",
                "split",
                "year",
                "season",
                "n_scenes",
                "n_published",
                "n_excluded",
                "n_eligible_cells_manifest",
                "n_cell_index_rows",
                "n_unique_cells",
            ]
        )
    del published
    return pd.DataFrame(rows)


def exclusion_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count excluded scenes by reason, split, and year."""
    excluded = manifest[manifest["status"].astype(str) != "published"]
    if excluded.empty:
        return _blank(["exclusion_reason", "split", "year", "n_scenes"])
    sizes = excluded.groupby(["exclusion_reason", "split", "year"], dropna=False, sort=False).size()
    counted = _require_series(sizes, "exclusion counts").rename("n_scenes").reset_index()
    return counted


def coverage_table(
    cells: pd.DataFrame,
    inside: pd.DataFrame,
) -> pd.DataFrame:
    """Compare eligible cells with the inside-AOI 100 m lattice."""
    inside_keys = set(zip(inside["row"].astype(int), inside["col"].astype(int), strict=True))
    n_inside = len(inside_keys)
    rows: list[dict[str, object]] = []

    def _row(grain: str, year: object, part: pd.DataFrame) -> dict[str, object]:
        keys = set(zip(part["row"].astype(int), part["col"].astype(int), strict=True))
        n_overlap = sum(1 for key in keys if key in inside_keys)
        return {
            "grain": grain,
            "year": year,
            "n_aoi_cells": n_inside,
            "n_unique_eligible_cells": len(keys),
            "n_unique_eligible_inside": n_overlap,
            "n_unique_eligible_outside": len(keys) - n_overlap,
            "fraction_aoi_eligible": (n_overlap / n_inside) if n_inside else math.nan,
        }

    rows.append(_row("all", pd.NA, cells))
    for year, part in cells.groupby("year", dropna=False, sort=True):
        rows.append(_row("year", _scalar_int(year), part))
    return pd.DataFrame(rows)


def patch_acceptance_table(manifest: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    """Separate patch-window acceptance from cell eligibility.

    A published scene that is absent from the index is ``no_valid_patch``.
    ``no_eligible_cells`` stays on the manifest and is not a patch count.
    The index is non-overlapping, so patch rows are not overlapping windows.
    """
    indexed = set(patches["scene_id"].astype(str)) if len(patches) else set()
    rows: list[dict[str, object]] = []
    grains: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("split_year", ("split", "year")),
        ("split", ("split",)),
        ("all", ()),
    )
    for grain, keys in grains:
        groups: list[tuple[object, pd.DataFrame]]
        if keys:
            groups = list(manifest.groupby(list(keys), dropna=False, sort=False))
        else:
            groups = [((), manifest)]
        for key, part in groups:
            key_tuple = _group_key(key)
            identity = _na_identity(("split", "year"), keys, key_tuple)
            scene_ids = set(part["scene_id"].astype(str))
            published = part[part["status"].astype(str) == "published"]
            published_ids = set(published["scene_id"].astype(str))
            no_eligible = part[part["exclusion_reason"].astype(str) == NO_ELIGIBLE_CELLS_REASON]
            in_index = scene_ids & indexed
            if len(patches):
                patch_part = patches[patches["scene_id"].astype(str).isin(list(scene_ids))]
            else:
                patch_part = patches
            rows.append(
                {
                    "grain": grain,
                    **identity,
                    "n_scenes": int(len(part)),
                    "n_published": int(len(published)),
                    "n_no_eligible_cells": int(len(no_eligible)),
                    "n_no_valid_patch": int(len(published_ids - indexed)),
                    "n_scenes_in_index": int(len(in_index)),
                    "n_patches": int(len(patch_part)),
                }
            )
    return pd.DataFrame(rows)


def _grouped(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> list[tuple[tuple[object, ...], pd.DataFrame]]:
    if frame.empty:
        return []
    if not keys:
        return [((), frame)]
    grouped = []
    for key, part in frame.groupby(list(keys), dropna=False, sort=False):
        grouped.append((_group_key(key), part))
    return grouped


def _identity(keys: tuple[str, ...], key: tuple[object, ...]) -> dict[str, object]:
    return _na_identity(("scene_id", "split", "year", "season"), keys, key)


def _scene_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    return float(values.mean())


def _scene_median(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    return float(values.median())


def lst_distribution_table(observations: pd.DataFrame, value: str) -> pd.DataFrame:
    """Distribute one thermal column at the observation grain and the equal-scene grain."""
    rows: list[dict[str, object]] = []
    for grain, keys in _LST_GRAINS:
        for key, part in _grouped(observations, keys):
            stats = distribution(part[value].to_numpy())
            scene_means = _require_series(
                part.groupby("scene_id", sort=False)[value].mean(),
                "scene means",
            )
            rows.append(
                {
                    "value": value,
                    "weighting": "observation",
                    "grain": grain,
                    **_identity(keys, key),
                    **stats,
                    "n_scenes": _nunique(part, "scene_id"),
                    "mean_of_scene_means": _scene_mean(scene_means),
                    "median_of_scene_medians": math.nan,
                }
            )
            medians = _require_series(
                part.groupby("scene_id", sort=False)[value].median(),
                "scene medians",
            )
            scene_stats = distribution(medians.to_numpy())
            rows.append(
                {
                    "value": value,
                    "weighting": "equal_scene",
                    "grain": grain,
                    **_identity(keys, key),
                    **scene_stats,
                    "n_scenes": int(len(medians)),
                    "mean_of_scene_means": _scene_mean(scene_means),
                    "median_of_scene_medians": _scene_median(medians),
                }
            )
    return pd.DataFrame(rows)


def between_within_table(observations: pd.DataFrame, value: str) -> pd.DataFrame:
    """One within/between row per split, plus a pooled row."""
    rows: list[dict[str, object]] = []
    pieces = [("all", observations)]
    for split, part in observations.groupby("split", dropna=False, sort=False):
        pieces.append((str(split), part))
    for split, part in pieces:
        stats = within_between(part["scene_id"].to_numpy(), part[value].to_numpy())
        rows.append({"value": value, "split": split, **stats})
    return pd.DataFrame(rows)


def hot_cold_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Per-scene hot and cold counts from that scene's own quantiles."""
    rows: list[dict[str, object]] = []
    for scene_id, part in observations.groupby("scene_id", sort=False):
        counts = hot_cold_counts(part["lst_k"].to_numpy())
        first = part.iloc[0]
        rows.append(
            {
                "scene_id": scene_id,
                "year": _scalar_int(first["year"]),
                "season": first["season"],
                "split": first["split"],
                **counts,
            }
        )
    return pd.DataFrame(rows)


def neighbor_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Per-scene distribution of absolute 4-neighbor LST differences."""
    rows: list[dict[str, object]] = []
    for scene_id, part in observations.groupby("scene_id", sort=False):
        diffs = neighbor_abs_diffs(
            part["row"].to_numpy(),
            part["col"].to_numpy(),
            part["lst_k"].to_numpy(),
        )
        first = part.iloc[0]
        rows.append(
            {
                "grain": "scene",
                "scene_id": scene_id,
                "year": _scalar_int(first["year"]),
                "season": first["season"],
                "split": first["split"],
                **distribution(diffs),
            }
        )
    per_scene = pd.DataFrame(rows)
    if per_scene.empty:
        return per_scene
    pooled: list[dict[str, object]] = []
    for grain, keys in (("year", ("year",)), ("split", ("split",)), ("all", ())):
        for key, part in _grouped(per_scene, keys):
            weight = part["n"].to_numpy(dtype=np.float64)
            mean = weighted_mean(part["mean"].to_numpy(), weight)
            pooled.append(
                {
                    "grain": grain,
                    **_identity(keys, key),
                    "n": _column_sum_int(part, "n"),
                    "mean": mean,
                    "median_of_scene_medians": _scalar_float(part["median"].median()),
                    "n_scenes": int(len(part)),
                }
            )
    pooled_frame = pd.DataFrame(pooled)
    return pd.concat([per_scene, pooled_frame], ignore_index=True)


def block_table(observations: pd.DataFrame) -> pd.DataFrame:
    """1 km block statistics from the eligible 100 m cells of one scene."""
    rows: list[dict[str, object]] = []
    group_keys = ["scene_id", "year", "season", "split", "block_row", "block_col"]
    for key, part in observations.groupby(group_keys, dropna=False, sort=False):
        stats = distribution(part["lst_k"].to_numpy())
        diffs = neighbor_abs_diffs(
            part["row"].to_numpy(),
            part["col"].to_numpy(),
            part["lst_k"].to_numpy(),
        )
        key_tuple = _group_key(key)
        identity = dict(zip(group_keys, key_tuple, strict=True))
        n_cells = int(stats["n"])
        rows.append(
            {
                **identity,
                "n_eligible": n_cells,
                "coverage": n_cells / float(BLOCK_CELLS * BLOCK_CELLS),
                "std": stats["std"],
                "lst_min": stats["min"],
                "lst_max": stats["max"],
                "lst_range": (
                    float(stats["max"]) - float(stats["min"])
                    if n_cells and math.isfinite(float(stats["min"]))
                    else math.nan
                ),
                "mean_abs_neighbor_diff": float(diffs.mean()) if diffs.size else math.nan,
                "n_neighbor_pairs": int(diffs.size),
            }
        )
    return pd.DataFrame(rows)


def feature_distribution_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Distribute each channel at the same grains as the thermal target."""
    frames: list[pd.DataFrame] = []
    names = feature_names_in(observations)
    for name in names:
        table = lst_distribution_table(observations, name)
        table.insert(0, "channel", name)
        table.insert(1, "family", CHANNEL_FAMILY[name])
        table.insert(2, "unit", CHANNEL_UNIT[name])
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_scale_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Within-scene and between-scene spread for every channel."""
    rows: list[dict[str, object]] = []
    names = feature_names_in(observations)
    for name in names:
        stats = within_between(observations["scene_id"].to_numpy(), observations[name].to_numpy())
        rows.append(
            {
                "channel": name,
                "family": CHANNEL_FAMILY[name],
                "unit": CHANNEL_UNIT[name],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def correlation_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson and Spearman with LST and among channels.

    Observation rows with a non-finite value in either member of the pair
    are dropped for that pair. Within-scene values are the median of
    per-scene correlations and do not weight a scene by its cell count.
    """
    names = ["lst_k", *feature_names_in(observations)]
    rows: list[dict[str, object]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pair = observations.loc[:, ["scene_id", left, right]].dropna()
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "grain": "observation",
                    "n": int(len(pair)),
                    "n_scenes_used": int(pair["scene_id"].nunique()) if len(pair) else 0,
                    "n_scenes_skipped": 0,
                    "pearson": pearson(pair[left].to_numpy(), pair[right].to_numpy()),
                    "spearman": spearman(pair[left].to_numpy(), pair[right].to_numpy()),
                }
            )
            within = within_scene_correlations(
                observations["scene_id"].to_numpy(),
                observations[left].to_numpy(),
                observations[right].to_numpy(),
            )
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "grain": "within_scene_median",
                    "n": int(len(observations)),
                    **within,
                    "pearson": within["median_pearson"],
                    "spearman": within["median_spearman"],
                }
            )
    return pd.DataFrame(rows)


def _proposal_column(shares: pd.DataFrame) -> pd.Series:
    def _label(value: object) -> str:
        number = _scalar_int(value)
        return PROPOSAL_B.get(number, f"unmapped_{number}")

    mapped = _require_series(shares["typ"].map(_label), "proposal labels")
    return mapped.astype(str)


def group_areas(shares: pd.DataFrame) -> pd.DataFrame:
    """Sum share area to one row per vintage, cell, and class.

    ``shares`` is already limited to cells whose centre is inside the AOI.
    Several ``typ`` values of one proposal-B group in one cell are added.
    """
    frame = shares.copy()
    frame["proposal_b"] = _proposal_column(frame)
    frame["ststrnr_id"] = frame["ststrnr"].map(_ststr_id)
    pieces = []
    for level, column in (("proposal_b", "proposal_b"), ("ststrnr", "ststrnr_id")):
        summed = frame.groupby(["vintage", "row", "col", column], dropna=False, sort=False)[
            "area_m2"
        ].sum()
        grouped = _require_series(summed, "class area").rename("area_m2").reset_index()
        grouped = grouped.rename(columns={column: "class_id"})
        grouped["level"] = level
        pieces.append(grouped)
    return pd.concat(pieces, ignore_index=True)


def _class_meta(level: str, class_id: str, names: dict[str, str]) -> tuple[str, str]:
    if level == "proposal_b" and not class_id.startswith("unmapped_"):
        return group_label(class_id), group_role(class_id)
    if level == "ststrnr":
        return names.get(class_id, class_id), "reference"
    return class_id, "data_gap"


def _name_lookup(shares: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    if "ststrname" not in shares.columns:
        return names
    for class_id, part in shares.groupby(shares["ststrnr"].map(_ststr_id), sort=False):
        labels = part["ststrname"].dropna().astype(str)
        if len(labels):
            names[str(class_id)] = str(labels.mode().iloc[0])
    return names


def _touch_counts(observations: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Count eligible scene-cells and unique cells that intersect each class."""
    columns = ["level", "class_id", "vintage", "n_unique_eligible_cells", "n_observations"]
    if observations.empty or areas.empty:
        return pd.DataFrame(columns=columns)
    observed = observations.loc[:, ["year", "row", "col", "scene_id"]].copy()
    observed["vintage"] = observed["year"].map(lambda year: scene_vintage(int(year)))
    observed = observed[observed["vintage"].notna()]
    if observed.empty:
        return pd.DataFrame(columns=columns)
    observed["vintage"] = observed["vintage"].astype(int)
    merged = observed.merge(areas, on=["vintage", "row", "col"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["cell_key"] = merged["row"].astype(str) + ":" + merged["col"].astype(str)
    return (
        merged.groupby(["level", "class_id", "vintage"], dropna=False, sort=False)
        .agg(n_unique_eligible_cells=("cell_key", "nunique"), n_observations=("scene_id", "size"))
        .reset_index()
    )


def support_table(shares: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Berlin class area, plus eligible cells that carry that class.

    Area is the vintage polygon area inside the AOI. It is not multiplied
    by the number of scenes. Eligible counts are scene×cell observations
    whose scene year maps to that vintage.
    """
    areas = group_areas(shares)
    names = _name_lookup(shares)
    area_sum = areas.groupby(["level", "class_id", "vintage"], dropna=False, sort=False)[
        "area_m2"
    ].sum()
    area_rows = _require_series(area_sum, "class area").rename("area_m2").reset_index()
    reference = area_rows[area_rows["vintage"].astype(int) == 2024].set_index(["level", "class_id"])
    touch = _touch_counts(observations, areas)
    joined = area_rows.merge(touch, on=["level", "class_id", "vintage"], how="left")
    joined["n_unique_eligible_cells"] = joined["n_unique_eligible_cells"].fillna(0).astype(int)
    joined["n_observations"] = joined["n_observations"].fillna(0).astype(int)
    rows: list[dict[str, object]] = []
    for record in joined.to_dict(orient="records"):
        level = str(record["level"])
        class_id = str(record["class_id"])
        label, role = _class_meta(level, class_id, names)
        area_m2 = _scalar_float(record["area_m2"])
        area_km2 = area_m2 / 1_000_000.0
        effective = area_m2 / 10_000.0
        sparse_2024 = False
        if (level, class_id) in reference.index:
            ref_area = _scalar_float(reference.loc[(level, class_id), "area_m2"])
            sparse_2024 = _sparse(ref_area / 1_000_000.0, ref_area / 10_000.0)
        rows.append(
            {
                "level": level,
                "class_id": class_id,
                "class_label": label,
                "role": role,
                "vintage": _scalar_int(record["vintage"]),
                "area_km2": area_km2,
                "effective_cells": effective,
                "sparse": _sparse(area_km2, effective),
                "sparse_in_2024": sparse_2024,
                "n_unique_eligible_cells": _scalar_int(record["n_unique_eligible_cells"]),
                "n_observations": _scalar_int(record["n_observations"]),
            }
        )
    return pd.DataFrame(rows)


def _with_vintage(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.copy()
    frame["vintage"] = frame["year"].map(lambda year: scene_vintage(_scalar_int(year)))
    selected = frame.loc[frame["vintage"].notna()]
    frame = _require_frame(selected, "vintage rows")
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["vintage"] = frame["vintage"].astype(int)
    return frame


def _weighted_record(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    level: str,
    class_id: str,
    label: str,
    role: str,
    vintage: int,
    split: str,
    weighting: str,
    value_name: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    total, weight, count = weighted_sum(values, weights)
    record: dict[str, object] = {
        "level": level,
        "class_id": class_id,
        "class_label": label,
        "role": role,
        "vintage": vintage,
        "split": split,
        "weighting": weighting,
        "value": value_name,
        "n": count,
        "weight_sum_m2": weight,
        "mean": (total / weight) if weight else math.nan,
        "q05": weighted_quantile(values, weights, 0.05),
        "q50": weighted_quantile(values, weights, 0.50),
        "q95": weighted_quantile(values, weights, 0.95),
    }
    if extra:
        record.update(extra)
    return record


def _class_lookup(shares: pd.DataFrame) -> dict[tuple[str, str], tuple[str, str]]:
    names = _name_lookup(shares)
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    areas = group_areas(shares)
    pairs = areas.loc[:, ["level", "class_id"]].drop_duplicates()
    for level, class_id in pairs.itertuples(index=False):
        lookup[(str(level), str(class_id))] = _class_meta(str(level), str(class_id), names)
    return lookup


def stadtstruktur_lst_table(observations: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted LST and scene-anomaly contrasts.

    ``observation_area`` weights every scene×cell by its class area.
    ``cell_area`` keeps each cell once and uses the mean scene anomaly, so
    a cell that is seen more often does not dominate the thermal contrast.
    Cells without a structure polygon are absent here and stay in the
    unstratified tables.
    """
    areas = group_areas(shares)
    lookup = _class_lookup(shares)
    keep = ["year", "split", "row", "col", "lst_k", "lst_anomaly_k"]
    observed = _with_vintage(observations.loc[:, keep])
    if observed.empty or areas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    splits = ["all", *sorted(set(observed["split"].astype(str)) & EVAL_SPLITS)]
    for split in splits:
        part = observed if split == "all" else observed[observed["split"].astype(str) == split]
        if part.empty:
            continue
        merged = part.merge(areas, on=["vintage", "row", "col"], how="inner")
        if merged.empty:
            continue
        for key, group in merged.groupby(
            ["level", "class_id", "vintage"], dropna=False, sort=False
        ):
            level, class_id, vintage_key = _triple(key)
            label, role = lookup.get((str(level), str(class_id)), (str(class_id), "data_gap"))
            weights = group["area_m2"].to_numpy()
            common = {
                "level": str(level),
                "class_id": str(class_id),
                "label": label,
                "role": role,
                "vintage": _scalar_int(vintage_key),
                "split": split,
            }
            for value_name in ("lst_k", "lst_anomaly_k"):
                rows.append(
                    _weighted_record(
                        group[value_name].to_numpy(),
                        weights,
                        weighting="observation_area",
                        value_name=value_name,
                        **common,
                    )
                )
        cell = (
            part.groupby(["vintage", "row", "col"], sort=False)
            .agg(lst_anomaly_k=("lst_anomaly_k", "mean"))
            .reset_index()
        )
        cell_merged = cell.merge(areas, on=["vintage", "row", "col"], how="inner")
        for key, group in cell_merged.groupby(
            ["level", "class_id", "vintage"], dropna=False, sort=False
        ):
            level, class_id, vintage_key = _triple(key)
            label, role = lookup.get((str(level), str(class_id)), (str(class_id), "data_gap"))
            rows.append(
                _weighted_record(
                    group["lst_anomaly_k"].to_numpy(),
                    group["area_m2"].to_numpy(),
                    level=str(level),
                    class_id=str(class_id),
                    label=label,
                    role=role,
                    vintage=_scalar_int(vintage_key),
                    split=split,
                    weighting="cell_area",
                    value_name="lst_anomaly_k",
                )
            )
    return pd.DataFrame(rows)


def stadtstruktur_feature_table(observations: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted channel means on the same class join as the LST table."""
    areas = group_areas(shares)
    lookup = _class_lookup(shares)
    names = feature_names_in(observations)
    observed = _with_vintage(observations.loc[:, ["year", "row", "col", *names]])
    if observed.empty or areas.empty or not names:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    # One pooled split. Split-specific thermal rows live in the LST table.
    merged = observed.merge(areas, on=["vintage", "row", "col"], how="inner")
    means = _require_frame(
        observed.groupby(["vintage", "row", "col"], sort=False)[names].mean(),
        "channel means",
    )
    cell = means.reset_index().merge(areas, on=["vintage", "row", "col"], how="inner")
    for name in names:
        extra: dict[str, object] = {
            "channel": name,
            "family": CHANNEL_FAMILY[name],
            "unit": CHANNEL_UNIT[name],
        }
        if not merged.empty:
            for key, group in merged.groupby(
                ["level", "class_id", "vintage"], dropna=False, sort=False
            ):
                level, class_id, vintage_key = _triple(key)
                label, role = lookup.get((str(level), str(class_id)), (str(class_id), "data_gap"))
                record = _weighted_record(
                    group[name].to_numpy(),
                    group["area_m2"].to_numpy(),
                    level=str(level),
                    class_id=str(class_id),
                    label=label,
                    role=role,
                    vintage=_scalar_int(vintage_key),
                    split="all",
                    weighting="observation_area",
                    value_name=name,
                    extra=extra,
                )
                rows.append(record)
        if not cell.empty:
            for key, group in cell.groupby(
                ["level", "class_id", "vintage"], dropna=False, sort=False
            ):
                level, class_id, vintage_key = _triple(key)
                label, role = lookup.get((str(level), str(class_id)), (str(class_id), "data_gap"))
                rows.append(
                    _weighted_record(
                        group[name].to_numpy(),
                        group["area_m2"].to_numpy(),
                        level=str(level),
                        class_id=str(class_id),
                        label=label,
                        role=role,
                        vintage=_scalar_int(vintage_key),
                        split="all",
                        weighting="cell_area",
                        value_name=name,
                        extra=extra,
                    )
                )
    return pd.DataFrame(rows)


def _lookup_weighted(table: pd.DataFrame, **query: object) -> pd.Series:
    mask = np.ones(len(table), dtype=bool)
    for column, value in query.items():
        mask &= table[column].astype(str) == str(value)
    found = table.loc[mask]
    if len(found) != 1:
        raise AssertionError(f"expected one row for {query}, found {len(found)}")
    return found.iloc[0]


def frames_self_check() -> list[str]:
    """Lock inventory counts and area weights on a two-class cell."""
    failures: list[str] = []
    manifest = pd.DataFrame(
        [
            {
                "scene_id": "s1",
                "year": 2024,
                "split": "validation",
                "status": "published",
                "eligible_cells": 1,
                "exclusion_reason": "",
            },
            {
                "scene_id": "s2",
                "year": 2026,
                "split": "inference",
                "status": "excluded",
                "eligible_cells": 0,
                "exclusion_reason": "inference_deferred",
            },
        ]
    )
    cells = pd.DataFrame(
        [
            {
                "scene_id": "s1",
                "year": 2024,
                "split": "validation",
                "cell_id": "c",
                "row": 0,
                "col": 0,
            },
            {
                "scene_id": "s1",
                "year": 2024,
                "split": "validation",
                "cell_id": "d",
                "row": 0,
                "col": 1,
            },
        ]
    )
    calendar = pd.DataFrame({"scene_id": ["s1", "s2"], "month": [7, 1], "season": ["JJA", "DJF"]})
    counted = inventory_table(prepare_manifest(manifest, calendar), cells)
    overall = counted[(counted["grain"] == "all")].iloc[0]
    if int(overall["n_scenes"]) != 2 or int(overall["n_published"]) != 1:
        failures.append("inventory scene counts drifted")
    if int(overall["n_unique_cells"]) != 2 or int(overall["n_cell_index_rows"]) != 2:
        failures.append("inventory cell counts drifted")
    shares = pd.DataFrame(
        [
            {
                "vintage": 2024,
                "row": 0,
                "col": 0,
                "cell_id": "0000_0000",
                "typ": 55,
                "ststrnr": 18,
                "ststrname": "Wald",
                "area_m2": 6000.0,
            },
            {
                "vintage": 2024,
                "row": 0,
                "col": 0,
                "cell_id": "0000_0000",
                "typ": 100,
                "ststrnr": pd.NA,
                "ststrname": "Wasser",
                "area_m2": 4000.0,
            },
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "scene_id": "s1",
                "year": 2024,
                "split": "validation",
                "row": 0,
                "col": 0,
                "lst_k": 300.0,
                "lst_anomaly_k": -5.0,
                "ndvi": 0.2,
            },
            {
                "scene_id": "s2",
                "year": 2024,
                "split": "validation",
                "row": 0,
                "col": 0,
                "lst_k": 310.0,
                "lst_anomaly_k": 5.0,
                "ndvi": 0.4,
            },
        ]
    )
    try:
        thermal = stadtstruktur_lst_table(observations, shares)
        forest = _lookup_weighted(
            thermal,
            level="proposal_b",
            class_id="forest",
            split="all",
            weighting="observation_area",
            value="lst_k",
        )
        mean_ok = abs(_scalar_float(forest["mean"]) - 305.0) > 1e-9
        weight_ok = abs(_scalar_float(forest["weight_sum_m2"]) - 12000.0) > 1e-6
        if mean_ok or weight_ok:
            failures.append(f"forest observation weight drifted: {forest['mean']}")
        anomaly = _lookup_weighted(
            thermal,
            level="proposal_b",
            class_id="forest",
            split="all",
            weighting="cell_area",
            value="lst_anomaly_k",
        )
        if abs(_scalar_float(anomaly["mean"])) > 1e-9:
            failures.append("cell-area anomaly still follows scene frequency")
        water = _lookup_weighted(
            thermal,
            level="proposal_b",
            class_id="water",
            split="all",
            weighting="observation_area",
            value="lst_k",
        )
        if (
            str(water["role"]) != "exclusion"
            or abs(_scalar_float(water["weight_sum_m2"]) - 8000.0) > 1e-6
        ):
            failures.append("water exclusion weight drifted")
    except (AssertionError, KeyError, ValueError) as exc:
        failures.append(f"stadtstruktur self-check failed: {exc}")
    return failures
