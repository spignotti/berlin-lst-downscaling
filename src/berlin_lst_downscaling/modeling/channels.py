"""Active feature-channel selection for Stage-1 and Tag-11 ablations.

The published V3 stack keeps a fixed 28-channel order (spectral → index →
morphology → ERA5 → shadow). Stage 1 and any contiguous prefix use
``v3_first_c``. Stage 3 needs shadows without ERA5, so it selects a named
subset that is not a prefix of that order. Scaler stats stay the full
train-only V3 fit; selection happens after scaling, on model input only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig, ListConfig, OmegaConf

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)

N_FEATURE_CHANNELS: int = len(FEATURE_CHANNEL_NAMES)

# Family slices into the fixed V3 order (inclusive start, exclusive end).
_SPECTRAL_INDEX = FEATURE_CHANNEL_NAMES[0:10]
_MORPHOLOGY = FEATURE_CHANNEL_NAMES[10:18]
_ERA5 = FEATURE_CHANNEL_NAMES[18:26]
_SHADOW = FEATURE_CHANNEL_NAMES[26:28]

# Cumulative ablation ladder (issue #57). Stage 5 reuses stage-4 inputs.
ABLATION_STAGE_CHANNELS: dict[int, tuple[str, ...]] = {
    1: _SPECTRAL_INDEX,
    2: _SPECTRAL_INDEX + _MORPHOLOGY,
    3: _SPECTRAL_INDEX + _MORPHOLOGY + _SHADOW,
    4: _SPECTRAL_INDEX + _MORPHOLOGY + _SHADOW + _ERA5,
    5: _SPECTRAL_INDEX + _MORPHOLOGY + _SHADOW + _ERA5,
}

# Single-family additions on the spectral block (issue #63). These are not
# rungs of the cumulative ladder: shadows without morphology, ERA5 without
# morphology or shadows. Loss stays masked L1.
ABLATION_ISOLATION_CHANNELS: dict[str, tuple[str, ...]] = {
    "shadows": _SPECTRAL_INDEX + _SHADOW,
    "era5": _SPECTRAL_INDEX + _ERA5,
}

# ── v6 / training-v3 ablation (feature schema version 5, 22 channels) ──
# Families are slices of the v6 registry roster by name; the registry order is
# spectral → morphology → ERA5 → shadow.
_V6_NAMES: tuple[str, ...] = feature_schema(FEATURE_SCHEMA_VERSION_V6).names


def _v6_family(*names: str) -> tuple[str, ...]:
    unknown = [name for name in names if name not in _V6_NAMES]
    if unknown:
        raise RuntimeError(f"v6 family names not in the v6 roster: {unknown}")
    return tuple(names)


V6_SPECTRAL_INDEX = _v6_family(
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
    "ndvi",
    "ndwi_mcfeeters",
    "ndbi",
    "s2_broadband_albedo",
)
V6_MORPHOLOGY = _v6_family(
    "building_height_mean",
    "building_coverage_ratio",
    "vegetation_height_mean",
    "canopy_cover_fraction",
    "imperviousness",
    "svf",
)
V6_ERA5 = _v6_family("t2m_scene", "ssrd_scene", "wind_speed_10m_scene", "tp_0_72h")
V6_SHADOW = _v6_family("shadow_building", "shadow_vegetation")

V6_FAMILIES: dict[str, tuple[str, ...]] = {
    "spectral_index": V6_SPECTRAL_INDEX,
    "morphology": V6_MORPHOLOGY,
    "era5": V6_ERA5,
    "shadow": V6_SHADOW,
}


def _assert_v6_families_partition() -> None:
    flat = [name for family in V6_FAMILIES.values() for name in family]
    if len(flat) != len(set(flat)) or set(flat) != set(_V6_NAMES) or len(flat) != len(_V6_NAMES):
        raise RuntimeError("v6 channel families do not partition the v6 roster")


_assert_v6_families_partition()


def v6_registry_order(names: Sequence[str] | set[str] | frozenset[str]) -> tuple[str, ...]:
    """Return ``names`` restricted to and sorted by the v6 registry order."""
    wanted = set(names)
    unknown = sorted(wanted - set(_V6_NAMES))
    if unknown:
        raise ValueError(f"names not in the v6 roster: {unknown}")
    return tuple(name for name in _V6_NAMES if name in wanted)


@dataclass(frozen=True)
class V6Arm:
    """One v6 ablation arm: model inputs in v6 registry order plus its loss."""

    arm_id: str
    config_name: str
    kind: str  # ladder | loo | input
    channels: tuple[str, ...]
    loss: str


def _v6_arm(arm_id: str, config_name: str, kind: str, channels: set[str], loss: str) -> V6Arm:
    return V6Arm(arm_id, config_name, kind, v6_registry_order(channels), loss)


_V6_FULL = set(_V6_NAMES)
# LOO-ERA5 (L4 minus ERA5) equals L3 and therefore has no arm of its own.
V6_ABLATION_ARMS: dict[str, V6Arm] = {
    arm.arm_id: arm
    for arm in (
        _v6_arm("L1", "v6_l1_locked", "ladder", set(V6_SPECTRAL_INDEX), "masked_l1"),
        _v6_arm("L2", "v6_l2_locked", "ladder", {*V6_SPECTRAL_INDEX, *V6_MORPHOLOGY}, "masked_l1"),
        _v6_arm(
            "L3",
            "v6_l3_locked",
            "ladder",
            {*V6_SPECTRAL_INDEX, *V6_MORPHOLOGY, *V6_SHADOW},
            "masked_l1",
        ),
        _v6_arm("L4", "v6_l4_locked", "ladder", _V6_FULL, "masked_l1"),
        _v6_arm("L5", "v6_l5_locked", "ladder", _V6_FULL, "thermal_aware"),
        _v6_arm("LOO-M", "v6_loo_morph_locked", "loo", _V6_FULL - set(V6_MORPHOLOGY), "masked_l1"),
        _v6_arm("LOO-S", "v6_loo_shadow_locked", "loo", _V6_FULL - set(V6_SHADOW), "masked_l1"),
        _v6_arm(
            "NO-NDVI-NDWI",
            "v6_no_ndvi_ndwi_locked",
            "input",
            _V6_FULL - {"ndvi", "ndwi_mcfeeters"},
            "masked_l1",
        ),
        _v6_arm(
            "NO-RED-NIR-IDX",
            "v6_no_red_nir_index_locked",
            "input",
            _V6_FULL - {"B04", "B08", "ndvi", "ndwi_mcfeeters", "ndbi", "s2_broadband_albedo"},
            "masked_l1",
        ),
    )
}

# Cumulative v6 ladder rungs L1–L4 (the RF v6 baseline fits one forest each).
V6_LADDER_STAGE_CHANNELS: dict[int, tuple[str, ...]] = {
    rung: V6_ABLATION_ARMS[f"L{rung}"].channels for rung in (1, 2, 3, 4)
}

V6_ABLATION_LOCKED_CONFIG_NAMES: tuple[str, ...] = tuple(
    arm.config_name for arm in V6_ABLATION_ARMS.values()
)

ABLATION_LOCKED_CONFIG_NAMES: tuple[str, ...] = (
    "stage2_locked",
    "stage3_locked",
    "stage4_locked",
    "stage5_locked",
    "isolate_shadows_locked",
    "isolate_era5_locked",
)


@dataclass(frozen=True)
class ActiveChannelSelection:
    """Resolved model-input channels in the order the U-Net sees them."""

    names: tuple[str, ...]
    indices: tuple[int, ...]

    @property
    def n_active(self) -> int:
        return len(self.names)


def _index_map() -> dict[str, int]:
    return {name: i for i, name in enumerate(FEATURE_CHANNEL_NAMES)}


def resolve_active_channels(
    *,
    n_active_channels: int,
    active_channel_names: Sequence[str] | None = None,
) -> ActiveChannelSelection:
    """Resolve ``n_active_channels`` / optional names into V3 indices.

    When ``active_channel_names`` is empty or ``None``, the selection is the
    first ``n_active_channels`` of the fixed V3 order (``v3_first_c``).
    When names are given, they must be unique V3 names and
    ``n_active_channels`` must equal their count.
    """
    if isinstance(n_active_channels, bool) or not isinstance(n_active_channels, int):
        raise ValueError(
            f"n_active_channels must be an int in [1, {N_FEATURE_CHANNELS}], "
            f"got {n_active_channels!r}"
        )
    if not 1 <= n_active_channels <= N_FEATURE_CHANNELS:
        raise ValueError(
            f"n_active_channels must be in [1, {N_FEATURE_CHANNELS}], got {n_active_channels}"
        )

    if active_channel_names is None or len(active_channel_names) == 0:
        names = FEATURE_CHANNEL_NAMES[:n_active_channels]
        return ActiveChannelSelection(names=names, indices=tuple(range(n_active_channels)))

    names_list = [str(name) for name in active_channel_names]
    if len(names_list) != n_active_channels:
        raise ValueError(
            f"data.n_active_channels={n_active_channels} does not match "
            f"len(active_channel_names)={len(names_list)}"
        )
    if len(set(names_list)) != len(names_list):
        raise ValueError("active_channel_names must be unique")

    index_by_name = _index_map()
    unknown = [name for name in names_list if name not in index_by_name]
    if unknown:
        raise ValueError(f"unknown active_channel_names (not in V3 stack): {unknown}")

    indices = tuple(index_by_name[name] for name in names_list)
    return ActiveChannelSelection(names=tuple(names_list), indices=indices)


def selection_from_config(cfg: DictConfig) -> ActiveChannelSelection:
    """Resolve the active channel selection from a modeling Hydra config.

    Schema version 3 keeps the published v5 path (``v3_first_c`` / named
    subset of the 28-channel roster). Schema version 4 resolves against
    the v6 roster. A null ``n_active_channels`` on v6 means the full roster.
    """
    raw_names = OmegaConf.select(cfg, "data.active_channel_names")
    names: list[str] | None
    if raw_names is None:
        names = None
    elif isinstance(raw_names, (list, tuple, ListConfig)):
        names = [str(item) for item in raw_names]
    else:
        raise ValueError(
            f"data.active_channel_names must be a list of names or null, got {raw_names!r}"
        )
    version = schema_version_from_config(cfg)
    raw_n = OmegaConf.select(cfg, "data.n_active_channels")
    if version == FEATURE_SCHEMA_VERSION:
        return resolve_active_channels(
            n_active_channels=int(raw_n),
            active_channel_names=names,
        )
    raw_threshold = OmegaConf.select(cfg, "data.canopy_cover_threshold_m")
    if raw_threshold is None:
        raw_threshold = OmegaConf.select(cfg, "canopy_cover_threshold_m")
    threshold = None if raw_threshold is None else float(raw_threshold)
    n_active = None if raw_n is None else int(raw_n)
    return resolve_schema_selection(
        version,
        n_active_channels=n_active,
        active_channel_names=names,
        canopy_cover_threshold_m=threshold,
    )


def feature_order_for(selection: ActiveChannelSelection, schema_version: int | None = None) -> str:
    """Return the contract feature-order label for a resolved selection.

    With ``schema_version`` set to the v6 schema, a proper subset resolves to
    ``v6_named_subset`` and must keep the v6 registry order. Without it (or
    for the V3 schema) the V3 labels apply unchanged.
    """
    v6 = feature_schema(FEATURE_SCHEMA_VERSION_V6)
    if selection.names == v6.names and selection.indices == tuple(range(v6.n_channels)):
        return "v6_registry"
    if schema_version == FEATURE_SCHEMA_VERSION_V6:
        if list(selection.indices) != sorted(selection.indices) or selection.names != tuple(
            v6.names[i] for i in selection.indices
        ):
            raise ValueError(
                f"v6 channel subset is not in v6 registry order: {list(selection.names)}"
            )
        return "v6_named_subset"
    prefix = FEATURE_CHANNEL_NAMES[: selection.n_active]
    if selection.names == prefix and selection.indices == tuple(range(selection.n_active)):
        return "v3_first_c"
    return "v3_named_subset"


def schema_version_from_config(cfg: DictConfig) -> int:
    """Feature schema version declared on a modeling or feature config."""
    for key in ("data.feature_schema_version", "feature_schema_version"):
        raw = OmegaConf.select(cfg, key)
        if raw is not None:
            return int(raw)
    return FEATURE_SCHEMA_VERSION


def resolve_schema_selection(
    schema_version: int,
    *,
    n_active_channels: int | None,
    active_channel_names: Sequence[str] | None = None,
    canopy_cover_threshold_m: float | None = None,
) -> ActiveChannelSelection:
    """Resolve model inputs against one feature roster.

    A missing count selects the whole roster. Indices are positions in
    that roster, which is the COG order for that schema.
    """
    if schema_version == FEATURE_SCHEMA_VERSION:
        schema = feature_schema(schema_version)
    else:
        schema = feature_schema(schema_version, canopy_cover_threshold_m=canopy_cover_threshold_m)
    if n_active_channels is None:
        n_active_channels = schema.n_channels
    if isinstance(n_active_channels, bool) or not isinstance(n_active_channels, int):
        raise ValueError(
            f"n_active_channels must be an int in [1, {schema.n_channels}], "
            f"got {n_active_channels!r}"
        )
    if not 1 <= n_active_channels <= schema.n_channels:
        raise ValueError(
            f"n_active_channels must be in [1, {schema.n_channels}], got {n_active_channels}"
        )
    if active_channel_names is None or len(active_channel_names) == 0:
        names = schema.names[:n_active_channels]
        return ActiveChannelSelection(names=names, indices=tuple(range(n_active_channels)))
    names_list = [str(name) for name in active_channel_names]
    if len(names_list) != n_active_channels:
        raise ValueError(
            f"n_active_channels={n_active_channels} does not match "
            f"len(active_channel_names)={len(names_list)}"
        )
    if len(set(names_list)) != len(names_list):
        raise ValueError("active_channel_names must be unique")
    index_by_name = {name: index for index, name in enumerate(schema.names)}
    unknown = [name for name in names_list if name not in index_by_name]
    if unknown:
        raise ValueError(
            f"unknown active_channel_names (not in schema v{schema.version}): {unknown}"
        )
    indices = tuple(index_by_name[name] for name in names_list)
    return ActiveChannelSelection(names=tuple(names_list), indices=indices)


__all__ = [
    "ABLATION_ISOLATION_CHANNELS",
    "ABLATION_LOCKED_CONFIG_NAMES",
    "ABLATION_STAGE_CHANNELS",
    "ActiveChannelSelection",
    "V6Arm",
    "V6_ABLATION_ARMS",
    "V6_ABLATION_LOCKED_CONFIG_NAMES",
    "V6_ERA5",
    "V6_FAMILIES",
    "V6_LADDER_STAGE_CHANNELS",
    "V6_MORPHOLOGY",
    "V6_SHADOW",
    "V6_SPECTRAL_INDEX",
    "v6_registry_order",
    "N_FEATURE_CHANNELS",
    "feature_order_for",
    "resolve_active_channels",
    "resolve_schema_selection",
    "schema_version_from_config",
    "selection_from_config",
]
