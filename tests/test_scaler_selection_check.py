"""Runtime scaler/selection check fails loudly on any roster disagreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from odc.geo.geobox import GeoBox
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V6,
    feature_schema,
)
from berlin_lst_downscaling.data.training.contracts import (
    expected_training_policy_hash,
    scaler_transform_for,
)
from berlin_lst_downscaling.data.training.scaler import fit_scaler
from berlin_lst_downscaling.modeling.channels import (
    V6_ABLATION_ARMS,
    ActiveChannelSelection,
    assert_scaler_matches_selection,
    resolve_schema_selection,
)
from berlin_lst_downscaling.modeling.patches import (
    ScalerChannel,
    ScalerSpec,
    assert_scaler_matches_config,
    assert_scaler_spec_matches,
    load_scaler,
)

_V = FEATURE_SCHEMA_VERSION_V6
_NAMES = feature_schema(_V).names


def _scaler(names: tuple[str, ...] = _NAMES) -> ScalerSpec:
    channels = tuple(ScalerChannel(i + 1, n, "identity", None, None) for i, n in enumerate(names))
    return ScalerSpec(channel_order=names, channels=channels, policy_hash="x")


def _full() -> ActiveChannelSelection:
    return resolve_schema_selection(_V, n_active_channels=None)


def _l1() -> ActiveChannelSelection:
    names = V6_ABLATION_ARMS["L1"].channels
    return resolve_schema_selection(_V, n_active_channels=len(names), active_channel_names=names)


def test_full_roster_passes() -> None:
    assert_scaler_spec_matches(_scaler(), schema_version=_V, selection=_full())


def test_subset_passes() -> None:
    assert_scaler_spec_matches(_scaler(), schema_version=_V, selection=_l1())


def test_reordered_scaler_fails() -> None:
    names = (_NAMES[1], _NAMES[0], *_NAMES[2:])
    with pytest.raises(ValueError, match="first_difference_position=0"):
        assert_scaler_spec_matches(_scaler(names), schema_version=_V, selection=_full())


def test_missing_channel_fails() -> None:
    with pytest.raises(ValueError, match=f"missing=\\['{_NAMES[-1]}'\\]"):
        assert_scaler_spec_matches(_scaler(_NAMES[:-1]), schema_version=_V, selection=_full())


def test_renamed_channel_fails() -> None:
    names = (*_NAMES[:-1], "renamed")
    with pytest.raises(ValueError, match="extra=\\['renamed'\\]"):
        assert_scaler_spec_matches(_scaler(names), schema_version=_V, selection=_full())


def test_selection_index_mapping_to_other_name_fails() -> None:
    bad = ActiveChannelSelection(names=(_NAMES[0], _NAMES[1]), indices=(0, 2))
    with pytest.raises(ValueError, match="do not map to the selected names"):
        assert_scaler_spec_matches(_scaler(), schema_version=_V, selection=bad)


def test_selection_index_out_of_range_fails() -> None:
    bad = ActiveChannelSelection(names=(_NAMES[0],), indices=(len(_NAMES),))
    with pytest.raises(ValueError, match="out of range"):
        assert_scaler_spec_matches(_scaler(), schema_version=_V, selection=bad)


def test_selection_name_not_in_scaler_fails() -> None:
    bad = ActiveChannelSelection(names=("nope",), indices=(0,))
    with pytest.raises(ValueError, match="not in the scaler"):
        assert_scaler_matches_selection(
            _NAMES, schema_names=_NAMES, schema_version=_V, selection=bad
        )


def test_entry_and_channel_order_disagreement_fails() -> None:
    scaler = _scaler()
    swapped = ScalerSpec(
        channel_order=scaler.channel_order,
        channels=(scaler.channels[1], scaler.channels[0], *scaler.channels[2:]),
        policy_hash="x",
    )
    with pytest.raises(ValueError, match="disagree with channel_order"):
        assert_scaler_spec_matches(swapped, schema_version=_V, selection=_full())
    wrong_index = ScalerSpec(
        channel_order=scaler.channel_order,
        channels=(ScalerChannel(5, _NAMES[0], "identity", None, None), *scaler.channels[1:]),
        policy_hash="x",
    )
    with pytest.raises(ValueError, match="disagree with channel_order"):
        assert_scaler_spec_matches(wrong_index, schema_version=_V, selection=_full())


# ── published scaler.json shape (1-based ``channel_index``) ──────────


def _published_payload(names: tuple[str, ...], schema_version: int) -> dict:
    """A scaler.json with the published keys and 1-based raster band numbers."""
    schema = feature_schema(schema_version)
    payload: dict = {
        "channel_order": list(names),
        "channels": [
            {
                "channel_index": i + 1,
                "channel_name": name,
                "transform": scaler_transform_for(name, schema),
                "count": 100,
                "mean": None if scaler_transform_for(name, schema) == "identity" else 0.5,
                "std": None if scaler_transform_for(name, schema) == "identity" else 2.0,
            }
            for i, name in enumerate(names)
        ],
    }
    if schema_version == FEATURE_SCHEMA_VERSION_V6:
        payload["v3_config_hash"] = "278427dea6ea41f0"
    payload["policy_hash"] = expected_training_policy_hash(payload)
    return payload


def _load(tmp_path: Path, payload: dict) -> ScalerSpec:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scaler.json").write_text(json.dumps(payload), encoding="utf-8")
    return load_scaler(str(tmp_path))


def test_published_v3_scaler_passes(tmp_path: Path) -> None:
    payload = _published_payload(_NAMES, _V)
    assert len(payload["channels"]) == 22
    assert (payload["channels"][0]["channel_index"], payload["channels"][0]["channel_name"]) == (
        1,
        "B02",
    )
    last = payload["channels"][-1]
    assert (last["channel_index"], last["channel_name"]) == (22, "shadow_vegetation")
    spec = _load(tmp_path, payload)
    assert_scaler_spec_matches(spec, schema_version=_V, selection=_full())
    assert_scaler_spec_matches(spec, schema_version=_V, selection=_l1())


def test_published_v1_scaler_passes(tmp_path: Path) -> None:
    payload = _published_payload(FEATURE_CHANNEL_NAMES, FEATURE_SCHEMA_VERSION)
    assert len(payload["channels"]) == 28
    spec = _load(tmp_path, payload)
    selection = resolve_schema_selection(FEATURE_SCHEMA_VERSION, n_active_channels=None)
    assert_scaler_spec_matches(spec, schema_version=FEATURE_SCHEMA_VERSION, selection=selection)


@pytest.mark.parametrize(
    ("schema_version", "names"),
    [(_V, _NAMES), (FEATURE_SCHEMA_VERSION, FEATURE_CHANNEL_NAMES)],
)
def test_publisher_scaler_round_trips_through_reader(
    tmp_path: Path, schema_version: int, names: tuple[str, ...]
) -> None:
    """``fit_scaler``'s own channel entries pass the modeling reader's check."""
    grid = GeoBox.from_bbox(
        (369190.0, 5838310.0, 369290.0, 5838410.0), crs="EPSG:25833", resolution=10
    )
    schema = None if schema_version == FEATURE_SCHEMA_VERSION else feature_schema(schema_version)
    payload = fit_scaler(
        [], features_root="unused", output_root="unused", grid_10m=grid, schema=schema
    )
    assert tuple(payload["channel_order"]) == names
    if schema_version == FEATURE_SCHEMA_VERSION_V6:
        payload["v3_config_hash"] = "278427dea6ea41f0"
    payload["policy_hash"] = expected_training_policy_hash(payload)
    spec = _load(tmp_path, payload)
    selection = resolve_schema_selection(schema_version, n_active_channels=None)
    assert_scaler_spec_matches(spec, schema_version=schema_version, selection=selection)


def test_zero_based_channel_index_fails(tmp_path: Path) -> None:
    payload = _published_payload(_NAMES, _V)
    for i, channel in enumerate(payload["channels"]):
        channel["channel_index"] = i
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="disagree with channel_order"):
        assert_scaler_spec_matches(spec, schema_version=_V, selection=_full())


def test_shuffled_channel_index_fails(tmp_path: Path) -> None:
    payload = _published_payload(_NAMES, _V)
    first, second = payload["channels"][0], payload["channels"][1]
    first["channel_index"], second["channel_index"] = 2, 1
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="disagree with channel_order"):
        assert_scaler_spec_matches(spec, schema_version=_V, selection=_full())


def test_published_entry_name_order_mismatch_fails(tmp_path: Path) -> None:
    payload = _published_payload(_NAMES, _V)
    first, second = payload["channels"][0], payload["channels"][1]
    first["channel_name"], second["channel_name"] = second["channel_name"], first["channel_name"]
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="disagree with channel_order"):
        assert_scaler_spec_matches(spec, schema_version=_V, selection=_full())


def _cfg(names: tuple[str, ...] | None) -> object:
    return OmegaConf.create(
        {
            "data": {
                "feature_schema_version": _V,
                "n_active_channels": None if names is None else len(names),
                "active_channel_names": None if names is None else list(names),
            }
        }
    )


def test_config_check_passes_and_fails() -> None:
    l1 = V6_ABLATION_ARMS["L1"].channels
    assert_scaler_matches_config(_cfg(l1), _scaler())
    assert_scaler_matches_config(_cfg(None), _scaler())
    with pytest.raises(ValueError, match="scaler channel names differ"):
        assert_scaler_matches_config(_cfg(l1), _scaler(tuple(reversed(_NAMES))))
