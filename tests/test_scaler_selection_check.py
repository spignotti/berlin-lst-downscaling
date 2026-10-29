"""Runtime scaler/selection check fails loudly on any roster disagreement."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6, feature_schema
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
)

_V = FEATURE_SCHEMA_VERSION_V6
_NAMES = feature_schema(_V).names


def _scaler(names: tuple[str, ...] = _NAMES) -> ScalerSpec:
    channels = tuple(ScalerChannel(i, n, "identity", None, None) for i, n in enumerate(names))
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
