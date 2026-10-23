"""TESSERA-arm evaluation guards: val 2024 / test 2025 pass, 2026 refused."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from berlin_lst_downscaling.data.embeddings.guards import (
    assert_tessera_eval_year,
    assert_tessera_not_2026,
)
from berlin_lst_downscaling.data.embeddings.tessera import TesseraYearError
from berlin_lst_downscaling.modeling.guards import contract_invariants

_REPO = Path(__file__).resolve().parents[1]


def test_eval_years_map_to_existing_splits() -> None:
    assert assert_tessera_eval_year(2024) == "validation"
    assert assert_tessera_eval_year(2025) == "test"
    assert assert_tessera_eval_year(2020) == "train"


def test_2026_excluded_for_tessera_arm() -> None:
    with pytest.raises(TesseraYearError, match="excludes 2026"):
        assert_tessera_eval_year(2026)
    with pytest.raises(TesseraYearError, match="excludes 2026"):
        assert_tessera_not_2026(2026)
    # Training years pass the 2026 guard silently.
    assert_tessera_not_2026(2023)


def test_tessera_arm_config_reuses_eval_path() -> None:
    with initialize_config_dir(config_dir=str(_REPO / "configs" / "modeling"), version_base=None):
        cfg = compose(config_name="tessera_arm")
    # Same frozen split/contract surface as the main arm: no new split.
    assert list(cfg.data.splits) == ["train", "validation"]
    contract_invariants(cfg)
    assert cfg.contract.feature_order == "v6_registry"
    assert cfg.data.tessera_enabled is True
    assert cfg.model.tessera_adapter_width == 8
