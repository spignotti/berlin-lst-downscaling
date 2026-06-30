"""Tests for the ARD mode translation helper."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from berlin_lst_downscaling.data.ard_modes import apply_mode


def test_mode_plan_sets_dry_run_true() -> None:
    """``plan`` maps to ``dry_run=True, smoke=False``."""
    cfg = OmegaConf.create({"mode": "plan", "dry_run": False, "smoke": True})
    apply_mode(cfg)
    assert cfg.dry_run is True
    assert cfg.smoke is False


def test_mode_smoke_sets_smoke_true() -> None:
    """``smoke`` maps to ``dry_run=False, smoke=True``."""
    cfg = OmegaConf.create({"mode": "smoke", "dry_run": True, "smoke": False})
    apply_mode(cfg)
    assert cfg.dry_run is False
    assert cfg.smoke is True


def test_mode_all_clears_both() -> None:
    """``all`` maps to ``dry_run=False, smoke=False``."""
    cfg = OmegaConf.create({"mode": "all", "dry_run": True, "smoke": True})
    apply_mode(cfg)
    assert cfg.dry_run is False
    assert cfg.smoke is False


def test_unknown_mode_raises() -> None:
    """An unknown mode string raises a clear ValueError."""
    cfg = OmegaConf.create({"mode": "garbage"})
    with pytest.raises(ValueError, match="Unknown mode"):
        apply_mode(cfg)
