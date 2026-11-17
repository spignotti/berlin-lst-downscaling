"""TESSERA arm under the spatial holdout: config, guard and launcher mode (offline)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from test_launch_vertex_v6 import _argv, launcher  # noqa: F401  (``launcher`` is a fixture)

from berlin_lst_downscaling.modeling.guards import guard_modeling_config
from berlin_lst_downscaling.modeling.spatial_holdout import TESSERA_SPATIAL_HOLDOUT_CONFIG

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_MODE = "tessera-spatial-holdout"
_SEEDED = ("--seed", "0", "--max-epochs", "40", "--lr-schedule", "cosine")
_OK = ["evaluate_test=false", "+data.patch_cache.source=prebuilt"]


def _compose(name: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def test_config_inherits_tessera_arm_and_adds_only_the_option_and_the_output_root() -> None:
    base = _compose("tessera_arm", [])
    holdout = _compose(TESSERA_SPATIAL_HOLDOUT_CONFIG, [])
    assert dict(holdout.spatial_holdout) == {"enabled": True, "assignment": "v1"}
    assert holdout.output_root != base.output_root and "holdout" in holdout.output_root
    rest = {k: v for k, v in holdout.items() if k not in ("spatial_holdout", "output_root")}
    assert rest == {k: v for k, v in base.items() if k != "output_root"}


def test_guard_admits_the_tessera_holdout_with_the_prebuilt_cache() -> None:
    guard_modeling_config(
        _compose(TESSERA_SPATIAL_HOLDOUT_CONFIG, _OK), TESSERA_SPATIAL_HOLDOUT_CONFIG
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            ["evaluate_test=true", "+data.patch_cache.source=prebuilt"],
            "evaluate_test must be false",
        ),
        (["evaluate_test=false"], "patch_cache.source must be prebuilt"),
        (
            [*_OK, "+data.patch_cache.allow_job_local_fallback=true"],
            "allow_job_local_fallback must be false",
        ),
        ([*_OK, "spatial_holdout.assignment=cv5-fold0"], "only 'v1' is admitted"),
        ([*_OK, "spatial_holdout.enabled=false"], "enabled must be true"),
    ],
)
def test_guard_refuses_what_the_tessera_holdout_does_not_admit(
    overrides: list[str], match: str
) -> None:
    cfg = _compose(TESSERA_SPATIAL_HOLDOUT_CONFIG, overrides)
    with pytest.raises(ValueError, match=match):
        guard_modeling_config(cfg, TESSERA_SPATIAL_HOLDOUT_CONFIG)


def test_the_option_is_refused_on_the_ordinary_tessera_config() -> None:
    cfg = _compose(TESSERA_SPATIAL_HOLDOUT_CONFIG, _OK)
    with pytest.raises(ValueError, match="only admitted on"):
        guard_modeling_config(cfg, "tessera_arm")
    guard_modeling_config(_compose("tessera_arm", ["evaluate_test=false"]), "tessera_arm")


def test_preflight_has_the_tessera_seed_bounds(
    launcher: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys, "argv", _argv(_MODE, *_SEEDED, "--skip-test", "--patch-cache", "prebuilt")
    )
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert f"{_MODE} / {TESSERA_SPATIAL_HOLDOUT_CONFIG}" in out
    assert "timeout        : 20400s  max-wait: 600s" in out and "~$5.83" in out
    assert "patch cache    : prebuilt" in out and "+data.patch_cache.source=prebuilt" in out
    assert "test pass      : skipped" in out and "evaluate_test=false" in out
    assert "/unit-seed0-ep40-cos-notest/eval_validation_patches.parquet" in out
    assert (launcher.TESSERA_TIMEOUT_SECONDS, launcher.TESSERA_MAX_EXPOSURE_USD) == (20400, 6.25)
    assert launcher.TESSERA_MAX_HOURLY_RATE_USD == 1.0
    assert _MODE not in launcher.SPATIAL_HOLDOUT_MODES  # the nine arms' set is unchanged
    assert launcher.MODE_EVIDENCE_PROFILE[_MODE] == "full"


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ((), "requires --skip-test"),
        (("--skip-test",), "pass --patch-cache prebuilt"),
        (
            ("--skip-test", "--patch-cache", "prebuilt", "--allow-job-local-fallback"),
            "without --allow-job-local-fallback",
        ),
        (
            ("--skip-test", "--patch-cache", "prebuilt", "--assignment", "cv5-fold0"),
            r"admitted for the arms \['L1', 'L4'\] only, not TESSERA",
        ),
        (("--skip-test", "--patch-cache", "prebuilt", "--timeout-seconds", "20401"), "timeout"),
        (
            ("--skip-test", "--patch-cache", "prebuilt", "--max-exposure-usd", "6.26"),
            "per-job exposure",
        ),
    ],
)
def test_launcher_refusals(
    launcher: ModuleType,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    match: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(_MODE, *_SEEDED, *extra))
    with pytest.raises(SystemExit, match=match):
        launcher.main()
