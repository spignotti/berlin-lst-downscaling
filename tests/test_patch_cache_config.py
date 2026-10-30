"""``data.patch_cache`` selection: guard validation and run-side resolution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.modeling.guards import (
    _check_patch_cache,
    assert_ablation_lock,
    contract_invariants,
    guard_modeling_config,
)
from berlin_lst_downscaling.modeling.run import _resolve_patch_cache

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_NAME = "v6_l4_locked"


@pytest.fixture
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):

        def _compose(overrides: list[str] | None = None) -> DictConfig:
            return compose(config_name=_NAME, overrides=overrides or [])

        yield _compose
    GlobalHydra.instance().clear()


def _with(block: dict[str, object]) -> DictConfig:
    return OmegaConf.create({"data": {"patch_cache": block}})


def test_absent_block_defaults_to_job_local(compose_modeling: Callable[..., DictConfig]) -> None:
    cfg = compose_modeling()
    guard_modeling_config(cfg, _NAME)
    resolved = _resolve_patch_cache(cfg)
    assert resolved["source"] == "job_local"
    assert resolved["allow_job_local_fallback"] is False
    assert resolved["uri_root"] == "gs://berlin-lst-training-data/patch-cache"
    assert (resolved["download_workers"], resolved["download_retries"]) == (8, 3)


@pytest.mark.parametrize(
    "overrides",
    [
        ["+data.patch_cache.source=prebuilt"],
        ["+data.patch_cache.source=prebuilt", "+data.patch_cache.allow_job_local_fallback=true"],
        ["+data.patch_cache.source=job_local"],
    ],
)
def test_valid_selection_passes_the_locked_profile_guards(
    compose_modeling: Callable[..., DictConfig], overrides: list[str]
) -> None:
    cfg = compose_modeling(overrides)
    guard_modeling_config(cfg, _NAME)
    contract_invariants(cfg)
    OmegaConf.set_struct(cfg, False)
    assert_ablation_lock(cfg)
    assert _resolve_patch_cache(cfg)["source"] == cfg.data.patch_cache.source


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"source": "remote"}, "source"),
        ({"source": None}, "source"),
        ({"source": "prebuilt", "allow_job_local_fallback": "yes"}, "allow_job_local_fallback"),
        ({"source": "prebuilt", "allow_job_local_fallback": 1}, "allow_job_local_fallback"),
        ({"source": "prebuilt", "uri_root": "relative/path"}, "uri_root"),
        ({"source": "prebuilt", "uri_root": 3}, "uri_root"),
        ({"source": "prebuilt", "download_workers": 0}, "download_workers"),
        ({"source": "prebuilt", "download_retries": True}, "download_retries"),
        ({"source": "prebuilt", "download_retries": 1.5}, "download_retries"),
        ({"source": "prebuilt", "unknown": 1}, "unknown"),
    ],
)
def test_invalid_selection_is_reported_naming_the_key(block: dict[str, object], match: str) -> None:
    cfg = _with(block)
    with pytest.raises(ValueError, match=match):
        guard_modeling_config(cfg, None)
    if "source" in block and block["source"] is not None:
        with pytest.raises(ValueError, match=match):
            _resolve_patch_cache(cfg)


@pytest.mark.parametrize(
    "block",
    [
        {"source": "prebuilt"},
        {"source": "prebuilt", "uri_root": "gs://bucket/patch-cache", "download_workers": 2},
        {"source": "job_local", "uri_root": "/mnt/cache", "download_retries": 5},
    ],
)
def test_valid_blocks_resolve_over_defaults(block: dict[str, object]) -> None:
    cfg = _with(block)
    _check_patch_cache(cfg)
    resolved = _resolve_patch_cache(cfg)
    for key, value in block.items():
        assert resolved[key] == value
