"""Launcher wiring for the v6 ablation modes (offline; nothing is submitted)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "operators" / "launch_vertex_modeling.py"
_IMAGE = "example.invalid/repo/image@sha256:" + "a" * 64
_MODES = {
    "v6-l1": "v6_l1_locked",
    "v6-l2": "v6_l2_locked",
    "v6-l3": "v6_l3_locked",
    "v6-l4": "v6_l4_locked",
    "v6-l5": "v6_l5_locked",
    "v6-loo-morph": "v6_loo_morph_locked",
    "v6-loo-shadow": "v6_loo_shadow_locked",
    "v6-no-ndvi-ndwi": "v6_no_ndvi_ndwi_locked",
    "v6-no-red-nir-index": "v6_no_red_nir_index_locked",
}


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The Vertex SDK lives in the optional ``operators`` dependency group. Stub it
    # when absent: these tests never create a client or submit anything.
    if importlib.util.find_spec("google.cloud.aiplatform_v1") is None:
        stub = MagicMock()
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1", stub)
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1.types", stub.types)
    spec = importlib.util.spec_from_file_location("launch_vertex_modeling_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _no_client(region: str) -> None:
        raise AssertionError("the Vertex client must not be created in these tests")

    monkeypatch.setattr(module, "_evidence_exists", lambda uri: False)
    monkeypatch.setattr(module, "_client", _no_client)
    return module


def _argv(mode: str, *extra: str, rate: bool = True) -> list[str]:
    argv = [
        "launch_vertex_modeling.py",
        "--mode",
        mode,
        "--image-uri",
        _IMAGE,
        "--source-sha",
        "0" * 40,
        "--run-label",
        "unit",
        "--service-account",
        "<service-account>",
        "--infisical-identity",
        "<identity>",
        "--infisical-project",
        "<project>",
        "--preflight",
    ]
    if rate:
        argv += ["--hourly-rate-usd", "1.0"]
    return [*argv, *extra]


def test_modes_map_to_configs(launcher: ModuleType) -> None:
    for mode, config in _MODES.items():
        assert launcher.MODE_CONFIG_NAME[mode] == config
        assert mode in launcher._V6_ABLATION_MODES
        assert mode in launcher._FULL_TEMPORAL_MODES
        assert mode in launcher._ABLATION_MODES
        assert launcher.MODE_EVIDENCE_PROFILE[mode] == "full"


@pytest.mark.parametrize("mode", list(_MODES))
def test_preflight_accepts_every_v6_mode(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(mode))
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert f"{mode} / {_MODES[mode]}" in out
    assert "seed/epochs    : 0 / 20" in out
    assert "unit-seed0-ep20" in out


def test_seed_and_epochs_reach_label_and_overrides(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", "--seed", "2", "--max-epochs", "40"))
    assert launcher.main() == 0
    out = capsys.readouterr().out
    assert "seed=2 trainer.max_epochs=40" in out
    assert "unit-seed2-ep40" in out
    assert "output_root=data/runs/v6-l4/unit-seed2-ep40/seed-2" in out


def test_env_carries_seed_epochs_and_seeded_output_root(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_submit(client: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "projects/p/locations/r/customJobs/1"

    monkeypatch.setattr(launcher, "_client", lambda region: object())
    monkeypatch.setattr(launcher, "_submit", _fake_submit)
    monkeypatch.setattr(
        launcher,
        "_poll_until_terminal",
        lambda *a, **k: (launcher.JobState.JOB_STATE_SUCCEEDED, False),
    )
    argv = _argv("v6-l5", "--seed", "2", "--max-epochs", "40")
    argv.remove("--preflight")
    monkeypatch.setattr(sys, "argv", argv)
    assert launcher.main() == 0
    env = dict(captured["env"])  # type: ignore[arg-type]
    assert env["VERTEX_SEED"] == "2"
    assert env["VERTEX_MAX_EPOCHS"] == "40"
    assert env["VERTEX_RUN_LABEL"] == "unit-seed2-ep40"
    assert env["VERTEX_OUTPUT_ROOT"] == "data/runs/v6-l5/unit-seed2-ep40/seed-2"
    assert env["VERTEX_EVIDENCE_URI"].endswith("/unit-seed2-ep40/evidence.json")
    assert env["VERTEX_CONFIG_NAME"] == "v6_l5_locked"
    assert str(captured["display_name"]) == "v6-l5-unit-seed2-ep40"


@pytest.mark.parametrize("flag", [["--seed", "1"], ["--max-epochs", "5"]])
def test_non_v6_mode_rejects_seed_and_epochs(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("stage4", *flag))
    with pytest.raises(SystemExit, match="only accepted by the v6 modes"):
        launcher.main()


def test_negative_seed_is_refused_by_the_lock(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", "--seed", "-1"))
    with pytest.raises(SystemExit, match="off-contract"):
        launcher.main()


def test_missing_rate_is_refused(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", rate=False))
    with pytest.raises(SystemExit, match="hourly-rate-usd"):
        launcher.main()


def _out(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys, *extra: str) -> str:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", *extra))
    assert launcher.main() == 0
    return capsys.readouterr().out


def test_skip_test_composes_and_passes_guards(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(launcher, monkeypatch, capsys, "--max-epochs", "2", "--skip-test")
    assert "evaluate_test=false" in out
    assert "test pass      : skipped" in out
    assert "run label      : unit-seed0-ep2-notest" in out
    assert "data/runs/v6-l4/unit-seed0-ep2-notest/seed-0" in out


def test_run_label_has_no_notest_suffix_without_skip_test(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(launcher, monkeypatch, capsys, "--max-epochs", "2")
    assert "run label      : unit-seed0-ep2\n" in out
    assert "notest" not in out


def test_early_stopping_is_off_by_default(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(launcher, monkeypatch, capsys, "--max-epochs", "40")
    assert "early_stopping" not in out.split("hydra overrides:")[1]
    assert "evaluate_test" not in out.split("hydra overrides:")[1]
    assert "early stopping : off" in out
    assert "test pass      : scored" in out


def _submit_env(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict:
    captured: dict[str, object] = {}

    def _fake_submit(client: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "projects/p/locations/r/customJobs/1"

    monkeypatch.setattr(launcher, "_client", lambda region: object())
    monkeypatch.setattr(launcher, "_submit", _fake_submit)
    monkeypatch.setattr(
        launcher,
        "_poll_until_terminal",
        lambda *a, **k: (launcher.JobState.JOB_STATE_SUCCEEDED, False),
    )
    argv = _argv("v6-l4", *extra)
    argv.remove("--preflight")
    monkeypatch.setattr(sys, "argv", argv)
    assert launcher.main() == 0
    return dict(captured["env"])  # type: ignore[arg-type]


def test_early_stopping_reaches_overrides_and_env(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = ("--early-stopping-patience", "5", "--early-stopping-min-delta", "0.01")
    out = _out(launcher, monkeypatch, capsys, *flags)
    assert "early_stopping.enabled=true early_stopping.patience=5" in out
    assert "early_stopping.min_delta=0.01" in out
    env = _submit_env(launcher, monkeypatch, *flags, "--skip-test")
    assert env["VERTEX_EARLY_STOPPING_PATIENCE"] == "5"
    assert env["VERTEX_EARLY_STOPPING_MIN_DELTA"] == "0.01"
    assert env["VERTEX_SKIP_TEST"] == "1"


def test_default_env_has_epoch_metrics_uri_and_no_es_or_skip(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _submit_env(launcher, monkeypatch)
    assert env["VERTEX_EPOCH_METRICS_URI"].endswith("/unit-seed0-ep20/epoch_metrics.json")
    assert not {"VERTEX_EARLY_STOPPING_PATIENCE", "VERTEX_EARLY_STOPPING_MIN_DELTA"} & set(env)
    assert "VERTEX_SKIP_TEST" not in env


@pytest.mark.parametrize(
    "flag",
    [
        ["--early-stopping-patience", "5"],
        ["--early-stopping-min-delta", "0.1"],
        ["--skip-test"],
    ],
)
def test_non_v6_mode_rejects_new_flags(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("stage4", *flag))
    with pytest.raises(SystemExit, match="only accepted by the v6 modes"):
        launcher.main()


@pytest.mark.parametrize(
    ("flag", "match"),
    [
        (["--early-stopping-min-delta", "0.1"], "requires --early-stopping-patience"),
        (["--early-stopping-patience", "0"], "must be >= 1"),
        (
            ["--early-stopping-patience", "3", "--early-stopping-min-delta", "-1"],
            "finite and >= 0",
        ),
    ],
)
def test_invalid_early_stopping_flags_rejected(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: list[str], match: str
) -> None:
    monkeypatch.setattr(sys, "argv", _argv("v6-l4", *flag))
    with pytest.raises(SystemExit, match=match):
        launcher.main()
