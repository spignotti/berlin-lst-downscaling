"""validate_ablation_configs.py: one PASS/FAIL line per v6 arm and cache mode.

Kept out of test_v6_ablation.py: its module-scoped Hydra context would clash
with the validator's own ``initialize_config_dir``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from berlin_lst_downscaling.modeling.channels import V6_ABLATION_ARMS

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cache_mode", ["both", "off", "on"])
def test_validate_ablation_configs_reports_each_v6_arm(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, cache_mode: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "validate_ablation_configs",
        _ROOT / "scripts" / "validators" / "validate_ablation_configs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The real checks run in the validate-ablation-configs nox session; this
    # test pins the per-arm report and exit code only.
    seen: list[tuple[str, str]] = []
    failing = next(iter(V6_ABLATION_ARMS))

    def fake_arm(arm_id: str, mode: str) -> list[str]:
        seen.append((arm_id, mode))
        return ["boom"] if arm_id == failing and mode == "off" else []

    monkeypatch.setattr(module, "_check_stage", lambda name: [])
    monkeypatch.setattr(module, "_check_v6_matrix", lambda: ([], "matrix"))
    monkeypatch.setattr(module, "_check_v6_arm", fake_arm)
    code = module.main(["--cache-mode", cache_mode])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert seen == [(arm_id, cache_mode) for arm_id in V6_ABLATION_ARMS]
    assert sum(line.startswith("v6 arm ") for line in lines) == len(V6_ABLATION_ARMS) == 9
    for arm_id, arm in V6_ABLATION_ARMS.items():
        status = "FAIL" if arm_id == failing and cache_mode == "off" else "PASS"
        expected = f"v6 arm {arm_id} ({arm.config_name}) cache_mode={cache_mode}: {status}"
        assert lines.count(expected) == 1
    if cache_mode == "off":
        assert code == 1
        assert "ablation config smoke NO-GO" in captured.err
    else:
        assert code == 0
        assert lines[-1].endswith(f"cache_mode={cache_mode}")
