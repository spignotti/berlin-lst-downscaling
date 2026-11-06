"""End-to-end provenance: deploy the pinned sha, then check the real Python helpers record it."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from _vm_remote_fixtures import (
    FOREIGN,
    HAVE_TOOLS,
    REPO,
    SHELL,
    SKIP_REASON,
    Sandbox,
    foreign_owner_supported,
    write_fake_stat,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason=SKIP_REASON)

_HELPERS_CODE = """
import json, sys
from berlin_lst_downscaling.data.io.run_logging import write_run_context
from berlin_lst_downscaling.modeling import prebuilt_cache, run_identity

uri = write_run_context(sys.argv[1], "random_forest", "run1")
print(json.dumps({
    "uri": uri,
    "local": run_identity._git_commit_local(),
    "commit": prebuilt_cache._git_commit(),
    "dirty": prebuilt_cache._git_dirty(),
}))
"""


@pytest.fixture
def deployed(tmp_path: Path) -> Sandbox:
    sb = Sandbox(tmp_path)
    res = sb.deploy()
    assert res.returncode == 0, res.stderr
    return sb


def _run_helpers(sb: Sandbox, env: dict[str, str]) -> dict[str, object]:
    full = {**env, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    out = sb.base / "out"
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _HELPERS_CODE, str(out)],
        cwd=sb.app,
        env=full,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_deploy_then_real_provenance_helpers_record_true_sha(deployed: Sandbox) -> None:
    sha = deployed.origin.sha
    prov = deployed.provenance()
    assert prov.returncode == 0, prov.stderr
    assert prov.stdout.strip() == sha

    result = _run_helpers(deployed, deployed.env())
    context = json.loads(Path(str(result["uri"])).read_text())
    assert context["git_commit"] == sha
    assert context["git_dirty"] is False
    assert result["local"] == sha
    assert result["commit"] == sha
    assert result["dirty"] is False


def _gate(sb: Sandbox, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    generated = sb.cmd("vm_remote_provenance_cmd")
    assert generated.returncode == 0, generated.stderr
    started = sb.base / "job-started"
    started.unlink(missing_ok=True)
    script = f'prov=$({SHELL} -c "$PROV_CMD") || exit 1\ntouch {started}\n'
    return subprocess.run(  # noqa: S603
        [SHELL, "-c", script],
        env={**env, "PROV_CMD": generated.stdout},
        capture_output=True,
        text=True,
        check=False,
    )


def test_foreign_owner_fails_before_job_start(deployed: Sandbox) -> None:
    started = deployed.base / "job-started"
    ok = _gate(deployed, deployed.env())
    assert ok.returncode == 0, ok.stderr
    assert started.exists()

    if foreign_owner_supported(deployed.base, deployed.app):
        env = deployed.env(FOREIGN)
        res = _gate(deployed, env)
        assert res.returncode != 0
        assert not started.exists()
        hazard = _run_helpers_foreign(deployed, env)
        assert hazard["local"] == "unknown"
        assert hazard["commit"] == "unknown"
        assert hazard["dirty"] is None
    else:
        pytest.skip("installed git ignores GIT_TEST_ASSUME_DIFFERENT_OWNER")


def _run_helpers_foreign(sb: Sandbox, env: dict[str, str]) -> dict[str, object]:
    code = (
        "import json\n"
        "from berlin_lst_downscaling.modeling import prebuilt_cache, run_identity\n"
        "print(json.dumps({'local': run_identity._git_commit_local(),"
        " 'commit': prebuilt_cache._git_commit(), 'dirty': prebuilt_cache._git_dirty()}))\n"
    )
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=sb.app,
        env={**env, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_fake_foreign_stat_fails_before_job_start(deployed: Sandbox) -> None:
    started = deployed.base / "job-started"
    write_fake_stat(deployed.bin, [("/berlin-lst-app", 4242), ("/.git", 4242)])
    res = _gate(deployed, deployed.env())
    assert res.returncode != 0
    assert not started.exists()


def _load_validator() -> ModuleType:
    path = REPO / "scripts" / "validators" / "validate_run_provenance.py"
    spec = importlib.util.spec_from_file_location("validate_run_provenance", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHA = "a" * 40
OTHER = "b" * 40

_BAD_REPORT_CASES = {
    "ok": ({"git_revision": SHA}, [{"git_commit": SHA, "git_dirty": False}], SHA, 0),
    "report_unknown": (
        {"git_revision": "unknown"},
        [{"git_commit": SHA, "git_dirty": False}],
        SHA,
        1,
    ),
    "report_missing": ({}, [{"git_commit": SHA, "git_dirty": False}], SHA, 1),
    "report_other": ({"git_revision": OTHER}, [{"git_commit": SHA, "git_dirty": False}], SHA, 1),
    "context_missing": ({"git_revision": SHA}, [], SHA, 1),
    "two_contexts": (
        {"git_revision": SHA},
        [{"git_commit": SHA, "git_dirty": False}, {"git_commit": SHA, "git_dirty": False}],
        SHA,
        1,
    ),
    "dirty_true": ({"git_revision": SHA}, [{"git_commit": SHA, "git_dirty": True}], SHA, 1),
    "dirty_missing": ({"git_revision": SHA}, [{"git_commit": SHA}], SHA, 1),
    "commit_mismatch": ({"git_revision": SHA}, [{"git_commit": OTHER, "git_dirty": False}], SHA, 1),
    "expected_not_hex": (
        {"git_revision": SHA},
        [{"git_commit": SHA, "git_dirty": False}],
        "abc123",
        1,
    ),
}


@pytest.mark.parametrize("case", list(_BAD_REPORT_CASES))
def test_post_run_validator(tmp_path: Path, case: str) -> None:
    report, contexts, expected, want = _BAD_REPORT_CASES[case]
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    for i, ctx in enumerate(contexts):
        (ctx_dir / f"run{i}.context.json").write_text(json.dumps(ctx))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    module = _load_validator()
    code = module.main(
        ["--report", str(report_path), "--expected-sha", expected, "--context-dir", str(ctx_dir)]
    )
    assert code == want
    assert os.path.exists(report_path)
