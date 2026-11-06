"""Offline tests for scripts/runners/vm/val-baselines-vm.sh (fake gcloud only)."""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "runners" / "vm" / "val-baselines-vm.sh"
BASH = shutil.which("bash")
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")

pytestmark = pytest.mark.skipif(
    BASH is None or PYTHON3 is None, reason="bash and python3 (in /usr/bin:/bin) are required"
)

FAKE_GCLOUD = f"""#!{PYTHON3}
import json, os, re, sys
tmp = os.environ["FAKE_TMP"]
argv = sys.argv[1:]
with open(os.path.join(tmp, "gcloud.log"), "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
fmt = next((a.split("=", 1)[1] for a in argv if a.startswith("--format=")), "")
expr = re.match(r"value\\((.*)\\)$", fmt).group(1)
table = {{
    "id": "6236232769523665407",
    "status": os.environ["FAKE_STATUS"],
    "disks[0].source.basename()": "berlin-lst-vm",
    "disks[0].autoDelete": "False",
    "serviceAccounts[0].email": "berlin-lst-vertex@berlin-lst-training.iam.gserviceaccount.com",
    "deletionProtection": "True",
    "machineType.basename()": "n2-highmem-2",
}}
print(table[expr])
"""


def _run(tmp: Path, *args: str, status: str = "TERMINATED") -> subprocess.CompletedProcess[str]:
    bin_dir = tmp / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        g = bin_dir / "gcloud"
        g.write_text(FAKE_GCLOUD)
        g.chmod(0o755)
    (tmp / "home").mkdir(exist_ok=True)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp / "home"),
        "FAKE_TMP": str(tmp),
        "FAKE_STATUS": status,
        "XDG_RUNTIME_DIR": str(tmp),
    }
    return subprocess.run(  # noqa: S603
        [BASH or "", str(SCRIPT), *args], env=env, capture_output=True, text=True, check=False
    )


def _verbs(tmp: Path) -> list[str]:
    log = tmp / "gcloud.log"
    return [json.loads(x)[2] for x in log.read_text().splitlines()] if log.exists() else []


def test_plan_is_default_and_offline(tmp_path: Path) -> None:
    r = _run(tmp_path, "plan", "main", "0.168788")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.index("rf-smoke") < out.index("rf-full") < out.index("naive")
    assert "0.168788" in out
    assert "re-verify" in out
    assert not (tmp_path / "gcloud.log").exists()


def test_step_dry_run_prints_command_and_exposure(tmp_path: Path) -> None:
    r = _run(tmp_path, "rf-full", "main", "0.168788")
    assert r.returncode == 0, r.stderr
    assert "run-random-forest-vm.sh v6-val-full main 0.168788" in r.stdout
    assert "1.35 USD" in r.stdout
    assert "not a hard cap" in r.stdout
    assert "[dry-run]" in r.stdout
    assert not (tmp_path / "gcloud.log").exists()


@pytest.mark.parametrize(
    "step",
    ["v6-full", "v6-smoke", "full", "smoke", "baseline_v6_full", "baseline_full", "test"],
)
def test_rejects_test_opening_steps(tmp_path: Path, step: str) -> None:
    r = _run(tmp_path, step, "main", "0.168788")
    assert r.returncode == 1
    assert "opens test 2025" in r.stderr
    assert not (tmp_path / "gcloud.log").exists()


@pytest.mark.parametrize("rate", ["", "0", "abc", "2"])
def test_requires_valid_rate(tmp_path: Path, rate: str) -> None:
    assert _run(tmp_path, "rf-smoke", "main", rate).returncode != 0


def test_execute_refuses_running_vm(tmp_path: Path) -> None:
    r = _run(tmp_path, "rf-smoke", "main", "0.168788", "--execute", status="RUNNING")
    assert r.returncode == 1
    assert "start" not in _verbs(tmp_path)
    assert set(_verbs(tmp_path)) == {"describe"}


def test_execute_refuses_concurrent_step(tmp_path: Path) -> None:
    lock = tmp_path / "berlin-lst-val-baselines.lock"
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = _run(tmp_path, "rf-smoke", "main", "0.168788", "--execute")
    assert r.returncode == 1
    assert "another val-baseline step" in r.stderr
    assert not (tmp_path / "gcloud.log").exists()


def test_naive_maps_to_baseline_v6_val_only(tmp_path: Path) -> None:
    r = _run(tmp_path, "naive", "main", "0.168788")
    assert r.returncode == 0, r.stderr
    assert "run-baseline-vm.sh main baseline_v6_val" in r.stdout
