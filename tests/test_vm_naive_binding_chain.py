"""Offline chain tests: naive evidence is secured, verified, uploaded and bound (no cloud calls)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _vm_remote_fixtures import BASH, HAVE_TOOLS, SKIP_REASON, git, write_exec

REPO = Path(__file__).resolve().parents[1]
VM_DIR = REPO / "scripts" / "runners" / "vm"
FIXTURE = REPO / "tests" / "fixtures" / "synthetic_naive_v6_val"
LAUNCHER = VM_DIR / "run-random-forest-vm.sh"
LIB = VM_DIR / "naive-evidence-lib.sh"

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason=SKIP_REASON)

FAKE_GCLOUD = f"""#!{sys.executable}
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

FAKE_UV = """#!/bin/bash
echo "$*" >> "$FAKE_TMP/uv.log"
shift 3
exec "$FAKE_PY" "$@"
"""

FAKE_SSH = """#!/bin/bash
echo "$*" >> "$FAKE_TMP/ssh.log"
exit 99
"""

BIN_PATH = "/usr/bin:/bin"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(VM_DIR, repo / "scripts" / "runners" / "vm")
    val = repo / "scripts" / "validators"
    val.mkdir(parents=True)
    for name in ("verify_naive_binding.py", "validate_run_provenance.py"):
        shutil.copy(REPO / "scripts" / "validators" / name, val / name)
    shutil.copytree(REPO / "configs" / "modeling", repo / "configs" / "modeling")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "sandbox")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exec(bin_dir / "gcloud", FAKE_GCLOUD)
    write_exec(bin_dir / "uv", FAKE_UV)
    write_exec(bin_dir / "ssh", FAKE_SSH)
    (tmp_path / "home").mkdir()
    return repo


def _env(repo: Path) -> dict[str, str]:
    tmp = repo.parent
    return {
        "PATH": f"{tmp / 'bin'}:{BIN_PATH}",
        "HOME": str(tmp / "home"),
        "FAKE_TMP": str(tmp),
        "FAKE_PY": sys.executable,
        "FAKE_STATUS": "RUNNING",
        "RUNLOG_ACTIVE": "1",
        "PYTHONPATH": str(REPO / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def _launch(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [BASH or "", str(repo / "scripts" / "runners" / "vm" / "run-random-forest-vm.sh"), *args],
        env=_env(repo),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _box_copy(tmp: Path) -> Path:
    dest = tmp / "box" / "vm" / "baseline-v6-notest" / "run1"
    shutil.copytree(FIXTURE, dest)
    return dest / "baseline_report.json"


def _verbs(tmp: Path) -> list[str]:
    log = tmp / "gcloud.log"
    return [json.loads(x)[2] for x in log.read_text().splitlines()] if log.exists() else []


def test_chain_naive_secures_then_rf_binds(sandbox: Path) -> None:
    tmp = sandbox.parent
    report = _box_copy(tmp)
    sha = _sha(report)
    rec = subprocess.run(  # noqa: S603
        [BASH or "", "-c", f'source "{LIB}"; naive_evidence_record "$1"', "_", str(report)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": BIN_PATH},
    )
    assert rec.returncode == 0, rec.stderr
    assert (report.parent / "baseline_report.json.sha256").read_text().split()[0] == sha
    assert f"--naive-sha256 {sha}" in rec.stdout
    for mode in ("v6-val-smoke", "v6-val-full"):
        (tmp / "gcloud.log").unlink(missing_ok=True)
        r = _launch(
            sandbox, mode, "main", "0.168788", "--naive-report", str(report), "--naive-sha256", sha
        )
        assert r.returncode != 0
        assert "OK: naive evidence bound" in r.stdout, r.stdout + r.stderr
        assert "verify_naive_binding.py" in (tmp / "uv.log").read_text()
        assert set(_verbs(tmp)) == {"describe"}
        assert not (tmp / "ssh.log").exists()


def _stale(report: Path) -> None:
    data = json.loads(report.read_text())
    data["fingerprints"]["source_policy_hash"] = "9f9287255803d759"
    data["fingerprints"]["patches_total"] = "10018"
    report.write_text(json.dumps(data))


def _wrong_dataset(report: Path) -> None:
    data = json.loads(report.read_text())
    data["fingerprints"]["source_policy_hash"] = "0000000000000000"
    report.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "case",
    [
        "missing_report",
        "wrong_sha",
        "wrong_dataset",
        "stale_v3",
        "missing_flags",
        "legacy_with_binding",
        "quote_in_path",
        "test_opening_mode_with_val_only_binding",
    ],
)
def test_invalid_binding_stops_before_any_gcloud(sandbox: Path, case: str) -> None:
    tmp = sandbox.parent
    report = _box_copy(tmp)
    sha = _sha(report)
    mode = "v6-val-full"
    args: list[str] = ["--naive-report", str(report), "--naive-sha256", sha]
    if case == "missing_report":
        args = ["--naive-report", str(tmp / "absent.json"), "--naive-sha256", sha]
    elif case == "wrong_sha":
        args = ["--naive-report", str(report), "--naive-sha256", "a" * 64]
    elif case in ("wrong_dataset", "stale_v3"):
        (_wrong_dataset if case == "wrong_dataset" else _stale)(report)
        args = ["--naive-report", str(report), "--naive-sha256", _sha(report)]
    elif case == "missing_flags":
        args = []
    elif case == "legacy_with_binding":
        mode = "smoke"
    elif case == "quote_in_path":
        args = ["--naive-report", str(tmp / "it's.json"), "--naive-sha256", sha]
    elif case == "test_opening_mode_with_val_only_binding":
        mode = "v6-smoke"
    r = _launch(sandbox, mode, "main", "0.168788", *args)
    assert r.returncode == 1, r.stdout + r.stderr
    assert not (tmp / "gcloud.log").exists()
    assert not (tmp / "ssh.log").exists()


def _bash(script: str, tmp: Path, *args: str, extra_env: dict[str, str] | None = None):
    env = {"PATH": f"{tmp / 'fakebin'}:{BIN_PATH}", "REMOTE_FILE": str(tmp / "unused")}
    env.update(extra_env or {})
    return subprocess.run(  # noqa: S603
        [BASH or "", "-c", script, "_", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


PRELUDE = f"""
source "{LIB}"
ssh_cmd() {{ bash -c "$1"; }}
ssh_upload_cmd() {{ bash -c "$1"; }}
"""


def test_transfer_and_explicit_validator_arg(tmp_path: Path) -> None:
    report = _box_copy(tmp_path)
    sha = _sha(report)
    app = tmp_path / "vm" / "app"
    app.mkdir(parents=True)
    remote = app / "data" / "run-naive" / "baseline_report.json"
    fake = tmp_path / "fakebin"
    fake.mkdir()
    write_exec(fake / "timeout", '#!/bin/bash\nshift 3\nexec "$@"\n')
    write_exec(fake / "uv", '#!/bin/bash\necho "$*" >> "$ARGV_LOG"\n')
    r = _bash(
        PRELUDE
        + """
naive_evidence_push "$1" "$2" "$3" || exit 10
naive_evidence_check_remote "$2" "$3" || exit 11
cmd=$(rf_validation_remote_cmd "$4" 60 "$4/out/random_forest_report.json" "$3")
bash -c "$cmd"
""",
        tmp_path,
        str(report),
        sha,
        str(remote),
        str(app),
        extra_env={"ARGV_LOG": str(tmp_path / "argv.log")},
    )
    assert r.returncode == 0, r.stderr
    assert remote.read_bytes() == report.read_bytes()
    argv = (tmp_path / "argv.log").read_text()
    assert f"--naive-report {remote}" in argv
    assert "--skip-naive" not in argv


def test_transfer_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    report = _box_copy(tmp_path)
    sha = _sha(report)
    remote = tmp_path / "vm" / "naive" / "baseline_report.json"
    corrupting = f"""
source "{LIB}"
ssh_cmd() {{ bash -c "$1"; }}
ssh_upload_cmd() {{ bash -c "$1" && printf x >> "$REMOTE_FILE"; }}
"""
    # Case A: the upload corrupts the remote bytes.
    a = _bash(
        corrupting + 'naive_evidence_push "$1" "$2" "$3"',
        tmp_path,
        str(report),
        sha,
        str(remote),
        extra_env={"REMOTE_FILE": str(remote)},
    )
    assert a.returncode != 0
    assert "sha256" in a.stderr
    # Case B: a valid push, then the remote file is tampered with.
    b = _bash(
        PRELUDE
        + """
naive_evidence_push "$1" "$2" "$3" || exit 10
printf x >> "$3"
naive_evidence_check_remote "$2" "$3"
""",
        tmp_path,
        str(report),
        sha,
        str(remote),
    )
    assert b.returncode not in (0, 10)
    # Case C: the local file changes after binding; ssh is never called.
    report.write_bytes(report.read_bytes() + b" ")
    calls = tmp_path / "ssh.calls"
    c = _bash(
        f"""
source "{LIB}"
ssh_cmd() {{ echo cmd >> "{calls}"; }}
ssh_upload_cmd() {{ echo up >> "{calls}"; }}
naive_evidence_push "$1" "$2" "$3"
""",
        tmp_path,
        str(report),
        sha,
        str(tmp_path / "other"),
    )
    assert c.returncode != 0
    assert not calls.exists()


def test_launcher_ssh_upload_cmd_forwards_stdin(tmp_path: Path) -> None:
    text = LAUNCHER.read_text()
    match = re.search(r"^UPLOAD_TIMEOUT_PY='.*?^ssh_upload_cmd\(\) \{.*?^\}\n", text, re.S | re.M)
    assert match is not None
    block = match.group(0)
    assert "<<" not in block
    assert "python3 -c" in block
    vm = tmp_path / "vm"
    vm.mkdir()
    write_exec(vm / "ssh-vm.sh", '#!/bin/bash\n[ "$1" = "--" ] || exit 2\nshift\nbash -c "$1"\n')
    src = tmp_path / "payload.bin"
    src.write_bytes(b"a\x00b\nc" * 1000)
    dest = tmp_path / "out" / "payload.bin"
    script = f"""
remaining_seconds() {{ echo 30; }}
VM_SCRIPTS="{vm}"
{block}
ssh_upload_cmd "mkdir -p '{dest.parent}' && cat > '{dest}'" < "{src}"
"""
    r = _bash(script, tmp_path)
    assert r.returncode == 0, r.stderr
    assert dest.read_bytes() == src.read_bytes()


def test_launcher_validation_only_after_remote_check() -> None:
    text = LAUNCHER.read_text()
    assert text.index("naive_evidence_push") < text.index("vm_launch_detached")
    assert text.rindex("naive_evidence_check_remote") < text.index('rf_validation_remote_cmd "')
    assert text.count("validate_random_forest.py") == 0
    assert "--skip-naive" not in text
