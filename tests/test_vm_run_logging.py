"""Offline checks for the VM launcher run-log helper (scripts/runners/vm/_logging.sh)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_VM = _ROOT / "scripts/runners/vm"

_GCLOUD_STUB = """#!/usr/bin/env bash
# gcloud storage cp SRC... DEST/ -> local copy; FAIL_UPLOAD=1 simulates an outage.
[[ "$1 $2" == "storage cp" ]] || exit 9
shift 2
dest="${@: -1}"
set -- "${@:1:$#-1}"
[[ -n "${FAIL_UPLOAD:-}" ]] && exit 1
mkdir -p "$dest" && cp "$@" "$dest"
"""

_FAKE_LAUNCHER = """#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
echo "launcher stdout $*"
echo "launcher stderr" >&2
echo "remote log" > "$RUNLOG_DIR/$RUNLOG_ID.remote-nohup.log"
exit "$1"
"""


def _run(
    tmp_path: Path, rc: int, *, fail_upload: bool
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    repo = tmp_path / "repo"
    vm = repo / "scripts/runners/vm"
    vm.mkdir(parents=True)
    shutil.copy(_VM / "_logging.sh", vm / "_logging.sh")
    launcher = vm / "run-fake-vm.sh"
    launcher.write_text(_FAKE_LAUNCHER)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gcloud"
    stub.write_text(_GCLOUD_STUB)
    stub.chmod(0o755)
    gcs = tmp_path / "gcs"
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "LOG_GCS_PREFIX": str(gcs),
        "FAIL_UPLOAD": "1" if fail_upload else "",
    }
    env.pop("RUNLOG_ACTIVE", None)
    proc = subprocess.run(  # noqa: S603
        ["bash", str(launcher), str(rc)],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, repo / ".runs/logs", gcs


@pytest.mark.parametrize("rc", [0, 3])
def test_log_uploaded_and_exit_code_preserved(tmp_path: Path, rc: int) -> None:
    proc, local, gcs = _run(tmp_path, rc, fail_upload=False)
    assert proc.returncode == rc
    uploaded = sorted(p.name for p in gcs.glob("*/*"))
    assert len(uploaded) == 2
    log = next(p for p in gcs.glob("*/*.log") if "remote" not in p.name)
    assert log.parent.name.count("-") == 1 and len(log.parent.name) == 7  # YYYY-MM
    assert log.name.startswith("run-fake-vm-")
    text = log.read_text()
    assert f"launcher stdout {rc}" in text and "launcher stderr" in text
    assert f"Run log uploaded: {gcs}/{log.parent.name}/{log.name}" in proc.stdout
    assert list(local.iterdir()) == []


def test_upload_failure_keeps_local_log_and_exit_code(tmp_path: Path) -> None:
    proc, local, gcs = _run(tmp_path, 5, fail_upload=True)
    assert proc.returncode == 5
    assert not gcs.exists()
    assert "WARNING: run log upload" in proc.stderr
    kept = sorted(p.name for p in local.iterdir())
    assert len(kept) == 2
    assert any(name.endswith(".remote-nohup.log") for name in kept)


def test_every_launcher_sources_the_helper_first() -> None:
    launchers = sorted(_VM.glob("run-*-vm.sh"))
    assert launchers
    for path in launchers:
        text = path.read_text()
        helper = text.index('/_logging.sh"')
        assert helper < text.index('/vm-runner-common.sh"'), path.name
        assert "vm-runs" not in text and "$HOME/vm" not in text, path.name
