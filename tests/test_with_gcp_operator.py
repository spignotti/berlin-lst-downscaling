"""Offline tests for scripts/operators/with-gcp-operator.sh.

Every subprocess gets a constructed environment and fake `infisical` / `gcloud`
binaries first on PATH, so no real credentials or network are ever used.
"""

from __future__ import annotations

import shutil
import signal
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "operators" / "with-gcp-operator.sh"
BASH = shutil.which("bash")
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")

pytestmark = pytest.mark.skipif(
    BASH is None or PYTHON3 is None,
    reason="bash and python3 (in /usr/bin:/bin) are required",
)

KEY_MARKER = "FAKE-PRIVATE-KEY-MARKER"
TOKEN_MARKER = "FAKE-TOKEN-VALUE"  # noqa: S105
CLIENT_SECRET = "FAKE-CLIENT-SECRET-VALUE"  # noqa: S105
EMAIL = "operator@example.iam.gserviceaccount.com"

FAKE_INFISICAL = f"""#!/usr/bin/env bash
set +x
log="$FAKE_LOG_DIR/infisical.log"
case "$1" in
  login)
    printf 'login\\n' >> "$log"
    printf 'login-args=%s\\n' "$*" >> "$FAKE_LOG_DIR/infisical.argv"
    if [[ -n "${{INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET:-}}" ]]; then
      printf '%s\\n' '{TOKEN_MARKER}'
    else
      exit 1
    fi
    ;;
  secrets)
    printf 'secrets\\n' >> "$log"
    printf 'secrets-args=%s\\n' "$*" >> "$FAKE_LOG_DIR/infisical.argv"
    printf 'token=%s\\n' "${{INFISICAL_TOKEN-UNSET}}" >> "$FAKE_LOG_DIR/infisical.env"
    case "${{FAKE_INFISICAL_MODE:-ok}}" in
      ok)
        printf '{{"type": "service_account", "client_email": "{EMAIL}", '
        printf '"private_key": "{KEY_MARKER}"}}\\n'
        ;;
      nonjson) printf '{KEY_MARKER} not json\\n' ;;
      empty) ;;
      fail) exit 1 ;;
    esac
    ;;
esac
"""

FAKE_GCLOUD = """#!/usr/bin/env bash
set +x
printf '%s\\n' "$*" >> "$FAKE_LOG_DIR/gcloud.args"
"""

FAKE_CMD = """#!/usr/bin/env bash
set +x
f="${GOOGLE_APPLICATION_CREDENTIALS-UNSET}"
{
  printf 'path=%s\\n' "$f"
  printf 'mode=%s\\n' "$(stat -c %a "$f")"
  printf 'override=%s\\n' "${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE-UNSET}"
  printf 'valid=%s\\n' "$(grep -c service_account "$f")"
} > "$FAKE_LOG_DIR/cmd.txt"
if [[ "${CMD_SIGTERM_PARENT:-}" == "1" ]]; then
  kill -TERM "$PPID"
  sleep 0.2
fi
exit "${CMD_EXIT:-0}"
"""


@dataclass
class Run:
    proc: subprocess.CompletedProcess[str]
    log_dir: Path
    tmp_dir: Path
    cmd: Path

    def cmd_record(self) -> dict[str, str]:
        text = (self.log_dir / "cmd.txt").read_text()
        return dict(line.split("=", 1) for line in text.splitlines())


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _run(
    tmp_path: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    bash_flags: tuple[str, ...] = (),
) -> Run:
    assert BASH is not None
    bin_dir = tmp_path / "bin"
    log_dir = tmp_path / "logs"
    tmp_dir = tmp_path / "tmp"
    for d in (bin_dir, log_dir, tmp_dir):
        d.mkdir(exist_ok=True)
    _write_exec(bin_dir / "infisical", FAKE_INFISICAL)
    _write_exec(bin_dir / "gcloud", FAKE_GCLOUD)
    cmd = bin_dir / "fake-cmd"
    _write_exec(cmd, FAKE_CMD)
    full_env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_dir),
        "FAKE_LOG_DIR": str(log_dir),
        **(env or {}),
    }
    expanded = [str(cmd) if a == "{CMD}" else a for a in args]
    proc = subprocess.run(  # noqa: S603
        [BASH, *bash_flags, str(SCRIPT), *expanded],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    run = Run(proc=proc, log_dir=log_dir, tmp_dir=tmp_dir, cmd=cmd)
    _assert_no_secrets(run)
    return run


def _assert_no_secrets(run: Run) -> None:
    combined = run.proc.stdout + run.proc.stderr
    for secret in (KEY_MARKER, TOKEN_MARKER, CLIENT_SECRET):
        assert secret not in combined


def _ua_env(**extra: str) -> dict[str, str]:
    return {
        "INFISICAL_THESIS_CLIENT_ID": "fake-client-id",
        "INFISICAL_THESIS_CLIENT_SECRET": CLIENT_SECRET,
        **extra,
    }


def _no_leftover(run: Run) -> None:
    assert list(run.tmp_dir.iterdir()) == []


def test_success_file_mode_env_and_cleanup(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env=_ua_env())
    assert run.proc.returncode == 0, run.proc.stderr
    rec = run.cmd_record()
    assert rec["mode"] == "600"
    assert rec["override"] == rec["path"]
    assert rec["valid"] == "1"
    assert Path(rec["path"]).parent == run.tmp_dir
    assert not Path(rec["path"]).exists()
    _no_leftover(run)


def test_command_exit_code_propagates(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env=_ua_env(CMD_EXIT="7"))
    assert run.proc.returncode == 7
    assert not Path(run.cmd_record()["path"]).exists()
    _no_leftover(run)


def test_sigterm_cleans_up(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env=_ua_env(CMD_SIGTERM_PARENT="1"))
    assert run.proc.returncode == 128 + signal.SIGTERM
    assert not Path(run.cmd_record()["path"]).exists()
    _no_leftover(run)


@pytest.mark.parametrize("mode", ["nonjson", "empty"])
def test_invalid_secret_fails_closed(tmp_path: Path, mode: str) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env=_ua_env(FAKE_INFISICAL_MODE=mode))
    assert run.proc.returncode != 0
    assert "GCP_OPERATOR_SA_KEY" in run.proc.stderr
    assert not (run.log_dir / "cmd.txt").exists()
    _no_leftover(run)


def test_fetch_failure_without_auth_explains_options(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env={"FAKE_INFISICAL_MODE": "fail"})
    assert run.proc.returncode != 0
    assert "INFISICAL_THESIS_CLIENT_ID" in run.proc.stderr
    assert "infisical login" in run.proc.stderr
    assert not (run.log_dir / "cmd.txt").exists()
    _no_leftover(run)


def test_client_id_without_secret(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        ["--", "{CMD}"],
        env={"INFISICAL_THESIS_CLIENT_ID": "fake-client-id"},
    )
    assert run.proc.returncode != 0
    assert "INFISICAL_THESIS_CLIENT_SECRET is missing" in run.proc.stderr
    assert not (run.log_dir / "infisical.log").exists()
    _no_leftover(run)


def test_xtrace_does_not_leak(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        ["--", "{CMD}"],
        env=_ua_env(SHELLOPTS="xtrace"),
        bash_flags=("-x",),
    )
    assert run.proc.returncode == 0, run.proc.stderr
    _no_leftover(run)


def test_check_prints_only_client_email(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--check"], env=_ua_env())
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.proc.stdout == f"{EMAIL}\n"
    _no_leftover(run)


def test_activate_uses_gcloud_without_override(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--activate", "--", "{CMD}"], env=_ua_env())
    assert run.proc.returncode == 0, run.proc.stderr
    rec = run.cmd_record()
    assert rec["override"] == "UNSET"
    gcloud_args = (run.log_dir / "gcloud.args").read_text().strip()
    assert gcloud_args == (f"auth activate-service-account --key-file={rec['path']} --quiet")
    _no_leftover(run)


def test_no_command_is_usage_error(tmp_path: Path) -> None:
    run = _run(tmp_path, [], env=_ua_env())
    assert run.proc.returncode == 2
    assert not (run.log_dir / "infisical.log").exists()


def test_universal_auth_keeps_secret_off_argv(tmp_path: Path) -> None:
    run = _run(tmp_path, ["--", "{CMD}"], env=_ua_env())
    assert run.proc.returncode == 0, run.proc.stderr
    argv = (run.log_dir / "infisical.argv").read_text()
    assert "login-args=login --method=universal-auth --plain --silent" in argv
    assert "--path=/operator" in argv
    assert "--env=dev" in argv
    assert "GCP_OPERATOR_SA_KEY" in argv
    for secret in (CLIENT_SECRET, TOKEN_MARKER):
        assert secret not in argv
    env_log = (run.log_dir / "infisical.env").read_text()
    assert f"token={TOKEN_MARKER}" in env_log


def test_script_file_is_not_world_writable() -> None:
    assert not SCRIPT.stat().st_mode & stat.S_IWOTH
