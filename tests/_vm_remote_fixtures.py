"""Shared offline helpers for the VM remote-git tests (real git, fake uv/stat/pgrep/git)."""

from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM_DIR = REPO / "scripts" / "runners" / "vm"
LIB = VM_DIR / "remote-git-lib.sh"
BASH = shutil.which("bash")
GIT = shutil.which("git")
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")
SHELL = shutil.which("dash") or "/bin/sh"
STAT = "/usr/bin/stat"
UID = os.getuid()
USER = pwd.getpwuid(UID).pw_name

FOREIGN = {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}

HAVE_TOOLS = (
    BASH is not None
    and GIT is not None
    and PYTHON3 is not None
    and os.path.exists(STAT)
    and UID != 0
)
SKIP_REASON = "bash, git, python3, /usr/bin/stat and a non-root user are required"


def write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def git(path: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603
        [GIT or "", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        },
    )
    return out.stdout.strip()


def init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@example.invalid")
    git(path, "config", "user.name", "t")


@dataclass
class Origin:
    """Local bare origin with branches ``main`` and ``feat/x`` plus a work clone to advance it."""

    path: Path
    work: Path
    sha: str

    def advance(self, name: str = "next.txt") -> str:
        """Push a new commit to ``feat/x`` and return its sha."""
        git(self.work, "checkout", "-q", "feat/x")
        (self.work / name).write_text(f"{name}\n")
        git(self.work, "add", name)
        git(self.work, "commit", "-q", "-m", f"add {name}")
        git(self.work, "push", "-q", "origin", "feat/x")
        self.sha = git(self.work, "rev-parse", "HEAD")
        return self.sha


def build_origin(base: Path) -> Origin:
    bare = base / "origin.git"
    git(base, "init", "-q", "--bare", "-b", "main", str(bare))
    work = base / "work"
    init_repo(work)
    (work / "a.txt").write_text("a\n")
    (work / ".gitignore").write_text("data/\nlogs/\n.venv/\n")
    git(work, "add", "a.txt", ".gitignore")
    git(work, "commit", "-q", "-m", "init")
    git(work, "remote", "add", "origin", str(bare))
    git(work, "push", "-q", "origin", "main")
    git(work, "checkout", "-q", "-b", "feat/x")
    (work / "b.txt").write_text("b\n")
    git(work, "add", "b.txt")
    git(work, "commit", "-q", "-m", "feature")
    git(work, "push", "-q", "origin", "feat/x")
    return Origin(bare, work, git(work, "rev-parse", "HEAD"))


def write_fake_uv(bin_dir: Path, log: Path, *, version_ok: bool = True) -> None:
    version = "echo uv 0; exit 0" if version_ok else "exit 1"
    write_exec(
        bin_dir / "uv",
        '#!/bin/sh\ncase "$1" in\n'
        f"  --version) {version};;\n  sync) echo sync >> {log};;\n"
        "esac\nexit 0\n",
    )


def write_fake_stat(bin_dir: Path, mapping: list[tuple[str, int]]) -> None:
    """``stat -c %u -- P`` prints the configured uid when P ends with a configured suffix."""
    write_exec(
        bin_dir / "stat",
        f"#!{PYTHON3}\nimport os, sys\nM = {mapping!r}\na = sys.argv[1:]\n"
        "if a[:3] == ['-c', '%u', '--'] and len(a) == 4:\n"
        "    for suffix, uid in M:\n"
        "        if a[3].endswith(suffix):\n"
        "            print(uid)\n"
        "            sys.exit(0)\n"
        f"os.execv({STAT!r}, [{STAT!r}] + a)\n",
    )


def write_logging_git(bin_dir: Path, log: Path, *, fail_on: str | None = None) -> None:
    """Log argv to ``log`` and exec the real git; optionally exit 128 when ``fail_on`` is an arg."""
    fail = f"if {fail_on!r} in sys.argv:\n    sys.exit(128)\n" if fail_on else ""
    write_exec(
        bin_dir / "git",
        f"#!{PYTHON3}\nimport json, os, sys\n"
        f"with open({str(log)!r}, 'a') as fh:\n    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"{fail}"
        f"os.execv({GIT!r}, [{GIT!r}] + sys.argv[1:])\n",
    )


def read_git_log(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def gen(fn: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [BASH or "", "-c", 'source "$1"; fn=$2; shift 2; "$fn" "$@"', "_", str(LIB), fn, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def foreign_owner_supported(base: Path, repo: Path) -> bool:
    plain = subprocess.run(  # noqa: S603
        [GIT or "", "-C", str(repo), "status"],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(base),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            **FOREIGN,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return "dubious ownership" in plain.stderr


class Sandbox:
    """tmp base with ``home``, ``bin`` (fake uv, pgrep), and a local origin."""

    def __init__(self, base: Path) -> None:
        self.base = base.resolve()
        self.home = self.base / "home"
        self.bin = self.base / "bin"
        self.home.mkdir()
        self.bin.mkdir()
        self.app = self.home / "berlin-lst-app"
        self.uv_log = self.base / "uv.log"
        self.git_log = self.base / "git.log"
        write_exec(self.bin / "pgrep", "#!/bin/sh\nexit 1\n")
        write_fake_uv(self.bin, self.uv_log)
        self.origin = build_origin(self.base)

    @property
    def uv_called(self) -> bool:
        return self.uv_log.exists()

    def env(
        self, extra: dict[str, str] | None = None, *, path: str | None = None
    ) -> dict[str, str]:
        git_dir = str(Path(GIT or "/usr/bin/git").parent)
        env = {
            "PATH": path or f"{self.bin}:{git_dir}:/usr/bin:/bin",
            "HOME": str(self.home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(self.base / "gitconfig-absent"),
        }
        env.update(extra or {})
        return env

    def cmd(
        self,
        fn: str,
        *extra_args: str,
        home: str | Path | None = None,
        user: str = USER,
        pin: str = "",
        origin: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return gen(
            fn,
            str(home if home is not None else self.home),
            user,
            str(origin if origin is not None else self.origin.path),
            pin,
            *extra_args,
        )

    def run(
        self,
        fn: str,
        *extra_args: str,
        env: dict[str, str] | None = None,
        **kwargs: str | Path | None,
    ) -> subprocess.CompletedProcess[str]:
        generated = self.cmd(fn, *extra_args, **kwargs)  # type: ignore[arg-type]
        assert generated.returncode == 0, generated.stderr
        return subprocess.run(  # noqa: S603
            [SHELL, "-c", generated.stdout],
            env=env if env is not None else self.env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def deploy(
        self,
        sha: str | None = None,
        branch: str = "feat/x",
        *,
        env: dict[str, str] | None = None,
        **kwargs: str | Path | None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run("vm_remote_deploy_cmd", branch, sha or self.origin.sha, env=env, **kwargs)

    def probe(self, env: dict[str, str] | None = None) -> str:
        res = self.run("vm_remote_idle_probe_cmd", env=env)
        return res.stdout.strip()

    def provenance(self, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return self.run("vm_remote_provenance_cmd", env=env)

    def head(self) -> str:
        return git(self.app, "rev-parse", "HEAD")


def require_tools() -> None:
    if not HAVE_TOOLS:
        pytest.skip(SKIP_REASON)
