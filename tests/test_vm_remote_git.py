"""Offline tests for scripts/runners/vm/remote-git-lib.sh (real git, fake pgrep/git/ls)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM_DIR = REPO / "scripts" / "runners" / "vm"
LIB = VM_DIR / "remote-git-lib.sh"
BASH = shutil.which("bash")
GIT = shutil.which("git")
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")

pytestmark = pytest.mark.skipif(
    BASH is None or GIT is None or PYTHON3 is None,
    reason="bash, git and python3 are required",
)

FOREIGN = {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}


def _write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def env_dir(tmp_path: Path) -> Path:
    base = tmp_path.resolve()
    (base / "home").mkdir()
    (base / "bin").mkdir()
    _write_exec(base / "bin" / "pgrep", "#!/bin/sh\nexit 1\n")
    return base


def _env(base: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": f"{base / 'bin'}:/usr/bin:/bin",
        "HOME": str(base / "home"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": str(base / "home" / ".gitconfig-absent"),
    }
    env.update(extra or {})
    return env


def _git(path: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603
        [GIT or "", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return out.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "t")
    (path / "a.txt").write_text("a\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "init")


def _gen(fn: str, arg: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [BASH or "", "-c", f'source "$1"; {fn} "$2"', "_", str(LIB), arg],
        capture_output=True,
        text=True,
        check=False,
    )


def _sh(snippet: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["/bin/sh", "-c", snippet], env=env, capture_output=True, text=True, check=False
    )


def _probe(path: Path | str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    gen = _gen("vm_remote_idle_probe_cmd", str(path))
    assert gen.returncode == 0, gen.stderr
    return _sh(gen.stdout, env)


def _preamble(path: Path | str) -> str:
    gen = _gen("vm_remote_git_preamble", str(path))
    assert gen.returncode == 0, gen.stderr
    return gen.stdout


@pytest.mark.parametrize(
    "arg",
    [
        "relative/app",
        "/workspace/app/",
        "/workspace/../etc",
        "/workspace/app;rm -rf /",
        "/work space",
        "",
    ],
)
def test_preamble_rejects_unsafe_literal(arg: str) -> None:
    gen = _gen("vm_remote_git_preamble", arg)
    assert gen.returncode != 0
    assert gen.stdout == ""
    probe = _gen("vm_remote_idle_probe_cmd", arg)
    assert probe.returncode != 0
    assert probe.stdout == ""


def test_probe_idle_and_dirty_for_foreign_owner(env_dir: Path) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    env = _env(env_dir, FOREIGN)
    plain = subprocess.run(  # noqa: S603
        [GIT or "", "-C", str(repo), "status"], env=env, capture_output=True, text=True, check=False
    )
    if "dubious ownership" not in plain.stderr:
        pytest.skip("installed git ignores GIT_TEST_ASSUME_DIFFERENT_OWNER")
    res = _probe(repo, env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "idle"
    (repo / "new.txt").write_text("x\n")
    assert _probe(repo, env).stdout.strip() == "dirty"
    assert not (env_dir / "home" / ".gitconfig").exists()
    assert not (env_dir / "home" / ".gitconfig-absent").exists()


def test_preamble_rgit_fetch_reset_for_foreign_owner(env_dir: Path) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    origin = env_dir / "origin.git"
    _git(env_dir, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    other = env_dir / "other"
    _git(env_dir, "clone", "-q", str(origin), str(other))
    _git(other, "config", "user.email", "t@example.invalid")
    _git(other, "config", "user.name", "t")
    _git(repo, "push", "-q", "origin", "HEAD:refs/heads/main")
    _git(other, "fetch", "-q", "origin", "main")
    _git(other, "checkout", "-q", "-B", "feat/x", "origin/main")
    (other / "b.txt").write_text("b\n")
    _git(other, "add", "b.txt")
    _git(other, "commit", "-q", "-m", "new")
    _git(other, "push", "-q", "origin", "feat/x")
    sha = _git(other, "rev-parse", "HEAD")

    snippet = _preamble(repo) + (
        "rgit fetch origin feat/x:refs/remotes/origin/feat/x && "
        "rgit checkout -B feat/x refs/remotes/origin/feat/x && "
        "rgit reset --hard refs/remotes/origin/feat/x && "
        "rgit rev-parse HEAD\n"
    )
    # The simulated foreign owner also applies to the local origin; trust it through a
    # test-only global config (the code under test never writes one).
    trust = env_dir / "trust.gitconfig"
    trust.write_text(f"[safe]\n\tdirectory = {origin}\n")
    extra = {**FOREIGN, "GIT_CONFIG_GLOBAL": str(trust)}
    res = _sh(snippet, _env(env_dir, extra))
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == sha


def test_probe_broken_when_app_dir_missing(env_dir: Path) -> None:
    missing = env_dir / "nope"
    assert _probe(missing, _env(env_dir)).stdout.strip() == "broken"
    res = _sh(_preamble(missing), _env(env_dir))
    assert res.returncode == 3
    assert "missing" in res.stderr


def test_probe_broken_when_app_dir_is_symlink(env_dir: Path) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    link = env_dir / "link"
    link.symlink_to(repo)
    assert _probe(link, _env(env_dir)).stdout.strip() == "broken"


def test_probe_broken_when_parent_is_symlink(env_dir: Path) -> None:
    _init_repo(env_dir / "repo")
    (env_dir / "parentlink").symlink_to(env_dir)
    assert _probe(env_dir / "parentlink" / "repo", _env(env_dir)).stdout.strip() == "broken"


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_probe_broken_when_dot_git_is_symlink_or_file(env_dir: Path, kind: str) -> None:
    other = env_dir / "other"
    _init_repo(other)
    app = env_dir / "app"
    app.mkdir()
    if kind == "symlink":
        (app / ".git").symlink_to(other / ".git")
    else:
        (app / ".git").write_text("gitdir: /elsewhere\n")
    assert _probe(app, _env(env_dir)).stdout.strip() == "broken"


@pytest.mark.parametrize("variant", ["status_only", "always"])
def test_probe_broken_on_git_error(env_dir: Path, variant: str) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    cond = "'status' in sys.argv" if variant == "status_only" else "True"
    _write_exec(
        env_dir / "bin" / "git",
        f"#!{PYTHON3}\nimport os, sys\n"
        f"if {cond}:\n    sys.exit(128)\n"
        f"os.execv({GIT!r}, [{GIT!r}] + sys.argv[1:])\n",
    )
    assert _probe(repo, _env(env_dir)).stdout.strip() == "broken"


@pytest.mark.parametrize("case", ["mismatch", "root"])
def test_probe_broken_on_owner_mismatch_or_root(env_dir: Path, case: str) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    uid = "0" if case == "root" else "4242"
    git_uid = "0" if case == "root" else str(os.getuid())
    dir_uid = "0" if case == "root" else uid
    _write_exec(
        env_dir / "bin" / "ls",
        f"#!{PYTHON3}\nimport sys\np = sys.argv[-1]\n"
        f"u = {git_uid!r} if p.endswith('/.git') else {dir_uid!r}\n"
        "print(f'drwxr-xr-x 2 {u} {u} 0 Jan 1 00:00 {p}')\n",
    )
    assert _probe(repo, _env(env_dir)).stdout.strip() == "broken"


def test_probe_occupied_short_circuits(env_dir: Path) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    log = env_dir / "git.log"
    _write_exec(env_dir / "bin" / "pgrep", "#!/bin/sh\nexit 0\n")
    _write_exec(env_dir / "bin" / "git", f"#!/bin/sh\necho called >> {log}\nexit 1\n")
    res = _probe(repo, _env(env_dir))
    assert res.stdout.strip() == "occupied"
    assert not log.exists()


def _logging_git(env_dir: Path, log: Path) -> None:
    _write_exec(
        env_dir / "bin" / "git",
        f"#!{PYTHON3}\nimport json, os, sys\n"
        f"with open({str(log)!r}, 'a') as fh:\n    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"os.execv({GIT!r}, [{GIT!r}] + sys.argv[1:])\n",
    )


def test_every_git_call_is_path_scoped(env_dir: Path) -> None:
    repo = env_dir / "repo"
    _init_repo(repo)
    log = env_dir / "argv.log"
    _logging_git(env_dir, log)
    env = _env(env_dir)
    assert _probe(repo, env).stdout.strip() == "idle"
    deploy = _preamble(repo) + "rgit rev-parse HEAD\n"
    assert _sh(deploy, env).returncode == 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls
    prefix = ["-c", f"safe.directory={repo}", "-C", str(repo)]
    for argv in calls:
        assert argv[:4] == prefix
        joined = " ".join(argv)
        assert "safe.directory=*" not in joined
        assert "--global" not in argv
        assert "--system" not in argv
        assert "config" not in argv


def test_launchers_use_shared_helper_and_no_forbidden_ops() -> None:
    common = (VM_DIR / "vm-runner-common.sh").read_text()
    for needle in (
        'source "$VM_SCRIPTS/remote-git-lib.sh"',
        "vm_remote_git_preamble",
        "rgit fetch",
        "rgit rev-parse HEAD",
    ):
        assert needle in common
    assert not re.search(r"cd \$APP_DIR\s*&&\s*\\?\s*git", common)
    rf = (VM_DIR / "run-random-forest-vm.sh").read_text()
    assert "vm_remote_idle_probe_cmd" in rf
    assert "git -C" not in rf
    assert "status --porcelain" not in rf

    forbidden = [
        r"safe\.directory=\*",
        r"git config --global",
        r"\bsudo\b",
        r"\bchown\b",
        r"chmod\s+0?777",
        r"setfacl",
        r"add-iam-policy-binding",
    ]
    for script in sorted(VM_DIR.glob("*.sh")):
        text = script.read_text()
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{script.name}: {pattern}"

    for script in (LIB, VM_DIR / "vm-runner-common.sh", VM_DIR / "run-random-forest-vm.sh"):
        res = subprocess.run(  # noqa: S603
            [BASH or "", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert res.returncode == 0, res.stderr
