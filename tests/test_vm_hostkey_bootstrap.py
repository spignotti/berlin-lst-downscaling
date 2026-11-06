"""Offline tests for hostkey-lib.sh and bootstrap-vm-hostkey.sh (fake gcloud/ssh only)."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM_DIR = REPO / "scripts" / "runners" / "vm"
LIB = VM_DIR / "hostkey-lib.sh"
BOOT = VM_DIR / "bootstrap-vm-hostkey.sh"
BASH = shutil.which("bash")
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")
SSH_KEYGEN = shutil.which("ssh-keygen")

pytestmark = pytest.mark.skipif(
    BASH is None or PYTHON3 is None or SSH_KEYGEN is None,
    reason="bash, python3 (in /usr/bin:/bin) and ssh-keygen are required",
)

VM_ID = "6236232769523665407"
ALIAS = f"compute.{VM_ID}"
SA = "berlin-lst-vertex@berlin-lst-training.iam.gserviceaccount.com"
MUTATING = ("start", "add-metadata", "remove-metadata", "get-serial-port-output")
ORIGINAL = "#!/bin/bash\necho original\n"

FAKE_GCLOUD = f"""#!{PYTHON3}
import json, os, re, sys
tmp = os.environ["FAKE_TMP"]
argv = sys.argv[1:]
with open(os.path.join(tmp, "gcloud.log"), "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
sp = os.path.join(tmp, "state.json")
st = json.load(open(sp))
def save():
    json.dump(st, open(sp, "w"))
verb = argv[2]
fmt = next((a.split("=", 1)[1] for a in argv if a.startswith("--format=")), "")
if verb == "describe":
    if fmt == "json":
        print(json.dumps({{"metadata": {{"items": st["items"]}}}}))
        sys.exit(0)
    m = re.match(r"value\\((.*)\\)$", fmt)
    expr = m.group(1)
    table = {{
        "id": st["id"],
        "status": st["status"],
        "disks[0].source.basename()": "berlin-lst-vm",
        "disks[0].autoDelete": "False",
        "serviceAccounts[0].email": st["sa"],
        "deletionProtection": "True" if st["protection"] else "False",
        "machineType.basename()": "n2-highmem-2",
    }}
    print(table[expr])
elif verb == "add-metadata":
    arg = next(a for a in argv if a.startswith("--metadata-from-file=startup-script="))
    content = open(arg.split("=", 2)[2]).read()
    mark = re.search(r"BERLIN-LST-HOSTKEY-([0-9a-f]+)", content)
    if mark is None and st.get("fail_restore"):
        sys.exit(1)
    st["items"] = [i for i in st["items"] if i["key"] != "startup-script"]
    st["items"].append({{"key": "startup-script", "value": content}})
    if mark:
        st["nonce"] = mark.group(1)
    save()
elif verb == "remove-metadata":
    st["items"] = [i for i in st["items"] if i["key"] != "startup-script"]
    save()
elif verb == "start":
    if st.get("fail_start"):
        sys.exit(1)
    st["status"] = "RUNNING"
    save()
elif verb == "get-serial-port-output":
    n = st["nonce"]
    print("NOISE-PREFIX-LINE boot message")
    print("startup-script: BERLIN-LST-HOSTKEY-deadbeefdeadbeef ssh-ed25519 AAAAstale")
    mode = st.get("serial_mode", "ok")
    if mode == "none":
        sys.exit(0)
    if mode == "malformed":
        print("startup-script: BERLIN-LST-HOSTKEY-%s ssh-ed25519 not*base64" % n)
    else:
        for t, b in st["serial_keys"]:
            print("startup-script: BERLIN-LST-HOSTKEY-%s %s %s" % (n, t, b))
    print("startup-script: BERLIN-LST-HOSTKEY-%s-END" % n)
else:
    sys.exit(2)
"""

FAKE_SSH = """#!/bin/sh
touch "$FAKE_TMP/ssh.called"
exit 99
"""


def _keygen(path: Path, kind: str) -> tuple[str, str]:
    subprocess.run(  # noqa: S603
        [SSH_KEYGEN or "", "-q", "-t", kind, "-N", "", "-f", str(path)],
        check=True,
        capture_output=True,
    )
    parts = Path(f"{path}.pub").read_text().split()
    return parts[0], parts[1]


@pytest.fixture
def keys(tmp_path: Path) -> dict[str, tuple[str, str]]:
    d = tmp_path / "keys"
    d.mkdir()
    return {
        "k1": _keygen(d / "k1", "ed25519"),
        "k2": _keygen(d / "k2", "ecdsa"),
        "k3": _keygen(d / "k3", "ed25519"),
    }


class Env:
    def __init__(self, tmp: Path, keys: dict[str, tuple[str, str]]) -> None:
        self.tmp = tmp
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.keys = keys
        for name, body in (("gcloud", FAKE_GCLOUD), ("ssh", FAKE_SSH)):
            p = self.bin / name
            p.write_text(body)
            p.chmod(0o755)
        self.state = {
            "id": VM_ID,
            "status": "TERMINATED",
            "sa": SA,
            "protection": True,
            "items": [],
            "nonce": "",
            "serial_mode": "ok",
            "serial_keys": [list(keys["k1"]), list(keys["k2"])],
        }
        self.save()

    def save(self) -> None:
        (self.tmp / "state.json").write_text(json.dumps(self.state))

    def load(self) -> dict:
        return json.loads((self.tmp / "state.json").read_text())

    def env(self) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "FAKE_TMP": str(self.tmp),
            "BERLIN_LST_HOSTKEY_POLL_INTERVAL": "0",
            "BERLIN_LST_HOSTKEY_POLL_TRIES": "2",
        }

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [BASH or "", str(BOOT), *args],
            env=self.env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def verbs(self) -> list[str]:
        log = self.tmp / "gcloud.log"
        if not log.exists():
            return []
        return [json.loads(line)[2] for line in log.read_text().splitlines()]

    def startup_value(self) -> str | None:
        for item in self.load()["items"]:
            if item["key"] == "startup-script":
                return item["value"]
        return None

    @property
    def kh(self) -> Path:
        return self.home / ".ssh" / "google_compute_known_hosts"


@pytest.fixture
def fx(tmp_path: Path, keys: dict[str, tuple[str, str]]) -> Env:
    return Env(tmp_path, keys)


def _lib(
    env: Env, snippet: str, stdin: str = "", args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            BASH or "",
            "-c",
            f"source {LIB}; VM_KNOWN_HOSTS_ALIAS={ALIAS}; {snippet}",
            "x",
            *(args or []),
        ],
        env=env.env(),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _serial(nonce: str, keys: list[tuple[str, str]], end: bool = True) -> str:
    lines = ["boot noise", "BERLIN-LST-HOSTKEY-deadbeefdeadbeef ssh-ed25519 AAAAstale"]
    lines += [f"[ 1.0] startup-script: BERLIN-LST-HOSTKEY-{nonce} {t} {b}" for t, b in keys]
    if end:
        lines.append(f"BERLIN-LST-HOSTKEY-{nonce}-END")
    return "\n".join(lines) + "\n"


def _keysfile(env: Env, names: list[str]) -> Path:
    p = env.tmp / "keys.txt"
    p.write_text("".join(f"{env.keys[n][0]} {env.keys[n][1]}\n" for n in names))
    return p


def test_parse_serial_extracts_only_nonce_tagged_public_keys(fx: Env) -> None:
    ks = [fx.keys["k1"], fx.keys["k2"]]
    r = _lib(fx, "hostkey_parse_serial abc123", _serial("abc123", ks))
    assert r.returncode == 0
    assert r.stdout.splitlines() == [f"{t} {b}" for t, b in ks]


def test_parse_serial_rejects_malformed_line(fx: Env) -> None:
    bad = "BERLIN-LST-HOSTKEY-abc123 ssh-ed25519 not*base64\nBERLIN-LST-HOSTKEY-abc123-END\n"
    r = _lib(fx, "hostkey_parse_serial abc123", bad)
    assert r.returncode == 2
    assert "malformed" in r.stderr
    assert "not*base64" not in r.stderr


def test_parse_serial_incomplete_without_end_marker(fx: Env) -> None:
    r = _lib(fx, "hostkey_parse_serial abc123", _serial("abc123", [fx.keys["k1"]], end=False))
    assert r.returncode == 3


def test_parse_serial_rejects_unknown_key_type(fx: Env) -> None:
    bad = "BERLIN-LST-HOSTKEY-abc123 ssh-dss AAAAB3NzaC1kc3M\nBERLIN-LST-HOSTKEY-abc123-END\n"
    assert _lib(fx, "hostkey_parse_serial abc123", bad).returncode == 2


def test_merge_known_hosts_appends_new_alias_with_mode_600(fx: Env) -> None:
    kf = _keysfile(fx, ["k1"])
    r = _lib(fx, f'hostkey_merge_known_hosts "{fx.kh}" {ALIAS} "{kf}"')
    assert r.returncode == 0, r.stderr
    t, b = fx.keys["k1"]
    assert f"{ALIAS} {t} {b}" in fx.kh.read_text()
    assert stat.S_IMODE(fx.kh.stat().st_mode) == 0o600
    found = subprocess.run(  # noqa: S603
        [SSH_KEYGEN or "", "-F", ALIAS, "-f", str(fx.kh)], capture_output=True, text=True
    )
    assert found.returncode == 0


def test_merge_known_hosts_idempotent_when_same_key(fx: Env) -> None:
    kf = _keysfile(fx, ["k1"])
    cmd = f'hostkey_merge_known_hosts "{fx.kh}" {ALIAS} "{kf}"'
    assert _lib(fx, cmd).returncode == 0
    before = fx.kh.read_bytes()
    r = _lib(fx, cmd)
    assert r.returncode == 0
    assert fx.kh.read_bytes() == before
    assert list(fx.kh.parent.glob("*.bak.*")) == []


def test_merge_known_hosts_rejects_changed_key(fx: Env) -> None:
    t3, b3 = fx.keys["k3"]
    fx.kh.parent.mkdir(mode=0o700)
    fx.kh.write_text(f"{ALIAS} {t3} {b3}\n")
    before = fx.kh.read_bytes()
    r = _lib(fx, f'hostkey_merge_known_hosts "{fx.kh}" {ALIAS} "{_keysfile(fx, ["k1"])}"')
    assert r.returncode == 4
    assert "changed host key" in r.stderr
    assert fx.kh.read_bytes() == before


def test_merge_known_hosts_ignores_other_aliases(fx: Env) -> None:
    t3, b3 = fx.keys["k3"]
    fx.kh.parent.mkdir(mode=0o700)
    fx.kh.write_text(f"compute.111 {t3} {b3}\n")
    r = _lib(fx, f'hostkey_merge_known_hosts "{fx.kh}" {ALIAS} "{_keysfile(fx, ["k1"])}"')
    assert r.returncode == 0, r.stderr
    text = fx.kh.read_text()
    assert f"compute.111 {t3} {b3}" in text
    assert ALIAS in text


def test_dry_run_does_not_mutate(fx: Env) -> None:
    r = fx.run()
    assert r.returncode == 0, r.stderr
    assert "[dry-run]" in r.stdout
    assert set(fx.verbs()) == {"describe"}
    assert not (fx.tmp / "ssh.called").exists()
    assert not (fx.home / ".ssh").exists()


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("id", "1", "Instance ID mismatch"),
        ("sa", "other@example.com", "service account mismatch"),
        ("protection", False, "deletion protection"),
    ],
)
def test_pin_mismatch_fails_before_any_mutation(
    fx: Env, field: str, value: object, needle: str
) -> None:
    fx.state[field] = value
    fx.save()
    r = fx.run("--execute")
    assert r.returncode != 0
    assert needle in r.stderr
    assert not set(fx.verbs()) & set(MUTATING)


def test_execute_refuses_running_vm(fx: Env) -> None:
    fx.state["status"] = "RUNNING"
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 1
    assert "RUNNING" in r.stderr
    assert not set(fx.verbs()) & set(MUTATING)


def test_execute_refuses_startup_script_url(fx: Env) -> None:
    fx.state["items"] = [{"key": "startup-script-url", "value": "gs://x/y"}]
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 1
    assert not set(fx.verbs()) & set(MUTATING)


def test_execute_happy_path_pins_key_and_removes_temp_metadata(fx: Env) -> None:
    r = fx.run("--execute")
    assert r.returncode == 0, r.stderr
    text = fx.kh.read_text()
    for name in ("k1", "k2"):
        t, b = fx.keys[name]
        assert f"{ALIAS} {t} {b}" in text
    assert fx.startup_value() is None
    assert "SHA256:" in r.stdout
    assert "BEGIN OPENSSH PRIVATE KEY" not in r.stdout + r.stderr
    assert "NOISE-PREFIX-LINE" not in r.stdout + r.stderr
    assert not (fx.tmp / "ssh.called").exists()


def test_execute_preserves_existing_startup_script(fx: Env) -> None:
    fx.state["items"] = [{"key": "startup-script", "value": ORIGINAL}]
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 0, r.stderr
    assert fx.startup_value() == ORIGINAL
    backups = fx.home / ".local" / "state" / "berlin-lst" / "hostkey-bootstrap"
    assert list(backups.glob("*.bak")) == []


def test_execute_restores_metadata_on_serial_timeout(fx: Env) -> None:
    fx.state["items"] = [{"key": "startup-script", "value": ORIGINAL}]
    fx.state["serial_mode"] = "none"
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 5
    assert fx.startup_value() == ORIGINAL
    assert not fx.kh.exists()


def test_execute_restores_metadata_on_malformed_key(fx: Env) -> None:
    fx.state["serial_mode"] = "malformed"
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 2
    assert fx.startup_value() is None
    assert not fx.kh.exists()


def test_execute_changed_key_fails_after_metadata_cleanup(fx: Env) -> None:
    t3, b3 = fx.keys["k3"]
    fx.kh.parent.mkdir(mode=0o700)
    fx.kh.write_text(f"{ALIAS} {t3} {b3}\n")
    before = fx.kh.read_bytes()
    r = fx.run("--execute")
    assert r.returncode == 4
    assert "changed host key" in r.stderr
    assert fx.startup_value() is None
    assert fx.kh.read_bytes() == before


def test_execute_restore_failure_is_hard_and_keeps_backup(fx: Env) -> None:
    fx.state["items"] = [{"key": "startup-script", "value": ORIGINAL}]
    fx.state["fail_restore"] = True
    fx.save()
    r = fx.run("--execute")
    assert r.returncode == 3
    assert "restore NOT verified" in r.stderr
    backups = list((fx.home / ".local/state/berlin-lst/hostkey-bootstrap").glob("*.bak"))
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert backups[0].read_text() == ORIGINAL


def test_execute_start_failure_still_cleans_metadata(fx: Env) -> None:
    fx.state["fail_start"] = True
    fx.save()
    r = fx.run("--execute")
    assert r.returncode != 0
    assert fx.startup_value() is None


def test_scripts_never_use_blind_trust() -> None:
    for path in (BOOT, LIB):
        text = path.read_text()
        for banned in ("ssh-keyscan", "StrictHostKeyChecking=no", "accept-new"):
            assert banned not in text, (path.name, banned)
    assert "StrictHostKeyChecking=yes" in (VM_DIR / "ssh-vm.sh").read_text()
    assert re.search(r"^REPO_ROOT=", BOOT.read_text(), re.M) is None
    assert os.access(BOOT, os.X_OK)
