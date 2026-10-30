"""Scene bootstrap CLI: synthetic, offline tests."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from test_scene_bootstrap import (
    effect_fn,
    export_table,
    make_world,
    write_export,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "analyze_scene_bootstrap.py"
_MODULE = _ROOT / "src" / "berlin_lst_downscaling" / "analysis" / "scene_bootstrap.py"


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_scene_bootstrap", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_arms(tmp_path: Path, split: str) -> tuple[list[str], list[str]]:
    world = make_world(8)
    fn = effect_fn(dict.fromkeys(range(8), -1.0))
    out: dict[str, list[str]] = {}
    for arm in ("A", "B"):
        files = []
        for seed in (0, 1, 2):
            t = export_table(arm, seed, split, world, fn)
            files.append(str(write_export(tmp_path / f"{arm}{seed}_{split}.parquet", t)))
        out[arm] = files
    return out["A"], out["B"]


def _argv(a: list[str], b: list[str], prefix: Path, *extra: str) -> list[str]:
    return [
        "--arm-a",
        *a,
        "--arm-b",
        *b,
        "--output-prefix",
        str(prefix),
        "--n-bootstrap",
        "50",
        *extra,
    ]


def test_defaults_to_validation(cli: ModuleType) -> None:
    args = cli.build_parser().parse_args(
        ["--arm-a", "a", "b", "c", "--arm-b", "d", "e", "f", "--output-prefix", "o"]
    )
    assert args.split == "validation"
    assert args.metric == "mae_100m"
    assert args.n_bootstrap == 10_000
    assert args.rng_seed == 0
    assert not args.i_know_test_is_frozen


def test_test_split_without_ack_fails_before_input_access(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("input accessed")

    monkeypatch.setattr(cli, "load_patch_exports", boom)
    monkeypatch.setattr("builtins.open", boom)
    names = ["nope1", "nope2", "nope3"]
    rc = cli.main(_argv(names, names, tmp_path / "o", "--split", "test"))
    assert rc != 0
    assert "--i-know-test-is-frozen" in capsys.readouterr().err


def test_acknowledged_test_split_reads_exports(cli: ModuleType, tmp_path: Path) -> None:
    a, b = _write_arms(tmp_path, "test")
    rc = cli.main(_argv(a, b, tmp_path / "out", "--split", "test", "--i-know-test-is-frozen"))
    assert rc == 0
    assert json.loads((tmp_path / "out.json").read_text())["split"] == "test"


def test_validation_rejects_test_export(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _write_arms(tmp_path, "test")
    rc = cli.main(_argv(a, b, tmp_path / "out"))
    assert rc == 1
    assert "footer split" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_writes_json_and_markdown(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _write_arms(tmp_path, "validation")
    rc = cli.main(_argv(a, b, tmp_path / "sub" / "res"))
    assert rc == 0
    assert (tmp_path / "sub" / "res.json").is_file()
    md = (tmp_path / "sub" / "res.md").read_text()
    assert "Paired scene bootstrap" in md
    assert "Paired scene bootstrap" in capsys.readouterr().out


def test_missing_file_is_concise_error(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = [str(tmp_path / f"m{i}.parquet") for i in range(3)]
    assert cli.main(_argv(names, names, tmp_path / "o")) == 1
    assert "not found" in capsys.readouterr().err


def test_output_input_collision_rejected(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _write_arms(tmp_path, "validation")
    clash = Path(a[0]).with_suffix("")
    # input named "<prefix>.json" would be overwritten
    target = tmp_path / "clash.json"
    target.write_bytes(Path(a[0]).read_bytes())
    a2 = [str(target), a[1], a[2]]
    before = target.read_bytes()
    rc = cli.main(_argv(a2, b, tmp_path / "clash"))
    assert rc != 0
    assert "collides" in capsys.readouterr().err
    assert target.read_bytes() == before
    del clash


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", [_SCRIPT, _MODULE])
def test_no_forbidden_imports(path: Path) -> None:
    forbidden = ("torch", "lightning", "subprocess", "scripts", "berlin_lst_downscaling.modeling",
                 "berlin_lst_downscaling.data", "google", "gcsfs", "fsspec")  # fmt: skip
    for name in _imports(path):
        assert not any(name == f or name.startswith(f + ".") for f in forbidden), name
