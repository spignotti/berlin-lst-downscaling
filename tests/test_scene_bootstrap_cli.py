"""Scene bootstrap CLI: synthetic, offline tests."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from test_scene_bootstrap import (
    effect_fn,
    export_table,
    make_world,
    write_export,
)

from berlin_lst_downscaling.analysis import scene_bootstrap as sb

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
    assert not hasattr(args, "i_know_test_is_frozen")


@pytest.mark.parametrize(
    "extra", [["--split", "test"], ["--split", "test", "--i-know-test-is-frozen"]]
)
def test_test_split_rejected_before_input_access(
    cli: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("input accessed")

    monkeypatch.setattr(cli, "load_patch_exports", boom)
    monkeypatch.setattr("builtins.open", boom)
    names = ["nope1", "nope2", "nope3"]
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(names, names, tmp_path / "o", *extra))
    assert exc.value.code == 2


def test_validation_rejects_test_export(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = make_world(8)
    fn = effect_fn(dict.fromkeys(range(8), -1.0))
    files = {"A": [], "B": []}
    for arm in files:
        for seed in (0, 1, 2):
            t = export_table(arm, seed, "validation", world, fn)
            path = write_export(tmp_path / f"{arm}{seed}.parquet", t, footer_split="test")
            files[arm].append(str(path))

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("row data must not be read")

    monkeypatch.setattr(sb.pq, "read_table", boom)
    rc = cli.main(_argv(files["A"], files["B"], tmp_path / "out"))
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


def test_ssim_cli_uses_main_fixed_range_metric(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from test_scene_bootstrap import _writer_arm

    world = make_world(8)
    a = _writer_arm(tmp_path, "A", world, main=0.6, l250=0.1, lpatch=0.05)
    b = _writer_arm(tmp_path, "B", world, main=0.4, l250=0.9, lpatch=None)
    rc = cli.main(
        _argv([str(p) for p in a], [str(p) for p in b], tmp_path / "ssim", "--metric", "ssim_100m")
    )
    assert rc == 0
    doc = json.loads((tmp_path / "ssim.json").read_text())
    assert doc["metric"] == "ssim_100m"
    assert doc["point_estimate"] == pytest.approx(0.2)


def test_cli_single_reference_file_for_arm_b(
    cli: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    import pyarrow as pa
    from test_scene_bootstrap import SCHEMA

    world = make_world(8)
    fn = effect_fn(dict.fromkeys(range(8), -1.0))
    a = [
        str(write_export(tmp_path / f"A{s}.parquet", export_table("A", s, "validation", world, fn)))
        for s in (0, 1, 2)
    ]
    ref = export_table("naive_prior_expand", 0, "validation", world, fn)
    ref = ref.set_column(
        ref.schema.get_field_index("seed"),
        SCHEMA.field("seed"),
        pa.array([None] * ref.num_rows, type=pa.int64()),
    )
    b = [str(write_export(tmp_path / "ref.parquet", ref))]
    rc = cli.main(_argv(a, b, tmp_path / "ref_out"))
    assert rc == 0
    doc = json.loads((tmp_path / "ref_out.json").read_text())
    assert doc["replicated_reference"] is True
    assert "Replicated reference:" in (tmp_path / "ref_out.md").read_text()
    capsys.readouterr()


def test_cli_two_files_per_arm_exit_code_2(cli: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(["a", "b"], ["c", "d", "e"], tmp_path / "o"))
    assert exc.value.code == 2


def test_cli_metric_choices_include_l250(cli: ModuleType) -> None:
    args = cli.build_parser().parse_args(
        ["--arm-a", "a", "--arm-b", "b", "--output-prefix", "o", "--metric", "ssim_100m_L250"]
    )
    assert args.metric == "ssim_100m_L250"
