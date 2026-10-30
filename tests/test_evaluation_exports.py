"""Offline tests for the per-patch evaluation export (Parquet sidecar)."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from berlin_lst_downscaling.data.features.contracts import FEATURE_SCHEMA_VERSION_V6
from berlin_lst_downscaling.modeling import baseline, patches
from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling.contracts import REAL_PATCH_CELLS, REAL_PATCH_PX
from berlin_lst_downscaling.modeling.evaluation_exports import (
    PATCH_EXPORT_METADATA_KEY,
    PATCH_EXPORT_SCHEMA,
    PATCH_EXPORT_SCHEMA_VERSION,
    EvaluationIdentity,
    patch_export_filename,
    score_batch_patches,
    write_patch_export,
)
from berlin_lst_downscaling.modeling.metrics import masked_abs_error_sums, masked_ssim_stats
from berlin_lst_downscaling.modeling.patches import (
    PatchRef,
    RealPatchReader,
    RealSample,
    RealSampleMeta,
    RealSourceConfig,
)
from berlin_lst_downscaling.modeling.random_forest import (
    ForestSpec,
    collect_eval_split,
    collect_train_pool,
    predict_10m,
    resolve_stage_selection,
    score_stage,
    stage_matrix,
)

N_V6 = 22
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"
_RUN_SOURCE = Path(run_module.__file__).read_text(encoding="utf-8")


def _row(patch_id: str, scene_id: str = "S1", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "patch_id": patch_id,
        "scene_id": scene_id,
        "scene_date": None,
        "split": "validation",
        "abs_error_sum": 12.5,
        "valid_cells": 240,
        "ssim_sum": 3.0,
        "ssim_windows": 4,
        "filled_feature_pixels": 0,
    }
    row.update(overrides)
    return row


def _identity(**overrides: object) -> EvaluationIdentity:
    kwargs: dict[str, object] = {
        "arm": "L3",
        "seed": 7,
        "run_id": "r1",
        "checkpoint": "ckpt/a.ckpt",
    }
    kwargs.update(overrides)
    return EvaluationIdentity(**kwargs)  # type: ignore[arg-type]


# --- writer ----------------------------------------------------------------


def test_no_torch_import_at_module_level() -> None:
    tree = ast.parse(Path(sys.modules[write_patch_export.__module__].__file__ or "").read_text())
    top = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    names = [a.name for n in top for a in n.names] + [
        n.module or "" for n in top if isinstance(n, ast.ImportFrom)
    ]
    assert not any(name.split(".")[0] in {"torch", "lightning"} for name in names)
    assert not any("modeling.run" in name or "metrics" in name for name in names)


def test_schema_and_identity_round_trip(tmp_path: Path) -> None:
    rows = [
        _row("S2:b", "S2", scene_date="2024-06-01"),
        _row("S1:z", "S1"),
        _row("S1:a", "S1", filled_feature_pixels=None, extra_field="ignored"),
    ]
    path = write_patch_export(
        rows,
        tmp_path / patch_export_filename("validation"),
        identity=_identity(),
        split="validation",
    )
    assert path.name == "eval_validation_patches.parquet"
    assert patch_export_filename("test", prefix="eval_stage2") == "eval_stage2_test_patches.parquet"
    table = pq.read_table(path)
    assert table.schema.remove_metadata().equals(PATCH_EXPORT_SCHEMA.remove_metadata())
    assert table.column_names == PATCH_EXPORT_SCHEMA.names
    assert table.column("patch_id").to_pylist() == ["S1:a", "S1:z", "S2:b"]
    assert table.column("scene_date").to_pylist() == [None, None, "2024-06-01"]
    assert table.column("filled_feature_pixels").to_pylist() == [None, 0, 0]
    assert set(table.column("schema_version").to_pylist()) == {PATCH_EXPORT_SCHEMA_VERSION}
    footer = json.loads(pq.read_schema(path).metadata[PATCH_EXPORT_METADATA_KEY])
    assert footer == {
        "schema_version": 1,
        "split": "validation",
        "arm": "L3",
        "seed": 7,
        "run_id": "r1",
        "checkpoint": "ckpt/a.ckpt",
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_null_seed_and_checkpoint(tmp_path: Path) -> None:
    path = write_patch_export(
        [_row("S1:a")],
        tmp_path / "x.parquet",
        identity=_identity(seed=None, checkpoint=None),
        split="validation",
    )
    table = pq.read_table(path)
    assert table.column("seed").to_pylist() == [None]
    assert table.column("checkpoint").to_pylist() == [None]
    footer = json.loads(pq.read_schema(path).metadata[PATCH_EXPORT_METADATA_KEY])
    assert footer["seed"] is None and footer["checkpoint"] is None


@pytest.mark.parametrize(
    ("records", "split", "identity"),
    [
        ([_row("a"), _row("a")], "validation", _identity()),
        ([_row("a", abs_error_sum=float("nan"))], "validation", _identity()),
        ([_row("a", ssim_sum=float("inf"))], "validation", _identity()),
        ([_row("a", valid_cells=-1)], "validation", _identity()),
        ([_row("a", ssim_windows=-2)], "validation", _identity()),
        ([_row("a", valid_cells=240.5)], "validation", _identity()),
        ([_row("a", ssim_windows=True)], "validation", _identity()),
        ([_row("a", valid_cells=0)], "validation", _identity()),
        ([_row("a", ssim_windows=0, ssim_sum=0.5)], "validation", _identity()),
        ([_row("a", scene_id="")], "validation", _identity()),
        ([_row("a")], "holdout", _identity()),
        ([_row("a", split="test")], "validation", _identity()),
        ([_row("a")], "validation", _identity(arm="")),
        ([_row("a")], "validation", _identity(run_id=" ")),
        ([{k: v for k, v in _row("a").items() if k != "ssim_sum"}], "validation", _identity()),
        ([], "validation", _identity()),
    ],
)
def test_writer_rejects_invalid(
    tmp_path: Path, records: list[dict[str, object]], split: str, identity: EvaluationIdentity
) -> None:
    target = tmp_path / "out.parquet"
    with pytest.raises(ValueError):
        write_patch_export(records, target, identity=identity, split=split)
    assert not target.exists()
    assert not list(tmp_path.glob("*"))


def test_writer_accepts_zero_ssim_windows_with_zero_sum(tmp_path: Path) -> None:
    write_patch_export(
        [_row("a", ssim_windows=0, ssim_sum=0.0)],
        tmp_path / "ok.parquet",
        identity=_identity(),
        split="validation",
    )


def test_writer_failure_removes_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(pq, "write_table", _boom)
    with pytest.raises(OSError):
        write_patch_export(
            [_row("a")], tmp_path / "x.parquet", identity=_identity(), split="validation"
        )
    assert list(tmp_path.iterdir()) == []


# --- batch scoring ---------------------------------------------------------


def _meta(patch_id: str, scene_id: str, split: str = "validation", date_: str | None = None):
    return RealSampleMeta(
        patch_id=patch_id,
        scene_id=scene_id,
        split=split,
        year=2024,
        row=0,
        col=0,
        filled_feature_pixels=3,
        scene_date=date_,
    )


def _tensors(batch: int = 2):
    g = torch.Generator().manual_seed(0)
    cells = REAL_PATCH_CELLS
    target = 290.0 + torch.randn(batch, 1, cells, cells, generator=g)
    pred = target + torch.randn(batch, 1, cells, cells, generator=g)
    mask = torch.ones(batch, 1, cells, cells, dtype=torch.bool)
    mask[:, :, 0, :] = False
    return pred, target, mask


def test_batch_rows_preserve_metadata_and_match_metrics() -> None:
    pred, target, mask = _tensors()
    metas = [_meta("A:1", "A", date_="2024-05-01"), _meta("B:1", "B")]
    rows = score_batch_patches(pred, target, mask, metas)
    assert [r["patch_id"] for r in rows] == ["A:1", "B:1"]
    assert [r["scene_date"] for r in rows] == ["2024-05-01", None]
    assert all(r["split"] == "validation" and r["filled_feature_pixels"] == 3 for r in rows)
    for i, row in enumerate(rows):
        sl = slice(i, i + 1)
        err, cnt = masked_abs_error_sums(pred[sl], target[sl], mask[sl])
        s_sum, s_cnt = masked_ssim_stats(pred[sl], target[sl], mask[sl], clamp_predictions=False)
        assert row["abs_error_sum"] == pytest.approx(float(err))
        assert row["valid_cells"] == int(cnt) == 240
        assert row["ssim_sum"] == pytest.approx(float(s_sum))
        assert row["ssim_windows"] == int(s_cnt)
    total_err, total_cnt = masked_abs_error_sums(pred, target, mask)
    assert sum(r["abs_error_sum"] for r in rows) == pytest.approx(float(total_err), rel=1e-5)
    assert sum(r["valid_cells"] for r in rows) == int(total_cnt)


def test_batch_rows_reject_metadata_length_mismatch() -> None:
    pred, target, mask = _tensors()
    with pytest.raises(ValueError, match="metadata length"):
        score_batch_patches(pred, target, mask, [_meta("A:1", "A")])


# --- scene_date plumbing ---------------------------------------------------


def _table(rows: list[dict[str, object]]) -> pa.Table:
    keys = sorted({k for r in rows for k in r})
    return pa.table({k: [r.get(k) for r in rows] for k in keys})


def _patch_index_mocks(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    monkeypatch.setattr(
        patches,
        "patch_index_fingerprints",
        lambda _r: {
            "source_policy_hash": "h",
            "patches_total": len(rows),
        },
    )
    monkeypatch.setattr(patches, "load_scaler", lambda _r: SimpleNamespace(policy_hash="h"))
    monkeypatch.setattr(patches, "_read_parquet", lambda _u: _table(rows))
    monkeypatch.setattr(patches, "_read_manifest_rows", lambda _r: {})
    monkeypatch.setattr(patches, "_verify_manifest_row", lambda _m, _r: None)


def _index_row(scene_id: str, **extra: object) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "year": 2024,
        "split": "validation",
        "s2_scene_id": "S2",
        "patch_id": f"{scene_id}:c",
        "row": 0,
        "col": 0,
        "n_eligible": 250,
        "eligibility_mask": "m",
        **extra,
    }


def test_patch_ref_optional_date_reaches_sample_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _index_row("A", scene_date="2024-05-01"),
        _index_row("B", acquisition_date=date(2024, 6, 2)),
        _index_row("C", datetime=datetime(2024, 7, 3, 10, 30)),
        _index_row("D", scene_date="2024-08-04T10:00:00Z"),
        _index_row("E"),
    ]
    _patch_index_mocks(monkeypatch, rows)
    cfg = RealSourceConfig("pi", "tr", "ft", "ard")
    refs = patches.load_patch_refs(cfg)
    assert [r.scene_date for r in refs] == [
        "2024-05-01",
        "2024-06-02",
        "2024-07-03",
        "2024-08-04",
        None,
    ]

    reader = object.__new__(RealPatchReader)
    reader.exclusions = Counter()
    stub_prior = SimpleNamespace(
        window=lambda *, row, col: np.full((REAL_PATCH_PX, REAL_PATCH_PX), 290.0, np.float32)
    )
    stub_tm = SimpleNamespace(
        target=np.zeros((REAL_PATCH_CELLS, REAL_PATCH_CELLS), np.float32),
        mask=np.ones((REAL_PATCH_CELLS, REAL_PATCH_CELLS), bool),
    )
    monkeypatch.setattr(reader, "_prior_for", lambda _s: stub_prior, raising=False)
    monkeypatch.setattr(
        reader, "_read_target_and_mask", lambda _r, _p: (stub_tm, ""), raising=False
    )
    monkeypatch.setattr(
        reader,
        "_read_features",
        lambda _r: (np.zeros((N_V6, REAL_PATCH_PX, REAL_PATCH_PX), np.float32), 0),
        raising=False,
    )
    assert reader.read_patch(refs[0]).meta.scene_date == "2024-05-01"  # type: ignore[union-attr]
    assert reader.read_patch(refs[-1]).meta.scene_date is None  # type: ignore[union-attr]


def test_unparseable_index_date_raises() -> None:
    with pytest.raises(ValueError):
        patches._index_scene_date({"scene_date": "not-a-date"})


# --- baseline --------------------------------------------------------------


def _sample(patch_id: str, scene_id: str, seed: int, date_: str | None = None) -> RealSample:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N_V6, REAL_PATCH_PX, REAL_PATCH_PX)).astype(np.float32)
    prior = (290.0 + rng.normal(size=(1, REAL_PATCH_PX, REAL_PATCH_PX))).astype(np.float32)
    target = (290.0 + rng.normal(size=(1, REAL_PATCH_CELLS, REAL_PATCH_CELLS))).astype(np.float32)
    mask = np.ones((1, REAL_PATCH_CELLS, REAL_PATCH_CELLS), dtype=bool)
    mask[:, 0, :] = False
    meta = RealSampleMeta(
        patch_id=patch_id,
        scene_id=scene_id,
        split="validation",
        year=2024,
        row=0,
        col=0,
        filled_feature_pixels=1,
        scene_date=date_,
    )
    return RealSample(meta, features, prior, target, mask)


def test_baseline_record_gains_scene_metadata_only() -> None:
    sample = _sample("A:1", "A", 0, "2024-05-01")
    record = baseline._score_patch(sample, aggregation="arithmetic")  # type: ignore[arg-type]
    assert record.scene_id == "A" and record.scene_date == "2024-05-01"
    legacy = {
        "patch_id",
        "split",
        "valid_cells",
        "abs_error_sum",
        "ssim_sum",
        "ssim_windows",
        "filled_feature_pixels",
    }
    from dataclasses import asdict

    payload = asdict(record)
    assert legacy <= set(payload)
    assert set(payload) - legacy == {"scene_id", "scene_date"}
    assert record.valid_cells == 240 and record.filled_feature_pixels == 1
    summary = baseline.SplitSummary(1, 1, 240, record.abs_error_sum, record.ssim_sum, 1, 1)
    assert summary.mae == pytest.approx(record.abs_error_sum / 240)
    assert not hasattr(summary, "scene_id")


# --- random forest ---------------------------------------------------------


class _FakeReader:
    def __init__(self, samples: dict[str, RealSample], skip: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.samples = samples
        self.skip = skip
        self.exclusions: Counter[str] = Counter()

    def read_patch(self, ref: PatchRef) -> RealSample | None:
        if ref.patch_id in self.skip:
            self.exclusions["skipped"] += 1
            return None
        return self.samples[ref.patch_id]


def _rf_fixture():
    ids = [("A:1", "A", "2024-05-01"), ("B:1", "B", None), ("C:1", "C", "2024-07-01")]
    samples = {pid: _sample(pid, sid, i, d) for i, (pid, sid, d) in enumerate(ids)}
    refs = [
        PatchRef(pid, sid, "S2", "validation", 2024, i * REAL_PATCH_CELLS, 0, 240, "m")
        for i, (pid, sid, _d) in enumerate(ids)
    ]
    return samples, refs


def test_rf_collect_eval_split_keeps_alignment_after_exclusion(tmp_path: Path) -> None:
    samples, refs = _rf_fixture()
    reader = _FakeReader(samples, skip={"B:1"})
    cache = collect_eval_split(
        reader,  # type: ignore[arg-type]
        refs,
        split="validation",
        cache_root=str(tmp_path),
    )
    assert cache.patch_ids == ["A:1", "C:1"]
    assert cache.scene_ids == ["A", "C"]
    assert cache.scene_dates == ["2024-05-01", "2024-07-01"]
    assert cache.requested_patches == 3


def test_rf_score_stage_scene_metadata_and_metric_values(tmp_path: Path) -> None:
    samples, refs = _rf_fixture()
    reader = _FakeReader(samples)
    cache = collect_eval_split(
        reader,  # type: ignore[arg-type]
        refs,
        split="validation",
        cache_root=str(tmp_path),
    )
    pool = collect_train_pool(
        reader,  # type: ignore[arg-type]
        refs,
        max_cells=1000,
        seed=0,
        n_feature_channels=N_V6,
    )
    stage = resolve_stage_selection(1, FEATURE_SCHEMA_VERSION_V6)
    matrix = stage_matrix(pool.features, pool.prior_100m, stage, n_feature_channels=N_V6)
    model = ForestSpec(n_estimators=2, max_depth=3, n_jobs=1).build().fit(matrix, pool.residual_k)
    records = score_stage(cache, model, stage, n_feature_channels=N_V6)
    assert [r.scene_id for r in records] == ["A", "B", "C"]
    assert [r.scene_date for r in records] == ["2024-05-01", None, "2024-07-01"]
    for i, record in enumerate(records):
        features = np.load(cache.feature_files[i])
        residual = predict_10m(model, features, cache.prior_100m[i], stage, n_feature_channels=N_V6)
        prior_10m = np.kron(cache.prior_100m[i], np.ones((10, 10), np.float32))
        pooled = (prior_10m + residual).reshape(16, 10, 16, 10).mean(axis=(1, 3))
        mask = cache.mask_100m[i]
        expected = np.abs(pooled - cache.target_100m[i])[mask].sum()
        assert record.valid_cells == int(mask.sum())
        assert record.abs_error_sum == pytest.approx(float(expected), rel=1e-4)
    out = write_patch_export(
        [__import__("dataclasses").asdict(r) for r in records],
        tmp_path / "rf.parquet",
        identity=_identity(arm="random_forest_stage1", seed=0, checkpoint="m.skops"),
        split="validation",
    )
    assert pq.read_table(out).num_rows == 3


# --- neural lifecycle ------------------------------------------------------


@pytest.fixture(scope="module")
def compose_modeling() -> Callable[..., DictConfig]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):

        def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
            return compose(config_name=name, overrides=overrides or [])

        yield _compose
    GlobalHydra.instance().clear()


def _smoke_cfg(
    compose_modeling: Callable[..., DictConfig], tmp_path: Path, extra: list[str]
) -> DictConfig:
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            compose_modeling(
                "contract_smoke", ["data.batch_size=1", "trainer.max_epochs=1", *extra]
            ),
            resolve=True,
        )
    )
    cfg.output_root = str(tmp_path / "data" / "runs" / "export")
    cfg.ablation_release = "v6"
    cfg.stage1_full = True
    return cfg


@pytest.fixture
def _no_run_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "enforce_run_lock", lambda cfg, identity: None)
    monkeypatch.setattr(run_module, "contract_invariants", lambda cfg: None)


@pytest.mark.usefixtures("_no_run_lock")
def test_validation_reload_writes_one_export_and_test_is_never_scored(
    compose_modeling: Callable[..., DictConfig],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("_score_test_once must not run")

    monkeypatch.setattr(run_module, "_score_test_once", _forbidden)
    cfg = _smoke_cfg(compose_modeling, tmp_path, ["evaluate_test=false"])
    result = run_module.run_contract_synthetic_training(cfg, run_id="exp-1")
    out = Path(cfg.output_root)
    exports = sorted(p.name for p in out.glob("*.parquet"))
    assert exports == ["eval_validation_patches.parquet"]
    table = pq.read_table(out / exports[0])
    summary = json.loads((out / "full_summary.json").read_text())
    assert table.num_rows > 0
    assert sum(table.column("abs_error_sum").to_pylist()) == pytest.approx(
        summary["validation_abs_error_sum"], rel=1e-4
    )
    assert sum(table.column("valid_cells").to_pylist()) == int(summary["validation_valid_cells"])
    assert set(table.column("split").to_pylist()) == {"validation"}
    footer = json.loads(pq.read_schema(out / exports[0]).metadata[PATCH_EXPORT_METADATA_KEY])
    assert footer["run_id"] == "exp-1" and footer["seed"] == int(cfg.seed)
    assert footer["checkpoint"] == Path(result.best_checkpoint).relative_to(out).as_posix()
    assert not (out / "eval_test_patches.parquet").exists()


# --- structural guards on run.py ------------------------------------------


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_score_test_once_is_gated_by_full_and_evaluate_test() -> None:
    tree = ast.parse(_RUN_SOURCE)
    lifecycle = _function(tree, "_fit_contract_lifecycle")
    assert len(_calls(lifecycle, "_score_test_once")) == 1
    gated = [
        n
        for n in ast.walk(lifecycle)
        if isinstance(n, ast.If)
        and {"is_full", "evaluate_test"} <= _names(n.test)
        and any(_calls(stmt, "_score_test_once") for stmt in n.body)
    ]
    assert len(gated) == 1
    assert not _calls(tree, "_score_test_once") or len(_calls(tree, "_score_test_once")) == 1


def test_test_export_is_written_only_inside_score_test_once() -> None:
    tree = ast.parse(_RUN_SOURCE)
    scorer = _function(tree, "_score_test_once")
    test_writes = [
        c
        for c in _calls(tree, "write_patch_export")
        if any(
            kw.arg == "split" and getattr(kw.value, "value", None) == "test" for kw in c.keywords
        )
    ]
    assert len(test_writes) == 1
    assert test_writes[0] in _calls(scorer, "write_patch_export")
    for other in _calls(tree, "write_patch_export"):
        if other is not test_writes[0]:
            assert other not in _calls(scorer, "write_patch_export")
