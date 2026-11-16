"""Per-cell validation export and its v6 matrix job: offline tests (mocks only)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import DictConfig
from test_v6_val_rescore import (
    SSIM,
    _expected_mae,
    _full_cache,
    _loader,
    _sample,
    _samples,
    _StubTask,
)

from berlin_lst_downscaling.analysis import spatial_error as se
from berlin_lst_downscaling.modeling import cell_exports as ce
from berlin_lst_downscaling.modeling import run as run_module
from berlin_lst_downscaling.modeling import v6_val_cell_export as job
from berlin_lst_downscaling.modeling import v6_val_rescore as rescore
from berlin_lst_downscaling.modeling.channels import selection_from_config
from berlin_lst_downscaling.modeling.contracts import REAL_PATCH_CELLS, RealSampleMeta
from berlin_lst_downscaling.modeling.evaluation_exports import (
    EvaluationIdentity,
    score_batch_patches,
)
from berlin_lst_downscaling.modeling.metrics import EVAL_PREDICTION_CLAMP_K
from berlin_lst_downscaling.modeling.patches import RealSample
from berlin_lst_downscaling.modeling.real_task import RealPatchDataModule
from berlin_lst_downscaling.modeling.ssim_range import SsimDataRange

IDENTITY = EvaluationIdentity(arm="L4", seed=0, run_id="run-a", checkpoint="gs://x/best.ckpt")
LABELS = ("v6m-070b33c-l1-seed0-ep40-cos", "v6m-070b33c-l4-seed2-ep40-cos")


def _meta(patch_id: str, row: int, col: int, split: str = "validation") -> RealSampleMeta:
    return RealSampleMeta(
        patch_id=patch_id,
        scene_id="LC08_X",
        split=split,
        year=2024,
        row=row,
        col=col,
        filled_feature_pixels=0,
    )


def _batch(n: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    shape = (n, 1, REAL_PATCH_CELLS, REAL_PATCH_CELLS)
    target = 300.0 + torch.randn(shape, generator=gen)
    prediction = target + torch.randn(shape, generator=gen)
    mask = torch.ones(shape, dtype=torch.bool)
    mask[:, :, 0, :] = False
    mask[0, 0, 5, 7] = False
    return prediction, target, mask


# ── columns ───────────────────────────────────────────────────────────


def test_columns_carry_global_cells_of_the_mask() -> None:
    prediction, target, mask = _batch()
    metadata = [_meta("p0", 100, 200), _meta("p1", 100, 216)]
    columns = ce.cell_columns_from_batch(prediction, target, mask, metadata)
    assert set(columns) == set(ce.CELL_EXPORT_COLUMNS)
    assert len(columns["row"]) == int(mask.sum())
    first = columns["patch_id"] == "p0"
    assert int(first.sum()) == int(mask[0].sum())
    assert columns["row"].min() == 101  # local row 0 is masked out
    assert columns["row"].max() == 100 + REAL_PATCH_CELLS - 1
    assert columns["col"][first].min() == 200 and columns["col"][~first].min() == 216
    assert not ((columns["row"] == 105) & (columns["col"] == 207) & first).any()
    cell = (columns["row"] == 103) & (columns["col"] == 216 + 4)
    assert columns["target_k"][cell][0] == pytest.approx(float(target[1, 0, 3, 4]))
    assert columns["prediction_k"][cell][0] == pytest.approx(float(prediction[1, 0, 3, 4]))


def test_columns_reproduce_the_per_patch_abs_error_sum() -> None:
    prediction, target, mask = _batch()
    prediction[0, 0, 3, 3] = 9999.0  # clamped to 400 K by both exports
    metadata = [_meta("p0", 0, 0), _meta("p1", 0, 16)]
    columns = ce.cell_columns_from_batch(prediction, target, mask, metadata)
    assert columns["prediction_k"].max() == EVAL_PREDICTION_CLAMP_K[1]
    rows = score_batch_patches(prediction, target, mask, metadata, data_range_k=30.0)
    error = np.abs(columns["prediction_k"].astype(np.float64) - columns["target_k"])
    for row in rows:
        select = columns["patch_id"] == row["patch_id"]
        assert int(select.sum()) == row["valid_cells"]
        assert error[select].sum() == pytest.approx(row["abs_error_sum"], rel=1e-5)


def test_columns_reject_other_splits_and_bad_shapes() -> None:
    prediction, target, mask = _batch()
    with pytest.raises(ValueError, match="not 'validation'"):
        ce.cell_columns_from_batch(
            prediction, target, mask, [_meta("p0", 0, 0), _meta("p1", 0, 16, split="test")]
        )
    with pytest.raises(ValueError, match="metadata"):
        ce.cell_columns_from_batch(prediction, target, mask, [_meta("p0", 0, 0)])
    with pytest.raises(ValueError, match="shape mismatch"):
        ce.cell_columns_from_batch(prediction[:, :, :8], target, mask, [])
    with pytest.raises(ValueError, match="no cell columns"):
        ce.concat_cell_columns([])


# ── writer ────────────────────────────────────────────────────────────


def _columns() -> dict[str, np.ndarray]:
    prediction, target, mask = _batch()
    return ce.cell_columns_from_batch(
        prediction, target, mask, [_meta("p0", 100, 200), _meta("p1", 100, 216)]
    )


def _write(columns: dict[str, np.ndarray], path: Path, split: str = "validation") -> Path:
    return ce.write_cell_export(
        columns, path, identity=IDENTITY, split=split, aggregation="arithmetic"
    )


def test_write_and_read_back_through_the_analysis_loader(tmp_path: Path) -> None:
    columns = _columns()
    path = _write(columns, tmp_path / "a" / ce.CELL_EXPORT_FILENAME)
    footer = json.loads(pq.read_schema(path).metadata[ce.CELL_EXPORT_METADATA_KEY])
    assert footer["arm"] == "L4" and footer["seed"] == 0 and footer["run_id"] == "run-a"
    assert footer["split"] == "validation" and footer["aggregation"] == "arithmetic"
    assert footer["prediction_clamp_k"] == [float(v) for v in EVAL_PREDICTION_CLAMP_K]
    assert pq.read_schema(path).remove_metadata() == ce.CELL_EXPORT_SCHEMA
    assert not list(path.parent.glob(".*tmp"))
    loaded = se.load_cell_exports([path])
    assert loaded.runs == (se.RunIdentity("L4", 0, "run-a"),)
    assert len(loaded.cells) == len(columns["row"])
    expected = np.abs(columns["prediction_k"].astype(np.float64) - columns["target_k"]).mean()
    assert np.abs(loaded.errors[0]).mean() == pytest.approx(expected)


def _drop_row(columns: dict[str, np.ndarray]) -> None:
    del columns["row"]


def _shorten(columns: dict[str, np.ndarray]) -> None:
    columns["col"] = columns["col"][:-1]


def _nan_target(columns: dict[str, np.ndarray]) -> None:
    columns["target_k"][0] = np.nan


def _unclamped(columns: dict[str, np.ndarray]) -> None:
    columns["prediction_k"][0] = 500.0


def _duplicate_cell(columns: dict[str, np.ndarray]) -> None:
    columns["col"][1] = columns["col"][0]


def _empty(columns: dict[str, np.ndarray]) -> None:
    for name in list(columns):
        columns[name] = columns[name][:0]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_drop_row, "missing"),
        (_shorten, "differ in length"),
        (_nan_target, "non-finite"),
        (_unclamped, "outside the evaluation clamp"),
        (_duplicate_cell, "more than once"),
        (_empty, "no cells"),
    ],
)
def test_writer_rejections(tmp_path: Path, edit, match: str) -> None:
    columns = _columns()
    edit(columns)
    with pytest.raises(ValueError, match=match):
        _write(columns, tmp_path / "x.parquet")
    assert not (tmp_path / "x.parquet").exists()


def test_writer_is_validation_only(tmp_path: Path) -> None:
    for split in ("test", "train"):
        with pytest.raises(ValueError, match="validation-only"):
            _write(_columns(), tmp_path / "x.parquet", split=split)
    assert not (tmp_path / "x.parquet").exists()


# ── job ───────────────────────────────────────────────────────────────


def test_select_runs_takes_matrix_labels_only() -> None:
    runs = job.select_runs(LABELS)
    assert [(r.arm_id, r.seed) for r in runs] == [("L1", 0), ("L4", 2)]
    assert all("evaluate_test=false" in r.hydra_overrides for r in runs)
    with pytest.raises(ValueError, match="outside the frozen v6 matrix"):
        job.select_runs(["v6-tessera-seed0"])
    with pytest.raises(ValueError, match="duplicate"):
        job.select_runs([LABELS[0], LABELS[0]])
    with pytest.raises(ValueError, match="no run labels"):
        job.select_runs([])


def _spread(samples: list[RealSample]) -> list[RealSample]:
    """Give every sample its own anchor; the rescore helper puts all at (0, 0)."""
    for i, sample in enumerate(samples):
        sample.meta.col = 16 * i
    return samples


def test_score_validation_cells_and_recheck() -> None:
    samples = _spread(_samples())
    columns, recomputed = job.score_validation_cells(
        _StubTask(5),  # type: ignore[arg-type]
        _loader(samples, 5),
        aggregation="arithmetic",
        device=torch.device("cpu"),
    )
    assert recomputed == pytest.approx(_expected_mae(samples), rel=1e-5)
    assert len(columns["row"]) == sum(int(s.mask_100m.sum()) for s in samples)
    assert set(columns["patch_id"]) == {s.meta.patch_id for s in samples}
    assert np.allclose(columns["prediction_k"], 290.0)
    error = np.abs(columns["prediction_k"].astype(np.float64) - columns["target_k"])
    assert error.mean() == pytest.approx(_expected_mae(samples), rel=1e-5)


def test_score_validation_cells_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        job.score_validation_cells(
            _StubTask(5, float("nan")),  # type: ignore[arg-type]
            _loader(_samples(), 5),
            aggregation="arithmetic",
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="split"):
        job.score_validation_cells(
            _StubTask(5),  # type: ignore[arg-type]
            _loader([_sample("t0", 0, split="test")], 5),
            aggregation="arithmetic",
            device=torch.device("cpu"),
        )


def _artifact_range(path: str) -> SsimDataRange:
    return dataclasses.replace(
        SSIM, source="artifact", version="v1", artifact_path=path, artifact_sha256="ab" * 32
    )


def test_ssim_artifact_path_follows_the_checkpoint_only_for_equal_content(tmp_path: Path) -> None:
    local = _artifact_range("/checkout/configs/modeling/ssim_data_range/v1.yaml")
    in_image = _artifact_range("/app/configs/modeling/ssim_data_range/v1.yaml")

    def ckpt(name: str, payload: object) -> Path:
        path = tmp_path / name
        torch.save({"hyper_parameters": {"ssim_data_range": payload}}, path)
        return path

    same = job.ssim_range_for_checkpoint(ckpt("same.ckpt", in_image.payload()), local)
    assert same.payload() == in_image.payload()
    other = dataclasses.replace(in_image, artifact_sha256="cd" * 32)
    assert job.ssim_range_for_checkpoint(ckpt("other.ckpt", other.payload()), local) == local
    wider = dataclasses.replace(in_image, data_range_k=in_image.data_range_k + 1.0)
    assert job.ssim_range_for_checkpoint(ckpt("wider.ckpt", wider.payload()), local) == local
    assert job.ssim_range_for_checkpoint(ckpt("none.ckpt", None), local) == local
    # a test-override range carries no artifact hash and is never rebound
    assert job.ssim_range_for_checkpoint(ckpt("override.ckpt", SSIM.payload()), SSIM) == SSIM
    bare = tmp_path / "bare.ckpt"
    torch.save({"state_dict": {}}, bare)
    assert job.ssim_range_for_checkpoint(bare, local) == local


class _Harness:
    """Replace every GCS and checkpoint touch of the job with local fakes."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.samples = _spread(_samples())
        self.expected_mae = _expected_mae(self.samples)
        self.cache_builds = 0
        self.fetched: list[str] = []
        self.bad_score = False
        self.out = tmp_path / "out"
        monkeypatch.setattr(rescore, "patch_index_fingerprints", lambda _root: {"index": "fp"})
        monkeypatch.setattr(rescore, "resolve_run_ssim_data_range", lambda *a, **k: SSIM)
        monkeypatch.setattr(rescore, "build_validation_cache", self._build_cache)
        monkeypatch.setattr(rescore, "fetch_checkpoint", self._fetch)
        monkeypatch.setattr(rescore, "checkpoint_selection_state", self._selection)
        monkeypatch.setattr(rescore, "load_rescore_task", self._load)
        monkeypatch.setattr(job, "ssim_range_for_checkpoint", lambda _path, ssim_range: ssim_range)
        monkeypatch.setattr(
            run_module,
            "_score_test_once",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("test split scored")),
        )
        monkeypatch.setattr(
            RealPatchDataModule,
            "test_dataloader",
            lambda self: (_ for _ in ()).throw(AssertionError("test loader built")),
        )

    def _build_cache(self, cfg: DictConfig, cache_root: Path) -> rescore.ValidationCache:
        self.cache_builds += 1
        return _full_cache(cache_root, self.samples)

    def _fetch(self, uri: str, dest: Path) -> rescore.CheckpointRef:
        self.fetched.append(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"ckpt:{uri}".encode())
        data = dest.read_bytes()
        return rescore.CheckpointRef(uri, 7, len(data), hashlib.sha256(data).hexdigest())

    def _selection(self, path: Path) -> dict[str, object]:
        return {"best_model_score": self.expected_mae + (0.5 if self.bad_score else 0.0)}

    def _load(self, path: Path, cfg: DictConfig, ssim_range: object) -> _StubTask:
        return _StubTask(selection_from_config(cfg).n_active)

    def run(self) -> dict[str, object]:
        return job.run_cell_export(run_labels=LABELS, output_root=self.out, device="cpu")


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def test_run_cell_export_end_to_end(harness: _Harness) -> None:
    manifest = harness.run()
    assert harness.cache_builds == 1
    assert harness.fetched == [run.checkpoint_uri for run in job.select_runs(LABELS)]
    assert not list((harness.out / "checkpoints").glob("*.ckpt"))
    assert manifest["evaluate_test"] is False and manifest["test_split"] == "not_read"
    assert manifest["splits_read"] == ["validation"] and manifest["uploaded"] is False
    on_disk = json.loads((harness.out / job.MANIFEST_NAME).read_text())
    assert on_disk["runs"] == manifest["runs"]
    entries = manifest["runs"]
    assert isinstance(entries, list) and len(entries) == 2
    cells = sum(int(s.mask_100m.sum()) for s in harness.samples)
    for label, entry in zip(LABELS, entries, strict=True):
        assert entry["run_label"] == label and entry["evaluate_test"] is False
        assert entry["cells"] == cells and entry["patches"] == 4 and entry["scenes"] == 2
        assert entry["clamped_validation_mae_100m"] == pytest.approx(harness.expected_mae, rel=1e-5)
        path = harness.out / entry["export"]["path"]
        assert path == harness.out / label / ce.CELL_EXPORT_FILENAME
        assert entry["export"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = se.load_cell_exports(
        [harness.out / label / ce.CELL_EXPORT_FILENAME for label in LABELS]
    )
    assert [(r.arm, r.seed, r.run_id) for r in loaded.runs] == [
        ("L1", 0, LABELS[0]),
        ("L4", 2, LABELS[1]),
    ]


def test_run_cell_export_is_create_only(harness: _Harness) -> None:
    existing = harness.out / LABELS[1] / ce.CELL_EXPORT_FILENAME
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"kept")
    with pytest.raises(FileExistsError):
        harness.run()
    assert harness.cache_builds == 0 and harness.fetched == []
    assert existing.read_bytes() == b"kept"


def test_run_cell_export_stops_on_a_selection_mismatch(harness: _Harness) -> None:
    harness.bad_score = True
    with pytest.raises(RuntimeError, match="best_model_score"):
        harness.run()
    assert not (harness.out / job.MANIFEST_NAME).exists()
    assert not list(harness.out.glob("*/" + ce.CELL_EXPORT_FILENAME))
