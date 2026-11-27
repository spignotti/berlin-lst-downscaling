# Spatial error analysis (validation)

Exploratory description of where the validation errors of a set of runs sit:
by scene, by Stadtstruktur group, and in space. It is not one of the
pre-declared comparison families of the matrix (thesis method notes, removed from this repo; `66ff28f:docs/thesis/03-methodik/evaluation-unsicherheit.md`),
its intervals are unadjusted, and it derives no verdict about a model.

Scope: validation split only. The loaders refuse any other split, the cell
export is validation-only by construction, and no test (2025) ref, metric or
export is read.

## Steps

| # | Step | Input |
|---|------|-------|
| 1 | Error per scene = scene bias + centred residual | per-cell export |
| 2 | MAE, bias, RMSE per Stadtstruktur group (proposal B) and the paired difference A − B per group | per-cell export, share table |
| 3 | Moran's I of the centred residuals per scene | per-cell export |
| 4 | Per-scene table of A − B next to scene properties; cell-pooled against scene-equal estimator | per-patch export, EDA table (optional) |
| 5 | Paired scene bootstrap between two single-run reference models (random forest stages) | two per-patch exports |

Code: `src/berlin_lst_downscaling/analysis/spatial_error.py`, CLI
`scripts/analyze_spatial_errors.py`. The CLI runs every step whose inputs are
given and lists the others under "Skipped steps".

Sign: error = prediction − target, in Kelvin. Positive means the model is too warm.

## Inputs

All inputs are explicit local files. Copy them from GCS first; the tool does not
read `gs://` paths.

- **Per-patch exports** (schema 2, `analysis.scene_bootstrap`), one per run:
  `<patch-export-root>/<run-label>/eval_validation_patches.parquet`. For the v6
  matrix they are retained under
  `gs://berlin-lst-training-data/qa/modeling/vertex-smoke/v6m-070b33c-val-rescore-notest-r2/`.
  They hold sums per patch (absolute error, valid cells, SSIM). They carry no
  signed error and no cell rows, so they serve steps 4 and 5 only.
- **Per-cell exports** (`modeling/cell_exports.py`, schema 1), one per run:
  `<cell-export-root>/<run-label>/eval_validation_cells.parquet`. One row per
  valid 100 m cell: `scene_id`, `patch_id`, global canonical `row`/`col`
  (`common/grid.py`), `target_k`, `prediction_k` (clamped like the per-patch
  export). Run identity is in the Parquet footer. See "Cell export" below.
- **Stadtstruktur shares**: one vintage of
  `gs://berlin-lst-training-data/evaluation/stadtstruktur/v1/grid/100m/<vintage>/shares.parquet`
  (`docs/stadtstruktur.md`). Types map to groups through `PROPOSAL_B` in
  `data/stadtstruktur/class_review.py`.
- **Scene properties** (optional):
  `gs://berlin-lst-training-data/evaluation/eda/v3/tables/lst_distribution.parquet`.
  Only its validation rows are used. Scene mean and SD of LST there cover all
  eligible cells of the scene, not only the evaluated patches.
- **Reference exports** (step 5): the per-patch exports of two single runs, for
  example `eval_stage1_validation_patches.parquet` and
  `eval_stage4_validation_patches.parquet` of a validation-only random forest run.

Runs are named by run label. Arm and seed come from the export footers, so runs
of other arms (a TESSERA arm, a spatial holdout run) join by adding their label,
as long as their exports follow the same schemas and cover the same cells.

## Commands

Steps 4 and 5 from per-patch exports:

```bash
uv run --locked python scripts/analyze_spatial_errors.py \
  --run-labels v6m-070b33c-l1-seed0-ep40-cos v6m-070b33c-l1-seed1-ep40-cos \
               v6m-070b33c-l1-seed2-ep40-cos v6m-070b33c-l4-seed0-ep40-cos \
               v6m-070b33c-l4-seed1-ep40-cos v6m-070b33c-l4-seed2-ep40-cos \
  --arm-a L4 --arm-b L1 \
  --patch-export-root <dir> \
  --scene-properties <dir>/lst_distribution.parquet \
  --reference-a <dir>/eval_stage4_validation_patches.parquet \
  --reference-b <dir>/eval_stage1_validation_patches.parquet \
  --output-dir <out>
```

All steps, with the per-cell exports and the share table:

```bash
uv run --locked python scripts/analyze_spatial_errors.py \
  --run-labels <labels> --arm-a L4 --arm-b L1 \
  --patch-export-root <dir> --cell-export-root <dir> \
  --stadtstruktur-shares <dir>/shares.parquet \
  --scene-properties <dir>/lst_distribution.parquet \
  --output-dir <out>
```

Outputs: `<out>/spatial_error_analysis.json` and `.md`. The JSON records every
parameter and every skipped step with its reason. When both export kinds are
given, the tool first checks that the absolute cell errors of each run
reproduce its per-patch `abs_error_sum` and valid-cell counts, and stops if
they do not.

## Cell export

`scripts/operators/export_v6_validation_cells.py` writes the per-cell exports
for a list of v6 matrix run labels from their retained `best.ckpt` files:

```bash
scripts/operators/with-gcp-operator.sh -- \
  uv run --locked python scripts/operators/export_v6_validation_cells.py \
  --run-labels <labels> \
  --output-root data/runs/v6-val-cells/<name> --device cpu
```

- Inference only on validation 2024. Every run is composed with
  `evaluate_test=false`; the shared cache is built from the validation split
  alone. Checkpoint download, strict load, hyperparameter comparison and the
  best-score recheck are those of the validation rescore
  (`modeling/v6_val_rescore.py`).
- It reads the checkpoints and the published validation sources from GCS and
  writes local files only: `<output-root>/<run-label>/eval_validation_cells.parquet`
  and `<output-root>/cell_export_manifest.json`. Nothing is uploaded. Existing
  output files stop the job before any work.
- It runs on a CPU. No Vertex job, VM or image build is involved. Reading from
  outside the bucket's region incurs GCS egress for the validation sources and
  one 377 MB checkpoint per run.
- `--list` prints the 27 matrix run labels without touching GCS.

## Method

### Bootstrap

The scene is the resampling unit: B = 10,000, rng seed 0, 95 % percentile
interval, as in the matrix evaluation. `scene_bootstrap.py` is unchanged. The
L4 − L1 comparison of step 4 calls its `compare_arms`. Step 5 and the group
intervals need a bootstrap over a plain vector of scene differences, which
`compare_arms` does not offer (it refuses two single-run arms), so
`spatial_error.scene_bootstrap_ci` draws the same index matrix with the same
generator, seed and batching. A test asserts identical intervals on the same
differences.

Seeds are not resampled. Per scene, the arm metric is the mean over the seeds.

### Step 1: scene bias and centred residuals

Per run and scene, bias is the mean signed error over the valid cells. The
centred residual of a cell is its error minus that bias. Then
`MSE = bias² + centred RMSE²` holds exactly per run and scene, and
`bias_share_of_mse = bias² / MSE`. The summary gives scene-equal means per arm
and the cell-pooled split of the MSE into the scene-bias part and the centred
part.

### Step 2: Stadtstruktur groups

A group's weight in a cell is the sum of the share column over the group's
types. Group metrics are weighted means over cells within a scene: bias, MAE,
RMSE, and the centred bias (group bias after the scene bias is removed).
"Effective cells" is the summed weight, in units of one 100 m cell.

A scene counts for a group only with at least `--min-effective-cells`. The
group summary gives the scene-equal mean over those scenes and the area-pooled
value. The paired difference A − B per group is the mean scene difference of the
group MAE with the scene bootstrap over the scenes that support the group; a
group with fewer than `--min-scenes` supported scenes is listed without an
interval. Every group interval is exploratory and unadjusted for the number of
groups. Water and construction are exclusion bins in proposal B; they are
reported with their role and not dropped.

### Step 3: Moran's I

Global Moran's I of the centred residuals per run and scene on the canonical
100 m grid, over the valid cells of all evaluated patches of the scene
(adjacent patches are neighbours). `I = (N / S0) · Σ w_ij z_i z_j / Σ z_i²`.
Cells without a present neighbour keep weight zero and are counted as islands.
The value is descriptive: no permutation test is run. The expected value under
no autocorrelation, −1/(N − 1), is reported next to it.

### Step 4: pooled against scene-equal

With per-scene MAE cell-pooled within the scene, the cell-pooled difference is
`Σ w_s d_s` with `w_s` the scene's share of all cells, and the scene-equal
difference is the plain mean of `d_s`. Their gap is exactly
`Σ (w_s − 1/S) d_s`; the table lists each scene's term (`gap part`). Both
estimators get a scene-bootstrap interval; the pooled one recomputes the ratio
on every resample. The table adds date, season (meteorological, as in
`data/eda/rules.py`), patches, cells, and the EDA scene mean LST, and reports
the correlation of the scene difference with those properties.

Sensitivity, not the frozen rule: scenes of one platform, path and date are
adjacent rows of one overpass. A second pair of intervals resamples overpasses
instead of scenes, with unchanged point estimates. The frozen rule of the
matrix evaluation resamples scenes.

### Step 5: single-run reference pair

Per-scene MAE of two single runs on identical patches, the paired scene
bootstrap of A − B, scene-equal and cell-pooled. A single run has no seed
spread; the interval covers scenes only.

## Parameters that are scientific choices

Each is a CLI parameter. The default is a proposal, not a decision.

| Parameter | Default | Source of the default | Alternatives |
|-----------|---------|-----------------------|--------------|
| `--min-effective-cells` | 50 | support floor in `docs/stadtstruktur.md` (sparse under 50 effective cells) | any positive value |
| `--min-scenes` | 10 | none; proposed here | any value from 2 |
| `--share-column` | `share_of_cell` | the share tables are the evaluation product; street land and unmapped area stay unassigned | `share_of_covered` rescales to the mapped area of the cell |
| `--min-share` | 0 | every mixed cell contributes with its area share | a threshold keeps only cells where the group reaches that share (0.8 is the dominance level used in `docs/stadtstruktur.md`) |
| share vintage | file given | same-year vintage for 2021–2024 scenes (`docs/stadtstruktur.md`) | — |
| `--moran-neighbourhood` | `queen` | none; proposed here | `rook` |
| `--moran-weights` | `row` | none; proposed here | `binary` |
| resampling unit | scene | frozen rule of the matrix evaluation | overpass, reported as a sensitivity in step 4 |

## Limits

- Validation 2024 only: 27 scenes of one year from 15 overpasses, 17 of them
  with at most six patches. Scene-level intervals are wide and season cells are
  thin.
- The share table describes block geometry at 1:5,000. About 11 % of Berlin is
  street land or otherwise unmapped and belongs to no group. More than half of
  the covered cells are mixed, so group metrics describe area-weighted mixtures
  unless `--min-share` is raised.
- Errors are measured against Landsat at 100 m. Nothing here validates a 10 m
  prediction.
- Group and scene results are many comparisons without adjustment. They are a
  description for the discussion chapter.
- The per-cell export is produced from the retained best checkpoints, which were
  selected on this validation split.
- Step 4 calls `compare_arms`, which requires seeds 0, 1 and 2 for each arm.
  Steps 1-3 average over whatever seeds an arm has and record the count.
- All runs of one analysis must cover identical cells with identical targets.
  A run on another patch set (a spatial holdout) needs its own analysis run.
- The cell export job only knows the 27 runs of the v6 matrix. Another arm
  needs its own job that writes the same cell export schema.
