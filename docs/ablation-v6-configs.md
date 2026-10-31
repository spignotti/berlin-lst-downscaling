# v6 ablation configs

Successor of the Tag-11 configs for feature release v6 (feature schema
version 5, 22 channels) on the training v3 chain. The v1 locks stay unchanged;
see [ablation-stage-configs.md](ablation-stage-configs.md). Nothing in this
document starts a run.

## Arms

| Arm | Config (`configs/modeling/`) | Channels | C | Loss |
|-----|------------------------------|----------|--:|------|
| L1 | `v6_l1_locked` | spectral/index (10) | 10 | masked L1 |
| L2 | `v6_l2_locked` | L1 + morphology (6) | 16 | masked L1 |
| L3 | `v6_l3_locked` | L2 + shadows (2) | 18 | masked L1 |
| L4 | `v6_l4_locked` | L3 + ERA5 (4) = full roster | 22 | masked L1 |
| L5 | `v6_l5_locked` | as L4 | 22 | thermal-aware |
| LOO-M | `v6_loo_morph_locked` | L4 without morphology | 16 | masked L1 |
| LOO-S | `v6_loo_shadow_locked` | L4 without shadows | 20 | masked L1 |
| NO-NDVI-NDWI | `v6_no_ndvi_ndwi_locked` | L4 without `ndvi`, `ndwi_mcfeeters` | 20 | masked L1 |
| NO-RED-NIR-IDX | `v6_no_red_nir_index_locked` | L4 without `B04`, `B08`, `ndvi`, `ndwi_mcfeeters`, `ndbi`, `s2_broadband_albedo` | 16 | masked L1 |

Channel families (slices of the v6 roster): spectral/index (10), morphology
(6), ERA5 (4), shadows (2).

LOO-ERA5 (L4 minus ERA5) selects exactly the L3 channel set, so it has no
config of its own: L3 is the LOO-ERA5 run.

## Channel order

Inputs always follow the v6 registry order restricted to the arm. L4 and L5
use the whole roster, so their `contract.feature_order` is `v6_registry`; all
other arms are proper subsets and declare `v6_named_subset`. A list that is
not in registry order is rejected. `v6_named_subset` is admitted only on the
v6 ablation lock; `v6.yaml` and `tessera_arm.yaml` keep their behavior.

## Parameters and pins

All arm configs share `_v6_ablation_base.yaml`, which is not runnable on its
own. Two values are parameters with defaults, overridable per run and recorded
in the run identity:

- `seed` (default 0): an int >= 0.
- `trainer.max_epochs` (default 20): an int >= 1.

Everything else is pinned and checked by `assert_ablation_lock`
(`ablation_release: "v6"`): real stream data without patch cap, shuffled
train, batch size 4, 0 workers, no pin memory, feature schema 5, depth 4,
base width 32, learning rate 1e-3, weight decay 0, 16-mixed precision, one
GPU, W&B online, arithmetic aggregation, residual prior, splits
`[train, validation]` with no scene filter, `data_release_id` `training/v3`,
and the `features/v6`, `training/v3`, `training/patch-index/v3` and ARD
roots. The cache budget must be at least 12 GiB x C/10.

## Output root

Each arm sets `output_root: data/runs/modeling-ablation-v6-<slug>/seed-${seed}`.
The lock requires a path segment `seed-<seed>` so seeds never share a
directory. `RunIdentity.seed` records the seed (it is not part of the run-lock
expectations).

## Run matrix

`configs/ablation/v6_matrix.yaml` lists the seeds per arm. The owner
froze the matrix: 9 arms x seeds 0, 1, 2 = 27 runs (LOO-ERA5 equals L3 and has
no entry). The training schema is matrix-level, one value for all 27 runs
(owner decision 2026-10-31); per-arm entries must not carry these keys:

- `max_epochs: 40`.
- `lr_schedule: cosine`: warmup 2 epochs, `T_max = max_epochs`, `eta_min = 0`,
  stepped per epoch (the v6 lock pins warmup and `eta_min`). The parser
  requires `max_epochs > 2` for cosine.
- `early_stopping: false` (default when absent: false).
- `patch_cache: prebuilt` (default when absent: `job_local`). The cache prefix
  is resolved at runtime from each arm's data config; no key is stored in the
  matrix or in code. L5 shares the L4 data config and so the L4 cache.

The planning estimates are PLACEHOLDERS until the timing smoke
measures them (not measured, not a verified rate): `hours_per_epoch_estimate`
(0.12), `fixed_overhead_hours_estimate` (0.1) and `usd_per_hour` (1.00). The
estimate per run is `fixed_overhead + max_epochs * hours_per_epoch`, so 27 x
(0.1 + 40 x 0.12) = 132.3 GPU hours. List the planned runs, the equivalent
launcher arguments and the estimated totals without submitting anything:

```
uv run python scripts/validators/plan_v6_ablation.py
```

`scripts/validators/validate_ablation_configs.py` composes every planned run
(seed, `trainer.max_epochs`, `trainer.lr_schedule.name`,
`early_stopping.enabled`, and `+data.patch_cache.source=prebuilt`) and checks
the lock, including distinct output roots per seed. Test-pass handling is
unchanged: the matrix sets no `evaluate_test` override and the plan helper
prints no `--skip-test`.

## Early stopping, test pass, epoch curve

- `early_stopping: {enabled: false, patience: 10, min_delta: 0.0}` sits in
  `configs/modeling/_base.yaml`, so every config resolves to off. When enabled
  (v6 arms only; the v1 locks and every other config reject it), Lightning
  `EarlyStopping` monitors `validation/mae_100m` (mode min), the same masked
  100 m validation MAE the checkpoint is selected on. The v6 lock checks only
  types and ranges (patience int >= 1, min_delta >= 0). v6 runs record the
  block in `run_identity.json`, in `run_start`, and in `full_summary.json`,
  together with `stopped_epoch` (1-based like `epoch_metrics.json`: the epoch
  of a patience stop, null when training ran to `max_epochs`). A non-finite `validation/mae_100m` with early
  stopping enabled fails the run (`RuntimeError` with epoch and value, no
  summary, non-zero exit, no evidence); it is never recorded as an early stop.
  v1 identity payloads carry no `seed` or
  `early_stopping` key.
- `evaluate_test: true` is the default. `evaluate_test=false` (v6 arms only)
  builds no test data module and writes `"test": {"skipped": true}`; the Vertex
  evidence accepts a skipped test only when the summary records
  `evaluate_test=false`. Use it for pilot and timing runs.
  `--skip-test` launcher runs get `-notest` in the run label and output path
  and record `evaluate_test: false`; they are pilot/timing runs only and never
  enter the final arm comparison. `scripts/validators/plan_v6_ablation.py` and
  the matrix set no `evaluate_test` override, so planned runs score the test
  split (default true). The test pass scores the best-validation checkpoint
  (`validation/mae_100m`, mode min), reloaded from `best_model_path`, not the
  last epoch.
- Before training, the real path (and the v6 random forest and naive baseline
  runners) checks that the published scaler's channel names equal the feature
  schema roster and that the arm's selected indices map back to its names.
- v6 Vertex runs upload `epoch_metrics.json` create-only next to the evidence
  record (`VERTEX_EPOCH_METRICS_URI`), also when the fit fails.
  `scripts/operators/read_epoch_curve.py <evidence-or-epoch-metrics>` prints
  epoch, train and validation MAE and marks the best epoch; it never reads test
  metrics.

## Read preflight (mandatory)

Before any VM or Vertex v6 run (arms, random forest, naive baseline), run the
read-only preflight for the exact config with the operator's ADC:

```
uv run python scripts/validators/preflight_v6_reads.py --config-name <v6 config> --limit 8
```

It composes the config and runs `load_scaler`, a scaler/channel-selection check
(per arm, per random-forest stage, or the full v6 roster for the naive
baseline), and `load_patch_refs` for train and validation only.
It never requests the test split, reads no patch raster, writes nothing, and
prints only hashes and counts. It must end with `PASS`; any error stops the
launch. It is not run in CI.

The training/v3 policy hash (`193fa9b23943a839`) is not the v1/v2 hash: it is
`v6_training_policy_hash(<scaler.json v3_config_hash>)` in
`data/training/contracts.py`, the computation the training/v3 publisher and the
patch-index/v3 builder use. The readers derive the expected hash from the
training root's `scaler.json` through `expected_training_policy_hash`, so the
patch index's `source_policy_hash` and the run identity's
`training_policy_hash` must match it.

## Launcher

The Vertex launcher has one mode per arm (`v6-l1` ... `v6-no-red-nir-index`)
plus `--seed` and `--max-epochs` (v6 modes only; defaults 0 and 20), plus
`--early-stopping-patience N` (enables early stopping), `--early-stopping-min-delta X`
and `--skip-test` (sets `evaluate_test=false`), all off by default. The
effective run label is `<run-label>-seed<seed>-ep<epochs>` (plus `-notest`
with `--skip-test`). Example with
placeholders:

```
uv run python scripts/operators/launch_vertex_modeling.py --mode v6-l4 \
  --seed 1 --max-epochs 20 --run-label <label> --image-uri <image@sha256:digest> \
  --source-sha <40-char-sha> --service-account <service-account> \
  --infisical-identity <identity-id> --infisical-project <project-id> \
  --hourly-rate-usd <verified-rate> --preflight
```

A modeling image rebuilt from this commit is required before any submit: the
entrypoint passes `seed=` and `trainer.max_epochs=` overrides that older
images do not contain.

## Baselines on the v3 chain

`random_forest_v6_full.yaml` (and `random_forest_v6_smoke.yaml`) and
`baseline_v6_full.yaml` run the random-forest and naive baselines on the same
v3 chain. The random forest fits L1-L4 from the v6 arm registry with unchanged
protocol constants.

### Validation-only baselines (test 2025 stays closed)

Test 2025 stays closed until the whole ablation matrix is frozen. Until then
run the baselines only through their validation-only variants; they fit on
train, score validation 2024, and never load a test patch ref or read a test
patch:

| Baseline | Config | VM launcher | Output root |
| --- | --- | --- | --- |
| RF smoke | `random_forest_v6_val_smoke` | `run-random-forest-vm.sh v6-val-smoke <branch> <rate>` | `data/runs/random-forest-v6-notest/<id>` |
| RF full | `random_forest_v6_val_full` | `run-random-forest-vm.sh v6-val-full <branch> <rate>` | `data/runs/random-forest-v6-notest/<id>` |
| Naive | `baseline_v6_val` | `run-baseline-vm.sh <branch> baseline_v6_val` | `data/runs/baseline-v6-notest/<id>` |

```
uv run python scripts/validators/preflight_v6_reads.py --config-name random_forest_v6_val_full --limit 8
scripts/runners/vm/run-random-forest-vm.sh v6-val-full <branch> <hourly-rate-usd>
scripts/runners/vm/run-baseline-vm.sh <branch> baseline_v6_val
```

The RF runner accepts exactly `splits: ["validation", "test"]` or
`["validation"]`. A validation-only run is named `rf-<profile>-notest-<id>` in
W&B and its report records `scope.test_split: "skipped"`;
`validate_random_forest.py` then requires validation only and rejects any test
split in the report. The naive launcher evaluates `validation` only for
`baseline_v6_val` (the config itself lists train and validation).

`random_forest_v6_full`, `random_forest_v6_smoke`, `baseline_v6_full` and the
launcher modes `v6-smoke`, `v6-full`, `baseline_v6_full` open test 2025. Do
not run them before the matrix is frozen.
