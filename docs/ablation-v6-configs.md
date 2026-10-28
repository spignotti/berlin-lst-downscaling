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

`configs/ablation/v6_matrix.yaml` lists seeds and epochs per arm plus the
planning estimates (hours per run, USD per hour). The defaults are pending the
owner's matrix decision; edit the file to change arms, seeds, or epochs. List
the planned runs and the estimated totals without submitting anything:

```
uv run python scripts/validators/plan_v6_ablation.py
```

`scripts/validators/validate_ablation_configs.py` composes every planned run
and checks the lock, including distinct output roots per seed.

## Launcher

The Vertex launcher has one mode per arm (`v6-l1` ... `v6-no-red-nir-index`)
plus `--seed` and `--max-epochs` (v6 modes only; defaults 0 and 20). The
effective run label is `<run-label>-seed<seed>-ep<epochs>`. Example with
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
