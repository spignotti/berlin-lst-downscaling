# Tag-11 ablation stage configs (issue #57)

Hydra profiles for the cumulative feature ladder under the Stage-1 residual
lock. Full temporal runs are issue #58.

| Config | Stage | Channels | `feature_order` | Loss |
|--------|------:|---------:|-----------------|------|
| `stage2_locked` | 2 | 18 | `v3_first_c` | masked L1 |
| `stage3_locked` | 3 | 20 | `v3_named_subset` | masked L1 |
| `stage4_locked` | 4 | 28 | `v3_named_subset` | masked L1 |
| `stage5_locked` | 5 | 28 | `v3_named_subset` | thermal-aware |
| `isolate_shadows_locked` | — | 12 | `v3_named_subset` | masked L1 |
| `isolate_era5_locked` | — | 18 | `v3_named_subset` | masked L1 |

Stage 2 is the first 18 bands of the published V3 order (spectral + index +
morphology). Stage 3 adds `shadow_building` / `shadow_vegetation` without ERA5;
those bands sit after meteorology in V3, so the loader selects by name after
the full train-only scaler runs. Stage 4 appends ERA5. Stage 5 keeps stage-4
inputs and switches only the training objective (`masked L1` + SSIM +
gradient penalties); checkpoint selection stays masked MAE @ 100 m.

`isolate_shadows_locked` is the spectral block plus the two shadow bands and
nothing else. `isolate_era5_locked` is the spectral block plus the eight ERA5
bands and nothing else. Both stay on masked L1. They exist so morphology and
shadows can be compared as separate additions, and so ERA5 can be measured
without the geometry stack (issue #63). They are not extra rungs of the
cumulative ladder.

Backbone, temporal split, patch geometry, residual prior, seeds, and the
cheap Stage-1 runtime (batch 4, AMP, workers 0) stay frozen. Job-local cache
budgets scale with channel count from the Stage-1 12 GiB / C=10 floor.

Smoke (GCS-free):

```bash
uv run python scripts/validators/validate_ablation_configs.py
```

## Full temporal submit (issue #58)

The Vertex launcher admits `--mode stage2` … `--mode stage5`, plus
`--mode isolate-shadows` and `--mode isolate-era5` (issue #63). Each mode maps
to the matching `*_locked` config, reuses the Stage-1 full evidence profile
(epoch curve, one-shot test, create-only checkpoint), and keeps the same
48 h / $50 ceilings and `--hourly-rate-usd` requirement as `--mode full`.

Rebuild the modeling image from the commit that carries these configs before
the first ablation submit; the Stage-1 digest does not include them. Run stages
sequentially (2 → 3 → 4 → 5); do not retune between stages.

```bash
uv run --group operators python scripts/operators/launch_vertex_modeling.py \
  --mode stage2 \
  --image-uri europe-west3-docker.pkg.dev/berlin-lst-training/berlin-lst-runners/modeling-vertex@sha256:<ablation-digest> \
  --source-sha <clean-commit> --run-label stage2-full-<utc>-<suffix> \
  --service-account berlin-lst-vertex-smoke@berlin-lst-training.iam.gserviceaccount.com \
  --infisical-identity 7ba603e5-b94d-42d1-bc57-d64658dde09d \
  --infisical-project 5da7dfb7-954d-4736-ba2e-4471ade9d766 \
  --infisical-env dev --infisical-path /vertex \
  --hourly-rate-usd 1.00 --preflight
```
