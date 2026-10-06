# Tag-11 ablation stage configs (issue #57)

Hydra profiles for the cumulative feature ladder under the Stage-1 residual
lock. Full temporal runs are issue #58.

| Config | Stage | Channels | `feature_order` | Loss |
|--------|------:|---------:|-----------------|------|
| `stage2_locked` | 2 | 18 | `v3_first_c` | masked L1 |
| `stage3_locked` | 3 | 20 | `v3_named_subset` | masked L1 |
| `stage4_locked` | 4 | 28 | `v3_named_subset` | masked L1 |
| `stage5_locked` | 5 | 28 | `v3_named_subset` | thermal-aware |

Stage 2 is the first 18 bands of the published V3 order (spectral + index +
morphology). Stage 3 adds `shadow_building` / `shadow_vegetation` without ERA5;
those bands sit after meteorology in V3, so the loader selects by name after
the full train-only scaler runs. Stage 4 appends ERA5. Stage 5 keeps stage-4
inputs and switches only the training objective (`masked L1` + SSIM +
gradient penalties); checkpoint selection stays masked MAE @ 100 m.

Backbone, temporal split, patch geometry, residual prior, seeds, and the
cheap Stage-1 runtime (batch 4, AMP, workers 0) stay frozen. Job-local cache
budgets scale with channel count from the Stage-1 12 GiB / C=10 floor.

Smoke (GCS-free):

```bash
uv run python scripts/validators/validate_ablation_configs.py
```
