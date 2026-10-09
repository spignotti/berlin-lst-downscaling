# Random-forest full temporal results (issue #59)

The four frozen random forests completed on 2026-10-08. Independent validation
and checksum-secured local recovery completed on 2026-10-09. All four improve
on the naive prior on validation and test. The corresponding U-Net stages have
lower errors on this same evaluation cohort.

## Method and cohort

The method is defined in [random-forest-baseline.md](random-forest-baseline.md).
Train years are 2017–2023, validation is 2024, and test is 2025. Each forest
uses the same seed-0 reservoir of 500,000 distinct eligible 100 m cells from
1,741,542 candidates. Training admits 6,872 of 6,873 indexed patches; one has
no valid prior block. There are no duplicate cells or overlapping windows.

The fixed profile uses 200 trees, depth 20, minimum leaf size 5, all input
columns per split, bootstrap, squared-error criterion, seed 0, and two workers.
Inputs are the active feature channels plus the physical prior in Kelvin.
Training uses aggregated 100 m features and Kelvin residual targets. Inference
uses native 10 m features and predicts a residual at each pixel; evaluation
pools the reconstructed temperature through exact 10×10 means.

Validation scores 1,341 of 1,342 patches and 341,588 valid cells; the same
invalid-prior exclusion applies. Test scores all 1,803 patches and 459,564
valid cells. Patch IDs, masks, feature-fill counts, and exclusion accounting
match the frozen naive evidence. MAE is weighted by valid cell count.
Validation and test scores did not change the profile or select a forest.

## Primary results

All values are MAE in Kelvin at 100 m. U-Net references are frozen in
[Stage-1 results](stage1-full-results.md) and
[Tag-11 results](ablation-full-results.md).

| Stage | Features | RF val | RF test | U-Net val | U-Net test |
|---|---|---:|---:|---:|---:|
| 1 | spectral + index, C=10 | 1.28859 | 1.27036 | 0.50285 | 0.53257 |
| 2 | + morphology, C=18 | 1.26061 | 1.23274 | 0.51167 | 0.51366 |
| 3 | + shadows, C=20 | 1.25810 | 1.25927 | 0.50601 | 0.49991 |
| 4 | + ERA5, C=28 | 1.18268 | 1.18341 | 0.49962 | 0.47535 |
| Naive prior | prior expansion | 1.61010 | 1.39523 | — | — |

The full-feature RF reduces test MAE by approximately 0.212 K (15.2%) against
the naive prior. The matched Stage-4 U-Net has approximately 0.708 K lower
error than this RF. Adding morphology lowers RF test error by 0.03762 K;
adding shadows raises it by 0.02653 K; adding ERA5 lowers it by 0.07586 K.
These are cumulative, single-seed feature effects. The shadow step has a
small validation improvement alongside a test deterioration.

Stage 5 uses Stage-4 features with a different U-Net loss. Its frozen test
MAE is 0.45931 K; it requires no additional RF feature arm. Single-family
isolation forests were outside this run's scope.

## Secondary metric

SSIM uses fully valid 7×7 uniform windows at 100 m. Each stage has 130,175
validation windows and 176,349 test windows. It is a descriptive metric.

| Stage | RF validation SSIM | RF test SSIM |
|---|---:|---:|
| 1 | 0.969997 | 0.975555 |
| 2 | 0.971084 | 0.976457 |
| 3 | 0.971205 | 0.976502 |
| 4 | 0.973711 | 0.978205 |

## Verification and provenance

Run: `random-forest-full-20261008T143403Z-5A7B6B4D`.
Executed source: `eb7a8fbc3381a76054070d56d5f518d59929375f`.
Python 3.12.14, NumPy 1.26.4, scikit-learn 1.9.1, skops 0.16.0,
SciPy 1.17.1, and Torch 2.2.2+cu121. This run used CPU inference and fitting.

Frozen repository evidence is under
`docs/results/random-forest-full-20261008T143403Z-5A7B6B4D/`:

- `random_forest_report.json`: exact aggregates, all patch contributions,
  feature names/order, model hashes, row-selection hashes, and provenance.
- `random_forest_config.yaml`: resolved full configuration.
- `validation.txt`: successful independent validation.
- `evidence_manifest.json`: SHA256 of the retained evidence files.
- `artifact.sha256`: checksum of the complete locally secured archive.

The validator audits 500,000 selected row identities, checks all four models
and complete cohorts, reconstructs five test and five validation patches from
published rasters, and rebuilds/replays two native-resolution probes across
all four stages. Cohort reconciliation reports zero mismatches. A separate
local aggregation of every patch contribution reproduces all MAE and SSIM
values within 1e-12. Local model hashes match the validated report.

The full archive, four `.skops` models, training-row Parquet, native probes,
logs, and offline W&B run `pbzqtb54` are secured locally under
`data/runs/random-forest/<run-id>/`. The archive checksum is
`1f52bbd98ff8efad5b4c5fa4b3bea9a8a73e77d93acdf640b56879250fcaa829`.
Use the retained checksum file as the authoritative checksum. Model artifacts
remain outside Git. Live W&B synchronization requires separate approval.
Canonical GCS products and prior retained QA evidence remain unchanged.
The owned remote output and transfer archive were removed after successful
checksum comparison and extraction. The VM was stopped successfully.

To reproduce the frozen profile on an authorized CPU host:

```bash
uv run python scripts/runners/run_random_forest.py \
  --config-name random_forest_full output_root=<new-run-root>
uv run python scripts/validators/validate_random_forest.py \
  --report <new-run-root>/random_forest_report.json --max-patches 5
```

## Resources and recovery

Pipeline runtime is 15,149.14 seconds (4 h 12 min 29 s), including train reads,
fitting, evaluation, and artifact writing. Train reads took 2,814.05 seconds.
Stage fit times were 923.13, 1,370.39, 1,400.94, and 2,000.13 seconds.
Peak process RSS was 5,081,489,408 bytes (4.73 GiB); the temporary evaluation
cache held 9,014,879,232 bytes (8.40 GiB) and was removed by the pipeline.

The machine was the pinned on-demand `n2-highmem-2` VM in `europe-west3-b`.
At the admitted $0.168788/hour rate, pipeline runtime alone corresponds to
approximately $0.71 of compute. This is a runtime-derived estimate. Total
billed VM uptime also includes startup, smoke, validation, transfer, and idle
connection-recovery time. The local computer being offline interrupted SSH
supervision; the remote pipeline completed successfully. The first lifecycle
exceeded its eight-hour aggregate bound during connection recovery. A separately
approved recovery of at most one hour validated, transferred, and stopped the
VM in approximately nine minutes. At the same rate that recovery corresponds
to about $0.03 of compute. Actual total billing, transfer, API, and disk costs
have not been independently retrieved.

## Limits on interpretation

- One seed and one fixed RF profile; this result describes that profile.
- RF trains on a sampled cell population. U-Net training uses patches and
  spatial neighborhoods. Training representation, objective, and capacity
  differ between the model families.
- The RF transfers from aggregated training features to native 10 m features
  at inference. That scale-transfer assumption is part of this comparison.
- The target is Landsat at 100 m with a same-scene degraded prior. MAE and
  SSIM support a temporal pseudo-pair reconstruction claim at recurring
  locations. Independent 10 m accuracy and spatial generalization remain
  unmeasured.
- Cumulative steps combine families already present. The shadow result does
  not isolate shadows from morphology, and small one-seed changes carry no
  statistical-significance claim.
