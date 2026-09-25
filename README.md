# Berlin LST Downscaling

Cloud-native land-surface-temperature downscaling pipeline for Berlin. It
combines 100 m thermal observations with fine-resolution predictors to build
a training-ready dataset for high-resolution LST modelling on a canonical
10 m grid (EPSG:25833).

## Architecture

```
Data sources
     │
     ▼
Manifest + ARD ──► Static / dynamic context ──► Feature stacks ──► Modelling (WIP)
```

## Inputs

- Landsat-8/9 Collection 2 Level-2 thermal data (100 m), the anchor target
- Sentinel-2 L2A surface reflectance (10 m), the primary predictor
- ECOSTRESS L2T LSTE v002 (70 m), independent validation
- Berlin 3D city morphology (LoD2 building geometry, terrain height,
  vegetation height, imperviousness) from the Berlin GDI, fixed per-scene
  vintage
- ERA5-Land meteorology via the Copernicus CDS API

## Pipeline

- **Manifest-driven selection.** Versioned, immutable manifest bundles define
  the scene universe and pair each Landsat anchor with its Sentinel-2 partner.
- **ARD processing.** Analysis-ready COGs with NaN-no-data semantics on the
  canonical grid.
- **Static context.** Official archive sources become 10 m geometry products:
  building and vegetation DSMs, shadow horizons, and sky-view factor, per
  geometry vintage.
- **Dynamic context.** Per-anchor ERA5-Land fields and building/vegetation
  shadow masks.
- **Feature stacks.** Per-anchor 28-band 10 m stacks (spectral indices,
  semantic morphology predictors, meteorology, shadows) with co-registered
  validity masks, ready for the training stage.
- **QA gates.** Read-only, mask-free validation of every published input,
  plus independent validators per pipeline stage.

Every product publishes four co-located artifacts: the COG, a STAC Item,
`provenance.json`, and a `complete.json` completion marker written last as
the publication gate. Ledgers in Google Cloud Storage are the reproducibility
basis; the uv lockfile and per-run config fingerprints pin the environment.

## Status

Preprocessing is complete: manifest selection, ARD, static and dynamic
context, and the per-anchor feature stacks (V3) are delivered and
validated, with the Stage-1 raw-input and Stage-2 feature-stack QA gates
green. Training-data preparation is delivered as the `training/v1`
release — eligibility masks, temporal splits, the cell index, and the
train-only scaler — together with the WB3 patch index
(`training/patch-index/v1`). The modelling stack implements the pseudo-pair
contract: a fixed 2D U-Net predicts LST at 10 m from the 28 feature
channels plus the coarse prior, checkpointed on the cell-weighted masked
MAE at 100 m, with a naive prior-expand baseline scored through the same
reader, mask, and metric. The scaffolding gates below prove the lifecycle
and the released-source wiring; a full Stage-1 training run is the next
step.

## Setup

```bash
uv sync
uv run nox -s lint typecheck
```

Local smoke gates (`uv run nox -s smoke-*`) may need Google Cloud ADC: the
modelling contract smokes run without credentials, while the real two-arm
comparison and the pipeline smokes read GCS. The dynamic pipeline also needs
a Copernicus CDS API key. Heavy production runs execute on a GCP VM against
GCS; managed model training will run on Vertex AI.

### Modelling gates

All three are technical checks, not quality thresholds.

```bash
# Deterministic synthetic MSE lifecycle (no GCS, no credentials)
uv run nox -s smoke-modeling

# Contract-shaped synthetic fixture through the masked-L1/masked-MAE
# lifecycle (no GCS): 28x160x160 features + prior, 16x16 target/mask
uv run nox -s smoke-modeling-contract

# Opt-in, requires Google Cloud ADC: one bounded read-only epoch over four
# published patches per split, the naive baseline on the identical patch
# universe, and the independent baseline validator
uv run nox -s smoke-real-comparison
```

Canonical storage since the 2026-09-25 account cutover is
`gs://berlin-lst-training-data/` (project `berlin-lst-training`,
`EUROPE-WEST3`), a verified byte-for-byte mirror of the previous bucket;
see `docs/gcs-inventory-and-transfer.md`.

Source roots are Hydra values and can be overridden per run, which is how the
cutover was applied without editing the configs:

```bash
uv run python scripts/runners/run_modeling.py --config-name real_smoke \
  patch_index_root=gs://<bucket>/training/patch-index/v1 \
  training_root=gs://<bucket>/training/v1 \
  ard_root=gs://<bucket>/ard/full/<release>
```

## Entrypoints

Scripts are grouped by role under `scripts/`:

- `scripts/runners/` — pipeline entrypoints (`run_*`, `build_manifest`)
- `scripts/validators/` — read-only validators (`validate_*`)
- `scripts/operators/` — release and diagnostic tooling
  (`compare_feature_releases`, `preflight_feature_release`,
  `retire_feature_release`, `audit_cloud_masking`)

Each script is self-documenting (`uv run python scripts/<group>/<name>.py
--help`) and runnable directly on the VM. VM lifecycle orchestration lives in
the `google-access` OpenCode skill, not in the repository.

## Documentation

- `docs/data-sources-and-contracts.md` — sources, canonical grid, manifest
  and product contracts.
- `docs/pseudo-pair-tensor-contract.md` — normative pseudo-pair and tensor
  contract for real-data training (WB3), implemented by `modeling/`.
- `docs/gcs-inventory-and-transfer.md` — GCS bucket inventory and the
  copy-first mirror runbook for a later account move.