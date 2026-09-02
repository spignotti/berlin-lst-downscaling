# Phase-2 preparation

Short handoff record between the delivered preprocessing phase and the
next QA/feature-engineering tasks. Normative contracts and the full
product inventory live in `data-sources-and-contracts.md` and
`phase-1-delivery.md`; this file records only the state a phase-2
session must know up front.

## Published inputs

- Manifest bundle (canonical, immutable):
  `gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/`
  — 509 scenes (345 Landsat anchors, 158 Sentinel-2 predictors,
  6 ECOSTRESS validation granules).
- ARD COG stack: `gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/`
  — per-scene COG + flag COG + STAC + provenance + complete; ledger 509/509 done.
  Sentinel-2 ARD is six-band (B02/B03/B04/B08/B11/B12; B11/B12 masked
  at native 20 m before bilinear 20→10 m resampling).
- Static geometry (sources + derived), dynamic per-scene ERA5-Land and
  shadows, and LoD vintages are delivered as documented in
  `phase-1-delivery.md`.

## ECOSTRESS validation role

The six ECOSTRESS granules are `role=validation` only — never training
input. They are published at 70 m native resolution and the
`scripts/validators/validate_ecostress_scenes.py` probe enforces, on the published
artifacts:

- structural COG/flag/grid contract (EPSG:25833, canonical 70 m grid),
- `NaN ⟺ FLAG_FILL` pixel consistency,
- STAC `lst`/`flag` assets declaring `spatial_resolution: 70`.

## Accepted schema state

The ARD ledger holds per-source schema versions: 345 Landsat rows at
v6, 6 ECOSTRESS rows at v7, 158 Sentinel-2 rows at v8 (six-band
spectral contract, republished 2026-08-14). A future full `run_ard`
rewrites the older rows deterministically. No intermediate migration
is planned.

## Cloud-mask audit evidence

The cloud-masking audit saves bounded, descriptive evidence under
`gs://berlin-lst-data/qa/cloud_masking/<run-id>/` (PNG overlays for the
top-risk pairs, plus `index.csv` and `summary.json`). It applies no
pass/fail decision and changes no published mask.

## Stage-1 raw-input QA gate (completed)

The Stage-1 raw-input gate (`scripts/runners/run_qa_stage1_raw.py`,
`data/qa/`) ran over the full training universe on the VM
(2026-08-14, run `9518fe0c`, after the six-band S2 republish):

- 345 pairings: 324 assessed, 21 excluded (2026 `role=inference`
  scenes, outside the training universe).
- **0 findings** — no contract/range/input faults; the gate passed
  clean and feature computation is unblocked. The S2 roster now
  includes B11/B12 (six-band metadata + reflectance range checks).
- Diagnostic support: 20,169,061 target-valid 100 m cells, of which
  5,914,125 are fully supported (valid Landsat target plus all 100
  10 m subpixels valid across S2, static morphology, ERA5-Land, and
  shadows). The numbers are unchanged from the pre-SWIR run
  (`71fab30d`) because S2 validity is flag-based (`flag == 0`) and
  B11/B12 add no validity constraint beyond the flag. Per-scene counts
  and the support histogram are in the report bundle.
- The gate is mask-free by design: only `summary.json`,
  `scenes.parquet`, `scenes.csv`, and logs are published. The 2026-08-14
  first run (`71fab30d`) was published before the evidence paths were
  decoupled from planning task ids, so it lives under the historical
  prefix `gs://berlin-lst-data/qa/wb2c-2/raw/71fab30d/`; newer runs
  write to `gs://berlin-lst-data/qa/stage1_raw/<run-id>/` (this run:
  `gs://berlin-lst-data/qa/stage1_raw/9518fe0c/`). No validity or
  training-selection mask exists yet — the final `training_eligible@100m`
  mask is decided after feature computation and the Stage-2 gate
  (user decision).

## Scene feature stacks (WB2c-3, delivered as V2)

Per paired Landsat anchor (2017-2025) the features pipeline publishes a
28-band 10 m stack + `feature_valid` mask on the canonical grid under
`gs://berlin-lst-data/features/v2/<scene_id>/` (ledger
`_state/features/ledger.parquet`). Contract details (channel order,
formulas, albedo proxy, mask semantics, AOI handling, vegetation-height
carry-forward) live in `data-sources-and-contracts.md`.

- **Corrected morphology contract (schema v2).** The initial 24-band
  contract mistakenly included the DSM auxiliaries (`building_dsm`,
  `vegetation_dsm`, `combined_dsm`) as model channels. Schema v2 replaces
  them with the eight semantic morphology predictors: 4 LoD2 bands
  (`building_height_mean/std/max`, `building_coverage_ratio`, per-scene
  vintage), 2 vegetation-height bands (`vegetation_height/2020`
  carry-forward), `imperviousness` (per scene year: 2016 for 2017-2019,
  2021 for 2020-2025), and `svf`. The DSMs remain internal auxiliaries
  for SVF/horizon/shadow computation only.
- Run: full 324-scene publication (2026-08-21/22, VM,
  `run_features_vm.sh`, isolated per-scene subprocesses with `--resume`;
  interrupted twice by SSH connection loss and resumed cleanly — the
  atomic `complete.json` visibility gate kept every published scene
  intact). Final ledger: **324 done, 0 failed**;
  `scripts/validators/validate_feature_stacks.py --root gs://berlin-lst-data/features/v2
  --expected-scenes 324` passes **324/324** with **88,282,271**
  feature-valid px. VM stopped (`TERMINATED`). A bounded cloud smoke gate
  (`nox -s cloud-smoke-features`) guards the same-process GCS/GDAL
  read-after-write path; the GDAL directory-cache root cause is documented
  in `data/features/product.py`.
- Sparse-support diagnostics: the isolated summary emits a non-gating
  `sparse_support_below_1pct` list (fraction `feature_valid_px /
  inside_aoi_px < 1%`). It flags `194023_20170929` (79 feature-valid px,
  ~0.0009%). `193023_20170821` and `193024_20170821` sit just above the
  threshold at ~1.40 % (124,708 px each) and are retained as sparse
  support regardless. All three are structurally valid published stacks;
  sparse support is diagnostic evidence, not a feature-stage exclusion —
  Stage 2 alone decides `training_eligible@100m`.
- The Stage-1 evidence (`qa/stage1_raw/9518fe0c/`) is the diagnostic
  baseline the feature stacks build on; the feature mask is AOI-aware
  (outside-Berlin pixels are masked, not counted as data gaps) and the
  per-scene coverage metrics split inside/outside AOI.
- Coverage vs. the AOI: provenance `aoi_frac` is 0.488874 — the canonical
  bbox rectangle is ~48.9 % inside the exact Berlin polygon. The 51.9 %
  `valid_frac` reported by the morphology products is their data-coverage
  extent (the source data covers the polygon plus a small buffer beyond it,
  ~51.9 % of the rectangle), not the polygon share. The `feature_valid`
  mask applies the polygon, so no outside-Berlin pixel is ever valid.
- The superseded 24-band V1 stacks were deleted from GCS after V2 passed
  both gates (`features/v1` prefix empty); historical QA evidence under
  `qa/` is retained.

## V3 semantic-availability correction (delivered)

The V2 release's all-channels-NaN validity rule destroyed ~95 % of the
AOI: the LoD2 source writes NaN for every non-building cell, and V2
blanked all 28 channels wherever one channel was unavailable. V3 fixes
the semantics at the source of the validity rule:

- **Per-channel availability.** Only unavailable bands are NaN (e.g. the
  S2 bands and derived indices under a non-clear flag); static and
  dynamic channels keep their values. All channels are NaN only outside
  the AOI. `feature_valid` remains the aggregate complete-vector
  availability mask.
- **LoD2 no-building vs source gap.** The four LoD bands are `0` in
  source-covered cells without a building and `NaN` in cells outside the
  LoD source-tile coverage. Coverage is reconstructed at composition time
  from the immutable raw archive manifests (2017 LoD1 ∩ LoD2, 2021,
  2022) and the 2024 LoD provenance tile receipts
  (`data/features/lod_coverage.py`); finite source values always win.
- **Schema v3, root `features/v3`.** The feature config hash now covers
  the static-sources ledger and the per-vintage LoD coverage + COG
  content fingerprints, so any change to these inputs invalidates the
  published stacks.
- Full run (2026-08-24, VM, `run_features_vm.sh` @ `03e085e`): **324/324
  stacks**, 0 failed; independent validator green with
  **1,584,712,041 feature-valid px** (vs 88,282,271 in V2); full V2→V3
  comparison green (every V2-valid pixel bit-identical, every newly valid
  pixel has zero LoD bands).
- Stage-2 QA on V3 (evidence `gs://berlin-lst-data/qa/stage2_features/cc00406a/`):
  **345 pairings, 324 assessed, 21 expected 2026 exclusions, 0 findings**,
  `ok: true`. `feature_valid_px` 1,584,712,041 (cross-validated);
  target-valid 100 m cells 20,169,061; **full-support cells 5,520,165**
  (all-100 == full-support; up from 742 in V2, where full support
  required all-100 building pixels). VM stopped (`TERMINATED`).
- Smoke gates (`smoke-features`, `cloud-smoke-features`, `smoke-qa-stage2`)
  cover one scene per LoD vintage on a canonical-aligned window with
  every semantic case (building, covered no-building, source gap, clear,
  flagged).
- V2 retirement plan recorded at
  `gs://berlin-lst-data/qa/retirements/v2/<timestamp>/plan.json`
  (2,596 objects, 28,694,365,422 bytes) — deletion happens only after
  explicit inventory-hash confirmation; historical `qa/` evidence is
  retained.

## Stage-2 feature-stack QA gate (WB2c-2, delivered)

The reusable QA gate was applied to the published feature stacks (identical
logic to Stage-1, on all stack channels). Read-only over `features/v3`;
evidence written to `qa/stage2_features/<run-id>/`. No validity or
selection mask is produced — `training_eligible@100m` is a WB2c-4
(training-data preparation) decision.

- Run: `qa-stage2-20260824T213350Z` (2026-08-24, VM), evidence
  `gs://berlin-lst-data/qa/stage2_features/cc00406a/` —
  `summary.json`, `scenes.parquet/csv`, `profiles.parquet/csv`.
- Result: **345 pairings, 324 assessed, 21 excluded** (all
  `dynamic role=inference (2026)`), **0 findings**, `ok: true`.
- Aggregate (canonical EPSG:25833 grid): `feature_valid_px` 1,584,712,041;
  target-valid 100 m cells 20,169,061; full-support cells 5,520,165
  (all-100 == full-support).
- Independent validator (`scripts/validators/validate_qa_stage2_features.py`) green:
  all source fingerprints verified, no `.tif` artifact under the prefix.
- Per-scene-channel profiles (fixed-bin histograms + count/min/max/mean/std)
  are the diagnostic record for WB2c-4; no values were filtered or removed.
- VM stopped (`TERMINATED`). The V2-era run
  (`qa-stage2-20260822T152539Z`, evidence `35eb283e`) remains under
  `qa/stage2_features/` as historical evidence but refers to the
  superseded V2 stacks.

## WB2c-4 training-data release (delivered)

The approved training contract is implemented as a reproducible release
under `gs://berlin-lst-data/training/v1` (pipeline `data/training/`,
runner `scripts/runners/run_training_data.py`, independent validator
`scripts/validators/validate_training_data.py`, smoke gate `nox -s smoke-training-data`,
VM wrapper `scripts/run_training_data_vm.sh`). Full normative details are
in `data-sources-and-contracts.md` § WB2c-4 training-data release.

- **Input basis pinned to Feature Release V3** (324 done scenes, config
  hash `d9eb25995b2f4911`); V1/V2 are hard-rejected.
- **Temporal contract:** 2017-2023 train, 2024 validation, 2025 test,
  2026 inference (metadata-only, `inference_deferred`).
- **Eligibility:** strict 100/100 support; scenes with zero eligible
  cells are excluded with `no_eligible_cells` (no sparse category).
- **Cells:** stable spatial `cell_id` from the canonical EPSG:25833 grid.
- **Scaler:** train-only, eligible cells only; z-score, log1p+z-score for
  precipitation, identity for shadows; Welford population statistics.
- **QA:** independent validator + publisher-side readback; deterministic
  smoke; guarded VM publication.

**2026 decision (user, 2026-08-26).** The 21 raw 2026 anchor pairings
are retained in the manifest/ARD but deliberately **not** materialised
into V3 features now. They are metadata-only in this release. After the
model is trained, a dedicated inference-preparation task reuses the
feature composer (not a second methodology) to produce 2026 features
under an inference-specific release root, then the trained model runs
over them.

## Next steps (separate session)

- WB3 training: consume `training/v1` (features, eligibility, splits,
  cells, scaler) — patch geometry, samplers, model training, spatial CV
  and Zarr materialisation are WB3 scope.
- Later: 2026 inference-preparation (V3 features for 2026 + trained-model
  application).
