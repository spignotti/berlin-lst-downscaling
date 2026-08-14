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
`scripts/validate_ecostress_scenes.py` probe enforces, on the published
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

The Stage-1 raw-input gate (`scripts/run_qa_stage1_raw.py`,
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

## Next steps (separate sessions)

- Scene feature-stack derivation (`WB2c-3` in Notion) — consume the
  Stage-1 evidence (`qa/stage1_raw/9518fe0c/`) as the diagnostic
  baseline; the feature stack re-runs the same gate logic in Stage 2.
  The six-band S2 ARD (B11/B12 included) is the spectral input basis.
- Stage-2 feature-stack QA gate (Notion `WB2c-2`, Stufe 2) — identical
  gate logic on the derived feature stacks, plus channel-level range
  checks; Stage 2 owns the sole authority to publish the
  training-eligibility mask.

Both are planned in Notion and scheduled as their own sessions.
