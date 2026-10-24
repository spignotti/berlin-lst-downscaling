# v6 republish chain (selection v4 → ARD v10 → features/v6 → training/v3)

One VM run per step, in order, launched from the operator's Mac via
`start-vm-run-tab.sh` (fail-closed lifecycle in `vm-runner-common.sh`:
start → deploy pinned commit → detached launch → poll → validate → stop;
the VM is stopped after each run). Nothing here has run yet — this page
is the runbook for the later VM executions.

## Upstream decision (Landsat/ECOSTRESS)

Landsat and ECOSTRESS ARD are **reused where they verify, reprocessed
where they do not**. Per `src/berlin_lst_downscaling/data/ard/contract.py`
(`_SCHEMA_VERSIONS`) and `docs/data-sources-and-contracts.md`, both
contracts are unchanged: `landsat-c2-l2` schema 7 (area-mean `lwir11`,
native bit-OR `qa_pixel`) and `ecostress` schema 7. The published rows
live in `gs://berlin-lst-training-data/ard/full/2017-2026-qa-avg-20261019`.
Step 2 automates the seeding: `run-ard-v10-vm.sh` first runs
`scripts/operators/seed_ard_v10_reuse.py`, which copies only the fully
verified `done` Landsat/ECOSTRESS rows present in the v4 bundle into the
new v10 ledger (COGs stay referenced in place, no COG is copied) and
records every other v4 reuse key — missing row, non-done row, or done
row with an empty COG/flag or a COG/flag missing at its resolved
canonical URI — as a processing gap in the full machine-readable gap
list (`seed_gaps.json` plus `GAPS_JSON` in the log; the human summary
line shows at most 10). Existence is checked on
`resolve_canonical_uri(path)` (`src/berlin_lst_downscaling/data/io/uri_mapping.py`,
see `docs/gcs-inventory-and-transfer.md`): the 6 ECOSTRESS reuse rows
that reference the retired `gs://berlin-lst-data` bucket (pre-cutover
provenance) verify against their byte-identical copies under the
canonical bucket and are seeded with the stored URIs kept unchanged as
provenance — downstream readers resolve stored URIs at read time, and
paths are never rewritten. Seeding them (instead of gap-processing)
matters because the gap pass would need a NASA Earthdata login
(`earthaccess.login()`), which does not exist on the VM, so those
scenes could not be reprocessed and `validate_ard` would fail. A
`gs://` URI on any other (foreign) bucket aborts the seed fail-closed
— a hard error, never a gap — consistent with `uri_mapping`'s
fail-closed intent. A non-done (failed/pending) destination row for a
key that now verifies from the reuse ledger is replaced by the verified
reuse row (logged), while a differing `done` destination row still
aborts the seed before anything is written. The launcher then runs the
S2-only `run_ard` (warm mosaic config, keeps seeded rows, processes
only S2) followed by the `landsat_ecostress_v10` gap pass (same
contracts, schema, and policy hash), which processes exactly the gap
scenes into the v10 root — a rerun is a no-op for seeded and processed
rows. Contract/hash mismatches and refusals to overwrite a differing
existing row stay hard errors: the seed writes nothing (neither ledger
nor gap file) and aborts. If either contract ever bumps, the seeded rows
must instead be reprocessed into a new root.

## Run order and launcher commands

| # | Step | Launcher command (from repo root) | Config | Expected outputs |
|---|------|-----------------------------------|--------|------------------|
| 1 | Selection v4 bundle | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-selection-v4-vm.sh main` | `configs/selection/full_2017_2026_v4.yaml` | `manifests/v4/2017-2026-cutoff-20260717T235959Z/{manifest.parquet,pairings.parquet,manifest_report.json}` |
| 2 | S2 ARD v10 + Landsat/ECOSTRESS gaps | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-ard-v10-vm.sh main` | `configs/ard/sentinel2_mosaic_v10.yaml` (`manifest_uri=` v4 bundle) then `configs/ard/landsat_ecostress_v10.yaml` (same roots) | `ard/full/2017-2026-s2-mosaic-v10/{*.cog,ledger.parquet,seed_gaps.json,...}` (S2 new; Landsat/ECOSTRESS seeded where verified, gap scenes processed, see above) |
| 3 | Static sources v6 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-static-sources-v6-vm.sh main` | `configs/static_sources/v6.yaml` | `static/sources/lod2-area-exact-v2/{ledger.parquet,...}` (LoD2 area-exact v2 incl. BuildingPart, canopy cover; imperviousness COGs stay published v3, newly pointed at) |
| 4 | Static derived v6 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-static-derived-v6-vm.sh main` | `configs/static_derived/v6.yaml` (`svf_max_radius: 10`, i.e. 100 m) | `static/derived/lod2-area-exact-v2/{_state/.../ledger.parquet,...}` (building/vegetation/combined DSMs, horizons, SVF) |
| 5 | Dynamic v6 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-dynamic-v6-vm.sh main` | `configs/dynamic/v6.yaml` (`manifest_uri=` v4 bundle) | `dynamic/lod2-area-exact-v2/{_state/.../ledger.parquet,...}` (shadows, ERA5 scene_value) |
| 6 | Features v6 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-features-v6-vm.sh main` | `configs/features/v6.yaml` (ARD v10 + v4 manifest) | `features/v6/{<scene_id>/*.cog,ledger.parquet,qa/...}` (22-channel roster) |
| 7 | Training v3 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-training-v3-vm.sh main` | `configs/training/v3.yaml` (from features/v6) | `training/v3/{manifest.parquet,cells.parquet,scaler.json,complete.json,...}` |
| 8 | Patch index v3 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-patch-index-v3-vm.sh main` | `configs/training/patch_index_v3.yaml` (from training/v3) | `training/patch-index/v3/{patch_index.parquet,qa.json,complete.json}` |
| 9 | EDA v3 | `.opencode/skills/google-access/scripts/start-vm-run-tab.sh run-training-eda-v3-vm.sh analysis/eda-v1` | `configs/eda/v3.yaml` (flags for `scripts/operators/run_training_eda.py`) | `evaluation/eda/v3/{tables,summary.json,observations/...}` |

Each launcher takes an optional `[branch]` (default `main`, step 9
defaults to `analysis/eda-v1` like its v2 template). Every launcher runs
`scripts/operators/preflight_republish_v6.py --step <step>` locally
before touching the VM and refuses to launch on any finding.

### Rerunning step 1 after a partial write

The v4 bundle root is immutable once `manifest.parquet` exists: the
selection preflight refuses to launch while the bundle URI exists, even
for a partial prefix (parquets present, `manifest_report.json` missing —
as after the failed run `selection-20261023T160311Z-36246C2F`, which
crashed hashing the GCS parquets with local `open()` before the report
was written). Before a rerun the operator deletes the partial prefix
(e.g. `gcloud storage rm -r
gs://berlin-lst-training-data/manifests/v4/2017-2026-cutoff-20260717T235959Z/*`).
The launcher never auto-deletes GCS data.

## Rough runtimes

No previous run of this exact chain exists, so there are no measured
step times. Poll budgets per launcher are upper bounds, not estimates:
selection/ dynamic/ training/ EDA 18 h (`64800`), patch index 6 h
(`21600`), ARD v10 / static sources / static derived / features 48 h
(`172800`) because each touches hundreds of scenes (S2 multi-tile
mosaics, LoD2 area-exact rebuild, full-Berlin DSM/SVF, 22-channel
compose). Feature build peaks at about 7–8 GB RAM per scene; the
`n2-highmem-2` VM covers that. The operator records actual runtimes here
after each run.

## Checks after each step

1. The launcher prints `SUCCESS` and the release root; otherwise it
   prints `FAIL` with the remote log path and leaves the VM state per
   `vm-runner-common.sh` (stopped on completion/failure, left `RUNNING`
   only on connection loss or ambiguous exit — then inspect with
   `status-dynamic-vm.sh --run-id <id>` before stopping).
2. Confirm the VM is `TERMINATED` (`status-vm.sh`) after the launcher
   stops it.
3. Confirm the step's completion marker exists in GCS:
    - steps 1–2: `manifest.parquet` + `pairings.parquet` +
      `manifest_report.json` (step 1), `ledger.parquet` plus `seed_gaps.json`
      (step 2, `validate_ard.py --ledger … --manifest …`, whose per-source
      done counts fail a skipped gap pass). S2 rows failed by the mosaic AOI
      coverage gate (`max_missing_aoi_fraction = 0.005`, `last_error`
      `Sentinel-2 mosaic AOI coverage failed: missing …`) are explicit,
      documented exclusions: the validator reports each with scene id and
      missing fraction and the release passes only if every non-done S2 row
      is such a gate exclusion and all Landsat/ECOSTRESS rows are done
      (ledger keeps `status="failed"`, no schema change; see
      `src/berlin_lst_downscaling/data/ard/coverage_gate.py`). ARD v10 run3
      (run `ard-v10-20261024T040705Z-83AB242C`, release
      `gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10`)
      has exactly two such exclusions, confirmed against the v4 bundle
      (`pairings.parquet`, 290 rows): 4 pairings use the two excluded S2
      scenes, so 286 pairings remain —
      `LC08_L2SP_193023_20210816_02_T1` and
      `LC08_L2SP_193024_20210816_02_T1` ↔
      `S2A_MSIL2A_20210814T102031_R065_T32UQD_20210814T215405` (missing
      560178/8922403 AOI px, 6.28 %), `LC09_L2SP_192023_20230503_02_T1` and
      `LC09_L2SP_192024_20230503_02_T1` ↔
      `S2A_MSIL2A_20230503T101021_R022_T32UQD_20240905T123554` (missing
      1629098/8922403 AOI px, 18.26 %). The 4 excluded pairings drop out of
      features v6 (see step 6); no replacement S2 partner is picked. The launcher
      and validator point GDAL at Application Default Credentials when no
      GDAL GCS credential is set, so `gs://` COG reads work on the operator
      Mac without manual env setup;
    - steps 3–4: `ledger.parquet` under the release root
      (`_state/...` for derived), all rows `done`;
    - step 5: `_state/dynamic/ledger.parquet` all rows `done` plus
      `validate_dynamic.py --output-root … --expected-role anchor
      --expected-scenes-from-manifest <v4 manifest>` (derived from the v4
      bundle, never hard-coded, so the AOI-overlap gate passes without
      manual edits);
   - step 6: `ledger.parquet` all `done` plus
     `validate_feature_stacks.py --root … --expected-scenes-from-manifest
     <v4 manifest> --ard-ledger <v10 ledger>` (expected count is the
     manifest-derived assessable universe minus pairings whose S2 partner is
     an ARD coverage-gate exclusion; excluded pairings drop out, no
     replacement partner is picked — the conservative rule consistent with
     the agreed 0.5 % gate. Dynamic v6 needs no adjustment: it is
     per-Landsat-anchor and never reads S2 ARD, so its manifest-derived
     assessable count is unchanged);
   - step 7: `complete.json` with the v6 policy hash plus
     `validate_training_data.py --release-root … --expected-scenes-from-manifest
     <v4 manifest>` (the count is derived from the v4 bundle, never
     hard-coded, so the AOI-overlap gate passes without manual edits);
   - step 8: `complete.json` plus
     `validate_patch_index.py --index-root … --source-root …`;
   - step 9: uploaded `evaluation/eda/v3` tables plus
     `validate_training_eda.py --eda-dir … --manifest
     training/v3/manifest.parquet`.
4. Confirm the validator output is green before starting the next step;
   a red validator blocks the chain (fix forward, never rewrite a
   published root).
5. `publish_guard` (`src/berlin_lst_downscaling/data/ard/publish_guard.py`)
   protects the new roots from legacy writes (S2 schema < 10 into the
   v10 root, feature schema ≠ 5 into features/v6, any write into the
   published v1/v2, patch-index v1/v2, and manifests/v3 roots). The new
   roots join the unconditional frozen sets in a follow-up once this
   chain is published (one line each).
