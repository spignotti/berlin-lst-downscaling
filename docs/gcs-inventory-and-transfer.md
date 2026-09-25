# GCS inventory and account transfer runbook

Records what must be preserved from the current GCS bucket, defines a
copy-first mirror procedure for a later move to a new GCP account, and
lists the cutover changes that a move would require.

This document is a record and a procedure. It performs no transfer,
changes no application path, and deletes nothing. The actual transfer is
a separate, explicitly approved operation.

## Bucket identity (observed 2026-09-23)

| Field | Value |
|-------|-------|
| Project | `masterarbeit-berlin-lst-v2` (only bucket in the project) |
| Bucket | `gs://berlin-lst-data` |
| Location | `EUROPE-WEST3` (regional) |
| Default storage class | `STANDARD` |
| Uniform bucket-level access | enabled |
| Object versioning | not enabled |
| Soft-delete retention | 7 days |
| Created | 2026-06-28 |

The region matches the pipeline VM (`europe-west3-a`). A destination
bucket in the same region avoids cross-region egress; a different region
is possible but costs more to read from the VM.

## Snapshot

Read-only inventory at `2026-09-23T10:17:25Z`: **12,567 objects /
246,174,774,802 bytes** (about 229.3 GiB) in the whole bucket.

Treat this as a dated baseline, not a transfer total. It changes when
pipelines run, and the mirror must refresh it before copying rather than
reuse the numbers below. Reconcile any difference per prefix instead of
trusting the bucket-level sum.

## What must be copied

`Count` and `Bytes` are the observed snapshot for that prefix. `Class`
is the disposition for the mirror.

| Prefix | Count | Bytes | Class | Role |
|--------|------:|------:|-------|------|
| `manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/` | 3 | 50,614 | canonical | Immutable scene universe: `manifest.parquet`, `pairings.parquet`, `manifest_report.json` (509 scenes / 345 pairings) |
| `boundaries/` | 3 | 535,808 | canonical | `aoi_10m.tif`, `aoi_100m.tif`, `berlin_landesgrenze.geojson` |
| `lod_vintages/` | 3 | 4,422,764,488 | canonical | SHA-256-pinned LoD archives for 2017, 2021, 2022 |
| `ard/` | 2,552 | 39,288,153,420 | canonical | 509 ARD scenes plus `ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet` |
| `static/` | 1,356 | 10,792,684,788 | canonical | Sources, derived products, and `static/geometry_vintages/v1/geometry_mapping.json`; includes both `ledger.parquet` files |
| `dynamic/full/` | 4,279 | 24,477,662,271 | canonical | `role=anchor` ERA5-Land and shadow products, including `_state/dynamic/ledger.parquet` |
| `dynamic/inference/2026/` | 302 | 1,391,235,470 | canonical | `role=inference` products, including their `_state` ledger |
| `features/v3/` | 2,594 | 152,375,498,958 | canonical | 324 published 28-band stacks plus `_state/features/ledger.parquet` |
| `training/v1/` | 982 | 1,191,080,151 | canonical | Eligibility masks, manifest, cells, scaler, release marker, `_state/training/ledger.parquet` |
| `qa/stage1_raw/` | 5 | 129,210 | evidence | Stage-1 run `9518fe0c`, `summary.json` |
| `qa/stage2_features/` | 21 | 7,799,922 | evidence | Stage-2 runs `cc00406a` (V3), `35eb283e` (V2 era), `0c8c8144` (see notes), plus `logs/` sidecars |
| `qa/cloud_masking/` | 42 | 65,142,561 | evidence | Three descriptive cloud-mask audit run roots |
| `qa/repairs/` | 394 | 12,154,297,863 | evidence | `s2-snow-ice-20260811T215612` repair evidence |
| `qa/retirements/` | 2 | 999,464 | evidence | V2 retirement `plan.json` and `receipt.json` |
| `qa/wb2c-2/` | 12 | 160,263 | evidence | Pre-decoupling Stage-1 evidence, kept under the historical prefix |
| `profiling/wb2c-1/` | 7 | 4,634,894 | evidence | Feature profiling profiles and logs |
| `dwd_validation/` | 7 | 1,895,342 | evidence (historical) | DWD r3 comparison, retained; not evidence for the current 8-band ERA5 fields |

Each canonical root carries its ledgers and `_state` with it, and the
mirror must copy them as part of the root:

- `ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet`
- `static/sources/full/ledger.parquet`,
  `static/derived/full/_state/static/derived/ledger.parquet`
- `dynamic/full/_state/dynamic/ledger.parquet`,
  `dynamic/inference/2026/_state/dynamic/ledger.parquet`
- `features/v3/_state/features/ledger.parquet`,
  `training/v1/_state/training/ledger.parquet`

`ard/full/` holds one cutoff root. `static/` splits into `sources/`
(1,274 objects / 7,124,990,465 bytes), `derived/` (81 objects /
3,667,692,802 bytes), and `geometry_vintages/` (1 object). The two
dynamic roots also carry ERA5-Land monthly caches under `_raw/`:
`dynamic/full/_raw/` (54 objects / 25,128,719 bytes) and
`dynamic/inference/2026/_raw/` (4 objects / 1,817,339 bytes). Those
caches are the reproducibility basis for the dynamic products, so they
belong to the canonical set even though they sit under `_raw/`.

The main table plus the non-`-r2` manifest bundle sums to the bucket
total of 12,567 objects / 246,174,774,802 bytes, with nothing left over
(verified 2026-09-23). A refresh is consistent when it reconciles the
same way.

### Historical and superseded

These are valid objects that no current pipeline reads. The runbook
copies them by default, because they are small relative to the canonical
set and they document how the current state was reached.

- `manifests/v3/2017-2026-cutoff-20260717T235959Z/` (no `-r2`), 3 objects
  / 49,315 bytes, is a superseded bundle generated 2026-07-18 with the
  same 509/345 counts as `-r2`. This is the only prefix in this section
  that is absent from the main table.
- `qa/stage2_features/35eb283e/` (5 objects / 2,617,313 bytes) is V2-era
  Stage-2 evidence.
- `dwd_validation/r3/` is historical scalar-ERA5 validation, retained
  after the DWD subsystem was removed.

The last two are subsets of rows already listed in the main table, so do
not add them again when totalling the mirror.

`features/v1/` and `features/v2/` no longer exist. The V2 retirement was
executed on 2026-08-25 (2,596 objects / 28,694,365,422 bytes) and its
`receipt.json` is under `qa/retirements/v2/20260825T105142Z/`. The
`plan.json` records the confirmed inventory hash.

### Ephemeral and excluded

The smoke roots (`features/smoke/`, `dynamic/smoke/`, `static/smoke/`)
hold no objects in this snapshot and are recreated per run. They are not
mirrored. Run outputs the pipeline creates later are not part of this
inventory.

### Notes on classification

- `qa/stage2_features/0c8c8144/` is not referenced in the repo docs. It
  is a Stage-2 run dated 2026-08-20 with `ok: true`, 324 assessed scenes,
  `feature_valid_px` 1,628,157,599, and no findings. That count differs
  from the released V3 run (`cc00406a`, 1,584,712,041). It predates the V3
  publication and is treated as retained interim evidence, not as the V3
  QA record. Copy it as evidence; do not read it as the current figure.
- The `-r2` manifest bundle is the canonical one; the earlier bundle is
  history only.
- Categories here come from the bucket listing plus the documented roots
  in `docs/phase-1-delivery.md` and `docs/phase-2-preparation.md`. No
  prefix in the bucket is left unclassified.

### Observed counts vs. release-report counts

The bucket counts above are object counts. The release reports state
domain counts that do not match object counts directly, and both are
correct for their purpose:

- `manifests/v3/...-r2/manifest_report.json`: 509 scenes, 345 pairings.
- `training/v1`: 982 objects; the run report states 324 processed scenes,
  0 failed, and 5,520,165 eligible cells.
- `qa/stage2_features/cc00406a/summary.json`: `ok: true`, 324 assessed,
  0 findings, `feature_valid_px` 1,584,712,041.

Verification compares objects between source and destination. It does not
compare domain counts to object counts.

## Mirror runbook

### 0. Principles

- Copy first, keep the source intact. No step here deletes or overwrites
  source objects.
- A completed copy does not authorize a cutover. Changing readers is a
  later, separately approved step (see below).
- The destination project and bucket are recorded under §1 below
  (2026-09-25). They stay preflight inputs and are fixed before a copy
  starts.
- The remaining old-account credit and its expiry are only visible in the
  Cloud Billing Console. Confirm them and record the cutoff date; they
  cannot be read from an API.

### 1. Preflight (record all of these before copying)

1. Destination project ID and bucket name. The destination bucket name
   must be globally unique; it may reuse `berlin-lst-data` only after the
   source bucket is deleted, which is not part of this runbook.
2. Destination region `EUROPE-WEST3` unless a different region is a
   deliberate decision with the egress cost accepted.
3. IAM on the destination: a principal with object-create rights on the
   new bucket. On the source: object-viewer rights. Grant roles to a
   service account or user; the runbook never carries credentials.
4. Billing enabled on the destination project, and a cost estimate. The
   copy itself does not bill a transfer fee for agentless bucket-to-bucket
   jobs, but it does incur destination storage, Class A write operations,
   listing on both sides, and any cross-region network use.
5. The mirror set: canonical roots plus the retained evidence roots, and
   an explicit decision on the historical roots.
6. The bucket snapshot refreshed at copy time (see below), so the
   destination can be checked against the exact source state.

### Destination (recorded 2026-09-25)

| Field | Value |
|---|---|
| New account identity | new account owner (personal Google identity; deliberately not named in this public repo). It has no access to the old project or bucket. |
| Billing | new billing linked; budget `berlin-lst-training budget` at 250 EUR with alerts at 50/90/100 %. Billing-account ID is deliberately not recorded in this public repo. |
| New credit | active, expires 2026-12-25 |
| Old credit cutoff | expires Sunday 2026-09-27 (console-only value; the deadline for the mirror) |
| Project ID | `berlin-lst-training` |
| Project number | `996559849187` |
| Bucket | `gs://berlin-lst-training-data` |
| Location | `EUROPE-WEST3` (regional, same as source) |
| Default storage class | `STANDARD` |
| Uniform bucket-level access | enabled |
| Object versioning | disabled |
| Soft-delete retention | 7 days (604800 s, matches source) |
| Copy principal (service account email) | `masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com` |
| Role on the source bucket | pre-existing `roles/storage.objectAdmin` (the old project's runner SA; the source owner grants are untouched) |
| Role on the destination bucket | `roles/storage.objectAdmin` (temporary, revoked after the mirror is verified and no later than 2026-09-27) |
| Role on the destination project | `roles/serviceusage.serviceUsageConsumer` (quota-project access only; temporary, revoked together with the destination bucket binding) |
| New VM principal (service account email) | `berlin-lst-vertex@berlin-lst-training.iam.gserviceaccount.com` |
| Role of the new VM principal | `roles/storage.objectAdmin` on the destination bucket only; no source access |

The new account cannot read the old project, so the old project's runner
service account is the copy principal: it already holds read/write on the
source bucket and receives a **temporary** destination grant (bucket
`objectAdmin` plus project `serviceusage.serviceUsageConsumer`) that is
revoked once the mirror is verified, and no later than the 2026-09-27
credit cutoff. This avoids creating a credential or touching the old
project's IAM. The new VM principal is a separate identity with
destination-only access.

Asymmetry to respect: the copy principal can **write and delete on the
source bucket** (its pre-existing `objectAdmin`, which this runbook does
not narrow). The "no source write/delete" rule is therefore procedural,
not IAM-enforced. Every copy command must be direction-checked before it
runs, and verification includes a source spot-checksum to show the source
bytes did not change.

### 2. Refresh the source inventory

Run these read-only commands and keep the output as the copy baseline.

```bash
# Bucket-level total
rclone size gcs-masterarbeit:berlin-lst-data --json

# Per prefix (repeat per top-level prefix)
rclone size gcs-masterarbeit:berlin-lst-data/features/v3 --json

# Confirm a prefix exists: `rclone size` reports 0 for a missing prefix too
rclone lsf gcs-masterarbeit:berlin-lst-data/features --dirs-only

# gcloud equivalent for a prefix
gcloud storage du gs://berlin-lst-data/features --project=masterarbeit-berlin-lst-v2 --summarize
```

### 3. Copy (non-deleting)

Dry run first, then the real copy. Replace `<DEST_BUCKET>`; the source
path never appears as a destination.

```bash
# Preview: shows what would be copied, changes nothing
gcloud storage rsync gs://berlin-lst-data gs://<DEST_BUCKET> \
    --recursive --dry-run

# Copy
gcloud storage rsync gs://berlin-lst-data gs://<DEST_BUCKET> --recursive
```

`gcloud storage rsync` between two buckets compares MD5 or CRC32C
checksums and skips unchanged objects, and the CLI validates each copy
and removes a bad copy if the checksum disagrees. Do not pass
`--delete-unmatched-destination-objects`; without it the command only
adds and updates at the destination.

For a large or long-running job, Storage Transfer Service is an
alternative: an agentless GCS-to-GCS job has no service fee, and a rerun
skips objects already present at the destination, but the same storage,
operation, and network charges apply.

### 4. Verify the destination

Verification is independent of the copy. Compare the two sides and
investigate any difference rather than assuming success.

```bash
# Per-prefix object count and byte total on both sides
rclone size gcs-masterarbeit:berlin-lst-data/features/v3 --json
rclone size <dest-remote>:<DEST_BUCKET>/features/v3 --json

# Spot-read the critical markers (must exist and parse)
gcloud storage cat gs://<DEST_BUCKET>/training/v1/complete.json
gcloud storage cat gs://<DEST_BUCKET>/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest_report.json
gcloud storage cat gs://<DEST_BUCKET>/qa/stage2_features/cc00406a/summary.json
```

Check, in order:

1. Every mirrored prefix has the same object count and byte total.
2. The critical markers read back: `training/v1/complete.json`, the `-r2`
   `manifest_report.json`, and the Stage-2 V3 `summary.json`.
3. The release ledgers are present under their roots
   (`features/v3/_state/features/ledger.parquet`,
   `training/v1/_state/training/ledger.parquet`, and the ARD, static, and
   dynamic ledgers).

The repo validators do not yet point at the destination. In particular
`scripts/validators/validate_training_data.py` pins the features root to
the current bucket, so it cannot validate a copy until the cutover
updates that constant. Use counts, byte totals, checksums, and the marker
spot-reads for this step; the validators become usable after cutover.

### 5. Safety and rollback

- The copy is additive. If verification fails, correct the destination
  and copy again; the source is untouched throughout.
- Soft-delete retention on the source is 7 days, which bounds recovery
  from an accidental delete. That is a safety net, not a backup plan.
- Never run a destructive command against the source to "finish" a
  mirror. Source retirement is a separate decision made only after the
  destination is verified and the cutover is approved.

### 6. Spend cutoff

The remaining old-account credit is reserved for inventory, mirror, and
cutover only. Until the cutover completes, do not start:

- a full Stage-1 or ablation training run,
- a full features or dynamic re-run,
- a speculative Zarr training-cube build.

Record the cutoff date during preflight and treat it as the deadline for
the mirror and verification steps.

## Cutover (later, separately approved)

Copying objects does not rewire the pipeline. After a verified copy, a
cutover must update every reference to the old bucket. These are the
known references; treat the list as a starting point and re-grep before
the cutover.

- Configuration roots: `configs/features/_base.yaml`,
  `configs/training/_base.yaml`, `configs/qa/_base.yaml`,
  `configs/qa/stage2_features_full.yaml`, `configs/modeling/_base.yaml`,
  `configs/dynamic/full.yaml`, `configs/dynamic/inference_2026.yaml`,
  `configs/static_sources/full.yaml`, `configs/static_derived/full.yaml`,
  `configs/ard/full_sentinel2_swir.yaml`.
- Hard-coded roots in code and scripts:
  `src/berlin_lst_downscaling/data/training/release.py` (`V3_FEATURES_ROOT`),
  `src/berlin_lst_downscaling/data/dynamic/geometry.py`,
  `src/berlin_lst_downscaling/data/secondary/lod_vintages.py`,
  `scripts/validators/validate_training_data.py` (`_V3_FEATURES_ROOT`),
  `scripts/validators/validate_lod_coverage.py`,
  `scripts/operators/preflight_feature_release.py`,
  `scripts/operators/retire_feature_release.py`,
  `scripts/operators/compare_feature_releases.py`, `noxfile.py`.
  Smoke configs and smoke baselines are also affected: the
  `configs/*/smoke*.yaml` files and the smoke sessions in `noxfile.py`.
- Local and VM access: the bucket name in the `google-access` skill, the
  rclone remote `gcs-masterarbeit`, the ADC service-account key path, and
  the VM service account's bucket access.
- Provenance records under the copied roots reference the old bucket in
  historical metadata. Those are records of past runs; do not rewrite
  them to match the new bucket.

Also note that some existing references point at `features/v2`, which no
longer exists (for example the smoke baselines in `noxfile.py` and
`scripts/operators/compare_feature_releases.py`). Those are stale
regardless of the transfer and should be resolved by the cutover.

## Non-goals for the old account

- No full Stage-1 or ablation training runs.
- No speculative full Zarr training cube.
- No recompute of products that already exist and can be mirrored.

## References

- Product inventory and handoff state: `docs/phase-1-delivery.md`.
- Phase-2 QA state and the `training/v1` release: `docs/phase-2-preparation.md`.
- Source, grid, manifest, ledger, and artifact contracts:
  `docs/data-sources-and-contracts.md`.
- Google Cloud, move data between buckets:
  <https://cloud.google.com/storage/docs/moving-buckets>
- Google Cloud, data validation with checksums:
  <https://cloud.google.com/storage/docs/data-validation>
- Google Cloud, Storage Transfer Service pricing:
  <https://cloud.google.com/storage-transfer/pricing>
- `gcloud storage rsync` reference:
  <https://cloud.google.com/sdk/gcloud/reference/storage/rsync>
