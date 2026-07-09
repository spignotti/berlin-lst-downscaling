# Cloud Runbook

Pipeline runs against `gs://berlin-lst-data` in `europe-west3`.
Two paths: local-ADC smoke (validates the cloud IO) and a GCP VM full run.

## Bucket Layout (`gs://berlin-lst-data`)

```
boundaries/                  AOI masks + boundary (aoi_10m.tif, aoi_100m.tif, berlin_landesgrenze.geojson)
manifests/                   Selection manifests uploaded for a cloud full run
ard/                         Final ARD COGs + STAC + ledger
  {source}/{year}/{scene_id}/<scene_id>.tif|.flag.tif|.stac.json
  ledger.parquet
  smoke/                      Cloud-pilot smoke outputs
_staging/                    Ephemeral raw staging (ECOSTRESS downloads)
  ecostress/{run_id}/
```

Staging is ephemeral — `StageSession.cleanup()` deletes it on exit.
Smoke outputs under `ard/smoke/` are kept for inspection; delete manually if disk pressure is needed.

## Prerequisites

- `uv` on PATH (project uses [uv](https://docs.astral.sh/uv/) for env + scripts)
- `.env` at repo root with:
  - `GOOGLE_APPLICATION_CREDENTIALS` (path to service-account JSON key)
  - `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` (NASA Earthdata Login for ECOSTRESS)
- A GCS service account with `roles/storage.objectAdmin` on `gs://berlin-lst-data`

`uv run` auto-loads `.env`, so `EARTHDATA_*` reach the Python process without extra wiring.
`rclone` is configured (`~/.config/rclone/rclone.conf`) with the `gcs-masterarbeit:` remote.

## Local — Cloud Pilot (ADC)

Validates the full cloud IO path against a 3-scene smoke manifest
(Landsat, Sentinel-2, ECOSTRESS). ECOSTRESS downloads to a GCS stage,
every source writes COGs to `gs://`, and AOI is read from `gs://`.

```bash
uv run nox -s cloud-pilot
```

Reads:

- ECOSTRESS from `gs://berlin-lst-data/_staging/ecostress/{run_id}/`
- AOI masks from `gs://berlin-lst-data/boundaries/`
- Planetary Computer STAC (Landsat + S2; no credentials)

Writes:

- COGs + STAC + ledger to `gs://berlin-lst-data/ard/smoke/smoke_primary/`

After the smoke, step 5 of the session prints the blob count and names.

## Local — Upload a Manifest for a Cloud Full Run

Build the manifest locally (selection `couple` mode), push to GCS, then pass the
GCS URI to the pipeline running on a VM.

```bash
# 1. Build full manifest locally
uv run python scripts/run_selection.py --config-name full_2017_2025 couple=true

# 2. Upload to GCS
uv run nox -s upload-manifest -- data/selection/manifest.parquet
# → prints: Uploaded: gs://berlin-lst-data/manifests/manifest.parquet
```

## Local — Inspect the Bucket

```bash
rclone tree --max-depth 3 gcs-masterarbeit:berlin-lst-data
mount-berlin   # mount as filesystem at ~/.mnt/berlin-lst
```

## GCP VM — Full ARD Run (Next Stage)

Provision a VM in `europe-west3` (same region as the bucket — same-region egress is free).

1. **VM**

   ```bash
   gcloud compute instances create berlin-lst-runner \
     --zone=europe-west3-a \
     --machine-type=n2-standard-8 \
     --image-family=debian-12 --image-project=debian-cloud \
     --boot-disk-size=100 \
     --service-account=masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com \
     --scopes=cloud-platform \
     --preemptible
   ```

2. **Provision the VM**

   ```bash
   gcloud compute ssh berlin-lst-runner -- apt install -y python3.12 curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/<user>/master-thesis.git && cd master-thesis
   uv sync
   # .env with GOOGLE_APPLICATION_CREDENTIALS + EARTHDATA_USERNAME/PASSWORD
   # (use Secret Manager or scp)
   ```

3. **Run**

   ```bash
   uv run nox -s upload-manifest -- data/selection/manifest.parquet
   uv run python scripts/run_ard.py --config-name full \
     manifest_uri=gs://berlin-lst-data/manifests/manifest.parquet \
     output_root=gs://berlin-lst-data/ard \
     ecostress.stage_base=gs://berlin-lst-data/_staging/ecostress \
     aoi.mask_base=gs://berlin-lst-data/boundaries
   ```

   Resume is automatic via the ledger at `gs://berlin-lst-data/ard/ledger.parquet`.

4. **Cleanup**

   ```bash
   gcloud compute instances delete berlin-lst-runner --zone=europe-west3-a
   ```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `earthaccess.login()` fails | `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` in `.env`; interactively run `python -c "import earthaccess; earthaccess.login()"` to bootstrap the token |
| `Bucket not found` from rclone | `~/.config/rclone/rclone.conf` must set `project_number = 469137882515` |
| `GOOGLE_APPLICATION_CREDENTIALS` not loaded | `uv run` reads `.env` automatically; verify with `uv run python -c "import os; print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))"` |
| `rclone ls gcs-masterarbeit:` hangs | IPv6 DNS issue; force IPv4 (see Lessons in `.opencode/PROJECT_STATE.md`) |
| Pipeline fails with `ECOSTRESS L2T layer not found` after staging | Check `gs://berlin-lst-data/_staging/ecostress/{run_id}/`; expect `{granule_id}/{granule_id}_{LST,cloud,water,QC}.tif` |
