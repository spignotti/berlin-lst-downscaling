---
name: google-access
description: Google Cloud Storage (rclone mount), ADC setup, and Compute Engine VM lifecycle for the berlin-lst-downscaling project
---

## Quick Reference

- Mount bucket:    `mount-berlin`
- List bucket:     `ls ~/.mnt/berlin-lst/`
- Start VM:        `.opencode/skills/google-access/scripts/start-vm.sh`
- Stop VM:         `.opencode/skills/google-access/scripts/stop-vm.sh`
- SSH into VM:     `.opencode/skills/google-access/scripts/ssh-vm.sh`
- Run Dynamic:     `.opencode/skills/google-access/scripts/run-dynamic-vm.sh <full|inference_2026> [branch]`
- Service account key: `~/.config/gcp-keys/masterarbeit-berlin-lst-v2.json`

## Purpose

This skill documents how to access Google Cloud resources for the berlin-lst-downscaling project: GCS bucket (via rclone mount, gcloud CLI, and Python), and ADC authentication.

## Account Info

| Field | Value |
|-------|-------|
| GCP project ID | `masterarbeit-berlin-lst-v2` |
| GCP project number | `469137882515` |
| Region | `europe-west3` (Frankfurt) |
| GCS bucket | `berlin-lst-data` |
| Service account | `masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com` |
| Auth method | Service account JSON key `masterarbeit-berlin-lst-v2.json` + ADC |

## Configuration Files

| File | Purpose |
|------|---------|
| `~/.config/gcp-keys/masterarbeit-berlin-lst-v2.json` | Service account JSON key (private key, used by rclone and ADC) |
| `~/.config/rclone/rclone.conf` | rclone remote `gcs-masterarbeit` pointing to bucket |
| `~/.config/gcloud/configurations/config_default` | gcloud CLI config (service account login) |

## Environment

`GOOGLE_APPLICATION_CREDENTIALS` is set in `~/.zshrc`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp-keys/masterarbeit-berlin-lst-v2.json"
```

This enables ADC for all Python Google libraries (`google.cloud.storage`, `google.auth`).

To verify:
```bash
gcloud auth application-default print-access-token
```
or in Python:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp-keys/masterarbeit-berlin-lst-v2.json"
uv run python -c "import google.auth; cr, proj = google.auth.default(); print(proj)"
```

## rclone Mount

The GCS bucket is mounted locally as a filesystem via rclone (not gcsfuse — gcsfuse doesn't support Intel Macs). rclone uses the same JSON key as ADC.

Aliases are defined in `~/.zshrc`:

```bash
alias mount-berlin='rclone mount gcs-masterarbeit:berlin-lst-data ~/.mnt/berlin-lst --vfs-cache-mode writes --dir-cache-time 10s --daemon'
alias umount-berlin='umount ~/.mnt/berlin-lst'
```

### Check if mount is running

```bash
pgrep -f "rclone mount" && echo "running" || echo "not running"
```

Or check the mount point:

```bash
ls ~/.mnt/berlin-lst/
```

### Start mount

```bash
mount-berlin
# or directly:
rclone mount gcs-masterarbeit:berlin-lst-data ~/.mnt/berlin-lst --vfs-cache-mode writes --dir-cache-time 10s --daemon
```

Wait 2 seconds, then verify:

```bash
ls ~/.mnt/berlin-lst/
```

### Stop mount

```bash
umount-berlin
# or:
pkill -f "rclone mount"
```

### Troubleshooting

- **macFUSE not loaded:** rclone mount requires macFUSE (already installed on this system). If mount fails, restart macFUSE: `/Library/Filesystems/macfuse.fs/Contents/Resources/load_macfuse`
- **Stale mount:** Sometimes macOS caches the empty directory. Run `pkill -f "rclone mount" && sleep 2 && mount-berlin`
- **"can't list buckets without project number":** The rclone config has `project_number = 469137882515` set.
- **VFS cache:** Cached files are stored in `~/Library/Caches/rclone/`. Can be cleared safely.

## GCS Access

### rclone CLI

List bucket root:

```bash
rclone ls gcs-masterarbeit:berlin-lst-data
```

List subdirectories:

```bash
rclone ls gcs-masterarbeit:berlin-lst-data/raw/
```

Copy from bucket:

```bash
rclone copy gcs-masterarbeit:berlin-lst-data/raw/ ./data/raw/
```

Copy to bucket:

```bash
rclone copy ./results/ gcs-masterarbeit:berlin-lst-data/results/
```

Sync (bidirectional, one-way):

```bash
rclone sync gcs-masterarbeit:berlin-lst-data/raw/ ./data/raw/
```

### gcloud storage CLI

```bash
gcloud storage ls gs://berlin-lst-data/ --project=masterarbeit-berlin-lst
gcloud storage cp gs://berlin-lst-data/test/smoke-test.txt ./data/
```

### Python GCS Client

Requires `GOOGLE_APPLICATION_CREDENTIALS` to be set.

```python
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket("berlin-lst-data")

# List blobs
for blob in bucket.list_blobs():
    print(blob.name)

# Read a blob
blob = bucket.blob("test/smoke-test.txt")
content = blob.download_as_text()
print(content)

# Write a blob
blob = bucket.blob("results/experiment.csv")
blob.upload_from_string("col1,col2\n1,2\n")
```

## Smoke Tests

### Manual / Debug

#### rclone mount
```bash
pgrep -f "rclone mount" && echo "mount ok" || echo "mount not running"
ls ~/.mnt/berlin-lst/
```

#### rclone CLI
```bash
rclone ls gcs-masterarbeit:berlin-lst-data
```

#### gcloud CLI
```bash
gcloud storage ls gs://berlin-lst-data/ --project=masterarbeit-berlin-lst-v2
```

#### Python GCS
```bash
uv run python -c "
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('berlin-lst-data')
for blob in bucket.list_blobs(max_results=5):
    print(blob.name)
"
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `rclone ls gcs-masterarbeit:` fails | `project_number` missing in rclone config | Add `project_number = 469137882515` to `~/.config/rclone/rclone.conf` |
| `gcloud auth application-default` fails | `GOOGLE_APPLICATION_CREDENTIALS` not set | `export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp-keys/masterarbeit-berlin-lst-v2.json"` |
| Mount directory empty after start | VFS cache not populated yet | Wait a few seconds (rclone lazily fetches files on first access). Run `ls` again. |
| `google.cloud` import fails | `google-cloud-storage` not installed | `uv add google-cloud-storage` |
| SSH to VM hangs on first connect | VM first-boot or `gcloud` needs to push SSH key | Wait 60s; `gcloud compute ssh` pushes the key automatically on first connect |
| Pipeline on VM says all scenes already `done` | Ledger at `gs://.../ledger.parquet` is from a previous run | `rclone delete gcs-masterarbeit:berlin-lst-data/<output_root>/ledger.parquet` before re-running |

## Compute Engine VM (berlin-lst-vm)

An On-Demand VM in `europe-west3-a` runs the Dynamic pipeline against the bucket.
Same SA as the local setup (`masterarbeit-vertex@...`) — it has
`storage.objectAdmin` on the bucket, so `google.cloud.storage.Client()` on
the VM uses VM ADC (no JSON key, no `GOOGLE_APPLICATION_CREDENTIALS`).

| Field | Value |
|-------|-------|
| Name | `berlin-lst-vm` |
| Zone | `europe-west3-a` |
| Machine | `n2-highmem-2` (2 vCPU / 16 GB) |
| Image | `debian-12` |
| Disk | 50 GB `pd-balanced` (retained between runs) |
| Provisioning | On-Demand (no preemption) |
| Service account | `masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com` |
| Required SA roles | `roles/compute.instanceAdmin.v1`, `roles/serviceusage.serviceUsageAdmin`, `roles/iam.serviceAccountUser` |
| Labels | `purpose=berlin-lst-runner,owner=silas` |

### Lifecycle

```bash
# Start (creates if missing, starts if stopped)
.opencode/skills/google-access/scripts/start-vm.sh

# Run a Dynamic config end-to-end (start → deploy → run → validate → stop)
.opencode/skills/google-access/scripts/run-dynamic-vm.sh full main
.opencode/skills/google-access/scripts/run-dynamic-vm.sh inference_2026 main

# SSH (interactive shell, or pass commands)
.opencode/skills/google-access/scripts/ssh-vm.sh
.opencode/skills/google-access/scripts/ssh-vm.sh -- uptime

# Stop (keeps disk, resumable; ~$1.50/month for 50-GB pd-balanced)
.opencode/skills/google-access/scripts/stop-vm.sh
```

The pipeline is ledger-aware and idempotent — re-running skips scenes already
at `status=done`. Products live in GCS, not on the VM disk. The boot disk
preserves the workspace, venv, and VM-side secrets between runs.

### One-time provisioning (inside the VM)

```bash
sudo apt-get update -qq && sudo apt-get install -y python3.12 python3.12-venv git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/spignotti/berlin-lst-downscaling.git /workspace/app
cd /workspace/app && uv sync

# .env (only EARTHDATA_TOKEN needed; ADC handles GCS auth)
cat > .env <<EOF
EARTHDATA_TOKEN=$(security find-generic-password -s earthdata -w 2>/dev/null || echo MISSING)
EOF
chmod 600 .env
```

### Costs

| State | Cost |
|-------|------|
| RUNNING (On-Demand, n2-highmem-2) | ~$0.065/h |
| STOPPED (disk preserved) | ~$1.50/month for 50-GB pd-balanced |
| DELETE | $0 (but loses disk + setup) |

Stop the VM when not actively running the pipeline. Start it again when you
need to resume a run.

## Key Files (outside repo)

| File | Purpose |
|------|---------|
| `~/.config/gcp-keys/masterarbeit-berlin-lst-v2.json` | Service account JSON key (private key) |
| `~/.config/rclone/rclone.conf` | rclone remote config |
| `~/.zshrc` | `GOOGLE_APPLICATION_CREDENTIALS` env var |
| `~/.mnt/berlin-lst/` | rclone mount point |
