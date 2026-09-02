---
name: google-access
description: Google Cloud Storage (rclone mount), ADC setup, and Compute Engine VM lifecycle for the berlin-lst-downscaling project
---

## Quick Reference

- Status:         `.opencode/skills/google-access/scripts/status-vm.sh`
- Start VM:       `.opencode/skills/google-access/scripts/start-vm.sh`
- Stop VM:        `.opencode/skills/google-access/scripts/stop-vm.sh`
- SSH into VM:    `.opencode/skills/google-access/scripts/ssh-vm.sh`
- SSH readiness:  `.opencode/skills/google-access/scripts/ssh-vm.sh --check`
- Run Dynamic:    `.opencode/skills/google-access/scripts/run-dynamic-vm.sh <full|inference_2026> [branch]`
- Run Features:   `.opencode/skills/google-access/scripts/run-features-vm.sh [branch]`
- Run Training:   `.opencode/skills/google-access/scripts/run-training-data-vm.sh [branch]`
- Run QA Stage 1: `.opencode/skills/google-access/scripts/run-qa-stage1-vm.sh [branch]`
- Run QA Stage 2: `.opencode/skills/google-access/scripts/run-qa-stage2-vm.sh [branch]`
- Run status:     `.opencode/skills/google-access/scripts/status-dynamic-vm.sh --run-id <id>`
- Service account key: `~/.config/gcp-keys/masterarbeit-berlin-lst-v2.json`

All `run-*-vm.sh` launchers source the shared fail-closed lifecycle in
`.opencode/skills/google-access/scripts/vm-runner-common.sh` (start → deploy pinned
commit → detached launch → poll → validate → stop; the VM is never stopped
automatically after a connection loss or an ambiguous exit). Pipeline-specific
parameters (runner, config, validator, output roots) live only in each
launcher; the application code itself stays VM-agnostic.

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

### SSH / VM lifecycle

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ssh-vm.sh` fails with "Instance ID mismatch" | VM was deleted and recreated with same name | Verify with `status-vm.sh`. If intentional, update pinned ID in `vm-identity.sh` and re-run host-key ceremony. |
| `ssh-vm.sh` fails with "No host-key entry" | Host key never provisioned or was cleared | Run the host-key ceremony (see above). |
| `ssh-vm.sh` fails with "Connection refused" | VM is stopped or SSH daemon not ready | Check state with `status-vm.sh`. Start if needed. |
| `ssh-vm.sh` fails with "Connection timed out" | No external IP or firewall rule missing | Check `status-vm.sh` for external IP. If absent, VM may need restart. |
| `start-vm.sh` fails with "does not exist" | VM was deleted | Requires manual recreation — this script will NOT create one. |
| `start-vm.sh` fails with "Unexpected state" | VM in PROVISIONING/STAGING/REPAIRING | Wait and retry. If persistent, investigate via GCP console. |
| `stop-vm.sh` shows "auto-delete=true" warning | Boot disk would be destroyed on `gcloud compute instances delete` | Run `gcloud compute instances set-disk-auto-delete berlin-lst-vm --disk=berlin-lst-vm --no-auto-delete --zone=europe-west3-a` to fix. Note: `--disk` takes the disk resource name (`berlin-lst-vm`), not the device name (`persistent-disk-0`). |
| Instance has no deletion protection | Instance can be deleted without a guard | Run `gcloud compute instances update berlin-lst-vm --deletion-protection --zone=europe-west3-a` |

### GCS / rclone

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `rclone ls gcs-masterarbeit:` fails | `project_number` missing in rclone config | Add `project_number = 469137882515` to `~/.config/rclone/rclone.conf` |
| `gcloud auth application-default` fails | `GOOGLE_APPLICATION_CREDENTIALS` not set | `export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp-keys/masterarbeit-berlin-lst-v2.json"` |
| Mount directory empty after start | VFS cache not populated yet | Wait a few seconds (rclone lazily fetches files on first access). Run `ls` again. |
| `google.cloud` import fails | `google-cloud-storage` not installed | `uv add google-cloud-storage` |
| Pipeline on VM says all scenes already `done` | Ledger at `gs://.../ledger.parquet` is from a previous run | `rclone delete gcs-masterarbeit:berlin-lst-data/<output_root>/ledger.parquet` before re-running |

## Compute Engine VM (berlin-lst-vm)

An On-Demand VM in `europe-west3-a` runs the Dynamic pipeline against the bucket.
Same SA as the local setup (`masterarbeit-vertex@...`) — it has
`storage.objectAdmin` on the bucket, so `google.cloud.storage.Client()` on
the VM uses VM ADC (no JSON key, no `GOOGLE_APPLICATION_CREDENTIALS`).

| Field | Value |
|-------|-------|
| Name | `berlin-lst-vm` |
| Instance ID | `8456019039456721311` |
| Zone | `europe-west3-a` |
| Machine | `n2-highmem-2` (2 vCPU / 16 GB) |
| Image | `debian-12` |
| Disk | 50 GB `pd-balanced` (resource name `berlin-lst-vm`, device name `persistent-disk-0`, retained between runs) |
| Protection | Boot disk not auto-delete, instance deletion protection enabled |
| Provisioning | On-Demand (no preemption) |
| Service account | `masterarbeit-vertex@masterarbeit-berlin-lst-v2.iam.gserviceaccount.com` |
| Required SA roles | `roles/compute.instanceAdmin.v1`, `roles/serviceusage.serviceUsageAdmin`, `roles/iam.serviceAccountUser` |
| Labels | `purpose=berlin-lst-runner,owner=silas` |

### Fail-closed lifecycle

All lifecycle scripts assert the pinned identity (project, zone, name, instance
ID, boot-disk attachment) before any operation. On mismatch they STOP — no
retry, no fallback, no creation.

- `start-vm.sh` **never creates** a new VM. If the pinned instance does not
  exist or has been replaced, it refuses to start. Use `--dry-run` to preview.
- `stop-vm.sh` asserts the exact instance and disk before stopping. Use
  `--dry-run` to preview.
- `ssh-vm.sh` uses direct OpenSSH with `-i ~/.ssh/google_compute_engine`,
  `IdentitiesOnly=yes`, `StrictHostKeyChecking=yes`, and a fixed
  `HostKeyAlias=compute.<instance-id>`. Host-key lookup uses `ssh-keygen -F`
  so it works with both plain and hashed known-hosts entries. It **never**
  writes known-hosts, generates keys, or provisions metadata keys.
- `status-vm.sh` is a read-only probe that reports identity, state, disk
  attachment, protection, external IP, and SSH host-key readiness.

### Pinned identity constants (vm-identity.sh)

| Constant | Value |
|----------|-------|
| `VM_PROJECT` | `masterarbeit-berlin-lst-v2` |
| `VM_ZONE` | `europe-west3-a` |
| `VM_NAME` | `berlin-lst-vm` |
| `VM_EXPECTED_ID` | `8456019039456721311` |
| `VM_SA` | `masterarbeit-vertex@…` |
| `VM_DISK_DEVICE` | `persistent-disk-0` |

Change these values only when the instance is knowingly re-created and the new
identity is independently verified.

### Lifecycle

```bash
# Read-only status (never changes anything)
.opencode/skills/google-access/scripts/status-vm.sh

# Start (fails closed; never creates; supports --dry-run)
.opencode/skills/google-access/scripts/start-vm.sh
.opencode/skills/google-access/scripts/start-vm.sh --dry-run

# SSH readiness check (read-only, no connection)
.opencode/skills/google-access/scripts/ssh-vm.sh --check

# SSH (direct OpenSSH, strict host-key)
.opencode/skills/google-access/scripts/ssh-vm.sh
.opencode/skills/google-access/scripts/ssh-vm.sh -- uptime

# Run a Dynamic config end-to-end (start → deploy → run → validate → stop)
.opencode/skills/google-access/scripts/run-dynamic-vm.sh full main
.opencode/skills/google-access/scripts/run-dynamic-vm.sh inference_2026 main

# Check status of a running or completed run
.opencode/skills/google-access/scripts/status-dynamic-vm.sh --run-id full-20260804T120000Z

# Stop (keeps disk, resumable; supports --dry-run)
.opencode/skills/google-access/scripts/stop-vm.sh
.opencode/skills/google-access/scripts/stop-vm.sh --dry-run
```

The pipeline is ledger-aware and idempotent — re-running skips scenes already
at `status=done`. Products live in GCS, not on the VM disk. The boot disk
preserves the workspace, venv, and VM-side secrets between runs.

### Run markers and reconnection

Each `run-dynamic-vm.sh` execution writes an immutable run marker on the VM:

```
/workspace/app/logs/runs/<config>-<timestamp>-<suffix>/marker.json
```

The marker contains: `run_id`, `config`, `branch`, `sha`, `started`, `pid`,
`log`, and `status_file`.  On process completion, the wrapper writes an
exit-status
file at the same location.

If the local session is interrupted (SSH failure, machine sleep, agent crash):

1. **Do NOT relaunch.** The remote process may still be running.
2. Run `status-dynamic-vm.sh --run-id <id>` to probe the marker.
3. It reports one of:
   - `RUNNING` — process is alive; last log line shown.
   - `COMPLETED` — exit code 0; safe to validate and stop VM.
   - `FAILED` — non-zero exit; check VM-side log.
   - `CONNECTION_LOST` — cannot reach VM; try later.
   - `AMBIGUOUS` — PID is dead but no exit status was written; investigate manually.
4. The VM is **never** stopped automatically after a connection loss. An
   operator must explicitly decide to stop after confirming the run state.

### Host-key recovery ceremony (manual, separately authorized)

If `ssh-vm.sh --check` reports no host-key entry or a key mismatch after a
VM reboot/recreate, the host key must be provisioned through an independently
authenticated channel — never by accepting whatever the VM presents.

1. Verify the VM is the expected instance (`status-vm.sh`).
2. Obtain the public host key through an authenticated out-of-band path:
   - Create a snapshot of the boot disk; clone it to a temporary disk;
     mount the clone read-only and inspect `/etc/ssh/ssh_host_*_key.pub`.
   - Or use an authenticated GCP console session to read guest attributes
     (requires guest attributes enabled on the instance).
3. Compute the fingerprint of the presented key and compare it with the
   offline-obtained fingerprint. If they differ: **STOP** — do not connect.
4. Back up `~/.ssh/google_compute_known_hosts`.
5. Remove only the `compute.<instance-id>` entry:
   ```bash
   ssh-keygen -R "compute.8456019039456721311"
   ```
6. Add the verified key:
   ```bash
   echo "compute.8456019039456721311 <key-type> <key-blob>" \
     >> ~/.ssh/google_compute_known_hosts
   ```
7. Re-run `ssh-vm.sh --check` to confirm.

If no independent fingerprint source exists: **halt**, do not trust the key.

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
