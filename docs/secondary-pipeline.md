# Secondary-Data Pipeline — Operations Runbook

Shared foundation for acquiring, processing, and storing all non-satellite
data sources used in the LST downscaling ablation study.

## Path Layout

All paths are relative to `output_root` (local path or `gs://bucket/prefix`).

| Path | Purpose |
|------|---------|
| `_raw/secondary/{source}/{period}/` | Raw downloaded archives — one per source and period/vintage |
| `_staging/secondary/{source}/{run_id}/` | Ephemeral processing scratch space |
| `ard/static/{category}/{source}/{vintage}/` | Final static COGs (morphology, terrain, SVF) |
| `qa/secondary/{run_id}/` | Run-specific QA reports |
| `ledger.parquet` | Persistent item-level processing ledger |

## Modes

| Mode | Output Root | Purpose |
|------|-------------|---------|
| `fixture` | Local `data/smoke/secondary/fixture` | Validate pipeline lifecycle, no real data |
| `cloud_smoke` | `gs://.../secondary/smoke/{run_id}` | Validate GCS write path |
| `full` | Configurable | Production runs with real sources |

## Ledger Semantics

The `SecondaryLedger` tracks every item (`item_id + source + period`) through
its lifecycle:

1. **pending** — newly added, not yet processed
2. **exporting** — processing in progress (crash recovery marker)
3. **done** — output COG written, validated, and checksummed
4. **failed** — processing error, will be retried on next run
5. **skipped** — explicitly skipped, never processed

### Idempotency / Resume

`reconcile()` applies the same logic as the ARD pipeline:

- Done items with matching `config_hash` + confirmed output → skipped
- Exporting items (crashed) → retry with **interrupted** reason
- Failed items → retry with **retry** reason
- Config hash mismatch → reprocess with **config_changed** reason

## Retry / Resume Behaviour

- HTTP downloads (via `download_to_raw`) use `tenacity` with
  3 attempts and exponential backoff (up to 10 s).
- Ledger upserts happen immediately — crash consistency is built in.
- An aborted run resumes by re-running the same command.  The second run
  processes only items that are missing, failed, or interrupted.

## Required Setup

### GCS access

- Install ADC: `gcloud auth application-default login`
- Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### VM

- Instance: `berlin-lst-vm`, zone `europe-west3-a`, machine `n2-standard-2`
- OS: Debian 12, Spot preemptible
- Auth: Service account via ADC (no keys in repo)
- Bucket: `gs://berlin-lst-data` (co-located in `europe-west3`)

### Disk Budget

Each full source run **must** declare a peak scratch estimate before
execution.  The run will fail preflight if the estimate exceeds the
configured budget.

Current budget: **20 GB** (VM boot disk).
Override with `disk_budget_gb=50` in the Hydra command line.
Use kachelweises/tilewise scratch cleanup when processing tile-based
sources (LoD2, DGM 1 m).

## Smoke Tests

```bash
# Local fixture (no GCS needed)
uv run nox -s smoke-secondary-fixture

# Cloud fixture (requires ADC)
uv run nox -s cloud-secondary-fixture

# Imperviousness end-to-end (downloads official ZIPs)
uv run nox -s smoke-secondary-imperviousness

# Imperviousness on GCS (requires ADC)
uv run nox -s cloud-secondary-imperviousness
```

## Added Sources

### Imperviousness (Versiegelung)

**Vintages:** 2016, 2021 (both processed unconditionally)

**Source:** Umweltatlas Berlin — uncorrected raster (2.5 m, uint8 class codes)

**Processing:**
1. Download ZIP from official ATOM feed (preserved under `_raw/secondary/imperviousness/{vintage}/`).
2. Extract GeoTIFF from ZIP.
3. Convert uint8 class codes to float32 sealing percent using the verified 16-code lookup.
4. Reproject to canonical 10 m EPSG:25833 grid with `Resampling.average`.
5. Write COG to `ard/static/morphology/imperviousness/{vintage}/imperviousness_{vintage}.tif`.

**Class codes verified (2026-07-14):**
| Code | Meaning | Output value |
|------|---------|-------------|
| 0 | Unsealed | 0 % |
| 5, 15, …, 95 | Sealing classes | class value (%) |
| 100 | Fully sealed (non-building) | 100 % |
| 101 | Building-shadow sealed | 100 % |
| 102 | Building footprint | 100 % |
| 103 | Rail ballast | 100 % |
| 110 | Shadow | 100 % |
| 255 | Nodata (2021 only) | NaN |

**Runbook:**
```bash
# Local smoke (downloads both vintages)
uv run python scripts/run_secondary.py --config-name smoke_imperviousness

# With custom output root (local or GCS)
uv run python scripts/run_secondary.py --config-name imperviousness \
    output_root=gs://berlin-lst-data/secondary/full_20260714

# Override disk budget on VM
uv run python scripts/run_secondary.py --config-name imperviousness \
    output_root=gs://berlin-lst-data/secondary/full_20260714 \
    disk_budget_gb=50
```

**Validation gates:**
- Structural: CRS, shape, origin, band count (reuses ARD `validate_cog`)
- Value range: all valid pixels ∈ [0, 100] with 0.01 tolerance
- Code set: hard fail on codes outside the verified 16-code scheme
- Idempotency: second run processes nothing

## Adding a New Source

1. Add source-specific band specs in a new contract (see `contract.py`).
2. Implement a prepare handler: acquire raw → convert → write COG.
3. Register the source in the pipeline runner (see `pipeline.py`).
4. Add a config and smoke session.

Each source module should expose a single `prepare()` entry point that
returns a QA dict.  The pipeline handles reconciliation, ledger, and
validation.
