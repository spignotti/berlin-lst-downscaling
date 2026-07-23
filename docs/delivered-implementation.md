# Delivered implementation

Records the production pipelines, published roots, commands, and
observability for the preprocessing phase. Production paths live in
`gs://berlin-lst-data/`; local paths are scratch only.

## Pipeline graph

```text
Selection (build_manifest)
  └─ assets/manifests/v3/<bundle>/-r2/
       ├── ARD pipeline (run_ard)
       └── Dynamic pipeline (run_dynamic)
              └── DWD validation (run_dwd_validation)

Static sources (run_static_sources)
  └─ Static derived (run_static_derived)
       └── consumed by Dynamic (geometry)
```

Each output is anchored to a single immutable bundle prefix; nothing
reproduces or mutates historical artifacts.

## Published artefacts

| Pipeline | GCS root | Counts (r3 2026-07-23) |
|----------|---------|--------------------:|
| Manifest bundle (canonical) | `gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/` | 509 manifest rows, 345 pairings |
| ARD ledger | `gs://berlin-lst-data/ard/full/2017-2026-cutoff-20260717T235959Z/ledger.parquet` | 509 rows |
| Static sources | `gs://berlin-lst-data/static/sources/full/` | 5 entries |
| Static derived | `gs://berlin-lst-data/static/derived/full/_state/static/derived/ledger.parquet` | 6 entries |
| Dynamic full | `gs://berlin-lst-data/dynamic/full/dyn-20260721T092945-4a4de9/` | 972 rows (`role=anchor`) |
| Dynamic inference | `gs://berlin-lst-data/dynamic/inference/2026/dyn-inf-r4-20260722T203148/` | 63 rows (`role=inference`) |
| DWD validation r3 | `gs://berlin-lst-data/dwd_validation/r3/runs/dwd/9d5269f5/` | 345 anchors, bias −0.03 °C, MAE 0.77 °C, RMSE 0.98 °C |

DWD r3 verifies every published Landsat anchor has a matching ERA5
anchor value: `n_anchors=345`, `n_anchors_with_era5=345`,
`n_pairs_era5_missing=0`. DWD never feeds model training.

## Commands

```bash
# ARD
uv run python scripts/run_ard.py --config-name full_all \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# Dynamic full (manifest_uri required)
uv run python scripts/run_dynamic.py --config-name full \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# Dynamic inference
uv run python scripts/run_dynamic.py --config-name inference_2026 \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet

# Static sources
uv run python scripts/run_static_sources.py --config-name full

# Static derived
uv run python scripts/run_static_derived.py --config-name full

# DWD validation
uv run python scripts/run_dwd_validation.py --config-name default \
    manifest_uri=gs://berlin-lst-data/manifests/v3/2017-2026-cutoff-20260717T235959Z-r2/manifest.parquet \
    dynamic_full_root=gs://berlin-lst-data/dynamic/full/dyn-20260721T092945-4a4de9 \
    dynamic_inference_root=gs://berlin-lst-data/dynamic/inference/2026/dyn-inf-r4-20260722T203148 \
    output_root=gs://berlin-lst-data/dwd_validation/<run-id>
```

Run validation through the dedicated scripts — they reject malformed
input before opening ledgers:

```bash
uv run python scripts/validate_manifest.py \
    --manifest gs://...-r2/manifest.parquet \
    --pairings gs://...-r2/pairings.parquet \
    --report gs://...-r2/manifest_report.json

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/full/dyn-20260721T092945-4a4de9 \
    --expected-role anchor --expected-scenes 324

uv run python scripts/validate_dynamic.py \
    --output-root gs://berlin-lst-data/dynamic/inference/2026/dyn-inf-r4-20260722T203148 \
    --expected-role inference --expected-scenes 21
```

## Smoke gates

The `noxfile.py` sessions are the documented smoke matrix:

- `smoke-primary` — bounded ARD with manifest-driven fixtures.
- `smoke-static-sources` / `smoke-static-derived` — bounded static runs.
- `smoke-dynamic` — single deterministic Landsat anchor through Dynamic.
- `smoke-dwd-validation` — bounded DWD run with a one-scene manifest.
- Cloud variants mirror local smokes and require ADC + CDS access.

Each smoke also runs the matching validator with assertion args.

## Logging contract

Every run emits one JSONL file at `<output_root>/logs/<pipeline>/<run_id>.jsonl`
(session handler in `data/io/run_logging.py`). For GCS runs the JSONL is
written to a local spool, uploaded atomically on session exit, and the
spool is deleted.

```python
from berlin_lst_downscaling.data.io import RunLogSession, log_event

with RunLogSession(output_root, pipeline="ard", run_id=run_id):
    log_event(_logger, logging.INFO, "scene_done",
              scene_id=..., source=..., attempts=..., elapsed_s=...)
```

## Accessibility

- Manifest bundle: `gs://berlin-lst-data/manifests/v3/.../manifest.parquet`.
- AOI mask: `data/boundaries/aoi_10m.tif` (default), `aoi_100m.tif`.
- AOI polygon: `data/boundaries/berlin_landesgrenze.geojson`.

## Manual smoke command catalogue

- `.opencode/skills/google-access/` — rclone mount and ADC setup.
- `AGENTS.md` — repository conventions and current operational focus.
- `noxfile.py` — exact session definitions for every smoke listed above.
