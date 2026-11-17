# TESSERA embedding ablation arm

Ablation-only arm that swaps hand-built morphology and shadow channels for
annual TESSERA embeddings. Hand-built features stay the main model and the
2026 inference; `features/v6` never depends on TESSERA in any way.

## Source

- Library: `geotessera >= 0.11` (default dataset `v1.1-dclimate`); cloud-native
  Zarr/Icechunk `GeoTesseraZarr().read_region(bbox, year=...)` /
  `read_region_quantized(...)` (int8 + float32 scale per pixel,
  dequantized = int8 × scale).
- Raw store (anonymous): `s3://tessera-embeddings/v1.1/dclimate.icechunk/`,
  zone group `33N`; needs `icechunk` and `numcodecs[pcodec]`.
- Berlin bbox `[13.08, 52.34, 13.76, 52.68]` lies entirely in UTM 33; data
  native `EPSG:32633`; project grid canonical `EPSG:25833` 10 m
  (`common/grid.py`). Origin/pixel raster need not align with the canonical
  grid. Check `years_complete` per zone (recorded in provenance).
- Citation: Feng et al. 2025, arXiv:2506.20380. License: CC0. Inputs are
  only S1/S2, no Landsat/thermal, so no target leakage.

## Grid / reprojection

Explicit reprojection `EPSG:32633 → EPSG:25833` 10 m with **nearest**
resampling only (`data/embeddings/reproject.py`). Embeddings are
representation vectors: values are copied, never interpolated or blended.
The int8 codes and the float32 scale are reprojected as separate nearest
fields, then recombined (`dequantized = int8 × scale`). Any other
resampling raises.

## Missing data

Non-finite scale or below-threshold pixels become the TESSERA-invalid mask
(`NaN` after dequantization), never zero-filled
(`data/embeddings/tessera.py:tessera_invalid_mask`). The default quality
threshold is `0.0` (config `quality_threshold` / `tessera_quality_threshold`,
recorded in provenance). A fully-zero embedding vector with finite scale
fails validation (zero-fill artifact).

## Common cell set

Both arms train and evaluate on the SAME cell set: the intersection of the
main arm's eligible cells and TESSERA-valid cells
(`data/embeddings/cells.py`). Main-arm eligibility is `100 m`
(target valid + all 100 subpixels `feature_valid`); TESSERA validity is
aggregated from `10 m` to `100 m` with the same strict `100/100` rule
(edge-truncated cells can never be valid). TESSERA gaps are not filled in the product (NaN/scale-NaN). At the model boundary, gap pixels are zero-filled to finite values (mirroring the real-path predictor rule); exclusion stays via `mask_100m` and the common cell set.
Both arms point at the identical patch/cell set via `common_cell_set` /
`intersect_eligibility`.

## 1×1 adapter

All 128 embedding channels go through a learnable `1×1` convolution (no
PCA, no fixed reduction) in front of the SAME U-Net
(`modeling/tessera_adapter.py`). Output width is the config value
`model.tessera_adapter_width` (default `8`, `TESSERA_ADAPTER_WIDTH_DEFAULT`).
The adapter output is concatenated with the remaining 14 v6 channels
(spectral + index + ERA5) and fed to the shared `UNet`.

## Channel replacement

Replaced (8): the 6 morphology channels + the 2 shadow channels.
Kept identical (14): spectral, index, ERA5 — see
`tessera_replaced_channel_names` / `tessera_kept_channel_names`. The
training-data path loads the per-year embedding for each scene's year and
keeps every other channel identical to the v6 main arm. Main-arm configs
and `features/v6` are byte-for-byte unaffected.

## Non-causal annual embedding note

Annual TESSERA embeddings include S1/S2 observations after the scene date
(non-causal). They are valid for this retrospective ablation but must not
be interpreted as a causal operational input.

## Splits and 2026 guard

Train 2017–2023, validation 2024, test 2025 (same split as the main model).
TESSERA has no 2026 (calendar-year store), so the arm is never used for
2026; `validate_tessera_year` and `assert_tessera_eval_year` /
`assert_tessera_not_2026` refuse 2026 with a clear error. Evaluation reuses
the existing eval path (val 2024, test 2025).

## Storage

Int8 + scale ≈ 2.3 GB per year on the Berlin bbox. Canonical product
(prefix, one directory per year):

    gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate/<year>/
        tessera_<year>_int8.tif, tessera_<year>_scale.tif,
        tessera_<year>_valid.tif, provenance.json, complete.json (last)

Years 2017–2025 are published and listed in `_FROZEN_TESSERA_ROOTS`.
Republish paths fail closed. A new release needs a new prefix.

## Spatial holdout (pm #524, pm #553)

The arm also runs under the spatial holdout `v1` (config `tessera_arm_spatial_holdout`, mode
`tessera-spatial-holdout`, `docs/spatial-holdout.md`, "TESSERA arm"): held-out anchors leave the
train split, the scaler is refit on the remaining train patches, the checkpoint is selected on
validation 2024 on seen anchors, test 2025 is never scored. A job-local TESSERA cache was never
measured on the T4 worker (window reads from GCS were far too slow), so the run reads the published
prebuilt cache (6,245 patches, about 35 GB) read-only under its unchanged key and re-scales its
features in the job. This is equivalent to a job-local holdout build with the refit scaler because:

1. The cache holds the 22 features scaled with the published scaler, non-finite pixels set to 0.0
   afterwards (`RealPatchReader._read_features`). Per kept channel the scaling is
   `(t - mean) / std`, `t` = the value, or `log1p` of it for precipitation. Hence
   `t = stored * std + mean` and the refit value is `(t - mean_refit) / std_refit`, the order of
   operations of `ScalerSpec.apply`; `src/berlin_lst_downscaling/modeling/published_cache_remap.py`.
2. A stored exact 0.0 is kept at 0.0: it is a filled pixel (a job-local build zero-fills it too).
   That holds only if no valid pixel scales to exactly 0.0. The reader evaluates the quotient in
   float64 on float32 input and stores float32. For `zscore` a zero needs an input value equal to the
   published mean, for `log1p_zscore` an input whose float64 `log1p` equals it. At load, per kept
   channel (all 14 are scaled, none `identity`), the nearest float32 values (3 steps each side) must
   differ from the mean by more than `1e-30 * std`, far above float32 underflow (1.2e-38), otherwise
   the run aborts. A float32-representable mean (for example 0.5) fails this check.
3. The recorded `filled_feature_pixels` counts non-finite pixels over all 22 channels, shadows
   included, so equality with the stored zeros cannot be checked. The load checks the sound
   inequality instead: per patch (kept train rows and all validation rows) the stored zeros in the
   14 kept channels must not exceed the recorded count; otherwise the run aborts before training,
   naming the patch and both counts.
4. The refit uses the pixels of eligible cells with a stored value other than 0.0 (eligible cells
   are finite), `scaler_moments` and `merge_scaler_moments` of the nine v6 holdout arms, and
   `scaler_from_moments` (8 decimals). Only the kept channels are refit and re-scaled; morphology
   and shadow channels, which the arm replaces, keep their published entries and are not used.
5. The embeddings path (`dequantize_tessera`, int8 plus per-pixel scale) is untouched.

The mean check (step 2) runs in the job right after the scaler is read, before admission and the
download; the launcher `--preflight` makes no read of `scaler.json` and so cannot run it.

Offline tests (`tests/test_tessera_spatial_holdout.py`) show on a synthetic cache with NaN features:
filled pixels are exactly 0.0 in both paths, other pixels differ by less than 1e-5 (float32
rounding of the stored value), the refit scaler equals the job-local holdout scaler within 1e-6, a
wrong filled count and a representable mean abort the load, and a training step opens no cloud
storage. The run summary records the cache key, the patches checked and refit, the recorded filled
pixels and the stored zeros (`spatial_holdout.feature_remap`).

Limits: the annual embeddings include imagery after the scene date and from the validation year,
also on held-out anchors. The arm replaces morphology and shadow channels, so it is a sibling of
L4, not a ladder stage. Because it drops patches with invalid embedding pixels, every comparison
with L4 uses the intersection of the patch sets and reports what each side lost. The only way to run
TESSERA seeds is `scripts/runners/vertex/launch_v6_matrix.sh holdout --arms TESSERA,TESSERA-TEMPORAL`
(the launchd watcher `retry_tessera_seeds.sh` is removed). A capacity miss ends with exit code 3;
re-run the same command with `V6M_RESUBMIT_CANCELLED=1`. A TESSERA batch must use the same
`V6M_RUN_DIR` as any v6 batch running at the same time (one lock, at most 4 jobs in flight overall).

## How to run the operator later

The 2017–2025 products are already in the bucket. Do not republish them.
A later release uses a new prefix, then the same launcher.

1. Do not rerun 2017–2025 on the canonical prefix; the launcher fails closed.
   A new release points `RELEASE_ROOT` at a new prefix, then runs one year
   at a time on the On-Demand VM.
   The launcher runs the acquisition operator in an ISOLATED uv script env
   (PEP 723 block at the top of `scripts/operators/run_tessera_embeddings.py`:
   `geotessera` / `icechunk` / `zarr>=3`, with `PYTHONPATH=src`), leaving the
   main training env (torch 2.2.2 / `zarr<3`, used by the validator below and
   by `uv sync --locked`) untouched. `uv.lock` carries no acquisition deps.
2. Dry-run to local disk first with `--output-root <local dir>` (same
   `uv run --script` form, run from the repo root); GCS writes need ADC and
   the VM.
3. Do not start any GCP VM, Vertex job, or write to
   `gs://berlin-lst-training-data` from CI or workstations. Tests use
   fakes/synthetic fixtures offline. Small anonymous read-only probes of the
   public TESSERA store are fine for exploration.
