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
(edge-truncated cells can never be valid). TESSERA gaps are not filled.
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

`publish_guard` adds protection for the new prefix once published: after a
year's `complete.json` is published, add its directory URI to
`_FROZEN_TESSERA_ROOTS` so republish paths fail closed.

## How to run the operator later

Code only in this PR; nothing is executed or published here.

1. From a clean checkout of this branch, run on the On-Demand VM (one year
   at a time, 2017–2025):
   `.opencode/skills/google-access/scripts/run-tessera-embeddings-vm.sh feat/tessera-arm 2024`
   (equivalent remote: `scripts/operators/run_tessera_embeddings.py --year 2024`
   then `scripts/validators/validate_tessera_embeddings.py --root <prefix> --year 2024`).
2. Dry-run to local disk first with `--output-root <local dir>`; GCS writes
   need ADC and the VM.
3. Do not start any GCP VM, Vertex job, or write to
   `gs://berlin-lst-training-data` from CI or workstations. Tests use
   fakes/synthetic fixtures offline. Small anonymous read-only probes of the
   public TESSERA store are fine for exploration.
