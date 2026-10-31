# SSIM data range artifacts

Each `v<N>.yaml` file holds one fixed SSIM data range `L = p99.5 - p0.5` of the valid 100 m
train targets (2017-2023), derived by `scripts/operators/derive_ssim_data_range.py`.
`SSIM_DATA_RANGE_ARTIFACT` in `src/berlin_lst_downscaling/modeling/ssim_range.py` names the active
version (currently `v1.yaml`, `L = 30.622 K`).

## Versioning rule

- An artifact is bound to the patch index it was derived from. It records
  `patch_index_fingerprints.index_policy_hash` and `patch_index_fingerprints.source_policy_hash`.
- Every loader that knows the run's patch index (U-Net runs, baseline, random forest) compares
  both hashes with the artifact. A mismatch raises and asks for a new artifact version. A missing
  or empty hash on either side also raises; a missing hash never counts as a match.
- A published version is immutable. The derivation script writes the next free version
  (`v2.yaml` after `v1.yaml`) and refuses to overwrite an existing file; `--out` must name a new
  `v<N>.yaml` file and the payload `version` must equal the file stem.
- Activating a new version is a reviewed change: point `SSIM_DATA_RANGE_ARTIFACT` at the new file
  in a pull request. Runs record the artifact SHA-256; a checkpoint trained with another artifact
  fails on reload.

## Reporting

The fixed-L SSIM (`ssim_100m`) is the main metric. Every report also carries
`ssim_100m_L250` (L = 250 K), the appendix comparison value to stages 1-4, and the diagnostic
`ssim_100m_Lpatch` (per-patch target range). See
[`docs/thesis/03-methodik/evaluation-unsicherheit.md`](../../../docs/thesis/03-methodik/evaluation-unsicherheit.md).
