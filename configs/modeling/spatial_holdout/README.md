# Spatial holdout assignments

`v<N>.json` is a committed assignment of patch-index anchors to the spatial
holdout (docs/spatial-holdout.md). A run names its file with
`spatial_holdout.assignment` and records the file's SHA-256.

- Written by `scripts/operators/build_spatial_holdout.py --output <file>`.
- The rule and the anchor table (train and validation index rows per anchor)
  are stored in the file. Loading re-derives the held-out anchors, blocks and
  positions from them and refuses a file that disagrees with its own rule.
- A file that a run may have used is never overwritten. A changed rule, seed
  or patch index gets a new version file.
- `--check <file>` compares a file with the published patch index.
