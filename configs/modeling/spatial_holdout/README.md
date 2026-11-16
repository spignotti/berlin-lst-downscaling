# Spatial holdout assignments

`v<N>.json` is a committed assignment of patch-index anchors to the spatial
holdout (docs/spatial-holdout.md). A run names its file with
`spatial_holdout.assignment` and records the file's SHA-256.

- Written by `scripts/operators/build_spatial_holdout.py --output <file>`.
- `cv5-fold0.json` to `cv5-fold4.json` are the five folds of the spatial cross-validation
  (all anchors, no random draw; rule in docs/spatial-holdout.md), written by
  `build_spatial_holdout.py --partition cv5 --anchors-from v1.json --output-dir <dir>`
  and admitted for arms L1 and L4 at seed 0 only. Loading re-derives the partition from
  the file's anchor table.
- The rule and the anchor table (train and validation index rows per anchor)
  are stored in the file. Loading re-derives the held-out anchors, blocks and
  positions from them and refuses a file that disagrees with its own rule.
- A file that a run may have used is never overwritten. A changed rule, seed
  or patch index gets a new version file.
- `--check <file>` compares a file with the published patch index.
