# Documentation map

This folder holds technical documentation only: contracts, operating guides,
evaluation design, and the machine-readable evidence that validators and tests
read. Thesis notes, result summaries and interpretation are kept outside this
repository; removed history and thesis material stays in the git history
(last at commit `66ff28f`). W&B owns live run metrics; GCS holds released
datasets, provenance, and checkpoints.

## Contracts

- `data-sources-and-contracts.md` — sources, canonical grid, manifest and
  product contracts.
- `features-v6.md` — features/v6 channel roster and contract.
- `pseudo-pair-tensor-contract.md` — normative pseudo-pair and tensor
  contract for training, implemented by `modeling/`.

## Running the pipeline

- `republish-v6.md` — the v6 republish chain step by step (selection v4 to
  patch index v3 and EDA v3).
- `vm-operations.md` — VM lifecycle, accounts and secrets, launcher logs.
- `vertex-gpu-training.md` — Vertex AI GPU launch recipe.
- `ablation-v6-configs.md` — v6 ablation arms, run matrix, launcher modes.

## Evaluation design

- `spatial-holdout.md` — spatial holdout and 5-fold spatial cross-validation:
  block assignment, training option, launcher, evaluation, limits.
- `spatial-error-analysis.md` — exploratory validation error analysis by
  scene, Stadtstruktur group and in space.
- `stadtstruktur.md` — Umweltatlas 06.07 / 06.08 evaluation layer; Proposal B
  is the evaluation grouping. Not a feature channel.
- `tessera-arm.md` — TESSERA embedding arm.
- `stage5-loss-audit.md` — thermal-aware loss audit and the fixed train-split
  SSIM data range (derivation command, parameters).

## Data layout

- `gcs-inventory-and-transfer.md` — bucket identity and prefix inventory, with the
  record of the completed 2026-09-25 bucket cutover.
- `s2-mosaic-membership.md` — Sentinel-2 mosaic membership rules.

## Pre-registered criteria read by code

The Stage-1 validators, guards and locked configs cite these documents.

- `stage1-full-results.md`, `stage1-debug-results.md`, `stage1-efficiency.md`,
  `archive/stage1-training-readiness.md`.

## Evidence

- `results/` — machine-readable experiment evidence. Do not rewrite these
  records; validators and tests read some of them.
