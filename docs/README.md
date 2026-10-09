# Documentation map

Thesis reasoning, method justification, and scientific interpretation live in
Notion. This repository owns normative technical contracts, source, configs,
validators, operating instructions, and immutable evidence records. W&B owns
live run metrics; GCS holds released datasets, provenance, and checkpoints.

## Current contracts and guides

- `data-sources-and-contracts.md` — sources, canonical grid, manifest and
  product contracts.
- `pseudo-pair-tensor-contract.md` — normative pseudo-pair and tensor
  contract for real-data training (WB3), implemented by `modeling/`.
- `vertex-gpu-training.md` — Vertex AI GPU launch recipe.
- `ablation-stage-configs.md` — frozen comparison configs for the ablation
  stages and isolation runs.

## Frozen evidence

Do not rewrite these records. Later runs are recorded separately.

- `baseline-full-results.md` — same-universe naive baseline and validation.
- `stage1-full-results.md` — Stage-1 full temporal GO screen, cohort,
  one-shot test, and checkpoint verification (issue #53).
- `stage1-efficiency.md` — closed efficiency gate and frozen cheap runtime.
- `stage1-probe-results.md` — historical absolute-head NO-GO context.
- `stage1-debug-results.md` — diagnostic observations and residual-recovery GO.
- `patch-read-timing.md` — bounded I/O measurement and no-change decision.
- `gcs-inventory-and-transfer.md` — completed 2026-09-25 bucket cutover.
- `ablation-full-results.md` — Tag-11 ladder and isolation runs (issues #58
  and #63).
- `random-forest-baseline.md` and `random-forest-full-results.md` — CPU
  random-forest comparison method, validation, and results (issue #59).
- `results/` — machine-readable experiment evidence.

## Archived handoffs

Historical context, not operational guidance. Paths inside these files may
predate the 2026-09-25 bucket cutover.

- `archive/phase-1-delivery.md` — preprocessing handoff.
- `archive/phase-2-preparation.md` — WB2 handoff.
- `archive/stage1-training-readiness.md` — cancelled verification gate; the
  frozen runtime reference lives in `stage1-efficiency.md`.
