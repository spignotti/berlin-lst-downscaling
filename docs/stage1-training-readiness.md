# Stage-1 training readiness

Verifies and freezes the lowest-cost supported Stage-1 setup before any full
Stage-1 run. Originally planned as one capped verification job after J1–J4.

**Status: cancelled.** The sole verification job was not run. The frozen
runtime for issue #53 is the J1–J4 recommendation in
`docs/stage1-efficiency.md`:

- job-local cache
- batch size 4
- `num_workers: 0`, `pin_memory: false`
- `trainer.precision: 16-mixed` (physical residual arithmetic, pooling,
  loss/metric reductions, validation, and reload remain FP32)

`configs/modeling/stage1_locked.yaml` and `assert_stage1_runtime` carry these
values. A cheap Vertex smoke (`--mode smoke`) proves the lifecycle on the new
image before the full temporal run.
