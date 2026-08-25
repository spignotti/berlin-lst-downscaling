# Project State

Last updated: 2026-08-21

**Current Focus:** WB2c-4 training-data preparation. As-built note delivered; ready to plan `training_eligible@100m` + splits.

**Active Work:** None.

### 2026-08-21
- Completed: WB2c-2 Stage-2 feature-stack QA gate — built (Hydra runner, blockwise `stage2_features.py` core, independent validator, `smoke-qa-stage2` nox session, VM wrapper), ran full on VM (`qa-stage2-20260820T183548Z`, evidence `gs://berlin-lst-data/qa/stage2_features/0c8c8144/`): 345 pairings, 324 assessed, 21 excluded (2026 inference), 0 findings; validator green incl. `features_ledger`; VM `TERMINATED`. Commits `1a89bd1`..`5e7a8b8`. Docs recorded observed delivery and corrected `training_eligible@100m` ownership to WB2c-4.
- Completed: Feature-engineering closure cleanup (`943379b`, `ef02ecd`) — centralized `INFERENCE_EXCLUSION_REASON` in `inventory.py` (single source, consumed by both gates + validator); removed dead `_TILE`/unused `mask_uri` param/`SCRIPTS_DIR`; validator now verifies `geometry_mapping` fingerprint and reports `--skip-source-verify` truthfully; all three Feature/QA VM wrappers gained guarded EXIT cleanup (stop on any post-start setup failure, preserve connection-loss `leave_running`). `uv run nox` + both smoke gates green; all final reviews clean (GO for WB2c closure).
- Completed: WB2c-4 as-built note (Notion) — compact inventory of feature-stack contract (24 channels), QA Stage 1 + 2 results, per-scene coverage/profiles, grouping identifiers, and missing artefacts. Verified against GCS evidence. Source: Notion task `1937cf56-900e-4f99-b20b-72d7e8823eab`.
- Open: none. WB2c-4 training-data preparation (`training_eligible@100m`, splits, normalization) is next. Deferred: Stage-1/2 QA VM wrappers lack the SSH-readiness wait loop (pre-existing, non-blocking); sibling Hydra runners discard exit codes.
- Context: HEAD `ef02ecd` (unpushed). All review gates clean. Notion as-built note delivered; WB2c-4 can proceed to planning.

### 2026-08-20
- Completed: WB2c-3 scene feature stacks — all 324 per-anchor 24-band 10 m stacks + `feature_valid` masks published to `gs://berlin-lst-data/features/v1/`; ledger `324 done / 0 failed`; independent validator `OK: 324 stacks`; VM `TERMINATED`; three blocked scenes recovered after hardening publication (bounded GCS retry + `SystemExit(1)` on failed report). Commits `b924317`..`81c2a43`.
- Completed: GitHub portfolio polish (`3455a51`) — README rewritten to a 69-line public overview; metadata/topics set; contributor question resolved (single owner, no external reviewers).
- Sparse-support policy (user-confirmed): `sparse_support_below_1pct` = `feature_valid_px/inside_aoi_px < 1%`, non-gating. 193023/193024 retained (~1.40% AOI), 194023 (~0.0009%, 79 px) kept as diagnostics; `training_eligible@100m` is a Stage-2/WB2c-4 decision.
- Open: none. Notion WB2c-3 closed; WB2c-2 (Stage-2 QA) was next, since delivered (2026-08-21).
- Context: HEAD `81c2a43`; next session completed Stage-2 QA.

---

## Key Decisions (≤5)

- 2026-08-04 VM access uses direct OpenSSH with pinned identity file, `IdentitiesOnly=yes`, `StrictHostKeyChecking=yes`, and `HostKeyAlias=compute.<instance-id>` — never `gcloud compute ssh`.
- 2026-08-02 ARD STAC uses Raster Extension 1.1 schema URLs with `"nan"` for float nodata, not JSON null.
- 2026-08-02 DVC removed — manifest/ledger/provenance GCS model is the reproducibility basis.
- 2026-07-20 Dynamic geometry policy: retrospective_static.
- 2026-07-19 GCS atomic writes use 5-attempt exponential backoff (1-60s).

---

## Lessons

- 2026-08-21 A detached VM-wrapper must stop the VM on any setup failure after `start-vm.sh`, but never after an ambiguous connection loss (a possibly-still-running job must not be killed mid-write). Arm a state-aware EXIT trap: `vm_started=1; trap cleanup EXIT` right after start; `cleanup() { local rc=$?; ...; exit "$rc"; }` with `local rc=$?` as the FIRST statement (any later command would clobber it); set `leave_running=1` before the connection-loss `exit 2`; disarm (`vm_started=0`) immediately before each explicit `stop-vm.sh` to avoid a double stop.
- 2026-08-15 Before bilinear resampling of masked raster data, declare NaN as nodata on the source (`rio.write_nodata(np.nan, encoded=False)`) — without it, GDAL's bilinear kernel smears NaN over a wider footprint (4×4 at 2× upscale); with it, invalid samples are excluded cleanly (2×2 footprint, valid neighbours interpolate).
- 2026-08-15 A `nohup`-detached bash wrapper's stdout is block-buffered — do not poll its log tail for progress; poll an explicit status file the wrapper writes (`echo "$state" > status`) and the remote marker/exit-status files.
- 2026-08-14 VM deploy of a stale local branch: `git checkout <branch>` does not fast-forward — after `git fetch origin`, always `git reset --hard origin/<branch>` before running. `run-dynamic-vm.sh` still carries the stale pattern; only `run_qa_stage1_vm.sh` was fixed.
- 2026-08-14 Bash: `( wait $PID; ... ) &` in a sibling subshell returns 127 (wait only sees children), and `while kill -0 $PID; sleep; done; echo $?` never captures the child's code. Capture the real exit code inside the detached process: `nohup sh -c 'cmd; rc=$?; echo "$rc" > status' &`.
- 2026-08-14 Published QA evidence is immutable: cosmetic semantics fixes (e.g. layer-fraction normalization) apply to future runs — do not re-run a costly VM pipeline to republish a corrected cosmetic number; document the semantics instead.
- 2026-08-13 Stage git commits by explicit file path. Never `git add -u` while `.opencode/PROJECT_STATE.md` is locally modified (it sweeps the `/end`-owned file into the commit — requires a soft-reset to undo), and never pass a directory you just `git rm`'d to `git add` (the pathspec abort fails the whole add, producing a partial commit).
- 2026-08-12 When bumping ARD `schema_version` for one sensor's semantics, use per-source versioning so `reconcile` reprocesses only that sensor's rows (`schema_changed`) — proven this session with the S2-only v8 bump selecting exactly the 158 S2 scenes.
- 2026-08-04 Separate fast strict-COG validation from blockwise profiling; remote per-block statistics are not an appropriate recovery gate.
- 2026-08-04 `ssh-keygen -H` hashes hostnames in known_hosts — host-key lookup must use `ssh-keygen -F`, not `rg`/`grep`.
- 2026-08-04 `gcloud compute instances describe value(a,b,c,...)` with multiple fields has tab-shift on empty values — use individual single-field calls instead.

---

## Local Skills

- `.opencode/skills/google-access/` — rclone mount, ADC, GCS access patterns, On-Demand VM lifecycle (fail-closed).

## Commit Conventions

- **Kein `Co-authored-by`** — Agent-Commits, nicht noetig.
- **Keine Notion-Task-Referenzen** in Commit-Messages — gehoert in den Workflow, nicht in den Code.
- Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) immer.
