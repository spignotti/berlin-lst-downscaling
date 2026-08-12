# Project State

Last updated: 2026-08-11

**Current Focus:** None — WB2c-1 closure complete; canonical data validated for the next QA task.

**Active Work:** None.

### 2026-08-11
- Completed: Repaired Base-DSM metadata-lineage defect — `derived_pipeline.py` upstream-hash keys now match `dsm.py` readers (terrain/lod2/vh), so ledger = provenance = STAC config_hash for building_dsm, vegetation_dsm, combined_dsm (`98af0fe21725` / `402c833f4b79` / `3dace53f22c1`, run `9baf3ba7`).
- Completed: Added scoped `force_dsm_products` re-publication hook (allowlist of the three DSMs, fail-fast on unknown); forced production re-finalization touched only those three products — horizons/SVF/Dynamic/source untouched.
- Completed: Re-validated canonical bucket — profiling 2,079 assets / 0 hard failures, manifest/ledger 509/509, Dynamic 972/972 + 63/63, zero `.tmp/` objects in touched prefixes.
- Completed: Published exactly one WB2c-1 closure note in Notion ("WB2c-1 — Datenprofiling abgeschlossen"); Notion task `961912eb` stayed Done (unchanged).
- Commits: `4157db1 fix(secondary): repair base DSM lineage metadata`, `ce6f9d8 fix(review): log skipped forced combined dsm`.
- Open: None.
- Context: Local `smoke-static-sources`/`smoke-static-derived` nox gates fail with `SSL: CERTIFICATE_VERIFY_FAILED` on GDI downloads (pre-existing, identical on clean HEAD) — canonical GCS validation unaffected; next QA task runs against GCS roots.

### 2026-08-04
- Completed: Removed the COG recovery implementation, configs, scripts, runbook, and residual helpers from the active repository; strict COG validation remains in `cog_layout.py` and the writer path is unchanged.
- Completed: Fixed the Dynamic validator to match published `source_metadata.era5_variables`; full and inference Dynamic validation passed.
- Completed: Verified 2,079/2,079 canonical COGs strict-clean and ARD 509/509 manifest/ledger/artifact validation passed.
- Completed: Deleted the recovery bucket, recovery artifacts, profiling smoke output, ARD smoke data, and DWD validation r1/r2; retained documented DWD r3 and all canonical roots.
- Open: None.
- Context: Repository and published bucket are clean for the next planned data task; `.opencode/PROJECT_STATE.md` is the only local modification owned by `/end`.

### 2026-08-02
- Completed: ARD publication contract enforced — all 509 scenes publish COG + flag + STAC + provenance + complete.json. Strict ARD validator deployed and validated on VM. DVC removed.
- Open: None.
- Context: All production pipelines complete and published; GCS is the source of truth.

---

## Key Decisions (≤5)

- 2026-08-04 VM access uses direct OpenSSH with pinned identity file, `IdentitiesOnly=yes`, `StrictHostKeyChecking=yes`, and `HostKeyAlias=compute.<instance-id>` — never `gcloud compute ssh`.
- 2026-08-02 ARD STAC uses Raster Extension 1.1 schema URLs with `"nan"` for float nodata, not JSON null.
- 2026-08-02 DVC removed — manifest/ledger/provenance GCS model is the reproducibility basis.
- 2026-07-20 Dynamic geometry policy: retrospective_static.
- 2026-07-19 GCS atomic writes use 5-attempt exponential backoff (1-60s).

---

## Lessons

- 2026-08-04 Separate fast strict-COG validation from blockwise profiling; remote per-block statistics are not an appropriate recovery gate.
- 2026-08-04 Direct OpenSSH requires explicit `-i` identity file — SSH auth fails silently without it even if the key exists at the expected path.
- 2026-08-04 `ssh-keygen -H` hashes hostnames in known_hosts — host-key lookup must use `ssh-keygen -F`, not `rg`/`grep`.
- 2026-08-04 `gcloud compute instances describe value(a,b,c,...)` with multiple fields has tab-shift on empty values — use individual single-field calls instead.
- 2026-08-02 STAC Raster Extension 1.1 requires float nodata as `"nan"` string, not JSON `null`.

---

## Local Skills

- `.opencode/skills/google-access/` — rclone mount, ADC, GCS access patterns, On-Demand VM lifecycle (fail-closed).

## Commit Conventions

- **Kein `Co-authored-by`** — Agent-Commits, nicht noetig.
- **Keine Notion-Task-Referenzen** in Commit-Messages — gehoert in den Workflow, nicht in den Code.
- Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) immer.
