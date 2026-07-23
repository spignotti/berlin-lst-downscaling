# Project State

Last updated: 2026-07-20

**Current Focus:** `feat/secondary-data-pipeline` branch — Dynamic Pipeline C implemented, contract repairs complete. CDS service down today — full ERA5 run pending.

**Active Work:** Dynamic pipeline ready for cloud execution once CDS recovers.

### 2026-07-20
- Completed: Dynamic pipeline (Pipeline C) implementation and contract repair. ERA5-Land adapter with vectorized SSRD, antecedent concatenation, and spatial extraction. Building/vegetation shadow adapters with proper provenance propagation. DWD station validation via directory listing. Dynamic QA report. Smoke/cloud configs. Data inventory note published to Notion.
- Open: CDS service outage — ERA5 full run pending. Local smoke test pending eccodes install on VM.
- Context: All static/derived products published. Dynamic pipeline code complete, awaiting CDS recovery for production run.

### 2026-07-19
- Completed: Static Pipeline B DSM validation. Full 2017-2026 v3 manifest (509 rows, 345 pairings). Full ARD run (509/509 scenes). GCS atomic write retries.
- Open: None.
- Context: All data pipelines complete. Ready for training/analysis work.

### 2026-07-18
- Completed: Manifest repair, ARD v3 bundle, item_href resolution.
- Open: None.

### 2026-07-17
- Completed: Static sources + derived products fully validated in GCS.
- Open: None.

---

## Key Decisions (≤5)

- 2026-07-20 Dynamic geometry policy: retrospective_static — LoD2-2024, DGM-2021, VH-2020 for all 2017-2025 scenes.
- 2026-07-19 Full ARD accepts manifest with 509 scenes, requires all three sources.
- 2026-07-19 GCS atomic writes use 5-attempt exponential backoff (1-60s).
- 2026-07-19 Combined DSM is exactly max(building_dsm, vegetation_dsm).
- 2026-07-08 Canonical raster grid: 10m base GeoBox, origin (369190, 5838410) EPSG:25833.

---

## Lessons

- 2026-07-20 ERA5-Land SSRD accumulates within each day and resets at 00:00 UTC — diff must be per-day, not continuous.
- 2026-07-20 ERA5-Land GRIB contains multi-cell lat/lon — never collapse with `.mean()` before spatial extraction.
- 2026-07-20 DWD historical archives use station-specific date ranges in filenames — discover from index, don't guess.
- 2026-07-19 PyArrow 14+ `pc.is_in()` requires SetLookupOptions, not plain set/list.
- 2026-07-19 GCS rate limits (429) require bounded retries in `_atomic_write_gcs`.
- 2026-07-19 `combined_dsm == max(building_dsm, vegetation_dsm)` is exact.
- 2026-07-16 `np.maximum.at(arr, mask, val)` with NaN-initialized arr returns NaN.
- 2026-07-15 Product contract: COG, STAC, provenance, complete.json (publication gate).

---

## Local Skills

- `.opencode/skills/google-access/` — rclone mount, ADC, GCS access patterns.

---

## Commit Conventions

- **Kein `Co-authored-by`** — Agent-Commits, nicht noetig.
- **Keine Notion-Task-Referenzen** in Commit-Messages — gehoert in den Workflow, nicht in den Code.
- Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) immer.
