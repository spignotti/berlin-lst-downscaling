# Pseudo-Pair- und Tensorvertrag (WB3)

> Migriert aus Notion am 2026-10-28 (Seite `bce40f9b320e431aa285d04e84a1343c`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

Freigegeben 2026-09-23. Variante B.

## Entscheidung

- Hauptvertrag: **B** (scale-matched, Output fein, Loss grob).
- LST-Prior als eigener Input-Kanal: ja.
- Loss-Aggregation 10 m → 100 m: block-mean über exakte 10×10-Nesting-Zellen des kanonischen EPSG:25833-Grids.

## Motivation

Produktziel ist 100 m → 10 m LST. Natives Thermal-Target existiert nur auf 100 m. Pseudo-Pairs üben denselben Faktor 10 eine Stufe gröber (1000 m → 100 m Supervision) und übertragen die Abbildung auf die Produktaufgabe.

## Patch-Geometrie und Zulassung

Freigegeben 2026-09-23.

- Hauptpatch: **160×160 px @ 10 m** = **16×16 Zellen @ 100 m** = 1,6 km Kante
- U-Net-Tiefe für Größen-Lock: **D = 4** (Kante teilbar durch 16); D = 5 optional später, 160er ist vorbereitet
- Smoke-Fallback: 80×80 @ 10 m (8×8 @ 100 m), nur Debug, nicht Hauptablation
- Patch-Zulassung: **≥ 95 %** der 100 m-Zellen im Fenster `training_eligible`; Zählbasis = alle 16×16 Fensterzellen
- Szenenregel: ≥ 1 gültiger Patch, sonst Ausschluss `no_valid_patch`
- Index: `patch_index.parquet` aus Eligibility-Masken, keine vorgebackenen Patch-Raster
- Stride/Overlap (2026-09-24): Stride = 160, kein Overlap; Geometrie und 95 % bleiben bis nach Stage-1-Smoke
- Sampling (2026-09-24): Patch-Zeile = Landsat-Szene × Spatial-Fenster; Batch-Diversität nach `s2_scene_id` (S2-Reuse ist der Redundanzpunkt, nicht Spatial-Reuse allein); keine Split-Regewichtung
- Noch offen: Batchgröße, konkrete Caps/Weights im Reader

### Patch-Index v3 (training/v3)

Der aktuelle Patch-Index v3 (`gs://berlin-lst-training-data/training/patch-index/v3`, `configs/training/patch_index_v3.yaml`) umfasst **8.078 Patches** (Train 5.523, Validation 1.047, Test 1.508). Geometrie (160×160 px @ 10 m) und Zulassungsschwelle (0,95) sind unverändert. Weitere Aufschlüsselung (Szenen, Anker, Jahre): TODO.

### Patch-Index Ergebnis (2026-09-23)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Abgelöst durch Patch-Index v3 auf `training/v3` (siehe oben).

Aus `patch_index.parquet` über 345 Manifestzeilen:

- Patches: **10.018**
- Szenen mit ≥ 1 Patch: **182**
- Distinct spatial anchors: 226
- Eligible 100 m-Zellen in Patches: **2.543.186**
- `eligible_frac_min`: **0.953125** = 244/256

Szenenbilanz: 182 mit Patch, 112 `no_valid_patch`, 30 `no_eligible_cells`, 21 Inferenz (2026) übersprungen.

Split: train 6.873 Patches / 126 Szenen, validation 1.342 / 30, test 1.803 / 26.<br>Jahr: 2017: 206, 2018: 1.199, 2019: 569, 2020: 656, 2021: 291, 2022: 1.482, 2023: 2.470, 2024: 1.342, 2025: 1.803.<br>Anker-Ausdehnung: Zeilen 48–304, Spalten 32–320 im 389×470-Grid (100 m).

**Befunde**

- `eligible_frac_min` = 244/256: der 95%-Floor ist am Rand die bindende Bedingung, wie vorgesehen.
- 112 von 324 bewertbaren Szenen (35 %) fallen als `no_valid_patch` heraus: eligible Zellen vorhanden, aber kein 1,6 km-Fenster mit 95 %. Die eingefrorene Geometrie ist streng gegen das Eligibility-Muster. Bei knappem Trainingsvolumen sind Stride/Overlap oder Schwelle die Stellschraube, kein Fehler.
- 2.543.186 / (10.018 × 256) = 99,2 %: Patches sind praktisch disjunkt, der aktuelle Stand ist ein überlappungsfreies Tiling.
- 226 distinct anchors bei 10.018 Patches: Fenster werden über Szenen hinweg wiederverwendet.
- Split ungleich verteilt, aber nirgends leer (126/30/26 Szenen): ein split-bewusster Sampler hat in allen drei Teilen Material.

## Tensorvertrag

| Feld | Form | Training | Inferenz |
| --- | --- | --- | --- |
| `features` | 22 × H10 × W10 | `features/v6`, Scaler `training/v3` | gleich |
| `lst_prior` | 1 × H10 × W10 | 100 m LST → mean-pool 10×10 → 1000 m, block-expand auf 10 m | natives 100 m LST, block-expand auf 10 m |
| `target_100m` | 1 × H100 × W100 | natives Landsat-LST | nicht verwendet |
| `mask_100m` | H100 × W100 | `training_eligible@100m` | nicht verwendet |
| `prediction_10m` | 1 × H10 × W10 | Modellausgabe | Produktausgabe |

Die historischen Läufe nutzten 28 × H10 × W10, V3 und Scaler `training/v1`.

## Loss

Nur gültige 100 m-Zellen:<br>`L = sum(m_100 * |pool10×10(pred_10) - y_100|) / count(m_100)`

Kein NaN-as-zero. Pool folgt dem kanonischen Nested Grid, kein freies Resampling.

**Stage 1 (2026-09-24):** Familie = masked L1. Selection-Metrik = masked MAE @ 100 m auf Validation; SSIM sekundär (log, kein Gate). Naive Baseline = Prior block-expand und optional bilinear, gleicher Split/Maske/Metriken. RF später als starke Baseline, nicht Stage-1-Scope. Output bleibt 10 m; 1000 m betrifft nur den Trainings-Prior.

## Bereits fest (nicht neu verhandeln)

- `features/v6`, 22-Kanal-Order (Feature-Schema 5); V3/`features/v5` mit 28 Kanälen historisch
- Split: Train 2017–2023, Val 2024, Test 2025; 2026 nur Inferenz
- `s2_scene_id`-Gruppierung, 100/100 Eligibility, kein Input-Impute
- Scaler train-only

## Trainings-I/O (2026-09-28, #30/#32)

- Default-Read-Pfad: **COG-Window-Streaming** (`data.mode=stream` für Full-Train). Eager nur bounded Smokes.
- Streaming read-invariant vs. Eager (IDs, Exclusions, Tensoren, Seed-Order) — Evidenz: [Notion-Seite c315449f](https://app.notion.com/p/c315449f6a3a45a3bdcb0d8449e8b08c) (Notion-Archiv).
- Augmentation Stage 1: **keine**.
- Zarr bleibt optionaler Benchmark-Kandidat nach stabilem Stage-1 auf Streaming, nicht Default.

## Bewusst offen (später)

- Konkrete Sampler-Caps/Weights im Dataset-Reader (`s2_scene_id`)
- **Historisch offen, inzwischen entschieden:** Stage 5 verwendet masked L1 + SSIM-/Gradienten-Strafterme; Stage 1–4 bleiben bei masked L1. Verbindliche Konfiguration im Repository, wissenschaftlicher Vergleich auf der Ergebnis-Seite.
- Zarr-Benchmark (nur wenn Streaming-Stage-1 I/O-bound ist)
- **Historisch offen, inzwischen ausgeführt:** gesperrte Stage-1–5-Läufe auf Vertex AI, Batch 4 und T4; technische Runtime-Parameter ausschließlich im Repository und in W&B.

## Option später

Variante C (Prior = natives 100 m auch im Training, Loss nur 10→100-Aggregat) als naive/produktnahe Baseline, nicht als Hauptvertrag.

## Bezug

- Datenvertrag Repo: `docs/data-sources-and-contracts.md`
- Grundlagen: DL-Training: Grundlagen und Thesis-Setup
- WB2c-4: Splits, Eligibility, Scaler
