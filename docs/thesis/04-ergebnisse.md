# 04 · Ergebnisse & Analysen

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66814589beffe8d5c803fe`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

## Gültiger Datenstand (28.10.2026)

- **Splits (fix):** Train 2017–2023, Validation 2024, Test 2025; 2026 nur Inferenz. Der Scaler wird ausschließlich auf den Trainingsdaten gefittet.
- **Selection v4:** Manifest-Bundle `manifests/v4/2017-2026-cutoff-20260717T235959Z` mit exakter Sentinel-2-Mosaikzugehörigkeit (Solartag-/Datatake-Gruppe) und Landsat-AOI-Overlap-Gate; 290 Paarungen ([Republish v6](../republish-v6.md)).
- **ARD v10:** Release `ard/full/2017-2026-s2-mosaic-v10` (Lauf `ard-v10-20261024T040705Z-83AB242C`). Das Sentinel-2-Mosaik-Gate (`max_missing_aoi_fraction = 0.005`) schließt genau zwei Szenen aus: `S2A_MSIL2A_20210814T102031_R065_T32UQD_20210814T215405` (6,28 % der AOI-Pixel fehlend) und `S2A_MSIL2A_20230503T101021_R022_T32UQD_20240905T123554` (18,26 % fehlend). Dadurch entfallen 4 Paarungen; 286 von 290 bleiben, ein Sentinel-2-Ersatzpartner wird nicht gewählt. Das Landsat-Target ist der Flächenmittelwert der 30-m-Schicht `lwir11` (`resampling=average`) mit QA `native_bit_or`.
- **features/v6:** 22 Kanäle (Schema-Version 5): 6 spektrale Bänder, 4 Indizes, 6 Morphologiekanäle (u. a. `canopy_cover_fraction`, `svf` mit Radius 100 m), 4 ERA5-Kanäle (je Szene ein Wert) und 2 Schattenkanäle. Die Morphologie stammt aus `lod2-area-exact-v2`. Details: [Datenbasis & Feature Engineering](03-methodik/datenbasis-feature-engineering.md) und [Features v6](../features-v6.md).
- **training/v3:** aus features/v6 mit Train-only-Scaler; die 4 durch das Gate ausgeschlossenen Paarungen tragen den Status `excluded` ([Republish v6](../republish-v6.md)).
- **Patch-Index v3:** 8.078 Patches (Train 5.523, Validation 1.047, Test 1.508) bei unveränderter Geometrie von 160×160 px ([QA](../results/patch-index-v3/patch_index_qa.json), [complete.json](../results/patch-index-v3/complete.json)).
- **EDA v3:** 266 gescannte Szenen (265 mit Beobachtungen), davon 201 Trainingsszenen, rund 5,97 Mio. Beobachtungen (Szene-Zelle-Kombinationen) und 83.602 verschiedene zulässige 100-m-Zellen ([summary.json](../results/eda-v3/summary.json)). TODO: Interpretation der EDA-v3-Befunde.

## Ergebnisse der Ablation auf training/v3

Ausstehend. Vorgesehene Eingabearme: (1) volle Optik (alle 22 Kanäle), (2) ohne NDVI, (3) ohne Rot, NIR und Indizes. Die finale Ablationsmatrix (Arme, Seeds, Loss-Varianten, Neuberechnung von RF- und naiver Baseline) ist offen. TODO: Ergebnisse eintragen, sobald die Läufe auf training/v3 vorliegen.

## Historische Befunde (training/v1, training/v2)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Der gültige Datenstand ist oben beschrieben; die Ablation wird auf training/v3 neu berechnet.

**Datenfreigabe (WB2):** Der publizierte V3-Datenbestand umfasst 324 aufbereitete 28-Kanal-Feature-Stacks für den Trainings-/Evaluationszeitraum. Das freigegebene Trainingdataset `training/v1` enthält 5.520.165 gültige 100-m-Zellbeobachtungen bei vollständigem 10-m-Support; der 2026er Ankerbestand bleibt für spätere Inferenz reserviert. Die dokumentierten Ablations- und Baselineergebnisse beruhen auf diesem Stand.

> **Historischer Stand (training/v2, features/v5, 28 Kanäle), abgelöst durch features/v6 / ARD v10 / training/v3.**

**Korrigierte Datenfreigabe (training/v2):** Nach Korrektur von Versiegelung, Landsat-QA und Landsat-Target sowie Ausschluss zweier unplausibler Szenen umfasst `training/v2` 283 publizierte Szenen mit 4.728.053 eligible 100-m-Zellbeobachtungen (Train 3.489.565, Validation 608.091, Test 630.397) und bei der Patchschwelle 244/256 6.619 Patches in 166 Szenen. Details und EDA-Befunde: [Datenbasis & Feature Engineering](03-methodik/datenbasis-feature-engineering.md).

**Naive Baseline und Stage 1 (WB3):** Auf dem 2025-Test erreicht die naive Prior-Expansion eine maskierte aggregierte MAE von **1,39523 K**, die spektrale residuale Stage 1 **0,53257 K**.

**Kumulative Ablation (WB3):** Die dokumentierten 2025-Testwerte betragen **0,51366 K** für Stage 2 (+ Morphologie), **0,49991 K** für Stage 3 (+ Schatten), **0,47535 K** für Stage 4 (+ ERA5) und **0,45931 K** für Stage 5 (gleiche Inputs, modifizierter Loss).

**Isolationsruns:** Spektral + Schatten: **0,52797 K**; spektral + ERA5: **0,47023 K** auf dem 2025-Test. Für alle dokumentierten Runs liegt eine quantitative Auswertung derselben primären Metrik vor.

## Bewertungsgrenze

Die genannten Fehler sind **100-m-zellgewichtete MAE nach Rückaggregation** des 10-m-Modelloutputs. Sie sind kein Nachweis gemessener thermischer 10-m-Detailgenauigkeit. Der Random-Forest-Vergleich liegt historisch auf training/v1 vor und ist auf training/v3 ausstehend; räumlicher Holdout, unabhängige thermale Validierung und die vollständige Berlin-Zeitreihe sind noch nicht als Ergebnisse belegt.

## Ergebnisberichte

- [WB2 Ergebnisse: Datenbestand und Pipelines](04-ergebnisse-wb2-datenbestand.md)

- [WB3 · Ergebnistabellen – Baselines und Ablation](04-ergebnisse-wb3-ergebnistabellen.md)
