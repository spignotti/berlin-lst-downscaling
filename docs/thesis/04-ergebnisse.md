# 04 · Ergebnisse & Analysen

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66814589beffe8d5c803fe`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.

## Zentrale empirische Befunde

**Datenfreigabe (WB2):** Der publizierte V3-Datenbestand umfasst 324 aufbereitete 28-Kanal-Feature-Stacks für den Trainings-/Evaluationszeitraum. Das freigegebene Trainingdataset `training/v1` enthält 5.520.165 gültige 100-m-Zellbeobachtungen bei vollständigem 10-m-Support; der 2026er Ankerbestand bleibt für spätere Inferenz reserviert. Die dokumentierten Ablations- und Baselineergebnisse beruhen auf diesem Stand.

**Korrigierte Datenfreigabe (training/v2):** Nach Korrektur von Versiegelung, Landsat-QA und Landsat-Target sowie Ausschluss zweier unplausibler Szenen umfasst `training/v2` 283 publizierte Szenen mit 4.728.053 eligible 100-m-Zellbeobachtungen (Train 3.489.565, Validation 608.091, Test 630.397) und bei der Patchschwelle 244/256 6.619 Patches in 166 Szenen. Details und EDA-Befunde: [Datenbasis & Feature Engineering](03-methodik/datenbasis-feature-engineering.md).

**Naive Baseline und Stage 1 (WB3):** Auf dem 2025-Test erreicht die naive Prior-Expansion eine maskierte aggregierte MAE von **1,39523 K**, die spektrale residuale Stage 1 **0,53257 K**.

**Kumulative Ablation (WB3):** Die dokumentierten 2025-Testwerte betragen **0,51366 K** für Stage 2 (+ Morphologie), **0,49991 K** für Stage 3 (+ Schatten), **0,47535 K** für Stage 4 (+ ERA5) und **0,45931 K** für Stage 5 (gleiche Inputs, modifizierter Loss).

**Isolationsruns:** Spektral + Schatten: **0,52797 K**; spektral + ERA5: **0,47023 K** auf dem 2025-Test. Für alle dokumentierten Runs liegt eine quantitative Auswertung derselben primären Metrik vor.

## Bewertungsgrenze

Die genannten Fehler sind **100-m-zellgewichtete MAE nach Rückaggregation** des 10-m-Modelloutputs. Sie sind kein Nachweis gemessener thermischer 10-m-Detailgenauigkeit. Random-Forest-Vergleich, räumlicher Holdout, unabhängige thermale Validierung und die vollständige Berlin-Zeitreihe sind noch nicht als Ergebnisse belegt.

## Ergebnisberichte

- [WB2 Ergebnisse: Datenbestand und Pipelines](04-ergebnisse-wb2-datenbestand.md)

- [WB3 · Ergebnistabellen – Baselines und Ablation](04-ergebnisse-wb3-ergebnistabellen.md)
