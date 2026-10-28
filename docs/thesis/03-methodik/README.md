# 03 · Methodik – aktueller Stand

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668138a32ae0c279cc18f4`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.

## Datenaufbau und Trainingskohorte

- **Zeitraum:** Training 2017–2023, Validation 2024, Test 2025; 2026 derzeit nur Inferenzreserve.
- **Thermisches Target:** Landsat 8/9 LST auf dem kanonischen 100-m-Raster, gebildet als Flächenmittel der 30-m-`lwir11`-Werte; QA bitweise auf dem nativen 30-m-Raster ausgewertet.
- **Eingangsgrößen:** Feature-Release v5 (Trainingsfreigabe `training/v2`; Vorgänger V3/`training/v1` für die dokumentierten Ablationsläufe) mit 28 möglichen 10-m-Kanälen aus Sentinel-2-Spektral-/Indexinformationen (10), urbaner Morphologie (8), Gebäude-/Vegetationsschatten (2) und ERA5-Land-Meteorologie (8).
- **Datenfreigabe:** Strikte Trainingseignung für vollständige 10-m-Featurevektoren in allen 100 Unterzellen eines gültigen 100-m-Thermalziels; Scaler ausschließlich auf Trainingdaten angepasst.

## Modell und Ablation

Fester 2D-U-Net-Backbone mit residualem LST-Prior. Training anhand synthetischer Pseudo-Pairs (1.000-m-Prior, 100-m-Supervision); Ausgabeauflösung 10 m, Trainingsloss und primäre Validierungsmetrik nach Rückaggregation auf 100 m.

| Stufe | Aktive Modellkanäle | Variation |
| --- | --- | --- |
| 1 | 10 | Spektralblock, masked L1 |
| 2 | 18 | • Morphologie |
| 3 | 20 | • Gebäude-/Vegetationsschatten |
| 4 | 28 | • ERA5-Land |
| 5 | 28 | Gleiche Inputs wie 4; masked L1 plus SSIM-/Gradientenstrafen |

Die dokumentierten Vollruns nutzen die fixierte Residualrepräsentation, dieselbe Patchgeometrie, den temporalen Split und Seed 0.

## Validierung und Berlin-Anwendung

**Bereits umgesetzt:** Full-Split-Vergleich mit naiver Prior-Expansion, 2024-Validation-basierte Checkpointwahl und 2025-Testauswertung der dokumentierten Ablationsruns.

**Noch nicht abgeschlossen:** Random-Forest-Baseline, unabhängige thermale Vergleichsdaten, räumlicher Holdout, strukturbezogene 10-m-Plausibilitätsprüfung und die vollständige 10-m-LST-Zeitreihe für Berlin. Zusätzliche Untersuchungen sind für die Einordnung der räumlichen Aussagekraft erforderlich.

**Technische Evidenz:** [Daten- und Release-Contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md) · [Modell- und Ablations-Contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-stage-configs.md) · [W&B-Projekt](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling).

## Methodische Teilbereiche

- [Datenbasis & Feature Engineering](datenbasis-feature-engineering.md)

- [Modell & Ablationsprotokoll](modell-ablationsprotokoll.md)

- [Evaluation & Unsicherheitsanalyse](evaluation-unsicherheit.md)

- [Berlin-Anwendung & geplante Auswertungen](berlin-anwendung.md)
